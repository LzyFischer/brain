"""
Module implementing the Cross-GNN model for use within the brain project.

This module vendors the implementation of the Cross-GNN architecture from
the standalone Cross-GNN repository and wraps it so that it can operate
directly on PyTorch Geometric ``Data`` objects produced by the brain
dataset.  The wrapper converts batched graphs into the dense 4D tensor
representation required by the Cross model.  Hyperparameters such as
``channel``, ``layer`` and ``gru`` are expected to be supplied via
``args`` at runtime (see ``main_brain.py`` for parser additions).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["masked_softmax", "Refine", "Cross", "CrossGNNBrain"]


def masked_softmax(src: torch.Tensor, mask: torch.Tensor | None = None, dim: int = 1) -> torch.Tensor:
    """Apply softmax along ``dim`` with optional masking.

    The original Cross-GNN implementation applies a full softmax without
    masking, so this helper mirrors that behaviour but keeps the API
    compatible should future work wish to add masking support.

    Args:
        src: Input tensor.
        mask: Boolean mask the same shape as ``src``, where ``False``
            entries would normally be masked out.  Currently unused.
        dim: Dimension along which to apply softmax.

    Returns:
        Softmax-normalised tensor along the specified dimension.
    """
    # The commented-out lines show how masking could be applied; the
    # Cross-GNN authors instead apply a plain softmax.
    # out = src.masked_fill(~mask, float('-inf'))
    out = torch.softmax(src, dim=dim)
    # out = out.masked_fill(~mask, 0)
    return out


class Refine(nn.Module):
    """Small feedforward network used inside the Cross-GNN model.

    Given an input sequence of ``channel`` hidden states, this module
    aggregates by taking the mean over the sequence dimension and then
    passes the result through a multilayer perceptron.  It returns both
    log-probabilities and probabilities for two output classes.
    """

    def __init__(self, channel: int, drop: float) -> None:
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(channel, 128),
            nn.LeakyReLU(),
            nn.Dropout(drop),
            nn.Linear(128, 30),
            nn.LeakyReLU(),
            nn.Dropout(drop),
            nn.Linear(30, 2),
        )

    def forward(self, x: torch.Tensor, tem: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute log- and softmax probabilities over two classes.

        Args:
            x: Input tensor of shape (batch, seq_len, channel).
            tem: Temperature for scaling logits before the softmax.

        Returns:
            A tuple ``(log_probs, probs)`` each of shape (batch, 2).
        """
        B = x.size(0)
        # Aggregate hidden states across the sequence dimension and flatten
        p1 = self.nn(x.mean(1).view(B, -1))
        return F.log_softmax(p1 / tem, dim=1), F.softmax(p1 / tem, dim=1)


class Cross(nn.Module):
    """Full Cross-GNN network.

    The model expects input with two modalities per subject, each
    represented as a sequence of length ``kernel_size``.  The model uses
    GRUs to process each modality independently, then computes an
    attention-like correspondence matrix to exchange information across
    modalities.  Two refinement networks produce auxiliary outputs used
    during training.

    Args:
        in_channel: Number of modalities (always 2 for functional and
            structural graphs).
        kernel_size: Sequence length / number of regions of interest
            (nodes).  Must match the dimension of the input graphs.
        num_classes: Number of target classes.
        args: Namespace of hyperparameters providing ``channel``,
            ``layer``, ``ab``, ``gru`` and ``alpha``.
    """

    def __init__(self, in_channel: int, kernel_size: int, num_classes: int = 2, args: object | None = None) -> None:
        super().__init__()
        self.d = kernel_size
        self.channel = args.channel if args and hasattr(args, "channel") else 32
        self.ll = args.layer if args and hasattr(args, "layer") else 2
        self.ab = args.ab if args and hasattr(args, "ab") else 0
        self.drop = 0.2
        num_layers = args.gru if args and hasattr(args, "gru") else 1

        # Two GRU encoders (one per modality)
        self.psi_1 = nn.GRU(input_size=self.d, hidden_size=self.channel, num_layers=num_layers)
        self.psi_2 = nn.GRU(input_size=self.d, hidden_size=self.channel, num_layers=num_layers)

        # Per-modality MLPs for projecting concatenated hidden states
        self.mlp1 = nn.Sequential(
            nn.Linear(self.channel * 2, self.channel * 4),
            nn.ReLU(),
            nn.Dropout(self.drop),
            nn.Linear(self.channel * 4, self.channel * 2),
            nn.ReLU(),
            nn.Dropout(self.drop),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(self.channel * 2, self.channel * 4),
            nn.ReLU(),
            nn.Dropout(self.drop),
            nn.Linear(self.channel * 4, self.channel * 2),
            nn.ReLU(),
            nn.Dropout(self.drop),
        )

        # Batch normalisation layers
        self.bn1 = nn.BatchNorm2d(self.channel * 2)
        self.bn2 = nn.BatchNorm2d(self.channel * 2)
        self.mlp_bn1 = nn.BatchNorm2d(self.channel)
        self.mlp_bn2 = nn.BatchNorm2d(self.channel)

        # Convolution over temporal dimension after concatenation
        self.nf_1 = nn.Conv2d(self.channel * 4, self.channel * 4, (self.d, 1))

        # Two refinement heads
        self.refine1 = Refine(self.channel, self.drop)
        self.refine2 = Refine(self.channel, self.drop)

        # Final classifier
        self.dense1 = nn.Linear(self.channel * 2 * 2, 128)
        self.dense2 = nn.Linear(128, 30)
        self.dense3 = nn.Linear(30, num_classes)

    def forward(self, x: torch.Tensor, tem: float = 1.0, get_corr: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of Cross-GNN.

        Args:
            x: Input tensor of shape (B, 2, d, d) where ``d`` is the
                number of nodes and modality dimension.
            tem: Temperature used for the refinement heads.
            get_corr: If ``True``, also return the correspondence matrix.

        Returns:
            Either ``(out_p, out1_log, out2_log)`` or
            ``(out_p, out1_log, out2_log, S_00)`` if ``get_corr`` is True.
        """
        B = x.size(0)

        # Split modalities: functional and structural
        h_s, _ = self.psi_1(x[:, 0])  # shape: (seq_len, B, hidden) -> we ignore seq_len dimension in GRU output
        h_t, _ = self.psi_1(x[:, 1])
        # Remove the unnecessary sequence dimension from GRU output
        h_s = h_s.squeeze(2)
        h_t = h_t.squeeze(2)

        # Apply batch normalisation in a channel-first manner
        h_s = self.mlp_bn1(h_s.permute(0, 2, 1)[..., None])[..., 0].permute(0, 2, 1)
        h_t = self.mlp_bn2(h_t.permute(0, 2, 1)[..., None])[..., 0].permute(0, 2, 1)

        # Compute symmetric correspondence matrix between modalities
        S_hat1 = h_s @ h_t.transpose(-1, -2)
        S_hat2 = h_t @ h_s.transpose(-1, -2)
        S_00 = masked_softmax((S_hat1 + S_hat2) / 2.0)

        # Cross-attend hidden states
        r_s = S_00 @ h_t
        r_t = S_00 @ h_s
        h_st1 = torch.cat((h_t, r_s), dim=2)
        h_st2 = torch.cat((h_s, r_t), dim=2)

        # Apply modality-specific MLPs and batch normalisation
        x1 = self.mlp1(h_st1)
        x2 = self.mlp2(h_st2)
        x1 = self.bn1(x1.permute(0, 2, 1)[..., None])[..., 0].permute(0, 2, 1)
        x2 = self.bn2(x2.permute(0, 2, 1)[..., None])[..., 0].permute(0, 2, 1)

        # Propagate through the correspondence matrix a configurable number of times
        for _ in range(self.ll):
            x1 = torch.einsum("npq,nqc->npc", S_00, x1)
            x1 = F.leaky_relu(x1)
            x1 = F.dropout(x1, self.drop, training=self.training)

            x2 = torch.einsum("npq,nqc->npc", S_00, x2)
            x2 = F.leaky_relu(x2)
            x2 = F.dropout(x2, self.drop, training=self.training)

        # Concatenate modality representations and collapse sequence dimension via conv
        x_cat = torch.cat((x1, x2), dim=2)  # shape: (B, d, channel*2)
        x_cat = x_cat.permute(0, 2, 1)      # (B, channel*2, d)
        out = self.nf_1(x_cat[..., None]).view(B, -1)

        # Final classification MLP
        out = F.dropout(F.leaky_relu(self.dense1(out)), p=self.drop, training=self.training)
        out = F.dropout(F.leaky_relu(self.dense2(out)), p=self.drop, training=self.training)
        logits = F.leaky_relu(self.dense3(out))
        out_p = F.softmax(logits, dim=1)

        # Refinement heads
        out1_log, out1_p = self.refine1(h_s, tem)
        out2_log, out2_p = self.refine2(h_t, tem)

        if get_corr:
            return out_p, out1_log, out2_log, S_00
        return out_p, out1_log, out2_log


class CrossGNNBrain(nn.Module):
    """Wrapper for Cross-GNN to work with brain ``Data`` objects.

    The brain pipeline works with batched PyG ``Data`` instances.  Each
    graph ``data`` contains two dense adjacency matrices: ``data.x``
    (functional connectivity) and ``data.x_SC`` (structural connectivity).
    When a batch of such graphs is collated, these tensors are stacked
    along their first dimension.  This wrapper reconstructs the per-graph
    adjacency matrices and stacks them into the shape expected by the
    Cross-GNN model: ``(B, 2, N, N)``.
    """

    def __init__(self, net_params: dict, args: object) -> None:
        super().__init__()
        self.num_nodes = net_params.get("in_channels")
        # Number of classes defaults to 2 for binary classification
        self.num_classes = net_params.get("num_classes", 2)
        self.args = args
        # Build the underlying Cross model
        self.model = Cross(
            in_channel=2,
            kernel_size=self.num_nodes,
            num_classes=self.num_classes,
            args=args,
        )

    def _batch_to_tensor(self, batch: object) -> torch.Tensor:
        """Convert a PyG batch into a dense tensor for Cross-GNN.

        Args:
            batch: A batch of PyG ``Data`` objects.  ``batch.x`` and
                ``batch.x_SC`` should each be of shape ``(B * N, N)``.

        Returns:
            A tensor ``(B, 2, N, N)`` where modality 0 is functional
            connectivity and modality 1 is structural connectivity.
        """
        # Determine batch size from labels or x dimension
        if hasattr(batch, "y"):
            B = batch.y.shape[0]
        else:
            # Fallback if labels are absent
            B = batch.x.size(0) // self.num_nodes
        # Reshape the adjacency matrices back to (B, N, N)
        fc = batch.x.view(B, self.num_nodes, self.num_nodes)
        sc = batch.x_SC.view(B, self.num_nodes, self.num_nodes)
        # Stack modalities along channel dimension
        return torch.stack([fc, sc], dim=1)

    def forward(self, data: object, tem: float = 1.0, get_corr: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the wrapper.

        Accepts either a batched PyG ``Data`` instance or an already
        formatted dense tensor.  When presented with a ``Data`` object
        the wrapper converts it into the shape required by the underlying
        Cross model.

        Args:
            data: Batched ``Data`` or dense tensor of shape ``(B, 2, N, N)``.
            tem: Temperature parameter passed to Cross-GNN.
            get_corr: Whether to also return the correspondence matrix.

        Returns:
            Whatever the underlying Cross model returns (either three or
            four tensors).
        """
        if isinstance(data, torch.Tensor):
            x = data
        else:
            x = self._batch_to_tensor(data)
        return self.model(x.to(next(self.parameters()).device), tem=tem, get_corr=get_corr)