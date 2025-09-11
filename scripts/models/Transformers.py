# brain_transformers.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    global_mean_pool,
    TransformerConv,
)
from torch_geometric.utils import to_dense_batch
# from utils.utils import plot_edge_weight
import numpy as np
import copy
import pdb          


# --------------------------------------------------------------------------- #
#                               Helper modules                                #
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    """Two‑layer MLP with GELU + dropout."""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, p: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def _sym_normalize(A, eps=1e-8):
    # A: (..., N, N), 返回 D^{-1/2} A D^{-1/2}
    deg = A.sum(dim=-1) + eps                       # (..., N)
    dinv2 = (deg.reciprocal().sqrt()).unsqueeze(-1) # (..., N,1)
    return dinv2 * A * dinv2.transpose(-1, -2)

# --------------------------------------------------------------------------- #
#                          Adaptive Prior Fusion (APF)                        #
# --------------------------------------------------------------------------- #
class AdaptivePriorFusion(nn.Module):
    """
    APF: Global matrix per prior (+free) + personalized symmetric rank-1 mask via hypernet.
    Generates subject-specific gate G(z) from priors and sample embedding z.

    priors: (B, P0, N, N)   – K prior adjacency matrices (0/1)
    z     : (B, N, H)       – node embeddings (current layer)
    A     : (B, N, N)       – subject connectivity (FC/SC), used if mode="gate" for A_tilde
    returns:
        if mode=="mask":  G ∈ (0,1)^{B,N,N}
        if mode=="gate":  A_tilde = (1 + gamma*G) ⊙ A,  and G
        if mode=="bias":  bias = beta*G, and G
    """
    def __init__(self, N: int, emb_dim: int, num_priors: int,
                 use_free: bool = True, tau: float = 2.0,
                 init_scale: float = 1e-2):
        super().__init__()
        self.N = N
        self.P0 = num_priors
        self.use_free = use_free
        self.P = num_priors + (1 if use_free else 0)
        self.tau = tau

        # Global matrices per prior(+free)
        self.Wg = nn.Parameter(torch.randn(self.P, N, N) * init_scale)

        # Hypernet to produce u_p(z) and α_p(z)
        hid = max(64, emb_dim // 2)
        self.hyper_uA = nn.Sequential(
            nn.Linear(emb_dim, hid), nn.ReLU(),
            nn.Linear(hid, self.P * N)
        )
        self.hyper_uB = nn.Sequential(
            nn.Linear(emb_dim, hid), nn.ReLU(),
            nn.Linear(hid, self.P * N)
        )
        self.hyper_alpha = nn.Sequential(
            nn.Linear(emb_dim, hid), nn.ReLU(),
            nn.Linear(hid, self.P), nn.Tanh()  # α ∈ (-1,1)
        )

        # Optional learned scales for stability
        self.scale_g = nn.Parameter(torch.zeros(self.P))
        self.scale_r = nn.Parameter(torch.zeros(self.P))

    @staticmethod
    def _sym_zero_diag(M):
        M = 0.5 * (M + M.transpose(-1, -2))
        M = M - torch.diag_embed(torch.diagonal(M, dim1=-2, dim2=-1))
        return M

    def _build_full_priors(self, priors: torch.Tensor):
        # priors: (B,P0,N,N). If use_free, append free mask = 1 - max_k priors_k
        if not self.use_free:
            return priors
        Pmax = priors.amax(dim=1)                         # (B,N,N)
        pri_free = (1.0 - Pmax).clamp(min=0.0)            # (B,N,N)
        pri_full = torch.cat([priors, pri_free.unsqueeze(1)], dim=1)  # (B,P,N,N)
        return pri_full

    def forward(self, priors: torch.Tensor, z: torch.Tensor,
                A: torch.Tensor = None, mode: str = "mask",
                gamma: float = 0.3, beta: float = 1.0):
        """
        mode: "mask" | "gate" | "bias"
        """
        device = z.device
        B, N, H = z.shape
        assert N == self.N, f"APF: N mismatch (got {N}, expected {self.N})"

        # Build full prior list (append free if enabled)
        P = self._build_full_priors(priors)               # (B,P,N,N)
        P = P.to(z.dtype).to(device)
        P_bin = (P > 0).to(z.dtype)

        # Global term (sym+zero_diag; bounded via tanh; scaled)
        Wg = self._sym_zero_diag(self.Wg)                 # (P,N,N)
        Wg = torch.tanh(Wg) * (1.0 + self.scale_g.view(-1,1,1))
        Wg_b = Wg.unsqueeze(0).expand(B, -1, -1, -1)      # (B,P,N,N)

        # Personalized rank-1 from subject embedding z̄
        z_bar = z.mean(dim=1)                             # (B,H)
        U_A = self.hyper_uA(z_bar).view(B, self.P, N)   # (B,P,N)
        U_B = self.hyper_uB(z_bar).view(B, self.P, N)   # (B,P,N)
        U_A = F.normalize(U_A, dim=-1)
        U_B = F.normalize(U_B, dim=-1)
        alpha = self.hyper_alpha(z_bar) * (1.0 + self.scale_r)  # (B,P)

        Wr_ab = torch.einsum("bpn,bpm->bpnm", U_A, U_B)     # uA uB^T  (B,P,N,N)
        Wr_ba = torch.einsum("bpn,bpm->bpnm", U_B, U_A)     # uB uA^T
        Wr = 0.5 * (Wr_ab + Wr_ba)
        Wr = alpha.view(B, self.P, 1, 1) * Wr
        Wr = 0.5 * (Wr + Wr.transpose(-1, -2))  # 若你始终想保持对称，保留这一行
        Wr = Wr - torch.diag_embed(torch.diagonal(Wr, dim1=-2, dim2=-1))  # 零对角

        # Mask per prior & sum
        S = (Wg_b + Wr) * P_bin                           # (B,P,N,N)
        S = self._sym_zero_diag(S.sum(dim=1))             # (B,N,N)

        # Temperature & sigmoid gate
        logits = torch.clamp(S / self.tau, -6.0, 6.0)
        G = torch.tanh(logits)                         # (B,N,N)
        if mode == "mask":
            return G
        elif mode == "gate":
            assert A is not None, "APF(mode='gate') requires A"
            A_tilde = (1.0 + gamma * G) * A
            return A_tilde, G
        elif mode == "bias":
            bias = beta * G
            return bias, G
        else:
            raise ValueError("mode must be 'mask', 'gate', or 'bias'")

class DualTransformerLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        distill_weights: torch.Tensor,
        dropout: float = 0.1,
        num_priors: int = 4,      # ← 你的 self.priors 里是 4 个
        N_nodes: int = 379,       # ← ROI 数
        apf_tau: float = 2.0,
        apf_use_free: bool = True,
    ):
        super().__init__()
        self.H = hidden_dim
        self.dw = distill_weights

        # –– Layer‑norms ––––––––––––––––––––––––––––––––––––––––––––––– #
        self.ln_attn_FC = nn.LayerNorm(hidden_dim)
        self.ln_attn_SC = nn.LayerNorm(hidden_dim)
        self.ln_ffn_FC  = nn.LayerNorm(hidden_dim)
        self.ln_ffn_SC  = nn.LayerNorm(hidden_dim)

        # –– QKV projections ––––––––––––––––––––––––––––––––––––––––––– #
        self.qkv_FC = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.qkv_SC = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)

        self.num_heads = num_heads
        self.scale = (hidden_dim // num_heads) ** -0.5

        # –– Feed‑forward (two‑layer MLP) ––––––––––––––––––––––––––––––– #
        self.ffn_FC = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_SC = copy.deepcopy(self.ffn_FC)

        # ---- NEW: APF 替换 WeightGenerator ----
        self.apf_FC = AdaptivePriorFusion(
            N=N_nodes, emb_dim=hidden_dim, num_priors=num_priors,
            use_free=apf_use_free, tau=apf_tau
        )
        self.apf_SC = AdaptivePriorFusion(
            N=N_nodes, emb_dim=hidden_dim, num_priors=num_priors,
            use_free=apf_use_free, tau=apf_tau
        )

        # Distillation heads
        self.kl = nn.KLDivLoss(reduction="batchmean")
        self.dist_in_FC  = nn.Linear(hidden_dim, hidden_dim)
        self.dist_in_SC  = nn.Linear(hidden_dim, hidden_dim)
        self.dist_out_FC = nn.Linear(hidden_dim, hidden_dim)
        self.dist_out_SC = nn.Linear(hidden_dim, hidden_dim)

    def _split_heads(self, x: torch.Tensor):
        B, N, _ = x.shape
        return (
            x.view(B, N, self.num_heads, -1)
            .transpose(1, 2)                    # (B, h, N, d_k)
        )

    def _merge_heads(self, x: torch.Tensor):
        B, h, N, d_k = x.shape
        return x.transpose(1, 2).contiguous().view(B, N, h * d_k)

    def _attention(self, q, k, v, gate_mask):
        # q,k,v: (B,h,N,d_k)   gate_mask: (B,N,N) in (0,1)
        attn_scores = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        attn_scores = attn_scores * (1.0 + gate_mask[:, None])    # multiplicative modulation
        attn_probs  = attn_scores.softmax(dim=-1)
        ctx = torch.einsum("bhij,bhjd->bhid", attn_probs, v)
        return ctx

    def forward(
        self,
        x_FC: torch.Tensor,    # (B,N,H)
        x_SC: torch.Tensor,    # (B,N,H)
        pri_FC: torch.Tensor,  # (B,4,N,N)
        pri_SC: torch.Tensor,  # (B,4,N,N)
        FC: torch.Tensor,      # (B,N,N)
        SC: torch.Tensor,      # (B,N,N)
    ):
        # 1) LN + QKV
        z_FC = self.ln_attn_FC(x_FC)
        z_SC = self.ln_attn_SC(x_SC)
        q_FC, k_FC, v_FC = map(self._split_heads, self.qkv_FC(z_FC).chunk(3, dim=-1))
        q_SC, k_SC, v_SC = map(self._split_heads, self.qkv_SC(z_SC).chunk(3, dim=-1))

        # 2) APF 生成每模态的门控 G(z)
        #    可选：如果想 gate 输入 A 再用于注意力，可用 mode="gate" 返回 A_tilde
        G_FC = self.apf_FC(pri_FC, z_FC, mode="mask")         # (B,N,N)
        G_SC = self.apf_SC(pri_SC, z_SC, mode="mask")         # (B,N,N)

        # 3) 注意力 + 残差
        ctx_FC = self._attention(q_FC, k_FC, v_FC, G_FC)
        ctx_SC = self._attention(q_SC, k_SC, v_SC, G_SC)
        ctx_FC = self._merge_heads(ctx_FC)
        ctx_SC = self._merge_heads(ctx_SC)

        attn_out_FC = x_FC + ctx_FC
        attn_out_SC = x_SC + ctx_SC

        # 4) FFN 残差
        ffn_FC = self.ffn_FC(self.ln_ffn_FC(attn_out_FC))
        ffn_SC = self.ffn_SC(self.ln_ffn_SC(attn_out_SC))
        x_FC_new = attn_out_FC + ffn_FC
        x_SC_new = attn_out_SC + ffn_SC

        # 5) 互蒸馏（与你原逻辑一致）
        wf, ws = self.dw
        proj_FC = self.dist_in_FC(x_FC_new)
        proj_SC = self.dist_in_SC(x_SC_new)

        l_FC = F.log_softmax(proj_FC.reshape(-1, self.H), dim=-1)
        p_SC = F.softmax     (proj_SC.reshape(-1, self.H), dim=-1)
        l_SC = F.log_softmax(proj_SC.reshape(-1, self.H), dim=-1)
        p_FC = F.softmax     (proj_FC.reshape(-1, self.H), dim=-1)

        x_FC_new = (1 - wf) * x_FC_new + wf * self.dist_out_FC(proj_FC)
        x_SC_new = (1 - ws) * x_SC_new + ws * self.dist_out_SC(proj_SC)

        distill_loss = 0.5 * (self.kl(l_FC, p_SC) * wf + self.kl(l_SC, p_FC) * ws)

        # 这里返回的 attn_weights_* 原先用于可视化；我们可返回 G_* 作为“prior gate”可视化
        return x_FC_new, x_SC_new, distill_loss, G_FC, G_SC


class BrainTransformers(nn.Module):
    def __init__(self, net_params, args):
        super(BrainTransformers, self).__init__()
        self.args = args
        H          = net_params["hidden_channels"]
        L          = net_params["num_layers"]
        C_out      = net_params["out_channels"]
        dropout    = net_params["dropout"]
        heads_gnn  = net_params.get("gnn_heads", 4)

        # ------------------------------------------------------------------ #
        self.register_buffer(
            "edge_mask_0",
            torch.from_numpy(
                np.load("./dataset/processed/att_connectivity.pkl", allow_pickle=True)
            ).float(),
        )
        self.register_buffer(
            "edge_mask_1",
            torch.from_numpy(
                np.load("./dataset/processed/inhibition_connectivity.pkl", allow_pickle=True)
            ).float(),
        )
        # # random generate a edge mask 1
        # self.edge_mask_1 = torch.rand(379, 379)
        # # > 0.5 -> 1, < 0.5 -> 0
        # self.edge_mask_1 = (self.edge_mask_1 > 0.5).float()

        self.register_buffer(
            "edge_mask_2",
            torch.from_numpy(
                np.load("./dataset/processed/updating_connectivity.pkl", allow_pickle=True)
            ).float(),
        )

        self.register_buffer(
            "edge_mask_3",
            torch.from_numpy(
                np.load("./dataset/processed/performance_connectivity.pkl", allow_pickle=True)
            ).float(),
        )

        self.priors = torch.stack(
            [self.edge_mask_0, self.edge_mask_1, self.edge_mask_2, self.edge_mask_3], dim=0
        )  # (4, 379, 379)
        N_nodes = self.priors.shape[-1]
        num_priors = self.priors.shape[0]
        # Input embedding & stacked dual‑transformers                        #
        self.input_proj = nn.Linear(net_params["in_channels"], H)
        # one distill‑weight pair per layer (you can also register nn.Parameter)
        layer_dw = torch.tensor([
            [0.1, 0.9],   # 第 1 层
            [0.4, 0.6],   # 第 2 层
            [0.7, 0.3],   # 第 3 层
        ])

        self.layers = nn.ModuleList(
            [
                DualTransformerLayer(
                    hidden_dim=H,
                    num_heads=8,
                    distill_weights=layer_dw[i],   # 每层用不同的权重
                    dropout=dropout,
                    num_priors=num_priors,
                    N_nodes=N_nodes,
                    apf_tau=2.0,
                    apf_use_free=True,
                )
                for i in range(len(layer_dw))
            ]
        )

        # ------------------------------------------------------------------ #
        # Two GNN layers for final message passing                           #
        self.conv1 = TransformerConv(
            H, H, heads=heads_gnn, dropout=dropout, edge_dim=1, beta=True
        )
        self.fc_encode = nn.Linear(H, H)                                     # skip‑connection

        self.conv2 = TransformerConv(
            H, H, heads=heads_gnn, dropout=dropout, edge_dim=1, beta=True
        )
        self.fc = nn.Linear(H, C_out)


    def forward(self, data):
        """
        Backbone: 
        1. input: FC, SC, three types of priors, 
        2. input processing: FC, SC into hidden states, priors need to be changed into masks,
         also FC and SC become masks for transformers as well.
        3. model: each layer contain a transformer, while we have to get the attention out because we are going to modify it.
            FC -> exp, SC -> log, or after we can softmax again
        4. after each mlp of transformers, we also need to do a mutual distill
        data.x        – (E) stacked FC features per node
        data.x_SC     – SC features
        data.edge_index, data.edge_attr
        """
        FC, SC, batch          = data.x, data.x_SC, data.batch
        edge_index, edge_weight = data.edge_index, data.edge_attr

        FC, mask_FC = to_dense_batch(FC, batch)      # (B,N,H), (B,N)
        SC, mask_SC = to_dense_batch(SC, batch)      # (B,N,H), (B,N)
        # 1. FC / SC input embedding  -------------------------------------- #
        FC_h = self.input_proj(FC)                            # (B,N,H)
        SC_h = self.input_proj(SC)

        B, N, _ = FC_h.shape
        pri_FC = self.priors.unsqueeze(0).expand(B, -1, -1, -1).to(FC_h.device)  # (B,3,N,N)
        pri_SC = pri_FC
        # 4. Stacked transformer layers with distillation ------------------- #
        extra_loss = 0.0
        att_FCs = []
        att_SCs = []
        x_FC, x_SC = FC_h, SC_h
        for layer in self.layers:
            x_FC, x_SC, dl, att_FC, att_SC = layer(x_FC, x_SC, pri_FC, pri_SC, FC, SC)
            att_FCs.append(att_FC)
            att_SCs.append(att_SC)
            extra_loss += dl
        
        # 5. Flatten padded nodes back to COO form -------------------------- #
        x_FC_flat = x_FC[mask_FC]                                            # (E,H)  same idx‑order as batch
        x_SC_flat = x_SC[mask_SC]

        x = x_FC_flat + x_SC_flat

        g = global_mean_pool(x, batch)                     # (B,H)
        out = self.fc(g)                                                    # (B,C_out)
        att_FCs = torch.stack(att_FCs, dim=1)  # (B,L,N,N)
        att_SCs = torch.stack(att_SCs, dim=1)
        if len(att_FCs.shape) == 3:
            att_FCs = att_FCs.unsqueeze(0)
            att_SCs = att_SCs.unsqueeze(0)
        return out, extra_loss * 0.1, att_FCs, att_SCs
    
    def plot_edge(self, edge_index, edge_weight):
        edge_index_first = edge_index[:, edge_index[0] < 379]
        edge_index_first = edge_index_first[:, edge_index_first[1] < 379]

        adj = torch.zeros((379, 379))
        adj[edge_index_first[0], edge_index_first[1]] = edge_weight.cpu()[:edge_index_first.shape[1]]

        # use plot_edge_weight function to plot the adj
        plot_edge_weight(adj, adj>0, 0, 0)



class DualTransformerLayerOLD(nn.Module):
    """
    One layer that simultaneously processes FC and SC streams, injects
    anatomical‑prior weights into the attention, and performs symmetric
    KL‑based mutual distillation.

    Args
    ----
    hidden_dim      : model width H
    num_heads       : multi‑head attention heads
    distill_weights : tensor([w_FC, w_SC]) – how much to trust the distilled branch
    dropout         : dropout probability used after attention & MLP
    """
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        distill_weights: torch.Tensor,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.H = hidden_dim
        self.dw = distill_weights

        # –– Layer‑norms ––––––––––––––––––––––––––––––––––––––––––––––– #
        self.ln_attn_FC = nn.LayerNorm(hidden_dim)
        self.ln_attn_SC = nn.LayerNorm(hidden_dim)
        self.ln_ffn_FC  = nn.LayerNorm(hidden_dim)
        self.ln_ffn_SC  = nn.LayerNorm(hidden_dim)

        # –– QKV projections ––––––––––––––––––––––––––––––––––––––––––– #
        self.qkv_FC = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.qkv_SC = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)

        self.num_heads = num_heads
        self.scale = (hidden_dim // num_heads) ** -0.5

        # –– Feed‑forward (two‑layer MLP) ––––––––––––––––––––––––––––––– #
        self.ffn_FC = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_SC = copy.deepcopy(self.ffn_FC)

        # –– Prior‑weight generator (same as before) ––––––––––––––––––– #
        self.weight_gen = WeightGenerator(hidden_dim, num_priors=3, num_heads=4)

        # –– Distillation loss ––––––––––––––––––––––––––––––––––––––––– #
        self.kl = nn.KLDivLoss(reduction="batchmean")

        self.dist_in_FC = nn.Linear(hidden_dim, hidden_dim)
        self.dist_in_SC = nn.Linear(hidden_dim, hidden_dim)
        self.dist_out_FC = nn.Linear(hidden_dim, hidden_dim)
        self.dist_out_SC = nn.Linear(hidden_dim, hidden_dim)

    # ------------------------------------------------------------------ #
    def _split_heads(self, x: torch.Tensor):
        B, N, _ = x.shape
        return (
            x.view(B, N, self.num_heads, -1)
            .transpose(1, 2)                    # (B, h, N, d_k)
        )

    def _merge_heads(self, x: torch.Tensor):
        B, h, N, d_k = x.shape
        return x.transpose(1, 2).contiguous().view(B, N, h * d_k)

    # ------------------------------------------------------------------ #
    def _attention(self, q, k, v, mask):
        # q,k,v : (B,h,N,d_k)   mask : (B,N,N)
        attn_scores = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        attn_scores = attn_scores * (1 + mask[:, None])               # broadcast over heads
        attn_probs  = attn_scores.softmax(dim=-1)
        ctx         = torch.einsum("bhij,bhjd->bhid", attn_probs, v)
        return ctx                                             # (B,h,N,d_k)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        x_FC: torch.Tensor,          # (B,N,H)
        x_SC: torch.Tensor,          # (B,N,H)
        pri_FC: torch.Tensor,        # (B,3,N,N)   anatomical priors
        pri_SC: torch.Tensor,        # (B,3,N,N)
        FC: torch.Tensor,          # (B,N,N)     original FC features
        SC: torch.Tensor,          # (B,N,N)     original SC features
    ):
        # ========= 1. SELF‑ATTENTION WITH PRIOR MASK =================== #
        # LN
        z_FC = self.ln_attn_FC(x_FC)
        z_SC = self.ln_attn_SC(x_SC)

        # QKV
        qkv_FC   = self.qkv_FC(z_FC).chunk(3, dim=-1)
        qkv_SC   = self.qkv_SC(z_SC).chunk(3, dim=-1)

        q_FC, k_FC, v_FC = map(self._split_heads, qkv_FC)
        q_SC, k_SC, v_SC = map(self._split_heads, qkv_SC)

        # Prior masks (multiply then log‑softmax)
        w_FC, attn_weights_FC, fused_nh_FC = self.weight_gen(pri_FC, z_FC, FC)  # (B,N,N)
        w_SC, attn_weights_SC, fused_nh_SC = self.weight_gen(pri_SC, z_SC, SC) 

        ctx_FC = self._attention(q_FC, k_FC, v_FC, w_FC)
        ctx_SC = self._attention(q_SC, k_SC, v_SC, w_SC)

        ctx_FC = self._merge_heads(ctx_FC)
        ctx_SC = self._merge_heads(ctx_SC)

        attn_out_FC = x_FC + ctx_FC + fused_nh_FC           # residual‑1
        attn_out_SC = x_SC + ctx_SC + fused_nh_SC

        # ========= 2. FEED‑FORWARD =============== #
        ffn_FC = self.ffn_FC(self.ln_ffn_FC(attn_out_FC))
        ffn_SC = self.ffn_SC(self.ln_ffn_SC(attn_out_SC))

        x_FC_new = attn_out_FC + ffn_FC       # residual‑2
        x_SC_new = attn_out_SC + ffn_SC

        # ========= 3. MUTUAL DISTILLATION ======== #
        wf, ws = self.dw
        proj_FC = self.dist_in_FC(x_FC_new)
        proj_SC = self.dist_in_SC(x_SC_new)

        l_FC = F.log_softmax(proj_FC.reshape(-1, self.H), dim=-1)
        p_SC = F.softmax     (proj_SC.reshape(-1, self.H), dim=-1)
        l_SC = F.log_softmax(proj_SC.reshape(-1, self.H), dim=-1)
        p_FC = F.softmax     (proj_FC.reshape(-1, self.H), dim=-1)

        x_FC_new = (1 - wf) * x_FC_new + wf * self.dist_out_FC(proj_FC)
        x_SC_new = (1 - ws) * x_SC_new + ws * self.dist_out_SC(proj_SC)

        distill_loss = 0.5 * (self.kl(l_FC, p_SC) * wf + self.kl(l_SC, p_FC) * ws)
        return x_FC_new, x_SC_new, distill_loss, attn_weights_FC, attn_weights_SC