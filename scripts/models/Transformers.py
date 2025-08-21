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
import pdb                        # already in your repo


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

class WeightGenerator(nn.Module):
    """
    Generate a learnable weighting mask for each anatomical prior.

    priors:  (B, P, N, N)         –  three prior adjacency matrices
    x:       (B, N, H)            –  node embeddings for the same graph
    returns: (B, P, N, N)         –  multiplicative attention mask
    """
    def __init__(self, hidden_dim: int, num_priors: int = 3, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_priors = num_priors

        self.prior_mlp = nn.Linear(379, hidden_dim)

        # Fuse prior‑edge features with an expanding of the node states
        self.edge_fuser = MLP(
            in_dim=hidden_dim * 2, hidden_dim=hidden_dim, out_dim=hidden_dim
        )

        # Self‑attention among the *three* priors to produce a single mask
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )

        self.alpha = nn.Parameter(torch.ones(1))  # learnable scaling factor


    def forward(self, priors: torch.Tensor, x: torch.Tensor, connectivity) -> torch.Tensor:
        B, P, N, _ = priors.shape  # P = 3 （num_priors）
        B, _, H = x.shape

        conn_flat   = connectivity.reshape(B, -1)                 # (B, N*N)
        pri_flat    = priors.reshape(B, P, -1)                    # (B, P, N*N)

        conn_norm   = F.normalize(conn_flat, dim=-1)
        pri_norm    = F.normalize(pri_flat,  dim=-1)
        sims        = torch.einsum("bd,bpd->bp", conn_norm, pri_norm)   # (B, P)

        sims_weights = F.softmax(sims, dim=-1) 

        mass = priors.abs().sum(dim=(2, 3), keepdim=True)  # (B, P, 1, 1)
        priors = priors / mass + torch.randn_like(priors) * 1e-6  # add noise to avoid zero division
        priors_emb = self.prior_mlp(priors).squeeze(-1)

        w = sims_weights.view(B, P, 1, 1)                        # (B, P, 1, 1)
        fused_nh = (w * priors_emb).sum(dim=1)
        
        #### above is for output embedding ,below is for attention weights ####
        x_exp      = x.unsqueeze(1).expand(-1, P, -1, -1)                      # (B, P, N, H)
        fused      = torch.cat([priors_emb, x_exp], dim=-1)                    # (B, P, N, 2H)
        fused      = self.edge_fuser(fused) # (B, )

        tokens     = fused.mean(dim=-2) # (B, P, H) mean among 379 nodes
        
        attn_out, attn_weights = self.attn(tokens, tokens, tokens)
        attn_weights = attn_weights.mean(dim=1)  # (B, P) average over heads
        attn_weights_ = F.softmax(attn_weights / 0.01, dim=-1)  # (B, P) normalize

        attn_weights = attn_weights_.unsqueeze(-1).unsqueeze(-1)  # (B, P, 1, 1)
        attn_weights = attn_weights.expand(-1, -1, N, N)          # (B, P, N, N)
        mask_weights = (attn_weights).sum(dim=1)          # (B, N, N)
        mask_weights = self.alpha / 2 * mask_weights + (1 - self.alpha / 2) * connectivity 
        mask_weights = torch.tanh(mask_weights)

        return mask_weights, attn_weights_, fused_nh

class DualTransformerLayer(nn.Module):
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
        proj_FC = wf * ffn_FC + (1 - wf) * x_FC_new
        proj_SC = ws * ffn_SC + (1 - ws) * x_SC_new

        l_FC = F.log_softmax(proj_FC.reshape(-1, self.H), dim=-1)
        p_SC = F.softmax     (proj_SC.reshape(-1, self.H), dim=-1)
        l_SC = F.log_softmax(proj_SC.reshape(-1, self.H), dim=-1)
        p_FC = F.softmax     (proj_FC.reshape(-1, self.H), dim=-1)

        distill_loss = 0.5 * (self.kl(l_FC, p_SC) + self.kl(l_SC, p_FC))
        return x_FC_new, x_SC_new, distill_loss, attn_weights_FC, attn_weights_SC
        


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
        
        # Input embedding & stacked dual‑transformers                        #
        self.input_proj = nn.Linear(net_params["in_channels"], H)

        # one distill‑weight pair per layer (you can also register nn.Parameter)
        layer_dw = torch.full((L, 2), 0.5)                                   # default 0.5/0.5 blend
        self.layers = nn.ModuleList(
            [
                DualTransformerLayer(
                    hidden_dim=H,
                    num_heads=8,
                    distill_weights=torch.tensor([0.5, 0.5]),
                    dropout=dropout,
                )
                for _ in range(L)
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
        return out, extra_loss * 0.1, att_FCs, att_SCs
    
    def plot_edge(self, edge_index, edge_weight):
        edge_index_first = edge_index[:, edge_index[0] < 379]
        edge_index_first = edge_index_first[:, edge_index_first[1] < 379]

        adj = torch.zeros((379, 379))
        adj[edge_index_first[0], edge_index_first[1]] = edge_weight.cpu()[:edge_index_first.shape[1]]

        # use plot_edge_weight function to plot the adj
        plot_edge_weight(adj, adj>0, 0, 0)









'''
class WeightGenerator(nn.Module):
    """
    Generate a learnable weighting mask for each anatomical prior.

    priors:  (B, P, N, N)         –  three prior adjacency matrices
    x:       (B, N, H)            –  node embeddings for the same graph
    returns: (B, P, N, N)         –  multiplicative attention mask
    """
    def __init__(self, hidden_dim: int, num_priors: int = 3, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_priors = num_priors

        self.prior_mlp = nn.Linear(379, hidden_dim)

        # Fuse prior‑edge features with an expanding of the node states
        self.edge_fuser = MLP(
            in_dim=hidden_dim * 2, hidden_dim=hidden_dim, out_dim=hidden_dim
        )

        # Self‑attention among the *three* priors to produce a single mask
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )

        self.alpha = nn.Parameter(torch.ones(1))  # learnable scaling factor


    def forward(self, priors: torch.Tensor, x: torch.Tensor, connectivity) -> torch.Tensor:
        B, P, N, _ = priors.shape  # P = 3 （num_priors）
        B, _, H = x.shape

        conn_flat   = connectivity.reshape(B, -1)                 # (B, N*N)
        pri_flat    = priors.reshape(B, P, -1)                    # (B, P, N*N)

        conn_norm   = F.normalize(conn_flat, dim=-1)
        pri_norm    = F.normalize(pri_flat,  dim=-1)
        sims        = torch.einsum("bd,bpd->bp", conn_norm, pri_norm)   # (B, P)

        sims_weights = F.softmax(sims, dim=-1) 

        priors_emb = self.prior_mlp(priors).squeeze(-1)

        w = sims_weights.view(B, P, 1, 1)                        # (B, P, 1, 1)
        fused_nh = (w * priors_emb).sum(dim=1)
        
        #### above is for output embedding ,below is for attention weights ####
        x_exp      = x.unsqueeze(1).expand(-1, P, -1, -1)                      # (B, P, N, H)
        fused      = torch.cat([priors_emb, x_exp], dim=-1)                    # (B, P, N, 2H)
        fused      = self.edge_fuser(fused) # (B, )

        tokens     = fused.mean(dim=-2) # (B, P, H) mean among 379 nodes
        
        attn_out, attn_weights = self.attn(tokens, tokens, tokens)
        attn_weights = attn_weights.mean(dim=1)  # (B, P) average over heads
        attn_weights_ = attn_weights.softmax(dim=-1)  # (B, P) normalize

        attn_weights = attn_weights_.unsqueeze(-1).unsqueeze(-1)  # (B, P, 1, 1)
        attn_weights = attn_weights.expand(-1, -1, N, N)          # (B, P, N, N)
        mask_weights = (attn_weights * priors).sum(dim=1)          # (B, N, N)
        mask_weights = self.alpha * mask_weights + (1 - self.alpha) * connectivity 
        mask_weights = torch.tanh(mask_weights)

        return mask_weights, attn_weights_, fused_nh
'''

''' tje 
class WeightGenerator(nn.Module):
    """
    Generate a learnable weighting mask for each anatomical prior.

    priors:  (B, P, N, N)         –  three prior adjacency matrices
    x:       (B, N, H)            –  node embeddings for the same graph
    returns: (B, P, N, N)         –  multiplicative attention mask
    """
    def __init__(self, hidden_dim: int, num_priors: int = 3, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_priors = num_priors

        self.prior_mlp = nn.Linear(379, hidden_dim)

        # Fuse prior‑edge features with an expanding of the node states
        self.edge_fuser = MLP(
            in_dim=hidden_dim * 2, hidden_dim=hidden_dim, out_dim=hidden_dim
        )

        # Self‑attention among the *three* priors to produce a single mask
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )

        self.alpha = nn.Parameter(torch.ones(1))  # learnable scaling factor


    def forward(self, priors: torch.Tensor, x: torch.Tensor, connectivity) -> torch.Tensor:
        B, P, N, _ = priors.shape  # P = 3 （num_priors）
        B, _, H = x.shape

        conn_flat   = connectivity.reshape(B, -1)                 # (B, N*N)
        pri_flat    = priors.reshape(B, P, -1)                    # (B, P, N*N)

        conn_norm   = F.normalize(conn_flat, dim=-1)
        pri_norm    = F.normalize(pri_flat,  dim=-1)
        sims        = torch.einsum("bd,bpd->bp", conn_norm, pri_norm)   # (B, P)

        sims_weights = F.softmax(sims, dim=-1) 

        priors_emb = self.prior_mlp(priors).squeeze(-1)

        w = sims_weights.view(B, P, 1, 1)                        # (B, P, 1, 1)
        fused_nh = (w * priors_emb).sum(dim=1)
        
        #### above is for output embedding ,below is for attention weights ####
        unit_k = 512  # 你可根据数据规模/显存调整

        B, P, N, _ = priors.shape
        device = priors.device

        def split_into_subpriors(adj_mat: torch.Tensor, k: int):
            """
            adj_mat: (N, N) 0/1 或非负权重矩阵（建议 0/1）
            返回: list[Tensor(N,N)], 每个是一个子-prior（与 adj_mat 同 dtype/设备）
            策略：对 adj 的非零位置随机打乱，按块大小 k 切成若干子块。
            """
            nz = (adj_mat > 0).nonzero(as_tuple=False)  # (nnz, 2)
            nnz = nz.size(0)
            if nnz == 0:
                # 没有边，仍然返回一个全零子-prior，避免后续维度为 0 的情况
                return [torch.zeros_like(adj_mat)]

            # 子块个数：ceil(nnz / k)
            num_sub = int(math.ceil(nnz / float(k)))
            perm = torch.randperm(nnz, device=adj_mat.device)
            nz = nz[perm]

            subpriors = []
            for s in range(num_sub):
                start = s * k
                end   = min((s + 1) * k, nnz)
                idx   = nz[start:end]
                sub   = torch.zeros_like(adj_mat)
                sub[idx[:, 0], idx[:, 1]] = adj_mat[idx[:, 0], idx[:, 1]]
                subpriors.append(sub)
            return subpriors

        # =============== 基于“子-prior”的注意力权重与 mask 构造 =================
        # 注意：为便于可变长度，我们按 batch 逐个样本计算注意力
        masks_per_b = []
        avg_weights_per_prior_all_b = []  # 仅用于打印/调试

        for b in range(B):
            # 1) 为该样本拆分所有 prior -> 子-prior
            sub_lists = []           # 长度 P 的 list，每个元素是当前 prior 的子-prior 列表
            parent_ids = []          # 展开后每个子-prior 对应的父 prior id，长度为 S (所有子prior 数量)
            for p in range(P):
                subp = split_into_subpriors(priors[b, p], unit_k)
                sub_lists.append(subp)
                parent_ids.extend([p] * len(subp))

            # 展开为统一序列：S 个子-prior
            S = sum(len(lst) for lst in sub_lists)
            # (S, N, N)
            sub_stack = torch.stack([sub for lst in sub_lists for sub in lst], dim=0).to(device)

            # 2) 为每个子-prior 生成 token 表示，再用注意力得到每个子-prior 的标量权重
            #    先用你已有的 edge_fuser 管道得到 (S, N, H) 再对 N 取平均 -> (S, H)
            #    这里需要把 prior 做嵌入：self.prior_mlp 对最后一维 N 做线性投影 -> (S, N, H)
            sub_emb  = self.prior_mlp(sub_stack)              # (S, N, H)
            x_b      = x[b].unsqueeze(0).expand(S, -1, -1)    # (S, N, H)
            fused_b  = torch.cat([sub_emb, x_b], dim=-1)      # (S, N, 2H)
            fused_b  = self.edge_fuser(fused_b)               # (S, N, H)
            tokens   = fused_b.mean(dim=1)                    # (S, H)

            # 使用“聚合查询”从 tokens 上做一轮注意力，得到对 S 个子-prior 的分布
            # 这里用 tokens 的均值作为查询（tgt_len=1），key/val= tokens（src_len=S）
            q = tokens.mean(dim=0, keepdim=True).unsqueeze(0)   # (1, 1, H)
            kv = tokens.unsqueeze(0)                            # (1, S, H)
            attn_out_b, attn_w_b = self.attn(q, kv, kv)         # attn_w_b: (1, 1, S)
            attn_w_b = attn_w_b.squeeze(0).squeeze(0)           # (S,)

            # 归一化成概率分布
            attn_w_b = F.softmax(attn_w_b, dim=-1)              # (S,)

            # 3) 计算最终 mask：所有子-prior 的加权和
            #    (S, N, N) 与 (S,) -> (N, N)
            mask_b = (attn_w_b.view(S, 1, 1) * sub_stack).sum(dim=0)  # (N, N)

            # 4) 统计每个 prior 的子-prior 平均权重（用于打印）
            parent_ids_t = torch.tensor(parent_ids, device=device, dtype=torch.long)
            avg_weights_per_prior = []
            for p in range(P):
                m = (parent_ids_t == p)
                if m.any():
                    avg_weights_per_prior.append(attn_w_b[m].mean().item())
                else:
                    avg_weights_per_prior.append(0.0)
            avg_weights_per_prior_all_b.append(avg_weights_per_prior)

            # 5) 融合 connectivity，并做 [-1,1] 压缩（沿用你之前的 tanh）
            mask_b = self.alpha * mask_b + (1 - self.alpha) * connectivity[b]
            mask_b = torch.tanh(mask_b)
            masks_per_b.append(mask_b)

        # (B, N, N)
        mask_weights = torch.stack(masks_per_b, dim=0)

        # 打印每个样本中，每个 prior 的“子-prior 平均权重”
        # 注意：训练中频繁 print 会很吵，你可加条件判断或 logging
        # for b in range(B):
        #     print(f"[WeightGenerator] batch {b} avg sub-prior weights per prior: {avg_weights_per_prior_all_b[b]}")

        # 我们仍然返回 (B,P) 级别的 attn “父 prior 平均权重”，
        # 便于与原先接口兼容（你也可以同时返回每个子-prior 的分布）
        attn_weights_parent = torch.tensor(avg_weights_per_prior_all_b, device=device, dtype=torch.float)

        return mask_weights, attn_weights_parent, fused_nh
'''