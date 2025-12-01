# reverse_pt_to_pkl.py
import pickle
import numpy as np
import torch
from types import SimpleNamespace

# 如果文件名不同，请改成你自己的类文件名/导入方式
from scripts.datasets.Brain_dataset import Brain

pt_path  = "data/processed"   # 你的 .pt 文件
out_pkl  = "recovered_dict.pkl"

# 不要设置 label_index，否则会得到被挑列后的 y
args = SimpleNamespace(data_name="SC_FC_att_selectedlabel", threshold=0.0, label_index=None)

# 直接用 processed_path 指向现成的 .pt；rawdata_path 无需提供
dataset = Brain(
    task="classification",
    x_attributes=None,
    # processed_path=pt_path,
    rawdata_path=None,
    args=args,
)

def data_to_dict(g):
    d = {
        # FC 原样保存在 x 中（形状 N x N）
        "FC": g.x.detach().cpu().numpy().astype(np.float32),

        # SC 在 x_SC 中：先把 log1p 去掉并除以 1e5，得到 [0,1] 归一化后的 SC
        # 想完全恢复原始尺度，需要每个样本的原始 SC.min()/SC.max()，.pt 里没有 -> 只能到这一步
        "SC": (np.expm1(g.x_SC.detach().cpu().numpy()) / 1e5).astype(np.float32),

        # label 最初是标量，这里取第一个元素
        "label": float(g.y.view(-1)[0].item()),
    }
    # feature 如果有，原来在 transform 里做了 unsqueeze(0)，这里去掉那一维
    if getattr(g, "feature", None) is not None and g.feature is not None:
        d["feature"] = g.feature.detach().cpu().numpy().squeeze(0).astype(np.float32)
    return d

recovered = [data_to_dict(dataset[i]) for i in range(len(dataset))]

with open(out_pkl, "wb") as f:
    pickle.dump(recovered, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Saved {len(recovered)} samples to {out_pkl}")