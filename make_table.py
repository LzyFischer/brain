#!/usr/bin/env python3
import re, glob, os
from collections import defaultdict

# 任务名与列顺序
TASK_ORDER = ["ADHD", "Anxiety", "ATT", "OCD"]
MODEL_ORDER = ["MLP", "Masked\\_GCN", "Ours"]  # 列顺序：MLP, Masked_GCN, Ours

# 文件名里关键字 -> 任务
TASK_MAP = {
    "att":     "ATT",
    "adhd":    "ADHD",
    "ocd":     "OCD",
    "anxiety": "Anxiety",
}

# 文件名里关键字 -> 模型列名
MODEL_MAP = {
    "MLP":          "MLP",
    "GCN":          "Masked\\_GCN",
    "Transformers": "Ours",
}

# 设置：含有 "_disease" 视为 disease，否则 ques
def setting_from_fname(name: str) -> str:
    return "disease" if "_disease" in name else "ques"

AVG_RE = re.compile(r"Average test result:\s*([0-9.]+)\s*(?:±|\+/-|\+-)\s*([0-9.]+)")

def parse_log(fp):
    mean, std = None, None
    with open(fp, "r", errors="ignore") as f:
        for line in f:
            m = AVG_RE.search(line)
            if m:
                mean, std = float(m.group(1)), float(m.group(2))
    return mean, std

def task_from_fname(name):
    lower = name.lower()
    for k, v in TASK_MAP.items():
        if k in lower:
            return v
    return None

def model_from_fname(name):
    for k, v in MODEL_MAP.items():
        if k in name:
            return v
    return None

def latex_cell(mean, std, bold=False):
    if mean is None or std is None:
        return r"--"
    cell = f"{mean:.4f}~\\footnotesize{{$({chr(92)}pm {std:.4f})$}}"
    if bold:
        cell = f"\\textbf{{{mean:.4f}}}~\\footnotesize{{$({chr(92)}pm {std:.4f})$}}"
    return cell

def row_for_setting(row_dict, setting):
    """row_dict: model->(mean,std) for one task+setting"""
    # 找该行的最优 mean 用于加粗
    best = None
    for m in MODEL_ORDER:
        if m in row_dict and row_dict[m][0] is not None:
            best = row_dict[m][0] if best is None else max(best, row_dict[m][0])
    cells = []
    for m in MODEL_ORDER:
        ms = row_dict.get(m, (None, None))
        bold = (ms[0] is not None and best is not None and abs(ms[0]-best) < 1e-12)
        cells.append(latex_cell(*ms, bold=bold))
    return cells

def main():
    # data[task][setting][model] = (mean,std)
    data = defaultdict(lambda: defaultdict(dict))

    for fp in glob.glob("logs/*.log"):
        base = os.path.basename(fp)
        task = task_from_fname(base)
        model = model_from_fname(base)
        setting = setting_from_fname(base)
        if not task or not model:
            continue
        mean, std = parse_log(fp)
        data[task][setting][model] = (mean, std)

    lines = []
    for task in TASK_ORDER:
        # 每个 task 两行：ques / disease
        ques_cells = row_for_setting(data[task].get("ques", {}), "ques")
        dis_cells  = row_for_setting(data[task].get("disease", {}), "disease")
        lines.append(
            rf"\multirow{{2}}{{*}}{{{task}}} " + "\n" +
            rf"& ques     & {ques_cells[0]} & {ques_cells[1]} & {ques_cells[2]} \\" + "\n" +
            rf"& disease  & {dis_cells[0]}  & {dis_cells[1]}  & {dis_cells[2]}  \\"
        )

    # 在任务之间加 \midrule
    block = ("\n\\midrule\n").join(lines)
    print(block)

if __name__ == "__main__":
    main()