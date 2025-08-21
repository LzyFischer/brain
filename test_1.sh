#!/bin/bash
set -euo pipefail

mkdir -p logs

# ====== 按你的真实数据集名修改这四个变量 ======
# DISEASE_ATT="SC_FC_att_selectedlabel_disease"          # TODO: 改成真实 disease 版 ATT 的 data_name
DISEASE_ADHD="disease_ADHD"      # TODO: 改成真实 disease 版 ADHD 的 data_name
DISEASE_OCD="disease_OCD"       # TODO: 改成真实 disease 版 OCD 的 data_name
DISEASE_ANX="disease_Anxiety"    # TODO: 改成真实 disease 版 Anxiety 的 data_name
# ===================================================

run_set() {
  local data_name="$1"   # e.g., SC_FC_att_selectedlabel
  local tag="$2"         # e.g., att
  local suffix="$3"      # "" for ques, "_disease" for disease

  python scripts/main_brain.py --config GCN_classification --data_name "${data_name}" \
      > "logs/SC_FC_${tag}_GCN${suffix}.log" 2>&1

  python scripts/main_brain.py --config Transformers_classification --data_name "${data_name}" \
      > "logs/SC_FC_${tag}_Transformers${suffix}.log" 2>&1

  python scripts/main_brain.py --config MLP_classification --data_name "${data_name}" \
      > "logs/SC_FC_${tag}_MLP${suffix}.log" 2>&1
}

# # ---------------- ques（question-only） ----------------
# run_set "SC_FC_att_selectedlabel"           "att"      ""
# run_set "SC_FC_adhd_selectedlabel_4"        "adhd"     ""
# run_set "SC_FC_ocd_selectedlabel_15"        "ocd"      ""
# run_set "SC_FC_anxiety_selectedlabel_3"     "anxiety"  ""

# ---------------- disease ----------------
# run_set "${DISEASE_ATT}"   "att"      "_disease"
run_set "${DISEASE_ADHD}"  "adhd"     "_disease"
run_set "${DISEASE_OCD}"   "ocd"      "_disease"
run_set "${DISEASE_ANX}"   "anxiety"  "_disease"