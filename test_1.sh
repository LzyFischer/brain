#!/bin/bash
# ========== SC_FC_ocd_selectedlabel_15 ==========
# python scripts/main_brain.py --config GCN_classification --data_name SC_FC_ocd_selectedlabel_15 \
#     > logs/SC_FC_ocd_GCN.log 2>&1

python scripts/main_brain.py --config Transformers_classification --data_name SC_FC_ocd_selectedlabel_15 \
    > logs/SC_FC_ocd_Transformers.log 2>&1 &

# python scripts/main_brain.py --config MLP_classification --data_name SC_FC_ocd_selectedlabel_15 \
#     > logs/SC_FC_ocd_MLP.log 2>&1


# ========== SC_FC_anxiety_selectedlabel_3 ==========
# python scripts/main_brain.py --config GCN_classification --data_name SC_FC_anxiety_selectedlabel_3 \
#     > logs/SC_FC_anxiety_GCN.log 2>&1

CUDA_VISIBLE_DEVICES=1 python scripts/main_brain.py --config Transformers_classification --data_name SC_FC_anxiety_selectedlabel_3 \
    > logs/SC_FC_anxiety_Transformers.log 2>&1

# python scripts/main_brain.py --config MLP_classification --data_name SC_FC_anxiety_selectedlabel_3 \
#     > logs/SC_FC_anxiety_MLP.log 2>&1