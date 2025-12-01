#!/bin/bash

# ========== SC_FC_att_selectedlabel ==========
# python scripts/main_brain.py --config GCN_classification --data_name SC_FC_att_selectedlabel \
#     > logs/SC_FC_att_GCN.log 2>&1

# python scripts/main_brain.py --config Transformers_classification --data_name SC_FC_att_selectedlabel \
#     > logs/SC_FC_att_Transformers.log 2>&1 &

# python scripts/main_brain.py --config MLP_classification --data_name SC_FC_att_selectedlabel \
#     > logs/SC_FC_att_MLP.log 2>&1


# ========== SC_FC_adhd_selectedlabel_4 ==========
# python scripts/main_brain.py --config GCN_classification --data_name SC_FC_adhd_selectedlabel_4 \
#     > logs/SC_FC_adhd_GCN.log 2>&1

# python scripts/main_brain.py --config Transformers_classification --data_name disease_ADHD

# python scripts/main_brain.py --config MLP_classification --data_name SC_FC_adhd_selectedlabel_4 \
#     > logs/SC_FC_adhd_MLP.log 2>&1




# python scripts/main_brain.py --config Transformers_classification --data_name SC_FC_ocd_selectedlabel_15 \
#     > logs/SC_FC_ocd_Transformers.log 2>&1 &


# CUDA_VISIBLE_DEVICES=1 python scripts/main_brain.py --config Transformers_classification --data_name SC_FC_anxiety_selectedlabel_3 \
#     > logs/SC_FC_anxiety_Transformers.log 2>&1

python scripts/main_brain.py --config Transformers_classification --data_name disease_ADHD --use_personal False \
> logs/disease_ADHD_no_personal.log 2>&1
python scripts/main_brain.py --config Transformers_classification --data_name disease_OCD --use_personal False \
> logs/disease_OCD_no_personal.log 2>&1
python scripts/main_brain.py --config Transformers_classification --data_name SC_FC_att_selectedlabel --use_personal False \
> logs/SC_FC_att_no_personal.log 2>&1
python scripts/main_brain.py --config Transformers_classification --data_name disease_Anx_nosf --use_personal False \
> logs/disease_Anx_nosf_no_personal.log 2>&1