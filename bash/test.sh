# #!/bin/bash

# python scripts/main_brain.py --config GCN_classification --data_name SC_FC_att_selectedlabel --seed 1 \
#     > logs/SC_FC_att_GCN_seed_1.log 2>&1

# python scripts/main_brain.py --config Transformers_classification --data_name SC_FC_att_selectedlabel --seed 1 \
#     > logs/SC_FC_att_Transformers_seed_1.log 2>&1 

# python scripts/main_brain.py --config MLP_classification --data_name SC_FC_att_selectedlabel --seed 1 \
#     > logs/SC_FC_att_MLP_seed_1.log 2>&1


# python scripts/main_brain.py --config GCN_classification --data_name disease_ADHD --seed 1 \
#     > logs/SC_FC_adhd_GCN_seed_1.log 2>&1

# python scripts/main_brain.py --config Transformers_classification --data_name disease_ADHD --seed 1 \
#     > logs/SC_FC_adhd_Transformers_seed_1.log 2>&1 

# python scripts/main_brain.py --config MLP_classification --data_name disease_ADHD --seed 1 \
#     > logs/SC_FC_adhd_MLP_seed_1.log 2>&1


# python scripts/main_brain.py --config GCN_classification --data_name disease_OCD --seed 1 \
#     > logs/SC_FC_ocd_GCN_seed_1.log 2>&1

# python scripts/main_brain.py --config Transformers_classification --data_name disease_OCD --seed 1 \
#     > logs/SC_FC_ocd_Transformers_seed_1.log 2>&1 

# python scripts/main_brain.py --config MLP_classification --data_name disease_OCD --seed 1 \
#     > logs/SC_FC_ocd_MLP_seed_1.log 2>&1


# python scripts/main_brain.py --config GCN_classification --data_name disease_Anx_nosf --seed 1 \
#     > logs/SC_FC_anxiety_GCN_seed_1.log 2>&1

# python scripts/main_brain.py --config Transformers_classification --data_name disease_Anx_nosf --seed 1 \
#     > logs/SC_FC_anxiety_Transformers_seed_1.log 2>&1 

# python scripts/main_brain.py --config MLP_classification --data_name disease_Anx_nosf --seed 1 \
#     > logs/SC_FC_anxiety_MLP_seed_1.log 2>&1


python scripts/main_brain.py --config Transformers_classification --data_name SC_FC_att_selectedlabel --seed 1 \
    > logs/aSC_FC_att_Transformers_seed_1.log 2>&1 

python scripts/main_brain.py --config Transformers_classification --data_name disease_Anx_nosf --seed 1 \
    > logs/aSC_FC_anx_Transformers_seed_1.log 2>&1 

python scripts/main_brain.py --config Transformers_classification --data_name disease_ADHD --seed 1 \
    > logs/aSC_FC_adhd_Transformers_seed_1.log 2>&1

python scripts/main_brain.py --config Transformers_classification --data_name disease_OCD --seed 1 \
    > logs/aSC_FC_ocd_Transformers_seed_1.log 2>&1