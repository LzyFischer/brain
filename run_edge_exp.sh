#!/bin/bash

DATASETS=("SC_FC_adhd_selectedlabel_3" "SC_FC_adhd_selectedlabel_35" "SC_FC_adhd_selectedlabel_4" "SC_FC_adhd_selectedlabel_45" "SC_FC_anxiety_selectedlabel_2" "SC_FC_anxiety_selectedlabel_25" "SC_FC_anxiety_selectedlabel_3" "SC_FC_anxiety_selectedlabel_35" "SC_FC_ocd_selectedlabel_1" "SC_FC_ocd_selectedlabel_15" "SC_FC_ocd_selectedlabel_2" "SC_FC_ocd_selectedlabel_25" "SC_FC_att_selectedlabel_2" "SC_FC_att_selectedlabel_25" "SC_FC_att_selectedlabel_3" "SC_FC_att_selectedlabel_35")

for DATA in "${DATASETS[@]}"
do
    echo "Running training for dataset: $DATA"

    # Choose least loaded GPU dynamically (optional)
    gpu_id=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | \
        awk '{print NR-1, $1}' | sort -k2 -n | head -1 | cut -d' ' -f1)

    export CUDA_VISIBLE_DEVICES=$gpu_id
    echo "Selected GPU: $CUDA_VISIBLE_DEVICES"

    # Run the script
    nohup python scripts/main_brain.py \
        --data_name $DATA \
        --num_run 3 \
        --device cuda:0 \
        --max_epochs 30 \
        --config GCN_classification \
        --task classification \
        --final_metric AUC > results/bash_logs/selectedlabel_${DATA}.log 2>&1 &

    wait
done