#!/bin/bash

# List of data files (without .pkl suffix)
data_files=("SC_FC_adhd_question" "SC_FC_anxiety_question" "SC_FC_att_question" "SC_FC_ocd_question")

# Function to get GPU memory usage
get_gpu_load() {
    local gpu_id=$1
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_id"
}

# Function to choose GPU with lowest memory usage
choose_gpu_with_least_load() {
    gpu_count=$(nvidia-smi --list-gpus | wc -l)
    min_load=$(get_gpu_load 0)
    chosen_gpu=0

    for ((gpu_id = 0; gpu_id < gpu_count; gpu_id++)); do
        load=$(get_gpu_load $gpu_id)
        if ((load < min_load)); then
            min_load=$load
            chosen_gpu=$gpu_id
        fi
    done

    echo "$chosen_gpu"
}

# Loop through each data file
for data_name in "${data_files[@]}"
do
    # Auto-detect number of labels from the data file
    label_count=$(python -c "
import pickle
with open('./dataset/processed/${data_name}.pkl', 'rb') as f:
    d = pickle.load(f)
print(len(d[0]['label']))
")

    echo "Detected $label_count labels in $data_name"

    # Loop through each label
    for ((label_index = 0; label_index < $label_count; label_index++))
    do
        # Select GPU with least load
        chosen_gpu=$(choose_gpu_with_least_load)
        export CUDA_VISIBLE_DEVICES=$chosen_gpu
        echo "Selected GPU $chosen_gpu for $data_name label $label_index"

        # Log and output file
        info="Data: ${data_name}, Label: ${label_index}"
        echo "Starting ${info}"
        output_file="results/bash_logs/${data_name}_label_${label_index}.log"

        # Launch training
        nohup python scripts/main_brain.py \
            --max_epochs 30 \
            --data_name $data_name \
            --label_index $label_index \
            --device cuda:0 > $output_file 2>&1 &

        pid=$!
        wait $pid
    done
done