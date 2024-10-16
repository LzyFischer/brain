# !/bin/bash


# node
for if_pos_weight in False 
do
    for modality in "SC" "FC"
    do
        for c in Nan 
        do
            for d in Nan
            do 
                for e in Nan
                do
                    # Function to get GPU utilization for a given GPU ID
                    get_gpu_load() {
                        local gpu_id=$1
                        local load=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_id")
                        printf "%d" "$load"
                        echo "$load"
                    }

                    # Function to choose the GPU with the least load
                    choose_gpu_with_least_load() {
                        gpu_count=$(nvidia-smi --list-gpus | wc -l)
                        if [ $gpu_count -eq 0 ]; then
                            echo "No GPUs available."
                            exit 1
                        fi

                        # Initialize variables
                        min_load=$(get_gpu_load 0)
                        chosen_gpu=""

                        # Loop through available GPUs
                        for ((gpu_id = 0; gpu_id < $gpu_count; gpu_id++)); do
                            load=$(get_gpu_load $gpu_id)
                            if [ -z "$load" ]; then
                                printf "Unable to determine GPU load for GPU %d.\n" $gpu_id
                                continue
                            fi

                            if ((load <= min_load)); then
                                min_load=$load
                                chosen_gpu=$gpu_id
                            fi
                        done

                        echo "$chosen_gpu"
                    }

                    # Choose GPU with the least load
                    chosen_gpu=$(choose_gpu_with_least_load)

                    if [ -z "$chosen_gpu" ]; then
                        echo "No available GPUs or unable to determine GPU load."
                        exit 1
                    fi


                    echo "Selected GPU: $chosen_gpu"

                    # Set the CUDA_VISIBLE_DEVICES environment variable to restrict execution to the chosen GPU
                    export CUDA_VISIBLE_DEVICES=$chosen_gpu


                    info="if_pos_weight: ${if_pos_weight}, modality: ${modality}"

                    echo "Start ${info}"
                    output_file="logs/if_pos_weight_${if_pos_weight}_modality_${modality}.txt"

                    nohup python scripts/main_brain.py \
                        --if_pos_weight $if_pos_weight \
                        --modality $modality > $output_file 2>&1 &
                    pid=$!
                    sleep 20
                done
            done
        done
    done
done