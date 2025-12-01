#!/bin/bash

# Seeds you want to run
seeds=(3 4 5)

# Datasets of interest
# datasets=("disease_ADHD" "disease_OCD" "SC_FC_att_selectedlabel" "disease_Anx_nosf")
datasets=("disease_Anx_nosf")

# Extra flags for ablations
flags=(
    # ""                                    # baseline (nothing extra)
    "--use_global False"
    "--use_personal False"
    "--disable_mutual_distill True"
)

# Run all combinations
for seed in "${seeds[@]}"; do
  for data in "${datasets[@]}"; do
    for flag in "${flags[@]}"; do

      # Generate a suffix for the log filename based on the flag
      if [[ "$flag" == *"use_global False"* ]]; then
        suffix="no_global"
      elif [[ "$flag" == *"use_personal False"* ]]; then
        suffix="no_personal"
      elif [[ "$flag" == *"disable_mutual_distill True"* ]]; then
        suffix="no_mutual_distill"
      else
        suffix="baseline"
      fi

      log_name="logs/${data}_${suffix}_seed_${seed}.log"

      echo "Running: data=$data, seed=$seed, flag=$suffix"
      python scripts/main_brain.py \
        --config Transformers_classification \
        --data_name "$data" \
        --seed "$seed" \
        $flag \
        > "$log_name" 2>&1

    done
  done
done