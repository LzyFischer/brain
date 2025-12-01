#!/bin/bash

seeds=(1 3 4 5)
datasets=("SC_FC_att_selectedlabel" "disease_ADHD" "disease_OCD" "disease_Anx_nosf")
configs=("GCN_classification" "Transformers_classification")

for seed in "${seeds[@]}"; do
  for config in "${configs[@]}"; do
    for data in "${datasets[@]}"; do
      log_name="logs/${data}_${config}_seed_${seed}.log"
      echo "Running: $config on $data with seed $seed"
      python scripts/main_brain.py --config "$config" --data_name "$data" --seed "$seed" \
        > "$log_name" 2>&1
    done
  done
done