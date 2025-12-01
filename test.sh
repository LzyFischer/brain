python scripts/main_brain.py \
  --config CrossGNN_classification \
  --data_name disease_Anx_nosf \
  --batch_size 16 \
  --max_epochs 15 \
  --channel 32 \
  --layer 2 \
  --gru 1 \
  --alpha 0.8 \
  --seed 0 \
  --device cuda:0 > logs/CrossGNN_Anx_nosf_seed_0.log 2>&1


  # python scripts/main_brain.py \
  # --config CrossGNN_classification \
  # --data_name disease_OCD \
  # --batch_size 16 \
  # --max_epochs 15 \
  # --channel 32 \
  # --layer 2 \
  # --gru 1 \
  # --alpha 0.8 \
  # --seed 0 \
  # --device cuda:0 > logs/CrossGNN_OCD_seed_0.log 2>&1

  # python scripts/main_brain.py \
  # --config CrossGNN_classification \
  # --data_name SC_FC_att_selectedlabel \
  # --batch_size 16 \
  # --max_epochs 15 \
  # --channel 32 \
  # --layer 2 \
  # --gru 1 \
  # --alpha 0.8 \
  # --seed 0 \
  # --device cuda:0 > logs/CrossGNN_SC_FC_att_selectedlabel_seed_0.log 2>&1