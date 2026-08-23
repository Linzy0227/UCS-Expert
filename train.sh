#!/usr/bin/env bash

python main.py \
  --model_type vit_b \
  --task_name UCS-Expert \
  --checkpoint sam_ckp/sam_vit_b_01ec64.pth \
  --data_path "${DATA_PATH:-dataset/CoralMask}" \
  --work_dir work_dir \
  --image_size 512 \
  --label_size 512 \
  --num_epochs 24 \
  --batch_size 4 \
  --use_amp
