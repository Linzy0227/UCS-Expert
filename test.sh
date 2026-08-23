#!/usr/bin/env bash

python infer.py \
  --model_type vit_b \
  --checkpoint sam_ckp/sam_vit_b_01ec64.pth \
  --resume checkpoint/ucs_b.pth \
  --data_path "${DATA_PATH:-sample}" \
  --save_dir output \
  --image_size 512 \
  --label_size 512 \
  --use_amp \
  --save_pic
