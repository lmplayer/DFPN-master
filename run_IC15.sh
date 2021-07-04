#!/usr/local/env bash
TF_CUDNN_USE_AUTOTUNE=0 python3 -u eval_curve.py \
--test_data_path=../../datasets/icdar15_test/ \
--text_scale=512 \
--gpu_list=$1 \
--checkpoint_path=./checkpoints/resnet_v1_50-model.ckpt-1 \
--output_dir=./results_IC15/ \
--resize_ratio=1.5 \
--no_write_images=True \
--link_method=Box \
--use_2branch=False \
--max_side_len=2400
