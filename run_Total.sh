#!/usr/local/env bash
TF_CUDNN_USE_AUTOTUNE=0 python3 -u eval_curve.py \
--test_data_path=../../datasets/total_test/ \
--text_scale=512 \
--gpu_list=$1 \
--checkpoint_path=./checkpoints/resnet_v1_50-model.ckpt-0 \
--output_dir=./results_Total/ \
--network='resnet_v1_50' \
--resize_ratio=1.0 \
--score_map_thresh=0.9 \
--mask_thresh=0.55 \
--no_write_images=True \
--link_method='RegLink' \
--use_2branch=True \
--max_side_len=1280 
