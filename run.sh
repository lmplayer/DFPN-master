#!/usr/local/env bash
TF_CUDNN_USE_AUTOTUNE=0 python3 eval_curve.py --test_data_path=./debug_img
