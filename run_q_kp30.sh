#!/bin/bash
CUDA_VISIBLE_DEVICES=$1,$2,$3,$4 TORCH_MODEL_ZOO=../torch ../python38 run.py --config config/taichi-256-q-kp30.yaml --device_ids 0,1,2,3
