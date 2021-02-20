#!/bin/bash
TORCH_MODEL_ZOO=../torch ../python37 run.py --config config/taichi-256-q-kp30.yaml --device_ids 0,1,2,3
