#!/bin/bash
#SBATCH --job-name=kp30
#SBATCH --output=kp30.out
#SBATCH --error=kp30.err
#SBATCH --partition=studentbatch
#SBATCH --time=1
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4


TORCH_MODEL_ZOO=../torch ../python37 run.py --config config/taichi-256-q-kp30.yaml --device_ids 0,1,2,3

