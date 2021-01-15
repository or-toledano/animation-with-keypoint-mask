CUDA_VISIBLE_DEVICES=$1 TORCH_MODEL_ZOO=../torch ../myenv37/bin/python run.py --config config/taichi-256.yaml --device_ids 0
