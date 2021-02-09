CUDA_VISIBLE_DEVICES=$1,$2 TORCH_MODEL_ZOO=../torch ../myenv37/bin/python run.py --config config/taichi-256-d.yaml --device_ids 0,1
