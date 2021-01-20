CUDA_VISIBLE_DEVICES=$1,$2,$3,$4 TORCH_MODEL_ZOO=../torch ../python run.py --config config/taichi-256-q.yaml --device_ids 0,1,2,3
