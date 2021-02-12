CUDA_VISIBLE_DEVICES=$1,$2 TORCH_HOME=../torch ../myenv37/bin/python run.py --config config/taichi-256-d.yaml --checkpoint_with_kp ../cpk/taichi-cpk.pth.tar --device_ids 0,1
