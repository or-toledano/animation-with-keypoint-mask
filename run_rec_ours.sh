CUDA_VISIBLE_DEVICES=$1 TORCH_MODEL_ZOO=../torch ../myenv37/bin/python run.py --config config/taichi-256-downscale.yaml --mode reconstruction --checkpoint "/home/dcor/ronmokady/workshop21/team3/animation-with-keypoint-mask/log/taichi-256-downscale 10_01_21_15.51.28/00000099-checkpoint.pth.tar"
