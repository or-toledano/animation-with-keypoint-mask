IMAGEIO_FFMPEG_EXE=/home/dcor/ronmokady/workshop21/team3/ffmpeg-4.3.1-amd64-static/ffmpeg CUDA_VISIBLE_DEVICES=$1 TORCH_HOME=../torch ../python run.py --config config/taichi-256-q.yaml --mode reconstruction --checkpoint "../cpk/taichi-cpk.pth.tar"

