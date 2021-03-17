
https://user-images.githubusercontent.com/49984106/111497559-df152780-8749-11eb-9527-f582bdf3c902.mp4

# animation-with-keypoint-mask
The right most square is the final result. Softmax mask (circles):\
<img src="examples/softmax003/first.gif" width="512">
<img src="examples/softmax003/second.gif" width="512">\
Heatmap mask:\
<img src="examples/heatmap/first.gif" width="512">
<img src="examples/heatmap/second.gif" width="512">



```conda env create -f environment.yml```\
```conda activate venv11```\
Please obtain pretrained keypoint module. You can do so by\
```git checkout fomm```\
Then, follow the instructions from the README of that branch.
### training

to train a model on specific dataset run:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py --config config/dataset_name.yaml --device_ids 0,1,2,3 --checkpoint_with_kp path/to/checkpoint/with/pretrained/kp
```
E.g. `taichi-256-q.yaml` for the keypoint mask based model, and `taichi-256-softmax-q.yaml` for drawn keypoints instead

the code will create a folder in the log directory (each run will create a time-stamped new directory).
checkpoints will be saved to this folder.
to check the loss values during training see ```log.txt```.
you can also check training data reconstructions in the ```train-vis``` sub-folder.
by default the batch size is tuned to run on 4 titan-x gpu (apart from speed it does not make much difference).
You can change the batch size in the train_params in corresponding ```.yaml``` file.

### evaluation on video reconstruction

To evaluate the reconstruction of the driving video from its first frame, run:
```
CUDA_VISIBLE_DEVICES=0 python run.py --config config/dataset_name.yaml --mode reconstruction --checkpoint path/to/checkpoint --checkpoint_with_kp path/to/checkpoint/with/pretrained/kp
```
you will need to specify the path to the checkpoint,
the ```reconstruction``` sub-folder will be created in the checkpoint folder.
the generated video will be stored to this folder, also generated videos will be stored in ```png``` subfolder in loss-less '.png' format for evaluation.
instructions for computing metrics from the paper can be found: https://github.com/aliaksandrsiarohin/pose-evaluation.

### image animation

In order to animate a source image with motion from driving, run:
```
CUDA_VISIBLE_DEVICES=0 python run.py --config config/dataset_name.yaml --mode animate --checkpoint path/to/checkpoint --checkpoint_with_kp path/to/checkpoint/with/pretrained/kp
```
you will need to specify the path to the checkpoint,
the ```animation``` sub-folder will be created in the same folder as the checkpoint.
you can find the generated video there and its loss-less version in the ```png``` sub-folder.
by default video from test set will be randomly paired, but you can specify the "source,driving" pairs in the corresponding ```.csv``` files. the path to this file should be specified in corresponding ```.yaml``` file in pairs_list setting.


### datasets
1) **taichi**. follow the instructions in [data/taichi-loading](data/taichi-loading/readme.md) or instructions from https://github.com/aliaksandrsiarohin/video-preprocessing.


### training on your own dataset
1) resize all the videos to the same size e.g 256x256, the videos can be in '.gif', '.mp4' or folder with images.
we recommend the later, for each video make a separate folder with all the frames in '.png' format. this format is loss-less, and it has better i/o performance.

2) create a folder ```data/dataset_name``` with 2 sub-folders ```train``` and ```test```, put training videos in the ```train``` and testing in the ```test```.

3) create a config ```config/dataset_name.yaml```, in dataset_params specify the root dir the ```root_dir:  data/dataset_name```. also adjust the number of epoch in train_params.

#### additional notes

citation:

