import yaml
import os
import torch
import imageio
import numpy as np
from modules.keypoint_detector import KPDetector
from argparse import ArgumentParser
from time import gmtime, strftime
from skimage.transform import resize
from logger import Visualizer
import matplotlib.pyplot as plot
from matplotlib.pyplot import imshow
from time import sleep
from modules.util import Hourglass, AntiAliasInterpolation2d, make_coordinate_grid, kp2gaussian

import matplotlib.pyplot as plt


def run_kp(config, checkpoint, image: np.ndarray):
    image_np = resize(image, (256, 256))[..., :3]
    image = torch.Tensor(image_np[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
    kp_params = config['model_params']['kp_detector_params']
    common_params = config['model_params']['common_params']  # TODO estimate_jacobian false
    kp_detector = KPDetector(**kp_params, **common_params)

    if kp_detector is not None:
        kp_detector_checkpoint = checkpoint['kp_detector']
        kp_detector.load_state_dict(kp_detector_checkpoint)

    kp_source = kp_detector(image)
    bs, _, h, w = image.shape

    # identity_grid = make_coordinate_grid((h, w), type=kp_source['value'].type())

    v = Visualizer(**config['visualizer_params'])
    kp_t: torch.Tensor = kp_source['value']
    kp_n: np.ndarray = np.squeeze(kp_t.detach().numpy())
    with_kp: np.ndarray = v.draw_image_with_kp(image_np, kp_n) # TODO debug draw_image_with_kp to extract kp_array from there which should be the tensor with the real coordinates (not between -1,1)
    imshow(with_kp)
    plot.show()
    return


def run_kp(config: dict, checkpoint, source: torch.Tensor, driving: torch.Tensor) -> torch.Tensor:
    """
    :param config: see prev function
    :param checkpoint: see prev function
    :param source: downscaled e.g. bsx3x64x64 (but implement for general case) source
    :param driving: downscaled e.g. bsx3x64x64 (but implement for general case) driving
    :return: [source|drawn_source_kp|drawn_driving_kp]_{BS}
    drawn kp should be white circle with size from config on black background
    """
    # torch cat (???)
    # image = np.copy(image) TODO torch??
    # spatial_size = np.array(image.shape[:2][::-1])[np.newaxis]
    # kp_array = spatial_size * (kp_array + 1) / 2
    ...



def main():
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--source", required=True, help="path to source")
    parser.add_argument("--mode", default="train", choices=["train", "reconstruction", "animate"])
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    parser.add_argument("--cpu", default=False, action="store_true", help="Run on cpu")
    parser.set_defaults(verbose=False)

    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f)

    if opt.checkpoint is not None:
        log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
    else:
        log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])
        log_dir += f' {strftime("%d_%m_%y_%H.%M.%S", gmtime())}'
    cpu = opt.cpu
    checkpoint = torch.load(opt.checkpoint, map_location='cpu' if cpu else None)
    source_image = imageio.imread(opt.source)
    # 0 1 2
    # 2 0 1
    # 1 2 0
    # plt.imshow(source_image.permute(0, 2, 3, 1))
    if not cpu:
        source_image = source_image.cuda()
    run_kp(config, checkpoint, source_image)


if __name__ == "__main__":
    print("a")
    #main()
