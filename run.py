import matplotlib
import os
import sys
import yaml
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy

from frames_dataset import FramesDataset

from modules.generator import Generator
from modules.keypoint_detector import KPDetector, FommKpDetector, VERBOSE

import torch
from train import train
from reconstruction import reconstruction
from animate import animate

if VERBOSE:
    matplotlib.use('TkAgg')
else:
    matplotlib.use('Agg')

if __name__ == '__main__':

    if sys.version_info[0] < 3:
        raise Exception('You must use Python 3 or higher. Recommended version is Python 3.7')

    parser = ArgumentParser()
    parser.add_argument('--config', required=True, help='path to config')
    parser.add_argument('--mode', default='train', choices=['train', 'reconstruction', 'animate'])
    parser.add_argument('--log_dir', default='log', help='path to log into')
    parser.add_argument('--checkpoint_with_kp', required=True, help='path to checkpoint with pretrained kp detector')
    parser.add_argument('--checkpoint', default=None, help='path to checkpoint to restore')
    parser.add_argument('--device_ids', default='0', type=lambda x: list(map(int, x.split(','))),
                        help='Names of the devices comma separated.')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Print model architecture')
    parser.add_argument('--cpu', default=False, action='store_true', help='Run on cpu')
    parser.set_defaults(verbose=False)

    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.safe_load(f)

    if opt.checkpoint is not None:
        log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
    else:
        log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])
        log_dir += ' ' + strftime('%d_%m_%y_%H.%M.%S', gmtime())

    generator = Generator(**config['model_params']['generator_params'],
                          **config['model_params']['common_params'])

    if torch.cuda.is_available():
        generator.to(opt.device_ids[0])
    if opt.verbose:
        print(generator)

    checkpoint_with_kp = torch.load(opt.checkpoint_with_kp, map_location='cpu' if opt.cpu else None)

    kp_detector = KPDetector(checkpoint_with_kp, **config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])

    if torch.cuda.is_available():
        kp_detector.to(opt.device_ids[0])

    if opt.verbose:
        print(kp_detector)

    dataset = FramesDataset(is_train=(opt.mode == 'train'), **config['dataset_params'])

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)

    if opt.mode == 'train':
        print('Training...')
        train(config, generator, kp_detector, opt.checkpoint, log_dir, dataset, opt.device_ids)
    elif opt.mode == 'reconstruction':
        print('Reconstruction...')
        reconstruction(config, generator, kp_detector, opt.checkpoint, log_dir, dataset)
    elif opt.mode == 'animate':
        print('Animate...')
        kp_after_softmax = config['model_params']['kp_detector_params']
        if kp_after_softmax:
            kp_detector = FommKpDetector(checkpoint_with_kp, **config['model_params']['kp_detector_params'],
                                         **config['model_params']['common_params'])
        animate(config, generator, kp_detector, opt.checkpoint, log_dir, dataset, kp_after_softmax)
