import numpy as np
import torch
import imageio

import os
from skimage.draw import circle

import matplotlib.pyplot as plt
import collections


class Logger:
    def __init__(self, log_dir, checkpoint_freq=100, visualizer_params=None, zfill_num=8, log_file_name='log.txt'):

        self.loss_list = []
        self.cpk_dir = log_dir
        self.visualizations_dir = os.path.join(log_dir, 'train-vis')
        if not os.path.exists(self.visualizations_dir):
            os.makedirs(self.visualizations_dir)
        self.log_file = open(os.path.join(log_dir, log_file_name), 'a')
        self.zfill_num = zfill_num
        self.visualizer = Visualizer(**visualizer_params)
        self.checkpoint_freq = checkpoint_freq
        self.epoch = 0
        self.best_loss = float('inf')
        self.names = None

    def log_scores(self, loss_names):
        loss_mean = np.array(self.loss_list).mean(axis=0)

        loss_string = "; ".join(["%s - %.5f" % (name, value) for name, value in zip(loss_names, loss_mean)])
        loss_string = str(self.epoch).zfill(self.zfill_num) + ") " + loss_string

        print(loss_string, file=self.log_file)
        self.loss_list = []
        self.log_file.flush()

    def visualize_rec(self, inp, out):
        image = self.visualizer.visualize(inp['driving'], inp['source'], out)
        imageio.imsave(os.path.join(self.visualizations_dir, "%s-rec.png" % str(self.epoch).zfill(self.zfill_num)),
                       image)

    def save_cpk(self, emergent=False):
        cpk = {k: v.state_dict() for k, v in self.models.items()}
        cpk['epoch'] = self.epoch
        cpk_path = os.path.join(self.cpk_dir, '%s-checkpoint.pth.tar' % str(self.epoch).zfill(self.zfill_num))
        if not (os.path.exists(cpk_path) and emergent):
            torch.save(cpk, cpk_path)

    @staticmethod
    def load_cpk(checkpoint_path, generator=None, kp_detector=None, optimizer_generator=None):
        if torch.cuda.is_available():
            map_location = None
        else:
            map_location = 'cpu'
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        if generator is not None:
            generator.load_state_dict(checkpoint['generator'])
        if kp_detector is not None:
            kp_detector.load_state_dict(checkpoint['kp_detector'])
        return checkpoint['epoch']

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if 'models' in self.__dict__:
            self.save_cpk()
        self.log_file.close()

    def log_iter(self, losses):
        losses = collections.OrderedDict(losses.items())
        if self.names is None:
            self.names = list(losses.keys())
        self.loss_list.append(list(losses.values()))

    def log_epoch(self, epoch, models, inp, out):
        self.epoch = epoch
        self.models = models
        if (self.epoch + 1) % self.checkpoint_freq == 0:
            self.save_cpk()
        self.log_scores(self.names)
        self.visualize_rec(inp, out)


class Visualizer:
    def __init__(self, draw_border=False, colormap='gist_rainbow'):
        self.draw_border = draw_border
        self.colormap = plt.get_cmap(colormap)

    def create_image_column(self, images):
        if self.draw_border:
            images = np.copy(images)
            images[:, :, [0, -1]] = (1, 1, 1)
        column = np.concatenate(list(images), axis=0)
        return column

    def create_image_grid(self, *args):
        out = []
        for arg in args:
            out.append(self.create_image_column(arg))
        return np.concatenate(out, axis=1)

    def visualize(self, driving, source, out):
        images = []
        source = source.data.cpu()
        source = np.transpose(source, [0, 2, 3, 1])
        images.append(source)
        kp_source_int = out['kp_source_int'].data.cpu().numpy()
        kp_source_int = np.transpose(kp_source_int, [0, 2, 3, 1])
        images.append(kp_source_int)
        driving = driving.data.cpu().numpy()
        driving = np.transpose(driving, [0, 2, 3, 1])
        images.append(driving)
        kp_driving_int = out['kp_driving_int'].data.cpu().numpy()
        kp_driving_int = np.transpose(kp_driving_int, [0, 2, 3, 1])
        images.append(kp_driving_int)
        low_res_prediction = out['low_res_prediction'].data.cpu().numpy()
        upscaled_prediction = out['upscaled_prediction'].data.cpu().numpy()
        low_res_prediction = np.transpose(low_res_prediction, [0, 2, 3, 1])
        upscaled_prediction = np.transpose(upscaled_prediction, [0, 2, 3, 1])
        images.append(low_res_prediction)
        images.append(upscaled_prediction)

        if 'kp_norm' in out:
            kp_norm = out['kp_norm']['value'].data.cpu().numpy()
            images.append(kp_norm)

        image = self.create_image_grid(*images)
        image = (255 * image).astype(np.uint8)
        return image
