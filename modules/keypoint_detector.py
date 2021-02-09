from torch import nn
import torch
from modules.util import Hourglass, make_coordinate_grid, AntiAliasInterpolation2d
import matplotlib.pyplot as plt
import imageio
import os


def viz_sum_old(x, m):
    viz = m.detach()
    viz = viz.permute(1, 2, 3, 0)
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(x.permute(2, 3, 1, 0).squeeze())
    fig.add_subplot(1, 2, 2)
    plt.imshow(viz[0])
    plt.show()


def viz_sum(m):
    viz = m.detach()
    viz = viz.permute(1, 2, 3, 0)
    viz = viz.detach()
    image = viz[0]
    imageio.imsave('debug/rec.png', image)


class KPDetector(nn.Module):
    """
    Detecting a keypoints. Return keypoint position and jacobian near each keypoint.
    """

    def __init__(self, checkpoint_with_kp, block_expansion, num_kp, num_channels, max_features,
                 num_blocks, temperature, estimate_jacobian=False, scale_factor=1,
                 single_jacobian_map=False, pad=0):
        super(KPDetector, self).__init__()

        self.predictor = Hourglass(block_expansion, in_features=num_channels,
                                   max_features=max_features, num_blocks=num_blocks)

        self.kp = nn.Conv2d(in_channels=self.predictor.out_filters, out_channels=num_kp, kernel_size=(7, 7),
                            padding=pad)

        if estimate_jacobian:
            self.num_jacobian_maps = 1 if single_jacobian_map else num_kp
            self.jacobian = nn.Conv2d(in_channels=self.predictor.out_filters,
                                      out_channels=4 * self.num_jacobian_maps, kernel_size=(7, 7), padding=pad)
            self.jacobian.weight.data.zero_()
            self.jacobian.bias.data.copy_(torch.tensor([1, 0, 0, 1] * self.num_jacobian_maps, dtype=torch.float))
        else:
            self.jacobian = None

        self.temperature = temperature
        self.scale_factor = scale_factor
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)

        self.load_state_dict(checkpoint_with_kp['kp_detector'])

    def gaussian2kp(self, heatmap):
        """
        Extract the mean and from a heatmap
        """
        shape = heatmap.shape
        heatmap = heatmap.unsqueeze(-1)
        grid = make_coordinate_grid(shape[2:], heatmap.type()).unsqueeze_(0).unsqueeze_(0)
        value = (heatmap * grid).sum(dim=(2, 3))
        kp = {'value': value}

        return kp

    def forward(self, x):
        if self.scale_factor != 1:
            x = self.down(x)

        feature_map = self.predictor(x)
        prediction = self.kp(feature_map)
        out = prediction.sum(1)
        pad = (x.shape[2] - out.shape[1]) // 2
        pad_layer = torch.nn.ZeroPad2d(pad)
        out = pad_layer(out)
        out = out.unsqueeze(1)
        return out
