from torch import nn
import torch
from modules.util import Hourglass, make_coordinate_grid, AntiAliasInterpolation2d
import torch.nn.functional as F
import matplotlib.pyplot as plt

VERBOSE = False  # To be used with bs=1 !


def vis(x, m):
    if VERBOSE:
        viz = m.detach()
        viz = viz.permute(1, 2, 0)
        fig = plt.figure()
        fig.add_subplot(1, 2, 1)
        plt.imshow(x.permute(2, 3, 1, 0).squeeze())
        fig.add_subplot(1, 2, 2)
        plt.imshow(viz)
        plt.show()


def vis_10(m):
    if VERBOSE:
        viz = m.detach()
        viz = viz.permute(1, 2, 3, 0)
        fig = plt.figure()

        for i in range(1, 11):
            fig.add_subplot(1, 10, i)
            plt.imshow(viz[i - 1])
        plt.show()


def kp2gaussian(kp, spatial_size, kp_variance):
    """
    Transform a keypoint into gaussian like representation
    """
    mean = kp['value']

    coordinate_grid = make_coordinate_grid(spatial_size, mean.type())
    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2)
    mean = mean.view(*shape)

    mean_sub = (coordinate_grid - mean)

    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)

    return out


def norm_mask(shape, mask):
    old_size = mask.size()
    out = mask.view(mask.size(0), -1)
    mn = out.min(1, keepdim=True)[0]
    out -= mn
    mx = out.max(1, keepdim=True)[0]
    out /= mx
    out = out.view(old_size)

    pad = (shape - out.shape[1]) // 2
    pad_layer = torch.nn.ZeroPad2d(pad)
    out = pad_layer(out)
    out = out.unsqueeze(1)
    return out


def draw_kp(shape, kp, kp_variance=0.003):
    res = kp2gaussian(kp, shape, kp_variance)
    vis_10(res)
    res = res.sum(1)
    return res


class KPDetector(nn.Module):
    """
    Detecting a keypoints. Return keypoint position and jacobian near each keypoint.
    """

    def __init__(self, checkpoint_with_kp, block_expansion, num_kp, kp_after_softmax, num_channels, max_features,
                 num_blocks, temperature, estimate_jacobian=False, scale_factor=1,
                 single_jacobian_map=False, pad=0, softmax_mask=False):
        super(KPDetector, self).__init__()

        self.predictor = Hourglass(block_expansion, in_features=num_channels,
                                   max_features=max_features, num_blocks=num_blocks)

        self.kp = nn.Conv2d(in_channels=self.predictor.out_filters, out_channels=num_kp, kernel_size=(7, 7),
                            padding=pad)

        self.kp_after_softmax = kp_after_softmax
        self.softmax_mask = softmax_mask

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
        final_shape = prediction.shape

        if self.kp_after_softmax:
            heatmap = prediction.view(final_shape[0], final_shape[1], -1)
            heatmap = F.softmax(heatmap / self.temperature, dim=2)
            heatmap = heatmap.view(*final_shape)
            out = self.gaussian2kp(heatmap)
            out = draw_kp(final_shape[2:], out)
        else:
            vis_10(prediction)
            if self.softmax_mask:
                prediction = prediction.view(final_shape[0], final_shape[1], -1)
                prediction = F.softmax(prediction / self.temperature, dim=2)
                prediction = prediction.view(*final_shape)
                vis_10(prediction)
            out = prediction.sum(1)
        vis(x, out)
        out = norm_mask(x.shape[2], out)

        return out


class FommKpDetector(nn.Module):
    """
    Detecting a keypoints. Return keypoint position and jacobian near each keypoint.
    """

    def __init__(self, checkpoint_with_kp, block_expansion, num_kp, num_channels, max_features,
                 num_blocks, temperature, kp_after_softmax, estimate_jacobian=False, scale_factor=1,
                 single_jacobian_map=False, pad=0):
        super(FommKpDetector, self).__init__()

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

        final_shape = prediction.shape
        heatmap = prediction.view(final_shape[0], final_shape[1], -1)
        heatmap = F.softmax(heatmap / self.temperature, dim=2)
        heatmap = heatmap.view(*final_shape)

        out = self.gaussian2kp(heatmap)

        if self.jacobian is not None:
            jacobian_map = self.jacobian(feature_map)
            jacobian_map = jacobian_map.reshape(final_shape[0], self.num_jacobian_maps, 4, final_shape[2],
                                                final_shape[3])
            heatmap = heatmap.unsqueeze(2)

            jacobian = heatmap * jacobian_map
            jacobian = jacobian.view(final_shape[0], final_shape[1], 4, -1)
            jacobian = jacobian.sum(dim=-1)
            jacobian = jacobian.view(jacobian.shape[0], jacobian.shape[1], 2, 2)
            out['jacobian'] = jacobian

        return out
