import torch
from torch import nn
import torch.nn.functional as F
from modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d, Hourglass, AntiAliasInterpolation2d


class Generator(nn.Module):
    def __init__(self, num_channels, num_kp, block_expansion, max_features, num_down_blocks,
                 num_bottleneck_blocks, estimate_occlusion_map=False, dense_motion_params=None,
                 estimate_jacobian=False, scale_factor=0.25, kp_after_softmax=False):
        super(Generator, self).__init__()
        self.source_first = AntiAliasInterpolation2d(num_channels, scale_factor)
        self.first = SameBlock2d(num_channels + 2, 256, kernel_size=(7, 7), padding=(3, 3))
        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        up_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.bottleneck = torch.nn.Sequential()
        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))

        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        self.num_channels = num_channels
        self.hourglass = Hourglass(block_expansion=64, in_features=8, max_features=1024, num_blocks=5)
        self.final_hourglass = nn.Conv2d(in_channels=self.hourglass.out_filters, out_channels=3, kernel_size=(7, 7),
                                         padding=(3, 3))

    def forward(self, source_image, kp_source, kp_driving):
        output_dict = {}
        smaller_source = self.source_first(source_image)
        out = torch.cat((smaller_source, kp_source, kp_driving), dim=1)  # Encoding part
        out = self.first(out)
        out = self.bottleneck(out)  # Decoding part
        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)
        out = self.final(out)
        low_res_prediction = torch.sigmoid(out)
        kp_source_int = F.interpolate(kp_source, size=source_image.shape[2:], mode='bilinear', align_corners=False)
        kp_driving_int = F.interpolate(kp_driving, size=source_image.shape[2:], mode='bilinear', align_corners=False)

        inp = torch.cat(
            (source_image, kp_source_int.detach(), kp_driving_int.detach(), low_res_prediction.detach()),
            dim=1)

        out = self.hourglass(inp)
        out = self.final_hourglass(out)
        upscaled_prediction = torch.sigmoid(out)

        output_dict['kp_source_int'] = kp_source_int.repeat(1, 3, 1, 1)
        output_dict['kp_driving_int'] = kp_driving_int.repeat(1, 3, 1, 1)
        output_dict['low_res_prediction'] = low_res_prediction
        output_dict['upscaled_prediction'] = upscaled_prediction
        return output_dict
