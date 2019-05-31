import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from mmcv.cnn.weight_init import caffe2_xavier_init

from ..utils import ConvModule
from ..registry import NECKS
GROUP_NUM=32

class RRB(nn.Module):

    def __init__(self, features, out_features=512, ks=1):
        super(RRB, self).__init__()

        self.unify = nn.Conv2d(features, out_features, kernel_size=ks, padding=ks//2, dilation=1, bias=False)
        self.residual = nn.Sequential(nn.Conv2d(out_features, out_features//4, kernel_size=3, padding=1, dilation=1, bias=False),
                                    # InPlaceABNSync(out_features//4),
                                    nn.GroupNorm(GROUP_NUM, out_features//4),
                                    nn.ReLU(True),
                                    nn.Conv2d(out_features//4, out_features, kernel_size=3, padding=1, dilation=1, bias=False))
        self.norm = nn.Sequential(nn.GroupNorm(GROUP_NUM, out_features), 
                                  nn.ReLU(True))

    def forward(self, feats):
        feats = self.unify(feats)
        residual = self.residual(feats)
        feats = self.norm(feats + residual)
        return feats

class CAB(nn.Module):
    def __init__(self, features):
        super(CAB, self).__init__()

        self.conv_reduce1 = nn.Sequential(
                        nn.Conv2d(features, 64, kernel_size=1, bias=False),
                        # InPlaceABNSync(features),
                        nn.GroupNorm(GROUP_NUM, 64),
                        nn.ReLU(True)
                        )
        self.conv_reduce2 = nn.Sequential(
                        nn.Conv2d(features, 64, kernel_size=1, bias=False),
                        # InPlaceABNSync(features),
                        nn.GroupNorm(GROUP_NUM, 64),
                        nn.ReLU(True)
                        )
        self.delta_gen = nn.Conv2d(128, 2, kernel_size=3, padding=1, bias=True)

        self.delta_gen.weight.data.zero_()
        self.delta_gen.bias.data.zero_()

    def bilinear_interpolate_torch_gridsample(self, input, size, delta=0):
        out_h, out_w = size
        n, c, h, w = input.shape
        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        w_list = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h_list = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h_list.unsqueeze(2), w_list.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + delta.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output

    def forward(self, low_stage, high_stage):
        h, w = low_stage.size(2), low_stage.size(3)
        low_stage_reduce = self.conv_reduce1(low_stage)
        high_stage_reduce = self.conv_reduce2(high_stage)
        high_stage_reduce_up = F.interpolate(input=high_stage_reduce, size=(h, w), mode='bilinear', align_corners=True)
        concat = torch.cat((low_stage_reduce, high_stage_reduce_up), 1)
        delta = self.delta_gen(concat)
        high_stage = self.bilinear_interpolate_torch_gridsample(high_stage, (h, w), delta)
        high_stage += low_stage
        return high_stage

@NECKS.register_module
class FAFPN(nn.Module):
    """HRFPN (High Resolution Feature Pyrmamids)

    arXiv: https://arxiv.org/abs/1904.04514

    Args:
        in_channels (list): number of channels for each branch.
        out_channels (int): output channels of feature pyramids.
        num_outs (int): number of output stages.
        pooling_type (str): pooling for generating feature pyramids
            from {MAX, AVG}.
        conv_cfg (dict): dictionary to construct and config conv layer.
        norm_cfg (dict): dictionary to construct and config norm layer.
        with_cp  (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs=5,
                 pooling_type='AVG',
                 conv_cfg=None,
                 norm_cfg=None,
                 with_cp=False):
        super(FAFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg


        self.RRB1a = RRB(in_channels[0], out_channels)
        # self.CAB1 = CAB(out_channels)
        self.RRB1b = RRB(out_channels, out_channels)
        self.RRB2a = RRB(in_channels[1], out_channels)
        self.CAB2 = CAB(out_channels)
        self.RRB2b = RRB(out_channels, out_channels)
        self.RRB3a = RRB(in_channels[2], out_channels)
        self.CAB3 = CAB(out_channels)
        self.RRB3b = RRB(out_channels, out_channels)
        self.RRB4a = RRB(in_channels[3], out_channels)
        self.CAB4 = CAB(out_channels)
        self.RRB4b = RRB(out_channels, out_channels)

        self.cem = nn.Sequential(
                    nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                    nn.Conv2d(in_channels[-1], out_channels, kernel_size=1, bias=False),
                    nn.GroupNorm(GROUP_NUM, out_channels),
                    nn.ReLU(True))
        
        self.fpn_convs = nn.ModuleList()
        for i in range(5):
            self.fpn_convs.append(nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            ))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                caffe2_xavier_init(m)

    def forward(self, inputs):
        assert len(inputs) == self.num_ins
        x1, x2, x3, x4 = inputs
        d1 = self.RRB1a(x1)

        d2 = self.RRB2a(x2)
        d2 = self.CAB2(d1, d2)
        d2 = self.RRB2b(d2)
        
        d3 = self.RRB3a(x3)
        d3 = self.CAB3(d2, d3)
        d3 = self.RRB3b(d3)

        d4 = self.RRB4a(x4)
        d4 = self.CAB4(d3, d4)
        d4 = self.RRB4b(d4)

        d5 = self.cem(x4)
        d5 = d5 + d4
        d5 = self.RRB1b(d5)

        outs = [d5]
        for i in range(1, self.num_outs):
            outs.append(F.interpolate(input=d5, scale_factor=1.0/2**i, mode='bilinear', align_corners=True))
        outputs = []

        for i in range(self.num_outs):
            if outs[i].requires_grad and self.with_cp:
                tmp_out = checkpoint(self.fpn_convs[i], outs[i])
            else:
                tmp_out = self.fpn_convs[i](outs[i])
            outputs.append(tmp_out)
        return tuple(outputs)
