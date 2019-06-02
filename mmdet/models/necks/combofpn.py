import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from mmcv.cnn.weight_init import caffe2_xavier_init

from ..utils import ConvModule
from ..registry import NECKS

from .fpn import FPN
from .fafpn import FAFPN


@NECKS.register_module
class COMBOFPN(nn.Module):
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
                 num_outs=[5, 1],
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None,
                 with_cp=False):
        super(COMBOFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.num_ins = len(in_channels)

        self.fpn = FPN(in_channels, out_channels[0], num_outs[0], start_level, end_level, \
                       add_extra_convs, extra_convs_on_inputs, relu_before_extra_convs, conv_cfg, norm_cfg, activation)
        self.fafpn = FAFPN(in_channels, out_channels[1], num_outs[1], with_cp)

    def init_weights(self):
        self.fpn.init_weights()
        self.fafpn.init_weights()

    def forward(self, inputs):
        return self.fpn(inputs), self.fafpn(inputs)
