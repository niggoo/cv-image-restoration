import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
import math
from functools import partial


# greatly inspired from the implementation of
# "https://github.com/facebookresearch/dinov2/blob/main/dinov2/hub/depth/decode_heads.py"
def resize(
    input,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=None,
    warning=False,
):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if (
                    (output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1)
                    and (output_h - 1) % (input_h - 1)
                    and (output_w - 1) % (input_w - 1)
                ):
                    warnings.warn(
                        f"When align_corners={align_corners}, "
                        "the output would more aligned if "
                        f"input size {(input_h, input_w)} is `x+1` and "
                        f"out size {(output_h, output_w)} is `nx+1`"
                    )
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class ConvModule(nn.Module):
    """A conv block that bundles conv/norm/activation layers.

    This block simplifies the usage of convolution layers, which are commonly
    used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
    It is based upon three build methods: `build_conv_layer()`,
    `build_norm_layer()` and `build_activation_layer()`.

    Besides, we add some additional features in this module.
    1. Automatically set `bias` of the conv layer.
    2. Spectral norm is supported.
    3. More padding modes are supported. Before PyTorch 1.5, nn.Conv2d only
    supports zero and circular padding, and we add "reflect" padding mode.

    Args:
        in_channels (int): Number of channels in the input feature map.
            Same as that in ``nn._ConvNd``.
        out_channels (int): Number of channels produced by the convolution.
            Same as that in ``nn._ConvNd``.
        kernel_size (int | tuple[int]): Size of the convolving kernel.
            Same as that in ``nn._ConvNd``.
        stride (int | tuple[int]): Stride of the convolution.
            Same as that in ``nn._ConvNd``.
        padding (int | tuple[int]): Zero-padding added to both sides of
            the input. Same as that in ``nn._ConvNd``.
        dilation (int | tuple[int]): Spacing between kernel elements.
            Same as that in ``nn._ConvNd``.
        groups (int): Number of blocked connections from input channels to
            output channels. Same as that in ``nn._ConvNd``.
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_layer. Bias will be set as True if `norm_layer` is None, otherwise
            False. Default: "auto".
        conv_layer (nn.Module): Convolution layer. Default: None,
            which means using conv2d.
        norm_layer (nn.Module): Normalization layer. Default: None.
        act_layer (nn.Module): Activation layer. Default: nn.ReLU.
        inplace (bool): Whether to use inplace mode for activation.
            Default: True.
        with_spectral_norm (bool): Whether use spectral norm in conv module.
            Default: False.
        padding_mode (str): If the `padding_mode` has not been supported by
            current `Conv2d` in PyTorch, we will use our own padding layer
            instead. Currently, we support ['zeros', 'circular'] with official
            implementation and ['reflect'] with our own implementation.
            Default: 'zeros'.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Common examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
            Default: ('conv', 'norm', 'act').
    """

    _abbr_ = "conv_block"

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias="auto",
        conv_layer=nn.Conv2d,
        norm_layer=None,
        act_layer=nn.ReLU,
        inplace=True,
        with_spectral_norm=False,
        padding_mode="zeros",
        order=("conv", "norm", "act"),
    ):
        super(ConvModule, self).__init__()
        official_padding_mode = ["zeros", "circular"]
        self.conv_layer = conv_layer
        self.norm_layer = norm_layer
        self.act_layer = act_layer
        self.inplace = inplace
        self.with_spectral_norm = with_spectral_norm
        self.with_explicit_padding = padding_mode not in official_padding_mode
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == set(["conv", "norm", "act"])

        self.with_norm = norm_layer is not None
        self.with_activation = act_layer is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == "auto":
            bias = not self.with_norm
        self.with_bias = bias

        if self.with_explicit_padding:
            if padding_mode == "zeros":
                padding_layer = nn.ZeroPad2d
            else:
                raise AssertionError(f"Unsupported padding mode: {padding_mode}")
            self.pad = padding_layer(padding)

        # reset padding to 0 for conv module
        conv_padding = 0 if self.with_explicit_padding else padding
        # build convolution layer
        self.conv = self.conv_layer(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=conv_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        if self.with_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            if order.index("norm") > order.index("conv"):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            norm = partial(norm_layer, num_features=norm_channels)
            self.add_module("norm", norm)
            if self.with_bias:
                from torch.nnModules.batchnorm import _BatchNorm
                from torch.nnModules.instancenorm import _InstanceNorm

                if isinstance(norm, (_BatchNorm, _InstanceNorm)):
                    warnings.warn("Unnecessary conv bias before batch/instance norm")
        else:
            self.norm_name = None

        # build activation layer
        if self.with_activation:
            # nn.Tanh has no 'inplace' argument
            # (nn.Tanh, nn.PReLU, nn.Sigmoid, nn.HSigmoid, nn.Swish, nn.GELU)
            if not isinstance(act_layer, (nn.Tanh, nn.PReLU, nn.Sigmoid, nn.GELU)):
                act_layer = partial(act_layer, inplace=inplace)
            self.activate = act_layer()

        # Use msra init by default
        self.init_weights()

    @property
    def norm(self):
        if self.norm_name:
            return getattr(self, self.norm_name)
        else:
            return None

    def init_weights(self):
        # 1. It is mainly for customized conv layers with their own
        #    initialization manners by calling their own ``init_weights()``,
        #    and we do not want ConvModule to override the initialization.
        # 2. For customized conv layers without their own initialization
        #    manners (that is, they don't have their own ``init_weights()``)
        #    and PyTorch's conv layers, they will be initialized by
        #    this method with default ``kaiming_init``.
        # Note: For PyTorch's conv layers, they will be overwritten by our
        #    initialization implementation using default ``kaiming_init``.
        if not hasattr(self.conv, "init_weights"):
            if self.with_activation and isinstance(self.act_layer, nn.LeakyReLU):
                nonlinearity = "leaky_relu"
                a = 0.01  # XXX: default negative_slope
            else:
                nonlinearity = "relu"
                a = 0
            if hasattr(self.conv, "weight") and self.conv.weight is not None:
                nn.init.kaiming_normal_(
                    self.conv.weight, a=a, mode="fan_out", nonlinearity=nonlinearity
                )
            if hasattr(self.conv, "bias") and self.conv.bias is not None:
                nn.init.constant_(self.conv.bias, 0)
        if self.with_norm:
            if hasattr(self.norm, "weight") and self.norm.weight is not None:
                nn.init.constant_(self.norm.weight, 1)
            if hasattr(self.norm, "bias") and self.norm.bias is not None:
                nn.init.constant_(self.norm.bias, 0)

    def forward(self, x, activate=True, norm=True):
        for layer in self.order:
            if layer == "conv":
                if self.with_explicit_padding:
                    x = self.pad(x)
                x = self.conv(x)
            elif layer == "norm" and norm and self.with_norm:
                x = self.norm(x)
            elif layer == "act" and activate and self.with_activation:
                x = self.activate(x)
        return x


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )
        return x


class ReassembleBlocks(nn.Module):
    """ViTPostProcessBlock, process cls_token in ViT backbone output and
    rearrange the feature vector to feature map.
    Args:
        in_channels (int): ViT feature channels. Default: 768.
        out_channels (List): output channels of each stage.
            Default: [96, 192, 384, 768].
        readout_type (str): Type of readout operation. Default: 'ignore'.
        patch_size (int): The patch size. Default: 16.
    """

    def __init__(self, in_channels=768, out_channels=[96, 192, 384, 768]):
        super(ReassembleBlocks, self).__init__()

        self.projects = nn.ModuleList(
            [
                ConvModule(
                    in_channels=in_channels,
                    out_channels=out_channel,
                    kernel_size=1,
                    act_layer=None,
                )
                for out_channel in out_channels
            ]
        )

        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0],
                    out_channels=out_channels[0],
                    kernel_size=4,
                    stride=4,
                    padding=0,
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1],
                    out_channels=out_channels[1],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3],
                    out_channels=out_channels[3],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )

    def forward(self, inputs):
        assert isinstance(inputs, list)
        out = []
        for i, x in enumerate(inputs):
            x, _ = x  # just ignore the cls token
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)
        return out


class PreActResidualConvUnit(nn.Module):
    """ResidualConvUnit, pre-activate residual unit.
    Args:
        in_channels (int): number of channels in the input feature map.
        act_layer (nn.Module): activation layer.
        norm_layer (nn.Module): norm layer.
        stride (int): stride of the first block. Default: 1
        dilation (int): dilation rate for convs layers. Default: 1.
    """

    def __init__(self, in_channels, act_layer, norm_layer, stride=1, dilation=1):
        super(PreActResidualConvUnit, self).__init__()

        self.conv1 = ConvModule(
            in_channels,
            in_channels,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            norm_layer=norm_layer,
            act_layer=act_layer,
            bias=False,
            order=("act", "conv", "norm"),
        )

        self.conv2 = ConvModule(
            in_channels,
            in_channels,
            3,
            padding=1,
            norm_layer=norm_layer,
            act_layer=act_layer,
            bias=False,
            order=("act", "conv", "norm"),
        )

    def forward(self, inputs):
        inputs_ = inputs.clone()
        x = self.conv1(inputs)
        x = self.conv2(x)
        return x + inputs_


class FeatureFusionBlock(nn.Module):
    """FeatureFusionBlock, merge feature map from different stages.
    Args:
        in_channels (int): Input channels.
        act_layer (nn.Module): activation layer for ResidualConvUnit.
        norm_layer (nn.Module): normalization layer.
        expand (bool): Whether expand the channels in post process block.
            Default: False.
        align_corners (bool): align_corner setting for bilinear upsample.
            Default: True.
    """

    def __init__(
        self, in_channels, act_layer, norm_layer, expand=False, align_corners=True
    ):
        super(FeatureFusionBlock, self).__init__()

        self.in_channels = in_channels
        self.expand = expand
        self.align_corners = align_corners

        self.out_channels = in_channels
        if self.expand:
            self.out_channels = in_channels // 2

        self.project = ConvModule(
            self.in_channels,
            self.out_channels,
            kernel_size=1,
            act_layer=None,
            bias=True,
        )

        self.res_conv_unit1 = PreActResidualConvUnit(
            in_channels=self.in_channels, act_layer=act_layer, norm_layer=norm_layer
        )
        self.res_conv_unit2 = PreActResidualConvUnit(
            in_channels=self.in_channels, act_layer=act_layer, norm_layer=norm_layer
        )

    def forward(self, *inputs):
        x = inputs[0]
        if len(inputs) == 2:
            if x.shape != inputs[1].shape:
                res = resize(
                    inputs[1],
                    size=(x.shape[2], x.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                res = inputs[1]
            x = x + self.res_conv_unit1(res)
        x = self.res_conv_unit2(x)
        x = resize(x, scale_factor=2, mode="bilinear", align_corners=self.align_corners)
        x = self.project(x)
        return x


class DPT(nn.Module):
    """Vision Transformers for Dense Prediction.
    This head is implemented of `DPT <https://arxiv.org/abs/2103.13413>`_.
    Args:
        embed_dims (int): The embed dimension of the ViT backbone.
            Default: 768.
        post_process_channels (List): Out channels of post process conv
            layers. Default: [96, 192, 384, 768].
        readout_type (str): Type of readout operation. Default: 'ignore'.
        patch_size (int): The patch size. Default: 16.
        expand_channels (bool): Whether expand the channels in post process
            block. Default: False.
    """

    def __init__(
        self,
        embed_dims=768,
        channels=48,
        post_process_channels=[48, 96, 192, 384],
        expand_channels=False,
        **kwargs,
    ):
        super(DPT, self).__init__(**kwargs)

        self.channels = channels
        self.expand_channels = expand_channels
        self.reassemble_blocks = ReassembleBlocks(embed_dims, post_process_channels)

        self.post_process_channels = [
            channel * math.pow(2, i) if expand_channels else channel
            for i, channel in enumerate(post_process_channels)
        ]
        self.convs = nn.ModuleList()
        for channel in self.post_process_channels:
            self.convs.append(
                ConvModule(
                    channel,
                    self.channels,
                    kernel_size=3,
                    padding=1,
                    act_layer=None,
                    bias=False,
                )
            )
        self.fusion_blocks = nn.ModuleList()
        for _ in range(len(self.convs)):
            self.fusion_blocks.append(FeatureFusionBlock(self.channels, nn.ReLU, None))
        self.fusion_blocks[0].res_conv_unit1 = None
        self.project = ConvModule(
            self.channels, self.channels, kernel_size=3, padding=1, norm_layer=None
        )
        self.head = nn.Sequential(
            nn.Conv2d(
                self.channels, self.channels // 2, kernel_size=3, stride=1, padding=1
            ),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(
                self.channels // 2,
                self.channels // 4,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(True),
            nn.Conv2d(self.channels // 4, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )
        self.num_fusion_blocks = len(self.fusion_blocks)
        self.num_reassemble_blocks = len(self.reassemble_blocks.resize_layers)
        self.num_post_process_channels = len(self.post_process_channels)
        assert self.num_fusion_blocks == self.num_reassemble_blocks
        assert self.num_reassemble_blocks == self.num_post_process_channels

    def forward(self, inputs):
        """
        Inputs should be a tuple with list of features from the ViT backbone and the skip connection
        """
        inputs, skip = inputs
        assert len(inputs) == self.num_reassemble_blocks
        x = [inp for inp in inputs]
        x = self.reassemble_blocks(x)
        x = [self.convs[i](feature) for i, feature in enumerate(x)]
        out = self.fusion_blocks[0](x[-1])
        for i in range(1, len(self.fusion_blocks)):
            out = self.fusion_blocks[i](out, x[-(i + 1)])
        out = self.project(out)
        out = self.head(out)
        out = out + skip if skip is not None else out  # fuse skip connection
        return out


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    module = DPT(embed_dims=384, post_process_channels=[48, 96, 192, 384]).to(device)
    mock = [
        (torch.rand((4, 384, 32, 32)).to(device), torch.rand(4, 384)) for _ in range(4)
    ]
    # torchinfo.summary(module, mock[0].shape)
    mock = module(mock)
    print(mock.shape)
