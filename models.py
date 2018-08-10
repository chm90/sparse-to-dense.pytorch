import torch
from torch import cat
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import collections
import math
from typing import Tuple, Any
oheight, owidth = 228, 304


class Unpool(nn.Module):
    # Unpool: 2*2 unpooling with zero padding
    def __init__(self, num_channels, stride=2):
        super(Unpool, self).__init__()

        self.num_channels = num_channels
        self.stride = stride

        # create kernel [1, 0; 0, 0]
        self.weights = torch.autograd.Variable(
            torch.zeros(
                num_channels, 1, stride,
                stride).cuda())  # currently not compatible with running on CPU
        self.weights[:, :, 0, 0] = 1

    def forward(self, x):
        return F.conv_transpose2d(
            x, self.weights, stride=self.stride, groups=self.num_channels)


def weights_init(m):
    # Initialize filters with Gaussian random weights
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


SKIP_TYPES = ["none", "cat", "sum", "proj"]


class Decoder(nn.Module):
    # Decoder is the base class for all decoders

    names = ['deconv2', 'deconv3', 'upconv', 'upproj']

    def __init__(self, skip_type, additional_input_ch):
        super(Decoder, self).__init__()
        assert (skip_type in SKIP_TYPES)
        self.skip_type = skip_type
        self.additional_input_ch = additional_input_ch
        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None
        if self.skip_type == "proj":
            self.proj_layers = dict()

    def skip(self, x, x_):
        if self.skip_type == "none" or x_ is None:
            return x
        if self.skip_type == "cat":
            return cat([x, x_], dim=1)
        if self.skip_type == "sum":
            return x + x_[:, :x.shape[1], :, :]
        if self.skip_type == "proj":
            layer_key = (x_.shape[1], x.shape[1])
            if not layer_key in self.proj_layers.keys():
                self.proj_layers[layer_key] = nn.Conv2d(
                    x_.shape[1], x.shape[1], 1, bias=None)
                self.proj_layers[layer_key].cuda()
            return x + self.proj_layers[layer_key](x_)
        raise ValueError(f"Invalid skip_type, {self.skip_type}")

    def forward(self, x, x1=None, x2=None, x3=None, x4=None, x5=None):
        x = self.skip(x, x1)
        x = self.layer1(x)
        #print("decoder layer1 output shape =", x.shape)
        # print("x2.shape =",x2.shape)

        x = self.skip(x, x2)
        x = self.layer2(x)
        #print("decoder layer2 output shape =", x.shape)
        #print("x3.shape =", x3.shape)
        x = self.skip(x, x3)
        x = self.layer3(x)
        #print("decoder layer3 output shape =", x.shape)
        x = self.skip(x, x4)
        x = self.layer4(x)
        #print("decoder layer4 output shape =", x.shape)
        x = self.skip(x, x5)
        return x


class DeConv(Decoder):
    def __init__(self, in_channels, kernel_size, skip_type,
                 additional_input_ch):
        assert kernel_size >= 2, "kernel_size out of range: {}".format(
            kernel_size)
        super(DeConv, self).__init__(skip_type, additional_input_ch)

        def convt(in_channels, i):
            stride = 2
            padding = (kernel_size - 1) // 2
            output_padding = kernel_size % 2
            assert -2 - 2 * padding + kernel_size + output_padding == 0, "deconv parameters incorrect"

            module_name = "deconv{}".format(kernel_size)
            return nn.Sequential(
                collections.OrderedDict([
                    (module_name,
                     nn.ConvTranspose2d(
                         in_channels + self.additional_input_ch[i],
                         in_channels // 2,
                         kernel_size,
                         stride,
                         padding,
                         output_padding,
                         bias=False)),
                    ('batchnorm', nn.BatchNorm2d(in_channels // 2)),
                    ('relu', nn.ReLU(inplace=True)),
                ]))

        self.layer1 = convt(in_channels, 0)
        self.layer2 = convt(in_channels // 2, 1)
        self.layer3 = convt(in_channels // (2**2), 2)
        self.layer4 = convt(in_channels // (2**3), 3)


class UpConv(Decoder):
    # UpConv decoder consists of 4 upconv modules with decreasing number of channels and increasing feature map size
    def upconv_module(self, in_channels):
        # UpConv module: unpool -> 5*5 conv -> batchnorm -> ReLU
        upconv = nn.Sequential(
            collections.OrderedDict([
                ('unpool', Unpool(in_channels)),
                ('conv',
                 nn.Conv2d(
                     in_channels,
                     in_channels // 2,
                     kernel_size=5,
                     stride=1,
                     padding=2,
                     bias=False)),
                ('batchnorm', nn.BatchNorm2d(in_channels // 2)),
                ('relu', nn.ReLU()),
            ]))
        return upconv

    def __init__(self, in_channels, skip_type, additional_input_ch):
        super(UpConv, self).__init__(skip_type, additional_input_ch)
        self.layer1 = self.upconv_module(in_channels + additional_input_ch[0])
        self.layer2 = self.upconv_module(
            in_channels // 2 + additional_input_ch[1])
        self.layer3 = self.upconv_module(
            in_channels // 4 + additional_input_ch[2])
        self.layer4 = self.upconv_module(
            in_channels // 8 + additional_input_ch[3])


class UpProj(Decoder):
    # UpProj decoder consists of 4 upproj modules with decreasing number of channels and increasing feature map sizes

    class UpProjModule(nn.Module):
        # UpProj module has two branches, with a Unpool at the start and a ReLu at the end
        #   upper branch: 5*5 conv -> batchnorm -> ReLU -> 3*3 conv -> batchnorm
        #   bottom branch: 5*5 conv -> batchnorm

        def __init__(self, in_channels, out_channels):
            super(UpProj.UpProjModule, self).__init__()
            self.unpool = Unpool(in_channels)
            self.upper_branch = nn.Sequential(
                collections.OrderedDict([
                    ('conv1',
                     nn.Conv2d(
                         in_channels,
                         out_channels,
                         kernel_size=5,
                         stride=1,
                         padding=2,
                         bias=False)),
                    ('batchnorm1', nn.BatchNorm2d(out_channels)),
                    ('relu', nn.ReLU()),
                    ('conv2',
                     nn.Conv2d(
                         out_channels,
                         out_channels,
                         kernel_size=3,
                         stride=1,
                         padding=1,
                         bias=False)),
                    ('batchnorm2', nn.BatchNorm2d(out_channels)),
                ]))
            self.bottom_branch = nn.Sequential(
                collections.OrderedDict([
                    ('conv',
                     nn.Conv2d(
                         in_channels,
                         out_channels,
                         kernel_size=5,
                         stride=1,
                         padding=2,
                         bias=False)),
                    ('batchnorm', nn.BatchNorm2d(out_channels)),
                ]))
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.unpool(x)
            x1 = self.upper_branch(x)
            x2 = self.bottom_branch(x)
            x = x1 + x2
            x = self.relu(x)
            return x

    def __init__(self, in_channels, skip_type, additional_input_ch):
        super(UpProj, self).__init__(skip_type, additional_input_ch)
        self.layer1 = self.UpProjModule(in_channels + additional_input_ch[0],
                                        in_channels // 2)
        self.layer2 = self.UpProjModule(
            in_channels // 2 + additional_input_ch[1], in_channels // 4)
        self.layer3 = self.UpProjModule(
            in_channels // 4 + additional_input_ch[2], in_channels // 8)
        self.layer4 = self.UpProjModule(
            in_channels // 8 + additional_input_ch[3], in_channels // 16)


def choose_decoder(decoder, in_channels, additional_input_ch, skip_type):
    # iheight, iwidth = 10, 8
    if decoder[:6] == 'deconv':
        assert len(decoder) == 7
        kernel_size = int(decoder[6])
        return DeConv(in_channels, kernel_size, skip_type, additional_input_ch)
    elif decoder == "upproj":
        return UpProj(in_channels, skip_type, additional_input_ch)
    elif decoder == "upconv":
        return UpConv(in_channels, skip_type, additional_input_ch)
    else:
        assert False, "invalid option for decoder: {}".format(decoder)


class ResNet(nn.Module):
    def __init__(self,
                 layers: int,
                 decoder: str,
                 in_channels: int = 3,
                 out_channels: int = 1,
                 pretrained: bool = True,
                 image_shape: Tuple[int, int] = (oheight, owidth),
                 skip_type: str = "sum",
                 square_width=50) -> None:
        self.image_shape = image_shape
        self.square_width = square_width
        if layers not in [18, 34, 50, 101, 152]:
            raise RuntimeError(
                'Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.
                format(layers))

        super(ResNet, self).__init__()
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(
            layers)](pretrained=pretrained)

        if in_channels == 3:
            self.conv1 = pretrained_model._modules['conv1']
            self.bn1 = pretrained_model._modules['bn1']
        else:
            self.conv1 = nn.Conv2d(
                in_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            weights_init(self.conv1)
            weights_init(self.bn1)

        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        self.conv2 = nn.Conv2d(
            num_channels, num_channels // 2, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels // 2)
        extra_channels = (0, 256, 128, 64) if skip_type == "cat" else (0, 0, 0,
                                                                       0)
        self.decoder = choose_decoder(decoder, num_channels // 2,
                                      extra_channels, skip_type)
        # setting bias=true doesn't improve accuracy
        self.conv3 = nn.Conv2d(
            num_channels // 32,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.bilinear = nn.Upsample(
            size=image_shape, mode='bilinear', align_corners=True)
        # weight init
        self.conv2.apply(weights_init)
        self.bn2.apply(weights_init)
        self.decoder.apply(weights_init)
        self.conv3.apply(weights_init)

    @property
    def in_channels(self):
        return self.conv1.weight.shape[1]

    @property
    def out_channels(self):
        return self.conv3.weight.shape[1]

    @property
    def input_shape(self):
        return (self.in_channels, *self.image_shape)

    def forward(self, x: torch.cuda.FloatTensor):
        # resnet
        assert x.shape[1:] == self.input_shape
        #print("input shape =", x.shape)
        x0 = self.conv1(x)
        self.conv1.output_shape = tuple(x0.shape[1:])
        #print("conv1 shape =", x0.shape)
        x = self.bn1(x0)
        x = self.relu(x)
        x = self.maxpool(x)
        #print("maxpool output shape =", x.shape)
        self.maxpool.output_shape = tuple(x.shape[1:])
        x1 = self.layer1(x)
        self.layer1.output_shape = tuple(x1.shape[1:])
        #print("layer1 output shape =", x1.shape)

        x2 = self.layer2(x1)
        self.layer2.output_shape = tuple(x2.shape[1:])
        #print("layer2 output shape =", x2.shape)

        x3 = self.layer3(x2)
        self.layer3.output_shape = tuple(x3.shape[1:])
        #print("layer3 output shape =", x3.shape)

        x4 = self.layer4(x3)
        self.layer4.output_shape = tuple(x4.shape[1:])
        #print("layer4 output shape =", x4.shape)
        x = self.conv2(x4)
        self.conv2.output_shape = tuple(x.shape[1:])
        #print("conv2 output shape =", x.shape)

        x = self.bn2(x)
        # decoder
        x = self.decoder(x, x2=x3, x3=x2, x4=x1)
        self.decoder.output_shape = tuple(x.shape[1:])
        #print("decoder output shape =", x.shape)
        x = self.conv3(x)
        self.conv3.output_shape = tuple(x.shape[1:])
        x = self.bilinear(x)
        self.bilinear.output_shape = tuple(x.shape[1:])
        #print("output shape =", x.shape)
        self.output_shape = self.bilinear.output_shape
        return x
