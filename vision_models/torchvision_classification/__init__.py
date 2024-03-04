#!/usr/bin/env python3
import math
import torch
import json
from torchvision.transforms._presets import InterpolationMode
from pdb import set_trace
from copy import deepcopy

from ...common import _IMAGENET_MEAN, _IMAGENET_STD
from ...registry import register_model

# pinning an earlier release for backwards compatiblity
# _HUB_URL = "pytorch/vision:v0.8.2"

# or update to the latest pytorch
_HUB_URL = "pytorch/vision"

default_meta = dict(    
    num_classes = 1000,
    input_size=(3,224,224),   
    crop_pct=0.875,
    resize=(256,256),
    input_range=[0, 1],
    mean=_IMAGENET_MEAN,
    std=_IMAGENET_STD,
    repo="https://github.com/pytorch/vision",
    task="supervised1k",
    dataset='imagenet1k',
    datasize="1.3M",
    bib=json.dumps('''""''')
)

weights = {}

__all__ = [
    'alexnet',
    'vgg11_bn',
    'vgg13_bn',
    'vgg16_bn',
    'vgg19_bn',
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152',
    'resnet50_gen2',
    'resnet101_gen2',
    'resnet152_gen2',
    'squeezenet1_0',
    'squeezenet1_1',
    'densenet121',
    'densenet169',
    'densenet201',
    'inception_v3',
    'shufflenet_v2_x0_5',
    'shufflenet_v2_x1_0',
    'shufflenet_v2_x1_5',
    'shufflenet_v2_x2_0',
    'mobilenet_v2',
    'mobilenet_v3_small',
    'mobilenet_v3_large',
    'resnext50_32x4d',
    'resnext101_32x8d',
    #'resnext101_64x4d',
    'resnext50_32x4d_gen2',
    'resnext101_32x8d_gen2',
    'wide_resnet50_2',
    'wide_resnet101_2',
    'wide_resnet50_2_gen2',
    'wide_resnet101_2_gen2',
    'mnasnet0_5',
    'mnasnet0_75',
    'mnasnet1_0',
    'mnasnet1_3',
    'efficientnet_b0',
    'efficientnet_b1',
    'efficientnet_b2',
    'efficientnet_b3',
    'efficientnet_b4',
    'efficientnet_b5',
    'efficientnet_b6',
    'efficientnet_b7',
    'efficientnet_v2_s',
    'efficientnet_v2_m',
    'efficientnet_v2_l',
    'efficientnet_b1_gen2',
    'regnet_y_400mf',
    'regnet_y_800mf',
    'regnet_y_1_6gf',
    'regnet_y_3_2gf',
    'regnet_y_8gf',
    'regnet_y_16gf',
    'regnet_y_32gf',
    'regnet_x_400mf',
    'regnet_x_800mf',
    'regnet_x_1_6gf',
    'regnet_x_3_2gf',
    'regnet_x_8gf',
    'regnet_x_16gf',
    'regnet_x_32gf',
    'regnet_y_400mf_gen2',
    'regnet_y_800mf_gen2',
    'regnet_y_1_6gf_gen2',
    'regnet_y_3_2gf_gen2',
    'regnet_y_8gf_gen2',
    'regnet_y_16gf_gen2',
    'regnet_y_32gf_gen2',
    'regnet_x_400mf_gen2',
    'regnet_x_800mf_gen2',
    'regnet_x_1_6gf_gen2',
    'regnet_x_3_2gf_gen2',
    'regnet_x_8gf_gen2',
    'regnet_x_16gf_gen2',
    'regnet_x_32gf_gen2',
    'regnet_y_16gf_swag_e2e_ft',
    'regnet_y_32gf_swag_e2e_ft',
    'regnet_y_128gf_swag_e2e_ft',
    'regnet_y_16gf_swag_linear',
    'regnet_y_32gf_swag_linear',
    'regnet_y_128gf_swag_linear',
    'vit_b_16',
    'vit_b_32',
    'vit_l_16',
    'vit_l_32',
    'vit_b_16_swag_e2e_ft',
    'vit_l_16_swag_e2e_ft',
    'vit_h_14_swag_e2e_ft',
    'vit_b_16_swag_linear',
    'vit_l_16_swag_linear',
    'vit_h_14_swag_linear',
    'convnext_tiny',
    'convnext_small',
    'convnext_base',
    'convnext_large',
    'weights'
]

# ===================================================
#  AlexNet
# ===================================================

from torchvision.models.alexnet import AlexNet_Weights

weights['alexnet'] = AlexNet_Weights.IMAGENET1K_V1

@register_model("torchvision", arch="alexnet", **default_meta, arxiv="https://arxiv.org/abs/1404.5997",
description="AlexNet model architecture from `One weird trick for parallelizing convolutional neural networks <https://arxiv.org/abs/1404.5997>`", weights_url=weights['alexnet'].url)
def alexnet(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "alexnet", verbose=True, 
                           weights=weights['alexnet'] if pretrained else None, **kwargs)

    return model

# ===================================================
#  VGG
# ===================================================

from torchvision.models.vgg import VGG11_BN_Weights, VGG13_BN_Weights, VGG16_BN_Weights, VGG19_BN_Weights

weights['vgg11_bn'] = VGG11_BN_Weights.IMAGENET1K_V1
weights['vgg13_bn'] = VGG13_BN_Weights.IMAGENET1K_V1
weights['vgg16_bn'] = VGG16_BN_Weights.IMAGENET1K_V1
weights['vgg19_bn'] = VGG19_BN_Weights.IMAGENET1K_V1

@register_model("torchvision", arch="vgg11_bn", **default_meta, arxiv="https://arxiv.org/abs/1409.1556",
description="VGG 11-layer model (configuration 'A') with batch normalization `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`", weights_url=weights['vgg11_bn'].url)
def vgg11_bn(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "vgg11_bn", verbose=True, 
                           weights=weights['vgg11_bn'] if pretrained else None, **kwargs)

    return model

@register_model("torchvision", arch="vgg13_bn", **default_meta, arxiv="https://arxiv.org/abs/1409.1556",
description="VGG 13-layer model (configuration 'B') with batch normalization `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`", weights_url=weights['vgg13_bn'].url)
def vgg13_bn(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "vgg13_bn", verbose=True, 
                           weights=weights['vgg13_bn'] if pretrained else None, **kwargs)

    return model

@register_model("torchvision", arch="vgg16_bn", **default_meta, arxiv="https://arxiv.org/abs/1409.1556",
description="VGG 16-layer model (configuration 'D') with batch normalization `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`", weights_url=weights['vgg16_bn'].url)
def vgg16_bn(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "vgg16_bn", verbose=True, 
                           weights=weights['vgg16_bn'] if pretrained else None, **kwargs)

    return model

@register_model("torchvision", arch="vgg19_bn", **default_meta, arxiv="https://arxiv.org/abs/1409.1556",
description="VGG 19-layer model (configuration 'E') with batch normalization `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`", weights_url=weights['vgg19_bn'].url)
def vgg19_bn(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "vgg19_bn", verbose=True, 
                           weights=weights['vgg19_bn'] if pretrained else None, **kwargs)

    return model

# ===================================================
#  ResNet
# ===================================================

from torchvision.models.resnet import (
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
)

weights['resnet18'] = ResNet18_Weights.IMAGENET1K_V1
weights['resnet34'] = ResNet34_Weights.IMAGENET1K_V1
weights['resnet50'] = ResNet50_Weights.IMAGENET1K_V1
weights['resnet101'] = ResNet101_Weights.IMAGENET1K_V1
weights['resnet152'] = ResNet152_Weights.IMAGENET1K_V1

@register_model("torchvision", arch="resnet18", **default_meta, arxiv="https://arxiv.org/pdf/1512.03385.pdf",
description="ResNet-18 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`", weights_url=weights['resnet18'].url)
def resnet18(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "resnet18", verbose=True, 
                           weights=weights['resnet18'] if pretrained else None, **kwargs)

    return model

@register_model("torchvision", arch="resnet34", **default_meta, arxiv="https://arxiv.org/pdf/1512.03385.pdf",
description="ResNet-34 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`", weights_url=weights['resnet34'].url)
def resnet34(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "resnet34", verbose=True, 
                           weights=weights['resnet34'] if pretrained else None, **kwargs)

    return model

@register_model("torchvision", arch="resnet50", **default_meta, arxiv="https://arxiv.org/pdf/1512.03385.pdf",
description="ResNet-50 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`", weights_url=weights['resnet50'].url)
def resnet50(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "resnet50", verbose=True, 
                           weights=weights['resnet50'] if pretrained else None, **kwargs)

    return model

@register_model("torchvision", arch="resnet101", **default_meta, arxiv="https://arxiv.org/pdf/1512.03385.pdf",
description="ResNet-101 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`", weights_url=weights['resnet101'].url)
def resnet101(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "resnet101", verbose=True, 
                           weights=weights['resnet101'] if pretrained else None, **kwargs)

    return model

@register_model("torchvision", arch="resnet152", **default_meta, arxiv="https://arxiv.org/pdf/1512.03385.pdf",
description="ResNet-152 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`", weights_url=weights['resnet152'].url)
def resnet152(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "resnet152", verbose=True, 
                           weights=weights['resnet152'] if pretrained else None, **kwargs)

    return model

# ===================================================
#  ResNet (pytorch second gen)
# ===================================================

from torchvision.models.resnet import (
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
)

# weights['resnet18_gen2'] = ResNet18_Weights.IMAGENET1K_V2
# weights['resnet34_gen2'] = ResNet34_Weights.IMAGENET1K_V2
weights['resnet50_gen2'] = ResNet50_Weights.IMAGENET1K_V2
weights['resnet101_gen2'] = ResNet101_Weights.IMAGENET1K_V2
weights['resnet152_gen2'] = ResNet152_Weights.IMAGENET1K_V2

# @register_model("torchvision", arch="resnet18", **default_meta, arxiv="https://arxiv.org/pdf/1512.03385.pdf",
# description="ResNet-18 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`. These weights improve upon the results of the original paper by using TorchVision's `new training recipe <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.", weights_url=weights['resnet18_gen2'].url)
# def resnet18_gen2(pretrained=True, **kwargs):
#     model = torch.hub.load(_HUB_URL, "resnet18", verbose=True, 
#                            weights=weights['resnet18_gen2'] if pretrained else None, **kwargs)

#     return model

# @register_model("torchvision", arch="resnet34", **default_meta, arxiv="https://arxiv.org/pdf/1512.03385.pdf",
# description="ResNet-18 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`. These weights improve upon the results of the original paper by using TorchVision's `new training recipe <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.", weights_url=weights['resnet34_gen2'].url)
# def resnet34_gen2(pretrained=True, **kwargs):
#     model = torch.hub.load(_HUB_URL, "resnet34", verbose=True, 
#                            weights=weights['resnet34_gen2'] if pretrained else None, **kwargs)

#     return model

@register_model("torchvision", arch="resnet50", **default_meta, arxiv="https://arxiv.org/pdf/1512.03385.pdf",
description="ResNet-18 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`. These weights improve upon the results of the original paper by using TorchVision's `new training recipe <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.", weights_url=weights['resnet50_gen2'].url)
def resnet50_gen2(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "resnet50", verbose=True, 
                           weights=weights['resnet50_gen2'] if pretrained else None, **kwargs)

    return model

@register_model("torchvision", arch="resnet101", **default_meta, arxiv="https://arxiv.org/pdf/1512.03385.pdf",
description="ResNet-18 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`. These weights improve upon the results of the original paper by using TorchVision's `new training recipe <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.", weights_url=weights['resnet101_gen2'].url)
def resnet101_gen2(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "resnet101", verbose=True, 
                           weights=weights['resnet101_gen2'] if pretrained else None, **kwargs)

    return model

@register_model("torchvision", arch="resnet152", **default_meta, arxiv="https://arxiv.org/pdf/1512.03385.pdf",
description="ResNet-18 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`. These weights improve upon the results of the original paper by using TorchVision's `new training recipe <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.", weights_url=weights['resnet152_gen2'].url)
def resnet152_gen2(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "resnet152", verbose=True, 
                           weights=weights['resnet152_gen2'] if pretrained else None, **kwargs)

    return model

# ===================================================
#  SqueezeNet
# ===================================================

from torchvision.models.squeezenet import SqueezeNet1_0_Weights, SqueezeNet1_1_Weights

weights['squeezenet1_0'] = SqueezeNet1_0_Weights.IMAGENET1K_V1
weights['squeezenet1_1'] = SqueezeNet1_1_Weights.IMAGENET1K_V1

@register_model("torchvision", arch="squeezenet1_0", **default_meta, arxiv="https://arxiv.org/abs/1602.07360",
description="SqueezeNet model architecture from the `SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size <https://arxiv.org/abs/1602.07360>`", weights_url=weights['squeezenet1_0'].url)
def squeezenet1_0(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "squeezenet1_0", verbose=True, 
                           weights=weights['squeezenet1_0'] if pretrained else None, **kwargs)

    return model

@register_model("torchvision", arch="squeezenet1_1", **default_meta, arxiv="https://arxiv.org/abs/1602.07360",
description="SqueezeNet 1.1 model from the `official SqueezeNet repo <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`. SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters than SqueezeNet 1.0, without sacrificing accuracy.", weights_url=weights['squeezenet1_1'].url)
def squeezenet1_1(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "squeezenet1_1", verbose=True, 
                           weights=weights['squeezenet1_1'] if pretrained else None, **kwargs)

    return model

# ===================================================
#  DenseNet
# ===================================================

from torchvision.models.densenet import DenseNet121_Weights, DenseNet161_Weights, DenseNet169_Weights, DenseNet201_Weights

weights['densenet121'] = DenseNet121_Weights.IMAGENET1K_V1
weights['densenet161'] = DenseNet161_Weights.IMAGENET1K_V1
weights['densenet169'] = DenseNet169_Weights.IMAGENET1K_V1
weights['densenet201'] = DenseNet201_Weights.IMAGENET1K_V1

@register_model("torchvision", arch="densenet121", **default_meta, arxiv="https://arxiv.org/abs/1608.06993",
description="Densenet-121 model from `Densely Connected Convolutional Networks <https://arxiv.org/abs/1608.06993>`", weights_url=weights['densenet121'].url)
def densenet121(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "densenet121", verbose=True, 
                           weights=weights['densenet121'] if pretrained else None, **kwargs)

    return model

@register_model("torchvision", arch="densenet169", **default_meta, arxiv="https://arxiv.org/abs/1608.06993",
description="Densenet-169 model from `Densely Connected Convolutional Networks <https://arxiv.org/abs/1608.06993>`", weights_url=weights['densenet169'].url)
def densenet169(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "densenet169", verbose=True, 
                           weights=weights['densenet169'] if pretrained else None, **kwargs)

    return model

@register_model("torchvision", arch="densenet201", **default_meta, arxiv="https://arxiv.org/abs/1608.06993",
description="Densenet-201 model from `Densely Connected Convolutional Networks <https://arxiv.org/abs/1608.06993>`", weights_url=weights['densenet201'].url)
def densenet201(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "densenet201", verbose=True, 
                           weights=weights['densenet201'] if pretrained else None, **kwargs)

    return model

# ===================================================
#  InceptionV3
# ===================================================

from torchvision.models.inception import Inception_V3_Weights

weights['inception_v3'] = Inception_V3_Weights.IMAGENET1K_V1

_meta = dict(input_size=[3,299,299], resize=[3,342,342], crop_pct=299/342)
@register_model("torchvision", arch="inception_v3", **{**default_meta, **_meta},
arxiv="http://arxiv.org/abs/1512.00567",
description="Inception v3 model architecture from `Rethinking the Inception Architecture for Computer Vision <http://arxiv.org/abs/1512.00567>`", weights_url=weights['inception_v3'].url)
def inception_v3(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "inception_v3", verbose=True, 
                           weights=weights['inception_v3'] if pretrained else None, **kwargs)

    return model

# ===================================================
#  ShuffleNet
# ===================================================

from torchvision.models.shufflenetv2 import (
    ShuffleNet_V2_X0_5_Weights, ShuffleNet_V2_X1_0_Weights, ShuffleNet_V2_X1_5_Weights, ShuffleNet_V2_X2_0_Weights
)

weights['shufflenet_v2_x0_5'] = ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1
weights['shufflenet_v2_x1_0'] = ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1
weights['shufflenet_v2_x1_5'] = ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V1
weights['shufflenet_v2_x2_0'] = ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1

@register_model("torchvision", arch="shufflenet_v2_x0_5", **default_meta, arxiv="https://arxiv.org/abs/1807.11164",
description="ShuffleNetV2 architecture with 0.5x output channels, as described in `ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design <https://arxiv.org/abs/1807.11164>`", weights_url=weights['shufflenet_v2_x0_5'].url)
def shufflenet_v2_x0_5(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "shufflenet_v2_x0_5", verbose=True, 
                           weights=weights['shufflenet_v2_x0_5'] if pretrained else None, **kwargs)

    return model

@register_model("torchvision", arch="shufflenet_v2_x1_0", **default_meta, arxiv="https://arxiv.org/abs/1807.11164",
description="ShuffleNetV2 architecture with 1.0x output channels, as described in `ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design <https://arxiv.org/abs/1807.11164>`", weights_url=weights['shufflenet_v2_x1_0'].url)
def shufflenet_v2_x1_0(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "shufflenet_v2_x1_0", verbose=True, 
                           weights=weights['shufflenet_v2_x1_0'] if pretrained else None, **kwargs)

    return model

_meta = dict(input_size=[3,224,224], resize=[232,232], crop_pct=224/232)
@register_model("torchvision", arch="shufflenet_v2_x1_5", **{**default_meta, **_meta}, arxiv="https://arxiv.org/abs/1807.11164",
description="ShuffleNetV2 architecture with 1.5x output channels, as described in `ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design <https://arxiv.org/abs/1807.11164>`", weights_url=weights['shufflenet_v2_x1_5'].url)
def shufflenet_v2_x1_5(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "shufflenet_v2_x1_5", verbose=True, 
                           weights=weights['shufflenet_v2_x1_5'] if pretrained else None, **kwargs)

    return model

_meta = dict(input_size=[3,224,224], resize=[232,232], crop_pct=224/232)
@register_model("torchvision", arch="shufflenet_v2_x2_0", **{**default_meta, **_meta}, arxiv="https://arxiv.org/abs/1807.11164",
description="ShuffleNetV2 architecture with 2.0x output channels, as described in `ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design <https://arxiv.org/abs/1807.11164>`", weights_url=weights['shufflenet_v2_x2_0'].url)
def shufflenet_v2_x2_0(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "shufflenet_v2_x2_0", verbose=True, 
                           weights=weights['shufflenet_v2_x2_0'] if pretrained else None, **kwargs)

    return model

# ===================================================
#  MobileNet
# ===================================================

from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights, MobileNet_V3_Large_Weights

weights['mobilenet_v2'] = MobileNet_V2_Weights.IMAGENET1K_V1
weights['mobilenet_v3_small'] = MobileNet_V3_Small_Weights.IMAGENET1K_V1
weights['mobilenet_v3_large'] = MobileNet_V3_Large_Weights.IMAGENET1K_V1

@register_model("torchvision", arch="mobilenet_v2", **default_meta, arxiv="https://arxiv.org/abs/1801.04381",
description="MobileNetV2 architecture from the `MobileNetV2: Inverted Residuals and Linear Bottlenecks <https://arxiv.org/abs/1801.04381>`_ paper.",
weights_url=weights['mobilenet_v2'].url)
def mobilenet_v2(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "mobilenet_v2", verbose=True, 
                           weights=weights['mobilenet_v2'] if pretrained else None, **kwargs)

    return model

@register_model("torchvision", arch="mobilenet_v3_small", **default_meta, arxiv="https://arxiv.org/abs/1905.02244",
description="Constructs a small MobileNetV3 architecture from `Searching for MobileNetV3 <https://arxiv.org/abs/1905.02244>`.",
weights_url=weights['mobilenet_v3_small'].url)
def mobilenet_v3_small(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "mobilenet_v3_small", verbose=True, 
                           weights=weights['mobilenet_v3_small'] if pretrained else None, **kwargs)

    return model

@register_model("torchvision", arch="mobilenet_v3_large", **default_meta, arxiv="https://arxiv.org/abs/1905.02244",
description="Constructs a small MobileNetV3 architecture from `Searching for MobileNetV3 <https://arxiv.org/abs/1905.02244>`.",
weights_url=weights['mobilenet_v3_large'].url)
def mobilenet_v3_large(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "mobilenet_v3_large", verbose=True, 
                           weights=weights['mobilenet_v3_large'] if pretrained else None, **kwargs)

    return model

# ===================================================
#  ResNext
# ===================================================

from torchvision.models.resnet import ResNeXt50_32X4D_Weights, ResNeXt101_32X8D_Weights, ResNeXt101_64X4D_Weights

weights['resnext50_32x4d'] = ResNeXt50_32X4D_Weights.IMAGENET1K_V1
weights['resnext101_32x8d'] = ResNeXt101_32X8D_Weights.IMAGENET1K_V1
# weights['resnext101_64x4d'] = ResNeXt101_64X4D_Weights.IMAGENET1K_V1

@register_model("torchvision", arch="resnext50_32x4d", **default_meta, arxiv="https://arxiv.org/abs/1611.05431",
description="ResNeXt-50 32x4d model from `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_.",
weights_url=weights['resnext50_32x4d'].url)
def resnext50_32x4d(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "resnext50_32x4d", verbose=True, 
                           weights=weights['resnext50_32x4d'] if pretrained else None, **kwargs)

    return model

@register_model("torchvision", arch="resnext101_32x8d", **default_meta, arxiv="https://arxiv.org/abs/1611.05431",
description="ResNeXt-101 32x8d model from `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_.",
weights_url=weights['resnext101_32x8d'].url)
def resnext101_32x8d(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "resnext101_32x8d", verbose=True, 
                           weights=weights['resnext101_32x8d'] if pretrained else None, **kwargs)

    return model

# @register_model("torchvision", arch="resnext101_64x4d", **default_meta, arxiv="https://arxiv.org/abs/1611.05431",
# description="ResNeXt-101 64x4d model from `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_.",
# weights_url=weights['resnext101_64x4d'].url)
# def resnext101_64x4d(pretrained=True, **kwargs):
#     model = torch.hub.load(_HUB_URL, "resnext101_64x4d", verbose=True, 
#                            weights=weights['resnext101_64x4d'] if pretrained else None, **kwargs)

#     return model

# ===================================================
#  ResNext_gen2 (pytorch new training)
# ===================================================

from torchvision.models.resnet import ResNeXt50_32X4D_Weights, ResNeXt101_32X8D_Weights, ResNeXt101_64X4D_Weights

weights['resnext50_32x4d_gen2'] = ResNeXt50_32X4D_Weights.IMAGENET1K_V2
weights['resnext101_32x8d_gen2'] = ResNeXt101_32X8D_Weights.IMAGENET1K_V2
# weights['resnext101_64x4d_gen2'] = ResNeXt101_64X4D_Weights.IMAGENET1K_V2

_meta = dict(input_size=[3,224,224], resize=[232,232], crop_pct=224/232)
@register_model("torchvision", arch="resnext50_32x4d", **{**default_meta, **_meta}, arxiv="https://arxiv.org/abs/1611.05431",
description="ResNeXt-50 32x4d model from `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_. These weights improve upon the results of the original paper by using TorchVision's `new training recipe <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.",
weights_url=weights['resnext50_32x4d_gen2'].url)
def resnext50_32x4d_gen2(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "resnext50_32x4d", verbose=True, 
                           weights=weights['resnext50_32x4d_gen2'] if pretrained else None, **kwargs)

    return model

_meta = dict(input_size=[3,224,224], resize=[232,232], crop_pct=224/232)
@register_model("torchvision", arch="resnext101_32x8d", **{**default_meta, **_meta}, arxiv="https://arxiv.org/abs/1611.05431",
description="ResNeXt-101 32x8d model from `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_. These weights improve upon the results of the original paper by using TorchVision's `new training recipe <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.",
weights_url=weights['resnext101_32x8d_gen2'].url)
def resnext101_32x8d_gen2(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "resnext101_32x8d", verbose=True, 
                           weights=weights['resnext101_32x8d_gen2'] if pretrained else None, **kwargs)

    return model

# _meta = dict(input_size=[3,224,224], resize=[232,232], crop_pct=224/232)
# @register_model("torchvision", arch="resnext101_64x4d_gen2", **{**default_meta, **_meta}, arxiv="https://arxiv.org/abs/1611.05431",
# description="ResNeXt-101 64x4d model from `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_. These weights improve upon the results of the original paper by using TorchVision's `new training recipe <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.",
# weights_url=weights['resnext101_64x4d_gen2'].url)
# def resnext101_64x4d_gen2(pretrained=True, **kwargs):
#     model = torch.hub.load(_HUB_URL, "resnext101_64x4d", verbose=True, 
#                            weights=weights['resnext101_64x4d_gen2'] if pretrained else None, **kwargs)

#     return model

# ===================================================
#  WideResnet
# ===================================================

from torchvision.models.resnet import Wide_ResNet50_2_Weights, Wide_ResNet101_2_Weights

weights['wide_resnet50_2'] = Wide_ResNet50_2_Weights.IMAGENET1K_V1
weights['wide_resnet101_2'] = Wide_ResNet101_2_Weights.IMAGENET1K_V1
_meta = dict(input_size=[3,224,224], resize=[232,232], crop_pct=224/232)

@register_model("torchvision", arch="wide_resnet50_2", **default_meta, arxiv="https://arxiv.org/abs/1605.07146",
description="Wide ResNet-50-2 model from `Wide Residual Networks <https://arxiv.org/abs/1605.07146>`_. The model is the same as ResNet except for the bottleneck number of channels which is twice larger in every block. The number of channels in outer 1x1 convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048 channels, and in Wide ResNet-50-2 has 2048-1024-2048.",
weights_url=weights['wide_resnet50_2'].url)
def wide_resnet50_2(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "wide_resnet50_2", verbose=True, 
                           weights=weights['wide_resnet50_2'] if pretrained else None, **kwargs)

    return model

@register_model("torchvision", arch="wide_resnet101_2", **default_meta, arxiv="https://arxiv.org/abs/1605.07146",
description="Wide ResNet-101-2 model from `Wide Residual Networks <https://arxiv.org/abs/1605.07146>`_. The model is the same as ResNet except for the bottleneck number of channels which is twice larger in every block. The number of channels in outer 1x1 convolutions is the same, e.g. last block in ResNet-101 has 2048-512-2048 channels, and in Wide ResNet-101-2 has 2048-1024-2048.",
weights_url=weights['wide_resnet101_2'].url)
def wide_resnet101_2(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "wide_resnet101_2", verbose=True, 
                           weights=weights['wide_resnet101_2'] if pretrained else None, **kwargs)

    return model

# ===================================================
#  WideResnet (gen2)
# ===================================================

weights['wide_resnet50_2_gen2'] = Wide_ResNet50_2_Weights.IMAGENET1K_V2
weights['wide_resnet101_2_gen2'] = Wide_ResNet101_2_Weights.IMAGENET1K_V2
_meta = dict(input_size=[3,224,224], resize=[232,232], crop_pct=224/232)

@register_model("torchvision", arch="wide_resnet50_2", **{**default_meta, **_meta}, arxiv="https://arxiv.org/abs/1605.07146",
description="Wide ResNet-50-2 model from `Wide Residual Networks <https://arxiv.org/abs/1605.07146>`_. The model is the same as ResNet except for the bottleneck number of channels which is twice larger in every block. The number of channels in outer 1x1 convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048 channels, and in Wide ResNet-50-2 has 2048-1024-2048. These weights improve upon the results of the original paper by using TorchVision's `new training recipe <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.",
weights_url=weights['wide_resnet50_2_gen2'].url)
def wide_resnet50_2_gen2(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "wide_resnet50_2", verbose=True, 
                           weights=weights['wide_resnet50_2_gen2'] if pretrained else None, **kwargs)

    return model

@register_model("torchvision", arch="wide_resnet101_2", **{**default_meta, **_meta}, arxiv="https://arxiv.org/abs/1605.07146",
description="Wide ResNet-101-2 model from `Wide Residual Networks <https://arxiv.org/abs/1605.07146>`_. The model is the same as ResNet except for the bottleneck number of channels which is twice larger in every block. The number of channels in outer 1x1 convolutions is the same, e.g. last block in ResNet-101 has 2048-512-2048 channels, and in Wide ResNet-101-2 has 2048-1024-2048. These weights improve upon the results of the original paper by using TorchVision's `new training recipe <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.",
weights_url=weights['wide_resnet101_2'].url)
def wide_resnet101_2_gen2(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "wide_resnet101_2", verbose=True, 
                           weights=weights['wide_resnet101_2_gen2'] if pretrained else None, **kwargs)

    return model

# ===================================================
#  MNASNet
# ===================================================

from torchvision.models.mnasnet import MNASNet0_5_Weights, MNASNet0_75_Weights, MNASNet1_0_Weights, MNASNet1_3_Weights

weights['mnasnet0_5'] = MNASNet0_5_Weights.IMAGENET1K_V1
weights['mnasnet0_75'] = MNASNet0_75_Weights.IMAGENET1K_V1
weights['mnasnet1_0'] = MNASNet1_0_Weights.IMAGENET1K_V1
weights['mnasnet1_3'] = MNASNet1_3_Weights.IMAGENET1K_V1

@register_model("torchvision", arch="mnasnet0_5", **default_meta, arxiv="https://arxiv.org/pdf/1807.11626.pdf",
description="MNASNet with depth multiplier of 0.5 from `MnasNet: Platform-Aware Neural Architecture Search for Mobile <https://arxiv.org/pdf/1807.11626.pdf>`_ paper.", 
weights_url=weights['mnasnet0_5'].url)
def mnasnet0_5(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "mnasnet0_5", verbose=True, 
                           weights=weights['mnasnet0_5'] if pretrained else None, **kwargs)

    return model

_meta = dict(input_size=[3,224,224], resize=[232,232], crop_pct=224/232)
@register_model("torchvision", arch="mnasnet0_75", **{**default_meta, **_meta}, arxiv="https://arxiv.org/pdf/1807.11626.pdf",
description="MNASNet with depth multiplier of 0.75 from `MnasNet: Platform-Aware Neural Architecture Search for Mobile <https://arxiv.org/pdf/1807.11626.pdf>`_ paper.", 
weights_url=weights['mnasnet0_75'].url)
def mnasnet0_75(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "mnasnet0_75", verbose=True, 
                           weights=weights['mnasnet0_75'] if pretrained else None, **kwargs)

    return model

@register_model("torchvision", arch="mnasnet1_0", **default_meta, arxiv="https://arxiv.org/pdf/1807.11626.pdf",
description="MNASNet with depth multiplier of 0.75 from `MnasNet: Platform-Aware Neural Architecture Search for Mobile <https://arxiv.org/pdf/1807.11626.pdf>`_ paper.", 
weights_url=weights['mnasnet1_0'].url)
def mnasnet1_0(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "mnasnet1_0", verbose=True, 
                           weights=weights['mnasnet1_0'] if pretrained else None, **kwargs)

    return model

_meta = dict(input_size=[3,224,224], resize=[232,232], crop_pct=224/232)
@register_model("torchvision", arch="mnasnet1_3", **{**default_meta, **_meta}, arxiv="https://arxiv.org/pdf/1807.11626.pdf",
description="MNASNet with depth multiplier of 1.3 from `MnasNet: Platform-Aware Neural Architecture Search for Mobile <https://arxiv.org/pdf/1807.11626.pdf>`_ paper.", 
weights_url=weights['mnasnet1_3'].url)
def mnasnet1_3(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "mnasnet1_3", verbose=True, 
                           weights=weights['mnasnet1_3'] if pretrained else None, **kwargs)

    return model

# ===================================================
#  EfficientNet
# ===================================================

from torchvision.models.efficientnet import (
    EfficientNet_B0_Weights, EfficientNet_B1_Weights, EfficientNet_B2_Weights, EfficientNet_B3_Weights,
    EfficientNet_B4_Weights, EfficientNet_B5_Weights, EfficientNet_B6_Weights, EfficientNet_B7_Weights,
)

weights['efficientnet_b0'] = EfficientNet_B0_Weights.IMAGENET1K_V1
weights['efficientnet_b1'] = EfficientNet_B1_Weights.IMAGENET1K_V1
weights['efficientnet_b2'] = EfficientNet_B2_Weights.IMAGENET1K_V1
weights['efficientnet_b3'] = EfficientNet_B3_Weights.IMAGENET1K_V1
weights['efficientnet_b4'] = EfficientNet_B4_Weights.IMAGENET1K_V1
weights['efficientnet_b5'] = EfficientNet_B5_Weights.IMAGENET1K_V1
weights['efficientnet_b6'] = EfficientNet_B6_Weights.IMAGENET1K_V1
weights['efficientnet_b7'] = EfficientNet_B7_Weights.IMAGENET1K_V1

_meta = dict(input_size=(3,224,224), resize=(256,256), crop_pct=224/256, interpolation=InterpolationMode.BICUBIC)
@register_model("torchvision", arch="efficientnet_b0", **{**default_meta, **_meta}, arxiv="https://arxiv.org/abs/1905.11946",
description="EfficientNet B0 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.", 
weights_url=weights['efficientnet_b0'].url)
def efficientnet_b0(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "efficientnet_b0", verbose=True, 
                           weights=weights['efficientnet_b0'] if pretrained else None, **kwargs)

    return model

_meta = dict(input_size=(3,240,240), resize=(256,256), crop_pct=240/256, interpolation=InterpolationMode.BICUBIC)
@register_model("torchvision", arch="efficientnet_b1", **{**default_meta, **_meta}, arxiv="https://arxiv.org/abs/1905.11946",
description="EfficientNet B1 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.", 
weights_url=weights['efficientnet_b1'].url)
def efficientnet_b1(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "efficientnet_b1", verbose=True,
                           weights=weights['efficientnet_b1'] if pretrained else None, **kwargs)

    return model

_meta = dict(input_size=(3,288,288), resize=(288,288), crop_pct=288/288, interpolation=InterpolationMode.BICUBIC)
@register_model("torchvision", arch="efficientnet_b2", **{**default_meta, **_meta}, arxiv="https://arxiv.org/abs/1905.11946",
description="EfficientNet B2 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.", 
weights_url=weights['efficientnet_b2'].url)
def efficientnet_b2(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "efficientnet_b2", verbose=True,
                           weights=weights['efficientnet_b2'] if pretrained else None, **kwargs)

    return model

_meta = dict(input_size=(3,300,300), resize=(320,320), crop_pct=300/320, interpolation=InterpolationMode.BICUBIC)
@register_model("torchvision", arch="efficientnet_b3", **{**default_meta, **_meta}, arxiv="https://arxiv.org/abs/1905.11946",
description="EfficientNet B3 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.", 
weights_url=weights['efficientnet_b3'].url)
def efficientnet_b3(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "efficientnet_b3", verbose=True,
                           weights=weights['efficientnet_b3'] if pretrained else None, **kwargs)

    return model

_meta = dict(input_size=(3,380,380), resize=(384,384), crop_pct=380/384, interpolation=InterpolationMode.BICUBIC)
@register_model("torchvision", arch="efficientnet_b4", **{**default_meta, **_meta}, arxiv="https://arxiv.org/abs/1905.11946",
description="EfficientNet B4 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.", 
weights_url=weights['efficientnet_b4'].url)
def efficientnet_b4(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "efficientnet_b4", verbose=True,
                           weights=weights['efficientnet_b4'] if pretrained else None, **kwargs)

    return model

_meta = dict(input_size=(3,456,456), resize=(456,456), crop_pct=456/456, interpolation=InterpolationMode.BICUBIC)
@register_model("torchvision", arch="efficientnet_b5", **{**default_meta, **_meta}, arxiv="https://arxiv.org/abs/1905.11946",
description="EfficientNet B5 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.", 
weights_url=weights['efficientnet_b5'].url)
def efficientnet_b5(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "efficientnet_b5", verbose=True,
                           weights=weights['efficientnet_b5'] if pretrained else None, **kwargs)

    return model

_meta = dict(input_size=(3,528,528), resize=(528,528), crop_pct=528/528, interpolation=InterpolationMode.BICUBIC)
@register_model("torchvision", arch="efficientnet_b6", **{**default_meta, **_meta}, arxiv="https://arxiv.org/abs/1905.11946",
description="EfficientNet B6 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.", 
weights_url=weights['efficientnet_b6'].url)
def efficientnet_b6(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "efficientnet_b6", verbose=True,
                           weights=weights['efficientnet_b6'] if pretrained else None, **kwargs)

    return model

_meta = dict(input_size=(3,600,600), resize=(600,600), crop_pct=600/600, interpolation=InterpolationMode.BICUBIC)
@register_model("torchvision", arch="efficientnet_b7", **{**default_meta, **_meta}, arxiv="https://arxiv.org/abs/1905.11946",
description="EfficientNet B7 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.", 
weights_url=weights['efficientnet_b7'].url)
def efficientnet_b7(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "efficientnet_b7", verbose=True,
                           weights=weights['efficientnet_b7'] if pretrained else None, **kwargs)

    return model

# ===================================================
#  EfficientNet_v2
# ===================================================

from torchvision.models.efficientnet import (
    EfficientNet_V2_S_Weights, EfficientNet_V2_M_Weights, EfficientNet_V2_L_Weights
)

weights['efficientnet_v2_s'] = EfficientNet_V2_S_Weights.IMAGENET1K_V1
weights['efficientnet_v2_m'] = EfficientNet_V2_M_Weights.IMAGENET1K_V1
weights['efficientnet_v2_l'] = EfficientNet_V2_L_Weights.IMAGENET1K_V1

_meta = dict(input_size=(3,384,384), resize=(384,384), crop_pct=384/384, interpolation=InterpolationMode.BILINEAR)
@register_model("torchvision", arch="efficientnet_v2_s", **{**default_meta, **_meta}, arxiv="https://arxiv.org/abs/1905.11946",
description=" Constructs an EfficientNetV2-S architecture from `EfficientNetV2: Smaller Models and Faster Training <https://arxiv.org/abs/2104.00298>`_.", 
weights_url=weights['efficientnet_v2_s'].url)
def efficientnet_v2_s(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "efficientnet_v2_s", verbose=True,
                           weights=weights['efficientnet_v2_s'] if pretrained else None, **kwargs)

    return model

_meta = dict(input_size=(3,480,480), resize=(480,480), crop_pct=480/480, interpolation=InterpolationMode.BILINEAR)
@register_model("torchvision", arch="efficientnet_v2_m", **{**default_meta, **_meta}, arxiv="https://arxiv.org/abs/1905.11946",
description=" Constructs an EfficientNetV2-M architecture from `EfficientNetV2: Smaller Models and Faster Training <https://arxiv.org/abs/2104.00298>`_.", 
weights_url=weights['efficientnet_v2_m'].url)
def efficientnet_v2_m(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "efficientnet_v2_m", verbose=True,
                           weights=weights['efficientnet_v2_m'] if pretrained else None, **kwargs)

    return model

_meta = dict(input_size=(3,480,480), resize=(480,480), crop_pct=480/480, interpolation=InterpolationMode.BILINEAR,
             mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
@register_model("torchvision", arch="efficientnet_v2_l", **{**default_meta, **_meta}, arxiv="https://arxiv.org/abs/1905.11946",
description=" Constructs an EfficientNetV2-L architecture from `EfficientNetV2: Smaller Models and Faster Training <https://arxiv.org/abs/2104.00298>`_.", 
weights_url=weights['efficientnet_v2_l'].url)
def efficientnet_v2_l(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "efficientnet_v2_l", verbose=True,
                           weights=weights['efficientnet_v2_l'] if pretrained else None, **kwargs)

    return model

# ===================================================
#  EfficientNet_gen2
# ===================================================

from torchvision.models.efficientnet import (
    EfficientNet_B1_Weights
)

weights['efficientnet_b1_gen2'] = EfficientNet_B1_Weights.IMAGENET1K_V2

_meta = dict(input_size=(3,240,240), resize=(255,255), crop_pct=240/255, interpolation=InterpolationMode.BILINEAR)
@register_model("torchvision", arch="efficientnet_b1", **{**default_meta, **_meta}, arxiv="https://arxiv.org/abs/1905.11946",
description="EfficientNet B1 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper. These weights improve upon the results of the original paper by using a modified version of TorchVision's `new training recipe <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.", 
weights_url=weights['efficientnet_b1_gen2'].url)
def efficientnet_b1_gen2(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "efficientnet_b1", verbose=True,
                           weights=weights['efficientnet_b1_gen2'] if pretrained else None, **kwargs)

    return model

# ===================================================
#  RegNets
# ===================================================

from torchvision.models.regnet import (
    RegNet_Y_400MF_Weights, RegNet_Y_800MF_Weights, RegNet_Y_1_6GF_Weights, RegNet_Y_3_2GF_Weights,
    RegNet_Y_8GF_Weights, RegNet_Y_16GF_Weights, RegNet_Y_32GF_Weights, RegNet_Y_128GF_Weights, 
    
    RegNet_X_400MF_Weights, RegNet_X_800MF_Weights, RegNet_X_1_6GF_Weights, RegNet_X_3_2GF_Weights,
    RegNet_X_8GF_Weights, RegNet_X_16GF_Weights, RegNet_X_32GF_Weights
)

weights['regnet_y_400mf'] = RegNet_Y_400MF_Weights.IMAGENET1K_V1
weights['regnet_y_800mf'] = RegNet_Y_800MF_Weights.IMAGENET1K_V1
weights['regnet_y_1_6gf'] = RegNet_Y_1_6GF_Weights.IMAGENET1K_V1
weights['regnet_y_3_2gf'] = RegNet_Y_3_2GF_Weights.IMAGENET1K_V1
weights['regnet_y_8gf'] = RegNet_Y_8GF_Weights.IMAGENET1K_V1
weights['regnet_y_16gf'] = RegNet_Y_16GF_Weights.IMAGENET1K_V1
weights['regnet_y_32gf'] = RegNet_Y_32GF_Weights.IMAGENET1K_V1
# weights['regnet_y_128gf'] = RegNet_Y_128GF_Weights.IMAGENET1K_V1

weights['regnet_x_400mf'] = RegNet_X_400MF_Weights.IMAGENET1K_V1
weights['regnet_x_800mf'] = RegNet_X_800MF_Weights.IMAGENET1K_V1
weights['regnet_x_1_6gf'] = RegNet_X_1_6GF_Weights.IMAGENET1K_V1
weights['regnet_x_3_2gf'] = RegNet_X_3_2GF_Weights.IMAGENET1K_V1
weights['regnet_x_8gf'] = RegNet_X_8GF_Weights.IMAGENET1K_V1
weights['regnet_x_16gf'] = RegNet_X_16GF_Weights.IMAGENET1K_V1
weights['regnet_x_32gf'] = RegNet_X_32GF_Weights.IMAGENET1K_V1

@register_model("torchvision", arch="regnet_y_400mf", **default_meta, arxiv="https://arxiv.org/abs/2003.13678",
description="Constructs a regnet_y_400mf architecture from `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.",
weights_url=weights["regnet_y_400mf"].url)
def regnet_y_400mf(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "regnet_y_400mf", verbose=True, 
                           weights=weights["regnet_y_400mf"] if pretrained else None, **kwargs)

    return model

@register_model("torchvision", arch="regnet_y_800mf", **default_meta, arxiv="https://arxiv.org/abs/2003.13678",
description="Constructs a regnet_y_800mf architecture from `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.",
weights_url=weights["regnet_y_800mf"].url)
def regnet_y_800mf(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "regnet_y_800mf", verbose=True, 
                           weights=weights["regnet_y_800mf"] if pretrained else None, **kwargs)

    return model

@register_model("torchvision", arch="regnet_y_1_6gf", **default_meta, arxiv="https://arxiv.org/abs/2003.13678",
description="Constructs a regnet_y_1_6gf architecture from `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.",
weights_url=weights["regnet_y_1_6gf"].url)
def regnet_y_1_6gf(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "regnet_y_1_6gf", verbose=True, 
                           weights=weights["regnet_y_1_6gf"] if pretrained else None, **kwargs)

    return model

@register_model("torchvision", arch="regnet_y_3_2gf", **default_meta, arxiv="https://arxiv.org/abs/2003.13678",
description="Constructs a regnet_y_3_2gf architecture from `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.",
weights_url=weights["regnet_y_3_2gf"].url)
def regnet_y_3_2gf(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "regnet_y_3_2gf", verbose=True, 
                           weights=weights["regnet_y_3_2gf"] if pretrained else None, **kwargs)

    return model

@register_model("torchvision", arch="regnet_y_8gf", **default_meta, arxiv="https://arxiv.org/abs/2003.13678",
description="Constructs a regnet_y_8gf architecture from `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.",
weights_url=weights["regnet_y_8gf"].url)
def regnet_y_8gf(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "regnet_y_8gf", verbose=True, 
                           weights=weights["regnet_y_8gf"] if pretrained else None, **kwargs)

    return model

@register_model("torchvision", arch="regnet_y_16gf", **default_meta, arxiv="https://arxiv.org/abs/2003.13678",
description="Constructs a regnet_y_16gf architecture from `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.",
weights_url=weights["regnet_y_16gf"].url)
def regnet_y_16gf(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "regnet_y_16gf", verbose=True, 
                           weights=weights["regnet_y_16gf"] if pretrained else None, **kwargs)

    return model

@register_model("torchvision", arch="regnet_y_32gf", **default_meta, arxiv="https://arxiv.org/abs/2003.13678",
description="Constructs a regnet_y_32gf architecture from `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.",
weights_url=weights["regnet_y_32gf"].url)
def regnet_y_32gf(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "regnet_y_32gf", verbose=True, 
                           weights=weights["regnet_y_32gf"] if pretrained else None, **kwargs)

    return model

# @register_model("torchvision", arch="regnet_y_128gf", **default_meta, arxiv="https://arxiv.org/abs/2003.13678",
# description="Constructs a regnet_y_128gf architecture from `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.",
# weights_url=weights["regnet_y_128gf"].url)
# def regnet_y_128gf(pretrained=True, **kwargs):
#     model = torch.hub.load(_HUB_URL, "regnet_y_128gf", verbose=True, 
#                            weights=weights["regnet_y_128gf"] if pretrained else None, **kwargs)

#     return model

@register_model("torchvision", arch="regnet_x_400mf", **default_meta, arxiv="https://arxiv.org/abs/2003.13678",
description="Constructs a regnet_x_400mf architecture from `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.",
weights_url=weights["regnet_x_400mf"].url)
def regnet_x_400mf(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "regnet_x_400mf", verbose=True, 
                           weights=weights["regnet_x_400mf"] if pretrained else None, **kwargs)

    return model

@register_model("torchvision", arch="regnet_x_800mf", **default_meta, arxiv="https://arxiv.org/abs/2003.13678",
description="Constructs a regnet_x_800mf architecture from `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.",
weights_url=weights["regnet_x_800mf"].url)
def regnet_x_800mf(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "regnet_x_800mf", verbose=True, 
                           weights=weights["regnet_x_800mf"] if pretrained else None, **kwargs)

    return model

@register_model("torchvision", arch="regnet_x_1_6gf", **default_meta, arxiv="https://arxiv.org/abs/2003.13678",
description="Constructs a regnet_x_1_6gf architecture from `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.",
weights_url=weights["regnet_x_1_6gf"].url)
def regnet_x_1_6gf(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "regnet_x_1_6gf", verbose=True, 
                           weights=weights["regnet_x_1_6gf"] if pretrained else None, **kwargs)

    return model

@register_model("torchvision", arch="regnet_x_3_2gf", **default_meta, arxiv="https://arxiv.org/abs/2003.13678",
description="Constructs a regnet_x_3_2gf architecture from `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.",
weights_url=weights["regnet_x_3_2gf"].url)
def regnet_x_3_2gf(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "regnet_x_3_2gf", verbose=True, 
                           weights=weights["regnet_x_3_2gf"] if pretrained else None, **kwargs)

    return model

@register_model("torchvision", arch="regnet_x_8gf", **default_meta, arxiv="https://arxiv.org/abs/2003.13678",
description="Constructs a regnet_x_8gf architecture from `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.",
weights_url=weights["regnet_x_8gf"].url)
def regnet_x_8gf(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "regnet_x_8gf", verbose=True, 
                           weights=weights["regnet_x_8gf"] if pretrained else None, **kwargs)

    return model

@register_model("torchvision", arch="regnet_x_16gf", **default_meta, arxiv="https://arxiv.org/abs/2003.13678",
description="Constructs a regnet_x_16gf architecture from `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.",
weights_url=weights["regnet_x_16gf"].url)
def regnet_x_16gf(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "regnet_x_16gf", verbose=True, 
                           weights=weights["regnet_x_16gf"] if pretrained else None, **kwargs)

    return model

@register_model("torchvision", arch="regnet_x_32gf", **default_meta, arxiv="https://arxiv.org/abs/2003.13678",
description="Constructs a regnet_x_32gf architecture from `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.",
weights_url=weights["regnet_x_32gf"].url)
def regnet_x_32gf(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "regnet_x_32gf", verbose=True, 
                           weights=weights["regnet_x_32gf"] if pretrained else None, **kwargs)

    return model

# ===================================================
#  RegNets_gen2
# ===================================================

from torchvision.models.regnet import (
    RegNet_Y_400MF_Weights, RegNet_Y_800MF_Weights, RegNet_Y_1_6GF_Weights, RegNet_Y_3_2GF_Weights,
    RegNet_Y_8GF_Weights, RegNet_Y_16GF_Weights, RegNet_Y_32GF_Weights, RegNet_Y_128GF_Weights, 
    
    RegNet_X_400MF_Weights, RegNet_X_800MF_Weights, RegNet_X_1_6GF_Weights, RegNet_X_3_2GF_Weights,
    RegNet_X_8GF_Weights, RegNet_X_16GF_Weights, RegNet_X_32GF_Weights
)

weights['regnet_y_400mf_gen2'] = RegNet_Y_400MF_Weights.IMAGENET1K_V2
weights['regnet_y_800mf_gen2'] = RegNet_Y_800MF_Weights.IMAGENET1K_V2
weights['regnet_y_1_6gf_gen2'] = RegNet_Y_1_6GF_Weights.IMAGENET1K_V2
weights['regnet_y_3_2gf_gen2'] = RegNet_Y_3_2GF_Weights.IMAGENET1K_V2
weights['regnet_y_8gf_gen2'] = RegNet_Y_8GF_Weights.IMAGENET1K_V2
weights['regnet_y_16gf_gen2'] = RegNet_Y_16GF_Weights.IMAGENET1K_V2
weights['regnet_y_32gf_gen2'] = RegNet_Y_32GF_Weights.IMAGENET1K_V2

weights['regnet_x_400mf_gen2'] = RegNet_X_400MF_Weights.IMAGENET1K_V2
weights['regnet_x_800mf_gen2'] = RegNet_X_800MF_Weights.IMAGENET1K_V2
weights['regnet_x_1_6gf_gen2'] = RegNet_X_1_6GF_Weights.IMAGENET1K_V2
weights['regnet_x_3_2gf_gen2'] = RegNet_X_3_2GF_Weights.IMAGENET1K_V2
weights['regnet_x_8gf_gen2'] = RegNet_X_8GF_Weights.IMAGENET1K_V2
weights['regnet_x_16gf_gen2'] = RegNet_X_16GF_Weights.IMAGENET1K_V2
weights['regnet_x_32gf_gen2'] = RegNet_X_32GF_Weights.IMAGENET1K_V2

_meta = dict(input_size=(3,224,224), resize=(232,232), crop_pct=224/232)

@register_model("torchvision", arch="regnet_y_400mf", **{**default_meta, **_meta}, arxiv="https://arxiv.org/abs/2003.13678",
description="Constructs a regnet_y_400mf architecture from `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_. These weights improve upon the results of the original paper by using a modified version of TorchVision's `new training recipe <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.",
weights_url=weights["regnet_y_400mf_gen2"].url)
def regnet_y_400mf_gen2(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "regnet_y_400mf", verbose=True, 
                           weights=weights["regnet_y_400mf_gen2"] if pretrained else None, **kwargs)

    return model

@register_model("torchvision", arch="regnet_y_800mf", **{**default_meta, **_meta}, arxiv="https://arxiv.org/abs/2003.13678",
description="Constructs a regnet_y_800mf architecture from `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_. These weights improve upon the results of the original paper by using a modified version of TorchVision's `new training recipe <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.",
weights_url=weights["regnet_y_800mf_gen2"].url)
def regnet_y_800mf_gen2(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "regnet_y_800mf", verbose=True, 
                           weights=weights["regnet_y_800mf_gen2"] if pretrained else None, **kwargs)

    return model

@register_model("torchvision", arch="regnet_y_1_6gf", **{**default_meta, **_meta}, arxiv="https://arxiv.org/abs/2003.13678",
description="Constructs a regnet_y_1_6gf architecture from `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_. These weights improve upon the results of the original paper by using a modified version of TorchVision's `new training recipe <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.",
weights_url=weights["regnet_y_1_6gf_gen2"].url)
def regnet_y_1_6gf_gen2(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "regnet_y_1_6gf", verbose=True, 
                           weights=weights["regnet_y_1_6gf_gen2"] if pretrained else None, **kwargs)

    return model

@register_model("torchvision", arch="regnet_y_3_2gf", **{**default_meta, **_meta}, arxiv="https://arxiv.org/abs/2003.13678",
description="Constructs a regnet_y_3_2gf architecture from `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_. These weights improve upon the results of the original paper by using a modified version of TorchVision's `new training recipe <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.",
weights_url=weights["regnet_y_3_2gf_gen2"].url)
def regnet_y_3_2gf_gen2(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "regnet_y_3_2gf", verbose=True, 
                           weights=weights["regnet_y_3_2gf_gen2"] if pretrained else None, **kwargs)

    return model

@register_model("torchvision", arch="regnet_y_8gf", **{**default_meta, **_meta}, arxiv="https://arxiv.org/abs/2003.13678",
description="Constructs a regnet_y_8gf architecture from `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_. These weights improve upon the results of the original paper by using a modified version of TorchVision's `new training recipe <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.",
weights_url=weights["regnet_y_8gf_gen2"].url)
def regnet_y_8gf_gen2(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "regnet_y_8gf", verbose=True, 
                           weights=weights["regnet_y_8gf_gen2"] if pretrained else None, **kwargs)

    return model

@register_model("torchvision", arch="regnet_y_16gf", **{**default_meta, **_meta}, arxiv="https://arxiv.org/abs/2003.13678",
description="Constructs a regnet_y_16gf architecture from `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_. These weights improve upon the results of the original paper by using a modified version of TorchVision's `new training recipe <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.",
weights_url=weights["regnet_y_16gf_gen2"].url)
def regnet_y_16gf_gen2(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "regnet_y_16gf", verbose=True, 
                           weights=weights["regnet_y_16gf_gen2"] if pretrained else None, **kwargs)

    return model

@register_model("torchvision", arch="regnet_y_32gf", **{**default_meta, **_meta}, arxiv="https://arxiv.org/abs/2003.13678",
description="Constructs a regnet_y_32gf architecture from `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_. These weights improve upon the results of the original paper by using a modified version of TorchVision's `new training recipe <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.",
weights_url=weights["regnet_y_32gf_gen2"].url)
def regnet_y_32gf_gen2(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "regnet_y_32gf", verbose=True, 
                           weights=weights["regnet_y_32gf_gen2"] if pretrained else None, **kwargs)

    return model

@register_model("torchvision", arch="regnet_x_400mf", **{**default_meta, **_meta}, arxiv="https://arxiv.org/abs/2003.13678",
description="Constructs a regnet_x_400mf architecture from `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_. These weights improve upon the results of the original paper by using a modified version of TorchVision's `new training recipe <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.",
weights_url=weights["regnet_x_400mf_gen2"].url)
def regnet_x_400mf_gen2(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "regnet_x_400mf", verbose=True, 
                           weights=weights["regnet_x_400mf_gen2"] if pretrained else None, **kwargs)

    return model

@register_model("torchvision", arch="regnet_x_800mf", **{**default_meta, **_meta}, arxiv="https://arxiv.org/abs/2003.13678",
description="Constructs a regnet_x_800mf architecture from `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_. These weights improve upon the results of the original paper by using a modified version of TorchVision's `new training recipe <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.",
weights_url=weights["regnet_x_800mf_gen2"].url)
def regnet_x_800mf_gen2(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "regnet_x_800mf", verbose=True, 
                           weights=weights["regnet_x_800mf_gen2"] if pretrained else None, **kwargs)

    return model

@register_model("torchvision", arch="regnet_x_1_6gf", **{**default_meta, **_meta}, arxiv="https://arxiv.org/abs/2003.13678",
description="Constructs a regnet_x_1_6gf architecture from `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_. These weights improve upon the results of the original paper by using a modified version of TorchVision's `new training recipe <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.",
weights_url=weights["regnet_x_1_6gf_gen2"].url)
def regnet_x_1_6gf_gen2(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "regnet_x_1_6gf", verbose=True, 
                           weights=weights["regnet_x_1_6gf_gen2"] if pretrained else None, **kwargs)

    return model

@register_model("torchvision", arch="regnet_x_3_2gf", **{**default_meta, **_meta}, arxiv="https://arxiv.org/abs/2003.13678",
description="Constructs a regnet_x_3_2gf architecture from `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_. These weights improve upon the results of the original paper by using a modified version of TorchVision's `new training recipe <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.",
weights_url=weights["regnet_x_3_2gf_gen2"].url)
def regnet_x_3_2gf_gen2(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "regnet_x_3_2gf", verbose=True, 
                           weights=weights["regnet_x_3_2gf_gen2"] if pretrained else None, **kwargs)

    return model

@register_model("torchvision", arch="regnet_x_8gf", **{**default_meta, **_meta}, arxiv="https://arxiv.org/abs/2003.13678",
description="Constructs a regnet_x_8gf architecture from `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_. These weights improve upon the results of the original paper by using a modified version of TorchVision's `new training recipe <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.",
weights_url=weights["regnet_x_8gf_gen2"].url)
def regnet_x_8gf_gen2(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "regnet_x_8gf", verbose=True, 
                           weights=weights["regnet_x_8gf_gen2"] if pretrained else None, **kwargs)

    return model

@register_model("torchvision", arch="regnet_x_16gf", **{**default_meta, **_meta}, arxiv="https://arxiv.org/abs/2003.13678",
description="Constructs a regnet_x_16gf architecture from `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_. These weights improve upon the results of the original paper by using a modified version of TorchVision's `new training recipe <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.",
weights_url=weights["regnet_x_16gf_gen2"].url)
def regnet_x_16gf_gen2(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "regnet_x_16gf", verbose=True, 
                           weights=weights["regnet_x_16gf_gen2"] if pretrained else None, **kwargs)

    return model

@register_model("torchvision", arch="regnet_x_32gf", **{**default_meta, **_meta}, arxiv="https://arxiv.org/abs/2003.13678",
description="Constructs a regnet_x_32gf architecture from `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_. These weights improve upon the results of the original paper by using a modified version of TorchVision's `new training recipe <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.",
weights_url=weights["regnet_x_32gf_gen2"].url)
def regnet_x_32gf_gen2(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "regnet_x_32gf", verbose=True, 
                           weights=weights["regnet_x_32gf_gen2"] if pretrained else None, **kwargs)

    return model

# ===================================================
#  RegNets_swag_ft (end-to-end fine-tuning)
# ===================================================

from torchvision.models.regnet import (
    RegNet_Y_16GF_Weights, RegNet_Y_32GF_Weights, RegNet_Y_128GF_Weights
)

weights['regnet_y_16gf_swag_e2e_ft'] = RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_E2E_V1
weights['regnet_y_32gf_swag_e2e_ft'] = RegNet_Y_32GF_Weights.IMAGENET1K_SWAG_E2E_V1
weights['regnet_y_128gf_swag_e2e_ft'] = RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_E2E_V1

_meta = dict(input_size=(3,384,384), resize=(384,384), crop_pct=384/384, interpolation=InterpolationMode.BICUBIC)

@register_model("torchvision", arch="regnet_y_16gf", **{**default_meta, **_meta}, arxiv=["https://arxiv.org/abs/2003.13678", "https://arxiv.org/abs/2201.08371"], description="Constructs a regnet_y_16gf architecture from `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_. These weights are learnt via transfer learning by end-to-end fine-tuning the original `SWAG (pretrained on IG-3.6B) <https://arxiv.org/abs/2201.08371>`_ weights on ImageNet-1K data.",
weights_url=weights["regnet_y_16gf_swag_e2e_ft"].url)
def regnet_y_16gf_swag_e2e_ft(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "regnet_y_16gf", verbose=True, 
                           weights=weights["regnet_y_16gf_swag_e2e_ft"] if pretrained else None, **kwargs)

    return model

@register_model("torchvision", arch="regnet_y_32gf", **{**default_meta, **_meta}, arxiv=["https://arxiv.org/abs/2003.13678", "https://arxiv.org/abs/2201.08371"], description="Constructs a regnet_y_32gf architecture from `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_. These weights are learnt via transfer learning by end-to-end fine-tuning the original `SWAG (pretrained on IG-3.6B) <https://arxiv.org/abs/2201.08371>`_ weights on ImageNet-1K data.",
weights_url=weights["regnet_y_32gf_swag_e2e_ft"].url)
def regnet_y_32gf_swag_e2e_ft(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "regnet_y_32gf", verbose=True, 
                           weights=weights["regnet_y_32gf_swag_e2e_ft"] if pretrained else None, **kwargs)

    return model

@register_model("torchvision", arch="regnet_y_128gf", **{**default_meta, **_meta}, arxiv=["https://arxiv.org/abs/2003.13678", "https://arxiv.org/abs/2201.08371"], description="Constructs a regnet_y_128gf architecture from `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_. These weights are learnt via transfer learning by end-to-end fine-tuning the original `SWAG (pretrained on IG-3.6B) <https://arxiv.org/abs/2201.08371>`_ weights on ImageNet-1K data.",
weights_url=weights["regnet_y_128gf_swag_e2e_ft"].url)
def regnet_y_128gf_swag_e2e_ft(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "regnet_y_128gf", verbose=True, 
                           weights=weights["regnet_y_128gf_swag_e2e_ft"] if pretrained else None, **kwargs)

    return model

# ===================================================
#  RegNets_swag_lin (frozen weights, linear)
# ===================================================

from torchvision.models.regnet import (
    RegNet_Y_16GF_Weights, RegNet_Y_32GF_Weights, RegNet_Y_128GF_Weights
)

weights['regnet_y_16gf_swag_linear'] = RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_LINEAR_V1
weights['regnet_y_32gf_swag_linear'] = RegNet_Y_32GF_Weights.IMAGENET1K_SWAG_LINEAR_V1
weights['regnet_y_128gf_swag_linear'] = RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_LINEAR_V1

_meta = dict(input_size=(3,224,224), resize=(224,224), crop_pct=224/224, interpolation=InterpolationMode.BICUBIC)

@register_model("torchvision", arch="regnet_y_16gf", **{**default_meta, **_meta}, arxiv=["https://arxiv.org/abs/2003.13678", "https://arxiv.org/abs/2201.08371"], description="Constructs a regnet_y_16gf architecture from `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_. These weights are composed of the original frozen `SWAG (pretrained IG-3.6B) <https://arxiv.org/abs/2201.08371>`_ trunk weights and a linear classifier learnt on top of them trained on ImageNet-1K data.",
weights_url=weights["regnet_y_16gf_swag_linear"].url)
def regnet_y_16gf_swag_linear(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "regnet_y_16gf", verbose=True, 
                           weights=weights["regnet_y_16gf_swag_linear"] if pretrained else None, **kwargs)

    return model

@register_model("torchvision", arch="regnet_y_32gf", **{**default_meta, **_meta}, arxiv=["https://arxiv.org/abs/2003.13678", "https://arxiv.org/abs/2201.08371"], description="Constructs a regnet_y_32gf architecture from `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_. These weights are composed of the original frozen `SWAG (pretrained IG-3.6B) <https://arxiv.org/abs/2201.08371>`_ trunk weights and a linear classifier learnt on top of them trained on ImageNet-1K data.",
weights_url=weights["regnet_y_32gf_swag_linear"].url)
def regnet_y_32gf_swag_linear(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "regnet_y_32gf", verbose=True, 
                           weights=weights["regnet_y_32gf_swag_linear"] if pretrained else None, **kwargs)

    return model

@register_model("torchvision", arch="regnet_y_128gf", **{**default_meta, **_meta}, arxiv=["https://arxiv.org/abs/2003.13678", "https://arxiv.org/abs/2201.08371"], description="Constructs a regnet_y_128gf architecture from `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_. These weights are composed of the original frozen `SWAG (pretrained IG-3.6B) <https://arxiv.org/abs/2201.08371>`_ trunk weights and a linear classifier learnt on top of them trained on ImageNet-1K data.",
weights_url=weights["regnet_y_128gf_swag_linear"].url)
def regnet_y_128gf_swag_linear(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "regnet_y_128gf", verbose=True, 
                           weights=weights["regnet_y_128gf_swag_linear"] if pretrained else None, **kwargs)

    return model

# ===================================================
#  Vision Transformers
# ===================================================

from torchvision.models.vision_transformer import (
    ViT_B_16_Weights, ViT_B_32_Weights, ViT_L_16_Weights, ViT_L_32_Weights, ViT_H_14_Weights
)

weights['vit_b_16'] = ViT_B_16_Weights.IMAGENET1K_V1
weights['vit_b_32'] = ViT_B_32_Weights.IMAGENET1K_V1
weights['vit_l_16'] = ViT_L_16_Weights.IMAGENET1K_V1
weights['vit_l_32'] = ViT_L_32_Weights.IMAGENET1K_V1
# weights['vit_h_14'] = ViT_H_14_Weights.IMAGENET1K_V1

@register_model("torchvision", arch="vit_b_16", **default_meta, arxiv="https://arxiv.org/abs/2201.03545",
description="Constructs a vit_b_16 architecture from `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.", 
weights_url=weights['vit_b_16'].url)
def vit_b_16(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "vit_b_16", verbose=True, 
                           weights=weights['vit_b_16'] if pretrained else None, **kwargs)

    return model

@register_model("torchvision", arch="vit_b_32", **default_meta, arxiv="https://arxiv.org/abs/2201.03545",
description="Constructs a vit_b_32 architecture from `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.", 
weights_url=weights['vit_b_32'].url)
def vit_b_32(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "vit_b_32", verbose=True, 
                           weights=weights['vit_b_32'] if pretrained else None, **kwargs)

    return model

_meta = dict(input_size=(3,224,224), resize=(242,242), crop_pct=224/242)
@register_model("torchvision", arch="vit_l_16", **{**default_meta, **_meta}, arxiv="https://arxiv.org/abs/2201.03545",
description="Constructs a vit_l_16 architecture from `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.", 
weights_url=weights['vit_l_16'].url)
def vit_l_16(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "vit_l_16", verbose=True, 
                           weights=weights['vit_l_16'] if pretrained else None, **kwargs)

    return model

@register_model("torchvision", arch="vit_l_32", **default_meta, arxiv="https://arxiv.org/abs/2201.03545",
description="Constructs a vit_l_32 architecture from `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.", 
weights_url=weights['vit_l_32'].url)
def vit_l_32(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "vit_l_32", verbose=True, 
                           weights=weights['vit_l_32'] if pretrained else None, **kwargs)

    return model

# @register_model("torchvision", arch="vit_h_14", **default_meta, arxiv="https://arxiv.org/abs/2201.03545",
# description="Constructs a vit_h_14 architecture from `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.", 
# weights_url=weights['vit_h_14'].url)
# def vit_h_14(pretrained=True, **kwargs):
#     model = torch.hub.load(_HUB_URL, "vit_h_14", verbose=True, 
#                            weights=weights['vit_h_14'] if pretrained else None, **kwargs)

#     return model

# ===================================================
#  Vision Transformers_swag_e2e_ft
# ===================================================

from torchvision.models.vision_transformer import (
    ViT_B_16_Weights, ViT_B_32_Weights, ViT_L_16_Weights, ViT_L_32_Weights, ViT_H_14_Weights
)

weights['vit_b_16_swag_e2e_ft'] = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
weights['vit_l_16_swag_e2e_ft'] = ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1
weights['vit_h_14_swag_e2e_ft'] = ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1

_meta = dict(input_size=(3,384,384), resize=(384,384), crop_pct=384/384, interpolation=InterpolationMode.BICUBIC)

@register_model("torchvision", arch="vit_b_16", **{**default_meta, **_meta}, arxiv=["https://arxiv.org/abs/2201.03545", "https://arxiv.org/abs/2201.08371"], description="Constructs a vit_b_16 architecture from `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_. These weights are learnt via transfer learning by end-to-end fine-tuning the original `SWAG (pretrained on IG-3.6B) <https://arxiv.org/abs/2201.08371>`_ weights on ImageNet-1K data.", 
weights_url=weights['vit_b_16_swag_e2e_ft'].url)
def vit_b_16_swag_e2e_ft(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "vit_b_16", verbose=True, 
                           weights=weights['vit_b_16_swag_e2e_ft'] if pretrained else None, **kwargs)

    return model

_meta = dict(input_size=(3,512,512), resize=(512,512), crop_pct=512/512, interpolation=InterpolationMode.BICUBIC)

@register_model("torchvision", arch="vit_l_16", **{**default_meta, **_meta}, arxiv=["https://arxiv.org/abs/2201.03545", "https://arxiv.org/abs/2201.08371"], description="Constructs a vit_l_16 architecture from `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_. These weights are learnt via transfer learning by end-to-end fine-tuning the original `SWAG (pretrained on IG-3.6B) <https://arxiv.org/abs/2201.08371>`_ weights on ImageNet-1K data.", 
weights_url=weights['vit_l_16_swag_e2e_ft'].url)
def vit_l_16_swag_e2e_ft(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "vit_l_16", verbose=True, 
                           weights=weights['vit_l_16_swag_e2e_ft'] if pretrained else None, **kwargs)

    return model

_meta = dict(input_size=(3,518,518), resize=(518,518), crop_pct=518/518, interpolation=InterpolationMode.BICUBIC)

@register_model("torchvision", arch="vit_h_14", **{**default_meta, **_meta}, arxiv=["https://arxiv.org/abs/2201.03545", "https://arxiv.org/abs/2201.08371"], description="Constructs a vit_h_14 architecture from `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_. These weights are learnt via transfer learning by end-to-end fine-tuning the original `SWAG (pretrained on IG-3.6B) <https://arxiv.org/abs/2201.08371>`_ weights on ImageNet-1K data.", 
weights_url=weights['vit_h_14_swag_e2e_ft'].url)
def vit_h_14_swag_e2e_ft(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "vit_h_14", verbose=True, 
                           weights=weights['vit_h_14_swag_e2e_ft'] if pretrained else None, **kwargs)

    return model

# ===================================================
#  Vision Transformers_swag_linear
# ===================================================

from torchvision.models.vision_transformer import (
    ViT_B_16_Weights, ViT_B_32_Weights, ViT_L_16_Weights, ViT_L_32_Weights, ViT_H_14_Weights
)

weights['vit_b_16_swag_linear'] = ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
weights['vit_l_16_swag_linear'] = ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
weights['vit_h_14_swag_linear'] = ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1

_meta = dict(input_size=(3,224,224), resize=(224,224), crop_pct=224/224, interpolation=InterpolationMode.BICUBIC)

@register_model("torchvision", arch="vit_b_16", **{**default_meta, **_meta}, arxiv=["https://arxiv.org/abs/2201.03545", "https://arxiv.org/abs/2201.08371"], description="Constructs a vit_b_16 architecture from `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_. These weights are composed of the original frozen `SWAG (pretrained IG-3.6B) <https://arxiv.org/abs/2201.08371>`_ trunk weights and a linear classifier learnt on top of them trained on ImageNet-1K data.", 
weights_url=weights['vit_b_16_swag_linear'].url)
def vit_b_16_swag_linear(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "vit_b_16", verbose=True, 
                           weights=weights['vit_b_16_swag_linear'] if pretrained else None, **kwargs)

    return model

_meta = dict(input_size=(3,224,224), resize=(224,224), crop_pct=224/224, interpolation=InterpolationMode.BICUBIC)

@register_model("torchvision", arch="vit_l_16", **{**default_meta, **_meta}, arxiv=["https://arxiv.org/abs/2201.03545", "https://arxiv.org/abs/2201.08371"], description="Constructs a vit_l_16 architecture from `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_. These weights are composed of the original frozen `SWAG (pretrained IG-3.6B) <https://arxiv.org/abs/2201.08371>`_ trunk weights and a linear classifier learnt on top of them trained on ImageNet-1K data.", 
weights_url=weights['vit_l_16_swag_linear'].url)
def vit_l_16_swag_linear(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "vit_l_16", verbose=True, 
                           weights=weights['vit_l_16_swag_linear'] if pretrained else None, **kwargs)

    return model

_meta = dict(input_size=(3,224,224), resize=(224,224), crop_pct=224/224, interpolation=InterpolationMode.BICUBIC)

@register_model("torchvision", arch="vit_h_14", **{**default_meta, **_meta}, arxiv=["https://arxiv.org/abs/2201.03545", "https://arxiv.org/abs/2201.08371"], description="Constructs a vit_h_14 architecture from `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_. These weights are composed of the original frozen `SWAG (pretrained IG-3.6B) <https://arxiv.org/abs/2201.08371>`_ trunk weights and a linear classifier learnt on top of them trained on ImageNet-1K data.", 
weights_url=weights['vit_h_14_swag_linear'].url)
def vit_h_14_swag_linear(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "vit_h_14", verbose=True, 
                           weights=weights['vit_h_14_swag_linear'] if pretrained else None, **kwargs)

    return model

# ===================================================
#  ConvNeXt
# ===================================================

from torchvision.models.convnext import (
    ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights, ConvNeXt_Base_Weights, ConvNeXt_Large_Weights
)
    
weights['convnext_tiny'] = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
weights['convnext_small'] = ConvNeXt_Small_Weights.IMAGENET1K_V1
weights['convnext_base'] = ConvNeXt_Base_Weights.IMAGENET1K_V1
weights['convnext_large'] = ConvNeXt_Large_Weights.IMAGENET1K_V1

_meta = dict(input_size=(3,224,224), resize=(236,236), crop_pct=224/236, weights_url=weights['convnext_tiny'].url)
@register_model("torchvision", arch="convnext_tiny", **{**default_meta, **_meta}, arxiv="https://arxiv.org/abs/2201.03545",
description="ConvNeXt Tiny model architecture from the `A ConvNet for the 2020s <https://arxiv.org/abs/2201.03545>`_ paper.")
def convnext_tiny(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "convnext_tiny", verbose=True, 
                           weights=weights['convnext_tiny'] if pretrained else None, **kwargs)

    return model

_meta = dict(input_size=(3,224,224), resize=(230,230), crop_pct=224/230, weights_url=weights['convnext_small'].url) # really 230?
@register_model("torchvision", arch="convnext_small", **{**default_meta, **_meta}, arxiv="https://arxiv.org/abs/2201.03545",
description="ConvNeXt Small model architecture from the `A ConvNet for the 2020s <https://arxiv.org/abs/2201.03545>`_ paper.")
def convnext_small(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "convnext_small", verbose=True, 
                           weights=weights['convnext_small'] if pretrained else None, **kwargs)

    return model

_meta = dict(input_size=(3,224,224), resize=(236,236), crop_pct=224/236, weights_url=weights['convnext_base'].url)
@register_model("torchvision", arch="convnext_base", **{**default_meta, **_meta}, arxiv="https://arxiv.org/abs/2201.03545",
description="ConvNeXt Base model architecture from the `A ConvNet for the 2020s <https://arxiv.org/abs/2201.03545>`_ paper.")
def convnext_base(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "convnext_base", verbose=True, 
                           weights=weights['convnext_base'] if pretrained else None, **kwargs)

    return model

_meta = dict(input_size=(3,224,224), resize=(236,236), crop_pct=224/236, weights_url=weights['convnext_large'].url)
@register_model("torchvision", arch="convnext_large", **{**default_meta, **_meta}, arxiv="https://arxiv.org/abs/2201.03545",
description="ConvNeXt Large model architecture from the `A ConvNet for the 2020s <https://arxiv.org/abs/2201.03545>`_ paper.")
def convnext_large(pretrained=True, **kwargs):
    model = torch.hub.load(_HUB_URL, "convnext_large", verbose=True, 
                           weights=weights['convnext_large'] if pretrained else None, **kwargs)

    return model