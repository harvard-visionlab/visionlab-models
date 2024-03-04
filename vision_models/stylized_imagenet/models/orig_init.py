import torch
import torch.nn as nn
from typing import Any, Optional
from torchvision.models._utils import _ovewrite_named_param

from torchvision import models as tv_models
from torchvision.models import AlexNet, ResNet, VGG
from torchvision import transforms
from torchvision.models._meta import _IMAGENET_CATEGORIES

from .._weights_api import Weights, WeightsEnum
from ..registry import register_model

__all__ = [
    'Resnet50_SIN_Weights',
    'VGG16_SIN_Weights',
    'AlexNet_SIN_Weights',
    'resnet50_SIN',
    'resnet50_SIN_and_IN',
    'resnet50_SIN_and_IN_finetuned_on_IN',
    'vgg16_SIN',
    'alexnet_SIN',
]

default_transforms = dict(
    train_transform=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]),
    val_transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    test_transform=transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
)

class Resnet50_SIN_Weights(WeightsEnum):        
    SIN = Weights(
        url="https://s3.us-east-1.wasabisys.com/visionlab-models/rgeirhos_stylized_imagenet/resnet50_train_60_epochs-c8e5653e.pth.tar",
        transforms=default_transforms,
        meta={
            "repo": "https://github.com/rgeirhos/texture-vs-shape",
            "log_url": None,
            "params_url": None,
            "train_script": None,
            "task": "supervised1k",
            "dataset": "stylized-imagenet",
            "arch": "resnet50",
            "num_params": 25557032,
            "categories": _IMAGENET_CATEGORIES,
            "_metrics": {
            },
            "_docs": """                
            """,
        },
    )
    SIN_and_IN = Weights(
        url="https://s3.us-east-1.wasabisys.com/visionlab-models/rgeirhos_stylized_imagenet/resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar",
        transforms=default_transforms,
        meta={
            "repo": "https://github.com/rgeirhos/texture-vs-shape",
            "log_url": None,
            "params_url": None,
            "train_script": None,
            "task": "supervised1k",
            "dataset": "stylized-imagenet+imagenet",
            "arch": "resnet50",
            "num_params": 25557032,
            "categories": _IMAGENET_CATEGORIES,
            "_metrics": {
            },
            "_docs": """                
            """,
        },
    )
    SIN_and_IN_finetuned_on_IN = Weights(
        url="https://s3.us-east-1.wasabisys.com/visionlab-models/rgeirhos_stylized_imagenet/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar",
        transforms=default_transforms,
        meta={
            "repo": "https://github.com/rgeirhos/texture-vs-shape",
            "log_url": None,
            "params_url": None,
            "train_script": None,
            "task": "supervised1k",
            "dataset": "stylized-imagenet+imagenet=>finetuned_on_imagenet",
            "arch": "resnet50",
            "num_params": 25557032,
            "categories": _IMAGENET_CATEGORIES,
            "_metrics": {
            },
            "_docs": """                
            """,
        },
    )
    DEFAULT = SIN
    
class VGG16_SIN_Weights(WeightsEnum):        
    SIN = Weights(
        url="https://s3.us-east-1.wasabisys.com/visionlab-models/rgeirhos_stylized_imagenet/vgg16_train_60_epochs_lr0.01-6c6fcc9f.pth.tar",
        transforms=default_transforms,
        meta={
            "repo": "https://github.com/rgeirhos/texture-vs-shape",
            "log_url": None,
            "params_url": None,
            "train_script": None,
            "task": "supervised1k",
            "dataset": "stylized-imagenet",
            "arch": "resnet50",
            "num_params": 138357544,
            "categories": _IMAGENET_CATEGORIES,
            "_metrics": {
            },
            "_docs": """                
            """,
        },
    )
    DEFAULT = SIN
    
class AlexNet_SIN_Weights(WeightsEnum):        
    SIN = Weights(
        url="https://s3.us-east-1.wasabisys.com/visionlab-models/rgeirhos_stylized_imagenet/alexnet_train_60_epochs_lr0.001-b4aa5238.pth.tar",
        transforms=default_transforms,
        meta={
            "repo": "https://github.com/rgeirhos/texture-vs-shape",
            "log_url": None,
            "params_url": None,
            "train_script": None,
            "task": "supervised1k",
            "dataset": "stylized-imagenet",
            "arch": "resnet50",
            "num_params": 61100840,
            "categories": _IMAGENET_CATEGORIES,
            "_metrics": {
            },
            "_docs": """                
            """,
        },
    )
    DEFAULT = SIN    
    
@register_model('stylized_imagenet')
def resnet50_SIN(*, pretrained=True, weights=None, progress: bool = True, **kwargs: Any) -> ResNet:
    if pretrained and weights is None:
        weights = Resnet50_SIN_Weights.SIN
        
    weights = Resnet50_SIN_Weights.verify(weights)
    
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
        
    model = tv_models.resnet50(**kwargs)
    
    if weights is not None:
        msg = model.load_state_dict(weights.get_state_dict(progress=progress))
        print(msg)
        
    return model, weights

@register_model('stylized_imagenet')
def resnet50_SIN_and_IN(*, pretrained=True, weights=None, progress: bool = True, **kwargs: Any) -> ResNet:
    if pretrained and weights is None:
        weights = Resnet50_SIN_Weights.SIN_and_IN
        
    weights = Resnet50_SIN_Weights.verify(weights)
    
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
        
    model = tv_models.resnet50(**kwargs)
    
    if weights is not None:
        msg = model.load_state_dict(weights.get_state_dict(progress=progress))
        print(msg)
        
    return model, weights

@register_model('stylized_imagenet')
def resnet50_SIN_and_IN_finetuned_on_IN(*, pretrained=True, weights=None, progress: bool = True, **kwargs: Any) -> ResNet:
    if pretrained and weights is None:
        weights = Resnet50_SIN_Weights.SIN_and_IN_finetuned_on_IN
        
    weights = Resnet50_SIN_Weights.verify(weights)
    
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
        
    model = tv_models.resnet50(**kwargs)
    
    if weights is not None:
        msg = model.load_state_dict(weights.get_state_dict(progress=progress))
        print(msg)
        
    return model, weights

# @register_model('stylized_imagenet')
# def vgg16_SIN(*, pretrained=True, weights=None, progress: bool = True, **kwargs: Any) -> VGG:
#     if pretrained and weights is None:
#         weights = VGG16_SIN_Weights.SIN
        
#     weights = VGG16_SIN_Weights.verify(weights)
    
#     if weights is not None:
#         _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
        
#     model = tv_models.vgg16(**kwargs)
    
#     if weights is not None:
#         msg = model.load_state_dict(weights.get_state_dict(progress=progress))
#         print(msg)
        
#     return model, weights

# @register_model('stylized_imagenet')
# def alexnet_SIN(*, pretrained=True, weights=None, progress: bool = True, **kwargs: Any) -> AlexNet:
#     if pretrained and weights is None:
#         weights = AlexNet_SIN_Weights.SIN
        
#     weights = AlexNet_SIN_Weights.verify(weights)
    
#     if weights is not None:
#         _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
        
#     model = tv_models.alexnet(**kwargs)
    
#     if weights is not None:
#         msg = model.load_state_dict(weights.get_state_dict(progress=progress))
#         print(msg)
        
#     return model, weights