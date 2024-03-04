import os
import torch
import torchvision
import json
from pathlib import Path

from torchvision.transforms import InterpolationMode

from ...common import _IMAGENET_MEAN, _IMAGENET_STD
from ...registry import register_model
from ...load_weights import load_weights, default_model_dir

base_url = "https://s3.us-east-1.wasabisys.com/visionlab-models/rgeirhos_stylized_imagenet/"

default_meta = dict(    
    num_classes = 1000,
    input_size=(3,224,224),   
    crop_pct=0.875,
    resize=256,
    input_range=[0, 1],
    mean=_IMAGENET_MEAN,
    std=_IMAGENET_STD,
    interpolation=InterpolationMode.BILINEAR,
    repo="https://github.com/rgeirhos/texture-vs-shape",
    task="supervised1k",
    dataset='in1k',
    datasize="1.3M",
    bib=json.dumps('''""'''),
)
            
metadata = dict(
    alexnet_SIN = {**default_meta, **dict(
        arch="alexnet_pytorch", 
        task="supervised1k",
        dataset="stylized-imagenet1k",
        hashid="b4aa5238",               
        weights_url=base_url+"alexnet_train_60_epochs_lr0.001-b4aa5238.pth.tar", 
        description="Alexnet(PyTorch) Trained on Stylized IN1K Classification", 
    )},
    alexnet_IN = {**default_meta, **dict(
        arch="alexnet_pytorch", 
        task="supervised1k",
        dataset="imagenet1k",
        hashid="7be5be79",               
        weights_url="https://download.pytorch.org/models/alexnet-owt-7be5be79.pth", 
        description="Alexnet(PyTorch) Trained on Standard IN1K Classification", 
    )},
    vgg16_SIN = {**default_meta, **dict(
        arch="vgg16_pytorch", 
        task="supervised1k",
        dataset="stylized-imagenet1k",
        hashid="6c6fcc9f",               
        weights_url=base_url+"vgg16_train_60_epochs_lr0.01-6c6fcc9f.pth.tar", 
        description="VGG16(PyTorch) Trained on Stylized IN1K Classification", 
    )},
    vgg16_IN = {**default_meta, **dict(
        arch="vgg16_pytorch", 
        task="supervised1k",
        dataset="imagenet1k",
        hashid="397923af",               
        weights_url="https://download.pytorch.org/models/vgg16-397923af.pth", 
        description="VGG16(PyTorch) Trained on Standard IN1K Classification", 
    )},
    resnet50_SIN = {**default_meta, **dict(
        arch="resnet50_pytorch", 
        task="supervised1k",
        dataset="stylized-imagenet1k",
        hashid="c8e5653e",               
        weights_url=base_url+"resnet50_train_60_epochs-c8e5653e.pth.tar", 
        description="Resnet50(PyTorch) Trained on Stylized IN1K Classification", 
    )},
    resnet50_IN = {**default_meta, **dict(
        arch="resnet50_pytorch", 
        task="supervised1k",
        dataset="imagenet1k",
        hashid="0676ba61",               
        weights_url="https://download.pytorch.org/models/resnet50-0676ba61.pth", 
        description="Resnet50(PyTorch) Trained on Standard IN1K Classification", 
    )},
    resnet50_SIN_and_IN = {**default_meta, **dict(
        arch="resnet50_pytorch", 
        task="supervised1k",
        dataset="stylized-imagenet1k+imagenet1k",
        hashid="2a0d100e",               
        weights_url=base_url+"resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar", 
        description="Resnet50(PyTorch) Trained on Sylized-IN1K AND IN1K Classification", 
    )},
    resnet50_SIN_and_IN_finetuned_on_IN = {**default_meta, **dict(
        arch="resnet50_pytorch", 
        task="supervised1k",
        dataset="stylized-imagenet1k+imagenet1k",
        hashid="ca06340c",               
        weights_url=base_url+"resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar", 
        description="Resnet50(PyTorch) Trained on Sylized-IN1K AND IN1K, Fine-Tuned on IN1K Classification", 
    )}    
)

@register_model("stylized_imagenet", **metadata['alexnet_SIN'])
def alexnet_SIN(pretrained=True, model_dir=default_model_dir, verbose=True):

    model = torchvision.models.alexnet()
    if pretrained:
        meta = metadata['alexnet_SIN']
        model_name = f"alexnet_SIN_{meta['hashid']}"
        url = meta['weights_url']
        cache_filename = f"stylized_imagenet_{Path(url).name}"
        if verbose: print(f"==> {model_name}")
        model = load_weights(model, url, cache_filename=cache_filename, model_dir=model_dir, verbose=verbose)
    return model

@register_model("stylized_imagenet", **metadata['alexnet_IN'])
def alexnet_IN(pretrained=True, model_dir=default_model_dir, verbose=True):

    model = torchvision.models.alexnet()
    if pretrained:
        meta = metadata['alexnet_IN']
        model_name = f"alexnet_IN_{meta['hashid']}"
        url = meta['weights_url']
        cache_filename = f"stylized_imagenet_{Path(url).name}"
        if verbose: print(f"==> {model_name}")
        model = load_weights(model, url, cache_filename=cache_filename, model_dir=model_dir, verbose=verbose)
    return model

@register_model("stylized_imagenet", **metadata['vgg16_SIN'])
def vgg16_SIN(pretrained=True, model_dir=default_model_dir, verbose=True):

    model = torchvision.models.vgg16()
    if pretrained:
        meta = metadata['vgg16_SIN']
        model_name = f"vgg16_SIN_{meta['hashid']}"
        url = meta['weights_url']
        cache_filename = f"stylized_imagenet_{Path(url).name}"
        if verbose: print(f"==> {model_name}")
        model = load_weights(model, url, cache_filename=cache_filename, model_dir=model_dir, verbose=verbose)
    return model

@register_model("stylized_imagenet", **metadata['vgg16_IN'])
def vgg16_IN(pretrained=True, model_dir=default_model_dir, verbose=True):

    model = torchvision.models.vgg16()
    if pretrained:
        meta = metadata['vgg16_IN']
        model_name = f"vgg16_IN_{meta['hashid']}"
        url = meta['weights_url']
        cache_filename = f"stylized_imagenet_{Path(url).name}"
        if verbose: print(f"==> {model_name}")
        model = load_weights(model, url, cache_filename=cache_filename, model_dir=model_dir, verbose=verbose)
    return model

@register_model("stylized_imagenet", **metadata['resnet50_SIN'])
def resnet50_SIN(pretrained=True, model_dir=default_model_dir, verbose=True):

    model = torchvision.models.resnet50()
    if pretrained:
        meta = metadata['resnet50_SIN']
        model_name = f"resnet50_SIN_{meta['hashid']}"
        url = meta['weights_url']
        cache_filename = f"stylized_imagenet_{Path(url).name}"
        if verbose: print(f"==> {model_name}")
        model = load_weights(model, url, cache_filename=cache_filename, model_dir=model_dir, verbose=verbose)
    return model

@register_model("stylized_imagenet", **metadata['resnet50_IN'])
def resnet50_IN(pretrained=True, model_dir=default_model_dir, verbose=True):

    model = torchvision.models.resnet50()
    if pretrained:
        meta = metadata['resnet50_IN']
        model_name = f"resnet50_IN_{meta['hashid']}"
        url = meta['weights_url']
        cache_filename = f"stylized_imagenet_{Path(url).name}"
        if verbose: print(f"==> {model_name}")
        model = load_weights(model, url, cache_filename=cache_filename, model_dir=model_dir, verbose=verbose)
    return model

@register_model("stylized_imagenet", **metadata['resnet50_SIN_and_IN'])
def resnet50_SIN_and_IN(pretrained=True, model_dir=default_model_dir, verbose=True):

    model = torchvision.models.resnet50()
    if pretrained:
        meta = metadata['resnet50_SIN_and_IN']
        model_name = f"resnet50_SIN_and_IN_{meta['hashid']}"
        url = meta['weights_url']
        cache_filename = f"stylized_imagenet_{Path(url).name}"
        if verbose: print(f"==> {model_name}")
        model = load_weights(model, url, cache_filename=cache_filename, model_dir=model_dir, verbose=verbose)
    return model

@register_model("stylized_imagenet", **metadata['resnet50_SIN_and_IN_finetuned_on_IN'])
def resnet50_SIN_and_IN_finetuned_on_IN(pretrained=True, model_dir=default_model_dir, verbose=True):

    model = torchvision.models.resnet50()
    if pretrained:
        meta = metadata['resnet50_SIN_and_IN_finetuned_on_IN']
        model_name = f"resnet50_SIN_and_IN_finetuned_on_IN_{meta['hashid']}"
        url = meta['weights_url']
        cache_filename = f"stylized_imagenet_{Path(url).name}"
        if verbose: print(f"==> {model_name}")
        model = load_weights(model, url, cache_filename=cache_filename, model_dir=model_dir, verbose=verbose)
    return model