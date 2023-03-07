import os
import torch
import torchvision
import json

from torchvision.transforms import InterpolationMode

from ...common import _IMAGENET_MEAN, _IMAGENET_STD
from ...registry import register_model

default_model_dir = os.path.join(torch.hub.get_dir(), "checkpoints")

cache_filenames = {
  'alexnet_pytorch-supervised1k-imagenet1k-7be5be79': 'alexnet-owt-7be5be79.pth',
}

model_urls = {
    'alexnet_pytorch-supervised1k-imagenet1k-7be5be79':
        'https://download.pytorch.org/models/alexnet-owt-7be5be79.pth',
}

default_meta = dict(    
    num_classes = 1000,
    input_size=(3,224,224),   
    crop_pct=0.875,
    resize=(256,256),
    input_range=[0, 1],
    mean=_IMAGENET_MEAN,
    std=_IMAGENET_STD,
    interpolation=InterpolationMode.BILINEAR,
    repo="https://github.com/pytorch/vision",
    task="supervised1k",
    dataset='imagenet-1k',
    datasize="1.3M",
    bib=json.dumps('''""'''),
)

def _load_weights(model, model_name, url, model_dir=default_model_dir, verbose=True):   
    cache_filename = cache_filenames[model_name]

    if verbose:
        print(f"==> {model_name}, loading checkpoint: {url}")

    checkpoint = torch.hub.load_state_dict_from_url(
        url = url,
        model_dir = model_dir,
        map_location = 'cpu',
        progress = True,
        check_hash = True,
        file_name = cache_filename
    )

    if 'model' in checkpoint:
        msg = model.load_state_dict(checkpoint['model'], strict=True)
    elif 'state_dict' in checkpoint:
        msg = model.load_state_dict(checkpoint['state_dict'], strict=True)
    else:
        msg = model.load_state_dict(checkpoint, strict=True)

    r = torch.hub.HASH_REGEX.search(cache_filename)  # r is Optional[Match[str]]
    model.hashid = r.group(1) if r else None
    model.weights_file = os.path.join(model_dir, cache_filename)

    if verbose:
        print(f"==> state loaded: {msg}")

    return model 
  
@register_model("alexnet_pytorch", arch="alexnet_pytorch", hashid="7be5be79", weights_url=model_urls["alexnet_pytorch-supervised1k-imagenet1k-7be5be79"], 
description="Alexnet(PyTorch) Trained on ImageNet1k Classification", **default_meta)    
def alexnet_7be5be79(task="supervised1k", dataset="imagenet1k", hashid="7be5be79", model_dir=default_model_dir, verbose=True):

    model = torchvision.models.alexnet(pretrained=False)
    if task is not None and dataset is not None:
        model_name = f"alexnet_pytorch-{task}-{dataset}-{hashid}"
        url = model_urls[model_name]
        model = _load_weights(model, model_name, url, model_dir=model_dir, verbose=verbose)
    return model