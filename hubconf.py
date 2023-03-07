import os
import torch
import torchvision

import models as vsl_models
from models.registry import *

dependencies = ['torch', 'torchvision']

default_model_dir = os.path.join(torch.hub.get_dir(), "checkpoints")

cache_filenames = {
  'alexnet_pytorch-category_supervised-imagenet1k-7be5be79': 'alexnet-owt-7be5be79.pth',
}

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
  
# def alexnet_pytorch(task="supervised1k", dataset="imagenet1k", hashid="7be5be79", model_dir=default_model_dir, verbose=True):
#     urls = {
#         "supervised1k": {
#             "imagenet1k": {
#                 "7be5be79":  "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth"
#             }
#         }
#     }

#     model = torchvision.models.alexnet(pretrained=False)
#     if task is not None and dataset is not None:
#         model_name = f"alexnet_pytorch-{task}-{dataset}-{rep}"
#         url = urls[task][dataset][rep]
#         model = _load_weights(model, model_name, url, model_dir=model_dir, verbose=verbose)
#     return 

def list_models(model_source=None, pattern=''):
    print(model_source, pattern)
    return ['alexnet_pytorch']