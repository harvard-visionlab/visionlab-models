import torch
import torchvision
import torchvision.transforms as transforms
from functools import partial
from collections import OrderedDict
from PIL import Image
from pdb import set_trace

from . import models as _alexnet_pytorch_models

from ..feature_extractor import FeatureExtractor, get_layer_names, get_layer_type
from ..common import OpenImage
from ..transforms import InverseNormalize
    
@torch.no_grad()
def forward_features(self, imgs, layer_names=None):
    self.eval()
    features = OrderedDict({})
    if layer_names is None:
        features = self(imgs)
    else:
        with FeatureExtractor(self, layer_names) as extractor:
            feats = extractor(imgs) 
            for layer_name, feat in feats.items():
                if isinstance(feat, tuple):
                    for idx,f in enumerate(feat):
                        name = f'{layer_name}.{idx}'
                        features[name] = f
                else:
                    # store the set of output features
                    features[layer_name] = feat   

    return features    
    
def load_model(model_name, *args, **kwargs):
    model = _alexnet_pytorch_models.__dict__[model_name](*args, **kwargs)
    meta = model.meta
    
    val_transforms = transforms.Compose([
        OpenImage(),
        transforms.Resize(meta['resize'][-1], interpolation=meta['interpolation']),
        transforms.CenterCrop(meta['input_size'][-2:]),
        transforms.ToTensor(),
        transforms.Normalize(mean=meta['mean'], std=meta['std']),
    ])
    
    test_transforms = transforms.Compose([
        OpenImage(),
        transforms.Resize(meta['input_size'][-1], interpolation=meta['interpolation']),
        transforms.CenterCrop(meta['input_size'][-2:]),
        transforms.ToTensor(),
        transforms.Normalize(mean=meta['mean'], std=meta['std']),
    ])
    
    to_pil = transforms.Compose([
        InverseNormalize(mean=meta['mean'], std=meta['std']),
        transforms.ToPILImage(),
    ])
    
    preprocess = dict(val_transforms=val_transforms, test_transforms=test_transforms, to_pil=to_pil)
    
    model.get_features = partial(forward_features, model)
    model.layer_type = partial(get_layer_type, model)
    model.layer_names = [ln for ln in get_layer_names(model) if 'aux' not in ln]
    model.layer_types = [get_layer_type(model, ln) for ln in model.layer_names]

    # get the names of the main blocks (useful for less fine-grained layerwise analysis)
    exclude = []
    selected_layers = [ln for ln,lt in zip(model.layer_names,model.layer_types)
                       if lt not in ['Identity'] and not "drop" in ln and ln not in exclude]
    block_names = list(OrderedDict.fromkeys([".".join(ln.split(".")[0:2]) for ln in selected_layers]))
    model.block_names = block_names

    return model, preprocess