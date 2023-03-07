from pdb import set_trace
from .registry import list_families, list_models, show_metadata
from . import alexnet_pytorch

# from . import bagnets
# from . import barlowtwins
# from . import bitm
# # from . import clip
# # from . import cornet
# # from . import debiased
# # from . import dino
# from . import efficientnet
# # from . import equivariant
# # from . import ipcl
# from . import mae
# from . import pycontrast
# # from . import rgeirhos
# from . import robust_models
# # from . import rotnet
# # from . import rsc
# # from . import simclr
# # from . import sparsenets
# # from . import swav
# from . import swsl
# # from . import taskonomy
# from . import timm as timm_models
# from . import torchvision_models as tv
# # from . import van
# from . import vicreg
# # from . import vissl
# # from . import vonenet
# # from . import vnet

def load_model(source, model_name, **kwargs):
    
    if source == "alexnet_pytorch":
        return alexnet_pytorch.load_model(model_name, **kwargs)
    
    if source == "bagnets":
        return bagnets.load_model(model_name, **kwargs)
    
    if source == "barlowtwins":
        return barlowtwins.load_model(model_name, **kwargs)
    
    #if source == "barlow_twins":
    #    from . import barlow_twins
    #    return barlow_twins.load_model(model_name, **kwargs)
    
    if source == "bitm":
        return bitm.load_model(model_name, **kwargs)
    
    if source == "clip":
        from . import clip
        return clip.load_model(model_name, **kwargs)
    
    if model_name.startswith("cornet"):
        from . import cornet
        return cornet.load_model(model_name, **kwargs)
    
    if source == "debiased":
        from . import debiased
        return debiased.load_model(model_name, **kwargs)
    
    if source == "dino":
        from . import dino
        return dino.load_model(model_name, **kwargs)
    
    if source == "efficientnet":
        return efficientnet.load_model(model_name, **kwargs)
    
    if source == "ipcl":
        from . import ipcl 
        return ipcl.load_model(model_name, **kwargs)
    
    if source == "mae":
        return mae.load_model(model_name, **kwargs)
    
    if source == "pycontrast":
        return pycontrast.load_model(model_name, **kwargs)
    
    if source == "rgeirhos":
        from . import rgeirhos
        return rgeirhos.load_model(model_name, **kwargs)
    
    if source == "robust_models":
        return robust_models.load_model(model_name, **kwargs)
    
    if source == "rsc":
        from . import rsc
        return rsc.load_model(model_name, **kwargs)
    
    if source == "simclr":
        from . import simclr
        return simclr.load_model(model_name, **kwargs)
    
    if source == "sparsenets":
        from . import sparsenets
        return sparsenets.load_model(model_name, **kwargs)
    
    if source == "swsl":
        return swsl.load_model(model_name, **kwargs)
    
    if source == "taskonomy":
        from . import taskonomy
        return taskonomy.load_model(model_name, **kwargs)
    
    if source == "timm":
        return timm_models.load_model(model_name, **kwargs)
    
    if source == "torchvision":        
        return tv.load_model(model_name, **kwargs)
    
    if source == "van":
        from . import van
        return van.load_model(model_name, **kwargs)
    
    if source == "vicreg":
        return vicreg.load_model(model_name, **kwargs)
    
    #if source == "vissl":
    #    from . import vissl
    #    return vissl.load_model(model_name, **kwargs)
    
    if source == "vnet":
        from . import vnet
        return vnet.load_model(model_name, **kwargs)
    
    raise ValueError(f'Unkown model source {source}')