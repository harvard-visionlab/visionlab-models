'''
    Adapted from https://github.com/bethgelab/model-vs-human/blob/master/modelvshuman/models/registry.py
'''
from collections import defaultdict
from pdb import set_trace

__all__ = ['list_families', 'list_models', 'show_metadata']

_model_registry = defaultdict(dict)  # mapping of model names to entrypoint fns

def register_model(model_family, **kwargs):
    def inner_decorator(fn):
        # add entries to registry dict/sets
        model_name = fn.__name__
        meta = dict(kwargs)
        meta['model_family'] = model_family
        meta['model_name'] = model_name
        
        if 'model_family' in _model_registry:
            assert model_name not in _model_registry[model_family], f"Oops, {model_name} alread in _model_registry[{model_family}]"
            
        _model_registry[model_family][model_name] = meta        
        
        # model wrapper allows us to append meta data when instantiating model
        def wrapper(*args, **kwargs):
            model = fn(*args, **kwargs)
            model.meta = meta
                
            return model
        
        return wrapper
    return inner_decorator

def list_families():
    """ Return list of available model families, sorted alphabetically
    """
    return list(_model_registry.keys())

def list_models(model_family):
    """ Return list of available model names, sorted alphabetically
    """
    return list(_model_registry[model_family].keys())

def show_metadata(model_family, model_name):
    """ Return list of available model names, sorted alphabetically
    """
    return _model_registry[model_family][model_name]