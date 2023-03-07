'''
    Adapted from https://github.com/bethgelab/model-vs-human/blob/master/modelvshuman/models/registry.py
'''
from collections import defaultdict
from pdb import set_trace

__all__ = ['list_sources', 'list_models', 'show_metadata']

_model_registry = defaultdict(dict)  # mapping of model names to entrypoint fns

def register_model(model_source, **kwargs):
    def inner_decorator(fn):
        # add entries to registry dict/sets
        model_name = fn.__name__
        meta = dict(kwargs)
        meta['model_source'] = model_source
        meta['model_name'] = model_name
        
        if 'model_source' in _model_registry:
            assert model_name not in _model_registry[model_source], f"Oops, {model_name} alread in _model_registry[{model_source}]"
            
        _model_registry[model_source][model_name] = meta        
        
        # model wrapper allows us to append meta data when instantiating model
        def wrapper(*args, **kwargs):
            model = fn(*args, **kwargs)
            model.meta = meta
                
            return model
        
        return wrapper
    return inner_decorator

def list_sources():
    """ Return list of available model names, sorted alphabetically
    """
    return list(_model_registry.keys())

def list_models(model_source):
    """ Return list of available model names, sorted alphabetically
    """
    return list(_model_registry[model_source].keys())

def show_metadata(model_source, model_name):
    """ Return list of available model names, sorted alphabetically
    """
    return _model_registry[model_source][model_name]