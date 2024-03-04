import os
import torch

default_model_dir = os.path.join(torch.hub.get_dir(), "checkpoints")

def load_weights(model, url, cache_filename=None, model_dir=default_model_dir, verbose=True):   
    
    if verbose:
        print(f"==> loading checkpoint: {url}")

    checkpoint = torch.hub.load_state_dict_from_url(
        url = url,
        model_dir = model_dir,
        map_location = 'cpu',
        progress = True,
        check_hash = True,
        file_name = cache_filename
    )
    
    # get state_dict from checkpoint, usually conforms to one of the following:
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # drop the "DistributedDataParallel" module prefix
    state_dict = {str.replace(k,'module.',''): v for k,v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=True)
    
    r = torch.hub.HASH_REGEX.search(cache_filename)  # r is Optional[Match[str]]
    model.hashid = r.group(1) if r else None
    model.weights_file = os.path.join(model_dir, cache_filename)

    if verbose:
        print(f"==> state loaded: {msg}")
        print(f"==> cache_filename: {cache_filename}")

    return model 