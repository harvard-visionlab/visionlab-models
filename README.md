# visionlab-models

(IN DEVELOPMENT!)

PyTorch models instrumented for research.

The goal is to make it easy to extract activations from model layers, to visualize attention-maps (e.g., for vision transformers), and to visualize receptive fields.

Models are organized into "families" that might be interesting for research questions. The same model may appear in multiple families (check the hashid to be sure). Some models are Private (yet-to-be-made-public), and attempting to download them without the necessary credentials will throw an "Unauthorized" error.

Any models that are ported (copied + instrumented) from official sources are unofficial implementations (with possible errors arising from the porting process). If you notice any discrepancies between our copies and original implementations, please submit an issue.

# list hub methods
```
    import torch
    
    torch.hub.list("harvard-visionlab/visionlab_models")
    
```

# list model families
```
    import torch
    
    models = torch.hub.load("harvard-visionlab/visionlab_models", 'visionlab_models', trust_repo=True, force_reload=True)
    models.list_families()
    
```

# list models within a family
```
    models.list_models('alexnet_pytorch')
```

# list metadata about a specific model
```
    models.show_metadata('alexnet_pytorch', 'alexnet_7be5be79')
```

# load a model
```
    model, transforms = models.load_model('alexnet_pytorch', 'alexnet_7be5be79')
```

# preprocess an image with test_transforms
```
transform = transforms['test_transforms']
inv_transform = transforms['to_pil']
url = 'https://www.dropbox.com/s/n8nr36vox86t25u/example.png?dl=1'
img = transform(url)
inv_transform(img) # visualize as PIL Image
```

# get model outputs (standards)
```
model.eval()
with torch.no_grad():
    out = model(img.unsqueeze(0))
out.shape   
```

# models to add
- [ ] debiased
- [ ] edgenets
- [ ] ijepa models
- [ ] iwm models
- [ ] ...
