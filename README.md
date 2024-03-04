# visionlab_models

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
    
    torch.hub.load("harvard-visionlab/visionlab_models", "list_families")
    
```

# list model families
```
    import torch
    
    torch.hub.list("harvard-visionlab/visionlab_models")
    
```