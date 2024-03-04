import torch 
from torchvision import transforms

class InverseNormalize(object):
    '''inverse transforms a tensor, then converts to PIL image to show it'''
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        
        self.mean = torch.as_tensor(mean)
        self.std = torch.as_tensor(std)
        std_inv = 1 / (self.std + 1e-7)
        mean_inv = -self.mean * std_inv
        self.transform = transforms.Normalize(mean=mean_inv, std=std_inv)
            
    def __call__(self, img, *kwargs):
        img = self.transform(img.clone())
        return img
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"