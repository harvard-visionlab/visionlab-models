from PIL import Image 
from enum import Enum
from typing import List, Tuple, Any, Optional
from collections import OrderedDict
from io import BytesIO
import requests

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_pil_interpolation_to_str = {
    Image.NEAREST: 'nearest',
    Image.BILINEAR: 'bilinear',
    Image.BICUBIC: 'bicubic',
    Image.BOX: 'box',
    Image.HAMMING: 'hamming',
    Image.LANCZOS: 'lanczos',
}

_str_to_pil_interpolation = {b: a for a, b in _pil_interpolation_to_str.items()}

class InterpolationMode(Enum):
    """Interpolation modes
    Available interpolation methods are ``nearest``, ``bilinear``, ``bicubic``, ``box``, ``hamming``, and ``lanczos``.
    """
    NEAREST = "nearest"
    NEAREST_EXACT = "nearest-exact"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    # For PIL compatibility
    BOX = "box"
    HAMMING = "hamming"
    LANCZOS = "lanczos"
    
def _interpolation_modes_from_int(i: int) -> InterpolationMode:
    inverse_modes_mapping = {
        0: InterpolationMode.NEAREST,
        2: InterpolationMode.BILINEAR,
        3: InterpolationMode.BICUBIC,
        4: InterpolationMode.BOX,
        5: InterpolationMode.HAMMING,
        1: InterpolationMode.LANCZOS,
    }
    return inverse_modes_mapping[i]
    
class OpenImage(object):
    def __call__(self, img):
        if isinstance(img, Image.Image):
            return img
        elif isinstance(img, str) and img.startswith('http'):
            return Image.open(BytesIO(requests.get(img).content)).convert('RGB')
        elif isinstance(img, str):
            return Image.open(img).convert('RGB')
    def __repr__(self):
        return f'{self.__class__.__name__}()' 
    
