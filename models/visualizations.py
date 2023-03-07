import math
import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
from torchvision.utils import make_grid

import colorsys
import random
import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from pdb import set_trace

def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def display_instances(image, mask, fname=None, figsize=(5, 5), blur=False, contour=True, alpha=0.5, title=''):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    ax.set_title(title)
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    if fname is not None:
        fig.savefig(fname)
        print(f"{fname} saved.")
    return

def download_image(url="https://dl.fbaipublicfiles.com/dino/img.png"):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.convert('RGB')
    
    return img

def show_attention_maps_dino(img, model, transform, patch_size=8, threshold=.60):
    img = transform(img)

    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
    img = img[:, :w, :h].unsqueeze(0)

    model.eval()
    with torch.no_grad():
        attentions = model.get_last_selfattention(img)
    
    nh = attentions.shape[1] # number of head
    
    # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
    
    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size
    
    if threshold is not None:
        # we keep only a certain percentage of the mass
        val, idx = torch.sort(attentions)
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)
        th_attn = cumval > (1 - threshold)
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]
        th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
        # interpolate
        th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()
    
    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()
    
    image = Image.fromarray((torchvision.utils.make_grid(batch[0].permute(1,2,0), normalize=True, scale_each=True)*255).numpy().astype(np.uint8))
    
    plt.imshow(image)
    plt.show()
    plt.imshow(attentions.mean(axis=0), extent=[0, 1, 0, 1])
    
    for j in range(nh):
        display_instances(np.array(image), th_attn[j], blur=False)
    
def show_attention_maps(batch, attentions, idx=0, global_pool=None, num_tokens=0, 
                        patch_size=8, max_cols=6, threshold=None, interpolation='nearest'):
    '''
        batch: a tensor image (CxHxW) or batch (BSxCxHxW)
        attentions: the attention maps from a SINGLE attention block
        idx: index into batch dimension, used to get image from batch and corresponding attention maps
        
        Usage:
        
        model,transform = dino or hacked-timm-vit
        img = download_image()
        batch = transform(img).unsqueeze(0)
        
        attns = get_attention_maps(model, batch)
        
        # show maps of last attention block
        show_attention_maps(batch, maps[-1], patch_size=16, threshold=0.8)
        
        # show maps of first attention block
        show_attention_maps(batch, maps[0], patch_size=16, threshold=0.8)
    
    '''
    
    if len(batch.shape) == 4:
        img = batch[idx]
    else:
        img = batch
        
    nh = attentions.shape[1] # number of heads
    num_rows = math.ceil(nh/max_cols)
    
    if global_pool == 'avg':
        # [thisImage, allHeads, theseTokens]
        attentions = attentions[idx, :, num_tokens:].mean(dim=1)[:,num_tokens:]
    else:
        # we keep only the output patch attention
        # [thisImage, allHeads, everything after the class token => i.e., the spatial patches]
        attentions = attentions[idx, :, 0, 1:].reshape(nh, -1)
    
    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size
        
    if threshold is not None:
        # we keep only a certain percentage of the mass
        val, idx = torch.sort(attentions)
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)
        th_attn = cumval > (1 - threshold)
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]
        th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
        # interpolate
        th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=patch_size, mode=interpolation)[0].cpu().numpy()
    
    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode=interpolation)[0].cpu().numpy()
    
    image = Image.fromarray((torchvision.utils.make_grid(img.permute(1,2,0), normalize=True, scale_each=True)*255).numpy().astype(np.uint8))
    
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(8,4))    
    
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title('original image')
    #plt.show()
    
    ax2.imshow(attentions.mean(axis=0), extent=[0, 1, 0, 1])
    ax2.axis('off')
    ax2.set_title('attn map (avg. over heads)')
    #plt.show()
    
    if threshold is not None:
        for j in range(nh):    
            display_instances(np.array(image), th_attn[j], blur=False, title=f'map{j}')
    else:
        fig, axes = plt.subplots(num_rows,max_cols,figsize=(3*max_cols,3))
        if num_rows==1: axes = [axes]
        j = 0
        for row in axes:
            for ax in row:
                if (j+1) >= nh:
                    fig.delaxes(ax)
                else:
                    ax.imshow(attentions[j], extent=[0, 1, 0, 1], aspect='equal')
                    ax.set_title(f'map{j}')
                    ax.axis('off')
                j+=1
                
def show_conv1(model, nrow=16):  
    # find first conv
    first_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            first_conv = m
            break
    
    if first_conv is not None:
        kernels = first_conv.weight.detach().clone().cpu()
        kernels = kernels - kernels.min()
        kernels = kernels / kernels.max()
        img = make_grid(kernels, nrow=nrow)
        plt.imshow(img.permute(1, 2, 0))
    else:
        print("failed to find first conv layer")