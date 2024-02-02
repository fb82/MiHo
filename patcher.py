#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torchvision.transforms as transforms

from PIL import Image
import scipy.io as sio


def save_patch(patch, grid=[40, 50], save_prefix='patch_', save_suffix='.png'):

    grid_el = grid[0] * grid[1]
    l = patch.size()[0]
    n = patch.size()[1]
    m = patch.size()[2]
    for i in range(0, l, grid_el):
        j = min(i+ grid_el, l)
        filename = f'{save_prefix}{i}_{j}{save_suffix}' 
        patch_ = patch[i:j]
        aux = torch.zeros((grid_el, n, m), dtype=torch.float16)
        aux[:j-i] = patch_
        
        print("doh!")
        
    return 0

def patchify(img, pts, H, r):

    wi = torch.arange(-r,r+1).unsqueeze(0)
    ws = r * 2 + 1
    n = pts.size()[0]
    _, y_sz, x_sz = img.size()
    
    x, y = pts.split(1, dim=1)
    
    widx = torch.full((n, 3, ws**2), 1, dtype=torch.float)
    
    widx[:, 0, :] = (wi + x).repeat(1,ws)
    widx[:, 1, :] = (wi + y).repeat_interleave(ws, dim=1)

    nidx = torch.matmul(H, widx)
    xx, yy, zz = nidx.split(1, dim=1)
    zz_ = zz.squeeze()
    xx_ = xx.squeeze() / zz_
    yy_ = yy.squeeze() / zz_
    
    xf = xx_.floor().type(torch.int)
    yf = yy_.floor().type(torch.int)
    xc = xf + 1
    yc = yf + 1

    nidx_mask = torch.logical_or(torch.logical_not(torch.isfinite(xx_)), torch.logical_not(torch.isfinite(yy_))) | torch.logical_or(xf < 0, yf < 0) | torch.logical_or(xc >= x_sz, yc >= y_sz)

    xf[nidx_mask] = 0
    yf[nidx_mask] = 0
    xc[nidx_mask] = 0
    yc[nidx_mask] = 0

    # for mask
    img_ = img.flatten()
    aux = img_[0]
    img_[0] = float('nan')

    a = xx_-xf    
    b = yy_-yf
    c = xc-xx_    
    d = yc-yy_

    patch = (a * (b * img_[yc * x_sz + xc] + d * img_[yf * x_sz + xc]) + c * (b * img_[yc * x_sz + xf] + d * img_[yf * x_sz + xf])).reshape((-1, ws, ws))
    img_[0] = aux

    return patch

    
if __name__ == '__main__':

    img1 = 'data/im1.png'
    img2 = 'data/im2_rot.png'
    match_file = 'data/matches_rot.mat'

    m12 = sio.loadmat(match_file, squeeze_me=True)
    m12 = m12['matches'][m12['midx'] > 0, :]

    im1 = Image.open(img1)
    im2 = Image.open(img2)
    
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.PILToTensor() 
        ]) 
        
    img1 = transform(im1).type(torch.float16)
    img2 = transform(im2).type(torch.float16)
    
    pt1, pt2 = torch.tensor(m12).round().type(torch.int).split(2, dim=1)
    
    patch1 = patchify(img1, pt1, torch.eye(3).unsqueeze(0), 15)   
    patch2 = patchify(img2, pt2, torch.eye(3).unsqueeze(0), 15)  
    
    save_patch(patch1, save_prefix='patch_', save_suffix='_a.png')
    save_patch(patch2, save_prefix='patch_', save_suffix='_b.png')

    print("doh!")
    
