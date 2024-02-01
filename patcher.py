#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torchvision.transforms as transforms

from PIL import Image
import scipy.io as sio


def patchify(img, pts, H, r):

    wi = torch.arange(-r,r+1).unsqueeze(0)
    ws = r * 2 + 1
    n = pts.size()[0]
    _, y_sz, x_sz = img.size()

    # for mask
    img[0, 0] = float('nan')
    
    x, y = pts.split(1, dim=1)
    
    widx = torch.full((n, 3, ws**2), 1, dtype=torch.float)
    
    widx[:, 0, :] = (wi + x).repeat(1,ws)
    widx[:, 1, :] = (wi + y).repeat_interleave(ws, dim=1)

    nidx = torch.matmul(H, widx)
    xx, yy, zz = nidx.split(1, dim=1)
    zz_ = zz.squeeze()
    xx_ = xx.squeeze().div(zz_)
    yy_ = yy.squeeze().div(zz_)
    
    xf = xx_.floor()
    yf = yy_.floor()
    xc = xx_.ceil()
    yc = yy_.ceil()

    nidx_mask = torch.logical_or(torch.logical_not(torch.isfinite(xx_)), torch.logical_not(torch.isfinite(yy_))) | torch.logical_or(xf < 0, yf < 0) | torch.logical_or(xc >= x_sz, yc >= y_sz)
    xx_[nidx_mask] = 0
    yy_[nidx_mask] = 0
    
    print("doh!")


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
    
    patchify(img1, pt1, torch.eye(3).unsqueeze(0), 15)   
    
    print("doh!")
    
