from PIL import Image
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
# import scipy.io as sio
import torch
import torchvision.transforms as transforms
import kornia as K
import pydegensac
import math

import OANet.learnedmatcher_mod as OANet_matcher

# from pprint import pprint
# import deep_image_matching as dim
# import yaml

from src.pipelines.keynetaffnethardnet_module_fabio import keynetaffnethardnet_module_fabio
from src.pipelines.keynetaffnethardnet_kornia_matcher_module import keynetaffnethardnet_kornia_matcher_module
from src.pipelines.superpoint_lightglue_module import superpoint_lightglue_module
from src.pipelines.superpoint_kornia_matcher_module import superpoint_kornia_matcher_module
from src.pipelines.disk_lightglue_module import disk_lightglue_module
from src.pipelines.aliked_lightglue_module import aliked_lightglue_module
from src.pipelines.loftr_module import loftr_module

cv2.ocl.setUseOpenCL(False)
matplotlib.use('tkagg')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPS_ = torch.finfo(torch.float32).eps
sqrt2 = np.sqrt(2)


def laf2homo(kps):
    c = kps[:, :, 2]
    s = torch.sqrt(torch.abs(kps[:, 0, 0] * kps[:, 1, 1] - kps[:, 0, 1] * kps[:, 1, 0]))   
    
    Hi = torch.zeros((kps.shape[0], 3, 3), device=device)
    Hi[:, :2, :] = kps / s.reshape(-1, 1, 1)
    Hi[:, 2, 2] = 1 

    H = torch.linalg.inv(Hi)
    
    return c, H, s


def refinement_laf(im1, im2, pt1=None, pt2=None, data1=None, data2=None, w=15, img_patches=True):
    if data1 is None:
        l = pt1.shape[0]
        Hs = torch.eye(3, device=device).repeat(l*2, 1).reshape(l, 2, 3, 3)
    else:
        l = data1.shape[0]
        pt1, H1, s1 = laf2homo(data1)
        pt2, H2, s2 = laf2homo(data2)

        s = torch.sqrt(s1 * s2)       
        H1[:, :2, :] = H1[:, :2, :] * (s / s1).reshape(-1, 1, 1)
        H2[:, :2, :] = H2[:, :2, :] * (s / s2).reshape(-1, 1, 1)

        Hs = torch.cat((H1.unsqueeze(1), H2.unsqueeze(1)), 1)        
    
    if img_patches:
        go_save_patches(im1, im2, pt1, pt2, Hs, w, save_prefix='laf_patch_')    
  
    return pt1, pt2, Hs


def get_inverse(pt1, pt2, Hs):
    l = Hs.size()[0] 
    Hs1, Hs2 = Hs.split(1, dim=1)
    Hs1 = Hs1.squeeze(1)
    Hs2 = Hs2.squeeze(1)
            
    pt1_ = Hs1.bmm(torch.hstack((pt1, torch.ones((pt1.size()[0], 1), device=device))).unsqueeze(-1)).squeeze(-1)
    pt1_ = pt1_[:, :2] / pt1_[:, 2].unsqueeze(-1)
    pt2_ = Hs2.bmm(torch.hstack((pt2, torch.ones((pt2.size()[0], 1), device=device))).unsqueeze(-1)).squeeze(-1)
    pt2_ = pt2_[:, :2] / pt2_[:, 2].unsqueeze(-1)
    
    Hi = torch.linalg.inv(Hs.reshape(l*2, 3, 3)).reshape(l, 2, 3, 3)    
    Hi1, Hi2 = Hi.split(1, dim=1)
    Hi1 = Hi1.squeeze(1)
    Hi2 = Hi2.squeeze(1)
    
    return pt1_, pt2_, Hi, Hi1, Hi2


def refinement_norm_corr(im1, im2, pt1, pt2, Hs, w=15, ref_image=['left', 'right'], subpix=True, img_patches=False, save_prefix='ncc_patch_'):    
    l = Hs.size()[0] 
    
    if l==0:
        return pt1, pt2, Hs, torch.zeros(0, device=device), torch.zeros((0, 3, 3), device=device)
            
    pt1_, pt2_, Hi, Hi1, Hi2 = get_inverse(pt1, pt2, Hs)    
                
    patch1 = patchify(im1, pt1_.squeeze(), Hi1, w*2)
    patch2 = patchify(im2, pt2_.squeeze(), Hi2, w*2)

    uidx = torch.arange(w, 3*w + 1)
    tmp = uidx.unsqueeze(0).repeat(w*2 + 1, 1)
    vidx = tmp.permute(1, 0) * (w*4 + 1) + tmp
        
    patch_val = torch.full((2, l), -1, device=device, dtype=torch.float)
    patch_offset = torch.zeros(2, l, 2, device=device)

    if ('left' in ref_image) or ('both' in ref_image):
        patch_offset0, patch_val0 = norm_corr(patch2, patch1.reshape(l, -1)[:, vidx.flatten()].reshape(l, w*2 + 1, w*2 + 1), subpix=subpix)
        patch_offset[0] = patch_offset0
        patch_val[0] = patch_val0

    if ('right' in ref_image) or ('both' in ref_image):
        patch_offset1, patch_val1 = norm_corr(patch1, patch2.reshape(l, -1)[:, vidx.flatten()].reshape(l, w*2 + 1, w*2 + 1), subpix=subpix)
        patch_offset[1] = patch_offset1
        patch_val[1] = patch_val1
        
    val, val_idx = patch_val.max(dim=0)
    
    zidx = (torch.arange(l, device=device) * 4 + (1 - val_idx) * 2).unsqueeze(1).repeat(1,2) + torch.tensor([0, 1], device=device)
    patch_offset = patch_offset.permute(1, 0, 2).flatten()
    patch_offset[zidx.flatten()] = 0
    patch_offset = patch_offset.reshape(l, 2, 2)
    
    pt1_ = pt1_ - patch_offset[:, 0]
    pt2_ = pt2_ - patch_offset[:, 1]

    pt1 = Hi1.bmm(torch.hstack((pt1_, torch.ones((pt1_.size()[0], 1), device=device))).unsqueeze(-1)).squeeze()
    pt1 = pt1[:, :2] / pt1[:, 2].unsqueeze(-1)
    pt2 = Hi2.bmm(torch.hstack((pt2_, torch.ones((pt2_.size()[0], 1), device=device))).unsqueeze(-1)).squeeze()
    pt2 = pt2[:, :2] / pt2[:, 2].unsqueeze(-1)
    
    T = torch.eye(3, device=device, dtype=torch.float).reshape(1, 1, 9).repeat(l, 2, 1).reshape(l*2, 9)
    aux = patch_offset.reshape(l*2, 2)
    T[:, 2] = -aux[:, 0]
    T[:, 5] = -aux[:, 1]
    T = T.reshape(l*2, 3, 3)
    
    if img_patches:
        go_save_patches(im1, im2, pt1, pt2, Hs, w, save_prefix=save_prefix)
        
    return pt1, pt2, Hs, val, T


def refinement_norm_corr_alternate(im1, im2, pt1, pt2, Hs, w=15, w_big=None, ref_image=['left', 'right'], angle=[0, ], scale=[[1, 1], ], subpix=True, img_patches=False,  save_prefix='ncc_alternate_patch_'):    
    l = Hs.size()[0] 
    
    if l==0:
        return pt1, pt2, Hs, torch.zeros(0, device=device), torch.zeros((0, 3, 3), device=device)
        
    if w_big is None:
        w_big = w * 2
        
    pt1_, pt2_, Hi, Hi1, Hi2 = get_inverse(pt1, pt2, Hs)    
                        
    patch_val = torch.full((2, l), -torch.inf, device=device, dtype=torch.float)
    patch_offset = torch.zeros(2, l, 2, device=device)
    patch_t = torch.eye(3, device=device, dtype=torch.float).reshape(1, 1, 9).repeat(l, 2, 1).reshape(l, 2, 3, 3)

    a = torch.tensor(angle, device=device) * np.pi / 180
    s = torch.tensor(scale, device=device)

    Ti = torch.eye(3, device=device, dtype=torch.float).reshape(1, 1, 9).repeat(l, 2, 1).reshape(l, 2, 3, 3)
    Ti[:, 0, 0, 2] = pt1_[:, 0]
    Ti[:, 0, 1, 2] = pt1_[:, 1]
    Ti[:, 1, 0, 2] = pt2_[:, 0]
    Ti[:, 1, 1, 2] = pt2_[:, 1]

    T = torch.eye(3, device=device, dtype=torch.float).reshape(1, 1, 9).repeat(l, 2, 1).reshape(l, 2, 3, 3)
    T[:, 0, 0, 2] = -pt1_[:, 0]
    T[:, 0, 1, 2] = -pt1_[:, 1]
    T[:, 1, 0, 2] = -pt2_[:, 0]
    T[:, 1, 1, 2] = -pt2_[:, 1]

    for i in torch.arange(a.shape[0]):
        for j in torch.arange(s.shape[0]):
            R = torch.eye(3, device=device)
            R[0, 0] = torch.cos(a[i])
            R[0, 1] = -torch.sin(a[i])
            R[1, 0] = torch.sin(a[i])
            R[1, 1] = torch.cos(a[i])

            S = torch.eye(3, device=device)
            S[0, 0] = s[j, 0]
            S[1, 1] = s[j, 1]
            
            _, _, Hiu, Hi1u, Hi2u = get_inverse(pt1, pt2, Ti @ S @ R @ T @ Hs)    

            if ('left' in ref_image) or ('both' in ref_image):
                patch2 = patchify(im2, pt2_, Hi2, w_big)
                patch1_small = patchify(im1, pt1_, Hi1u, w)
        
                patch_offset0, patch_val0 = norm_corr(patch2, patch1_small, subpix=subpix)

                mask = patch_val0 > patch_val[0]                
                patch_offset[0, mask] = patch_offset0[mask]
                patch_val[0, mask] = patch_val0[mask]
                patch_t[mask, 0] = Hi1u[mask]                
        
            if ('right' in ref_image) or ('both' in ref_image):
                patch1 = patchify(im1, pt1_, Hi1, w_big)
                patch2_small = patchify(im2, pt2_, Hi2u, w)  
                
                patch_offset1, patch_val1 = norm_corr(patch1, patch2_small, subpix=subpix)
                
                mask = patch_val1 > patch_val[1]                
                patch_offset[1, mask] = patch_offset1[mask]
                patch_val[1, mask] = patch_val1[mask]
                patch_t[mask, 1] = Hi2u[mask]
        
    val, val_idx = patch_val.max(dim=0)
    
    zidx = (torch.arange(l, device=device) * 4 + (1 - val_idx) * 2).unsqueeze(1).repeat(1,2) + torch.tensor([0, 1], device=device)
    patch_offset = patch_offset.permute(1, 0, 2).flatten()
    patch_offset[zidx.flatten()] = 0
    patch_offset = patch_offset.reshape(l, 2, 2)

    pt1_ = pt1_ - patch_offset[:, 0]
    pt2_ = pt2_ - patch_offset[:, 1]

    Hiu = Hi
    valid_idx = val >= 0
    Hiu[(val_idx==0) & valid_idx, 0] = patch_t[(val_idx==0) & valid_idx, 0] 
    Hiu[(val_idx==1) & valid_idx, 1] = patch_t[(val_idx==1) & valid_idx, 1] 
    
    Hsu = torch.linalg.inv(Hiu.reshape(l*2, 3, 3)).reshape(l, 2, 3, 3)        
    
    pt1 = Hiu[:, 0].bmm(torch.hstack((pt1_, torch.ones((pt1_.size()[0], 1), device=device))).unsqueeze(-1)).squeeze(-1)
    pt1 = pt1[:, :2] / pt1[:, 2].unsqueeze(-1)
    pt2 = Hiu[:, 1].bmm(torch.hstack((pt2_, torch.ones((pt2_.size()[0], 1), device=device))).unsqueeze(-1)).squeeze(-1)
    pt2 = pt2[:, :2] / pt2[:, 2].unsqueeze(-1)
    
    T = torch.eye(3, device=device, dtype=torch.float).reshape(1, 1, 9).repeat(l, 2, 1).reshape(l*2, 9)
    aux = patch_offset.reshape(l*2, 2)
    T[:, 2] = -aux[:, 0]
    T[:, 5] = -aux[:, 1]
    T = T.reshape(l*2, 3, 3)
    
    if img_patches:
        go_save_patches(im1, im2, pt1, pt2, Hsu, w, save_prefix=save_prefix)
        
    return pt1, pt2, Hsu, val, T


def go_save_patches(im1, im2, pt1, pt2, Hs, w, save_prefix='patch_'):        
    pt1_, pt2_, _, Hi1, Hi2 = get_inverse(pt1, pt2, Hs) 
            
    patch1 = patchify(im1, pt1_, Hi1, w)
    patch2 = patchify(im2, pt2_, Hi2, w)

    save_patch(patch1, save_prefix=save_prefix, save_suffix='_a.png')
    save_patch(patch2, save_prefix=save_prefix, save_suffix='_b.png')


def refinement_miho(im1, im2, pt1, pt2, mihoo=None, Hs_laf=None, remove_bad=True, w=15, img_patches=False):
    l = pt1.shape[0]
    idx = torch.ones(l, dtype=torch.bool, device=device)

    if mihoo is None:
        if Hs_laf is not None:
            return pt1, pt2, Hs_laf, idx
        else:
            Hs = torch.eye(3, device=device).repeat(l*2, 1).reshape(l, 2, 3, 3)
            return pt1, pt2, Hs, idx

    Hs = torch.zeros((l, 2, 3, 3), device=device)
    for i in range(l):
        ii = mihoo.Hidx[i]
        if ii > -1:
            Hs[i, 0] = mihoo.Hs[ii][0]
            Hs[i, 1] = mihoo.Hs[ii][1]
        elif Hs_laf is not None:
            Hs[i, 0] = Hs_laf[i, 0]
            Hs[i, 1] = Hs_laf[i, 1]
        else:
            Hs[i, 0] = torch.eye(3, device=device)
            Hs[i, 1] = torch.eye(3, device=device)
            
    if remove_bad:
        mask = mihoo.Hidx > -1
        pt1 = pt1[mask]
        pt2 = pt2[mask]
        Hs = Hs[mask]
        idx = mask
        
    if img_patches:
        go_save_patches(im1, im2, pt1, pt2, Hs, w, save_prefix='miho_patch_')
    
    return pt1, pt2, Hs, idx


def norm_corr(patch1, patch2, subpix=True):     
    w = patch2.size()[1]
    ww = w * w
    n = patch1.size()[0]
    
    with torch.no_grad():
        conv_ = torch.nn.Conv2d(1, 1, (w, w), padding='valid', bias=False, device=device)
        conv_.weight = torch.nn.Parameter(torch.ones((1, 1, w, w), device=device).to(torch.float))    
        m1 = conv_(patch1.unsqueeze(1)).squeeze()
        e1 = conv_((patch1**2).unsqueeze(1)).squeeze()
        s1 = ww * e1 - m1**2
    
    m2 = patch2.sum(dim=[1, 2])
    e2 = (patch2**2).sum(dim=[1, 2])    
    s2 = ww * e2 - m2**2
    
    with torch.no_grad():
        conv__ = torch.nn.Conv2d(n, n, (w, w), padding='valid', bias=False, groups=n, device=device)
        conv__.weight = torch.nn.Parameter(patch2.unsqueeze(1))    
        cc = conv__(patch1.unsqueeze(0)).squeeze()

    nc = ((ww * cc) - (m1 * m2.reshape(n, 1, 1))) / torch.sqrt(s1 * s2.reshape(n, 1, 1))   
    nc.flatten()[~torch.isfinite(nc.flatten())] = -torch.inf

    w_ = nc.shape[1]
    ww_ = w_ * w_    
    r = (w_ - 1) / 2

    idx = nc.reshape(n, ww_).max(dim=1)
    offset = (torch.vstack((idx[1] % w_, torch.div(idx[1], w_, rounding_mode='trunc')))).permute(1, 0).to(torch.float)
    
    if subpix:    
        t = ((offset > 0) & ( offset < w_ - 1)).all(dim=1).to(torch.float)
        tidx = (torch.tensor([-1, 0, 1], device=device).unsqueeze(0) * t.unsqueeze(1)).squeeze()
    
        tx = offset[:, 0].unsqueeze(1) + tidx
        v = nc.flatten()[(torch.arange(n, device=device).unsqueeze(1) * ww_ + offset[:, 1].unsqueeze(1) * w_ + tx).to(torch.long).flatten()].reshape(n, 3)
        sx = (v[:, 2] - v[:, 0]) / (2 * (2 * v[:, 1] - v[:, 0] - v[:, 2]))
        sx[~sx.isfinite()] = 0
    
        ty = offset[:, 1].unsqueeze(1) + tidx
        v = nc.flatten()[(torch.arange(n, device=device).unsqueeze(1) * ww_ + ty * w_ + offset[:, 0].unsqueeze(1)).to(torch.long).flatten()].reshape(n, 3)
        sy = (v[:, 2] - v[:, 0]) / (2 * (2 * v[:, 1] - v[:, 0] - v[:, 2]))
        sy[~sy.isfinite()] = 0
        
        offset[:, 0] = offset[:, 0] + sx
        offset[:, 1] = offset[:, 1] + sy

    offset -= r
    offset[~torch.isfinite(idx[0])] = 0

    return offset, idx[0]


def save_patch(patch, grid=[40, 50], save_prefix='patch_', save_suffix='.png', normalize=False):

    grid_el = grid[0] * grid[1]
    l = patch.size()[0]
    n = patch.size()[1]
    m = patch.size()[2]
    transform = transforms.ToPILImage()
    for i in range(0, l, grid_el):
        j = min(i+ grid_el, l)
        filename = f'{save_prefix}{i}_{j}{save_suffix}' 
        
        patch_ = patch[i:j]
        aux = torch.zeros((grid_el, n, m), dtype=torch.float32, device=device)
        aux[:j-i] = patch_
        
        mask = aux.isfinite()
        aux[~mask] = 0
        
        if not normalize:
            aux = aux.type(torch.uint8)
        else:
            aux[~mask] = -1        
            avg = ((mask * aux).sum(dim=(1,2)) / mask.sum(dim=(1,2))).reshape(-1, 1, 1).repeat(1, n, m)
            avg[mask] = aux[mask]
            m_ = avg.reshape(grid_el, -1).min(dim=1)[0]
            M_ = avg.reshape(grid_el, -1).max(dim=1)[0]
            aux = (((aux - m_.reshape(-1, 1, 1)) / (M_ - m_).reshape(-1, 1, 1)) * 255).type(torch.uint8)
           
        # if not needed do not add alpha channel
        all_mask = mask.all()
        c = 1 + (3 * ~all_mask)
        aux = aux.reshape(grid[0], grid[1], n, m).permute(0, 2, 1, 3).reshape(grid[0] * n, grid[1] * m).contiguous().unsqueeze(0).repeat(c, 1, 1)
        if not all_mask:        
            aux[3, :, :] = (mask *255).type(torch.uint8).reshape(grid[0], grid[1], n, m).permute(0, 2, 1, 3).reshape(grid[0] * n, grid[1] * m).contiguous()
        transform(aux).save(filename)
        

def patchify(img, pts, H, r):

    wi = torch.arange(-r,r+1, device=device)
    ws = r * 2 + 1
    n = pts.size()[0]
    _, y_sz, x_sz = img.size()
    
    x, y = pts.split(1, dim=1)
    
    widx = torch.zeros((n, 3, ws**2), dtype=torch.float, device=device)
    
    widx[:, 0] = (wi + x).repeat(1,ws)
    widx[:, 1] = (wi + y).repeat_interleave(ws, dim=1)
    widx[:, 2] = 1

    nidx = torch.matmul(H, widx)
    xx, yy, zz = nidx.split(1, dim=1)
    zz_ = zz.squeeze()
    xx_ = xx.squeeze() / zz_
    yy_ = yy.squeeze() / zz_
    
    xf = xx_.floor().type(torch.long)
    yf = yy_.floor().type(torch.long)
    xc = xf + 1
    yc = yf + 1

    nidx_mask = ~torch.isfinite(xx_) | ~torch.isfinite(yy_) | (xf < 0) | (yf < 0) | (xc >= x_sz) | (yc >= y_sz)

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


def get_error_duplex(H12, pt1, pt2, ptm, sidx_par):
    l2 = sidx_par.size()[0]        
    n = pt1.size()[1]
    
    ptm_reproj = torch.cat((torch.matmul(H12[:l2], pt1), torch.matmul(H12[l2:], pt2)), dim=0)
    sign_ptm = torch.sign(ptm_reproj[:, 2])

    pt12_reproj = torch.linalg.solve(H12, ptm)
    sign_pt12 = torch.sign(pt12_reproj[:, 2])

    idx_aux = torch.arange(l2*2, device=device)*n + sidx_par[:, 0].repeat(2)

    sa = sign_ptm.flatten()[idx_aux.flatten()].reshape(idx_aux.size())
    sb = sign_pt12.flatten()[idx_aux.flatten()].reshape(idx_aux.size())
    
    ssa = sa.unsqueeze(1) == sign_ptm
    ssb = sb.unsqueeze(1) == sign_pt12

    mask = torch.logical_and(ssa, ssb)
    
    err_m = ptm_reproj[:, :2] / ptm_reproj[:, 2].unsqueeze(1) - ptm[:2]
    err_12 = torch.cat((pt12_reproj[:l2, :2] / pt12_reproj[:l2, 2].unsqueeze(1) - pt1[:2], pt12_reproj[l2:, :2] / pt12_reproj[l2:, 2].unsqueeze(1) - pt2[:2]), dim=0)

    err = torch.maximum(torch.sum(err_m ** 2, dim=1), torch.sum(err_12 ** 2, dim=1))
    err[torch.logical_or(~torch.isfinite(err), ~mask)] = float('inf')

    return torch.maximum(err[:l2], err[l2:])


def get_inlier_duplex(H12, pt1, pt2, ptm, sidx_par, th):
    l2 = sidx_par.size()[0]        
    n = pt1.size()[1]
    
    ptm_reproj = torch.cat((torch.matmul(H12[:l2], pt1), torch.matmul(H12[l2:], pt2)), dim=0)
    sign_ptm = torch.sign(ptm_reproj[:, 2])
    
    # CUDA crash on bad matrices! ##########################
    bad_matrix = ~(torch.isfinite(torch.linalg.cond(H12))) #
    H12[bad_matrix] = torch.eye(3, device=device)          #
    ########################################################
    pt12_reproj = torch.linalg.solve(H12, ptm.unsqueeze(0))
    # CUDA crash on bad matrices! ##
    pt12_reproj[bad_matrix, 2] = 0 #
    ################################
    sign_pt12 = torch.sign(pt12_reproj[:, 2])

    idx_aux = torch.arange(l2*2, device=device).unsqueeze(1)*n + sidx_par.repeat(2,1)

    sa = sign_ptm.flatten()[idx_aux.flatten()].reshape(idx_aux.size())
    sb = sign_pt12.flatten()[idx_aux.flatten()].reshape(idx_aux.size())
    
    ssa = torch.all(sa[:, 0].unsqueeze(1) == sa, dim=1)
    ssb = torch.all(sb[:, 0].unsqueeze(1) == sb, dim=1)

    ssa_ = sa[:, 0].unsqueeze(1) == sign_ptm
    ssb_ = sb[:, 0].unsqueeze(1) == sign_pt12

    mask = (ssa_ & ssb_) & (ssa & ssb).unsqueeze(1)
    
    err_m = ptm_reproj[:, :2] / ptm_reproj[:, 2].unsqueeze(1) - ptm[:2]
    err_12 = torch.cat((pt12_reproj[:l2, :2] / pt12_reproj[:l2, 2].unsqueeze(1) - pt1[:2], pt12_reproj[l2:, :2] / pt12_reproj[l2:, 2].unsqueeze(1) - pt2[:2]), dim=0)

    err = torch.maximum(torch.sum(err_m ** 2, dim=1), torch.sum(err_12 ** 2, dim=1))
    err[~torch.isfinite(err)] = float('inf')
    err_ = err < th

    final_mask = mask & err_
    return torch.logical_and(final_mask[:, :l2], final_mask[:, l2:]).squeeze(dim=0)


def compute_homography_duplex(pt1, pt2, ptm, sidx_par):
    
    if sidx_par.dtype != torch.bool:
        l0 = sidx_par.size()[0]
        l1 = sidx_par.size()[1]
        
        pt1_par = pt1[:, sidx_par.flatten()].reshape(3, l0, l1).permute(1, 0, 2)
        pt2_par = pt2[:, sidx_par.flatten()].reshape(3, l0, l1).permute(1, 0, 2)
        ptm_par = ptm[:, sidx_par.flatten()].reshape(3, l0, l1).permute(1, 0, 2)
    else:
        l0 = 1
        l1 = sidx_par.sum()
        
        pt1_par = pt1[:, sidx_par].reshape(3, l0, l1).permute(1, 0, 2)
        pt2_par = pt2[:, sidx_par].reshape(3, l0, l1).permute(1, 0, 2)
        ptm_par = ptm[:, sidx_par].reshape(3, l0, l1).permute(1, 0, 2)
        

    c1 = torch.mean(pt1_par[:, :2], dim=2)
    c2 = torch.mean(pt2_par[:, :2], dim=2)
    cm = torch.mean(ptm_par[:, :2], dim=2)

    norm_diff_1 = torch.sqrt(torch.sum((pt1_par[:, :2] - c1.unsqueeze(2))**2, dim=1))
    norm_diff_2 = torch.sqrt(torch.sum((pt2_par[:, :2] - c2.unsqueeze(2))**2, dim=1))
    norm_diff_m = torch.sqrt(torch.sum((ptm_par[:, :2] - cm.unsqueeze(2))**2, dim=1))

    s1 = sqrt2 / (torch.mean(norm_diff_1, dim=1) + EPS_)
    s2 = sqrt2 / (torch.mean(norm_diff_2, dim=1) + EPS_)
    sm = sqrt2 / (torch.mean(norm_diff_m, dim=1) + EPS_)

    T12 = torch.zeros((l0*2, 3, 3), dtype=torch.float32, device=device)
    T12[:l0, 0, 0] = s1
    T12[:l0, 1, 1] = s1        
    T12[:l0, 2, 2] = 1
    T12[:l0, 0, 2] = -c1[:, 0] * s1
    T12[:l0, 1, 2] = -c1[:, 1] * s1

    T12[l0:, 0, 0] = s2
    T12[l0:, 1, 1] = s2
    T12[l0:, 2, 2] = 1
    T12[l0:, 0, 2] = -c2[:, 0] * s2
    T12[l0:, 1, 2] = -c2[:, 1] * s2

    Tm = torch.zeros((l0*2, 3, 3), dtype=torch.float32, device=device)
    Tm[l0:, 0, 0] = 1/sm
    Tm[l0:, 1, 1] = 1/sm
    Tm[l0:, 2, 2] = 1
    Tm[l0:, 0, 2] = cm[:, 0]
    Tm[l0:, 1, 2] = cm[:, 1]

    Tm[:l0, 0, 0] = 1/sm
    Tm[:l0, 1, 1] = 1/sm
    Tm[:l0, 2, 2] = 1
    Tm[:l0, 0, 2] = cm[:, 0]
    Tm[:l0, 1, 2] = cm[:, 1]


    p1x = s1.unsqueeze(1) * (pt1_par[:, 0] - c1[:, 0].unsqueeze(1))
    p1y = s1.unsqueeze(1) * (pt1_par[:, 1] - c1[:, 1].unsqueeze(1))

    p2x = s2.unsqueeze(1) * (pt2_par[:, 0] - c2[:, 0].unsqueeze(1))
    p2y = s2.unsqueeze(1) * (pt2_par[:, 1] - c2[:, 1].unsqueeze(1))

    pmx = sm.unsqueeze(1) * (ptm_par[:, 0] - cm[:, 0].unsqueeze(1))
    pmy = sm.unsqueeze(1) * (ptm_par[:, 1] - cm[:, 1].unsqueeze(1))


    A = torch.zeros((l0*2, l1*3, 9), dtype=torch.float32, device=device)


    A[:l0, :l1, 3] = -p1x
    A[:l0, :l1, 4] = -p1y
    A[:l0, :l1, 5] = -1

    A[:l0, :l1, 6] = torch.mul(pmy, p1x)
    A[:l0, :l1, 7] = torch.mul(pmy, p1y)
    A[:l0, :l1, 8] = pmy

    A[:l0, l1:2*l1, 0] = p1x
    A[:l0, l1:2*l1, 1] = p1y
    A[:l0, l1:2*l1, 2] = 1

    A[:l0, l1:2*l1, 6] = -torch.mul(pmx, p1x)
    A[:l0, l1:2*l1, 7] = -torch.mul(pmx, p1y)
    A[:l0, l1:2*l1, 8] = -pmx

    A[:l0, 2*l1:, 0] = -torch.mul(pmy, p1x)
    A[:l0, 2*l1:, 1] = -torch.mul(pmy, p1y)
    A[:l0, 2*l1:, 2] = -pmy

    A[:l0, 2*l1:, 3] = torch.mul(pmx, p1x)
    A[:l0, 2*l1:, 4] = torch.mul(pmx, p1y)
    A[:l0, 2*l1:, 5] = pmx


    A[l0:, :l1, 3] = -p2x
    A[l0:, :l1, 4] = -p2y
    A[l0:, :l1, 5] = -1

    A[l0:, :l1, 6] = torch.mul(pmy, p2x)
    A[l0:, :l1, 7] = torch.mul(pmy, p2y)
    A[l0:, :l1, 8] = pmy

    A[l0:, l1:2*l1, 0] = p2x
    A[l0:, l1:2*l1, 1] = p2y
    A[l0:, l1:2*l1, 2] = 1

    A[l0:, l1:2*l1, 6] = -torch.mul(pmx, p2x)
    A[l0:, l1:2*l1, 7] = -torch.mul(pmx, p2y)
    A[l0:, l1:2*l1, 8] = -pmx

    A[l0:, 2*l1:, 0] = -torch.mul(pmy, p2x)
    A[l0:, 2*l1:, 1] = -torch.mul(pmy, p2y)
    A[l0:, 2*l1:, 2] = -pmy

    A[l0:, 2*l1:, 3] = torch.mul(pmx, p2x)
    A[l0:, 2*l1:, 4] = torch.mul(pmx, p2y)
    A[l0:, 2*l1:, 5] = pmx


    _, D, V = torch.linalg.svd(A, full_matrices=True)
    H12 = V[:, -1].reshape(l0*2, 3, 3).permute(0, 2, 1)
    H12 = Tm @ H12 @ T12

    # H12 = H12.reshape(2, l0 ,3, 3)
    sv = torch.amax(D[:, -2].reshape(2, l0), dim=0)
    
    return H12, sv


def compute_homography_unduplex(pts1, pts2, sidx_par):
    if sidx_par.dtype != torch.bool:
        l0 = sidx_par.size()[0]
        l1 = sidx_par.size()[1]
        
        pt1_par = pt1[:, sidx_par.flatten()].reshape(3, l0, l1).permute(1, 0, 2)
        pt2_par = pt2[:, sidx_par.flatten()].reshape(3, l0, l1).permute(1, 0, 2)
    else:
        l0 = 1
        l1 = sidx_par.sum()
        
        pt1_par = pt1[:, sidx_par].reshape(3, l0, l1).permute(1, 0, 2)
        pt2_par = pt2[:, sidx_par].reshape(3, l0, l1).permute(1, 0, 2)

    c1 = torch.mean(pt1_par[:, :2], dim=2)
    c2 = torch.mean(pt2_par[:, :2], dim=2)

    norm_diff_1 = torch.sqrt(torch.sum((pt1_par[:, :2] - c1.unsqueeze(2))**2, dim=1))
    norm_diff_2 = torch.sqrt(torch.sum((pt2_par[:, :2] - c2.unsqueeze(2))**2, dim=1))

    s1 = sqrt2 / (torch.mean(norm_diff_1, dim=1) + EPS_)
    s2 = sqrt2 / (torch.mean(norm_diff_2, dim=1) + EPS_)

    T1 = torch.zeros((l0, 3, 3), dtype=torch.float32, device=device)
    T1[:, 0, 0] = s1
    T1[:, 1, 1] = s1        
    T1[:, 2, 2] = 1
    T1[:, 0, 2] = -c1[:, 0] * s1
    T1[:, 1, 2] = -c1[:, 1] * s1

    T2 = torch.zeros((l0, 3, 3), dtype=torch.float32, device=device)
    T2[:, 0, 0] = 1/s2
    T2[:, 1, 1] = 1/s2
    T2[:, 2, 2] = 1
    T2[:, 0, 2] = c2[:, 0]
    T2[:, 1, 2] = c2[:, 1]

    p1x = s1.unsqueeze(1) * (pt1_par[:, 0] - c1[:, 0].unsqueeze(1))
    p1y = s1.unsqueeze(1) * (pt1_par[:, 1] - c1[:, 1].unsqueeze(1))

    p2x = s2.unsqueeze(1) * (pt2_par[:, 0] - c2[:, 0].unsqueeze(1))
    p2y = s2.unsqueeze(1) * (pt2_par[:, 1] - c2[:, 1].unsqueeze(1))

    A = torch.zeros((l0, l1*3, 9), dtype=torch.float32, device=device)

    A[:, :l1, 3] = -p1x
    A[:, :l1, 4] = -p1y
    A[:, :l1, 5] = -1

    A[:, :l1, 6] = torch.mul(p2y, p1x)
    A[:, :l1, 7] = torch.mul(p2y, p1y)
    A[:, :l1, 8] = p2y

    A[:, l1:2*l1, 0] = p1x
    A[:, l1:2*l1, 1] = p1y
    A[:, l1:2*l1, 2] = 1

    A[:, l1:2*l1, 6] = -torch.mul(p2x, p1x)
    A[:, l1:2*l1, 7] = -torch.mul(p2x, p1y)
    A[:, l1:2*l1, 8] = -p2x

    A[:, 2*l1:, 0] = -torch.mul(p2y, p1x)
    A[:, 2*l1:, 1] = -torch.mul(p2y, p1y)
    A[:, 2*l1:, 2] = -p2y

    A[:, 2*l1:, 3] = torch.mul(p2x, p1x)
    A[:, 2*l1:, 4] = torch.mul(p2x, p1y)
    A[:, 2*l1:, 5] = p2x

    _, D, V = torch.linalg.svd(A, full_matrices=True)
    H12 = V[:, -1].reshape(l0, 3, 3).permute(0, 2, 1)
    H12 = T2 @ H12 @ T1

    sv = torch.amax(D[:, -2].reshape(2, l0), dim=0)

    return H12, sv


def data_normalize(pts):
    c = torch.mean(pts, dim=1)
    norm_diff = torch.sqrt((pts[0] - c[0])**2 + (pts[1] - c[1])**2)
    s = torch.sqrt(torch.tensor(2.0)) / (torch.mean(norm_diff) + EPS_)

    T = torch.tensor([
        [s, 0, -c[0] * s],
        [0, s, -c[1] * s],
        [0, 0, 1]
    ], dtype=torch.float32, device=device)

    return T


def steps(pps, inl, p):
    e = 1 - inl
    r = torch.log(torch.tensor(1.0) - p) / torch.log(torch.tensor(1.0) - (torch.tensor(1.0) - e)**pps)
    return r


def compute_homography(pts1, pts2):
    T1 = data_normalize(pts1)
    T2 = data_normalize(pts2)

    npts1 = torch.matmul(T1, pts1)
    npts2 = torch.matmul(T2, pts2)

    l = npts1.shape[1]
    A = torch.zeros((l*3, 9), dtype=torch.float32, device=device)
    A[:l, 3:6] = -torch.mul(torch.tile(npts2[2], (3, 1)).t(), npts1.t())
    A[:l, 6:] = torch.mul(torch.tile(npts2[1], (3, 1)).t(), npts1.t())
    A[l:2*l, :3] = torch.mul(torch.tile(npts2[2], (3, 1)).t(), npts1.t())
    A[l:2*l, 6:] = -torch.mul(torch.tile(npts2[0], (3, 1)).t(), npts1.t())
    A[2*l:, :3] = -torch.mul(torch.tile(npts2[1], (3, 1)).t(), npts1.t())
    A[2*l:, 3:6] = torch.mul(torch.tile(npts2[0], (3, 1)).t(), npts1.t())

    try:
        _, D, V = torch.linalg.svd(A, full_matrices=True)
        H = V[-1, :].reshape(3, 3).T
        H = torch.inverse(T2) @ H @ T1
    except:
        H = None
        D = torch.zeros(9, dtype=torch.float32)

    return H, D


def get_inliers(pt1, pt2, H, ths, sidx):

    l = pt1.shape[1]
    ths_ = [ths] if not isinstance(ths, list) else ths
    m = len(ths_)

    pt2_ = torch.matmul(H, pt1)
    s2_ = torch.sign(pt2_[2])
    s2 = s2_[sidx[0]]

    if not torch.all(s2_[sidx] == s2):
        nidx = torch.zeros((m, l), dtype=torch.bool, device=device)
        if not isinstance(ths, list):
            nidx = nidx[0]
        return nidx

    tmp2_ = pt2_[:2] / pt2_[2] - pt2[:2]
    err2 = torch.sum(tmp2_**2, axis=0)

    pt1_ = torch.matmul(torch.inverse(H), pt2)
    s1_ = torch.sign(pt1_[2])
    s1 = s1_[sidx[0]]

    if not torch.all(s1_[sidx] == s1):
        nidx = torch.zeros((m, l), dtype=torch.bool, device=device)
        if not isinstance(ths, list):
            nidx = nidx[0]
        return nidx

    tmp1_ = pt1_[:2] / pt1_[2] - pt1[:2]
    err1 = torch.sum(tmp1_**2, axis=0)

    err = torch.maximum(err1, err2)
    err[~torch.isfinite(err)] = float('inf')

    nidx = torch.zeros((m, l), dtype=torch.bool, device=device)
    for i in range(m):
        nidx[i] = torch.all(torch.stack((err < ths_[i], s2_ == s2, s1_ == s1)), dim=0)

    if not isinstance(ths, list):
        nidx = nidx[0]

    return nidx


def get_error(pt1, pt2, H, sidx):
    l = pt1.shape[1]

    pt2_ = torch.matmul(H, pt1)
    s2_ = torch.sign(pt2_[2])
    s2 = s2_[sidx[0]]

    if not torch.all(s2_[sidx] == s2):
        return torch.full((l,), float('inf'))

    tmp2_ = pt2_[:2] / pt2_[2] - pt2[:2]
    err2 = torch.sum(tmp2_**2, dim=0)

    pt1_ = torch.matmul(torch.inverse(H), pt2)
    s1_ = torch.sign(pt1_[2])
    s1 = s1_[sidx[0]]

    if not torch.all(s1_[sidx] == s1):
        return torch.full((l,), float('inf'))

    tmp1_ = pt1_[:2] / pt1_[2] - pt1[:2]
    err1 = torch.sum(tmp1_**2, dim=0)

    err = torch.maximum(err1, err2)
    err[~torch.isfinite(err)] = float('inf')

    return err


def sampler4_par(n_par, m):
    nn = n_par.size()[0]

    n_par = n_par.repeat(m)

    sidx = (torch.rand((m * nn, 4), device=device) * torch.stack((n_par, n_par-1, n_par-2, n_par-3)).permute(1,0)).type(torch.long)  

    for k in range(1,4):
        sidx[:, 0:k] = torch.sort(sidx[:, 0:k])[0]
        for kk in range(k):
            sidx[:, k] = sidx[:, k] + (sidx[:, k] >= sidx[:, kk])

    return sidx.reshape(m, nn, 4)


def ransac_middle(pt1, pt2, dd=None, th_grid=15, th_in=7, th_out=15, max_iter=500, min_iter=50, p=0.9, svd_th=0.05, buffers=5, ssidx=None, par_value=100000):
    n = pt1.shape[1]

    th_in = th_in ** 2
    th_out = th_out ** 2

    ptm = (pt1 + pt2) / 2

    th = torch.tensor(th_out, device=device).reshape(1, 1, 1)
    ths = torch.tensor([th_in, th_out], device=device).reshape(2, 1, 1)

    if n < 4:
        H1 = torch.tensor([], device=device)
        H2 = torch.tensor([], device=device)
        iidx = torch.zeros(n, dtype=torch.bool, device=device)
        oidx = torch.zeros(n, dtype=torch.bool, device=device)
        vidx = torch.zeros((n, 0), dtype=torch.bool, device=device)
        sidx_ = torch.zeros((4,), dtype=torch.int32, device=device)
        return H1, H2, iidx, oidx, vidx, sidx_
    
    min_iter = min(min_iter, n*(n-1)*(n-2)*(n-3) // 12)

    vidx = torch.zeros((n, buffers), dtype=torch.bool, device=device)
    midx = torch.zeros((n, buffers+1), dtype=torch.bool, device=device)

    sum_midx = 0
    Nc = float('inf')
    min_th_stats = 3

    sn = ssidx.shape[1]
    sidx_ = torch.zeros((4,), dtype=torch.long, device=device)
    
    ssidx_sz = torch.sum(ssidx, dim=0)

    par_run = par_value // n

    c = 0
    best_model = torch.zeros((2, 3, 3), device=device)
    while c < max_iter:
        c_par = torch.arange(c, min(max_iter, c + par_run), device=device)
        c_par_sz = c_par.size()[0]

        n_par = torch.full((c_par_sz,), n, dtype=torch.int, device=device)
        for i in range(c_par_sz):
            if (c_par[i] < sn):
                if ssidx_sz[c_par[i]] > 4:
                    n_par[i] = ssidx_sz[c_par[i]]
            else:
                break

        sidx = sampler4_par(n_par, min_iter)

        for i in range(c_par_sz):
            if (c_par[i] < sn):
                if ssidx_sz[c_par[i]] > 4:
                    aux = sidx[:, c_par[i]].flatten()
                    tmp = torch.nonzero(ssidx[:, c_par[i]]).squeeze()
                    aux = tmp[aux]
                    sidx[:, c_par[i]] = aux.reshape(min_iter, 4)                
            else:
                break


        dr = sidx.repeat(1, 1, 4).flatten()
        dc = sidx.repeat_interleave(4, dim=2).flatten()
        
        dd_check = (((pt1[0, dr] - pt1[0, dc])**2 + (pt1[1, dr] - pt1[1, dc])**2 > th_grid**2) &
                    ((pt2[0, dr] - pt2[0, dc])**2 + (pt2[1, dr] - pt2[1, dc])**2 > th_grid**2)).reshape(min_iter, c_par_sz, -1)

        # dd_check = dd.flatten()[sidx.repeat((1, 1, 4)).flatten() * dd.size()[0] + sidx.repeat_interleave(4, dim=2).flatten()].reshape(min_iter, c_par_sz, -1)
        dd_good = torch.sum(dd_check, dim=-1) >= 12

        # good_sample_par = torch.zeros(c_par_sz, dtype=torch.bool, device=device)
        # sidx_par = torch.zeros((c_par_sz, 4), dtype=torch.long, device=device)
        #        
        # for i in range(c_par_sz):
        #     for j in range(min_iter):
        #         if dd_good[j, i]:
        #             good_sample_par[i] = True
        #             sidx_par[i, :] = sidx[j, i, :]
        #             break
                
        good_sample_par, good_idx = dd_good.max(dim=0)
        sidx_par = sidx.reshape(min_iter * c_par_sz, 4)[good_idx * c_par_sz + torch.arange(c_par_sz, device=device)]

        sidx_par = sidx_par[good_sample_par]
        c_par = c_par[good_sample_par]             
        
        H12, sv = compute_homography_duplex(pt1, pt2, ptm, sidx_par)
        good_H = sv > svd_th

        H12 = H12[good_H.repeat(2)]            
        sidx_par = sidx_par[good_H]
        c_par = c_par[good_H]
        
        if not c_par.size()[0]:
            if (c + par_run > Nc) and (c + par_run > min_iter):
                break
            else:
                continue
                
        nidx_par = get_inlier_duplex(H12, pt1, pt2, ptm, sidx_par, th)                        
        sum_nidx_par = nidx_par.sum(dim=1)
        l2 = sidx_par.size()[0]

        sum_nidx_par, sort_idx = torch.sort(sum_nidx_par, descending=True)
        nidx_par = nidx_par[sort_idx]
        sidx_par = sidx_par[sort_idx]

        for i in range(l2):
            sum_nidx = sum_nidx_par[i]

            nidx = nidx_par[i]
            
            sidx_i = sidx_par[i]

            H1 = H12[sort_idx[i]]
            H2 = H12[sort_idx[i] + l2]

            updated_model = False
    
            midx[:, -1] = nidx
    
            if sum_nidx > min_th_stats:
    
                idxs = torch.arange(buffers+1)
                q = torch.tensor(n+1)
    
                for t in range(buffers):
                    uidx = ~torch.any(midx[:, idxs[:t]], dim=1).unsqueeze(1)
    
                    tsum = uidx & midx[:, idxs[t:]]
                    ssum = torch.sum(tsum, dim=0)
                    vidx[:, t] = tsum[:, torch.argmax(ssum)]
    
                    tt = torch.argmax((ssum[-1] > ssum[:-1]).type(torch.long))
                    if ssum[-1] > ssum[tt]:
                        aux = idxs[-1].clone()
                        idxs[-1] = idxs[t+tt].clone()
                        idxs[t+tt] = aux
                        if t == 0 and tt == 0:
                            sidx_ = sidx_i
    
                    q = torch.minimum(q, torch.max(ssum))
    
                min_th_stats = torch.maximum(torch.tensor(4), q)
    
                updated_model = idxs[0] != 0
                midx = midx[:, idxs]
    
            if updated_model:
                sum_midx = torch.sum(midx[:, 0])
                best_model[0] = H1
                best_model[1] = H2
                Nc = steps(4, sum_midx / n, p)
    
        if (c + par_run > Nc) and (c + par_run > min_iter):
            break
            
        c += par_run

    vidx = vidx[:, 1:]

    if sum_midx >= 4:
        bidx = midx[:, 0]

        H12, _ = compute_homography_duplex(pt1, pt2, ptm, bidx)
        H1, H2 = H12

        iidx, oidx = get_inlier_duplex(H12, pt1, pt2, ptm, sidx_.unsqueeze(0), ths)                        

        if sum_midx > torch.sum(oidx):
            H1, H2 = best_model

            iidx, oidx = get_inlier_duplex(best_model, pt1, pt2, ptm, sidx_.unsqueeze(0), ths)
    else:
        H1 = torch.tensor([], device=device)
        H2 = torch.tensor([], device=device)

        iidx = torch.zeros(n, dtype=torch.bool, device=device)
        oidx = torch.zeros(n, dtype=torch.bool, device=device)        
        
    return H1, H2, iidx, oidx, vidx, sidx_


def rot_best(pt1, pt2, n=4):
    # start = time.time()    
    
    n = int(n)
    if n < 1:
        n = 1
    
    # current MiHo formulation is translation invariant
    pt1 = pt1[:2]
    pt2 = pt2[:2]

    d0 = dist2(pt1.t())
    d2 = dist2(pt2.t())

    a = torch.arange(n) * 2 * np.pi / n
    R = torch.stack((torch.stack((torch.cos(a), -torch.sin(a)), dim=1), torch.stack((torch.sin(a), torch.cos(a)), dim=1)), dim=1).to(device)    

    pt2_ = torch.matmul(R, pt2)
    # ptm = (pt1[None, :] + pt2_) / 2
    ptm = (pt1 + pt2_) / 2

    d1 = dist2_batch(ptm.permute(0, 2, 1))
    
    # in_middle = (torch.sign(d0[None, :, :] - d1) * torch.sign(d1 - d2[None, :, :])) > 0    
    in_middle = (torch.sign(d0 - d1) * torch.sign(d1 - d2)) > 0    

    sum_all_k = torch.sum(in_middle,(1, 2))    
    # print(sum_all_k)
    
    best_i = torch.argmax(sum_all_k)
    
    aux = torch.eye(3).to(device)
    aux[:2, :2] = R[best_i]
               
    # end = time.time()
    # print("Elapsed rot_best time = %s" % (end - start))    
    
    return aux


def rot_best_block(pt1, pt2, n=4, split_sz=2048):
    # start = time.time()    
    
    n = int(n)
    if n < 1:
        n = 1
    
    # current MiHo formulation is translation invariant
    pt1 = pt1[:2]
    pt2 = pt2[:2]

    sum_all_k = torch.zeros(4, device=device)

    a = torch.arange(n) * 2 * np.pi / n
    R = torch.stack((torch.stack((torch.cos(a), -torch.sin(a)), dim=1), torch.stack((torch.sin(a), torch.cos(a)), dim=1)), dim=1).to(device)    

    pt1_blk = torch.split(pt1, split_sz, dim=1)
    pt2_blk = torch.split(pt2, split_sz, dim=1)

    for i in np.arange(len(pt1_blk)):
        for j in np.arange(len(pt1_blk)):
            d0 = dist2p(pt1_blk[i].t(), pt1_blk[j].t())
            d2 = dist2p(pt2_blk[i].t(), pt2_blk[j].t())

            pt2_i_ = torch.matmul(R, pt2_blk[i])
            pt2_j_ = torch.matmul(R, pt2_blk[j])

            ptm_i = (pt1_blk[i] + pt2_i_) / 2
            ptm_j = (pt1_blk[j] + pt2_j_) / 2

            d1 = dist2p_batch(ptm_i.permute(0, 2, 1), ptm_j.permute(0, 2, 1))
    
            # in_middle = (torch.sign(d0[None, :, :] - d1) * torch.sign(d1 - d2[None, :, :])) > 0    
            in_middle = (torch.sign(d0 - d1) * torch.sign(d1 - d2)) > 0    

            sum_all_k = sum_all_k + torch.sum(in_middle,(1, 2))    
    # print(sum_all_k)
    
    best_i = torch.argmax(sum_all_k)
    
    aux = torch.eye(3).to(device)
    aux[:2, :2] = R[best_i]
               
    # end = time.time()
    # print("Elapsed rot_best time = %s" % (end - start))    
    
    return aux


def get_avg_hom(pt1, pt2, ransac_middle_args={}, min_plane_pts=4, min_pt_gap=4,
                max_fail_count=3, random_seed_init=123, th_grid=15,
                rot_check=4):

    # set to 123 for debugging and profiling
    if random_seed_init is not None:
        torch.manual_seed(random_seed_init)

    H1 = torch.eye(3, device=device)
    H2 = torch.eye(3, device=device)

    Hdata = []
    l = pt1.shape[0]

    midx = torch.zeros(l, dtype=torch.bool, device=device)
    tidx = torch.zeros(l, dtype=torch.bool, device=device)

    # d1 = dist2(pt1) > th_grid**2
    # d2 = dist2(pt2) > th_grid**2
    # dd = d1 & d2

    pt1 = torch.cat((pt1.t(), torch.ones(1, l, device=device)))
    pt2 = torch.cat((pt2.t(), torch.ones(1, l, device=device)))

    if rot_check > 1:
        H2 = rot_best_block(pt1, pt2, rot_check)
        # H2 = rot_best(pt1, pt2, rot_check)


    fail_count = 0
    midx_sum = 0
    ssidx = torch.zeros((l, 0), dtype=torch.bool, device=device)
    sidx = torch.arange(l, device=device)

    pt1 = torch.matmul(H1, pt1)
    pt1 = pt1 / pt1[2]

    pt2 = torch.matmul(H2, pt2)
    pt2 = pt2 / pt2[2]

    while torch.sum(midx) < l - 4:
        pt1_ = pt1[:, ~midx]
        pt2_ = pt2[:, ~midx]

        # dd_ = dd[~midx, :][:, ~midx].to(device)
        dd_ = None

        ssidx = ssidx[~midx, :]

        H1_, H2_, iidx, oidx, ssidx, sidx_ = ransac_middle(pt1_, pt2_, dd_, th_grid, ssidx=ssidx, **ransac_middle_args)

        sidx_ = sidx[~midx][sidx_]

        # print(torch.sum(ssidx, dim=0))
        good_ssidx = torch.logical_not(torch.sum(ssidx, dim=0) == 0)
        ssidx = ssidx[:, good_ssidx]
        tsidx = torch.zeros((l, ssidx.shape[1]), dtype=torch.bool, device=device)
        tsidx[~midx, :] = ssidx
        ssidx = tsidx

        idx = torch.zeros(l, dtype=torch.bool, device=device)
        idx[~midx] = oidx

        midx[~midx] = iidx
        tidx = tidx | idx

        midx_sum_old = midx_sum
        midx_sum = torch.sum(midx)

        H_failed = torch.sum(oidx) <= min_plane_pts
        inl_failed = midx_sum - midx_sum_old <= min_pt_gap
        if H_failed or inl_failed:
            fail_count += 1
            if fail_count > max_fail_count:
                break
            if inl_failed:
                midx = tidx
            if H_failed:
                continue
        else:
            fail_count = 0

        # print(f"{torch.sum(tidx)} {torch.sum(midx)} {fail_count}")

        Hdata.append([torch.matmul(H1_, H1), torch.matmul(H2_, H2), idx, sidx_])

    return Hdata, H1, H2


def dist2_batch(pt):
    pt = pt.type(torch.float32)
    d = (pt.unsqueeze(-1)[:, :, 0] - pt.unsqueeze(1)[:, :, :, 0])**2 + (pt.unsqueeze(-1)[:, :, 1] - pt.unsqueeze(1)[:, :, :, 1])**2
    # d = (pt[:, :, 0, None] - pt[:, None, :, 0])**2 + (pt[:, :, 1, None] - pt[:, None, :, 1])**2
    return d


def dist2p_batch(pt1, pt2):
    pt1 = pt1.type(torch.float32)
    pt2 = pt2.type(torch.float32)
    d = (pt1.unsqueeze(-1)[:, :, 0] - pt2.unsqueeze(1)[:, :, :, 0])**2 + (pt1.unsqueeze(-1)[:, :, 1] - pt2.unsqueeze(1)[:, :, :, 1])**2
    return d


def dist2(pt):
    pt = pt.type(torch.float32)
    d = (pt.unsqueeze(-1)[:, 0] - pt.unsqueeze(0)[:, :, 0])**2 + (pt.unsqueeze(-1)[:, 1] - pt.unsqueeze(0)[:, :, 1])**2
    # d = (pt[:, 0, None] - pt[None, :, 0])**2 + (pt[:, 1, None] - pt[None, :, 1])**2
    return d


def dist2p(pt1, pt2):
    pt1 = pt1.type(torch.float32)
    pt2 = pt2.type(torch.float32)
    d = (pt1.unsqueeze(-1)[:, 0] - pt2.unsqueeze(0)[:, :, 0])**2 + (pt1.unsqueeze(-1)[:, 1] - pt2.unsqueeze(0)[:, :, 1])**2
    return d


def cluster_assign_base(Hdata, pt1, pt2, H1_pre, H2_pre, **dummy_args):
    l = len(Hdata)
    n = Hdata[0][2].shape[0]
    
    if not((l>0) and (n>0)):
        return torch.full((n, ), -1, dtype=torch.int, device=device)
    
    inl_mask = torch.zeros((n, l), dtype=torch.bool, device=device)
    for i in range(l):
        inl_mask[:, i] = Hdata[i][2]

    alone_idx = torch.sum(inl_mask, dim=1) == 0
    set_size = torch.sum(inl_mask, dim=0)

    max_size_idx = torch.argmax(
        set_size.view(1, -1).expand(inl_mask.shape[0], -1) * inl_mask, dim=1)
    max_size_idx[alone_idx] = -1

    return max_size_idx


def cluster_assign(Hdata, pt1, pt2, H1_pre, H2_pre, median_th=5, err_th=15, **dummy_args):    
    l = len(Hdata)
    n = pt1.shape[0]

    if not((l>0) and (n>0)):
        return torch.full((n, ), -1, dtype=torch.int, device=device)

    pt1 = torch.vstack((pt1.T, torch.ones((1, n), device=device)))
    pt2 = torch.vstack((pt2.T, torch.ones((1, n), device=device)))

    pt1_ = torch.matmul(H1_pre, pt1)
    pt1_ = pt1_ / pt1_[2]

    pt2_ = torch.matmul(H2_pre, pt2)
    pt2_ = pt2_ / pt2_[2]

    ptm = (pt1_ + pt2_) / 2

    # err = torch.zeros((n, l), device=device)
    # inl_mask = torch.zeros((n, l), dtype=torch.bool, device=device)
    #
    # for i in range(l):
    #     H1 = Hdata[i][0]
    #     H2 = Hdata[i][1]
    #     sidx = Hdata[i][3]
    #
    #     inl_mask[:, i] = Hdata[i][2]
    #
    #     err[:, i] = torch.maximum(get_error(pt1, ptm, H1, sidx), get_error(pt2, ptm, H2, sidx))

    H12 = torch.zeros((l*2, 3, 3), device=device)
    sidx_par = torch.zeros((l, 4), device=device, dtype=torch.long)
    inl_mask = torch.zeros((n, l), dtype=torch.bool, device=device)

    for i in range(l):
        H12[i] = Hdata[i][0]
        H12[i+l] = Hdata[i][1]
        sidx_par[i] = Hdata[i][3]

        inl_mask[:, i] = Hdata[i][2]

    err = get_error_duplex(H12, pt1, pt2, ptm, sidx_par).permute(1,0)

    # min error
    abs_err_min_val, abs_err_min_idx = torch.min(err, dim=1)

    set_size = torch.sum(inl_mask, dim=0)
    size_mask = torch.repeat_interleave(set_size.unsqueeze(0), n, dim=0) * inl_mask

    # take a cluster if its cardinality is more than the median of the top median_th ones
    ssize_mask, _ = torch.sort(size_mask, descending=True, dim=1)

    median_idx = torch.sum(ssize_mask[:, :median_th] > 0, dim=1) / 2
    median_idx[median_idx == 1] = 1.5
    median_idx = torch.maximum(torch.ceil(median_idx).to(torch.int)-1, torch.tensor(0))

    # flat_indices = torch.arange(n) * ssize_mask.shape[1] + median_idx
    # top_median = ssize_mask.view(-1)[flat_indices]
    top_median = ssize_mask.flatten()[torch.arange(n, device=device) * ssize_mask.shape[1] + median_idx]

    # take among the selected the one which gives less error
    discarded_mask = size_mask < top_median.unsqueeze(1)
    err[discarded_mask] = float('inf')
    err_min_idx = torch.argmin(err, dim=1)

    # remove match with no cluster
    alone_idx = torch.sum(inl_mask, dim=1) == 0
    really_alone_idx = alone_idx & (abs_err_min_val > err_th**2)

    err_min_idx[alone_idx] = abs_err_min_idx[alone_idx]
    err_min_idx[really_alone_idx] = -1

    return err_min_idx


def cluster_assign_other(Hdata, pt1, pt2, H1_pre, H2_pre, err_th_only=15, **dummy_args):
    l = len(Hdata)
    n = pt1.shape[0]

    if not((l>0) and (n>0)):
        return torch.full((n, ), -1, dtype=torch.int, device=device)

    pt1 = torch.vstack((pt1.T, torch.ones((1, n), device=device)))
    pt2 = torch.vstack((pt2.T, torch.ones((1, n), device=device)))

    pt1_ = torch.matmul(H1_pre, pt1)
    pt1_ = pt1_ / pt1_[2]

    pt2_ = torch.matmul(H2_pre, pt2)
    pt2_ = pt2_ / pt2_[2]

    ptm = (pt1_ + pt2_) / 2

    # err = torch.zeros((n, l), device=device)
    #
    # for i in range(l):
    #     H1 = Hdata[i][0]
    #     H2 = Hdata[i][1]
    #     sidx = Hdata[i][3]
    #
    #     err[:, i] = torch.maximum(get_error(pt1, ptm, H1, sidx), get_error(pt2, ptm, H2, sidx))

    H12 = torch.zeros((l*2, 3, 3), device=device)
    sidx_par = torch.zeros((l, 4), device=device, dtype=torch.long)

    for i in range(l):
        H12[i] = Hdata[i][0]
        H12[i+l] = Hdata[i][1]
        sidx_par[i] = Hdata[i][3]

    err = get_error_duplex(H12, pt1, pt2, ptm, sidx_par).permute(1,0)

    # err_min_idx = torch.argmin(err, dim=1)
    # err_min_val = err.flatten()[torch.arange(n, device=device) * l + err_min_idx]

    err_min_val, err_min_idx = torch.min(err, dim=1)

    err_min_idx[(err_min_val > err_th_only**2) | torch.isnan(err_min_val)] = -1

    return err_min_idx


def show_fig(im1, im2, pt1, pt2, Hidx, tosave='miho_buffered_rot_pytorch_gpu.pdf', fig_dpi=300,
             colors = ['#FF1F5B', '#00CD6C', '#009ADE', '#AF58BA', '#FFC61E', '#F28522'],
             markers = ['o','x','8','p','h'], bad_marker = 'd', bad_color = '#000000',
             plot_opt = {'markersize': 2, 'markeredgewidth': 0.5,
                         'markerfacecolor': "None", 'alpha': 0.5}):

    im12 = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)))
    im12.paste(im1, (0, 0))
    im12.paste(im2, (im1.width, 0))

    plt.figure()
    plt.axis('off')
    plt.imshow(im12)

    cn = len(colors)
    mn = len(markers)

    for i, idx in enumerate(np.ndarray.tolist(np.unique(Hidx))):
        mask = Hidx == idx
        x = np.vstack((pt1[mask, 0], pt2[mask, 0]+im1.width))
        y = np.vstack((pt1[mask, 1], pt2[mask, 1]))
        if (idx == -1):
            color = bad_color
            marker = bad_marker
        else:
            color = colors[((i-1)%(cn*mn))%cn]
            marker = markers[((i-1)%(cn*mn))//cn]
        plt.plot(x, y, linestyle='', color=color, marker=marker, **plot_opt)

    plt.savefig(tosave, dpi = fig_dpi, bbox_inches='tight')


def go_assign(Hdata, pt1, pt2, H1_pre, H2_pre, method=cluster_assign, method_args={}):
    return method(Hdata, pt1, pt2, H1_pre, H2_pre, **method_args)


def purge_default(dict1, dict2):

    if type(dict1) != type(dict2):
        return None

    if not (isinstance(dict2, dict) or isinstance(dict2, list)):
        if dict1 == dict2:
            return None
        else:
            return dict2

    elif isinstance(dict2, dict):

        keys1 = list(dict1.keys())
        keys2 = list(dict2.keys())

        for i in keys2:
            if i not in keys1:
                dict2.pop(i, None)
            else:
                aux = purge_default(dict1[i], dict2[i])
                if not aux:
                    dict2.pop(i, None)
                else:
                    dict2[i] = aux

        return dict2

    elif isinstance(dict2, list):

        if len(dict1) != len(dict2):
            return dict2

        for i in range(len(dict2)):
            aux = purge_default(dict1[i], dict2[i])
            if aux:
               return dict2

        return None


def merge_params(dict1, dict2):

    keys1 = dict1.keys()
    keys2 = dict2.keys()

    for i in keys1:
        if (i in keys2):
            if (not isinstance(dict1[i], dict)):
                dict1[i] = dict2[i]
            else:
                dict1[i] = merge_params(dict1[i], dict2[i])

    return dict1


class miho:
    def __init__(self, params=None):
        """initiate MiHo"""
        self.set_default()

        if params is not None:
            self.update_params(params)


    def set_default(self):
        """set default MiHo parameters"""
        self.params = { 'get_avg_hom': {}, 'go_assign': {}, 'show_clustering': {}}


    def get_current(self):
        """get current MiHo parameters"""
        tmp_params = self.params.copy()

        for i in ['get_avg_hom', 'show_clustering', 'go_assign']:
            if i not in tmp_params:
                tmp_params[i] = {}

        return merge_params(self.all_params(), tmp_params)


    def update_params(self, params):
        """update current MiHo parameters"""
        all_default_params = self.all_params()
        clear_params = purge_default(all_default_params, params.copy())

        for i in ['get_avg_hom', 'show_clustering', 'go_assign']:
            if i in clear_params:
                self.params[i] = clear_params[i]


    @staticmethod
    def all_params():
        """all MiHo parameters with default values"""
        ransac_middle_params = {'th_in': 7, 'th_out': 15, 'max_iter': 500,
                                'min_iter': 50, 'p' :0.9, 'svd_th': 0.05,
                                'buffers': 5}
        get_avg_hom_params = {'ransac_middle_args': ransac_middle_params,
                              'min_plane_pts': 4, 'min_pt_gap': 4,
                              'max_fail_count': 3, 'random_seed_init': 123,
                              'th_grid': 15, 'rot_check': 4}

        method_args_params = {'median_th': 5, 'err_th': 15, 'err_th_only': 15, 'par_value': 100000}
        go_assign_params = {'method': cluster_assign,
                            'method_args': method_args_params}

        show_clustering_params = {'tosave': 'miho_buffered_rot_pytorch_gpu.pdf', 'fig_dpi': 300,
             'colors': ['#FF1F5B', '#00CD6C', '#009ADE', '#AF58BA', '#FFC61E', '#F28522'],
             'markers': ['o','x','8','p','h'], 'bad_marker': 'd', 'bad_color': '#000000',
             'plot_opt': {'markersize': 2, 'markeredgewidth': 0.5,
                          'markerfacecolor': "None", 'alpha': 0.5}}

        return {'get_avg_hom': get_avg_hom_params,
                'go_assign': go_assign_params,
                'show_clustering': show_clustering_params}


    def planar_clustering(self, pt1, pt2):
        """run MiHo"""
        self.pt1 = pt1
        self.pt2 = pt2

        Hdata, H1_pre, H2_pre = get_avg_hom(self.pt1, self.pt2, **self.params['get_avg_hom'])
        self.Hs = Hdata
        self.H1_pre = H1_pre
        self.H2_pre = H2_pre

        self.Hidx = go_assign(Hdata, self.pt1, self.pt2, H1_pre, H2_pre, **self.params['go_assign'])

        return self.Hs, self.Hidx


    def attach_images(self, im1, im2):
        """" add image pair to MiHo and tensorify it"""
        
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.PILToTensor() 
            ]) 

        self.img1 = im1.copy()
        self.img2 = im2.copy()
        
        self.im1 = transform(im1).type(torch.float16).to(device)
        self.im2 = transform(im2).type(torch.float16).to(device)
        
        return True


    def show_clustering(self):
        """ show MiHo clutering"""
        if hasattr(self, 'Hs') and hasattr(self, 'img1'):
            show_fig(self.img1, self.img2, self.pt1.cpu(), self.pt2.cpu(), self.Hidx.cpu(), **self.params['show_clustering'])
        else:
            warnings.warn("planar_clustering must run before!!!")


import os
import warnings
import _pickle as cPickle
import bz2
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


# Pickle a file and then compress it into a file with extension 
def compressed_pickle(title, data, add_ext=False):
    if add_ext:
        ext = '.pbz2'
    else:
        ext = ''
        
    with bz2.BZ2File(title + ext, 'w') as f: 
        cPickle.dump(data, f)
        

# Load any compressed pickle file
def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data


def progress_bar(text=''):
    return Progress(
        TextColumn(text + " [progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    )


def megadepth_1500_list(ppath='bench_data/gt_data/megadepth'):
    npz_list = [i for i in os.listdir(ppath) if (os.path.splitext(i)[1] == '.npz')]

    data = {'im1': [], 'im2': [], 'K1': [], 'K2': [], 'T': [], 'R': []}
    for name in npz_list:
        scene_info = np.load(os.path.join(ppath, name), allow_pickle=True)
    
        # Collect pairs
        for pair_info in scene_info['pair_infos']:
            (id1, id2), overlap, _ = pair_info
            im1 = scene_info['image_paths'][id1].replace('Undistorted_SfM/', '')
            im2 = scene_info['image_paths'][id2].replace('Undistorted_SfM/', '')                        
            K1 = scene_info['intrinsics'][id1].astype(np.float32)
            K2 = scene_info['intrinsics'][id2].astype(np.float32)
    
            # Compute relative pose
            T1 = scene_info['poses'][id1]
            T2 = scene_info['poses'][id2]
            T12 = np.matmul(T2, np.linalg.inv(T1))
    
            data['im1'].append(im1)
            data['im2'].append(im2)
            data['K1'].append(K1)
            data['K2'].append(K2)
            data['T'].append(T12[:3, 3])
            data['R'].append(T12[:3, :3])   
    return data


def scannet_1500_list(ppath='bench_data/gt_data/scannet'):
    intrinsic_path = 'intrinsics.npz'
    npz_path = 'test.npz'

    data = np.load(os.path.join(ppath, npz_path))
    data_names = data['name']
    intrinsics = dict(np.load(os.path.join(ppath, intrinsic_path)))
    rel_pose = data['rel_pose']
    
    data = {'im1': [], 'im2': [], 'K1': [], 'K2': [], 'T': [], 'R': []}
    
    for idx in range(data_names.shape[0]):
        scene_name, scene_sub_name, stem_name_0, stem_name_1 = data_names[idx]
        scene_name = f'scene{scene_name:04d}_{scene_sub_name:02d}'
    
        # read the grayscale image which will be resized to (1, 480, 640)
        im1 = os.path.join(scene_name, 'color', f'{stem_name_0}.jpg')
        im2 = os.path.join(scene_name, 'color', f'{stem_name_1}.jpg')
        
        # read the intrinsic of depthmap
        K1 = intrinsics[scene_name]
        K2 = intrinsics[scene_name]
    
        # pose    
        T12 = np.concatenate((rel_pose[idx],np.asarray([0, 0, 0, 1.0]))).reshape(4,4)
        
        data['im1'].append(im1)
        data['im2'].append(im2)
        data['K1'].append(K1)
        data['K2'].append(K2)  
        data['T'].append(T12[:3, 3])
        data['R'].append(T12[:3, :3])     
    return data


def bench_init(bench_file='megadepth_scannet', bench_path='bench_data', bench_gt='gt_data'):
    data_file = os.path.join(bench_path, 'megadepth_scannet' + '.pbz2')
    if not os.path.isfile(data_file + '.pbz2'):      
        megadepth_data = megadepth_1500_list(os.path.join(bench_path, bench_gt, 'megadepth'))
        scannet_data = scannet_1500_list(os.path.join(bench_path, bench_gt, 'scannet'))
        compressed_pickle(data_file, (megadepth_data, scannet_data))
    else:
        megadepth_data, scannet_data = decompress_pickle(data_file)
    
    return megadepth_data, scannet_data, data_file


def resize_megadepth(im, res_path='imgs/megadepth', bench_path='bench_data', force=False):
    mod_im = os.path.join(bench_path, res_path, os.path.splitext(im)[0] + '.png')
    ori_im= os.path.join(bench_path, 'megadepth_test_1500/Undistorted_SfM', im)

    if os.path.isfile(mod_im) and not force:
        # PIL does not load image, so it's faster to get only image size
        return np.asarray(Image.open(ori_im).size) / np.asarray(Image.open(mod_im).size) 
        # return np.array(cv2.imread(ori_im).shape)[:2][::-1] / np.array(cv2.imread(mod_im).shape)[:2][::-1]

    img = cv2.imread(ori_im)
    sz_ori = np.array(img.shape)[:2][::-1]
    sz_max = float(max(sz_ori))

    if sz_max > 1200:
        cf = 1200 / sz_max                    
        sz_new = np.ceil(sz_ori * cf).astype(int) 
        img = cv2.resize(img, tuple(sz_new), interpolation=cv2.INTER_LANCZOS4)
        sc = sz_ori/sz_new
        os.makedirs(os.path.dirname(mod_im), exist_ok=True)                 
        cv2.imwrite(mod_im, img)
        return sc
    else:
        os.makedirs(os.path.dirname(mod_im), exist_ok=True)                 
        cv2.imwrite(mod_im, img)
        return np.array([1., 1.])


def resize_scannet(im, res_path='imgs/scannet', bench_path='bench_data', force=False):
    mod_im = os.path.join(bench_path, res_path, os.path.splitext(im)[0] + '.png')
    ori_im= os.path.join(bench_path, 'scannet_test_1500', im)

    if os.path.isfile(mod_im) and not force:
        # PIL does not load image, so it's faster to get only image size
        return np.asarray(Image.open(ori_im).size) / np.asarray(Image.open(mod_im).size) 
        # return np.array(cv2.imread(ori_im).shape)[:2][::-1] / np.array(cv2.imread(mod_im).shape)[:2][::-1]

    img = cv2.imread(ori_im)
    sz_ori = np.array(img.shape)[:2][::-1]

    sz_new = np.array([640, 480])
    img = cv2.resize(img, tuple(sz_new), interpolation=cv2.INTER_LANCZOS4)
    sc = sz_ori/sz_new
    os.makedirs(os.path.dirname(mod_im), exist_ok=True)                 
    cv2.imwrite(mod_im, img)
    return sc


def setup_images(megadepth_data, scannet_data, data_file='bench_data/megadepth_scannet.pbz2', bench_path='bench_data', bench_imgs='imgs'):
    if not ('im_pair_scale' in megadepth_data.keys()):        
        n = len(megadepth_data['im1'])
        im_pair_scale = np.zeros((n, 2, 2))
        res_path = os.path.join(bench_imgs, 'megadepth')
        with progress_bar('MegaDepth - image setup completion') as p:
            for i in p.track(range(n)):
                im_pair_scale[i, 0] = resize_megadepth(megadepth_data['im1'][i], res_path, bench_path)
                im_pair_scale[i, 1] = resize_megadepth(megadepth_data['im2'][i], res_path, bench_path)
        megadepth_data['im_pair_scale'] = im_pair_scale

        n = len(scannet_data['im1'])
        im_pair_scale = np.zeros((n, 2, 2))
        res_path = os.path.join(bench_imgs, 'scannet')
        with progress_bar('ScanNet - image setup completion') as p:
            for i in p.track(range(n)):
                im_pair_scale[i, 0] = resize_scannet(scannet_data['im1'][i], res_path, bench_path)
                im_pair_scale[i, 1] = resize_scannet(scannet_data['im2'][i], res_path, bench_path)
        scannet_data['im_pair_scale'] = im_pair_scale
        
        compressed_pickle(data_file, (megadepth_data, scannet_data))
 
    return megadepth_data, scannet_data


def relative_pose_error(R_gt, t_gt, R, t, ignore_gt_t_thr=0.0):
    # angle error between 2 vectors
    # t_gt = T_0to1[:3, 3]
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
    t_err = np.minimum(t_err, 180 - t_err)  # handle E ambiguity
    if np.linalg.norm(t_gt) < ignore_gt_t_thr:  # pure rotation is challenging
        t_err = 0

    # angle error between 2 rotation matrices
    # R_gt = T_0to1[:3, :3]
    cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # handle numercial errors
    R_err = np.rad2deg(np.abs(np.arccos(cos)))

    return t_err, R_err


def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999, max_iters=10000):
    if len(kpts0) < 5:
        return None
    # normalize keypoints
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    # normalize ransac threshold
    ransac_thr = thresh / np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])

    # compute pose with cv2
    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=ransac_thr, prob=conf, method=cv2.RANSAC, maxIters=max_iters)
    if E is None:
        print("\nE is None while trying to recover pose.\n")
        return None

    # recover pose from E
    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(
            _E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            ret = (R, t[:, 0], mask.ravel() > 0)
            best_num_inliers = n

    return ret


def error_auc(errors, thr):
    errors = [0] + sorted(errors)
    recall = list(np.linspace(0, 1, len(errors)))

    last_index = np.searchsorted(errors, thr)
    y = recall[:last_index] + [recall[last_index-1]]
    x = errors[:last_index] + [thr]
    return np.trapz(y, x) / thr    


def run_pipe(pipe, dataset_data, dataset_name, bar_name, bench_path='bench_data' , bench_im='imgs', bench_res='res', force=False):

    n = len(dataset_data['im1'])
    im_path = os.path.join(bench_im, dataset_name)        
    with progress_bar(bar_name + ' - pipeline completion') as p:
        for i in p.track(range(n)):
            #if i == 10:
            #    break
            im1 = os.path.join(bench_path, im_path, os.path.splitext(dataset_data['im1'][i])[0]) + '.png'
            im2 = os.path.join(bench_path, im_path, os.path.splitext(dataset_data['im2'][i])[0]) + '.png'

            pipe_name_base = os.path.join(bench_path, bench_res, dataset_name)
            for pipe_module in pipe:
                pipe_name_base = os.path.join(pipe_name_base, pipe_module.get_id())
                pipe_f = os.path.join(pipe_name_base, 'base', str(i) + '.pbz2')
            
                if os.path.isfile(pipe_f) and not force:
                    out_data = decompress_pickle(pipe_f)
                else:                      
                    out_data = eval(pipe_module.eval_args())
                    os.makedirs(os.path.dirname(pipe_f), exist_ok=True)                 
                    compressed_pickle(pipe_f, out_data)
                    
                exec(pipe_module.eval_out())


def eval_pipe(pipe, dataset_data,  dataset_name, bar_name, bench_path='bench_data', bench_res='res', essential_th_list=[0.5, 1, 1.5], save_to='res.pbz2', force=False, use_scale=False):
    warnings.filterwarnings("ignore", category=UserWarning)

    angular_thresholds = [5, 10, 20]

    K1 = dataset_data['K1']
    K2 = dataset_data['K2']
    R_gt = dataset_data['R']
    t_gt = dataset_data['T']

    if os.path.isfile(save_to):
        eval_data = decompress_pickle(save_to)
    else:
        eval_data = {}
        
    for essential_th in essential_th_list:            
        n = len(dataset_data['im1'])
        
        pipe_name_base = os.path.join(bench_path, bench_res, dataset_name)
        pipe_name_base_small = ''
        for pipe_module in pipe:
            pipe_name_base = os.path.join(pipe_name_base, pipe_module.get_id())
            pipe_name_base_small = os.path.join(pipe_name_base_small, pipe_module.get_id())

            print(bar_name + ' evaluation with RANSAC essential matrix threshold ' + str(essential_th) + ' px')
            print('Pipeline: ' + pipe_name_base_small)

            if ((pipe_name_base + '_essential_th_list_' + str(essential_th)) in eval_data.keys()) and not force:
                eval_data_ = eval_data[pipe_name_base + '_essential_th_list_' + str(essential_th)]                
                for a in angular_thresholds:
                    print(f"mAA@{str(a)} : {eval_data_['pose_error_auc_' + str(a)]}")
                
                continue
                    
            eval_data_ = {}
            eval_data_['R_errs'] = []
            eval_data_['t_errs'] = []
            eval_data_['inliers'] = []
                
            with progress_bar('Completion') as p:
                for i in p.track(range(n)):            
                    pipe_f = os.path.join(pipe_name_base, 'base', str(i) + '.pbz2')
                    
                    if os.path.isfile(pipe_f):
                        out_data = decompress_pickle(pipe_f)
    
                        pts1 = out_data[0]
                        pts2 = out_data[1]
                                                
                        if torch.is_tensor(pts1):
                            pts1 = pts1.detach().cpu().numpy()
                            pts2 = pts2.detach().cpu().numpy()
                        
                        if not pts1.size:
                            Rt = None

                        else:
                            if use_scale:
                                scales = dataset_data['im_pair_scale'][i]
                            
                                pts1 = pts1 * scales[0]
                                pts2 = pts2 * scales[1]
                            
                            Rt = estimate_pose(pts1, pts2, K1[i], K2[i], essential_th)
                    else:
                        Rt = None
        
                    if Rt is None:
                        eval_data_['R_errs'].append(np.inf)
                        eval_data_['t_errs'].append(np.inf)
                        eval_data_['inliers'].append(np.array([]).astype('bool'))
                    else:
                        R, t, inliers = Rt
                        t_err, R_err = relative_pose_error(R_gt[i], t_gt[i], R, t, ignore_gt_t_thr=0.0)
                        eval_data_['R_errs'].append(R_err)
                        eval_data_['t_errs'].append(t_err)
                        eval_data_['inliers'].append(inliers)
        
                aux = np.stack(([eval_data_['R_errs'], eval_data_['t_errs']]), axis=1)
                max_Rt_err = np.max(aux, axis=1)
        
                tmp = np.concatenate((aux, np.expand_dims(
                    np.max(aux, axis=1), axis=1)), axis=1)
        
                for a in angular_thresholds:       
                    auc_R = error_auc(np.squeeze(eval_data_['R_errs']), a)
                    auc_t = error_auc(np.squeeze(eval_data_['t_errs']), a)
                    auc_max_Rt = error_auc(np.squeeze(max_Rt_err), a)
                    eval_data_['pose_error_auc_' + str(a)] = np.asarray([auc_R, auc_t, auc_max_Rt])
                    eval_data_['pose_error_acc_' + str(a)] = np.sum(tmp < a, axis=0)/np.shape(tmp)[0]

                    print(f"mAA@{str(a)} : {eval_data_['pose_error_auc_' + str(a)]}")

            eval_data[pipe_name_base + '_essential_th_list_' + str(essential_th)] = eval_data_
            compressed_pickle(save_to, eval_data)
            

class keynetaffnethardnet_module:
    def __init__(self, **args):
        self.upright = False
        self.th = 0.99
        with torch.inference_mode():
            self.detector = K.feature.KeyNetAffNetHardNet(upright=self.upright, device=device)
        
        for k, v in args.items():
           setattr(self, k, v)
        
        
    def get_id(self):
        return ('keynetaffnethardnet_upright_' + str(self.upright) + '_th_' + str(self.th)).lower()

    
    def eval_args(self):
        return "pipe_module.run(im1, im2)"


    def eval_out(self):
        return "pt1, pt2, kps1, kps2, Hs = out_data"               


    def run(self, *args):    
        with torch.inference_mode():
            kps1, _ , descs1 = self.detector(K.io.load_image(args[0], K.io.ImageLoadType.GRAY32, device=device).unsqueeze(0))
            kps2, _ , descs2 = self.detector(K.io.load_image(args[1], K.io.ImageLoadType.GRAY32, device=device).unsqueeze(0))
            dists, idxs = K.feature.match_smnn(descs1.squeeze(), descs2.squeeze(), self.th)        
        
        pt1 = None
        pt2 = None
        kps1 = kps1.squeeze().detach()[idxs[:, 0]].to(device)
        kps2 = kps2.squeeze().detach()[idxs[:, 1]].to(device)
        
        pt1, pt2, Hs_laf = refinement_laf(None, None, data1=kps1, data2=kps2, img_patches=False)    
    
        return pt1, pt2, kps1, kps2, Hs_laf


class miho_module:
    def __init__(self, **args):
        self.miho = miho()
        
        for k, v in args.items():
           setattr(self, k, v)
        
        
    def get_id(self):
        return ('miho_default').lower()

    
    def eval_args(self):
        return "pipe_module.run(pt1, pt2, Hs)"


    def eval_out(self):
        return "pt1, pt2, Hs, inliers = out_data"               


    def run(self, *args):
        self.miho.planar_clustering(args[0], args[1])
        
        pt1, pt2, Hs_miho, inliers = refinement_miho(None, None, args[0], args[1], self.miho, args[2], remove_bad=True, img_patches=False)        
            
        return pt1, pt2, Hs_miho, inliers


class ncc_module:
    def __init__(self, **args):
        self.w = 10
        self.w_big = None
        self.angle = [-30, -15, 0, 15, 30]
        self.scale = [[10/14, 1], [10/12, 1], [1, 1], [1, 12/10], [1, 14/10]]
        self.subpix = True
        self.ref_images = 'both'
        
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.PILToTensor() 
            ]) 
        
        for k, v in args.items():
           setattr(self, k, v)
        
        
    def get_id(self):
        return ('nnc_subpix_' + str(self.subpix) + '_w_' + str(self.w) + '_w_big_' + str(self.w_big) + '_ref_images_' + str(self.ref_images) + '_scales_' +  str(len(self.scale)) + '_angles_' + str(len(self.scale))).lower()

    
    def eval_args(self):
        return "pipe_module.run(pt1, pt2, Hs, im1, im2)"


    def eval_out(self):
        return "pt1, pt2, Hs, val, T = out_data"            


    def run(self, *args):
        im1 = Image.open(args[3])
        im2 = Image.open(args[4])

        im1 = self.transform(im1).type(torch.float16).to(device)
        im2 = self.transform(im2).type(torch.float16).to(device)        
        
        pt1, pt2, Hs_ncc, val, T = refinement_norm_corr_alternate(im1, im2, args[0], args[1], args[2], w=self.w, w_big=self.w_big, ref_image=[self.ref_images], angle=self.angle, scale=self.scale, subpix=self.subpix, img_patches=False)   
                    
        return pt1, pt2, Hs_ncc, val, T


class pydegensac_module:
    def __init__(self, **args):
        self.px_th = 3
        self.conf = 0.9999
        self.max_iters = 100000
              
        for k, v in args.items():
           setattr(self, k, v)
       
        
    def get_id(self):
        return ('pydegensac_th_' + str(self.px_th) + '_conf_' + str(self.conf) + '_max_iters_' + str(self.max_iters)).lower()

    
    def eval_args(self):
        return "pipe_module.run(pt1, pt2, Hs)"

        
    def eval_out(self):
        return "pt1, pt2, Hs, F, mask = out_data"
    
    
    def run(self, *args):  
        pt1 = args[0]
        pt2 = args[1]
        Hs = args[2]
        
        if torch.is_tensor(pt1):
            pt1 = pt1.detach().cpu()
            pt2 = pt1.detach().cpu()
            
        if (np.ascontiguousarray(pt1).shape)[0] > 7:                        
            F, mask = pydegensac.findFundamentalMatrix(np.ascontiguousarray(pt1), np.ascontiguousarray(pt2), px_th=self.px_th, conf=self.conf, max_iters=self.max_iters)
    
            pt1 = args[0][mask]
            pt2 = args[1][mask]     
            Hs = args[2][mask]
        else:            
            F = None
            mask = None
            
        return pt1, pt2, Hs, F, mask


THRESHOLD_FACTOR = 6

ROTATION_PATTERNS = [
    [1, 2, 3,
     4, 5, 6,
     7, 8, 9],

    [4, 1, 2,
     7, 5, 3,
     8, 9, 6],

    [7, 4, 1,
     8, 5, 2,
     9, 6, 3],

    [8, 7, 4,
     9, 5, 1,
     6, 3, 2],

    [9, 8, 7,
     6, 5, 4,
     3, 2, 1],

    [6, 9, 8,
     3, 5, 7,
     2, 1, 4],

    [3, 6, 9,
     2, 5, 8,
     1, 4, 7],

    [2, 3, 6,
     1, 5, 9,
     4, 7, 8]]


class Size:
    def __init__(self, width, height):
        self.width = width
        self.height = height


class GmsMatcher:
    def __init__(self, kp1, kp2, m12):
        self.kp1=kp1
        self.kp2=kp2
        self.m12=m12

        self.scale_ratios = [1.0, 1.0 / 2, 1.0 / math.sqrt(2.0), math.sqrt(2.0), 2.0]
        
        # Normalized vectors of 2D points
        self.normalized_points1 = []
        self.normalized_points2 = []
        # Matches - list of pairs representing numbers
        self.matches = []
        self.matches_number = 0
        # Grid Size
        self.grid_size_right = Size(0, 0)
        self.grid_number_right = 0
        # x      : left grid idx
        # y      :  right grid idx
        # value  : how many matches from idx_left to idx_right
        self.motion_statistics = []

        self.number_of_points_per_cell_left = []
        # Inldex  : grid_idx_left
        # Value   : grid_idx_right
        self.cell_pairs = []

        # Every Matches has a cell-pair
        # first  : grid_idx_left
        # second : grid_idx_right
        self.match_pairs = []

        # Inlier Mask for output
        self.inlier_mask = []
        self.grid_neighbor_right = []

        # Grid initialize
        self.grid_size_left = Size(20, 20)
        self.grid_number_left = self.grid_size_left.width * self.grid_size_left.height

        # Initialize the neihbor of left grid
        self.grid_neighbor_left = np.zeros((self.grid_number_left, 9))

#       self.descriptor = descriptor
#       self.matcher = matcher
        self.gms_matches = []
        self.keypoints_image1 = []
        self.keypoints_image2 = []

    def empty_matches(self):
        self.normalized_points1 = []
        self.normalized_points2 = []
        self.matches = []
        self.gms_matches = []

#   def compute_matches(self, img1, img2):
    def compute_matches(self, sz1r, sz1c, sz2r, sz2c):
        self.keypoints_image1=self.kp1
        self.keypoints_image2=self.kp2

#       self.keypoints_image1, descriptors_image1 = self.descriptor.detectAndCompute(img1, np.array([]))
#       self.keypoints_image2, descriptors_image2 = self.descriptor.detectAndCompute(img2, np.array([]))
                        
#       size1 = Size(img1.shape[1], img1.shape[0])
#       size2 = Size(img2.shape[1], img2.shape[0])

        size1 = Size(sz1c, sz1r)
        size2 = Size(sz2c, sz2r)

        if self.gms_matches:
            self.empty_matches()

        all_matches=self.m12
#       all_matches = self.matcher.match(descriptors_image1, descriptors_image2)
                
        self.normalize_points(self.keypoints_image1, size1, self.normalized_points1)
        self.normalize_points(self.keypoints_image2, size2, self.normalized_points2)
        self.matches_number = len(all_matches)
        self.convert_matches(all_matches, self.matches)
                
        self.initialize_neighbours(self.grid_neighbor_left, self.grid_size_left)
        
        mask, num_inliers = self.get_inlier_mask(False, False)
        # print('Found', num_inliers, 'matches')

        for i in range(len(mask)):
            if mask[i]:
                self.gms_matches.append(all_matches[i])
        return self.gms_matches, mask

    # Normalize Key points to range (0-1)
    def normalize_points(self, kp, size, npts):
        for keypoint in kp:
            npts.append((keypoint.pt[0] / size.width, keypoint.pt[1] / size.height))

    # Convert OpenCV match to list of tuples
    def convert_matches(self, vd_matches, v_matches):
        for match in vd_matches:
            v_matches.append((match.queryIdx, match.trainIdx))

    def initialize_neighbours(self, neighbor, grid_size):
        for i in range(neighbor.shape[0]):
            neighbor[i] = self.get_nb9(i, grid_size)

    def get_nb9(self, idx, grid_size):
        nb9 = [-1 for _ in range(9)]
        idx_x = idx % grid_size.width
        idx_y = idx // grid_size.width

        for yi in range(-1, 2):
            for xi in range(-1, 2):
                idx_xx = idx_x + xi
                idx_yy = idx_y + yi

                if idx_xx < 0 or idx_xx >= grid_size.width or idx_yy < 0 or idx_yy >= grid_size.height:
                    continue
                nb9[xi + 4 + yi * 3] = idx_xx + idx_yy * grid_size.width

        return nb9

    def get_inlier_mask(self, with_scale, with_rotation):
        max_inlier = 0
        self.set_scale(0)

        if not with_scale and not with_rotation:
            max_inlier = self.run(1)
            return self.inlier_mask, max_inlier
        elif with_scale and with_rotation:
            vb_inliers = []
            for scale in range(5):
                self.set_scale(scale)
                for rotation_type in range(1, 9):
                    num_inlier = self.run(rotation_type)
                    if num_inlier > max_inlier:
                        vb_inliers = self.inlier_mask
                        max_inlier = num_inlier

            if vb_inliers != []:
                return vb_inliers, max_inlier
            else:
                return self.inlier_mask, max_inlier
        elif with_rotation and not with_scale:
            vb_inliers = []
            for rotation_type in range(1, 9):
                num_inlier = self.run(rotation_type)
                if num_inlier > max_inlier:
                    vb_inliers = self.inlier_mask
                    max_inlier = num_inlier

            if vb_inliers != []:
                return vb_inliers, max_inlier
            else:
                return self.inlier_mask, max_inlier
        else:
            vb_inliers = []
            for scale in range(5):
                self.set_scale(scale)
                num_inlier = self.run(1)
                if num_inlier > max_inlier:
                    vb_inliers = self.inlier_mask
                    max_inlier = num_inlier

            if vb_inliers != []:
                return vb_inliers, max_inlier
            else:
                return self.inlier_mask, max_inlier

    def set_scale(self, scale):
        self.grid_size_right.width = self.grid_size_left.width * self.scale_ratios[scale]
        self.grid_size_right.height = self.grid_size_left.height * self.scale_ratios[scale]
        self.grid_number_right = self.grid_size_right.width * self.grid_size_right.height

        # Initialize the neighbour of right grid
        self.grid_neighbor_right = np.zeros((int(self.grid_number_right), 9))
        self.initialize_neighbours(self.grid_neighbor_right, self.grid_size_right)

    def run(self, rotation_type):
        self.inlier_mask = [False for _ in range(self.matches_number)]

        # Initialize motion statistics
        self.motion_statistics = np.zeros((int(self.grid_number_left), int(self.grid_number_right)))
        self.match_pairs = [[0, 0] for _ in range(self.matches_number)]

        for GridType in range(1, 5):
            self.motion_statistics = np.zeros((int(self.grid_number_left), int(self.grid_number_right)))
            self.cell_pairs = [-1 for _ in range(self.grid_number_left)]
            self.number_of_points_per_cell_left = [0 for _ in range(self.grid_number_left)]

            self.assign_match_pairs(GridType)
            self.verify_cell_pairs(rotation_type)

            # Mark inliers
            for i in range(self.matches_number):
                if self.cell_pairs[int(self.match_pairs[i][0])] == self.match_pairs[i][1]:
                    self.inlier_mask[i] = True

        return sum(self.inlier_mask)

    def assign_match_pairs(self, grid_type):
        for i in range(self.matches_number):
            lp = self.normalized_points1[self.matches[i][0]]
            rp = self.normalized_points2[self.matches[i][1]]
            lgidx = self.match_pairs[i][0] = self.get_grid_index_left(lp, grid_type)

            if grid_type == 1:
                rgidx = self.match_pairs[i][1] = self.get_grid_index_right(rp)
            else:
                rgidx = self.match_pairs[i][1]

            if lgidx < 0 or rgidx < 0:
                continue
            self.motion_statistics[int(lgidx)][int(rgidx)] += 1
            self.number_of_points_per_cell_left[int(lgidx)] += 1

    def get_grid_index_left(self, pt, type_of_grid):
        x = pt[0] * self.grid_size_left.width
        y = pt[1] * self.grid_size_left.height

        if type_of_grid == 2:
            x += 0.5
        elif type_of_grid == 3:
            y += 0.5
        elif type_of_grid == 4:
            x += 0.5
            y += 0.5

        x = math.floor(x)
        y = math.floor(y)

        if x >= self.grid_size_left.width or y >= self.grid_size_left.height:
            return -1
        return x + y * self.grid_size_left.width

    def get_grid_index_right(self, pt):
        x = int(math.floor(pt[0] * self.grid_size_right.width))
        y = int(math.floor(pt[1] * self.grid_size_right.height))
        return x + y * self.grid_size_right.width

    def verify_cell_pairs(self, rotation_type):
        current_rotation_pattern = ROTATION_PATTERNS[rotation_type - 1]

        for i in range(self.grid_number_left):
            if sum(self.motion_statistics[i]) == 0:
                self.cell_pairs[i] = -1
                continue
            max_number = 0
            for j in range(int(self.grid_number_right)):
                value = self.motion_statistics[i]
                if value[j] > max_number:
                    self.cell_pairs[i] = j
                    max_number = value[j]

            idx_grid_rt = self.cell_pairs[i]
            nb9_lt = self.grid_neighbor_left[i]
            nb9_rt = self.grid_neighbor_right[idx_grid_rt]
            score = 0
            thresh = 0
            numpair = 0

            for j in range(9):
                ll = nb9_lt[j]
                rr = nb9_rt[current_rotation_pattern[j] - 1]
                if ll == -1 or rr == -1:
                    continue

                score += self.motion_statistics[int(ll), int(rr)]
                thresh += self.number_of_points_per_cell_left[int(ll)]
                numpair += 1

            thresh = THRESHOLD_FACTOR * math.sqrt(thresh/numpair)
            if score < thresh:
                self.cell_pairs[i] = -2


class gms_module:
    def __init__(self, **args):
              
        for k, v in args.items():
           setattr(self, k, v)
       
        
    def get_id(self):
        return ('gms').lower()

    
    def eval_args(self):
        return "pipe_module.run(pt1, pt2, im1, im2, Hs)"

        
    def eval_out(self):
        return "pt1, pt2, Hs, mask = out_data"
    
    
    def run(self, *args):  
        pt1 = np.ascontiguousarray(args[0].detach().cpu())
        pt2 = np.ascontiguousarray(args[1].detach().cpu())

        sz1 = Image.open(args[2]).size
        sz2 = Image.open(args[3]).size        
                
        l = pt1.shape[0]
        
        if l > 0:    
            kpt1 = [cv2.KeyPoint(pt1[i, 0], pt1[i, 1], 1) for i in range(l)]
            kpt2 = [cv2.KeyPoint(pt2[i, 0], pt2[i, 1], 1) for i in range(l)]
            m12 = [cv2.DMatch(i, i, 0) for i in range(l)]    
    
            gms = GmsMatcher(kpt1, kpt2, m12)
            _, mask = gms.compute_matches(sz1[1], sz1[0], sz2[1], sz2[0]);
        
            pt1 = args[0][mask]
            pt2 = args[1][mask]            
            Hs = args[4][mask]            
        else:
            pt1 = args[0]
            pt2 = args[1]           
            Hs = args[4]   
            mask = []
            
        return pt1, pt2, Hs, mask




class oanet_module:
    def __init__(self, **args):  
        oanet_dir = 'src/OANet'
        model_file = os.path.join(oanet_dir, 'model_best.pth.tar.gz')
        file_to_download = os.path.join(oanet_dir, 'sift-gl3d.tar.gz')    
        if not os.path.isfile(model_file):    
            url = "https://drive.google.com/file/d/1JxXYuuSa_sS-IXbL-VzJs4OVrC9O0bXc/view?usp=drive_link"
            gdown.download(url, file_to_download, fuzzy=True)
    
            with tarfile.open(file_to_download,"r") as tar_ref:
                tar_ref.extract('gl3d/sift-4000/model_best.pth', path=oanet_dir)
            
            shutil.copy(os.path.join(oanet_dir, 'gl3d/sift-4000/model_best.pth'), model_file)
            shutil.rmtree(os.path.join(oanet_dir, 'gl3d'))
            os.remove(file_to_download)
                
        self.lm = OANet_matcher.LearnedMatcher(model_file, inlier_threshold=1, use_ratio=0, use_mutual=0, corr_file=-1)        
        
        for k, v in args.items():
           setattr(self, k, v)
       
        
    def get_id(self):
        return ('oanet').lower()

    
    def eval_args(self):
        return "pipe_module.run(pt1, pt2, Hs)"

        
    def eval_out(self):
        return "pt1, pt2, Hs, mask = out_data"
    
    
    def run(self, *args):  
        pt1 = np.ascontiguousarray(args[0].detach().cpu())
        pt2 = np.ascontiguousarray(args[1].detach().cpu())
                
        l = pt1.shape[0]
        
        if l > 0:                
            _, _, _, _, mask = self.lm.infer(pt1, pt2)
                    
            pt1 = args[0][mask]
            pt2 = args[1][mask]            
            Hs = args[2][mask]            
        else:
            pt1 = args[0]
            pt2 = args[1]           
            Hs = args[2]   
            mask = []
                        
        return pt1, pt2, Hs, mask


def download_megadepth_scannet_data(bench_path ='bench_data'):   
    os.makedirs(os.path.join(bench_path, 'downloads'), exist_ok=True)   

    file_to_download = os.path.join(bench_path, 'downloads', 'megadepth_test_1500.tar')    
    if not os.path.isfile(file_to_download):    
        url = "https://drive.google.com/file/d/12yKniNWebDHRTCwhBNJmxYMPgqYX3Nhv/view?usp=drive_link"
        gdown.download(url, file_to_download, fuzzy=True)
    
    out_dir = os.path.join(bench_path, 'megadepth_test_1500')
    if not os.path.isdir(out_dir):    
        with tarfile.open(file_to_download,"r") as tar_ref:
            tar_ref.extractall(bench_path)
    
    file_to_download = os.path.join(bench_path, 'downloads', 'scannet_test_1500.tar')    
    if not os.path.isfile(file_to_download):    
        url = "https://drive.google.com/file/d/1wtl-mNicxGlXZ-UQJxFnKuWPvvssQBwd/view?usp=drive_link"
        gdown.download(url, file_to_download, fuzzy=True)

    out_dir = os.path.join(bench_path, 'scannet_test_1500')
    if not os.path.isdir(out_dir):    
        with tarfile.open(file_to_download,"r") as tar_ref:
            tar_ref.extractall(bench_path)
    
    return


if __name__ == '__main__':
    # megadepth & scannet
    bench_path = '../miho_megadepth_scannet_bench_data'   
    bench_gt = 'gt_data'
    bench_im = 'imgs'
    bench_file = 'megadepth_scannet'
    bench_res = 'res'
    save_to = os.path.join(bench_path, bench_res, 'res_')

    pipes = [
        #[
        #    superpoint_lightglue_module(nmax_keypoints=4000),
        #    #superpoint_kornia_matcher_module(nmax_keypoints=4000, th=0.97),
        #    #keynetaffnethardnet_kornia_matcher_module(nmax_keypoints=4000, upright=False, th=0.99),
        #    #disk_lightglue_module(nmax_keypoints=4000),
        #    #aliked_lightglue_module(nmax_keypoints=4000),
        #    #loftr_module(pretrained='outdoor'),
        #    #keynetaffnethardnet_module(upright=False, th=0.99),
        #    miho_module(),
        #    pydegensac_module(px_th=3)
        #],

        [
            keynetaffnethardnet_module(upright=False, th=0.99),
            miho_module(),
            pydegensac_module(px_th=3)
        ],

        [
            keynetaffnethardnet_module(upright=False, th=0.99),
            pydegensac_module(px_th=3)
        ],

        [
            keynetaffnethardnet_module(upright=False, th=0.99),
            ncc_module(),
            pydegensac_module(px_th=3)
        ],

        [
            keynetaffnethardnet_module(upright=False, th=0.99),
            miho_module(),
            ncc_module(),
            pydegensac_module(px_th=3)
        ],

        [
            keynetaffnethardnet_module(upright=False, th=0.99),
            gms_module(),
            pydegensac_module(px_th=3)
        ],

        [
            keynetaffnethardnet_module(upright=False, th=0.99),
            gms_module(),
            ncc_module(),
            pydegensac_module(px_th=3)
        ],

        [
            keynetaffnethardnet_module(upright=False, th=0.99),
            oanet_module(),
            pydegensac_module(px_th=3)
        ],

        [
            keynetaffnethardnet_module(upright=False, th=0.99),
            oanet_module(),
            ncc_module(),
            pydegensac_module(px_th=3)
        ]        
    ]
               
    megadepth_data, scannet_data, data_file = bench_init(bench_file=bench_file, bench_path=bench_path, bench_gt=bench_gt)
    megadepth_data, scannet_data = setup_images(megadepth_data, scannet_data, data_file=data_file, bench_path=bench_path, bench_imgs=bench_im)

    for i, pipe in enumerate(pipes):
        print(f"--== Running pipeline {i+1}/{len(pipes)} ==--")
        run_pipe(pipe, megadepth_data, 'megadepth', 'MegaDepth', bench_path=bench_path , bench_im=bench_im, bench_res=bench_res)
        run_pipe(pipe, scannet_data, 'scannet', 'ScanNet', bench_path=bench_path , bench_im=bench_im, bench_res=bench_res)

        eval_pipe(pipe, megadepth_data, 'megadepth', 'MegaDepth', bench_path=bench_path, bench_res='res', essential_th_list=[0.5, 1, 1.5], save_to=save_to + 'megadepth.pbz2', use_scale=True)
        eval_pipe(pipe, scannet_data, 'scannet', 'ScanNet', bench_path=bench_path, bench_res='res', essential_th_list=[0.5, 1, 1.5], save_to=save_to + 'scannet.pbz2', use_scale=False)

    # demo code
    
    img1 = 'data/im1.png'
    img2 = 'data/im2_rot.png'
    # match_file = 'data/matches_rot.mat'

    # img1 = 'data/dc0.png'
    # img2 = 'data/dc2.png'

    # *** NCC / NCC+ ***
    # window radius
    w = 10
    w_big = 15
    # filter outliers by MiHo
    remove_bad=False
    # NCC+ patch angle offset
    angle=[-30, -15, 0, 15, 30]
    # NCC+ patch anisotropic scales
    scale=[[10/14, 1], [10/12, 1], [1, 1], [1, 12/10], [1, 14/10]]

    im1 = Image.open(img1)
    im2 = Image.open(img2)

    # generate matches with kornia, LAF included, check upright!
    upright=False
    with torch.inference_mode():
        detector = K.feature.KeyNetAffNetHardNet(upright=upright, device=device)
        kps1, _ , descs1 = detector(K.io.load_image(img1, K.io.ImageLoadType.GRAY32, device=device).unsqueeze(0))
        kps2, _ , descs2 = detector(K.io.load_image(img2, K.io.ImageLoadType.GRAY32, device=device).unsqueeze(0))
        dists, idxs = K.feature.match_smnn(descs1.squeeze(), descs2.squeeze(), 0.99)        
    kps1 = kps1.squeeze().detach()[idxs[:, 0]].to(device)
    kps2 = kps2.squeeze().detach()[idxs[:, 1]].to(device)

    # import from a match file with only kpts
    #
    # m12 = sio.loadmat(match_file, squeeze_me=True)
    # m12 = m12['matches'][m12['midx'] > 0, :]
    # # m12 = m12['matches']
    # pt1 = torch.tensor(m12[:, :2], dtype=torch.float32, device=device)
    # pt2 = torch.tensor(m12[:, 2:], dtype=torch.float32, device=device)

    params = miho.all_params()
    params['get_avg_hom']['rot_check'] = True
    mihoo = miho(params)

    # miho paramas examples
    #
    # params = miho.all_params()
    # params['go_assign']['method'] = cluster_assign_base
    # params['go_assign']['method_args']['err_th'] = 16
    # mihoo = miho(params)
    #
    # params = mihoo.get_current()
    # params['get_avg_hom']['min_plane_pts'] = 16
    # mihoo.update_params(params)

    mihoo.attach_images(im1, im2)

    # # offset kpt shift, for testing
    # pt1, pt2, Hs_laf = refinement_laf(mihoo.im1, mihoo.im2, data1=kps1, data2=kps2, w=w, img_patches=False)    
    # pt1 = pt1.round()
    # if w_big is None:
    #     ww_big = w * 2
    # else:
    #     ww_big = w_big
    # test_idx = (torch.rand((pt1.shape[0], 2), device=device) * (((ww_big-w) * 2) - 1) - (ww_big-w-1)).round()    
    # pt2 = pt1 + test_idx
    # pt1, pt2, Hs_laf = refinement_laf(mihoo.im1, mihoo.im1, pt1=pt1, pt2=pt2, w=w, img_patches=True)    
    # # pt1__, pt2__, Hs_ncc, val, T = refinement_norm_corr(mihoo.im1, mihoo.im1, pt1, pt2, Hs_laf, w=w, ref_image=['both'], subpix=True, img_patches=True)   
    # pt1__p, pt2__p, Hs_ncc_p, val_p, T_p = refinement_norm_corr_alternate(mihoo.im1, mihoo.im1, pt1, pt2, Hs_laf, w=w, w_big=w_big, ref_image=['both'], subpix=True, img_patches=True)   

    # data formatting 
    pt1, pt2, Hs_laf = refinement_laf(mihoo.im1, mihoo.im2, data1=kps1, data2=kps2, w=w, img_patches=True)    
    # pt1, pt2, Hs_laf = refinement_laf(mihoo.im1, mihoo.im2, pt1=pt1, pt2=pt2, w=w, img_patches=True)    

    ###
    start = time.time()
    
    mihoo.planar_clustering(pt1, pt2)

    end = time.time()
    print("Elapsed = %s (MiHo clustering)" % (end - start))

    # save MiHo
    # import pickle
    #
    # with open('miho.pt', 'wb') as file:
    #     torch.save(mihoo, file)
    #
    # with open('miho.pt', 'rb') as miho_pt:
    #     mihoo = torch.load(miho_pt)
  
    # *** MiHo inlier mask ***
    good_matches = mihoo.Hidx > -1  
  
    start = time.time()
        
    # offset kpt shift, for testing - LAF -> NCC | NCC+
    # pt1__, pt2__, Hs_ncc, val, T = refinement_norm_corr(mihoo.im1, mihoo.im1, pt1, pt2, Hs_laf, w=w, ref_image=['both'], subpix=True, img_patches=True)   
    # pt1__p, pt2__p, Hs_ncc_p, val_p, T_p = refinement_norm_corr_alternate(mihoo.im1, mihoo.im1, pt1, pt2, Hs_laf, w=w, ref_image=['both'], angle=angle, scale=scale, subpix=True, img_patches=True)   
            
    # LAF -> NCC | NCC+
    # pt1__, pt2__, Hs_ncc, val, T = refinement_norm_corr(mihoo.im1, mihoo.im2, pt1, pt2, Hs_laf, w=w, ref_image=['both'], subpix=True, img_patches=True)   
    # pt1__p, pt2__p, Hs_ncc_p, val_p, T_p = refinement_norm_corr_alternate(mihoo.im1, mihoo.im2, pt1, pt2, Hs_laf, w=w, ref_image=['both'], angle=angle, scale=scale, subpix=True, img_patches=True)   
    
    # LAF -> MiHo -> NCC | NCC+   
    pt1_, pt2_, Hs_miho, inliers = refinement_miho(mihoo.im1, mihoo.im2, pt1, pt2, mihoo, Hs_laf, remove_bad=remove_bad, w=w, img_patches=True)        
    pt1__, pt2__, Hs_ncc, val, T = refinement_norm_corr(mihoo.im1, mihoo.im2, pt1_, pt2_, Hs_miho, w=w, ref_image=['both'], subpix=True, img_patches=True)   
    pt1__p, pt2_p_, Hs_ncc_p, val_p, T_p = refinement_norm_corr_alternate(mihoo.im1, mihoo.im2, pt1_, pt2_, Hs_miho, w=w, ref_image=['both'], angle=angle, scale=scale, subpix=True, img_patches=True)   
    
    end = time.time()
    print("Elapsed = %s (NCC refinement)" % (end - start))
    
    mihoo.show_clustering()
