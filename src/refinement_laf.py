import torch
from src.laf2homo import laf2homo

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
