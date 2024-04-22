import numpy as np
import torch
import kornia as K
import pydegensac
from .ncc import refinement_laf


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
