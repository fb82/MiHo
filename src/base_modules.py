import numpy as np
import torch
import kornia as K
import pydegensac
import cv2
import poselib
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
        return "pt1, pt2, kps1, kps2, Hs, val = out_data"               


    def run(self, *args):    
        with torch.inference_mode():
            kps1, _ , descs1 = self.detector(K.io.load_image(args[0], K.io.ImageLoadType.GRAY32, device=device).unsqueeze(0))
            kps2, _ , descs2 = self.detector(K.io.load_image(args[1], K.io.ImageLoadType.GRAY32, device=device).unsqueeze(0))
            val, idxs = K.feature.match_smnn(descs1.squeeze(), descs2.squeeze(), self.th)        
        
        pt1 = None
        pt2 = None
        kps1 = kps1.squeeze().detach()[idxs[:, 0]].to(device)
        kps2 = kps2.squeeze().detach()[idxs[:, 1]].to(device)
        
        pt1, pt2, Hs_laf = refinement_laf(None, None, data1=kps1, data2=kps2, img_patches=False)    
    
        return pt1, pt2, kps1, kps2, Hs_laf, val


class pydegensac_module:
    def __init__(self, **args):
        self.px_th = 3
        self.conf = 0.9999
        self.max_iters = 100000
        self.mode = 'fundamental_matrix'
              
        for k, v in args.items():
           setattr(self, k, v)
       
        
    def get_id(self):
        return ('pydegensac_' + self.mode + '_th_' + str(self.px_th) + '_conf_' + str(self.conf) + '_max_iters_' + str(self.max_iters)).lower()

    
    def eval_args(self):
        return "pipe_module.run(pt1, pt2, Hs)"

        
    def eval_out(self):
        return "pt1, pt2, Hs, F, mask = out_data"
    
    
    def run(self, *args):  
        pt1 = args[0]
        pt2 = args[1]
        Hs = args[2]
        
        if torch.is_tensor(pt1):
            pt1 = np.ascontiguousarray(pt1.detach().cpu())
            pt2 = np.ascontiguousarray(pt2.detach().cpu())

        F = None
        mask = []
            
        if self.mode == 'fundamental_matrix':           
            if (pt1.shape)[0] > 7:                        
                F, mask = pydegensac.findFundamentalMatrix(pt1, pt2, px_th=self.px_th, conf=self.conf, max_iters=self.max_iters)
        
            pt1 = args[0][mask]
            pt2 = args[1][mask]     
            Hs = args[2][mask]
        else:
            if (pt1.shape)[0] > 3:                        
                F, mask = pydegensac.findHomography(pt1, pt2, px_th=self.px_th, conf=self.conf, max_iters=self.max_iters)
                
            pt1 = args[0][mask]
            pt2 = args[1][mask]     
            Hs = args[2][mask]            
            
        return pt1, pt2, Hs, F, mask


class magsac_module:
    def __init__(self, **args):
        self.px_th = 3
        self.conf = 0.9999
        self.max_iters = 100000
        self.mode = 'fundamental_matrix'
                              
        for k, v in args.items():
           setattr(self, k, v)
       
        
    def get_id(self):
        return ('opencv_magsac_' + self.mode + '_th_' + str(self.px_th) + '_conf_' + str(self.conf) + '_max_iters_' + str(self.max_iters)).lower()

    
    def eval_args(self):
        return "pipe_module.run(pt1, pt2, Hs)"

        
    def eval_out(self):
        return "pt1, pt2, Hs, F, mask = out_data"
    
    
    def run(self, *args):  
        pt1 = args[0]
        pt2 = args[1]
        Hs = args[2]
        
        if torch.is_tensor(pt1):
            pt1 = np.ascontiguousarray(pt1.detach().cpu())
            pt2 = np.ascontiguousarray(pt2.detach().cpu())

        F = None
        mask = []
            
        if self.mode == 'fundamental_matrix':           
            if (pt1.shape)[0] > 7:                        
                F, mask = cv2.findFundamentalMat(pt1, pt2, cv2.USAC_MAGSAC, self.px_th, self.conf, self.max_iters)
                
            mask = mask.squeeze(1) > 0
        
            pt1 = args[0][mask]
            pt2 = args[1][mask]     
            Hs = args[2][mask]
        else:
            if (pt1.shape)[0] > 3:                        
                F, mask = cv2.findHomography(pt1, pt2, cv2.USAC_MAGSAC, self.px_th, self.conf, self.max_iters)

            mask = mask.squeeze(1) > 0
                
            pt1 = args[0][mask]
            pt2 = args[1][mask]     
            Hs = args[2][mask]            
            
        return pt1, pt2, Hs, F, mask


class poselib_module:
    def __init__(self, **args):
        self.px_th = 3
        self.conf = 0.9999
        self.max_iters = 100000
        self.min_iters = 1000                    
        self.mode = 'fundamental_matrix'
          
        for k, v in args.items():
           setattr(self, k, v)
       
        
    def get_id(self):
        return ('poselib_' + self.mode + '_th_' + str(self.px_th) + '_conf_' + str(self.conf) + '_max_iters_' + str(self.max_iters) + '_min_iters_' + str(self.max_iters)).lower()

    
    def eval_args(self):
        return "pipe_module.run(pt1, pt2, Hs)"

        
    def eval_out(self):
        return "pt1, pt2, Hs, F, mask = out_data"
    
    
    def run(self, *args):  
        pt1 = args[0]
        pt2 = args[1]
        Hs = args[2]
        
        if torch.is_tensor(pt1):
            pt1 = np.ascontiguousarray(pt1.detach().cpu())
            pt2 = np.ascontiguousarray(pt2.detach().cpu())

        F = None
        mask = []
        
        params = {         
            'max_iterations' : self.max_iters,
            'min_iterations' : self.min_iters,
            'success_prob' : self.conf,
            'max_epipolar_error' : self.px_th
            }
            
        if self.mode == 'fundamental_matrix':           
            if (pt1.shape)[0] > 7:  

                F, info = poselib.estimate_fundamental(pt1, pt2, params, {})
                mask = info['inliers']

            pt1 = args[0][mask]
            pt2 = args[1][mask]     
            Hs = args[2][mask]
        else:
            if (pt1.shape)[0] > 3:                        
                F, info = poselib.estimate_homography(pt1, pt2, params, {})
                mask = info['inliers']

            mask = mask.squeeze(1) > 0
                
            pt1 = args[0][mask]
            pt2 = args[1][mask]     
            Hs = args[2][mask]            
            
        return pt1, pt2, Hs, F, mask