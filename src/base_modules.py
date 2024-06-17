import numpy as np
import torch
import kornia as K
import pydegensac
import cv2
from PIL import Image
import poselib
from .ncc import refinement_laf

from .DeDoDe import dedode_detector_L, dedode_descriptor_B, dedode_descriptor_G
from .DeDoDe.matchers.dual_softmax_matcher import DualSoftMaxMatcher


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# from kornia moons
def laf_from_opencv_kpts(kpts, mrSize=6.0, device=torch.device('cpu')):
    N = len(kpts)
    xy = torch.tensor([(x.pt[0], x.pt[1]) for x in kpts ], device=device, dtype=torch.float).view(1, N, 2)
    scales = torch.tensor([(mrSize * x.size) for x in kpts ], device=device, dtype=torch.float).view(1, N, 1, 1)
    angles = torch.tensor([(-x.angle) for x in kpts ], device=device, dtype=torch.float).view(1, N, 1)
    laf = K.feature.laf_from_center_scale_ori(xy, scales, angles).reshape(1, -1, 2, 3)
    return laf.reshape(1, -1, 2, 3)


class keynetaffnethardnet_module:
    def __init__(self, **args):
        self.upright = False
        self.th = 0.99
        self.num_features = 8000
        
        for k, v in args.items():
           setattr(self, k, v)

        with torch.inference_mode():
            self.detector = K.feature.KeyNetAffNetHardNet(num_features=self.num_features, upright=self.upright, device=device)
        
        
    def get_id(self):
        return ('keynetaffnethardnet_upright_' + str(self.upright) + '_th_' + str(self.th) + '_nfeat_' + str(self.num_features)).lower()

    
    def run(self, **args):    
        with torch.inference_mode():
            kps1, _ , descs1 = self.detector(K.io.load_image(args['im1'], K.io.ImageLoadType.GRAY32, device=device).unsqueeze(0))
            kps2, _ , descs2 = self.detector(K.io.load_image(args['im2'], K.io.ImageLoadType.GRAY32, device=device).unsqueeze(0))
            val, idxs = K.feature.match_smnn(descs1.squeeze(), descs2.squeeze(), self.th)        
        
        pt1 = None
        pt2 = None
        kps1 = kps1.squeeze().detach()[idxs[:, 0]].to(device)
        kps2 = kps2.squeeze().detach()[idxs[:, 1]].to(device)
        
        pt1, pt2, Hs_laf = refinement_laf(None, None, data1=kps1, data2=kps2, img_patches=False)    
    
        return {'pt1': pt1, 'pt2': pt2, 'kp1': kps1, 'kp2': kps2, 'Hs': Hs_laf, 'val': val}


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

        
    def run(self, **args):  
        pt1 = args['pt1']
        pt2 = args['pt2']
        Hs = args['Hs']
        
        if torch.is_tensor(pt1):
            pt1 = np.ascontiguousarray(pt1.detach().cpu())
            pt2 = np.ascontiguousarray(pt2.detach().cpu())

        F = None
        mask = []
            
        if self.mode == 'fundamental_matrix':           
            if (pt1.shape)[0] > 7:                        
                F, mask = pydegensac.findFundamentalMatrix(pt1, pt2, px_th=self.px_th, conf=self.conf, max_iters=self.max_iters)
        
            pt1 = args['pt1'][mask]
            pt2 = args['pt2'][mask]     
            Hs = args['Hs'][mask]
        else:
            if (pt1.shape)[0] > 3:                        
                F, mask = pydegensac.findHomography(pt1, pt2, px_th=self.px_th, conf=self.conf, max_iters=self.max_iters)
                
            pt1 = args['pt1'][mask]
            pt2 = args['pt2'][mask]     
            Hs = args['Hs'][mask]            
            
        return {'pt1': pt1, 'pt2': pt2, 'Hs': Hs, 'F': F, 'mask': mask}


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

        
    def run(self, **args):  
        pt1 = args['pt1']
        pt2 = args['pt2']
        Hs = args['Hs']
        
        if torch.is_tensor(pt1):
            pt1 = np.ascontiguousarray(pt1.detach().cpu())
            pt2 = np.ascontiguousarray(pt2.detach().cpu())

        F = None
        mask = []
            
        if self.mode == 'fundamental_matrix':           
            if (pt1.shape)[0] > 7:  
                try:                     
                    F, mask = cv2.findFundamentalMat(pt1, pt2, cv2.USAC_MAGSAC, self.px_th, self.conf, self.max_iters)
                except:
                    try:
                        idx = np.random.permutation(pt1.shape[0])
                        jdx = np.argsort(idx)
                        F, mask = cv2.findFundamentalMat(pt1[idx], pt2[idx], cv2.USAC_MAGSAC, self.px_th, self.conf, self.max_iters)
                        mask = mask[jdx]
                    except:
                        F, mask = pydegensac.findFundamentalMatrix(pt1, pt2, px_th=self.px_th, conf=self.conf, max_iters=self.max_iters)
                        
            if not isinstance(mask, np.ndarray):
                mask = []
            else:
                if len(mask.shape) > 1: mask = mask.squeeze(1) > 0
        
            pt1 = args['pt1'][mask]
            pt2 = args['pt2'][mask]     
            Hs = args['Hs'][mask]
        else:
            if (pt1.shape)[0] > 3:    
                try:                    
                    F, mask = cv2.findHomography(pt1, pt2, cv2.USAC_MAGSAC, self.px_th, self.conf, self.max_iters)
                except:
                    try:
                        idx = np.random.permutation(pt1.shape[0])
                        jdx = np.argsort(idx)
                        F, mask = cv2.findHomography(pt1[idx], pt2[idx], cv2.USAC_MAGSAC, self.px_th, self.conf, self.max_iters)
                        mask = mask[jdx]
                    except:                    
                        F, mask = pydegensac.findHomography(pt1, pt2, px_th=self.px_th, conf=self.conf, max_iters=self.max_iters)

            if not isinstance(mask, np.ndarray):
                mask = []
            else:
                if len(mask.shape) > 1: mask = mask.squeeze(1) > 0
                
            pt1 = args['pt1'][mask]
            pt2 = args['pt2'][mask]     
            Hs = args['Hs'][mask]            
            
        return {'pt1': pt1, 'pt2': pt2, 'Hs': Hs, 'F': F, 'mask': mask}


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

        
    def run(self, **args):  
        pt1 = args['pt1']
        pt2 = args['pt2']
        Hs = args['Hs']
        
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

            pt1 = args['pt1'][mask]
            pt2 = args['pt2'][mask]     
            Hs = args['Hs'][mask]
        else:
            if (pt1.shape)[0] > 3:                        
                F, info = poselib.estimate_homography(pt1, pt2, params, {})
                mask = info['inliers']
                
            pt1 = args['pt1'][mask]
            pt2 = args['pt2'][mask]     
            Hs = args['Hs'][mask]            
            
        return {'pt1': pt1, 'pt2': pt2, 'Hs': Hs, 'F': F, 'mask': mask}


class sift_module:
    def __init__(self, **args):
        self.upright = False
        self.th = 0.99
        self.num_features = 8000
        self.rootsift = True
        
        for k, v in args.items():
           setattr(self, k, v)

        with torch.inference_mode():
            self.detector = cv2.SIFT_create(self.num_features, contrastThreshold=-10000, edgeThreshold=10000)


    def get_id(self):
        return ('sift_upright_' + str(self.upright) + '_rootsift_' + str(self.rootsift) + '_th_' + str(self.th) + '_nfeat_' + str(self.num_features)).lower()


    def run(self, **args):    
        
        im1 = cv2.imread(args['im1'], cv2.IMREAD_GRAYSCALE)
        kps1 = self.detector.detect(im1, None)

        if self.upright:
            idx = np.unique(np.asarray([[k.pt[0], k.pt[1]] for k in kps1]), axis=0, return_index=True)[1]
            kps1 = [kps1[ii] for ii in idx]
            for ii in range(len(kps1)):
                kps1[ii].angle = 0           
        kps1, descs1 = self.detector.compute(im1, kps1)

        if self.rootsift:
            descs1 /= descs1.sum(axis=1, keepdims=True) + 1e-8
            descs1 = np.sqrt(descs1)

        im2 = cv2.imread(args['im2'], cv2.IMREAD_GRAYSCALE)
        kps2 = self.detector.detect(im2, None)

        if self.upright:
            idx = np.unique(np.asarray([[k.pt[0], k.pt[1]] for k in kps2]), axis=0, return_index=True)[1]
            kps2 = [kps2[ii] for ii in idx]
            for ii in range(len(kps2)):
                kps2[ii].angle = 0           
        kps2, descs2 = self.detector.compute(im2, kps2)

        if self.rootsift:
            descs2 /= descs2.sum(axis=1, keepdims=True) + 1e-8
            descs2 = np.sqrt(descs2)

        with torch.inference_mode():            
            val, idxs = K.feature.match_smnn(torch.from_numpy(descs1).cuda(), torch.from_numpy(descs2).cuda(), self.th)

        pt1 = None
        pt2 = None
        kps1 = laf_from_opencv_kpts(kps1, device=device)
        kps2 = laf_from_opencv_kpts(kps2, device=device)
                
        kps1 = kps1.squeeze().detach()[idxs[:, 0]].to(device)
        kps2 = kps2.squeeze().detach()[idxs[:, 1]].to(device)

        pt1, pt2, Hs_laf = refinement_laf(None, None, data1=kps1, data2=kps2, img_patches=False)    

        return {'pt1': pt1, 'pt2': pt2, 'kp1': kps1, 'kp2': kps2, 'Hs': Hs_laf, 'val': val}
    

class dedodev2_module:
    def __init__(self, **args):
        self.threshold = 0.01
        self.num_features = 8000
        
        for k, v in args.items():
           setattr(self, k, v)

        with torch.inference_mode():
            self.detector = dedode_detector_L(weights = None)
            self.descriptor = dedode_descriptor_B(weights = None)
            self.matcher = DualSoftMaxMatcher()
        
        
    def get_id(self):
        return ('dedodev2_th_' + str(self.threshold) + '_nfeat_' + str(self.num_features)).lower()

    
    def run(self, **args):
        im1 = Image.open(args['im1'])
        im2 = Image.open(args['im2'])
        W1, H1 = im1.size
        W2, H2 = im2.size

        with torch.inference_mode():
            detections1 = self.detector.detect_from_path(args['im1'], num_keypoints = self.num_features, H = H1, W = W1)
            kps1, p1 = detections1["keypoints"], detections1["confidence"]
            descs1 = self.descriptor.describe_keypoints_from_path(args['im1'], kps1, H = H1, W = W1)["descriptions"]

            detections2 = self.detector.detect_from_path(args['im2'], num_keypoints = self.num_features, H = H2, W = W2)
            kps2, p2 = detections2["keypoints"], detections2["confidence"]
            descs2 = self.descriptor.describe_keypoints_from_path(args['im2'], kps2, H = H2, W = W2)["descriptions"]
            
            # val, idxs = K.feature.match_smnn(descs1.squeeze(), descs2.squeeze(), self.th)

            matches1, matches2, batch_ids = self.matcher.match(kps1, descs1, 
                                                               kps2, descs2,
                                                               P_A = p1, P_B = p2,
                                                               normalize = True, inv_temp=20, threshold = self.threshold)
            
           
        pt1 = None
        pt2 = None
        # kps1 = kps1.squeeze()[idxs[:, 0],:]
        # kps2 = kps2.squeeze()[idxs[:, 1],:]
        kps1, kps2 = self.matcher.to_pixel_coords(matches1, matches2, H1, W1, H2, W2)

        pt1, pt2, Hs_laf = refinement_laf(None, None, pt1=kps1, pt2=kps2, img_patches=False) # No refinement LAF!!!
        # print('************* dedodev2 kps1 shape', kps1.shape)
        # print('************* dedodev2 pt1 shape', pt1.shape)
   
        # return {'pt1': pt1, 'pt2': pt2, 'kp1': kps1, 'kp2': kps2, 'Hs': Hs_laf, 'val': val}
        return {'pt1': pt1, 'pt2': pt2, 'kp1': kps1, 'kp2': kps2, 'Hs': Hs_laf}