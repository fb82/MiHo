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
        kps1, kps2 = self.matcher.to_pixel_coords(matches1, matches2, H1, W1, H2, W2)

        pt1, pt2, Hs_laf = refinement_laf(None, None, pt1=kps1, pt2=kps2, img_patches=False) # No refinement LAF!!!

        return {'pt1': pt1, 'pt2': pt2, 'Hs': Hs_laf}
