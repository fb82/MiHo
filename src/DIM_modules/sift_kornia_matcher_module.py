import torch
import deep_image_matching as dim

from ..ncc import refinement_laf
from .load_image import load_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class sift_kornia_matcher_module:
    def __init__(self, **args):
        self.n_features = 8000
        self.th = 0.99
        self.contrastThreshold = 0.04
        for k, v in args.items():
           setattr(self, k, v)

        self.grayscale = True
        self.as_float = None
        params = {
            "dir": "./thirdparty/deep-image-matching/assets/example_cyprus", # placeholder
            "pipeline": "sift+kornia_matcher",
            "strategy": "bruteforce",
            "quality": "high",
            "tiling": "none",
            "skip_reconstruction": False,
            "force": True,
            "camera_options": "./thirdparty/deep-image-matching/config/cameras.yaml", # placeholder
            "openmvg": None,
            "verbose": False,
        }
        self.config = dim.Config(params)

        self.config.extractor['n_features'] = self.n_features
        self.config.extractor['contrastThreshold'] = self.contrastThreshold
        self.config.matcher['match_mode'] = 'smnn'
        self.config.matcher['th'] = self.th
        print('extractor', self.config.extractor)
        print('matcher', self.config.matcher)

        self.extractor =  dim.extractors.SIFTExtractor(self.config)
        self.matcher = dim.matchers.KorniaMatcher(self.config)

    def get_id(self):
        return ('sift_kornia_matcher_' + '_th_' + str(self.th)).lower()

    def eval_args(self):
        return "pipe_module.run(im1, im2)"

    def eval_out(self):
        return "pt1, pt2, kps1, kps2, Hs = out_data"               

    def run(self, *args):

        with torch.inference_mode():
            image1 = load_image(args[0], self.grayscale, self.as_float)
            image2 = load_image(args[1], self.grayscale, self.as_float)
            feats1 = self.extractor._extract(image1)
            feats2 = self.extractor._extract(image2)
            matches = self.matcher._match_pairs(feats1, feats2)
            # print(feats1['keypoints'].shape, feats2['keypoints'].shape, matches.shape)

        kps1 = torch.tensor(feats1['keypoints'][matches[:,0],:]).to(device)
        kps2 = torch.tensor(feats2['keypoints'][matches[:,1],:]).to(device)
        pt1, pt2, Hs_laf = refinement_laf(None, None, pt1=kps1, pt2=kps2, img_patches=False) # No refinement LAF!!!
        #print(pt1.shape)
        #print(Hs_laf.shape)
        #with open('/home/threedom/Desktop/prova.txt', 'w') as out:
        #    for i in range(pt1.shape[0]):
        #        out.write(f"{pt1[i,0]} {pt1[i,1]} {pt2[i,0]} {pt2[i,1]}\n")
        #    print(args[0])
        #    print(args[1])
        #    quit()
    
        return pt1, pt2, kps1, kps2, Hs_laf
