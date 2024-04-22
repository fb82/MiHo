import torch
import kornia as K

from ..ncc import refinement_laf

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class keynetaffnethardnet_module_fabio:
    def __init__(self, **args):
        self.upright = False
        self.th = 0.99
        with torch.inference_mode():
            self.detector = K.feature.KeyNetAffNetHardNet(upright=self.upright, device=device)
        for k, v in args.items():
           setattr(self, k, v)

    def get_id(self):
        return ('fabio_keynetaffnethardnet_upright_' + str(self.upright) + '_th_' + str(self.th)).lower()

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
        # pt1, pt2, Hs_laf = refinement_laf(None, None, pt1=kps1, pt2=kps2, img_patches=False) # No refinement LAF!!!

        return pt1, pt2, kps1, kps2, Hs_laf
