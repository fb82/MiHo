import torch
import numpy as np
import pydegensac
from deep_image_matching.utils.geometric_verification import geometric_verification

class magsac_module:
    def __init__(self, **args):
        self.px_th = 3
        self.conf = 0.9999
        self.max_iters = 100000
              
        for k, v in args.items():
           setattr(self, k, v)
       
        
    def get_id(self):
        return ('magsac_th_' + str(self.px_th) + '_conf_' + str(self.conf) + '_max_iters_' + str(self.max_iters)).lower()

    
    def eval_args(self):
        return "pipe_module.run(pt1, pt2, Hs)"

        
    def eval_out(self):
        return "pt1, pt2, Hs, F, mask = out_data"
    
    
    def run(self, **args):
        pt1 = args['pt1']
        pt2 = args['pt2']
        Hs = args['Hs']
        
        if torch.is_tensor(pt1):
            pt1 = pt1.detach().cpu()
            pt2 = pt1.detach().cpu()
            
        if (np.ascontiguousarray(pt1).shape)[0] > 7:                      
            F, mask = geometric_verification(
                kpts0=np.ascontiguousarray(pt1),
                kpts1=np.ascontiguousarray(pt2),
                method="MAGSAC",
                threshold=self.px_th,
                confidence=self.conf,
                max_iters=self.max_iters,
                quiet=False,
            )
    
            pt1 = args['pt1'][mask]
            pt2 = args['pt2'][mask]     
            Hs = args['Hs'][mask]
        else:            
            F = None
            mask = []
        
        return {'pt1': pt1, 'pt2': pt2, 'Hs': Hs, 'F': F, 'mask': mask}