import numpy as np

try:
    import pyprogressivex
    pyprogressivex_off = False
except:
    pyprogressivex_off = True
    import warnings
    warnings.warn("cannot load pyprogressivex - Progressive-X module will return no matches")

from PIL import Image

class progressivex_module:
    def __init__(self, **args):
        self.threshold = 7.0
        self.conf = 0.5
        self.spatial_coherence_weight = 0.0
        self.neighborhood_ball_radius = 200.0
        self.maximum_tanimoto_similarity = 0.4
        self.max_iters = 1000
        self.minimum_point_number = 10
        self.maximum_model_number = -1
        self.sampler_id = 3
                
        for k, v in args.items():
           setattr(self, k, v)
        
        
    def get_id(self):
        return ('progressivex_th_' + str(self.threshold)).lower()


    if pyprogressivex_off:
        def run(self, **args):                
            mask = []
            pt1 = args['pt1']
            pt2 = args['pt2']     
            Hs = args['Hs']
                       
            return {'pt1': pt1, 'pt2': pt2, 'Hs': Hs, 'mask': mask}
    else:
        def run(self, **args):        
            k1 = np.ascontiguousarray(args['pt1'].detach().cpu())
            k2 = np.ascontiguousarray(args['pt2'].detach().cpu())
        
            if k1.shape[0] <= 4:
                mask = []
                pt1 = args['pt1']
                pt2 = args['pt2']     
                Hs = args['Hs']
           
                return {'pt1': pt1, 'pt2': pt2, 'Hs': Hs, 'mask': mask}
        
            correspondences = np.ascontiguousarray(np.concatenate([k1, k2], axis=1))      
        
            sz1 = Image.open(args['im1']).size
            sz2 = Image.open(args['im2']).size        

            H, labels = pyprogressivex.findHomographies(
                np.ascontiguousarray(correspondences), 
                sz1[1], sz1[0], 
                sz2[1], sz2[0],
                threshold = self.threshold,
                conf = self.conf,
                spatial_coherence_weight = self.spatial_coherence_weight,
                neighborhood_ball_radius = self.neighborhood_ball_radius,
                maximum_tanimoto_similarity = self.maximum_tanimoto_similarity,
                max_iters = self.max_iters,
                minimum_point_number = self.minimum_point_number,
                maximum_model_number = self.maximum_model_number,
                sampler_id = self.sampler_id,
                do_logging = False)    
        
            H = np.asarray(H)
            labels = np.asarray(labels)
        
            n_models = int(H.shape[0] / 3)
        
            if n_models > 0:        
                mask = labels < np.max(n_models)
            else:
                mask = labels == 0
                    
            pt1 = args['pt1'][mask]
            pt2 = args['pt2'][mask]     
            Hs = args['Hs'][mask]
            
            return {'pt1': pt1, 'pt2': pt2, 'Hs': Hs, 'mask': mask}
