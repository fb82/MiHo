import torch
import numpy as np
from collections import namedtuple
from .oan_cuda import OANet

class LearnedMatcher(object):
    def __init__(self, model_path, inlier_threshold=0, use_ratio=2, use_mutual=2, corr_file=-1, device='cuda'):
        self.default_config = {}
        self.default_config['net_channels'] = 128
        self.default_config['net_depth'] = 12
        self.default_config['clusters'] = 500
        self.default_config['use_ratio'] = use_ratio
        self.default_config['use_mutual'] = use_mutual
        self.default_config['iter_num'] = 1
        self.default_config['inlier_threshold'] = inlier_threshold
        self.corr_file = corr_file
        self.default_config = namedtuple("Config", self.default_config.keys())(*self.default_config.values())

        self.model = OANet(self.default_config)
        self.device = device

        # print('load model from ' + model_path)
        checkpoint = torch.load(model_path,map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def normalize_kpts(self, kpts):
        x_mean = np.mean(kpts, axis=0)
        dist = kpts - x_mean
        meandist = np.sqrt((dist**2).sum(axis=1)).mean()
        scale = np.sqrt(2) / meandist
        T = np.zeros([3,3])
        T[0,0], T[1,1], T[2,2] = scale, scale, 1
        T[0,2], T[1,2] = -scale*x_mean[0], -scale*x_mean[1]
        nkpts = kpts * np.asarray([T[0, 0], T[1, 1]]) + np.array([T[0, 2], T[1, 2]])
        return nkpts

    def infer(self, kp1, kp2, sides=[]):
        with torch.no_grad():
            kp1_ = torch.from_numpy(self.normalize_kpts(kp1).astype(np.float32)).to(self.device)
            kp2_ = torch.from_numpy(self.normalize_kpts(kp2).astype(np.float32)).to(self.device)
            
            corr = torch.hstack((kp1_,kp2_)).unsqueeze(0).unsqueeze(0)
            corr_idx = np.arange(corr.shape[2]).reshape(-1,1).repeat(2, axis=1)
            
            data = {}
            data['xs'] = corr
            # currently supported mode:
            if self.default_config.use_ratio==2 and self.default_config.use_mutual==2:
                data['sides'] = sides
            elif self.default_config.use_ratio==0 and self.default_config.use_mutual==1:
                mutual = sides[0,:,1]>0
                data['xs'] = corr[:,:,mutual,:]
                data['sides'] = []
                corr_idx = corr_idx[mutual,:]
            elif self.default_config.use_ratio==1 and self.default_config.use_mutual==0:
                ratio = sides[0,:,0] < 0.8
                data['xs'] = corr[:,:,ratio,:]
                data['sides'] = []
                corr_idx = corr_idx[ratio,:]
            elif self.default_config.use_ratio==1 and self.default_config.use_mutual==1:
                mask = (sides[0,:,0] < 0.8) & (sides[0,:,1]>0)
                data['xs'] = corr[:,:,mask,:]
                data['sides'] = []
                corr_idx = corr_idx[mask,:]
            elif self.default_config.use_ratio==0 and self.default_config.use_mutual==0:
                data['sides'] = []
            else:
                raise NotImplementedError
            
            y_hat, e_hat = self.model(data)
            y = y_hat[-1][0, :].cpu().numpy()
            mask = y > self.default_config.inlier_threshold
            inlier_idx = np.where(mask)
            matches = corr_idx[inlier_idx[0], :].astype('int32')
            y_ = y[inlier_idx[0]]              
        corr0 = kp1[matches[:, 0]]
        corr1 = kp2[matches[:, 1]]
        return matches, corr0, corr1, y_, mask