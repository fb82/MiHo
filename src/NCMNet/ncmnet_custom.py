import os
import numpy as np
import torch
import gdown
import zipfile
from PIL import Image

from .code.ncmnet import NCMNet as Model
from .code.config import get_config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_grad_enabled(False)


class ncmnet_module:    
    def __init__(self, **args):
        ncmnet_dir = os.path.split(__file__)[0]
        model_dir = os.path.join(ncmnet_dir, 'NCMNet_models')

        file_to_download = os.path.join(ncmnet_dir, 'ncmnet_weights.zip')    
        if not os.path.isfile(file_to_download):    
            url = "https://drive.google.com/file/d/1UZA8ypbwu1jozWJg7vvSSqrQY9HfsbIP/view?usp=drive_link"
            gdown.download(url, file_to_download, fuzzy=True)

        file_to_unzip = file_to_download
        if not os.path.isdir(model_dir):    
            with zipfile.ZipFile(file_to_unzip,"r") as zip_ref:
                zip_ref.extractall(path=ncmnet_dir)

        self.config, unparsed = get_config()
        self.obj_geod_th = self.config.obj_geod_th

        self.model = Model(self.config)
        checkpoint = torch.load(os.path.join(model_dir, 'model_best.pth'))
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval().to(device)

        for k, v in args.items():
           setattr(self, k, v)


    def get_id(self):
        return ('ncmnet_obj_geod_th_' + str(self.obj_geod_th)).lower()

    
    def norm_kp(self, cx, cy, fx, fy, kp):
        # New kp
        kp = (kp - np.array([[cx, cy]])) / np.asarray([[fx, fy]])
        return kp


    def run(self, **args):
        sz1 = Image.open(args['im1']).size
        sz2 = Image.open(args['im2']).size

        pt1 = np.ascontiguousarray(args['pt1'].detach().cpu())
        pt2 = np.ascontiguousarray(args['pt2'].detach().cpu())

        l = pt1.shape[0]

        if l > 11:
            cx1 = (sz1[0] - 1.0) * 0.5
            cy1 = (sz1[1] - 1.0) * 0.5
            f1 = max(sz1[1] - 1.0, sz1[0] - 1.0)
            
            cx2 = (sz2[0] - 1.0) * 0.5
            cy2 = (sz2[1] - 1.0) * 0.5
            f2 = max(sz2[1] - 1.0, sz2[0] - 1.0)

            x1 = self.norm_kp(cx1, cy1, f1, f1, pt1)
            x2 = self.norm_kp(cx2, cy2, f2, f2, pt2)

            xs = np.concatenate([x1, x2], axis=1).reshape(1,-1,4)
            ys = np.ones(xs.shape[1]).reshape(-1,1)

            xs = torch.from_numpy(xs).float().unsqueeze(0).to(device)
            ys = torch.from_numpy(ys).float().unsqueeze(0).to(device)

            _, _, _, y_hat = self.model(xs, ys.squeeze(-1))

            mask = y_hat.squeeze(0) < self.obj_geod_th

            pt1 = args['pt1'][mask]
            pt2 = args['pt2'][mask]            
            Hs = args['Hs'][mask]
        else:
            pt1 = args['pt1']
            pt2 = args['pt2']           
            Hs = args['Hs']   
            mask = []

        return {'pt1': pt1, 'pt2': pt2, 'Hs': Hs, 'mask': mask}          


