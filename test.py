import scipy.io as sio
import numpy as np
import cv2
import os
import os.path as osp
import warnings
import argparse
import subprocess
from pathlib import Path
import shutil

from PIL import Image
from miho_buffered_rot_patch_pytorch import *

def relative_pose_error(R_gt, t_gt, R, t, ignore_gt_t_thr=0.0):
    # angle error between 2 vectors
    # t_gt = T_0to1[:3, 3]
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
    t_err = np.minimum(t_err, 180 - t_err)  # handle E ambiguity
    if np.linalg.norm(t_gt) < ignore_gt_t_thr:  # pure rotation is challenging
        t_err = 0

    # angle error between 2 rotation matrices
    # R_gt = T_0to1[:3, :3]
    cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # handle numercial errors
    R_err = np.rad2deg(np.abs(np.arccos(cos)))

    return t_err, R_err


def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None
    # normalize keypoints
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    # normalize ransac threshold
    ransac_thr = thresh / np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])

    # compute pose with cv2
    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=ransac_thr, prob=conf, method=cv2.RANSAC)
    if E is None:
        print("\nE is None while trying to recover pose.\n")
        return None

    # recover pose from E
    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(
            _E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            ret = (R, t[:, 0], mask.ravel() > 0)
            best_num_inliers = n

    return ret

def error_auc(errors, thr):
    errors = [0] + sorted(errors)
    recall = list(np.linspace(0, 1, len(errors)))

    last_index = np.searchsorted(errors, thr)
    y = recall[:last_index] + [recall[last_index-1]]
    x = errors[:last_index] + [thr]
    return np.trapz(y, x) / thr    


if __name__ == '__main__':

    #matchers = ['disk', 'dkm', 'hz', 'hz_upright', 'keynet', 'keynet_upright', 'loftr', 'loftr_se',
    #            'matchformer', 'quadtreeattention', 'rootsift', 'rootsift_upright', 'slime', 'slime_upright', 'superglue']
    
    angular_thresholds = [5, 10, 20]
    pixel_thr = 0.5 # ??? correct???
    sac_th = [0, 5, 3, 2, 1]
    data = sio.loadmat('img_test/scannet.mat', simplify_cells=True,
                       matlab_compatible=True, chars_as_strings=True)


    im1 = ["".join(data['im1'][i]).strip()
           for i in range(data['im1'].shape[0])]
    im2 = ["".join(data['im2'][i]).strip()
           for i in range(data['im2'].shape[0])]
    K1 = data['K1']
    K2 = data['K2']
    R_gt = data['R']
    t_gt = data['T']


    pipeline = "superpoint+lightglue"
    working_dir = Path("./img_test/scannet_test_1500")

    # Store all the image pairs per scene
    pairs_per_scene = {}
    for pair_idx in range(len(im1)):
        scene1, _, image1 = im1[pair_idx].split("/", 2)
        scene2, _, image2 = im2[pair_idx].split("/", 2)
        if scene1 not in list(pairs_per_scene.keys()):
            pairs_per_scene[scene1] = []
        assert(scene1== scene2)
        pairs_per_scene[scene1].append((image1, image2, pair_idx))
    


    for sac_th_idx in range(len(sac_th)):
        th_str = 'th_'+str(sac_th[sac_th_idx])
        #print(matcher+' '+str(pixel_thr)+' '+th_str)
        data[th_str] = {}
        data[th_str]['R_errs'] = []
        data[th_str]['t_errs'] = []
        data[th_str]['inliers'] = []

        #for scene in list(pairs_per_scene.keys()):
        for scene in ["scene0707_00", "scene0708_00"]:
            with open(working_dir / scene / "pairs.txt", 'w') as pair_file:
                for pair in pairs_per_scene[scene]:
                    pair_file.write(f'{pair[0]} {pair[1]}\n')

            shutil.copytree(working_dir / scene / 'color', working_dir / scene / 'images', dirs_exist_ok=True)

            # STILL TO BE DISABLED RANSAC
            # Extract and match features
            p = subprocess.Popen([
                "python", 
                "./thirdparty/deep-image-matching/main.py", 
                "--dir", f"{working_dir / scene}", 
                "--pipeline", 
                f"{pipeline}", 
                "--strategy", 
                "custom_pairs", 
                "--force", 
                "--skip_reconstruction",
                "--pair_file",
                f"{working_dir / scene}/pairs.txt",
                ])
            p.wait()

            # Save matches in x1 y1 x2 y2 format
            p = subprocess.Popen([
                "python", 
                "./thirdparty/deep-image-matching/scripts/export_from_database.py", 
                f"{working_dir / scene}/results_{pipeline}_custom_pairs_quality_high/database.db", 
                "x1y1x2y2",
                ])
            p.wait()

            for pair in pairs_per_scene[scene]:
                img0, img1, pair_idx = pair[0], pair[1], pair[2]
                match_file1 = f"{working_dir / scene}/results_{pipeline}_custom_pairs_quality_high/matches/{Path(img0).stem}_{Path(img1).stem}.txt"
                match_file2 = f"{working_dir / scene}/results_{pipeline}_custom_pairs_quality_high/matches/{Path(img1).stem}_{Path(img0).stem}.txt"

                #p = subprocess.Popen([
                #    "python",
                #    "./miho_buffered_rot_patch_pytorch.py",
                #    f"{working_dir / scene}/images/{img0}",
                #    f"{working_dir / scene}/images/{img1}",
                #    f"{working_dir / scene}/results_{pipeline}_custom_pairs_quality_high/matches/{Path(img0).stem}_{Path(img1).stem}.txt",
                #    ])
                #p.wait()

                im1 = Image.open(working_dir / scene / "images" / img0)            
                im2 = Image.open(working_dir / scene / "images" / img1)
                
                if os.path.exists(match_file1):
                    m12 = np.loadtxt(match_file1)
                if os.path.exists(match_file2):
                    m21 = np.loadtxt(match_file2)
                    m12 = m21[:, [2, 3, 0, 1]]
                
                if os.path.exists(match_file1) or os.path.exists(match_file2):

                    params = miho.all_params()
                    params['get_avg_hom']['rot_check'] = True
                    mihoo = miho(params)
                    mihoo.planar_clustering(m12[:, :2], m12[:, 2:])
                    # Ransac
                    # Comprimere i file txt!
                    mihoo.attach_images(im1, im2)

                    w = 15  

                    pt1_, pt2_, Hs_ = refinement_init(mihoo.im1, mihoo.im2, mihoo.Hidx, mihoo.Hs, mihoo.pt1, mihoo.pt2, mihoo, w=w, img_patches=True)        
                    pt1__, pt2__, Hs__, val, T = refinement_norm_corr(mihoo.im1, mihoo.im2, Hs_, pt1_, pt2_, w=w, ref_image=['both'], subpix=True, img_patches=True)

                    if (pt1__.shape[0] > 8):
                        Rt = estimate_pose(
                            pt1__.numpy(), pt2__.numpy(), K1[pair_idx], K2[pair_idx], pixel_thr)
                    else:
                        Rt = None
                
                else:
                    Rt = None

                if Rt is None:
                    data[th_str]['R_errs'].append(np.inf)
                    data[th_str]['t_errs'].append(np.inf)
                    data[th_str]['inliers'].append(
                        np.array([]).astype('bool'))
                else:
                    R, t, inliers = Rt
                    t_err, R_err = relative_pose_error(
                        R_gt[pair_idx], t_gt[pair_idx], R, t, ignore_gt_t_thr=0.0)
                    data[th_str]['R_errs'].append(R_err)
                    data[th_str]['t_errs'].append(t_err)
                    data[th_str]['inliers'].append(inliers)


        aux = np.stack(
            ([data[th_str]['R_errs'], data[th_str]['t_errs']]), axis=1)
        max_Rt_err = np.max(aux, axis=1)

        tmp = np.concatenate((aux, np.expand_dims(
            np.max(aux, axis=1), axis=1)), axis=1)

        for a in angular_thresholds:
            auc_R = error_auc(np.squeeze(data[th_str]['R_errs']), a)
            auc_t = error_auc(np.squeeze(data[th_str]['t_errs']), a)
            auc_max_Rt = error_auc(np.squeeze(max_Rt_err), a)
            data[th_str]['pose_error_auc_@' +
                         str(a)] = np.asarray([auc_R, auc_t, auc_max_Rt])

            data[th_str]['pose_error_acc_@' +
                         str(a)] = np.sum(tmp < a, axis=0)/np.shape(tmp)[0]

    sio.savemat(osp.join('table_res', pipeline +'_'+ "scannet" +
                '_pose_error_'+str(pixel_thr)+'.mat'), data, do_compression=True)