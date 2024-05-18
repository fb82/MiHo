from PIL import Image
import cv2
import numpy as np
import torch
import gdown
import tarfile
import zipfile
import os
import warnings
import _pickle as cPickle
import bz2
import shutil
import src.ncc as ncc

from matplotlib import colormaps
import matplotlib.pyplot as plt
import src.plot.viz2d as viz
import src.plot.utils as viz_utils


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pipe_color = ['red', 'blue', 'lime', 'fuchsia', 'yellow']


from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


# Pickle a file and then compress it into a file with extension 
def compressed_pickle(title, data, add_ext=False):
    if add_ext:
        ext = '.pbz2'
    else:
        ext = ''
        
    with bz2.BZ2File(title + ext, 'w') as f: 
        cPickle.dump(data, f)
        

# Load any compressed pickle file
def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data


def progress_bar(text=''):
    return Progress(
        TextColumn(text + " [progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    )


def megadepth_1500_list(ppath='bench_data/gt_data/megadepth'):
    npz_list = [i for i in os.listdir(ppath) if (os.path.splitext(i)[1] == '.npz')]

    data = {'im1': [], 'im2': [], 'K1': [], 'K2': [], 'T': [], 'R': []}
    for name in npz_list:
        scene_info = np.load(os.path.join(ppath, name), allow_pickle=True)
    
        # Collect pairs
        for pair_info in scene_info['pair_infos']:
            (id1, id2), overlap, _ = pair_info
            im1 = scene_info['image_paths'][id1].replace('Undistorted_SfM/', '')
            im2 = scene_info['image_paths'][id2].replace('Undistorted_SfM/', '')                        
            K1 = scene_info['intrinsics'][id1].astype(np.float32)
            K2 = scene_info['intrinsics'][id2].astype(np.float32)
    
            # Compute relative pose
            T1 = scene_info['poses'][id1]
            T2 = scene_info['poses'][id2]
            T12 = np.matmul(T2, np.linalg.inv(T1))
    
            data['im1'].append(im1)
            data['im2'].append(im2)
            data['K1'].append(K1)
            data['K2'].append(K2)
            data['T'].append(T12[:3, 3])
            data['R'].append(T12[:3, :3])   
    return data


def scannet_1500_list(ppath='bench_data/gt_data/scannet'):
    intrinsic_path = 'intrinsics.npz'
    npz_path = 'test.npz'

    data = np.load(os.path.join(ppath, npz_path))
    data_names = data['name']
    intrinsics = dict(np.load(os.path.join(ppath, intrinsic_path)))
    rel_pose = data['rel_pose']
    
    data = {'im1': [], 'im2': [], 'K1': [], 'K2': [], 'T': [], 'R': []}
    
    for idx in range(data_names.shape[0]):
        scene_name, scene_sub_name, stem_name_0, stem_name_1 = data_names[idx]
        scene_name = f'scene{scene_name:04d}_{scene_sub_name:02d}'
    
        # read the grayscale image which will be resized to (1, 480, 640)
        im1 = os.path.join(scene_name, 'color', f'{stem_name_0}.jpg')
        im2 = os.path.join(scene_name, 'color', f'{stem_name_1}.jpg')
        
        # read the intrinsic of depthmap
        K1 = intrinsics[scene_name]
        K2 = intrinsics[scene_name]
    
        # pose    
        T12 = np.concatenate((rel_pose[idx],np.asarray([0, 0, 0, 1.0]))).reshape(4,4)
        
        data['im1'].append(im1)
        data['im2'].append(im2)
        data['K1'].append(K1)
        data['K2'].append(K2)  
        data['T'].append(T12[:3, 3])
        data['R'].append(T12[:3, :3])     
    return data


def bench_init(bench_file='megadepth_scannet', bench_path='bench_data', bench_gt='gt_data'):
    download_megadepth_scannet_data(bench_path)
        
    data_file = os.path.join(bench_path, 'megadepth_scannet' + '.pbz2')
    if not os.path.isfile(data_file):      
        megadepth_data = megadepth_1500_list(os.path.join(bench_path, bench_gt, 'megadepth'))
        scannet_data = scannet_1500_list(os.path.join(bench_path, bench_gt, 'scannet'))
        compressed_pickle(data_file, (megadepth_data, scannet_data))
    else:
        megadepth_data, scannet_data = decompress_pickle(data_file)
    
    return megadepth_data, scannet_data, data_file


def resize_megadepth(im, res_path='imgs/megadepth', bench_path='bench_data', force=False):
    mod_im = os.path.join(bench_path, res_path, os.path.splitext(im)[0] + '.png')
    ori_im= os.path.join(bench_path, 'megadepth_test_1500/Undistorted_SfM', im)

    if os.path.isfile(mod_im) and not force:
        # PIL does not load image, so it's faster to get only image size
        return np.asarray(Image.open(ori_im).size) / np.asarray(Image.open(mod_im).size) 
        # return np.array(cv2.imread(ori_im).shape)[:2][::-1] / np.array(cv2.imread(mod_im).shape)[:2][::-1]

    img = cv2.imread(ori_im)
    sz_ori = np.array(img.shape)[:2][::-1]
    sz_max = float(max(sz_ori))

    if sz_max > 1200:
        cf = 1200 / sz_max                    
        sz_new = np.ceil(sz_ori * cf).astype(int) 
        img = cv2.resize(img, tuple(sz_new), interpolation=cv2.INTER_LANCZOS4)
        sc = sz_ori/sz_new
        os.makedirs(os.path.dirname(mod_im), exist_ok=True)                 
        cv2.imwrite(mod_im, img)
        return sc
    else:
        os.makedirs(os.path.dirname(mod_im), exist_ok=True)                 
        cv2.imwrite(mod_im, img)
        return np.array([1., 1.])


def resize_scannet(im, res_path='imgs/scannet', bench_path='bench_data', force=False):
    mod_im = os.path.join(bench_path, res_path, os.path.splitext(im)[0] + '.png')
    ori_im= os.path.join(bench_path, 'scannet_test_1500', im)

    if os.path.isfile(mod_im) and not force:
        # PIL does not load image, so it's faster to get only image size
        return np.asarray(Image.open(ori_im).size) / np.asarray(Image.open(mod_im).size) 
        # return np.array(cv2.imread(ori_im).shape)[:2][::-1] / np.array(cv2.imread(mod_im).shape)[:2][::-1]

    img = cv2.imread(ori_im)
    sz_ori = np.array(img.shape)[:2][::-1]

    sz_new = np.array([640, 480])
    img = cv2.resize(img, tuple(sz_new), interpolation=cv2.INTER_LANCZOS4)
    sc = sz_ori/sz_new
    os.makedirs(os.path.dirname(mod_im), exist_ok=True)                 
    cv2.imwrite(mod_im, img)
    return sc


def setup_images(megadepth_data, scannet_data, data_file='bench_data/megadepth_scannet.pbz2', bench_path='bench_data', bench_imgs='imgs'):
    if not ('im_pair_scale' in megadepth_data.keys()):        
        n = len(megadepth_data['im1'])
        im_pair_scale = np.zeros((n, 2, 2))
        res_path = os.path.join(bench_imgs, 'megadepth')
        with progress_bar('MegaDepth - image setup completion') as p:
            for i in p.track(range(n)):
                im_pair_scale[i, 0] = resize_megadepth(megadepth_data['im1'][i], res_path, bench_path)
                im_pair_scale[i, 1] = resize_megadepth(megadepth_data['im2'][i], res_path, bench_path)
        megadepth_data['im_pair_scale'] = im_pair_scale

        n = len(scannet_data['im1'])
        im_pair_scale = np.zeros((n, 2, 2))
        res_path = os.path.join(bench_imgs, 'scannet')
        with progress_bar('ScanNet - image setup completion') as p:
            for i in p.track(range(n)):
                im_pair_scale[i, 0] = resize_scannet(scannet_data['im1'][i], res_path, bench_path)
                im_pair_scale[i, 1] = resize_scannet(scannet_data['im2'][i], res_path, bench_path)
        scannet_data['im_pair_scale'] = im_pair_scale
        
        compressed_pickle(data_file, (megadepth_data, scannet_data))
 
    return megadepth_data, scannet_data


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


def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999, max_iters=10000):
    if len(kpts0) < 5:
        return None
    # normalize keypoints
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    # normalize ransac threshold
    ransac_thr = thresh / np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])

    # compute pose with cv2
    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=ransac_thr, prob=conf, method=cv2.RANSAC, maxIters=max_iters)
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


def run_pipe(pipe, dataset_data, dataset_name, bar_name, bench_path='bench_data' , bench_im='imgs', bench_res='res', force=False):

    n = len(dataset_data['im1'])
    im_path = os.path.join(bench_im, dataset_name)        
    with progress_bar(bar_name + ' - pipeline completion') as p:
        for i in p.track(range(n)):
            im1 = os.path.join(bench_path, im_path, os.path.splitext(dataset_data['im1'][i])[0]) + '.png'
            im2 = os.path.join(bench_path, im_path, os.path.splitext(dataset_data['im2'][i])[0]) + '.png'

            pipe_data = {'im1': im1, 'im2': im2}
            pipe_name_base = os.path.join(bench_path, bench_res, dataset_name)
            for pipe_module in pipe:
                pipe_name_base = os.path.join(pipe_name_base, pipe_module.get_id())
                pipe_f = os.path.join(pipe_name_base, 'base', str(i) + '.pbz2')
            
                if os.path.isfile(pipe_f) and not force:
                    out_data = decompress_pickle(pipe_f)
                else:                      
                    out_data = pipe_module.run(**pipe_data)
                    os.makedirs(os.path.dirname(pipe_f), exist_ok=True)                 
                    compressed_pickle(pipe_f, out_data)
                    
                for k, v in out_data.items(): pipe_data[k] = v


# original benchmark metric
def eval_pipe_essential(pipe, dataset_data,  dataset_name, bar_name, bench_path='bench_data', bench_res='res', essential_th_list=[0.5, 1, 1.5], save_to='res_essential.pbz2', force=False, use_scale=False):
    warnings.filterwarnings("ignore", category=UserWarning)

    angular_thresholds = [5, 10, 20]

    K1 = dataset_data['K1']
    K2 = dataset_data['K2']
    R_gt = dataset_data['R']
    t_gt = dataset_data['T']

    if os.path.isfile(save_to):
        eval_data = decompress_pickle(save_to)
    else:
        eval_data = {}
        
    for essential_th in essential_th_list:            
        n = len(dataset_data['im1'])
        
        pipe_name_base = os.path.join(bench_path, bench_res, dataset_name)
        pipe_name_base_small = ''
        for pipe_module in pipe:
            pipe_name_base = os.path.join(pipe_name_base, pipe_module.get_id())
            pipe_name_base_small = os.path.join(pipe_name_base_small, pipe_module.get_id())

            print(bar_name + ' evaluation with RANSAC essential matrix threshold ' + str(essential_th) + ' px')
            print('Pipeline: ' + pipe_name_base_small)

            if ((pipe_name_base + '_essential_th_list_' + str(essential_th)) in eval_data.keys()) and not force:
                eval_data_ = eval_data[pipe_name_base + '_essential_th_list_' + str(essential_th)]                
                for a in angular_thresholds:
                    print(f"mAA@{str(a).ljust(2,' ')} (E) : {eval_data_['pose_error_e_auc_' + str(a)]}")                                    
                continue
                    
            eval_data_ = {}
            eval_data_['R_errs_e'] = []
            eval_data_['t_errs_e'] = []
            eval_data_['inliers_e'] = []
                
            with progress_bar('Completion') as p:
                for i in p.track(range(n)):            
                    pipe_f = os.path.join(pipe_name_base, 'base', str(i) + '.pbz2')
                                        
                    if os.path.isfile(pipe_f):
                        out_data = decompress_pickle(pipe_f)
    
                        pts1 = out_data['pt1']
                        pts2 = out_data['pt2']
                                                
                        if torch.is_tensor(pts1):
                            pts1 = pts1.detach().cpu().numpy()
                            pts2 = pts2.detach().cpu().numpy()
                        
                        if use_scale:
                            scales = dataset_data['im_pair_scale'][i]
                        
                            pts1 = pts1 * scales[0]
                            pts2 = pts2 * scales[1]
                            
                        nn = pts1.shape[0]
                                                                            
                        if nn < 5:
                            Rt = None
                        else:                            
                            Rt = estimate_pose(pts1, pts2, K1[i], K2[i], essential_th)                                                        
                    else:
                        Rt = None
        
                    if Rt is None:
                        eval_data_['R_errs_e'].append(np.inf)
                        eval_data_['t_errs_e'].append(np.inf)
                        eval_data_['inliers_e'].append(np.array([]).astype('bool'))
                    else:
                        R, t, inliers = Rt
                        t_err, R_err = relative_pose_error(R_gt[i], t_gt[i], R, t, ignore_gt_t_thr=0.0)
                        eval_data_['R_errs_e'].append(R_err)
                        eval_data_['t_errs_e'].append(t_err)
                        eval_data_['inliers_e'].append(inliers)
        
                aux = np.stack(([eval_data_['R_errs_e'], eval_data_['t_errs_e']]), axis=1)
                max_Rt_err = np.max(aux, axis=1)
        
                tmp = np.concatenate((aux, np.expand_dims(max_Rt_err, axis=1)), axis=1)
        
                for a in angular_thresholds:       
                    auc_R = error_auc(np.squeeze(eval_data_['R_errs_e']), a)
                    auc_t = error_auc(np.squeeze(eval_data_['t_errs_e']), a)
                    auc_max_Rt = error_auc(np.squeeze(max_Rt_err), a)
                    eval_data_['pose_error_e_auc_' + str(a)] = np.asarray([auc_R, auc_t, auc_max_Rt])
                    eval_data_['pose_error_e_acc_' + str(a)] = np.sum(tmp < a, axis=0)/np.shape(tmp)[0]

                    print(f"mAA@{str(a).ljust(2,' ')} (E) : {eval_data_['pose_error_e_auc_' + str(a)]}")

            eval_data[pipe_name_base + '_essential_th_list_' + str(essential_th)] = eval_data_
            compressed_pickle(save_to, eval_data)


def eval_pipe_fundamental(pipe, dataset_data,  dataset_name, bar_name, bench_path='bench_data', bench_res='res', save_to='res_fundamental.pbz2', force=False, use_scale=False, err_th_list=list(range(1,16))):
    warnings.filterwarnings("ignore", category=UserWarning)

    angular_thresholds = [5, 10, 20]

    K1 = dataset_data['K1']
    K2 = dataset_data['K2']
    R_gt = dataset_data['R']
    t_gt = dataset_data['T']

    if os.path.isfile(save_to):
        eval_data = decompress_pickle(save_to)
    else:
        eval_data = {}
        
    n = len(dataset_data['im1'])
    
    pipe_name_base = os.path.join(bench_path, bench_res, dataset_name)
    pipe_name_base_small = ''
    pipe_name_root = os.path.join(pipe_name_base, pipe[0].get_id())

    for pipe_module in pipe:
        pipe_name_base = os.path.join(pipe_name_base, pipe_module.get_id())
        pipe_name_base_small = os.path.join(pipe_name_base_small, pipe_module.get_id())

        print('Pipeline: ' + pipe_name_base_small)

        if (pipe_name_base in eval_data.keys()) and not force:
            eval_data_ = eval_data[pipe_name_base]                
            for a in angular_thresholds:
                print(f"mAA@{str(a).ljust(2,' ')} (F) : {eval_data_['pose_error_f_auc_' + str(a)]}")

            print(f"precision(F) : {eval_data_['epi_global_prec_f']}")
            print(f"recall (F) : {eval_data_['epi_global_recall_f']}")
                                                
            continue
                
        eval_data_ = {}
        eval_data_['R_errs_f'] = []
        eval_data_['t_errs_f'] = []
        eval_data_['epi_max_error_f'] = []
        eval_data_['epi_inliers_f'] = []
        eval_data_['epi_prec_f'] = []
        eval_data_['epi_recall_f'] = []
            
        with progress_bar('Completion') as p:
            for i in p.track(range(n)):            
                pipe_f = os.path.join(pipe_name_base, 'base', str(i) + '.pbz2')

                epi_max_err =[]
                inl_sum = []
                avg_prec = 0
                avg_recall = 0

                if os.path.isfile(pipe_f):
                    out_data = decompress_pickle(pipe_f)

                    pts1 = out_data['pt1']
                    pts2 = out_data['pt2']
                                            
                    if torch.is_tensor(pts1):
                        pts1 = pts1.detach().cpu().numpy()
                        pts2 = pts2.detach().cpu().numpy()

                    if use_scale:
                        scales = dataset_data['im_pair_scale'][i]
                    else:
                        scales = np.asarray([[1.0, 1.0], [1.0, 1.0]])
                    
                    pts1 = pts1 * scales[0]
                    pts2 = pts2 * scales[1]
                        
                    nn = pts1.shape[0]
                                                
                    if nn < 8:
                        Rt_ = None
                    else:
                        F = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)[0]
                        E = K2[i].T @ F @ K1[i]
                        Rt_ = cv2.decomposeEssentialMat(E)

                    if nn > 0:
                        F_gt = torch.tensor(K2[i].T, device=device, dtype=torch.float64).inverse() @ \
                               torch.tensor([[0, -t_gt[i][2], t_gt[i][1]],
                                            [t_gt[i][2], 0, -t_gt[i][0]],
                                            [-t_gt[i][1], t_gt[i][0], 0]], device=device) @ \
                               torch.tensor(R_gt[i], device=device) @ \
                               torch.tensor(K1[i], device=device, dtype=torch.float64).inverse()
                        F_gt = F_gt / F_gt.sum()

                        pt1_ = torch.vstack((torch.tensor(pts1.T, device=device), torch.ones((1, nn), device=device)))
                        pt2_ = torch.vstack((torch.tensor(pts2.T, device=device), torch.ones((1, nn), device=device)))
                        
                        l1_ = F_gt @ pt1_
                        d1 = pt2_.permute(1,0).unsqueeze(-2).bmm(l1_.permute(1,0).unsqueeze(-1)).squeeze().abs() / (l1_[:2]**2).sum(0).sqrt()
                        
                        l2_ = F_gt.T @ pt2_
                        d2 = pt1_.permute(1,0).unsqueeze(-2).bmm(l2_.permute(1,0).unsqueeze(-1)).squeeze().abs() / (l2_[:2]**2).sum(0).sqrt()
                        
                        epi_max_err = torch.maximum(d1, d2);                                
                        inl_sum = (epi_max_err.unsqueeze(-1) < torch.tensor(err_th_list, device=device).unsqueeze(0)).sum(dim=0).type(torch.int)
                        avg_prec = inl_sum.type(torch.double).mean()/nn
                                                
                        if pipe_name_base==pipe_name_root:
                            recall_normalizer = torch.tensor(inl_sum, device=device)
                        else:
                            recall_normalizer = torch.tensor(eval_data[pipe_name_root]['epi_inliers_f'][i], device=device)
                        avg_recall = inl_sum.type(torch.double) / recall_normalizer
                        avg_recall[~avg_recall.isfinite()] = 0
                        avg_recall = avg_recall.mean()
                        
                        epi_max_err = epi_max_err.detach().cpu().numpy()
                        inl_sum = inl_sum.detach().cpu().numpy()
                        avg_prec = avg_prec.item()
                        avg_recall = avg_recall.item()
                else:
                    Rt_ = None
                    
                    
                if Rt_ is None:
                    eval_data_['R_errs_f'].append(np.inf)
                    eval_data_['t_errs_f'].append(np.inf)
                else:
                    R_a, t_a, = Rt_[0], Rt_[2].squeeze()
                    t_err_a, R_err_a = relative_pose_error(R_gt[i], t_gt[i], R_a, t_a, ignore_gt_t_thr=0.0)

                    R_b, t_b, = Rt_[1], Rt_[2].squeeze()
                    t_err_b, R_err_b = relative_pose_error(R_gt[i], t_gt[i], R_b, t_b, ignore_gt_t_thr=0.0)

                    if max(R_err_a, t_err_a) < max(R_err_b, t_err_b):
                        R_err, t_err = R_err_a, t_err_b
                    else:
                        R_err, t_err = R_err_b, t_err_b

                    eval_data_['R_errs_f'].append(R_err)
                    eval_data_['t_errs_f'].append(t_err)
                    
                eval_data_['epi_max_error_f'].append(epi_max_err)  
                eval_data_['epi_inliers_f'].append(inl_sum)
                eval_data_['epi_prec_f'].append(avg_prec)                           
                eval_data_['epi_recall_f'].append(avg_recall)
                    
            aux = np.stack(([eval_data_['R_errs_f'], eval_data_['t_errs_f']]), axis=1)
            max_Rt_err = np.max(aux, axis=1)
        
            tmp = np.concatenate((aux, np.expand_dims(max_Rt_err, axis=1)), axis=1)
    
            for a in angular_thresholds:       
                auc_R = error_auc(np.squeeze(eval_data_['R_errs_f']), a)
                auc_t = error_auc(np.squeeze(eval_data_['t_errs_f']), a)
                auc_max_Rt = error_auc(np.squeeze(max_Rt_err), a)
                eval_data_['pose_error_f_auc_' + str(a)] = np.asarray([auc_R, auc_t, auc_max_Rt])
                eval_data_['pose_error_f_acc_' + str(a)] = np.sum(tmp < a, axis=0)/np.shape(tmp)[0]

                print(f"mAA@{str(a).ljust(2,' ')} (F) : {eval_data_['pose_error_f_auc_' + str(a)]}")
            
            eval_data_['epi_global_prec_f'] = torch.tensor(eval_data_['epi_prec_f'], device=device).mean().item()
            eval_data_['epi_global_recall_f'] = torch.tensor(eval_data_['epi_recall_f'], device=device).mean().item()
        
            print(f"precision (F) : {eval_data_['epi_global_prec_f']}")
            print(f"recall (F) : {eval_data_['epi_global_recall_f']}")

            eval_data[pipe_name_base] = eval_data_
            compressed_pickle(save_to, eval_data)


def download_megadepth_scannet_data(bench_path ='bench_data'):   
    os.makedirs(os.path.join(bench_path, 'downloads'), exist_ok=True)   

    file_to_download = os.path.join(bench_path, 'downloads', 'megadepth_scannet_gt_data.zip')    
    if not os.path.isfile(file_to_download):    
        url = "https://drive.google.com/file/d/1GtpHBN6RLcgSW5RPPyqYLyfbn7ex360G/view?usp=drive_link"
        gdown.download(url, file_to_download, fuzzy=True)

    out_dir = os.path.join(bench_path, 'gt_data')
    if not os.path.isdir(out_dir):    
        with zipfile.ZipFile(file_to_download, "r") as zip_ref:
            zip_ref.extractall(bench_path)    

    file_to_download = os.path.join(bench_path, 'downloads', 'megadepth_test_1500.tar.gz')    
    if not os.path.isfile(file_to_download):    
        url = "https://drive.google.com/file/d/1Vwk_htrvWmw5AtJRmHw10ldK57ckgZ3r/view?usp=drive_link"
        gdown.download(url, file_to_download, fuzzy=True)
    
    out_dir = os.path.join(bench_path, 'megadepth_test_1500')
    if not os.path.isdir(out_dir):    
        with tarfile.open(file_to_download, "r") as tar_ref:
            tar_ref.extractall(bench_path)
    
    file_to_download = os.path.join(bench_path, 'downloads', 'scannet_test_1500.tar.gz')    
    if not os.path.isfile(file_to_download):    
        url = "https://drive.google.com/file/d/13KCCdC1k3IIZ4I3e4xJoVMvDA84Wo-AG/view?usp=drive_link"
        gdown.download(url, file_to_download, fuzzy=True)

    out_dir = os.path.join(bench_path, 'scannet_test_1500')
    if not os.path.isdir(out_dir):    
        with tarfile.open(file_to_download, "r") as tar_ref:
            tar_ref.extractall(bench_path)
    
    return


def show_pipe(pipe, dataset_data, dataset_name, bar_name, bench_path='bench_data' , bench_im='imgs', bench_res='res', bench_plot='plot', force=False):

    n = len(dataset_data['im1'])
    im_path = os.path.join(bench_im, dataset_name)    
    fig = plt.figure()    
    
    with progress_bar(bar_name + ' - pipeline completion') as p:
        for i in p.track(range(n)):
            im1 = os.path.join(bench_path, im_path, os.path.splitext(dataset_data['im1'][i])[0]) + '.png'
            im2 = os.path.join(bench_path, im_path, os.path.splitext(dataset_data['im2'][i])[0]) + '.png'
                        
            pair_data = []            
            pipe_name_base = os.path.join(bench_path, bench_res, dataset_name)
            pipe_img_save = os.path.join(bench_path, bench_plot, dataset_name)
            for pipe_module in pipe:
                pipe_name_base = os.path.join(pipe_name_base, pipe_module.get_id())
                pipe_img_save = os.path.join(pipe_img_save, pipe_module.get_id())

                pipe_f = os.path.join(pipe_name_base, 'base', str(i) + '.pbz2')            
                pair_data.append(decompress_pickle(pipe_f))
            
            os.makedirs(pipe_img_save, exist_ok=True)
            pipe_img_save = os.path.join(pipe_img_save, str(i) + '.png')
            if os.path.isfile(pipe_img_save) and not force:
                continue
            
            img1 = viz_utils.load_image(im1)
            img2 = viz_utils.load_image(im2)
            fig, axes = viz.plot_images([img1, img2], fig_num=fig.number)              
            
            pt1 = pair_data[0]['pt1']
            pt2 = pair_data[0]['pt2']
            l = pt1.shape[0]
            
            idx = torch.arange(l, device=device)                            
            clr = 0
            for j in range(1, len(pair_data)):
                if 'mask' in pair_data[j].keys():
                    mask = pair_data[j]['mask']
                    if isinstance(mask, list): mask = np.asarray(mask, dtype=bool)
                    if mask.shape[0] > 0:
                        mpt1 = pt1[idx[~mask]]
                        mpt2 = pt2[idx[~mask]]
                        viz.plot_matches(mpt1, mpt2, color=pipe_color[clr], lw=0.2, ps=6, a=0.3, axes=axes, fig_num=fig.number)
                        idx = idx[mask]
                    clr = clr + 1
            mpt1 = pt1[idx]
            mpt2 = pt2[idx]
            viz.plot_matches(mpt1, mpt2, color=pipe_color[clr], lw=0.2, ps=6, a=0.3, axes=axes, fig_num=fig.number)

            viz.save_plot(pipe_img_save, fig)
            viz.clear_plot(fig)
    plt.close(fig)


planar_scenes = [
    'aerial',
    'aerialrot',
    'apprendices'
    'artisans',
    'bark'
    'barkrot'
    'birdwoman',
    'boat',
    'boatrot',
    'calder',
    'chatnoir',
    'colors',
    'DD',
    'dogman',
    'duckhunt',
    'floor',
#   'graf',       # actually there are two planes :(
    'home',
    'marilyn',
    'mario',
    'maskedman',
    'op',
    'oprot',
    'outside',
    'posters',
    'screen',
    'spidey',
    'sunseason',
    'there',
    'wall',
    'zero'
    ]


def planar_bench_setup(planar_scenes=planar_scenes, max_imgs=6, bench_path='bench_data', bench_imgs='imgs', out_path='planar_out', bench_plot='plot', save_to='planar.pbz2'):        
    os.makedirs(os.path.join(bench_path, 'downloads'), exist_ok=True)

    file_to_download = os.path.join(bench_path, 'downloads', 'planar_data.zip')    
    if not os.path.isfile(file_to_download):    
        url = "https://drive.google.com/file/d/1XkP4RR9KKbCV94heI5JWlue2l32H0TNs/view?usp=drive_link"
        gdown.download(url, file_to_download, fuzzy=True)

    out_dir = os.path.join(bench_path, 'planar')
    if not os.path.isdir(out_dir):    
        with zipfile.ZipFile(file_to_download, "r") as zip_ref:
            zip_ref.extractall(out_dir)        
    
    in_path = out_dir
    out_path = os.path.join(bench_path, bench_imgs, 'planar')
    check_path = os.path.join(bench_path, bench_plot, 'planar_check')
    save_to_full = os.path.join(bench_path, save_to)

    os.makedirs(out_path, exist_ok=True)
    os.makedirs(check_path, exist_ok=True)

    im1 = []
    im2 = []
    sz1 = []
    sz2 = []
    H = []
    H_inv = []
    im1_mask = []
    im2_mask = []
    im1_full_mask = []
    im2_full_mask = []

    im1_use_mask = []
    im2_use_mask = []
    im_pair_scale = []
    
    for scene in planar_scenes:
        img1 = scene + '1.png'
        img1_mask = 'mask_' + scene + '1.png'
        img1_mask_bad = 'mask_bad_' + scene + '1.png'

        for i in range(2, max_imgs+1):
            img2 = scene + str(i) + '.png'
            img2_mask = 'mask_' + scene  + str(i) + '.png'
            img2_mask_bad = 'mask_bad_' + scene + str(i) + '.png'

            H12 = scene + '_H1' + str(i) + '.txt'
                        
            im1s = os.path.join(in_path, img1)
            im1s_mask = os.path.join(in_path, img1_mask)
            im1s_mask_bad = os.path.join(in_path, img1_mask_bad)
            im2s = os.path.join(in_path, img2)
            im2s_mask = os.path.join(in_path, img2_mask)
            im2s_mask_bad = os.path.join(in_path, img2_mask_bad)
            H12s = os.path.join(in_path, H12)
            
            if not os.path.isfile(H12s):
                continue
            
            im1d = os.path.join(out_path, img1)
            im2d = os.path.join(out_path, img2)
 
            shutil.copyfile(im1s, im1d)
            shutil.copyfile(im2s, im2d)
            
            H_ = np.loadtxt(H12s)
            H_inv_ = np.linalg.inv(H_)
            
            im1.append(img1)
            im2.append(img2)
            H.append(H_)
            H_inv.append(H_inv_)
            
            im1i=cv2.imread(im1s)
            im2i=cv2.imread(im2s)
            
            sz1.append(np.array(im1i.shape)[:2][::-1])
            sz2.append(np.array(im2i.shape)[:2][::-1])
                        
            im2i_ = cv2.warpPerspective(im1i,H_,(im2i.shape[1],im2i.shape[0]))
            im1i_ = cv2.warpPerspective(im2i,H_inv_,(im1i.shape[1],im1i.shape[0]))
            
            im_pair_scale.append(np.ones((2, 2)))
            
            im1_mask_ = torch.ones((sz1[-1][1],sz1[-1][0]), device=device, dtype=torch.bool)
            im2_mask_ = torch.ones((sz2[-1][1],sz2[-1][0]), device=device, dtype=torch.bool)
            im1_use_mask_ = False
            im2_use_mask_ = False
            
            if os.path.isfile(im1s_mask):
                im1_mask_ = torch.tensor((cv2.imread(im1s_mask, cv2.IMREAD_GRAYSCALE)==0), device=device)
                im1_use_mask_ = True
 
            if os.path.isfile(im2s_mask):
                im2_mask_ = torch.tensor((cv2.imread(im2s_mask, cv2.IMREAD_GRAYSCALE)==0), device=device)
                im2_use_mask_ = True

            if os.path.isfile(im1s_mask_bad):
                im1_mask_ = torch.tensor((cv2.imread(im1s_mask_bad, cv2.IMREAD_GRAYSCALE)==0), device=device)

            if os.path.isfile(im2s_mask_bad):
                im2_mask_ = torch.tensor((cv2.imread(im1s_mask_bad, cv2.IMREAD_GRAYSCALE)==0), device=device)

            im1_mask.append(im1_mask_.detach().cpu().numpy())
            im2_mask.append(im2_mask_.detach().cpu().numpy())

            im1_use_mask.append(im1_use_mask_)
            im2_use_mask.append(im2_use_mask_)

            im1_full_mask_ = refine_mask(im1_mask_, im2_mask_, sz1[-1], sz2[-1], H_)
            im2_full_mask_ = refine_mask(im2_mask_, im1_full_mask_, sz2[-1], sz1[-1], H_inv_)
            
            im1_full_mask.append(im1_full_mask_.detach().cpu().numpy())
            im2_full_mask.append(im2_full_mask_.detach().cpu().numpy())
            
            iname = os.path.splitext(img1)[0] + '_' + os.path.splitext(img2)[0]
                        
            cv2.imwrite(os.path.join(check_path, iname + '_1a.png'), im1i)
            cv2.imwrite(os.path.join(check_path, iname + '_1b.png'), im1i_)
            cv2.imwrite(os.path.join(check_path, iname + '_2a.png'), im2i)
            cv2.imwrite(os.path.join(check_path, iname + '_2b.png'), im2i_)
    
    H = np.asarray(H)
    H_inv = np.asarray(H_inv)

    sz1 = np.asarray(sz1)
    sz2 = np.asarray(sz2)

    im1_use_mask = np.asarray(im1_use_mask)
    im2_use_mask = np.asarray(im2_use_mask)

    im_pair_scale = np.asarray(im_pair_scale)
    
    data = {'im1': im1, 'im2': im2, 'H': H, 'H_inv': H_inv,
            'im1_mask': im1_mask, 'im2_mask': im2_mask, 'sz1': sz1, 'sz2': sz2,
            'im1_use_mask': im1_use_mask, 'im2_use_mask': im2_use_mask,
            'im1_full_mask': im1_full_mask, 'im2_full_mask': im2_full_mask}

    compressed_pickle(save_to_full, data)
    return data, save_to_full


def  refine_mask(im1_mask, im2_mask, sz1, sz2, H):
                
    x = torch.arange(sz1[0], device=device).unsqueeze(0).repeat(sz1[1],1).unsqueeze(-1)
    y = torch.arange(sz1[1], device=device).unsqueeze(1).repeat(1,sz1[0]).unsqueeze(-1)
    z = torch.ones((sz1[1],sz1[0]), device=device).unsqueeze(-1)
    pt1 = torch.cat((x, y, z), dim=-1).reshape((-1, 3))
    pt2_ = torch.tensor(H, device=device, dtype=torch.float) @ pt1.permute(1,0)
    pt2_ = (pt2_[:2] / pt2_[-1].unsqueeze(0)).reshape(2, sz1[1], -1).round()
    mask1_reproj = torch.isfinite(pt2_).all(dim=0) & (pt2_ >= 0).all(dim=0) & (pt2_[0] < sz2[0]) & (pt2_[1] < sz2[1])
    mask1_reproj = mask1_reproj & im1_mask
    masked_pt2 = pt2_[:, mask1_reproj]
    idx = masked_pt2[1] * sz2[0] + masked_pt2[0]
    mask1_reproj[mask1_reproj.clone()] = im2_mask.flatten()[idx.type(torch.long)]
    
    return mask1_reproj


def eval_pipe_homography(pipe, dataset_data,  dataset_name, bar_name, bench_path='bench_data', bench_res='res', save_to='res_homography.pbz2', force=False, use_scale=False, rad=15, err_th_list=list(range(1,16)), bench_plot='plot', save_acc_images=True):
    warnings.filterwarnings("ignore", category=UserWarning)

    # these are actually pixel errors
    angular_thresholds = [5, 10, 15]

    if os.path.isfile(save_to):
        eval_data = decompress_pickle(save_to)
    else:
        eval_data = {}
        
    n = len(dataset_data['im1'])
    
    pipe_name_base = os.path.join(bench_path, bench_res, dataset_name)
    pipe_name_base_small = ''
    pipe_name_root = os.path.join(pipe_name_base, pipe[0].get_id())
    pipe_img_save = os.path.join(bench_path, bench_plot, 'planar_accuracy')

    for pipe_module in pipe:
        pipe_name_base = os.path.join(pipe_name_base, pipe_module.get_id())
        pipe_name_base_small = os.path.join(pipe_name_base_small, pipe_module.get_id())
        pipe_img_save = os.path.join(pipe_img_save, pipe_module.get_id())

        print('Pipeline: ' + pipe_name_base_small)

        if (pipe_name_base in eval_data.keys()) and not force:
            eval_data_ = eval_data[pipe_name_base]                
            for a in angular_thresholds:
                print(f"mAA@{str(a).ljust(2,' ')} (F) : {eval_data_['pose_error_h_auc_' + str(a)]}")

            print(f"precision(H) : {eval_data_['reproj_global_prec_h']}")
            print(f"recall (H) : {eval_data_['reproj_global_recall_h']}")
                                                
            continue
                
        eval_data_ = {}

        eval_data_['err_plane_1_h'] = []
        eval_data_['err_plane_2_h'] = []        

        eval_data_['acc_1_h'] = []
        eval_data_['acc_2_h'] = []        
        
        eval_data_['reproj_max_error_h'] = []
        eval_data_['reproj_inliers_h'] = []
        eval_data_['reproj_valid_h'] = []
        eval_data_['reproj_prec_h'] = []
        eval_data_['reproj_recall_h'] = []
            
        with progress_bar('Completion') as p:
            for i in p.track(range(n)):            
                pipe_f = os.path.join(pipe_name_base, 'base', str(i) + '.pbz2')

                reproj_max_err =[]
                inl_sum = []
                avg_prec = 0
                avg_recall = 0

                if os.path.isfile(pipe_f):
                    out_data = decompress_pickle(pipe_f)

                    pts1 = out_data['pt1']
                    pts2 = out_data['pt2']
                                            
                    if torch.is_tensor(pts1):
                        pts1 = pts1.detach().cpu().numpy()
                        pts2 = pts2.detach().cpu().numpy()

                    if use_scale:
                        scales = dataset_data['im_pair_scale'][i]
                    else:
                        scales = np.asarray([[1.0, 1.0], [1.0, 1.0]])
                    
                    pts1 = pts1 * scales[0]
                    pts2 = pts2 * scales[1]
                        
                    nn = pts1.shape[0]
                                                
                    if nn < 4:
                        H = None
                    else:
                        H = torch.tensor(cv2.findHomography(pts1, pts2, 0)[0], device=device)

                    if nn > 0:
                        H_gt = torch.tensor(dataset_data['H'][i], device=device)
                        H_inv_gt = torch.tensor(dataset_data['H_inv'][i], device=device)
                        
                        pt1_ = torch.vstack((torch.tensor(pts1.T, device=device), torch.ones((1, nn), device=device)))
                        pt2_ = torch.vstack((torch.tensor(pts2.T, device=device), torch.ones((1, nn), device=device)))
                        
                        pt1_reproj = H_gt @ pt1_
                        pt1_reproj = pt1_reproj[:2] / pt1_reproj[2].unsqueeze(0)
                        d1 = ((pt2_[:2] - pt1_reproj)**2).sum(0).sqrt()
                        
                        pt2_reproj = H_inv_gt @ pt2_
                        pt2_reproj = pt2_reproj[:2] / pt2_reproj[2].unsqueeze(0)
                        d2 = ((pt1_[:2] - pt2_reproj)**2).sum(0).sqrt()
                        
                        valid_matches = torch.ones(nn, device=device, dtype=torch.bool)
                        
                        if dataset_data['im1_use_mask'][i]:
                            valid_matches = valid_matches & ~invalid_matches(dataset_data['im1_mask'][i], dataset_data['im2_full_mask'][i], pts1, pts2, rad)

                        if dataset_data['im2_use_mask'][i]:
                            valid_matches = valid_matches & ~invalid_matches(dataset_data['im2_mask'][i], dataset_data['im1_full_mask'][i], pts2, pts1, rad)
                                                
                        reproj_max_err_ = torch.maximum(d1, d2);                                
                        reproj_max_err = reproj_max_err_[valid_matches]
                        inl_sum = (reproj_max_err.unsqueeze(-1) < torch.tensor(err_th_list, device=device).unsqueeze(0)).sum(dim=0).type(torch.int)
                        avg_prec = inl_sum.type(torch.double).mean()/nn
                                                
                        if pipe_name_base==pipe_name_root:
                            recall_normalizer = torch.tensor(inl_sum, device=device)
                        else:
                            recall_normalizer = torch.tensor(eval_data[pipe_name_root]['reproj_inliers_h'][i], device=device)
                        avg_recall = inl_sum.type(torch.double) / recall_normalizer
                        avg_recall[~avg_recall.isfinite()] = 0
                        avg_recall = avg_recall.mean()
                        
                        reproj_max_err = reproj_max_err_.detach().cpu().numpy()
                        valid_matches = valid_matches.detach().cpu().numpy()
                        inl_sum = inl_sum.detach().cpu().numpy()
                        avg_prec = avg_prec.item()
                        avg_recall = avg_recall.item()
                else:
                    H = None
                    
                    
                if H is None:
                    eval_data_['err_plane_1_h'].append([])
                    eval_data_['err_plane_2_h'].append([])

                    eval_data_['acc_1_h'].append(np.inf) 
                    eval_data_['acc_2_h'].append(np.inf)        
                else:
                    heat1 = homography_error_heat_map(H_gt, H, torch.tensor(dataset_data['im1_full_mask'][i], device=device))
                    heat2 = homography_error_heat_map(H_inv_gt, H.inverse(), torch.tensor(dataset_data['im2_full_mask'][i], device=device))
                    
                    eval_data_['acc_1_h'].append(heat1[heat1 != 1].mean().detach().cpu().numpy()) 
                    eval_data_['acc_2_h'].append(heat2[heat2 != 1].mean().detach().cpu().numpy())       

                    eval_data_['err_plane_1_h'].append(heat1.type(torch.half).detach().cpu().numpy())
                    eval_data_['err_plane_2_h'].append(heat2.type(torch.half).detach().cpu().numpy())

                    if save_acc_images:
                        pipe_img_save_base = os.path.join(pipe_img_save, 'base')
                        os.makedirs(pipe_img_save_base, exist_ok=True)
                        iname = os.path.splitext(dataset_data['im1'][i])[0] + '_' + os.path.splitext(dataset_data['im2'][i])[0]
    
                        pipe_img_save1 = os.path.join(pipe_img_save_base, iname + '_1.png')
                        if not (os.path.isfile(pipe_img_save1) and not force):
                            im1s = os.path.join(bench_path,'planar',dataset_data['im1'][i])
                            colorize_plane(im1s, heat1, cmap_name='viridis', max_val=45, cf=0.7, save_to=pipe_img_save1)
    
                        pipe_img_save2 = os.path.join(pipe_img_save_base, iname + '_2.png')
                        if not (os.path.isfile(pipe_img_save2) and not force):
                            im2s = os.path.join(bench_path,'planar',dataset_data['im2'][i])
                            colorize_plane(im2s, heat2, cmap_name='viridis', max_val=45, cf=0.7, save_to=pipe_img_save2)
   
                eval_data_['reproj_max_error_h'].append(reproj_max_err)  
                eval_data_['reproj_inliers_h'].append(inl_sum)
                eval_data_['reproj_valid_h'].append(valid_matches)
                eval_data_['reproj_prec_h'].append(avg_prec)                           
                eval_data_['reproj_recall_h'].append(avg_recall)
                    
            aux = np.stack(([eval_data_['acc_1_h'], eval_data_['acc_2_h']]), axis=1)
            max_acc_err = np.max(aux, axis=1)        
            tmp = np.concatenate((aux, np.expand_dims(max_acc_err, axis=1)), axis=1)
    
            for a in angular_thresholds:       
                auc_1 = error_auc(np.squeeze(eval_data_['acc_1_h']), a)
                auc_2 = error_auc(np.squeeze(eval_data_['acc_2_h']), a)
                auc_max_12 = error_auc(np.squeeze(max_acc_err), a)
                eval_data_['pose_error_h_auc_' + str(a)] = np.asarray([auc_1, auc_2, auc_max_12])
                eval_data_['pose_error_h_acc_' + str(a)] = np.sum(tmp < a, axis=0)/np.shape(tmp)[0]

                # accuracy using 1st, 2nd image as reference, and the maximum accuracy
                print(f"mAA@{str(a).ljust(2,' ')} (H) : {eval_data_['pose_error_h_auc_' + str(a)]}")
            
            eval_data_['reproj_global_prec_h'] = torch.tensor(eval_data_['reproj_prec_h'], device=device).mean().item()
            eval_data_['reproj_global_recall_h'] = torch.tensor(eval_data_['reproj_recall_h'], device=device).mean().item()
        
            print(f"precision (H) : {eval_data_['reproj_global_prec_h']}")
            print(f"recall (H) : {eval_data_['reproj_global_recall_h']}")

            eval_data[pipe_name_base] = eval_data_
            compressed_pickle(save_to, eval_data)


def colorize_plane(ims, heat, cmap_name='viridis', max_val=45, cf=0.7, save_to='plane_acc.png'):
    im_gray = cv2.imread(ims, cv2.IMREAD_GRAYSCALE)
    im_gray = torch.tensor(im_gray, device=device).unsqueeze(0).repeat(3,1,1).permute(1,2,0)
    heat_mask = heat != -1
    heat_ = heat.clone()
    cmap = (colormaps[cmap_name](np.arange(0,(max_val + 1)) / max_val))[:,[2, 1, 0]]
    heat_[heat_ > max_val - 1] = max_val - 1
    heat_[heat_ == -1] = max_val
    cmap = torch.tensor(cmap, device=device)
    heat_im = cmap[heat_.type(torch.long)]
    heat_im = heat_im.type(torch.float) * 255
    blend_mask = heat_mask.unsqueeze(-1).type(torch.float) * cf
    imm = heat_im * blend_mask + im_gray.type(torch.float) * (1-blend_mask)                    
    cv2.imwrite(save_to, imm.type(torch.uint8).detach().cpu().numpy())   
 

def invalid_matches(mask1, mask2, pts1, pts2, rad):
    dmask2 = cv2.dilate(mask2.astype(np.ubyte),np.ones((rad*2+1, rad*2+1)))
    
    pt1 = torch.tensor(pts1, device=device).round().permute(1, 0)
    pt2 = torch.tensor(pts2, device=device).round().permute(1, 0)

    invalid_ = torch.zeros(pt1.shape[1], device=device, dtype=torch.bool)

    to_exclude = (pt1[0] < 0) & (pt2[0] < 0) & (pt1[0] >= mask1.shape[1]) & (pt2[0] >= mask2.shape[1]) & (pt1[1] < 0) & (pt2[1] < 0) & (pt1[1] >= mask1.shape[0]) & (pt2[1] >= mask2.shape[0])

    pt1 = pt1[:, ~to_exclude]
    pt2 = pt2[:, ~to_exclude]
    
    l1 = (pt1[1, :] * mask1.shape[1] + pt1[0,:]).type(torch.long)
    l2 = (pt2[1, :] * mask2.shape[1] + pt2[0,:]).type(torch.long)

    invalid_check = ~(torch.tensor(mask1, device=device).flatten()[l1]) & ~(torch.tensor(dmask2, device=device, dtype=torch.bool).flatten()[l2])
    invalid_[~to_exclude] = invalid_check 

    return invalid_


def homography_error_heat_map(H12_gt, H12, mask1):
    pt1 = mask1.argwhere()
    
    pt1 = torch.cat((pt1, torch.ones(pt1.shape[0], 1, device=device)), dim=1).permute(1,0)   

    pt2_gt_ = H12_gt.type(torch.float) @ pt1
    pt2_gt_ = pt2_gt_[:2] / pt2_gt_[2].unsqueeze(0)

    pt2_ = H12.type(torch.float) @ pt1
    pt2_ = pt2_[:2] / pt2_[2].unsqueeze(0)

    d1 = ((pt2_gt_ - pt2_)**2).sum(dim=0).sqrt()

    heat_map = torch.full(mask1.shape, -1, device=device, dtype=torch.float)
    heat_map[mask1] = d1
    
    return heat_map
