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
import src.ncc as ncc

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
    
    out_dir = os.path.join(bench_path, 'gt_data')
    if not os.path.isdir(out_dir):    
        with zipfile.ZipFile('data/megadepth_scannet_gt_data.zip',"r") as zip_ref:
            zip_ref.extractall(bench_path)    
    
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
        
                tmp = np.concatenate((aux, np.expand_dims(np.max(aux, axis=1), axis=1)), axis=1)
        
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
        
            tmp = np.concatenate((aux, np.expand_dims(np.max(aux, axis=1), axis=1)), axis=1)
    
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

    file_to_download = os.path.join(bench_path, 'downloads', 'megadepth_test_1500.tar.gz')    
    if not os.path.isfile(file_to_download):    
        url = "https://drive.google.com/file/d/12yKniNWebDHRTCwhBNJmxYMPgqYX3Nhv/view?usp=drive_link"
        gdown.download(url, file_to_download, fuzzy=True)
    
    out_dir = os.path.join(bench_path, 'megadepth_test_1500')
    if not os.path.isdir(out_dir):    
        with tarfile.open(file_to_download,"r") as tar_ref:
            tar_ref.extractall(bench_path)
    
    file_to_download = os.path.join(bench_path, 'downloads', 'scannet_test_1500.tar.gz')    
    if not os.path.isfile(file_to_download):    
        url = "https://drive.google.com/file/d/1wtl-mNicxGlXZ-UQJxFnKuWPvvssQBwd/view?usp=drive_link"
        gdown.download(url, file_to_download, fuzzy=True)

    out_dir = os.path.join(bench_path, 'scannet_test_1500')
    if not os.path.isdir(out_dir):    
        with tarfile.open(file_to_download,"r") as tar_ref:
            tar_ref.extractall(bench_path)
    
    return


def show_pipe(pipe, dataset_data, dataset_name, bar_name, bench_path='bench_data' , bench_im='imgs', bench_res='res', bench_plot='plot', force=False):

    n = len(dataset_data['im1'])
    im_path = os.path.join(bench_im, dataset_name)        
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
            fig, axes = viz.plot_images([img1, img2])              
            
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
                        viz.plot_matches(mpt1, mpt2, color=pipe_color[clr], lw=0.2, ps=6, a=0.3, axes=axes)
                        idx = idx[mask]
                    clr = clr + 1
            mpt1 = pt1[idx]
            mpt2 = pt2[idx]
            viz.plot_matches(mpt1, mpt2, color=pipe_color[clr], lw=0.2, ps=6, a=0.3, axes=axes)

            viz.save_plot(pipe_img_save)
            viz.close_plot(fig)
