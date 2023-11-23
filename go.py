import os
import scipy.io as sio
from scipy import ndimage
import cv2
from PIL import Image
import numpy as np
import math
import torch


def data_normalize(pts):
    c = np.mean(pts, axis=1)
    s = np.sqrt(2) / np.mean(np.sqrt((pts[0, :] - c[0])**2 + (pts[1, :] - c[1])**2))

    T = np.array([
        [s, 0, -c[0] * s],
        [0, s, -c[1] * s],
        [0, 0, 1]
    ])

    return T


def compute_homography(pts1, pts2):
    T1 = data_normalize(pts1)
    T2 = data_normalize(pts2)

    npts1 = np.dot(T1, pts1)
    npts2 = np.dot(T2, pts2)

    l = npts1.shape[1]
    A = np.vstack((
        np.hstack((np.zeros((l, 3)), -np.multiply(np.tile(npts2[2, :], (3, 1)).T, npts1.T), np.multiply(np.tile(npts2[1, :], (3, 1)).T, npts1.T))),
        np.hstack((np.multiply(np.tile(npts2[2, :], (3, 1)).T, npts1.T), np.zeros((l, 3)), -np.multiply(np.tile(npts2[0, :], (3, 1)).T, npts1.T))),
        np.hstack((-np.multiply(np.tile(npts2[1, :], (3, 1)).T, npts1.T), np.multiply(np.tile(npts2[0, :], (3, 1)).T, npts1.T), np.zeros((l, 3))))
    ))

    _, D, V = np.linalg.svd(A, full_matrices=True)
    # print(f"@@@@@@@ D: {D.shape}")
    # print(f"@@@@@@@ V: {V.shape}")
    # D = np.diag(D)
    H = V[-1, :].reshape(3, 3).T
    H = np.linalg.inv(T2) @ H @ T1

    return H, D


def get_hom_inliers(pt1, pt2, H, th, sidx):
    pt2_ = np.dot(H, pt1)
    s2_ = np.sign(pt2_[2, :])
    tmp2_ = pt2_[:2, :] / pt2_[2, :] - pt2[:2, :]
    err2 = np.sum(tmp2_**2, axis=0)
    s2 = s2_[sidx[0]]
    
    if not np.all(s2_[sidx] == s2):
        nidx = np.zeros(pt1.shape[1], dtype=bool)
        err = np.inf
        return nidx#, err

    pt1_ = np.dot(np.linalg.inv(H), pt2)
    s1_ = np.sign(pt1_[2, :])
    tmp1_ = pt1_[:2, :] / pt1_[2, :] - pt1[:2, :]
    err1 = np.sum(tmp1_**2, axis=0)
    s1 = s1_[sidx[0]]
    
    if not np.all(s1_[sidx] == s1):
        nidx = np.zeros(pt1.shape[1], dtype=bool)
        err = np.inf
        return nidx#, err

    err = np.maximum(err1, err2)
    err[~np.isfinite(err)] = np.inf
    nidx = (err < th) & (s2_ == s2) & (s1_ == s1)
    
    return nidx#, err


def ransac_middle(pt1, pt2, th, th_out):
    # np.random.seed(42)  # 设置随机种子以保持结果的可重复性

    max_iter = 100#10000
    min_iter = 100
    p = 0.9
    c = 0

    n = pt1.shape[0]
    th = th ** 2
    th_out = th_out ** 2

    if n < 4:
        H1 = np.array([])
        H2 = np.array([])
        midx = np.zeros(n, dtype=bool)
        oidx = np.zeros(n, dtype=bool)
        return H1, H2, midx, oidx, c

    min_iter = min(min_iter, int(np.math.comb(n, 2)))

    pt1 = np.hstack((pt1, np.ones((n, 1)))).T
    pt2 = np.hstack((pt2, np.ones((n, 1)))).T

    midx = np.zeros(n, dtype=bool)
    oidx = np.zeros(n, dtype=bool)
    Nc = float('inf')

    while c < max_iter:
        sidx = np.random.choice(n, size=4, replace=False)
        ptm = (pt1 + pt2) / 2
        H1, eD = compute_homography(pt1[:, sidx], ptm[:, sidx])
        if eD[-2] < 0.05:
            continue
        H2, eD = compute_homography(pt2[:, sidx], ptm[:, sidx])
        if eD[-2] < 0.05:
            continue

        nidx = get_hom_inliers(pt1, ptm, H1, th, sidx) & get_hom_inliers(pt2, ptm, H2, th, sidx)
        if np.sum(nidx) > np.sum(midx):
            midx = nidx
            sidx_ = sidx
            Nc = 4 * np.sum(midx) / n
            if c > Nc and c > min_iter:
                break
        c += 1

    if np.any(midx):
        H1, _ = compute_homography(pt1[:, midx], ptm[:, midx])
        H2, _ = compute_homography(pt2[:, midx], ptm[:, midx])
        midx = get_hom_inliers(pt1, ptm, H1, th, sidx_) & get_hom_inliers(pt2, ptm, H2, th, sidx_)
        oidx = get_hom_inliers(pt1, ptm, H1, th_out, sidx_) & get_hom_inliers(pt2, ptm, H2, th_out, sidx_)
    else:
        H1 = np.array([])
        H2 = np.array([])
        midx = np.zeros(n, dtype=bool)
        oidx = np.zeros(n, dtype=bool)

    return H1, H2, midx, oidx, c

def get_avg_hom(pt1, pt2, th, th_out):
    H1 = np.eye(3)
    H2 = np.eye(3)

    Hdata = []
    max_iter = 5
    midx = np.zeros(pt1.shape[0], dtype=bool)
    tidx = np.zeros(pt1.shape[0], dtype=bool)
    hc = 1

    while True:
        pt1_ = pt1[~midx, :]
        pt1_ = np.hstack((pt1_, np.ones((pt1_.shape[0], 1))))
        pt1_ = np.dot(H1, pt1_.T)
        pt1_ = pt1_[:2, :] / pt1_[2, :]

        pt2_ = pt2[~midx, :]
        pt2_ = np.hstack((pt2_, np.ones((pt2_.shape[0], 1))))
        pt2_ = np.dot(H2, pt2_.T)
        pt2_ = pt2_[:2, :] / pt2_[2, :]

        H1_, H2_, nidx, oidx, c = ransac_middle(pt1_.T, pt2_.T, th, th_out)

        if np.sum(nidx) <= 4:
            break

        zidx = np.zeros(midx.shape, dtype=bool)
        zidx[~midx] = nidx

        tidx[~midx] = tidx[~midx] | nidx
        midx[~midx] = oidx * hc

        H1_new = np.dot(H1_, H1)
        H2_new = np.dot(H2_, H2)

        for i in range(max_iter):
            pt1_ = pt1[zidx, :]
            pt1_ = np.hstack((pt1_, np.ones((pt1_.shape[0], 1))))
            pt1_ = np.dot(H1_new, pt1_.T)
            pt1_ = pt1_ / pt1_[2, :]

            pt2_ = pt2[zidx, :]
            pt2_ = np.hstack((pt2_, np.ones((pt2_.shape[0], 1))))
            pt2_ = np.dot(H2_new, pt2_.T)
            pt2_ = pt2_ / pt2_[2, :]

            ptm = (pt1_ + pt2_) / 2
            H1_, _ = compute_homography(pt1_, ptm)
            H2_, _ = compute_homography(pt2_, ptm)

            H1_new = np.dot(H1_, H1_new)
            H2_new = np.dot(H2_, H2_new)

        Hdata.append([H1_new, H2_new, zidx])
        hc += 1

    return Hdata

def middle_homo(im1_path, im2_path, matches, th, th_out):
    pt1 = matches[:, :2]
    pt2 = matches[:, 2:]

    Hdata = get_avg_hom(pt1, pt2, th, th_out)

    print(f'**************************Hdata: {Hdata}')

    midx = np.stack([Hdata[i][2] for i in range(len(Hdata))], axis=0)

    print(f'**************************midx: {midx.shape}')
    sidx = np.sum(midx, axis=1)
    print(f'**************************sidx: {sidx}')
    print(f"##########################")
    print(f"{(np.tile(sidx, (midx.shape[1], 1)).T * midx)}")
    print(f"###########################")
    didx = np.argmax(np.tile(sidx, (midx.shape[1], 1)).T * midx, axis=0)
    print(f'**************************didx: {didx}')

    Hdata = [entry[:2] for entry in Hdata]

    return didx, Hdata


def pdist2(X, Y):
    # Calculate pairwise Euclidean distances
    # Reshape to enable broadcasting for the distance calculation
    dists = np.sqrt(np.sum((X[:, np.newaxis] - Y) ** 2, axis=2))
    return dists


if __name__ == '__main__':

    s = 0.2
    dpath = 'img_test'
    th_sac = 15
    th_cf = 2

    p = os.listdir(dpath)
    p_valid = [os.path.isdir(os.path.join(dpath, folder)) for folder in p]
    p = [p[i] for i in range(len(p)) if p_valid[i]]

    method = [
        # 'keynet': keynet,
        # 'keynet_upright': keynet_upright,
        'hz',
        # 'hz_upright': hz_upright
    ]

    corr_method = {
        'lsm',
        'norm_corr',
        'fast_match'
    }

    # noise offsets
    err = 11
    e = np.empty((0, 2))

    for k in range(1, err + 1):
        if k % 2 != 0:
            e = np.vstack((e, k * np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])))
        else:
            e = np.vstack((e, k * np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])))

    for z in range(len(method)):
        widx = z
        # widx = np.where(method[:,0] == what) # Uncomment for equivalent to find(strcmp(method(:,1),what)) in MATLAB
        
        for i in range(len(p)):
            bpath = os.path.join(dpath, p[i])
            ppath = os.path.join(dpath, p[i], 'working_' + method[widx])

            os.makedirs(ppath, exist_ok=True)

            im1h_name = os.path.join(bpath, 'img1.jpg')
            im1l_name = os.path.join(ppath, f'img1_scale_{s}.png')
            if not os.path.exists(im1l_name):
                im = cv2.imread(im1h_name)
                im = cv2.resize(im, None, fx=s, fy=s)
                cv2.imwrite(im1l_name, im)

            im2h_name = os.path.join(bpath, 'img2.jpg')
            im2l_name = os.path.join(ppath, f'img2_scale_{s}.png')
            if not os.path.exists(im2l_name):
                im = cv2.imread(im2h_name)
                im = cv2.resize(im, None, fx=s, fy=s)
                cv2.imwrite(im2l_name, im)

            gt_file = os.path.join(bpath, 'gt.mat')
            if os.path.exists(gt_file):
                gt = sio.loadmat(gt_file)['gt']
            else:
                gt = np.loadtxt(os.path.join(bpath, 'gt.txt'))
                sio.savemat(gt_file, {'gt': gt})
                os.remove(os.path.join(bpath, 'gt.txt'))

            match_file = os.path.join(ppath, f'matches_scale_{s}_{method[widx]}_sac_{th_sac}.mat')
            if os.path.exists(match_file):
                data = sio.loadmat(match_file)
                matches = data['matches']
                midx = data['midx']
            else:
                os.chdir('matcher')
                matches = method[widx][1](os.path.join('..', im1l_name), os.path.join('..', im2l_name))
                midx = fun_sac_matrix(matches[:, [0, 1]], matches[:, [2, 3]], th_sac)
                os.chdir('..')

                sio.savemat(match_file, {'matches': matches, 'midx': midx})

            gt_scaled = gt * s

            # matches = matches[:20,:]

            mm1 = pdist2(matches[:, :2], gt_scaled[:, :2])
            mm2 = pdist2(matches[:, 2:], gt_scaled[:, 2:])

            # print(f"*********mm1: {mm1.shape}")
            # print(f"*********mm2: {mm2}")

            # remove matches within 2r (th_cf*th_sac) of GT matches before including the noisy matches
            to_remove_matches = np.any((mm1 < th_sac * th_cf) | (mm2 < th_sac * th_cf), axis=1)
            print(f"*********to_remove_matches: {np.sum(to_remove_matches!=0)}")
            hom_matches = matches[~to_remove_matches]
            print(f"*********hom_matches: {hom_matches.shape}")

            for k in range(len(e)):
                aux = np.copy(gt_scaled)
                aux[:, 2:] += np.tile(e[k], (gt_scaled.shape[0], 1))
                all_matches = np.vstack((aux, hom_matches))
                print(f"*********all_matches: {all_matches.shape}")

                middle_homo_file = os.path.join(ppath, f'matches_scale_{s}_{method[widx][0]}_sac_{th_sac}_err_{e[k,0]}_{e[k,1]}_middle_homo.mat')
                th_out = np.ceil(th_sac / 2)

                if os.path.exists(middle_homo_file):
                    data = sio.loadmat(middle_homo_file)
                    Hdata = data['Hdata']
                    didx = data['didx']
                else:
                    Hdata, didx = middle_homo(im1l_name, im2l_name, all_matches, th_sac, th_out)
                    # sio.savemat(middle_homo_file, {'Hdata': Hdata, 'didx': didx})

                im1 = cv2.imread(im1l_name)
                im2 = cv2.imread(im2l_name)

#                 to_check_matches = all_matches[:gt_scaled.shape[0], :]
#                 hom_data = {'Hdata': Hdata, 'didx': didx[:gt_scaled.shape[0]]}

#                 for j in range(len(corr_method)):
#                     for hom in range(2):
#                         middle_homo_file = os.path.join(ppath, f'matches_scale_{s}_{method[widx][0]}_sac_{th_sac}_err_{e[k,0]}_{e[k,1]}_{corr_method[j][0]}_hom_{hom}.mat')

#                         if not os.path.exists(middle_homo_file):
#                             data_mm1, ttime1 = kpt_improver(im1, im2, to_check_matches, corr_method[j][1], th_sac, 1, 1, hom, hom_data)
#                             data_mm2, ttime2 = kpt_improver(im1, im2, to_check_matches, corr_method[j][1], [int(np.fix(th_sac / 2)), th_sac], [0.5, 1], 1, hom, hom_data)

#                             data = {
#                                 'mm1': data_mm1,
#                                 'mm2': data_mm2,
#                                 'time1': ttime1,
#                                 'time2': ttime2,
#                                 'err1': np.sqrt(np.sum((data_mm1[:, 2:] - gt_scaled[:, 2:]) ** 2, axis=1)),
#                                 'err2': np.sqrt(np.sum((data_mm2[:, 2:] - gt_scaled[:, 2:]) ** 2, axis=1))
#                             }

#                             sio.savemat(middle_homo_file, {'data': data})
#                         else:
#                             data = sio.loadmat(middle_homo_file)['data']