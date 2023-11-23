import os
import scipy.io as sio
from scipy import ndimage
import cv2
from PIL import Image
import numpy as np
import math
import torch
from skimage.color import rgb2gray
import multiprocessing


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
    max_iter = 10000
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

    # print(f'**************************Hdata: {Hdata}')

    midx = np.stack([Hdata[i][2] for i in range(len(Hdata))], axis=0)

    # print(f'**************************midx: {midx.shape}')
    sidx = np.sum(midx, axis=1)
    # print(f'**************************sidx: {sidx}')
    # print(f"##########################")
    # print(f"{(np.tile(sidx, (midx.shape[1], 1)).T * midx)}")
    # print(f"###########################")
    didx = np.argmax(np.tile(sidx, (midx.shape[1], 1)).T * midx, axis=0)
    # print(f'**************************didx: {didx}')

    Hdata = [entry[:2] for entry in Hdata]

    return didx, Hdata


def apply_H(p, H, HH, wr, im):
    p_status = 0

    Hp = np.dot(H, np.append(p, 1))
    Hp = Hp[:2] / Hp[2]

    x = np.arange(Hp[0] - wr, Hp[0] + wr + 1)
    y = np.arange(Hp[1] - wr, Hp[1] + wr + 1)
    aux1, aux2 = np.meshgrid(x, y)
    sx = len(x)
    sy = len(y)

    aux = np.dot(np.linalg.inv(H), np.dot(HH, np.vstack((aux1.flatten(), aux2.flatten(), np.ones_like(aux1.flatten())))))
    x_ = aux[0, :] / aux[2, :]
    y_ = aux[1, :] / aux[2, :]
    xf = np.floor(x_).astype(int)
    yf = np.floor(y_).astype(int)
    xc = xf + 1
    yc = yf + 1

    if any(x_ < 1) or any(x_ > im.shape[1]) or any(y_ < 1) or any(y_ > im.shape[0]):
        im_ = np.array([])
        p_status = 1
        return im_, p_status

    im_ = np.zeros((sy, sx))
    for i in range(sy):
        for j in range(sx):
            im_[i, j] = im[yf[i, j] - 1, xf[i, j] - 1] * (xc[i, j] - x_[i, j]) * (yc[i, j] - y_[i, j]) + \
                         im[yc[i, j] - 1, xc[i, j] - 1] * (x_[i, j] - xf[i, j]) * (y_[i, j] - yf[i, j]) + \
                         im[yf[i, j] - 1, xc[i, j] - 1] * (x_[i, j] - xf[i, j]) * (yc[i, j] - y_[i, j]) + \
                         im[yc[i, j] - 1, xf[i, j] - 1] * (xc[i, j] - x_[i, j]) * (y_[i, j] - yf[i, j])

    return im_, p_status


def lsm(im1, im2, p1, p2, wr, Hs, s, T_):
    max_iter = 500
    alpha = 0.5
    px_err = 1
    tm_err = 10
    what = 3

    err_count = np.zeros(tm_err + 1) + np.inf
    v = np.array([1, 0, 0, 0, 1, 0, 1, 0])
    p2_new_old = p2.copy()
    p2_new = p2.copy()
    p2_err_old = np.inf
    p2_err = np.inf
    p2_err_base = np.inf

    T = np.array([[v[0], v[1], v[2]], [v[3], v[4], v[5]], [0, 0, 1]])
    r = v[6:8]

    sH = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]])
    Hs[0] = np.dot(sH, Hs[0])
    Hs[1] = np.dot(sH, np.dot(T_, Hs[1]))

    tmp_im1, p2_status = apply_H(p1, Hs[0], np.eye(3), wr, im1)
    if p2_status == 1:
        return p2_new, p2_status, p2_err, p2_err_base, T

    p2_status = 4
    for i in range(max_iter):
        im2_, p2_status = apply_H(p2, Hs[1], T, wr + 1, im2)
        if p2_status == 1:
            p2_new = p2_new_old
            p2_err = p2_err_old
            p2_status = 2
            break

        Hp = np.dot(Hs[1], np.append(p2, 1))
        Hp = Hp[:2] / Hp[2]

        x = np.arange(Hp[0] - wr - 1, Hp[0] + wr + 2)
        y = np.arange(Hp[1] - wr - 1, Hp[1] + wr + 2)
        aux1, aux2 = np.meshgrid(x, y)

        x2 = aux1
        y2 = aux2

        tmp_gx = np.reshape(im2_[1:-1, 2:] - im2_[1:-1, :-2] / 2, (-1, 1))
        tmp_gy = np.reshape(im2_[2:, 1:-1] - im2_[:-2, 1:-1] / 2, (-1, 1))
        tmp_im2 = np.reshape(im2_[1:-1, 1:-1], (-1, 1))
        tmp_x2 = np.reshape(x2[1:-1, 1:-1], (-1, 1))
        tmp_y2 = np.reshape(y2[1:-1, 1:-1], (-1, 1))
        tmp_ones = np.ones((len(tmp_im2), 1))

        if what == 0:
            tmp_more = np.empty((0, 1))  # Empty array
        elif what == 1:
            tmp_more = tmp_ones
        elif what == 2:
            tmp_more = tmp_im2
        elif what == 3:
            tmp_more = np.hstack((tmp_im2, tmp_ones))
        else:
            pass

        b = tmp_im1.flatten() - r[0] * tmp_im2 - r[1]
        A = np.column_stack([
            r[0] * tmp_gx * tmp_x2,
            r[0] * tmp_gx * tmp_y2,
            r[0] * tmp_gx,
            r[0] * tmp_gy * tmp_x2,
            r[0] * tmp_gy * tmp_y2,
            r[0] * tmp_gy,
            tmp_more
        ])

        v = np.linalg.pinv(A) @ b
        v = v.flatten()

        if i == 1:
            p2_err_base = np.mean(np.abs(b))

        curr_err = np.mean(np.abs(b - A @ v))
        err_count = np.roll(err_count, -1)
        err_count[-1] = curr_err

        if np.all(np.abs(err_count - err_count[-1]) < px_err):
            p2_status = 0
            break

        T_old = T.copy()
        T[0] += alpha * v[0:3]
        T[1] += alpha * v[3:6]

        if what == 1:
            r[1] += alpha * v[-1]
        elif what == 2:
            r[0] += alpha * v[-1]
        elif what == 3:
            r += alpha * v[6:]

        if i == 1:
            p2_err_old = p2_err_base
        else:
            p2_err_old = p2_err

        p2_err = curr_err
        p2_new_old = p2_new

        tmp_p2_new = np.dot(np.linalg.inv(Hs[2]) @ T @ np.linalg.inv(Hs[2]), np.hstack((p2, 1)))
        p2_new = tmp_p2_new[:2] / tmp_p2_new[2]

        if np.linalg.norm(p2_new - p2) > wr / s:
            p2_new = p2_new_old
            p2_err = p2_err_old
            p2_status = 3
            T = T_old.copy()
        
    return p2_new, p2_status, p2_err, p2_err_base, T


def ncorr(im1, im2, p1, p2, wr, Hs, s, T_):
    p2_new = p2.copy()
    p2_err = np.inf
    p2_err_base = np.inf
    T = np.eye(3)

    sH = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]])
    Hs[0] = np.dot(sH, Hs[0])
    Hs[1] = np.dot(np.dot(sH, T_), Hs[1])

    tmp1, p2_status = apply_H(p1, Hs[0], np.eye(3), wr, im1)
    if p2_status == 1:
        return p2_new, p2_status, p2_err, p2_err_base, T

    tmp2, p2_status = apply_H(p2, Hs[1], np.eye(3), 2 * wr, im2)
    if p2_status == 1:
        return p2_new, p2_status, p2_err, p2_err_base, T

    m = np.correlate(tmp1.ravel(), tmp2.ravel(), mode='full')
    m = m[2 * wr + 1:-2 * wr, 2 * wr + 1:-2 * wr]
    p2_err = np.max(m)

    i, j = np.where(m == p2_err)
    i = i[0] - wr - 1
    j = j[0] - wr - 1

    Hp2 = np.dot(Hs[1], np.array([p2[0], p2[1], 1]))
    Hp2 = Hp2[:2] / Hp2[2]

    T = np.array([[1, 0, j], [0, 1, i], [0, 0, 1]])

    p2_new = np.dot(np.linalg.inv(Hs[1]), np.array([Hp2[0] + j, Hp2[1] + i, 1]))
    p2_new = p2_new[:2] / p2_new[2]
    p2_status = 0

    p2_err = -abs(p2_err)

    return p2_new, p2_status, p2_err, p2_err_base, T


def kpt_improver(im1, im2, matches, what, wr, s, ref, hom, hom_data):
    nthreads = 10  # Number of threads
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(nthreads)

    if hom != 0:
        didx = hom_data['didx']
        Hdata = hom_data['Hdata']
    else:
        didx = np.ones((1, matches.shape[0]))
        Hdata = [np.eye(3), np.eye(3)]

    img1 = torch.tensor(rgb2gray(im1)).to(device).double()
    img2 = torch.tensor(rgb2gray(im2)).to(device).double()

    matches_new = torch.tensor(matches).to(device).double()

    method = {
        'lsm': lsm,
        'norm_corr': ncorr
    }

    T = [torch.eye(3) for _ in range(matches.shape[0])]

    stime = torch.cuda.Event(enable_timing=True)
    etime = torch.cuda.Event(enable_timing=True)

    stime.record()

    for k in range(len(what)):
        widx = [idx for idx, val in enumerate(method.keys()) if val == what[k]]
        wr_ = wr[k]

        def process_matches(i):
            aux_match = matches[i, :]
            tmp_match = matches_new[i, :]
            T_ = T[i]

            if not ref or ref == 1:
                p2_new, p2_status, p2_err, p2_err_base, T2 = method[what[k]](
                    img1, img2, aux_match[0:2], aux_match[2:4], wr_, Hdata[int(didx[i]) - 1][0:2], s[k], T_)

            if not ref or ref == 2:
                p1_new, p1_status, p1_err, p1_err_base, T1 = method[what[k]](
                    img2, img1, aux_match[2:4], aux_match[0:2], wr_, Hdata[int(didx[i]) - 1][1::-1], s[k], T_)

            if ref == 0:
                if not p2_status and p2_err < p1_err:
                    if p2_err < p2_err_base:
                        tmp_match[2:4] = p2_new
                        T_ = torch.matmul(torch.inverse(T2), T_)

                if not p1_status and p1_err < p2_err:
                    if p1_err < p1_err_base:
                        tmp_match[0:2] = p1_new
                        T_ = torch.matmul(T1, T_)

            elif ref == 1:
                if not p2_status:
                    if p2_err < p2_err_base:
                        tmp_match[2:4] = p2_new
                        T_ = torch.matmul(torch.inverse(T2), T_)

            elif ref == 2:
                if not p1_status:
                    if p1_err < p1_err_base:
                        tmp_match[0:2] = p1_new
                        T_ = torch.matmul(T1, T_)

            matches_new[i, :] = tmp_match
            return T_

        with multiprocessing.Pool(processes=nthreads) as pool:
            T = pool.map(process_matches, range(matches.shape[0]))

        matches = matches_new

    return matches, ttime


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

    corr_method = [
        'lsm',
        'norm_corr',
        'fast_match'
    ]

    # noise offsets
    err = 1#11
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
                matches = method[widx](os.path.join('..', im1l_name), os.path.join('..', im2l_name))
                midx = fun_sac_matrix(matches[:, [0, 1]], matches[:, [2, 3]], th_sac)
                os.chdir('..')

                sio.savemat(match_file, {'matches': matches, 'midx': midx})

            gt_scaled = gt * s

            # matches = matches[:100,:]

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

                middle_homo_file = os.path.join(ppath, f'matches_scale_{s}_{method[widx]}_sac_{th_sac}_err_{e[k,0]}_{e[k,1]}_middle_homo.mat')
                th_out = np.ceil(th_sac / 2)

                if os.path.exists(middle_homo_file):
                    data = sio.loadmat(middle_homo_file)
                    Hdata = data['Hdata']
                    didx = data['didx']
                else:
                    Hdata, didx = middle_homo(im1l_name, im2l_name, all_matches, th_sac, th_out)
                    sio.savemat(middle_homo_file, {'Hdata': Hdata, 'didx': didx})

                im1 = cv2.imread(im1l_name)
                im2 = cv2.imread(im2l_name)

                to_check_matches = all_matches[:gt_scaled.shape[0], :]
                hom_data = {'Hdata': Hdata, 'didx': didx[:gt_scaled.shape[0]]}

                print(f"*********to_check_matches: {to_check_matches.shape}")

                for j in range(len(corr_method)):
                    for hom in range(2):
                        middle_homo_file = os.path.join(ppath, f'matches_scale_{s}_{method[widx]}_sac_{th_sac}_err_{e[k,0]}_{e[k,1]}_{corr_method[j]}_hom_{hom}.mat')

                        if not os.path.exists(middle_homo_file):
                            data_mm1, ttime1 = kpt_improver(im1, im2, to_check_matches, corr_method[j][1], th_sac, 1, 1, hom, hom_data)
                            data_mm2, ttime2 = kpt_improver(im1, im2, to_check_matches, corr_method[j][1], [int(np.fix(th_sac / 2)), th_sac], [0.5, 1], 1, hom, hom_data)

                            data = {
                                'mm1': data_mm1,
                                'mm2': data_mm2,
                                'time1': ttime1,
                                'time2': ttime2,
                                'err1': np.sqrt(np.sum((data_mm1[:, 2:] - gt_scaled[:, 2:]) ** 2, axis=1)),
                                'err2': np.sqrt(np.sum((data_mm2[:, 2:] - gt_scaled[:, 2:]) ** 2, axis=1))
                            }

#                             sio.savemat(middle_homo_file, {'data': data})
#                         else:
#                             data = sio.loadmat(middle_homo_file)['data']