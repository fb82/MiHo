import os
import scipy.io as sio
from scipy import ndimage
import cv2
import numpy as np


def middle_homo(im1, im2, matches, th, th_out):
    pt1 = matches[:, :2]
    pt2 = matches[:, 2:]

    Hdata = get_avg_hom(pt1, pt2, th, th_out)

    im1 = ndimage.imread(im1)
    im2 = ndimage.imread(im2)

    midx = []
    for i in range(len(Hdata)):
        midx.append(Hdata[i][2])
    midx = np.concatenate(midx, axis=0)
    midx = midx > 0

    sidx = np.sum(midx, axis=1)
    didx = np.argmax(np.tile(sidx, (1, midx.shape[1])) * midx, axis=1)

    Hdata = [entry[:2] for entry in Hdata]

    return didx, Hdata

def get_avg_hom(pt1, pt2, th, th_out):
    H1 = np.eye(3)
    H2 = np.eye(3)
    Hdata = []
    max_iter = 5
    midx = np.zeros(pt1.shape[0], dtype=bool)
    tidx = np.zeros(pt1.shape[0], dtype=bool)
    hc = 1

    while True:
        pt1_ = np.hstack((pt1[~midx, :], np.ones((np.sum(~midx), 1))))
        pt1_ = np.dot(H1, pt1_.T)
        pt1_ = pt1_[:2] / pt1_[2]

        pt2_ = np.hstack((pt2[~midx, :], np.ones((np.sum(~midx), 1))))
        pt2_ = np.dot(H2, pt2_.T)
        pt2_ = pt2_[:2] / pt2_[2]

        H1_, H2_, nidx, oidx = ransac_middle(pt1_.T, pt2_.T, th, th_out)

        if np.sum(nidx) <= 4:
            break

        zidx = np.zeros(midx.shape[0], dtype=bool)
        zidx[~midx] = nidx

        tidx[~midx] = tidx[~midx] | nidx
        midx[~midx] = oidx * hc

        H1_new = np.dot(H1_, H1)
        H2_new = np.dot(H2_, H2)

        for i in range(max_iter):
            pt1_ = pt1[zidx]
            pt1_ = np.hstack((pt1_, np.ones((pt1_.shape[0], 1))))
            pt1_ = np.dot(H1_new, pt1_.T)
            pt1_ = pt1_ / pt1_[2]

            pt2_ = pt2[zidx]
            pt2_ = np.hstack((pt2_, np.ones((pt2_.shape[0], 1))))
            pt2_ = np.dot(H2_new, pt2_.T)
            pt2_ = pt2_ / pt2_[2]

            ptm = (pt1_ + pt2_) / 2
            H1_ = compute_homography(pt1_, ptm)
            H2_ = compute_homography(pt2_, ptm)

            H1_new = np.dot(H1_, H1_new)
            H2_new = np.dot(H2_, H2_new)

        Hdata.append([H1_new, H2_new, zidx])
        hc += 1

    return Hdata

if __name__ == '__main__':

    s = 0.2
    dpath = 'data'
    th_sac = 15
    th_cf = 2

    p = os.listdir(dpath)
    p_valid = [os.path.isdir(os.path.join(dpath, folder)) for folder in p]
    p_valid[:2] = False
    p = [p[i] for i in range(len(p)) if p_valid[i]]

    method = {
        'keynet': keynet,
        'keynet_upright': keynet_upright,
        'hz': hz,
        'hz_upright': hz_upright
    }

    corr_method = {
        'lsm': lsm,
        'norm_corr': ncorr,
        'fast_match': fmatch
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
            ppath = os.path.join(dpath, p[i], 'working_' + method[widx][0])

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
                gt = loadmat(gt_file)['gt']
            else:
                gt = np.loadtxt(os.path.join(bpath, 'gt.txt'))
                sio.savemat(gt_file, {'gt': gt})
                os.remove(os.path.join(bpath, 'gt.txt'))

            match_file = os.path.join(ppath, f'matches_scale_{s}_{method[widx][0]}_sac_{th_sac}.mat')
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

            mm1 = np.linalg.norm(matches[:, :2] - gt_scaled[:, :2], axis=1)
            mm2 = np.linalg.norm(matches[:, 2:] - gt_scaled[:, 2:], axis=1)

            to_remove_matches = np.any((mm1 < th_sac * th_cf) | (mm2 < th_sac * th_cf), axis=0)
            hom_matches = matches[~to_remove_matches]

            for k in range(len(e)):
                aux = np.copy(gt_scaled)
                aux[:, 2:] += np.tile(e[k], (gt_scaled.shape[0], 1))
                all_matches = np.vstack((aux, hom_matches))

                middle_homo_file = os.path.join(ppath, f'matches_scale_{s}_{method[widx][0]}_sac_{th_sac}_err_{e[k,0]}_{e[k,1]}_middle_homo.mat')
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

                for j in range(len(corr_method)):
                    for hom in range(2):
                        middle_homo_file = os.path.join(ppath, f'matches_scale_{s}_{method[widx][0]}_sac_{th_sac}_err_{e[k,0]}_{e[k,1]}_{corr_method[j][0]}_hom_{hom}.mat')

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

                            sio.savemat(middle_homo_file, {'data': data})
                        else:
                            data = sio.loadmat(middle_homo_file)['data']