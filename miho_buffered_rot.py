from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io as sio
import warnings

def data_normalize(pts):

    c = np.mean(pts, axis=1)
    s = np.sqrt(2) / (np.mean(np.sqrt((pts[0, :] - c[0])**2 + (pts[1, :] - c[1])**2)) +
                      np.finfo(float).eps)

    T = np.array([
        [s, 0, -c[0] * s],
        [0, s, -c[1] * s],
        [0, 0, 1]
    ])

    return T


def steps(pps, inl, p):

    e = 1 - inl
    r = np.log(1 - p) / np.log(1 - (1 - e)**pps)

    return r


def compute_homography(pts1, pts2):

    T1 = data_normalize(pts1)
    T2 = data_normalize(pts2)

    npts1 = np.dot(T1, pts1)
    npts2 = np.dot(T2, pts2)

    l = npts1.shape[1]
    A = np.zeros((l*3,9))
    A[:l,3:6] = -np.multiply(np.tile(npts2[2, :], (3, 1)).T, npts1.T)
    A[:l,6:] = np.multiply(np.tile(npts2[1, :], (3, 1)).T, npts1.T)
    A[l:2*l,:3] = np.multiply(np.tile(npts2[2, :], (3, 1)).T, npts1.T)
    A[l:2*l,6:] = -np.multiply(np.tile(npts2[0, :], (3, 1)).T, npts1.T)
    # TODO: last block in the matrix A can be removed for speeding the computation
    A[2*l:,:3] = -np.multiply(np.tile(npts2[1, :], (3, 1)).T, npts1.T)
    A[2*l:,3:6] = np.multiply(np.tile(npts2[0, :], (3, 1)).T, npts1.T)

    # A = np.vstack((
    #     np.hstack((np.zeros((l, 3)), -np.multiply(np.tile(npts2[2, :], (3, 1)).T, npts1.T), np.multiply(np.tile(npts2[1, :], (3, 1)).T, npts1.T))),
    #     np.hstack((np.multiply(np.tile(npts2[2, :], (3, 1)).T, npts1.T), np.zeros((l, 3)), -np.multiply(np.tile(npts2[0, :], (3, 1)).T, npts1.T))),
    #     np.hstack((-np.multiply(np.tile(npts2[1, :], (3, 1)).T, npts1.T), np.multiply(np.tile(npts2[0, :], (3, 1)).T, npts1.T), np.zeros((l, 3))))
    # ))

    try:
        _, D, V = np.linalg.svd(A, full_matrices=True)
        H = V[-1, :].reshape(3, 3).T
        H = np.linalg.inv(T2) @ H @ T1
    except:
        H = None
        D = np.zeros(9)

    return H, D


def sampler(n,k,m):
    p = np.floor(np.random.rand(m,k) * np.arange(n,n-k,-1)[np.newaxis, :]).astype(int)  
    for i in range(1,k):
        p[:, 0:i] = np.sort(p[:, 0:i])
        for j in range(i):
            p[:,i] = p[:,i] + (p[:,i] >= p[:,j])

    return p
    

def get_inliers(pt1, pt2, H, ths, sidx):

    l = pt1.shape[1]
    ths_ = [ths] if not isinstance(ths, list) else ths
    m = len(ths_)

    pt2_ = np.dot(H, pt1)
    s2_ = np.sign(pt2_[2, :])
    s2 = s2_[sidx[0]]

    if not np.all(s2_[sidx] == s2):
        nidx = np.zeros((m, l), dtype=bool)
        if not isinstance(ths, list): nidx = nidx[0]
        return nidx

    tmp2_ = pt2_[:2, :] / pt2_[2, :] - pt2[:2, :]
    err2 = np.sum(tmp2_**2, axis=0)

    pt1_ = np.dot(np.linalg.inv(H), pt2)
    s1_ = np.sign(pt1_[2, :])
    s1 = s1_[sidx[0]]

    if not np.all(s1_[sidx] == s1):
        nidx = np.zeros((m, l), dtype=bool)
        if not isinstance(ths, list): nidx = nidx[0]
        return nidx

    tmp1_ = pt1_[:2, :] / pt1_[2, :] - pt1[:2, :]
    err1 = np.sum(tmp1_**2, axis=0)

    err = np.maximum(err1, err2)
    err[~np.isfinite(err)] = np.inf

    nidx = np.zeros((m, l), dtype=bool)
    for i in range(m):
        nidx[i, :] = np.all(np.vstack((err < ths_[i], s2_ == s2,s1_ == s1)), axis=0)

    if not isinstance(ths, list): nidx = nidx[0]

    return nidx


def get_error(pt1, pt2, H, sidx):

    l = pt1.shape[1]

    pt2_ = np.dot(H, pt1)
    s2_ = np.sign(pt2_[2, :])
    s2 = s2_[sidx[0]]

    if not np.all(s2_[sidx] == s2):
        return np.full(l, np.inf)

    tmp2_ = pt2_[:2, :] / pt2_[2, :] - pt2[:2, :]
    err2 = np.sum(tmp2_**2, axis=0)

    pt1_ = np.dot(np.linalg.inv(H), pt2)
    s1_ = np.sign(pt1_[2, :])
    s1 = s1_[sidx[0]]

    if not np.all(s1_[sidx] == s1):
        return np.full(l, np.Inf)

    tmp1_ = pt1_[:2, :] / pt1_[2, :] - pt1[:2, :]
    err1 = np.sum(tmp1_**2, axis=0)

    err = np.maximum(err1, err2)
    err[~np.isfinite(err)] = np.inf

    return err


def ransac_middle(pt1, pt2, dd, th_in=7, th_out=15, max_iter=500, min_iter=50, p=0.9, svd_th=0.05, buffers=5, ssidx=None):

    n = pt1.shape[1]

    th_in = th_in ** 2
    th_out = th_out ** 2

    ptm = (pt1 + pt2) / 2

    if n < 4:
        H1 = np.array([])
        H2 = np.array([])
        iidx = np.zeros(n, dtype=bool)
        oidx = np.zeros(n, dtype=bool)
        vidx = np.zeros((n, 0), dtype=bool)
        sidx_ = np.zeros((4,), dtype=int)
        return H1, H2, iidx, oidx, vidx, sidx_

    min_iter = min(min_iter, n*(n-1)*(n-2)*(n-3) / 12)

    vidx = np.zeros((n, buffers), dtype=bool)
    midx = np.zeros((n, buffers+1), dtype=bool)

    sum_midx = 0
    Nc = np.Inf
    min_th_stats = 3

    sn = ssidx.shape[1]
    sidx_ = np.zeros((4,), dtype=int)

    for c in range(1, max_iter):

        good_sample = False
        for i in range(min_iter):
            if c < sn:
                aux = np.argwhere(ssidx[:, c])
                if aux.shape[0] > 4:
                    aux_idx = np.random.choice(aux.shape[0], size=4, replace=False)
                    sidx = np.squeeze(aux[aux_idx])
                else:
                    sidx = np.random.choice(n, size=4, replace=False)
            else:
                sidx = np.random.choice(n, size=4, replace=False)

            if np.all(np.sum(dd[sidx, :][:, sidx], axis=0) >= 3):
                good_sample = True
                break

        if not good_sample:
            break

        H1, eD = compute_homography(pt1[:, sidx], ptm[:, sidx])
        if eD[-2] < svd_th:
            if (c > Nc) and (c > min_iter):
                break
            else:
                continue

        H2, eD = compute_homography(pt2[:, sidx], ptm[:, sidx])
        if eD[-2] < svd_th:
            if (c > Nc) and (c > min_iter):
                break
            else:
                continue

        nidx = get_inliers(pt1, ptm, H1, th_out, sidx) * get_inliers(pt2, ptm, H2, th_out, sidx)

        updated_model = False

        midx[:,-1] = nidx
        sum_nidx = np.sum(nidx)

        if sum_nidx > min_th_stats:

            idxs = np.arange(buffers+1)
            q = n+1

            for t in range(buffers):
                uidx = np.logical_not(np.any(midx[:, idxs[:t]], axis=1)[:, np.newaxis])

                tsum = uidx & midx[:, idxs[t:]]
                ssum = np.sum(tsum, axis=0)
                vidx[:, t] = tsum[:, np.argmax(ssum)]

                tt = np.argmax(ssum[-1] > ssum[:-1])
                if (ssum[-1] > ssum[tt]):
                    aux = idxs[-1]
                    idxs[-1] = idxs[t+tt]
                    idxs[t+tt] = aux
                    if (t==0) and (tt==0):
                        sidx_ = sidx

                q = np.minimum(q, np.max(ssum))

            min_th_stats = np.maximum(4, q)

            updated_model = idxs[0] != 0
            midx = midx[:, idxs]

        if updated_model:
            sum_midx = np.sum(midx[:, 0])
            best_model = [H1, H2]
            Nc = steps(4, sum_midx / n, p)

        if (c > Nc) and (c > min_iter):
            break

    vidx = vidx[:,1:]

    if sum_midx >= 4:
        bidx = midx[:, 0]
        H1, _ = compute_homography(pt1[:, bidx], ptm[:, bidx])
        H2, _ = compute_homography(pt2[:, bidx], ptm[:, bidx])

        inl1 = get_inliers(pt1, ptm, H1, [th_in, th_out], sidx_)
        inl2 = get_inliers(pt2, ptm, H2, [th_in, th_out], sidx_)
        iidx = inl1[0] & inl2[0]
        oidx = inl1[1] & inl2[1]

        if sum_midx > np.sum(oidx):
            H1, H2 = best_model

            inl1 = get_inliers(pt1, ptm, H1, [th_in, th_out], sidx_)
            inl2 = get_inliers(pt2, ptm, H2, [th_in, th_out], sidx_)
            iidx = inl1[0] & inl2[0]
            oidx = inl1[1] & inl2[1]

    else:
        H1 = np.array([])
        H2 = np.array([])

        iidx = np.zeros(n, dtype=bool)
        oidx = np.zeros(n, dtype=bool)

    return H1, H2, iidx, oidx, vidx, sidx_


def rot_best(pt1, pt2, n=4):    
    # start = time.time()    

    n = int(n)
    if n < 1: n = 1
    
    # current MiHo formulation is translation invariant    
    pt1 = pt1[:2,:]
    pt2 = pt2[:2,:]    

    a = 2 * np.pi / n
    R = np.asarray([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

    M = np.eye(2)
    
    d0 = dist2(pt1.T)
    d2 = dist2(pt2.T)
    
    sum_best = -np.inf
    R_best = np.eye(3)
    
    for i in range(n):
        pt2_ = M @ pt2
        ptm = (pt1 + pt2_) / 2

        d1 = dist2(ptm.T)
                
        in_middle = (np.sign(d0 - d1) * np.sign(d1 - d2)) > 0
        sum_k = np.sum(in_middle)        
        
        # print(f"{i} {sum_k}")
        
        if sum_k > sum_best:
            sum_best = sum_k
            R_best = M
        
        M = R @ M
        
    aux = np.eye(3)
    aux[:2, :2] = R_best
              
    # end = time.time()
    # print("Elapsed rot_best time = %s" % (end - start))       
    
    return aux

def get_avg_hom(pt1, pt2, ransac_middle_args= {}, min_plane_pts=4, min_pt_gap=4,
                max_fail_count=3, random_seed_init=123, th_grid=15,
                rot_check=4):

    # set to 123 for debugging and profiling
    if random_seed_init is not None:
        np.random.seed(random_seed_init)

    H1 = np.eye(3)
    H2 = np.eye(3)

    Hdata = []
    l = pt1.shape[0]

    midx = np.zeros(l, dtype=bool)
    tidx = np.zeros(l, dtype=bool)

    d1 = dist2(pt1) > th_grid**2
    d2 = dist2(pt2) > th_grid**2
    dd = d1 & d2

    pt1 = np.vstack((pt1.T, np.ones((1, l))))
    pt2 = np.vstack((pt2.T, np.ones((1, l))))

    if rot_check > 1:
        H2 = rot_best(pt1, pt2, rot_check)

    fail_count = 0
    midx_sum = 0
    ssidx = np.zeros((l,0), dtype=bool)
    sidx = np.arange(l)

    while (np.sum(midx) < l - 4):
        pt1_ = pt1[:, ~midx]
        pt1_ = np.dot(H1, pt1_)
        pt1_ = pt1_ / pt1_[2, :]

        pt2_ = pt2[:, ~midx]
        pt2_ = np.dot(H2, pt2_)
        pt2_ = pt2_ / pt2_[2, :]

        dd_ = dd[~midx, :][:, ~midx]

        ssidx = ssidx[~midx, :]

        H1_, H2_, iidx, oidx, ssidx, sidx_ = ransac_middle(pt1_, pt2_, dd_, ssidx=ssidx, **ransac_middle_args)

        sidx_ = sidx[~midx][sidx_]

        # print(np.sum(ssidx, axis=0))
        good_ssidx = np.logical_not(np.sum(ssidx, axis=0) == 0)
        ssidx = ssidx[:, good_ssidx]
        tsidx = np.zeros((l, ssidx.shape[1]), dtype=bool)
        tsidx[~midx,:] = ssidx
        ssidx = tsidx

        idx = np.zeros(l, dtype=bool)
        idx[~midx] = oidx

        midx[~midx] = iidx
        tidx = tidx | idx

        midx_sum_old = midx_sum
        midx_sum = np.sum(midx)

        H_failed = np.sum(oidx) <= min_plane_pts
        inl_failed = midx_sum - midx_sum_old <= min_pt_gap
        if H_failed or inl_failed:
            fail_count+=1
            if fail_count > max_fail_count: break
            if inl_failed: midx = tidx
            if H_failed: continue
        else:
            fail_count = 0

        # print(f"{np.sum(tidx)} {np.sum(midx)} {fail_count}")

        Hdata.append([H1_ @ H1, H2_ @ H2, idx, sidx_])

    return Hdata, H1, H2


def dist2(pt):

    pt = pt.astype(np.single)
    d = (pt[:, 0, np.newaxis] - pt[np.newaxis :, 0])**2 + (pt[:, 1, np.newaxis] - pt[np.newaxis :, 1])**2

    return d


def cluster_assign_base(Hdata, pt1, pt2, H1_pre, H2_pre, **dummy_args):
    l = len(Hdata)
    n = Hdata[0][2].shape[0]

    inl_mask = np.zeros((n, l), dtype=bool)
    for i in range(l): inl_mask[:, i] = Hdata[i][2]

    alone_idx = np.sum(inl_mask, axis=1)==0
    set_size = np.sum(inl_mask, axis=0)

    max_size_idx = np.argmax(
        np.repeat(set_size[np.newaxis, :], inl_mask.shape[0], axis=0) * inl_mask, axis=1)
    max_size_idx[alone_idx] = -1

    return max_size_idx


def cluster_assign(Hdata, pt1, pt2, H1_pre, H2_pre, median_th=5, err_th=15, **dummy_args):
    l = len(Hdata)
    n = pt1.shape[0]

    pt1 = np.vstack((pt1.T, np.ones((1, n))))
    pt2 = np.vstack((pt2.T, np.ones((1, n))))

    pt1_ = np.dot(H1_pre, pt1)
    pt1_ = pt1_ / pt1_[2, :]

    pt2_ = np.dot(H2_pre, pt2)
    pt2_ = pt2_ / pt2_[2, :]

    ptm = (pt1_ + pt2_) / 2

    err = np.zeros((n,l))

    for i in range(l):
        H1 = Hdata[i][0]
        H2 = Hdata[i][1]
        sidx = Hdata[i][3]

        err[:, i] = np.maximum(get_error(pt1, ptm, H1, sidx), get_error(pt2, ptm, H2, sidx))

    # min error
    abs_err_min_val = np.min(err, axis=1)
    abs_err_min_idx = np.argmin(err, axis=1)

    inl_mask = np.zeros((n, l), dtype=bool)
    for i in range(l): inl_mask[:, i] = Hdata[i][2]

    set_size = np.sum(inl_mask, axis=0)
    size_mask = np.repeat(set_size[np.newaxis, :], inl_mask.shape[0], axis=0) * inl_mask

    # take a cluster if its cardinality is more than the median of the top median_th ones
    ssize_mask = -np.sort(-size_mask, axis=1)

    median_idx = np.sum(ssize_mask[:, :median_th]>0, axis=1) / 2
    median_idx[median_idx==1] = 1.5
    median_idx = np.maximum(np.ceil(median_idx).astype(int)-1, 0)

    top_median = ssize_mask.flatten(
        )[np.ravel_multi_index([np.arange(n), median_idx], ssize_mask.shape)]

    # take among the selected the one which gives less error
    discarded_mask = size_mask < top_median[:, np.newaxis]
    err[discarded_mask] = np.Inf
    err_min_idx = np.argmin(err, axis=1)

    # remove match with no cluster
    alone_idx = np.sum(inl_mask, axis=1)==0
    really_alone_idx = alone_idx & (abs_err_min_val > err_th**2)

    err_min_idx[alone_idx] = abs_err_min_idx[alone_idx]
    err_min_idx[really_alone_idx] = -1

    return err_min_idx


def cluster_assign_other(Hdata, pt1, pt2, H1_pre, H2_pre, err_th_only=15, **dummy_args):
    l = len(Hdata)
    n = pt1.shape[0]

    pt1 = np.vstack((pt1.T, np.ones((1, n))))
    pt2 = np.vstack((pt2.T, np.ones((1, n))))

    pt1_ = np.dot(H1_pre, pt1)
    pt1_ = pt1_ / pt1_[2, :]

    pt2_ = np.dot(H2_pre, pt2)
    pt2_ = pt2_ / pt2_[2, :]

    ptm = (pt1_ + pt2_) / 2

    err = np.zeros((n,l))

    for i in range(l):
        H1 = Hdata[i][0]
        H2 = Hdata[i][1]
        sidx = Hdata[i][3]

        err[:, i] = np.maximum(get_error(pt1, ptm, H1, sidx), get_error(pt2, ptm, H2, sidx))

    err_min_idx = np.argmin(err, axis=1)
    err_min_val = err.flatten()[np.ravel_multi_index([np.arange(n), err_min_idx], err.shape)]

    err_min_idx[(err_min_val > err_th_only**2) | np.isnan(err_min_val)] = -1

    return err_min_idx


def show_fig(im1, im2, pt1, pt2, Hdata, Hidx, tosave='miho_buffered_rot.pdf', fig_dpi=300,
             colors = ['#FF1F5B', '#00CD6C', '#009ADE', '#AF58BA', '#FFC61E', '#F28522'],
             markers = ['o','x','8','p','h'], bad_marker = 'd', bad_color = '#000000',
             plot_opt = {'markersize': 2, 'markeredgewidth': 0.5,
                         'markerfacecolor': "None", 'alpha': 0.5}):

    im12 = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)))
    im12.paste(im1, (0, 0))
    im12.paste(im2, (im1.width, 0))

    plt.figure()
    plt.axis('off')
    plt.imshow(im12)

    cn = len(colors)
    mn = len(markers)

    for i, idx in enumerate(np.ndarray.tolist(np.unique(Hidx))):
        mask = Hidx == idx
        x = np.vstack((pt1[mask, 0], pt2[mask, 0]+im1.width))
        y = np.vstack((pt1[mask, 1], pt2[mask, 1]))
        if (idx == -1):
            color = bad_color
            marker = bad_marker
        else:
            color = colors[((i-1)%(cn*mn))%cn]
            marker = markers[((i-1)%(cn*mn))//cn]
        plt.plot(x, y, linestyle='', color=color, marker=marker, **plot_opt)

    plt.savefig(tosave, dpi = fig_dpi, bbox_inches='tight')


def go_assign(Hdata, pt1, pt2, H1_pre, H2_pre, method=cluster_assign, method_args={}):
    return method(Hdata, pt1, pt2, H1_pre, H2_pre, **method_args)


def purge_default(dict1, dict2):

    if type(dict1) != type(dict2):
        return None

    if not (isinstance(dict2, dict) or isinstance(dict2, list)):
        if dict1 == dict2:
            return None
        else:
            return dict2

    elif isinstance(dict2, dict):

        keys1 = list(dict1.keys())
        keys2 = list(dict2.keys())

        for i in keys2:
            if i not in keys1:
                dict2.pop(i, None)
            else:
                aux = purge_default(dict1[i], dict2[i])
                if not aux:
                    dict2.pop(i, None)
                else:
                    dict2[i] = aux

        return dict2

    elif isinstance(dict2, list):

        if len(dict1) != len(dict2):
            return dict2

        for i in range(len(dict2)):
            aux = purge_default(dict1[i], dict2[i])
            if aux:
               return dict2

        return None


def merge_params(dict1, dict2):

    keys1 = dict1.keys()
    keys2 = dict2.keys()

    for i in keys1:
        if (i in keys2):
            if (not isinstance(dict1[i], dict)):
                dict1[i] = dict2[i]
            else:
                dict1[i] = merge_params(dict1[i], dict2[i])

    return dict1


class miho:
    def __init__(self, params=None):
        """initiate MiHo"""
        self.set_default()

        if params is not None:
            self.update_params(params)


    def set_default(self):
        """set default MiHo parameters"""
        self.params = { 'get_avg_hom': {}, 'go_assign': {}, 'show_clustering': {}}


    def get_current(self):
        """get current MiHo parameters"""
        tmp_params = self.params.copy()

        for i in ['get_avg_hom', 'show_clustering', 'go_assign']:
            if i not in tmp_params:
                tmp_params[i] = {}

        return merge_params(self.all_params(), tmp_params)


    def update_params(self, params):
        """update current MiHo parameters"""
        all_default_params = self.all_params()
        clear_params = purge_default(all_default_params, params.copy())

        for i in ['get_avg_hom', 'show_clustering', 'go_assign']:
            if i in clear_params:
                self.params[i] = clear_params[i]


    @staticmethod
    def all_params():
        """all MiHo parameters with default values"""
        ransac_middle_params = {'th_in': 7, 'th_out': 15, 'max_iter': 500,
                                'min_iter': 50, 'p' :0.9, 'svd_th': 0.05,
                                'buffers': 5}
        get_avg_hom_params = {'ransac_middle_args': ransac_middle_params,
                              'min_plane_pts': 4, 'min_pt_gap': 4,
                              'max_fail_count': 3, 'random_seed_init': 123,
                              'th_grid': 15, 'rot_check': 4}

        method_args_params = {'median_th': 5, 'err_th': 15, 'err_th_only': 15}
        go_assign_params = {'method': cluster_assign,
                            'method_args': method_args_params}

        show_clustering_params = {'tosave': 'miho_buffered_rot.pdf', 'fig_dpi': 300,
             'colors': ['#FF1F5B', '#00CD6C', '#009ADE', '#AF58BA', '#FFC61E', '#F28522'],
             'markers': ['o','x','8','p','h'], 'bad_marker': 'd', 'bad_color': '#000000',
             'plot_opt': {'markersize': 2, 'markeredgewidth': 0.5,
                          'markerfacecolor': "None", 'alpha': 0.5}}

        return {'get_avg_hom': get_avg_hom_params,
                'go_assign': go_assign_params,
                'show_clustering': show_clustering_params}


    def planar_clustering(self, pt1, pt2):
        """run MiHo"""
        self.pt1 = pt1
        self.pt2 = pt2

        Hdata, H1_pre, H2_pre = get_avg_hom(pt1, pt2, **self.params['get_avg_hom'])

        self.Hs = Hdata
        self.H1_pre = H1_pre
        self.H2_pre = H2_pre

        self.Hidx = go_assign(Hdata, pt1, pt2, H1_pre, H2_pre, **self.params['go_assign'])

        return self.Hs, self.Hidx


    def show_clustering(self, im1, im2):
        """ show MiHo clutering"""
        if hasattr(self, 'Hs'):
            self.im1 = im1
            self.im2 = im2

            show_fig(im1, im2, self.pt1, self.pt2, self.Hs, self.Hidx,
                     **self.params['show_clustering'])
        else:
            warnings.warn("planar_clustering must run before!!!")


if __name__ == '__main__':

    # start = time.time()
    # for t in range(500):
    #     sampler(8000,4,50)        
    # end = time.time()
    # print("Elapsed = %s" % (end - start))

    # start = time.time()
    # for t in range(500):
    #     tmp = np.zeros((50,4)).astype(int)
    #     for i in range(50):
    #         tmp[i,:] = np.random.choice(8000, size=4, replace=False)         
    # end = time.time()
    # print("Elapsed = %s" % (end - start))

    img1 = 'data/im1.png'
    img2 = 'data/im2_rot.png'
    match_file = 'data/matches_rot.mat'

    im1 = Image.open(img1)
    im2 = Image.open(img2)

    m12 = sio.loadmat(match_file, squeeze_me=True)
    m12 = m12['matches'][m12['midx'] > 0, :]

    start = time.time()

    params = miho.all_params()
    params['get_avg_hom']['rot_check'] = True
    mihoo = miho(params)

    # mihoo = miho(params)

    # params = miho.all_params()
    # params['go_assign']['method'] = cluster_assign_base
    # params['go_assign']['method_args']['err_th'] = 16
    # mihoo = miho(params)

    # params = mihoo.get_current()
    # params['get_avg_hom']['min_plane_pts'] = 16
    # mihoo.update_params(params)

    mihoo.planar_clustering(m12[:, :2], m12[:, 2:])
    # mihoo.planar_clustering(m12[:, :2], m12[:, 2:])

    end = time.time()
    print("Elapsed = %s" % (end - start))

    # import pickle
    #
    # with open('miho.pkl', 'wb') as file:
    #     pickle.dump(mihoo, file)
    #
    # with open('miho.pkl', 'rb') as miho_pkl:
    #     mihoo = pickle.load(miho_pkl)

    mihoo.show_clustering(im1, im2)