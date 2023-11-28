import scipy.io as sio
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pickle


def data_normalize(pts):
    c = np.mean(pts, axis=1)
    s = np.sqrt(2) / (np.mean(np.sqrt((pts[0, :] - c[0])**2 + (pts[1, :] - c[1])**2)) + np.finfo(float).eps)

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
    # TODO: last line in the matrix A can be removed for speeding the computation
    A = np.vstack((
        np.hstack((np.zeros((l, 3)), -np.multiply(np.tile(npts2[2, :], (3, 1)).T, npts1.T), np.multiply(np.tile(npts2[1, :], (3, 1)).T, npts1.T))),
        np.hstack((np.multiply(np.tile(npts2[2, :], (3, 1)).T, npts1.T), np.zeros((l, 3)), -np.multiply(np.tile(npts2[0, :], (3, 1)).T, npts1.T))),
        np.hstack((-np.multiply(np.tile(npts2[1, :], (3, 1)).T, npts1.T), np.multiply(np.tile(npts2[0, :], (3, 1)).T, npts1.T), np.zeros((l, 3))))
    ))

    try:
        _, D, V = np.linalg.svd(A, full_matrices=True)        
        H = V[-1, :].reshape(3, 3).T
        H = np.linalg.inv(T2) @ H @ T1
    except:
        H = None
        D = np.zeros(9)

    return H, D


def get_inliers(pt1, pt2, H, ths, sidx):
    l = pt1.shape[1]

    pt2_ = np.dot(H, pt1)
    s2_ = np.sign(pt2_[2, :])
    tmp2_ = pt2_[:2, :] / pt2_[2, :] - pt2[:2, :]
    err2 = np.sum(tmp2_**2, axis=0)
    s2 = s2_[sidx[0]]

    if not np.all(s2_[sidx] == s2):
        nidx = np.zeros(l, dtype=bool)
        return nidx

    pt1_ = np.dot(np.linalg.inv(H), pt2)
    s1_ = np.sign(pt1_[2, :])
    tmp1_ = pt1_[:2, :] / pt1_[2, :] - pt1[:2, :]
    err1 = np.sum(tmp1_**2, axis=0)
    s1 = s1_[sidx[0]]

    if not np.all(s1_[sidx] == s1):
        nidx = np.zeros(l, dtype=bool)
        return nidx

    err = np.maximum(err1, err2)
    err[~np.isfinite(err)] = np.inf
    
    ths_ = [ths] if not isinstance(ths, list) else ths
    nidx = [np.all(np.vstack((err < th,s2_ == s2,s1_ == s1)), axis=0) for th in ths_]
    if not isinstance(ths, list): nidx = nidx[0]

    return nidx


def ransac_middle(pt1, pt2, th_in=7, th_out=15, max_iter=10000, min_iter=100, p=0.9, svd_th=0.05):
    n = pt1.shape[1]
    th_in = th_in ** 2
    th_out = th_out ** 2

    ptm = (pt1 + pt2) / 2

    if n < 4:
        H1 = np.array([])
        H2 = np.array([])
        iidx = np.zeros(n, dtype=bool)
        oidx = np.zeros(n, dtype=bool)
        return H1, H2, iidx, oidx

    min_iter = min(min_iter, int(np.math.comb(n, 2)))

    midx = np.zeros(n, dtype=bool)
    oidx = np.zeros(n, dtype=bool)
    sum_midx = 0
    Nc = np.Inf

    for c in range(1, max_iter):
        sidx = np.random.choice(n, size=4, replace=False)
        H1, eD = compute_homography(pt1[:, sidx], ptm[:, sidx])
        if eD[-2] < svd_th:
            continue
        H2, eD = compute_homography(pt2[:, sidx], ptm[:, sidx])
        if eD[-2] < svd_th:
            continue

        nidx = get_inliers(pt1, ptm, H1, th_out, sidx) * get_inliers(pt2, ptm, H2, th_out, sidx)
        sum_nidx = np.sum(nidx)
        if sum_nidx > sum_midx:
            midx = nidx
            sum_midx = sum_nidx
            sidx_ = sidx
            Nc = steps(4, sum_midx / n, p)
            if (c > Nc) and (c > min_iter):
                break

    if (sum_midx > 0):
        H1, _ = compute_homography(pt1[:, midx], ptm[:, midx])
        H2, _ = compute_homography(pt2[:, midx], ptm[:, midx])
        inl1 = get_inliers(pt1, ptm, H1, [th_in, th_out], sidx_)
        inl2 = get_inliers(pt2, ptm, H2, [th_in, th_out], sidx_)
        iidx = inl1[0] & inl2[0]
        oidx = inl1[1] & inl2[1]
    else:
        H1 = np.array([])
        H2 = np.array([])
        iidx = np.zeros(n, dtype=bool)
        oidx = np.zeros(n, dtype=bool)

    return H1, H2, iidx, oidx


def get_avg_hom(pt1, pt2, th_in=7, th_out=15, min_plane_pts=4, min_pt_gap=4, max_ref_iter=5, max_fail_count=2):
    H1 = np.eye(3)
    H2 = np.eye(3)

    Hdata = []
    l = pt1.shape[0]

    midx = np.zeros(l, dtype=bool)
    tidx = np.zeros(l, dtype=bool)

    pt1 = np.vstack((pt1.T, np.ones((1, l))))
    pt2 = np.vstack((pt2.T, np.ones((1, l))))

    Hdata = []

    fail_count = 0
    midx_sum = 0
    while (np.sum(midx) < l - 4):
        pt1_ = pt1[:, ~midx]
        pt1_ = np.dot(H1, pt1_)
        pt1_ = pt1_ / pt1_[2, :]

        pt2_ = pt2[:, ~midx]
        pt2_ = np.dot(H2, pt2_)
        pt2_ = pt2_ / pt2_[2, :]

        H1_, H2_, iidx, oidx = ransac_middle(pt1_, pt2_, th_in, th_out)
                        
        idx = np.zeros(l, dtype=bool)
        idx[~midx] = oidx

        midx[~midx] = iidx
        tidx = tidx | idx

        midx_sum_old = midx_sum
        midx_sum = np.sum(midx)
        
        if (np.sum(oidx) <= min_plane_pts) or (midx_sum - midx_sum_old <= min_pt_gap):
            fail_count+=1
            if fail_count > max_fail_count: break
            if midx_sum - midx_sum_old <= min_pt_gap: midx = tidx        
            if np.sum(oidx) <= min_plane_pts: continue
        else:
            fail_count = 0
       
        # print(f"{np.sum(tidx)} {np.sum(midx)} {fail_count}")
                        
        H1_new = np.dot(H1_, H1)
        H2_new = np.dot(H2_, H2)

        H1_ = np.eye(3)
        H2_ = np.eye(3)

        ptm_err_old = np.Inf

        for i in range(max_ref_iter):

            H1_new = np.dot(H1_, H1_new)
            H2_new = np.dot(H2_, H2_new)

            pt1_ = pt1[:, idx]
            pt1_ = np.dot(H1_new, pt1_)
            pt1_ = pt1_ / pt1_[2, :]

            pt2_ = pt2[:, idx]
            pt2_ = np.dot(H2_new, pt2_)
            pt2_ = pt2_ / pt2_[2, :]

            ptm_err = np.mean(np.sqrt(np.sum((pt1_ - pt2_)**2, axis=0)))
            if (ptm_err_old < ptm_err):
                break
            ptm_err_old = ptm_err

            ptm = (pt1_ + pt2_) / 2
            H1_, _ = compute_homography(pt1_, ptm)
            H2_, _ = compute_homography(pt2_, ptm)

        Hdata.append([H1_new, H2_new, idx])

    return Hdata


def cluster_assign_base(Hdata):
    l = len(Hdata)
    midx = np.hstack([Hdata[i][2][:, np.newaxis] for i in range(l)])
    qidx = np.sum(midx, axis=1)==0
    sidx = np.sum(midx, axis=0)
    vidx = np.argmax(np.repeat(sidx[np.newaxis, :], midx.shape[0], axis=0) * midx, axis=1)
    vidx[qidx] = -1
    return vidx


def cluster_assign(Hdata, pt1=None, pt2=None, median_th=5):
    l = len(Hdata)
    n = pt1.shape[0]

    pt1 = np.vstack((pt1.T, np.ones((1, n))))
    pt2 = np.vstack((pt2.T, np.ones((1, n))))

    eidx = np.zeros((n,l))

    for i in range(l):
        H1 = Hdata[i][0]
        H2 = Hdata[i][1]

        pt1_ = np.dot(H1, pt1)
        pt1_ = pt1_ / pt1_[2, :]

        pt2_ = np.dot(H2, pt2)
        pt2_ = pt2_ / pt2_[2, :]

        err = np.sqrt(np.sum((pt1_ - pt2_)**2, axis=0)).T
        eidx[:, i] = err

    inl_mak = np.hstack([Hdata[i][2][:, np.newaxis] for i in range(l)])
    sidx = np.sum(inl_mask, axis=0)
    vidx = np.repeat(sidx[np.newaxis, :], inl_mask.shape[0], axis=0) * inl_mask
 
    # take a cluster if its cardinality is more than the median of the top median_th ones
    zidx = -np.sort(-vidx, axis=1)
    pidx = np.sum(zidx[:,:median_th]>0, axis=1) / 2
    pidx[pidx==1] = 1.5
    pidx = np.maximum(np.ceil(pidx).astype(int)-1,0)    
    tidx = zidx.flatten()[np.ravel_multi_index([np.arange(n),pidx],zidx.shape)]        
    # take among the selected the one which gives less error
    xidx = vidx < tidx[:, np.newaxis]
    eidx[xidx] = np.Inf
    lidx = np.argmin(eidx,axis=1)    
    qidx = np.sum(inl_mask, axis=1)==0    
    lidx[qidx] = -1
    return lidx


def show_fig(im1, im2, pt1, pt2, Hdata, Hidx, tosave='miho.pdf', fig_dpi=300):
    im12 = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)))
    im12.paste(im1, (0, 0))
    im12.paste(im2, (im1.width, 0))

    plt.figure()
    plt.axis('off')            
    plt.imshow(im12)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#bcbd22', '#17becf']
    markers = ['o','x','8','p','h']
    bad_marker = 'd'
    bad_color = '#000000'
    cn = len(colors)
    mn = len(markers)

    plot_opt = {'markersize': 2, 'markeredgewidth': 0.5, 'markerfacecolor': "None", 'alpha': 0.5}    
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


class miho:
    def __init__(self, th_in=7, th_out=15, cluster_assign=cluster_assign, assign_args={}):
        """initiate MiHo"""
        self.th_in = th_in
        self.th_out = th_out
        self.assign = cluster_assign
        self.assign_args = assign_args


    def planar_clustering(self, pt1, pt2):
        """run MiHo"""
        self.pt1 = pt1
        self.pt2 = pt2
        Hdata = get_avg_hom(pt1, pt2, self.th_in, self.th_out)
        
        self.Hs = Hdata
        if self.assign == cluster_assign:
            self.assign_args = {'pt1': self.pt1, 'pt2': self.pt2}
        self.Hidx = self.assign(Hdata, **self.assign_args)
        return self.Hs, self.Hidx


    def show_clustering(self, im1, im2, saveto=None):
        """ todo: show MiHo clutering as in the paper"""
        if hasattr(self, 'Hs'):
            self.im1 = im1
            self.im2 = im2

            show_fig(im1, im2, self.pt1, self.pt2, self.Hs, self.Hidx)
            
            
if __name__ == '__main__':

    img1 = 'data/im1.png'
    img2 = 'data/im2.png'
    match_file = 'data/matches.mat'

    im1 = Image.open(img1)
    im2 = Image.open(img2)

    m12 = sio.loadmat(match_file, squeeze_me=True)
    m12 = m12['matches'][m12['midx'] > 0, :]

    mihoo = miho()
    mihoo.planar_clustering(m12[:, :2], m12[:, 2:])

    # with open('miho.pkl', 'wb') as file:      
    #     pickle.dump(mihoo, file) 

    # with open('miho.pkl', 'rb') as miho_pkl:       
    #     mihoo = pickle.load(miho_pkl) 

    mihoo.show_clustering(im1, im2)
