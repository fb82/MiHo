import scipy.io as sio
from PIL import Image
import numpy as np


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
    # TO DO: last line in the matrix A can be removed for speeding the computation
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


# todo: preallocate numpy arrays instead of contatenation for faster computation ?
def cluster_assign(Hdata, l):
    """maybe we can change method, it would be also good to retain
    as Hdata[3] the reprojection error """
    midx = np.stack([Hdata[i][2] for i in range(len(Hdata))], axis=0)
    sidx = np.sum(midx, axis=1)
    return np.argmax(np.tile(sidx, (l, 1)).T * midx, axis=0)


class miho:
    def __init__(self, th_in = 7, th_out = 15, cluster_assign = cluster_assign):
        """initiate MiHo"""
        self.th_in = th_in
        self.th_out = th_out
        self.assign = cluster_assign


    def planar_clustering(self, pt1, pt2):
        """run MiHo"""
        self.pt1 = pt1
        self.pt2 = pt2
        Hdata = get_avg_hom(pt1, pt2, self.th_in, self.th_out)
#       retain all for now
#       Hdata = [entry[:2] for entry in Hdata]           
        self.Hs = Hdata
        self.Hidx = self.assign(Hdata, pt1.shape[0])
        return self.Hs, self.Hidx


    def show_clustering(self, im1, im2, saveto=None):
        """ todo: show MiHo clutering as in the paper"""
        if hasattr(self, 'Hs'):
            self.im1 = im1
            self.im2 = im2


if __name__ == '__main__':

    img1 = 'data/im1.png'
    img2 = 'data/im2.png'
    match_file = 'data/matches.mat'
    
    im1 = Image.open(img1)
    im2 = Image.open(img2)
    
    m12 = sio.loadmat(match_file, squeeze_me=True)
    m12 = m12['matches'][m12['midx'],:]

    mihoo = miho()
    mihoo.planar_clustering(m12[:,:2], m12[:,2:])
    mihoo.show_clustering(im1, im2)