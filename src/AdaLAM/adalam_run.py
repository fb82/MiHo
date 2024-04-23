import cv2 as cv
import numpy as np
import argparse
import sys
import scipy.io as sio
import torch
import time
from adalam import AdalamFilter

def extract_keypoints(impath):
    im = cv.imread(impath, cv.IMREAD_COLOR)
    d = cv.xfeatures2d.SIFT_create(nfeatures=8000, contrastThreshold=1e-5)
    kp1, desc1 = d.detectAndCompute(im, mask=np.ones(shape=im.shape[:-1] + (1,),
                                                              dtype=np.uint8))
    pts = np.array([k.pt for k in kp1], dtype=np.float32)
    ors = np.array([k.angle for k in kp1], dtype=np.float32)
    scs = np.array([k.size for k in kp1], dtype=np.float32)
    return pts, ors, scs, desc1, im


def show_matches(img1, img2, k1, k2, target_dim=800.):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    def resize_horizontal(h1, w1, h2, w2, target_height):
        scale_to_align = float(h1) / h2
        current_width = w1 + w2 * scale_to_align
        scale_to_fit = target_height / h1
        target_w1 = int(w1 * scale_to_fit)
        target_w2 = int(w2 * scale_to_align * scale_to_fit)
        target_h = int(target_height)
        return (target_w1, target_h), (target_w2, target_h), scale_to_fit, scale_to_fit * scale_to_align, [target_w1, 0]

    target_1, target_2, scale1, scale2, offset = resize_horizontal(h1, w1, h2, w2, target_dim)

    im1 = cv.resize(img1, target_1, interpolation=cv.INTER_AREA)
    im2 = cv.resize(img2, target_2, interpolation=cv.INTER_AREA)

    h1, w1 = target_1[::-1]
    h2, w2 = target_2[::-1]

    vis = np.ones((max(h1, h2), w1 + w2, 3), np.uint8) * 255
    vis[:h1, :w1] = im1
    vis[:h2, w1:w1 + w2] = im2

    p1 = [np.int32(k * scale1) for k in k1]
    p2 = [np.int32(k * scale2 + offset) for k in k2]

    for (x1, y1), (x2, y2) in zip(p1, p2):
        cv.line(vis, (x1, y1), (x2, y2), [0, 255, 0], 1)

    cv.imshow("AdaLAM example", vis)
    cv.waitKey()


if __name__ == '__main__':
#   p = argparse.ArgumentParser()
#   p.add_argument("--im1", required=True)
#   p.add_argument("--im2", required=True)
#   opt = p.parse_args()

#   k1, o1, s1, d1, im1 = extract_keypoints(opt.im1)
#   k2, o2, s2, d2, im2 = extract_keypoints(opt.im2)

    aux = sio.loadmat(sys.argv[1])
    k1 = aux["data"]["k1"][0][0].astype(np.float32)
    k2 = aux["data"]["k2"][0][0].astype(np.float32)
    o1 = aux["data"]["o1"][0][0].astype(np.float32)
    o2 = aux["data"]["o2"][0][0].astype(np.float32)
    s1 = aux["data"]["s1"][0][0].astype(np.float32)
    s2 = aux["data"]["s2"][0][0].astype(np.float32)                    
    im1shape = aux["data"]["im1shape"][0][0][0].astype(np.int64)
    im2shape = aux["data"]["im2shape"][0][0][0].astype(np.int64)  
    im1shape =  im1shape[0], im1shape[1]                  
    im2shape =  im2shape[0], im2shape[1]  

    putative_matches = aux["data"]["putative_matches"][0][0].astype(np.int64)
    scores = aux["data"]["scores"][0][0].astype(np.float32)                    
    mnn = aux["data"]["mnn"][0][0].astype(np.bool)                    

    th = aux["data"]["th"][0][0][0][0]

    t1 = time.time()
    matcher = AdalamFilter()
    matcher.config['th']=th
    matches = matcher.match_and_filter(k1=k1, k2=k2,
                                       o1=o1, o2=o2,
                                       d1=None, d2=None,
                                       s1=s1, s2=s2,
                                       im1shape=im1shape, im2shape=im2shape, putative_matches=putative_matches, scores=scores, mnn=mnn).cpu().numpy()                                       
#                                      im1shape=im1.shape[:2], im2shape=im2.shape[:2]).cpu().numpy()
    t2 = time.time() 
    
    if matches.size == 0:
        np.savetxt(sys.argv[2],np.array([-1]), fmt='%d', delimiter=' ') 
    else:
        np.savetxt(sys.argv[2],matches, fmt='%d', delimiter=' ') 
    np.savetxt(sys.argv[3], np.array([t2-t1]), fmt='%.6f', delimiter='\n')

#   show_matches(im1, im2, k1=k1[matches[:, 0]], k2=k2[matches[:, 1]])






