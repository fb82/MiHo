import numpy as np
import torch
import cv2
import sys
from config import get_config
torch.set_grad_enabled(False)

sys.path.append('../core')
from dematch import DeMatch

def norm_kp(cx, cy, fx, fy, kp):
    # New kp
    kp = (kp - np.array([[cx, cy]])) / np.asarray([[fx, fy]])
    return kp

def computeNN(desc_ii, desc_jj):
    desc_ii, desc_jj = torch.from_numpy(desc_ii), torch.from_numpy(desc_jj)
    d1 = (desc_ii**2).sum(1)
    d2 = (desc_jj**2).sum(1)
    distmat = (d1.unsqueeze(1) + d2.unsqueeze(0) - 2*torch.matmul(desc_ii, desc_jj.transpose(0,1))).sqrt()
    distVals, nnIdx1 = torch.topk(distmat, k=2, dim=1, largest=False)
    nnIdx1 = nnIdx1[:,0]
    _, nnIdx2 = torch.topk(distmat, k=1, dim=0, largest=False)
    nnIdx2= nnIdx2.squeeze()
    mutual_nearest = (nnIdx2[nnIdx1] == torch.arange(0, nnIdx1.shape[0]).long()).numpy()
    ratio_test = (distVals[:,0] / distVals[:,1].clamp(min=1e-10)).numpy()
    idx_sort = [np.arange(nnIdx1.shape[0]), nnIdx1.numpy()]
    return idx_sort, ratio_test, mutual_nearest

def draw_matching(img1, img2, pt1, pt2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1,h2), w1+w2, 3), np.uint8)
    vis[:h1, :w1] = img1

    vis[:h2, w1:w1+w2] = img2

    green = (0, 255, 0)
    red = (0, 0, 255)
    white = (255, 255, 255)
    thickness = 1
    num = 0

    for i in range(pt1.shape[0]):
        x1 = int(pt1[i, 0])
        y1 = int(pt1[i, 1])
        x2 = int(pt2[i, 0] + w1)
        y2 = int(pt2[i, 1])

        cv2.line(vis, (x1, y1), (x2, y2), green, int(thickness))
    return vis

class ExtractSIFT(object):
    def __init__(self, num_kp, contrastThreshold=1e-5):
        self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=num_kp, contrastThreshold=contrastThreshold)
        self.num_kp = num_kp
    def run(self, img):
        img = img.astype(np.uint8)
    #    img = cv2.imread(img)
        cv_kp, desc = self.sift.detectAndCompute(img, None)

        kp = np.array([[_kp.pt[0], _kp.pt[1]] for _kp in cv_kp]) # N*2

        return kp[:self.num_kp], desc[:self.num_kp]

def demo(opt, img1_path, img2_path):
    print("=======> Loading images")
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    print("=======> Generating initial matching")
    SIFT = ExtractSIFT(num_kp=2000)

    kpts1, desc1 = SIFT.run(img1)
    kpts2, desc2 = SIFT.run(img2)

    idx_sort, ratio_test, mutual_nearest = computeNN(desc1, desc2)

    kpts2 = kpts2[idx_sort[1],:]

    cx1 = (img1.shape[1] - 1.0) * 0.5
    cy1 = (img1.shape[0] - 1.0) * 0.5
    f1 = max(img1.shape[1] - 1.0, img1.shape[0] - 1.0)

    cx2 = (img2.shape[1] - 1.0) * 0.5
    cy2 = (img2.shape[0] - 1.0) * 0.5
    f2 = max(img2.shape[1] - 1.0, img2.shape[0] - 1.0)

    kpts1_n = norm_kp(cx1, cy1, f1, f1, kpts1)
    kpts2_n = norm_kp(cx2, cy2, f2, f2, kpts2)

    xs = np.concatenate([kpts1_n, kpts2_n], axis=-1)
    ys = np.ones(xs.shape[0])

    print("=======> Loading pretrained model")
    model = DeMatch(opt)
    checkpoint = torch.load('../pretrained-model/yfcc100m/model_best.pth', map_location=torch.device('cuda'))

    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()
    model.eval()

    xs = torch.from_numpy(xs).float().cuda()

    print("=======> Pruning")
    data = {}
    data['xs'] = xs.unsqueeze(0).unsqueeze(1)
    y_hat, e_hat = model(data)
    y = y_hat[-1][0, :].cpu().numpy()  
    matching = draw_matching(img1, img2, kpts1[y > opt.inlier_threshold], kpts2[y > opt.inlier_threshold])
    cv2.imwrite('./inliers.jpg', matching)

if __name__ == '__main__':
    opt, unparsed = get_config()

    img1_path = './test_img1.jpg'
    img2_path = './test_img2.jpg'

    demo(opt, img1_path, img2_path)
