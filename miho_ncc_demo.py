from PIL import Image
import time
import torch
import kornia as K
from src import miho as miho
from src import ncc as ncc
import scipy.io as sio


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    # demo code
    load_matches = False # used pre-computed matches without LAF
    ncc_check = False    # img2 patches are img1 patches randomly traslated, only for testing NCC / NCC+
    no_miho = False      # compute NCC / NCC+ on LAF without MiHo
        
    img1 = 'data/im1.png'
    img2 = 'data/im2_rot.png'
    if load_matches: match_file = 'data/matches_rot.mat'

    # *** NCC / NCC+ ***
    # window radius
    w = 10
    w_big = 15
    # filter outliers by MiHo
    remove_bad=False
    # NCC+ patch angle offset
    angle=[-30, -15, 0, 15, 30]
    # NCC+ patch anisotropic scales
    scale=[[10/14, 1], [10/12, 1], [1, 1], [1, 12/10], [1, 14/10]]

    im1 = Image.open(img1)
    im2 = Image.open(img2)

    if not load_matches:
    # generate matches with kornia, LAF included, check upright!
        upright=False
        with torch.inference_mode():
            detector = K.feature.KeyNetAffNetHardNet(upright=upright, device=device)
            kps1, _ , descs1 = detector(K.io.load_image(img1, K.io.ImageLoadType.GRAY32, device=device).unsqueeze(0))
            kps2, _ , descs2 = detector(K.io.load_image(img2, K.io.ImageLoadType.GRAY32, device=device).unsqueeze(0))
            dists, idxs = K.feature.match_smnn(descs1.squeeze(), descs2.squeeze(), 0.99)        
        kps1 = kps1.squeeze().detach()[idxs[:, 0]].to(device)
        kps2 = kps2.squeeze().detach()[idxs[:, 1]].to(device)
    else:
    # import from a match file with only kpts    
        m12 = sio.loadmat(match_file, squeeze_me=True)
        m12 = m12['matches'][m12['midx'] > 0, :]
        # m12 = m12['matches']
        pt1 = torch.tensor(m12[:, :2], dtype=torch.float32, device=device)
        pt2 = torch.tensor(m12[:, 2:], dtype=torch.float32, device=device)

    params = miho.miho.all_params()
    params['get_avg_hom']['rot_check'] = 4
    mihoo = miho.miho(params)

    # miho paramas examples:
    #
    # params = miho.all_params()
    # params['go_assign']['method'] = cluster_assign_base
    # params['go_assign']['method_args']['err_th'] = 16
    # mihoo = miho(params)
    #
    # but also:
    #
    # params = mihoo.get_current()
    # params['get_avg_hom']['min_plane_pts'] = 16
    # mihoo.update_params(params)

    mihoo.attach_images(im1, im2)

    if ncc_check:
    # offset kpt shift, for testing
        if not load_matches:
            pt1, pt2, Hs_laf = ncc.refinement_laf(mihoo.im1, mihoo.im2, data1=kps1, data2=kps2, w=w, img_patches=False)    
        else:
              pt1, pt2, Hs_laf = ncc.refinement_laf(mihoo.im1, mihoo.im1, pt1=pt1, pt2=pt2, w=w, img_patches=False)      
        pt1 = pt1.round()
        if w_big is None:
            ww_big = w * 2
        else:
            ww_big = w_big
        test_idx = (torch.rand((pt1.shape[0], 2), device=device) * (((ww_big-w) * 2) - 1) - (ww_big-w-1)).round()    
        pt2 = pt1 + test_idx
        pt1, pt2, Hs_laf = ncc.refinement_laf(mihoo.im1, mihoo.im1, pt1=pt1, pt2=pt2, w=w, img_patches=True)    
        # pt1__, pt2__, Hs_ncc, val, T = ncc.refinement_norm_corr(mihoo.im1, mihoo.im1, pt1, pt2, Hs_laf, w=w, ref_image=['both'], subpix=True, img_patches=True)   
        pt1__p, pt2__p, Hs_ncc_p, val_p, T_p = ncc.refinement_norm_corr_alternate(mihoo.im1, mihoo.im1, pt1, pt2, Hs_laf, w=w, w_big=w_big, ref_image=['both'], subpix=True, img_patches=True)   
    else:
    # data formatting for NCC / NCC+
        if not load_matches:
            pt1, pt2, Hs_laf = ncc.refinement_laf(mihoo.im1, mihoo.im2, data1=kps1, data2=kps2, w=w, img_patches=True)    
        else:
            pt1, pt2, Hs_laf = ncc.refinement_laf(mihoo.im1, mihoo.im2, pt1=pt1, pt2=pt2, w=w, img_patches=True)    

    ### MiHo
    start = time.time()
    
    mihoo.planar_clustering(pt1, pt2)

    end = time.time()
    print("Elapsed = %s (MiHo clustering)" % (end - start))

    # *** MiHo inlier mask ***
    good_matches = mihoo.Hidx > -1  
  
    
    ### NCC / NCC+
    start = time.time()

    if ncc_check:        
    # offset kpt shift, for testing - LAF -> NCC | NCC+
        pt1__, pt2__, Hs_ncc, val, T = ncc.refinement_norm_corr(mihoo.im1, mihoo.im1, pt1, pt2, Hs_laf, w=w, ref_image=['both'], subpix=True, img_patches=True)   
        pt1__p, pt2__p, Hs_ncc_p, val_p, T_p = ncc.refinement_norm_corr_alternate(mihoo.im1, mihoo.im1, pt1, pt2, Hs_laf, w=w, ref_image=['both'], angle=angle, scale=scale, subpix=True, img_patches=True)   
    else:     
        if no_miho:
        # LAF -> NCC | NCC+
            pt1__, pt2__, Hs_ncc, val, T = ncc.refinement_norm_corr(mihoo.im1, mihoo.im2, pt1, pt2, Hs_laf, w=w, ref_image=['both'], subpix=True, img_patches=True)   
            pt1__p, pt2__p, Hs_ncc_p, val_p, T_p = ncc.refinement_norm_corr_alternate(mihoo.im1, mihoo.im2, pt1, pt2, Hs_laf, w=w, ref_image=['both'], angle=angle, scale=scale, subpix=True, img_patches=True)   
        else:
        # LAF -> MiHo -> NCC | NCC+   
            pt1_, pt2_, Hs_miho, inliers = ncc.refinement_miho(mihoo.im1, mihoo.im2, pt1, pt2, mihoo, Hs_laf, remove_bad=remove_bad, w=w, img_patches=True)        
            pt1__, pt2__, Hs_ncc, val, T = ncc.refinement_norm_corr(mihoo.im1, mihoo.im2, pt1_, pt2_, Hs_miho, w=w, ref_image=['both'], subpix=True, img_patches=True)   
            pt1__p, pt2_p_, Hs_ncc_p, val_p, T_p = ncc.refinement_norm_corr_alternate(mihoo.im1, mihoo.im2, pt1_, pt2_, Hs_miho, w=w, ref_image=['both'], angle=angle, scale=scale, subpix=True, img_patches=True)   
    
    end = time.time()
    print("Elapsed = %s (NCC refinement)" % (end - start))

    # display MiHo clusters, outliers are black diamonds    
    if not ncc_check:
        mihoo.show_clustering()