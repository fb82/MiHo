import os
import src.base_modules as pipe_base
import src.miho as miho_duplex
import src.miho_other as miho_unduplex
import src.ncc as ncc
import src.GMS.gms_custom as gms
import src.OANet.learnedmatcher_custom as oanet
import src.ACNe.acne_custom as acne
import src.AdaLAM.adalam_custom as adalam
import src.bench_utils as bench

# from pprint import pprint
# import deep_image_matching as dim
# import yaml

#from src.pipelines.keynetaffnethardnet_module_fabio import keynetaffnethardnet_module_fabio
#from src.DIM_modules.keynetaffnethardnet_kornia_matcher_module import keynetaffnethardnet_kornia_matcher_module
#from src.DIM_modules.superpoint_lightglue_module import superpoint_lightglue_module
#from src.DIM_modules.superpoint_kornia_matcher_module import superpoint_kornia_matcher_module
#from src.DIM_modules.disk_lightglue_module import disk_lightglue_module
#from src.DIM_modules.aliked_lightglue_module import aliked_lightglue_module
#from src.DIM_modules.sift_kornia_matcher_module import sift_kornia_matcher_module
#from src.DIM_modules.magsac_module import magsac_module
#from src.DIM_modules.loftr_module import loftr_module


if __name__ == '__main__':
    bench_path = '../bench_data'   
    bench_gt = 'gt_data'
    bench_im = 'imgs'
    bench_res = 'res'
    bench_plot = 'plot'
    save_to = os.path.join(bench_path, bench_res, 'res_')

    pipes = [
        #[
        #    superpoint_lightglue_module(nmax_keypoints=2048),
        #    #magsac_module(px_th=3, conf=0.95, max_iters=100000)
        #    pipe_base.pydegensac_module(px_th=1)
        #],
#
        #[
        #    superpoint_lightglue_module(nmax_keypoints=2048),
        #    #magsac_module(px_th=3, conf=0.95, max_iters=100000)
        #    pipe_base.pydegensac_module(px_th=3)
        #],
#
        #[
        #    superpoint_lightglue_module(nmax_keypoints=2048),
        #    #magsac_module(px_th=3, conf=0.95, max_iters=100000)
        #    pipe_base.pydegensac_module(px_th=5)
        #],
#
        #[
        #    superpoint_lightglue_module(nmax_keypoints=2048),
        #    #magsac_module(px_th=3, conf=0.95, max_iters=100000)
        #    pipe_base.magsac_module(px_th=1)
        #],
#
        #[
        #    superpoint_lightglue_module(nmax_keypoints=2048),
        #    #magsac_module(px_th=3, conf=0.95, max_iters=100000)
        #    pipe_base.magsac_module(px_th=3)
        #],
#
        #[
        #    superpoint_lightglue_module(nmax_keypoints=2048),
        #    #magsac_module(px_th=3, conf=0.95, max_iters=100000)
        #    pipe_base.magsac_module(px_th=5)
        #],

        #[
        #    superpoint_lightglue_module(nmax_keypoints=2048),
        #    pipe_base.pydegensac_module(px_th=3)
        #],

        #[
        #    superpoint_kornia_matcher_module(nmax_keypoints=2048, th=0.99),
        #    miho.miho_module(),
        #    pipe_base.pydegensac_module(px_th=3)
        #],

        #[
        #    superpoint_kornia_matcher_module(nmax_keypoints=2048, th=0.99),
        #    pipe_base.pydegensac_module(px_th=3)
        #],

        #[
        #    sift_kornia_matcher_module(n_features=2048, contrastThreshold=0.04, th=0.99),
        #    miho.miho_module(),
        #    pipe_base.pydegensac_module(px_th=3)
        #],

        #[
        #    sift_kornia_matcher_module(n_features=2048, contrastThreshold=0.04, th=0.99),
        #    pipe_base.pydegensac_module(px_th=3)
        #],

        #[
        #   #superpoint_lightglue_module(nmax_keypoints=4000),
        #   #sift_kornia_matcher_module(n_features=8000, contrastThreshold=0.04, th=0.99),
        #   superpoint_kornia_matcher_module(nmax_keypoints=8000, th=0.99),
        #   # keynetaffnethardnet_kornia_matcher_module(nmax_keypoints=4000, upright=False, th=0.99),
        #   # disk_lightglue_module(nmax_keypoints=4000),
        #   # aliked_lightglue_module(nmax_keypoints=4000),
        #   # loftr_module(pretrained='outdoor'),
        #   # keynetaffnethardnet_module(upright=False, th=0.99),
        #   #miho.miho_module(),
        #   pipe_base.pydegensac_module(px_th=3)
        #],

        [
            pipe_base.keynetaffnethardnet_module(upright=False, th=0.99),
            miho.miho_module(),
            pipe_base.pydegensac_module(px_th=3)
        ],

        # available RANSAC: pydegensac, magsac, poselib        
        [
            pipe_base.keynetaffnethardnet_module(upright=False, th=0.99),
            pipe_base.pydegensac_module(px_th=3)
        ],

        [
            pipe_base.keynetaffnethardnet_module(upright=False, th=0.99),
            ncc.ncc_module(),
            pipe_base.pydegensac_module(px_th=3)
        ],

        [
            pipe_base.keynetaffnethardnet_module(upright=False, th=0.99),
            miho_duplex.miho_module(),
            pipe_base.pydegensac_module(px_th=3)
        ],

        [
            pipe_base.keynetaffnethardnet_module(upright=False, th=0.99),
            miho_duplex.miho_module(),
            ncc.ncc_module(),
            pipe_base.pydegensac_module(px_th=3)
        ],

        [
            pipe_base.keynetaffnethardnet_module(upright=False, th=0.99),
            miho_unduplex.miho_module(),
            pipe_base.pydegensac_module(px_th=3)
        ],

        [
            pipe_base.keynetaffnethardnet_module(upright=False, th=0.99),
            miho_unduplex.miho_module(),
            ncc.ncc_module(),
            pipe_base.pydegensac_module(px_th=3)
        ],

        [
            pipe_base.keynetaffnethardnet_module(upright=False, th=0.99),
            gms.gms_module(),
            pipe_base.pydegensac_module(px_th=3)
        ],

        [
            pipe_base.keynetaffnethardnet_module(upright=False, th=0.99),
            gms.gms_module(),
            ncc.ncc_module(),
            pipe_base.pydegensac_module(px_th=3)
        ],

        [
            pipe_base.keynetaffnethardnet_module(upright=False, th=0.99),
            oanet.oanet_module(),
            pipe_base.pydegensac_module(px_th=3)
        ],

        [
            pipe_base.keynetaffnethardnet_module(upright=False, th=0.99),
            oanet.oanet_module(),
            ncc.ncc_module(),
            pipe_base.pydegensac_module(px_th=3)
        ],

        [
            pipe_base.keynetaffnethardnet_module(upright=False, th=0.99),
            acne.acne_module(),
            pipe_base.pydegensac_module(px_th=3)
        ],        
        
        [
            pipe_base.keynetaffnethardnet_module(upright=False, th=0.99),
            acne.acne_module(),
            ncc.ncc_module(),
            pipe_base.pydegensac_module(px_th=3)
        ],
        
        [
            pipe_base.keynetaffnethardnet_module(upright=False, th=0.99),
            adalam.adalam_module(),
            pipe_base.pydegensac_module(px_th=3)
        ],        
       
        [
            pipe_base.keynetaffnethardnet_module(upright=False, th=0.99),
            adalam.adalam_module(),
            ncc.ncc_module(),
            pipe_base.pydegensac_module(px_th=3)
        ]            
    ]

###

    bench_file = 'megadepth_scannet'
    megadepth_data, scannet_data, data_file = bench.bench_init(bench_file=bench_file, bench_path=bench_path, bench_gt=bench_gt)
    megadepth_data, scannet_data = bench.setup_images(megadepth_data, scannet_data, data_file=data_file, bench_path=bench_path, bench_imgs=bench_im)

    print("*** M e g a D e p t h ***")    
    for i, pipe in enumerate(pipes):
        print(f"--== Running pipeline {i+1}/{len(pipes)} ==--")        
        for pipe_module in pipe:
            if hasattr(pipe_module, 'mode'): setattr(pipe_module, 'mode', 'fundamental_matrix')
        bench.run_pipe(pipe, megadepth_data, 'megadepth', 'MegaDepth', bench_path=bench_path , bench_im=bench_im, bench_res=bench_res)
        bench.eval_pipe_fundamental(pipe, megadepth_data, 'megadepth', 'MegaDepth', bench_path=bench_path, bench_res='res', save_to=save_to + '_fundamental_megadepth.pbz2', use_scale=True, err_th_list=list(range(1, 16)))
        bench.eval_pipe_essential(pipe, megadepth_data, 'megadepth', 'MegaDepth', bench_path=bench_path, bench_res='res', essential_th_list=[0.5], save_to=save_to + '_essential_megadepth.pbz2', use_scale=True)
        bench.show_pipe(pipe, megadepth_data, 'megadepth', 'MegaDepth', bench_path=bench_path , bench_im=bench_im, bench_res=bench_res, bench_plot=bench_plot)

    print("*** S c a n N e t ***")    
    for i, pipe in enumerate(pipes):
        print(f"--== Running pipeline {i+1}/{len(pipes)} ==--")        
        for pipe_module in pipe:
            if hasattr(pipe_module, 'mode'): setattr(pipe_module, 'mode', 'fundamental_matrix')
        bench.run_pipe(pipe, scannet_data, 'scannet', 'ScanNet', bench_path=bench_path , bench_im=bench_im, bench_res=bench_res)
        bench.eval_pipe_fundamental(pipe, scannet_data, 'scannet', 'ScanNet', bench_path=bench_path, bench_res='res', save_to=save_to + '_fundamental_scannet.pbz2', use_scale=False, err_th_list=list(range(1, 16)))
        bench.eval_pipe_essential(pipe, scannet_data, 'scannet', 'ScanNet', bench_path=bench_path, bench_res='res', essential_th_list=[0.5], save_to=save_to + '_essential_scannet.pbz2', use_scale=False)
        bench.show_pipe(pipe, scannet_data, 'scannet', 'ScanNet', bench_path=bench_path , bench_im=bench_im, bench_res=bench_res, bench_plot=bench_plot)

###

    planar_data, _ = bench.planar_bench_setup(bench_path=bench_path, bench_imgs=bench_im)

    print("*** P l a n a r ***")    
    for i, pipe in enumerate(pipes):
        print(f"--== Running pipeline {i+1}/{len(pipes)} ==--")        
        for pipe_module in pipe:
            if hasattr(pipe_module, 'mode'): setattr(pipe_module, 'mode', 'homography')
        bench.run_pipe(pipe, planar_data, 'planar', 'Planar', bench_path=bench_path , bench_im=bench_im, bench_res=bench_res)
        bench.eval_pipe_homography(pipe, planar_data, 'planar', 'Planar', bench_path=bench_path, bench_res='res', save_to=save_to + '_homography_planar.pbz2', use_scale=False, err_th_list=list(range(1, 16)))
        bench.show_pipe(pipe, planar_data, 'planar', 'Planar', bench_path=bench_path , bench_im=bench_im, bench_res=bench_res, bench_plot=bench_plot)