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

# from src.DIM_modules.keynetaffnethardnet_kornia_matcher_module import keynetaffnethardnet_kornia_matcher_module
# from src.DIM_modules.superpoint_lightglue_module import superpoint_lightglue_module
# from src.DIM_modules.superpoint_kornia_matcher_module import superpoint_kornia_matcher_module
# from src.DIM_modules.disk_lightglue_module import disk_lightglue_module
# from src.DIM_modules.aliked_lightglue_module import aliked_lightglue_module
# from src.DIM_modules.loftr_module import loftr_module


if __name__ == '__main__':

    pipes = [
        # [
        #     superpoint_lightglue_module(nmax_keypoints=4000),
        #     # superpoint_kornia_matcher_module(nmax_keypoints=4000, th=0.97),
        #     # keynetaffnethardnet_kornia_matcher_module(nmax_keypoints=4000, upright=False, th=0.99),
        #     # disk_lightglue_module(nmax_keypoints=4000),
        #     # aliked_lightglue_module(nmax_keypoints=4000),
        #     # loftr_module(pretrained='outdoor'),
        #     # keynetaffnethardnet_module(upright=False, th=0.99),
        #     miho_duplex.miho_module(),
        #     pipe_base.pydegensac_module(px_th=3)
        # ],

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

    bench_path = '../bench_data'   
    save_to = os.path.join(bench_path, 'res', 'res')
    
###

    megadepth_data, scannet_data, _ = bench.megadepth_scannet_bench_setup(bench_path=bench_path)

    print("*** M e g a D e p t h ***")    
    for i, pipe in enumerate(pipes):
        print(f"--== Running pipeline {i+1}/{len(pipes)} ==--")        
        for pipe_module in pipe:
            if hasattr(pipe_module, 'mode'): setattr(pipe_module, 'mode', 'fundamental_matrix')
        bench.run_pipe(pipe, megadepth_data, 'megadepth', 'MegaDepth', bench_path=bench_path, ext='.png')
        bench.eval_pipe_fundamental(pipe, megadepth_data, 'megadepth', 'MegaDepth', bench_path=bench_path, save_to=save_to + '_fundamental_megadepth.pbz2', use_scale=True)
        bench.eval_pipe_essential(pipe, megadepth_data, 'megadepth', 'MegaDepth', bench_path=bench_path, essential_th_list=[0.5], save_to=save_to + '_essential_megadepth.pbz2', use_scale=True)
        bench.show_pipe(pipe, megadepth_data, 'megadepth', 'MegaDepth', bench_path=bench_path , ext='.png')

    print("*** S c a n N e t ***")    
    for i, pipe in enumerate(pipes):
        print(f"--== Running pipeline {i+1}/{len(pipes)} ==--")        
        for pipe_module in pipe:
            if hasattr(pipe_module, 'mode'): setattr(pipe_module, 'mode', 'fundamental_matrix')
        bench.run_pipe(pipe, scannet_data, 'scannet', 'ScanNet', bench_path=bench_path , ext='.png')
        bench.eval_pipe_fundamental(pipe, scannet_data, 'scannet', 'ScanNet', bench_path=bench_path, save_to=save_to + '_fundamental_scannet.pbz2', use_scale=False)
        bench.eval_pipe_essential(pipe, scannet_data, 'scannet', 'ScanNet', bench_path=bench_path, essential_th_list=[0.5], save_to=save_to + '_essential_scannet.pbz2', use_scale=False)
        bench.show_pipe(pipe, scannet_data, 'scannet', 'ScanNet', bench_path=bench_path , ext='.png')

###

    planar_data, _ = bench.planar_bench_setup(bench_path=bench_path, upright=True)

    print("*** P l a n a r ***")    
    for i, pipe in enumerate(pipes):
        print(f"--== Running pipeline {i+1}/{len(pipes)} ==--")        
        for pipe_module in pipe:
            if hasattr(pipe_module, 'mode'): setattr(pipe_module, 'mode', 'homography')
        bench.run_pipe(pipe, planar_data, 'planar', 'Planar', bench_path=bench_path, ext='.png')
        bench.eval_pipe_homography(pipe, planar_data, 'planar', 'Planar', bench_path=bench_path, save_to=save_to + '_homography_planar.pbz2', use_scale=False, save_acc_images=True)
        bench.show_pipe(pipe, planar_data, 'planar', 'Planar', bench_path=bench_path , ext='.png')

###

    imc_data, _ = bench.imc_phototourism_bench_setup(bench_path=bench_path)
    
    print("*** I M C   P h o t o t o u r i s m ***")
    for i, pipe in enumerate(pipes):
        print(f"--== Running pipeline {i+1}/{len(pipes)} ==--")        
        for pipe_module in pipe:
            if hasattr(pipe_module, 'mode'): setattr(pipe_module, 'mode', 'fundamental_matrix')
        bench.run_pipe(pipe, imc_data, 'imc_phototourism', 'IMC Phototourism', bench_path=bench_path, ext='.jpg')
        bench.eval_pipe_fundamental(pipe, imc_data, 'imc_phototourism', 'IMC Phototourism', bench_path=bench_path, save_to=save_to + '_fundamental_imc_phototourism.pbz2', use_scale=False, also_metric=True)
        bench.eval_pipe_essential(pipe, imc_data, 'imc_phototourism', 'IMC Phototourism', bench_path=bench_path, essential_th_list=[0.5], save_to=save_to + '_essential_imc_phototourism.pbz2', use_scale=False, also_metric=True)
        bench.show_pipe(pipe, imc_data, 'imc_phototourism', 'IMC Phototourism', bench_path=bench_path, ext='.jpg')
