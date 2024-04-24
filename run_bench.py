import os
import src.base_modules as pipe_base
import src.miho as miho
import src.ncc as ncc
import src.GMS.gms_custom as gms
import src.OANet.learnedmatcher_custom as oanet
import src.ACNe.acne_custom as acne
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
    # megadepth & scannet
    bench_path = '../miho_megadepth_scannet_bench_data'   
    bench_gt = 'gt_data'
    bench_im = 'imgs'
    bench_file = 'megadepth_scannet'
    bench_res = 'res'
    save_to = os.path.join(bench_path, bench_res, 'res_')

    # current deep pipelines are for outdoor !!!
    acne_unique_module = acne.acne_module()
    pipes = [
        # [
        #    superpoint_lightglue_module(nmax_keypoints=4000),
        #    # superpoint_kornia_matcher_module(nmax_keypoints=4000, th=0.97),
        #    # keynetaffnethardnet_kornia_matcher_module(nmax_keypoints=4000, upright=False, th=0.99),
        #    # disk_lightglue_module(nmax_keypoints=4000),
        #    # aliked_lightglue_module(nmax_keypoints=4000),
        #    # loftr_module(pretrained='outdoor'),
        #    # keynetaffnethardnet_module(upright=False, th=0.99),
        #    miho.miho_module(),
        #    pipe_base.pydegensac_module(px_th=3)
        # ],

        [
            pipe_base.keynetaffnethardnet_module(upright=False, th=0.99),
            miho.miho_module(),
            pipe_base.pydegensac_module(px_th=3)
        ],

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
            miho.miho_module(),
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
            acne_unique_module,
            pipe_base.pydegensac_module(px_th=3)
        ],        
        
        [
            pipe_base.keynetaffnethardnet_module(upright=False, th=0.99),
            acne_unique_module,
            ncc.ncc_module(),
            pipe_base.pydegensac_module(px_th=3)
        ]        
    ]
               
    megadepth_data, scannet_data, data_file = bench.bench_init(bench_file=bench_file, bench_path=bench_path, bench_gt=bench_gt)
    megadepth_data, scannet_data = bench.setup_images(megadepth_data, scannet_data, data_file=data_file, bench_path=bench_path, bench_imgs=bench_im)

    for i, pipe in enumerate(pipes):
        print(f"--== Running pipeline {i+1}/{len(pipes)} ==--")
        bench.run_pipe(pipe, megadepth_data, 'megadepth', 'MegaDepth', bench_path=bench_path , bench_im=bench_im, bench_res=bench_res)
        bench.eval_pipe(pipe, megadepth_data, 'megadepth', 'MegaDepth', bench_path=bench_path, bench_res='res', essential_th_list=[0.5, 1, 1.5], save_to=save_to + 'megadepth.pbz2', use_scale=True)

    for i, pipe in enumerate(pipes):
        print(f"--== Running pipeline {i+1}/{len(pipes)} ==--")
        bench.run_pipe(pipe, scannet_data, 'scannet', 'ScanNet', bench_path=bench_path , bench_im=bench_im, bench_res=bench_res)
        bench.eval_pipe(pipe, scannet_data, 'scannet', 'ScanNet', bench_path=bench_path, bench_res='res', essential_th_list=[0.5, 1, 1.5], save_to=save_to + 'scannet.pbz2', use_scale=False)
