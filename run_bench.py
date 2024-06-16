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

from src.DIM_modules.superpoint_lightglue_module import superpoint_lightglue_module
from src.DIM_modules.disk_lightglue_module import disk_lightglue_module
from src.DIM_modules.aliked_lightglue_module import aliked_lightglue_module
from src.DIM_modules.loftr_module import loftr_module


if __name__ == '__main__':

    # available RANSAC: pydegensac, magsac, poselib        
    pipe_init = None
    pipe_end = None

    pipes = [
        [
            pipe_init,
            pipe_end
        ],

        [
            pipe_init,
            ncc.ncc_module(also_prev=True),
            pipe_end
        ],

        [
            pipe_init,
            miho_duplex.miho_module(),
            pipe_end
        ],

        [
            pipe_init,
            miho_duplex.miho_module(),
            ncc.ncc_module(also_prev=True),
            pipe_end
        ],

        [
            pipe_init,
            miho_unduplex.miho_module(),
            pipe_end
        ],


        [
            pipe_init,
            miho_unduplex.miho_module(),
            ncc.ncc_module(also_prev=True),            
            pipe_end
        ],
    
        [
            pipe_init,
            gms.gms_module(),
            pipe_end
        ],

        [
            pipe_init,
            oanet.oanet_module(),
            pipe_end
        ],
        
        [
            pipe_init,
            adalam.adalam_module(),
            pipe_end
        ],                  
        
        [
            pipe_init,
            acne.acne_module(),
            pipe_end
        ],        
    ]

    pipe_inits = [
        pipe_base.keynetaffnethardnet_module(num_features=8000, upright=True, th=0.99),
        pipe_base.sift_module(rootsift=True, num_features=8000, upright=True, th=0.95),     
        superpoint_lightglue_module(nmax_keypoints=8000),
        aliked_lightglue_module(nmax_keypoints=8000),
        disk_lightglue_module(nmax_keypoints=8000),
        loftr_module(nmax_keypoints=8000),
        ]
    
    pipe_ends = [
        pipe_base.magsac_module(th=1.00),
        pipe_base.magsac_module(th=0.75),
        ]

###

    bench_path = '../bench_data'   
    save_to = 'res'
    show_matches = False
    
    benchmark_data = {
            'megadepth': {'name': 'megadepth', 'Name': 'MegaDepth', 'setup': bench.megadepth_bench_setup, 'is_outdoor': True, 'is_not_planar': True, 'ext': '.png', 'use_scale': True, 'also_metric': False},
            'scannet': {'name': 'scannet', 'Name': 'ScanNet', 'setup': bench.scannet_bench_setup, 'is_outdoor': False, 'is_not_planar': True, 'ext': '.png', 'use_scale': False, 'also_metric': False},
            'planar': {'name': 'planar', 'Name': 'Planar', 'setup': bench.planar_bench_setup, 'is_outdoor': True, 'is_not_planar': False, 'ext': '.png', 'use_scale': False, 'also_metric': False},
            'imc_phototourism': {'name': 'imc_phototourism', 'Name': 'IMC PhotoTourism', 'setup': bench.imc_phototourism_bench_setup, 'is_outdoor': True, 'is_not_planar': False, 'ext': '.jpg', 'use_scale': False, 'also_metric': True},
        }
    
    for b in benchmark_data.keys():
        print("*** " + benchmark_data[b]['Name'] + " ***")
        
        b_data, _ = benchmark_data[b]['setup'](bench_path=bench_path, upright=True)
        
        if benchmark_data[b]['is_not_planar']:
            bench_mode = 'fundamental_matrix'
        else:
            bench_mode = 'homography'
            
        for ip in range(len(pipe_inits)):
            pipe_init = pipe_inits[ip]
            to_save_file =  os.path.join(bench_path, save_to, save_to + '_' + pipe_init.get_id() + '_')
            to_save_file_suffix ='_' + benchmark_data[b]['name'] + '.pbz2'
            
            for jp in range(len(pipe_ends)):
                pipe_end = pipe_ends[jp]
                
                for i, pipe in enumerate(pipes):
                    print(f"--== Running pipeline {i+1}/{len(pipes)} ==--")        
                    for pipe_module in pipe:
                        if hasattr(pipe_module, 'mode'): setattr(pipe_module, 'mode', bench_mode)
                        if hasattr(pipe_module, 'outdoor'): setattr(pipe_module, 'outdoor', benchmark_data[b]['is_outdoor'])
                    bench.run_pipe(pipe, b_data, benchmark_data[b]['name'], benchmark_data[b]['Name'], bench_path=bench_path, ext=benchmark_data[b]['ext'])

                    if benchmark_data[b]['is_not_planar']:
                        bench.eval_pipe_fundamental(pipe, b_data, benchmark_data[b]['name'], benchmark_data[b]['Name'], bench_path=bench_path, save_to=to_save_file + 'fundamental' + to_save_file_suffix, use_scale=benchmark_data[b]['use_scale'], also_metric=benchmark_data[b]['also_metric'])
                        bench.eval_pipe_essential(pipe, b_data, benchmark_data[b]['name'], benchmark_data[b]['Name'], bench_path=bench_path, essential_th_list=[0.5], save_to=to_save_file + 'essential' + to_save_file_suffix, use_scale=benchmark_data[b]['use_scale'], also_metric=benchmark_data[b]['also_metric'])
                    else:
                        bench.eval_pipe_homography(pipe, b_data, benchmark_data[b]['name'], benchmark_data[b]['Name'], bench_path=bench_path, save_to=to_save_file + 'homography' + to_save_file_suffix, use_scale=benchmark_data[b]['use_scale'], save_acc_images=show_matches)

                    if show_matches:
                        bench.show_pipe(pipe, b_data, benchmark_data[b]['name'], benchmark_data[b]['Name'], bench_path=bench_path, ext=benchmark_data[b]['ext'])