import os
import numpy as np
import src.base_modules as pipe_base
import src.miho as miho_duplex
import src.miho_other as miho_unduplex
import src.ncc as ncc
import src.GMS.gms_custom as gms
import src.OANet.learnedmatcher_custom as oanet
import src.ACNe.acne_custom as acne
import src.AdaLAM.adalam_custom as adalam
import src.DeDoDe2.dedode2_custom as dedode2
import src.DeMatch.dematch_custom as dematch
import src.CLNet.clnet_custom as clnet
import src.FCGNN.fcgnn_custom as fcgnn
import src.MS2DGNet.ms2dgnet_custom as ms2dgnet
import src.NCMNet.ncmnet_custom as ncmnet
import src.bench_utils as bench
import src.ConvMatch.convmatch_custom as convmatch
import src.ConsensusClustering.consensusclustering_custom as consensusclustering

# from src.DIM_modules.superpoint_lightglue_module import superpoint_lightglue_module
# from src.DIM_modules.disk_lightglue_module import disk_lightglue_module
# from src.DIM_modules.aliked_lightglue_module import aliked_lightglue_module
# from src.DIM_modules.loftr_module import loftr_module

# plot matches found by the pipeline with and without RANSAC
# change index in benchmark_data to run on different image pairs
# color matches are green, orange, red, purple, blue respectively for
# error ranges [0, 1], [1, 3], [3, 7], [7, 15], [15, np.Inf] according to GT
# gray is used for data with no GT in planar scenes

if __name__ == '__main__':    
    # available RANSAC: pydegensac, magsac, poselib        

    pipe_head = lambda: None
    pipe_head.placeholder = 'head'

    pipe_ransac = lambda: None
    pipe_ransac.placeholder = 'ransac'

    pipes = [
        [
            pipe_head,
            pipe_ransac
        ],

        [
            pipe_head,
            ncc.ncc_module(also_prev=True),
            pipe_ransac
        ],

        [
            pipe_head,
            miho_duplex.miho_module(),
            pipe_ransac
        ],

        [
            pipe_head,
            miho_duplex.miho_module(),
            ncc.ncc_module(also_prev=True),
            pipe_ransac
        ],

        [
            pipe_head,
            miho_unduplex.miho_module(),
            pipe_ransac
        ],

        [
            pipe_head,
            miho_unduplex.miho_module(),
            ncc.ncc_module(also_prev=True),            
            pipe_ransac
        ],
    
        [
            pipe_head,
            gms.gms_module(),
            pipe_ransac
        ],

        [
            pipe_head,
            oanet.oanet_module(),
            pipe_ransac
        ],
        
        [
            pipe_head,
            adalam.adalam_module(),
            pipe_ransac
        ],                  
        
        [
            pipe_head,
            acne.acne_module(),
            pipe_ransac
        ],        

        [
            pipe_head,
            consensusclustering.consensusclustering_module(),
            pipe_ransac
        ],
        
        [
            pipe_head,
            dematch.dematch_module(),
            pipe_ransac
        ], 

        [
            pipe_head,
            convmatch.convmatch_module(),
            pipe_ransac
        ], 

        [
            pipe_head,
            fcgnn.fcgnn_module(),
            pipe_ransac
        ],  

        [
            pipe_head,
            clnet.clnet_module(),
            pipe_ransac
        ],
        
        [
            pipe_head,
            ms2dgnet.ms2dgnet_module(),
            pipe_ransac
        ],
        
        [
            pipe_head,
            ncmnet.ncmnet_module(),
            pipe_ransac
        ],            
    ]

    pipe_heads = [
        pipe_base.keynetaffnethardnet_module(num_features=8000, upright=True, th=0.99),
        pipe_base.sift_module(num_features=8000, upright=True, th=0.95, rootsift=True),     
        pipe_base.lightglue_module(num_features=8000, upright=True, what='superpoint'),
        pipe_base.lightglue_module(num_features=8000, upright=True, what='aliked'),
        pipe_base.lightglue_module(num_features=8000, upright=True, what='disk'),  
        pipe_base.loftr_module(num_features=8000, upright=True),        
        dedode2.dedode2_module(num_features=8000, upright=True),                
        # superpoint_lightglue_module(nmax_keypoints=8000),
        # aliked_lightglue_module(nmax_keypoints=8000),
        # disk_lightglue_module(nmax_keypoints=8000),
        # loftr_module(nmax_keypoints=8000),  
        ]
    
    pipe_ransacs = [
        pipe_base.magsac_module(px_th=1.00),
        # pipe_base.magsac_module(px_th=0.75),
        ]
    
    for pipe_module in pipe_heads: pipe_module.placeholder = 'head'
    for pipe_module in pipe_ransacs: pipe_module.placeholder = 'ransac'
    
    pipe_save_to = [pipe_head.get_id() for pipe_head in pipe_heads]

###

    bench_path = '../bench_data'   
    save_to = 'res'
    save_mode = 'as_bench' # 'as_bench' / 'flat' - showcase folder structure
    
    benchmark_data = {
            'megadepth': {'name': 'megadepth', 'Name': 'MegaDepth', 'setup': bench.megadepth_bench_setup, 'is_outdoor': True, 'is_not_planar': True, 'ext': '.png', 'use_scale': True, 'also_metric': False, 'index': [1352, 584]},
            'scannet': {'name': 'scannet', 'Name': 'ScanNet', 'setup': bench.scannet_bench_setup, 'is_outdoor': False, 'is_not_planar': True, 'ext': '.png', 'use_scale': False, 'also_metric': False, 'index': [0, 69]},
            'planar': {'name': 'planar', 'Name': 'Planar', 'setup': bench.planar_bench_setup, 'is_outdoor': True, 'is_not_planar': False, 'ext': '.png', 'use_scale': False, 'also_metric': False, 'index': [86, 130]},
          # 'imc_phototourism': {'name': 'imc_phototourism', 'Name': 'IMC PhotoTourism', 'setup': bench.imc_phototourism_bench_setup, 'is_outdoor': True, 'is_not_planar': True, 'ext': '.jpg', 'use_scale': False, 'also_metric': True, 'index': [0]},
        }
        
    for b in benchmark_data.keys():
        print("*** " + benchmark_data[b]['Name'] + " ***")
        
        b_data, _ = benchmark_data[b]['setup'](bench_path=bench_path, upright=True)
        
        # select image pair subset
        b_index = benchmark_data[b]['index']
        if not (b_index is None):
            for bname in b_data.keys():
                if isinstance(b_data[bname], list): b_data[bname] = [b_data[bname][bb] for bb in b_index]
                if isinstance(b_data[bname], np.ndarray): b_data[bname] = b_data[bname][b_index]
        
        if benchmark_data[b]['is_not_planar']:
            bench_mode = 'fundamental_matrix'
        else:
            bench_mode = 'homography'
            
        for ip in range(len(pipe_heads)):
            pipe_head = pipe_heads[ip]
            
            print("*** " + pipe_head.get_id() + " ***")
            
            to_save_file =  os.path.join(bench_path, save_to, save_to + '_' + pipe_save_to[ip] + '_')
            to_save_file_suffix ='_' + benchmark_data[b]['name']
            
            for jp in range(len(pipe_ransacs)):
                pipe_ransac = pipe_ransacs[jp]

                print("*** " + pipe_ransac.get_id() + " ***")
                
                for i, pipe in enumerate(pipes):                                        
                    print(f"*** Pipeline {i+1}/{len(pipes)} ***")        

                    for k, pipe_module in enumerate(pipe):
                        if hasattr(pipe_module, 'placeholder'):
                            if pipe_module.placeholder == 'head': pipe[k] = pipe_head
                            if pipe_module.placeholder == 'ransac': pipe[k] = pipe_ransac

                    for pipe_module in pipe:
                        if hasattr(pipe_module, 'mode'): setattr(pipe_module, 'mode', bench_mode)
                        if hasattr(pipe_module, 'outdoor'): setattr(pipe_module, 'outdoor', benchmark_data[b]['is_outdoor'])

                    bench.run_pipe(pipe, b_data, benchmark_data[b]['name'], benchmark_data[b]['Name'], bench_path=bench_path, ext=benchmark_data[b]['ext'])
                    bench.show_pipe_other(pipe, b_data, benchmark_data[b]['name'], benchmark_data[b]['Name'], bench_path=bench_path, ext=benchmark_data[b]['ext'], save_ext='.pdf', save_mode=save_mode, b_index=benchmark_data[b]['index'], bench_mode=bench_mode, use_scale=benchmark_data[b]['use_scale'])
                        
