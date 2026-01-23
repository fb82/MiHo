import os
import src.base_modules as pipe_base
import src.miho as miho_duplex
import src.miho_other as miho_unduplex
import src.ncc as ncc
import src.bench_utils as bench

if __name__ == '__main__':    
    # available RANSAC: pydegensac, magsac, poselib        

    pipe_head = lambda: None
    pipe_head.placeholder = 'head'

    pipe_ransac = lambda: None
    pipe_ransac.placeholder = 'ransac'

    pipes = [
        [
            pipe_head,
            miho_duplex.miho_module(),
            ncc.ncc_module(also_prev=True, use_covariance=True, search_gauss_mask=0.5, covariance_gauss_mask=0.5),
            pipe_ransac
        ],
                
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
            miho_unduplex.miho_module(half=True),
            pipe_ransac
        ],

        [
            pipe_head,
            miho_unduplex.miho_module(half=True),
            ncc.ncc_module(also_prev=True),            
            pipe_ransac
        ],

###

        [
            pipe_head,
            ncc.ncc_module(also_prev=True, use_covariance=True),
            pipe_ransac
        ],

        [
            pipe_head,
            miho_duplex.miho_module(),
            ncc.ncc_module(also_prev=True, use_covariance=True),
            pipe_ransac
        ],

        [
            pipe_head,
            miho_unduplex.miho_module(),
            ncc.ncc_module(also_prev=True, use_covariance=True),          
            pipe_ransac
        ],

        [
            pipe_head,
            miho_unduplex.miho_module(half=True),
            ncc.ncc_module(also_prev=True, use_covariance=True),           
            pipe_ransac
        ],
    
###

        [
            pipe_head,
            ncc.ncc_module(also_prev=True, use_covariance=True, search_gauss_mask=0.5),
            pipe_ransac
        ],

        [
            pipe_head,
            miho_duplex.miho_module(),
            ncc.ncc_module(also_prev=True, use_covariance=True, search_gauss_mask=0.5),
            pipe_ransac
        ],

        [
            pipe_head,
            miho_unduplex.miho_module(),
            ncc.ncc_module(also_prev=True, use_covariance=True, search_gauss_mask=0.5),          
            pipe_ransac
        ],

        [
            pipe_head,
            miho_unduplex.miho_module(half=True),
            ncc.ncc_module(also_prev=True, use_covariance=True, search_gauss_mask=0.5),           
            pipe_ransac
        ],

###

        [
            pipe_head,
            ncc.ncc_module(also_prev=True, use_covariance=True, covariance_gauss_mask=0.5),
            pipe_ransac
        ],

        [
            pipe_head,
            miho_duplex.miho_module(),
            ncc.ncc_module(also_prev=True, use_covariance=True, covariance_gauss_mask=0.5),
            pipe_ransac
        ],

        [
            pipe_head,
            miho_unduplex.miho_module(),
            ncc.ncc_module(also_prev=True, use_covariance=True, covariance_gauss_mask=0.5),          
            pipe_ransac
        ],

        [
            pipe_head,
            miho_unduplex.miho_module(half=True),
            ncc.ncc_module(also_prev=True, use_covariance=True, covariance_gauss_mask=0.5),           
            pipe_ransac
        ],        
    ]

    pipe_heads = [
        pipe_base.keynetaffnethardnet_module(num_features=8000, upright=True, th=0.99),
        pipe_base.sift_module(num_features=8000, upright=True, th=0.95, rootsift=True),     
        pipe_base.lightglue_module(num_features=8000, upright=True, what='superpoint'),
        pipe_base.loftr_module(num_features=8000, upright=True),        
        ]
    
    pipe_ransacs = [
        pipe_base.magsac_module(px_th=1.00),
        pipe_base.magsac_module(px_th=0.75),
        ]
    
    for pipe_module in pipe_heads: pipe_module.placeholder = 'head'
    for pipe_module in pipe_ransacs: pipe_module.placeholder = 'ransac'
    
    pipe_save_to = [pipe_head.get_id() for pipe_head in pipe_heads]

###

    bench_path = '../bench_data'   
    save_to = 'res'
    show_matches = False
    
    benchmark_data = {
            'megadepth': {'name': 'megadepth', 'Name': 'MegaDepth', 'setup': bench.megadepth_bench_setup, 'is_outdoor': True, 'is_not_planar': True, 'ext': '.png', 'use_scale': True, 'also_metric': False},
            'scannet': {'name': 'scannet', 'Name': 'ScanNet', 'setup': bench.scannet_bench_setup, 'is_outdoor': False, 'is_not_planar': True, 'ext': '.png', 'use_scale': False, 'also_metric': False},
            'planar': {'name': 'planar', 'Name': 'Planar', 'setup': bench.planar_bench_setup, 'is_outdoor': True, 'is_not_planar': False, 'ext': '.png', 'use_scale': False, 'also_metric': False},
#           'imc_phototourism': {'name': 'imc_phototourism', 'Name': 'IMC PhotoTourism', 'setup': bench.imc_phototourism_bench_setup, 'is_outdoor': True, 'is_not_planar': True, 'ext': '.jpg', 'use_scale': False, 'also_metric': True},
        }
    
    for b in benchmark_data.keys():
        print("*** " + benchmark_data[b]['Name'] + " ***")
        
        b_data, _ = benchmark_data[b]['setup'](bench_path=bench_path, upright=True)
        
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
                
                if 'keypt2subpx' in pipe_head.get_id():
                    pl = 1
                else:
                    pl = len(pipes)
                                
                for i, pipe in enumerate(pipes[:pl]):                                        
                    print(f"*** Pipeline {i+1}/{len(pipes)} ***")        

                    for k, pipe_module in enumerate(pipe):
                        if hasattr(pipe_module, 'placeholder'):
                            if pipe_module.placeholder == 'head': pipe[k] = pipe_head
                            if pipe_module.placeholder == 'ransac': pipe[k] = pipe_ransac

                    for pipe_module in pipe:
                        if hasattr(pipe_module, 'mode'): setattr(pipe_module, 'mode', bench_mode)
                        if hasattr(pipe_module, 'outdoor'): setattr(pipe_module, 'outdoor', benchmark_data[b]['is_outdoor'])

                    bench.run_pipe(pipe, b_data, benchmark_data[b]['name'], benchmark_data[b]['Name'], bench_path=bench_path, ext=benchmark_data[b]['ext'])
                    bench.count_pipe_match(pipe, b_data, benchmark_data[b]['name'], bench_path=bench_path, save_to=to_save_file + 'match_count' + to_save_file_suffix + '.pbz2')

                    if benchmark_data[b]['is_not_planar']:
                        bench.eval_pipe_fundamental(pipe, b_data, benchmark_data[b]['name'], benchmark_data[b]['Name'], bench_path=bench_path, save_to=to_save_file + 'fundamental' + to_save_file_suffix + '.pbz2', use_scale=benchmark_data[b]['use_scale'], also_metric=benchmark_data[b]['also_metric'])
                        bench.eval_pipe_essential(pipe, b_data, benchmark_data[b]['name'], benchmark_data[b]['Name'], bench_path=bench_path, essential_th_list=[0.5], save_to=to_save_file + 'essential' + to_save_file_suffix + '.pbz2', use_scale=benchmark_data[b]['use_scale'], also_metric=benchmark_data[b]['also_metric'])
                    else:
                        bench.eval_pipe_homography(pipe, b_data, benchmark_data[b]['name'], benchmark_data[b]['Name'], bench_path=bench_path, save_to=to_save_file + 'homography' + to_save_file_suffix + '.pbz2', use_scale=benchmark_data[b]['use_scale'], save_acc_images=show_matches)

                    if show_matches:
                        bench.show_pipe(pipe, b_data, benchmark_data[b]['name'], benchmark_data[b]['Name'], bench_path=bench_path, ext=benchmark_data[b]['ext'], save_ext='.jpg')
                        
            if benchmark_data[b]['is_not_planar']:
                bench.csv_summary_non_planar(essential_th_list=[0.5], essential_load_from=to_save_file + 'essential' + to_save_file_suffix + '.pbz2', fundamental_load_from=to_save_file + 'fundamental' + to_save_file_suffix + '.pbz2', match_count_load_from=to_save_file + 'match_count' + to_save_file_suffix + '.pbz2', save_to=to_save_file + 'fundamental_and_essential' + to_save_file_suffix + '.csv', also_metric=benchmark_data[b]['also_metric'], to_remove_prefix=pipe_head.get_id())
            else:
                bench.csv_summary_planar(load_from=to_save_file + 'homography' + to_save_file_suffix + '.pbz2', save_to=to_save_file + 'homography' + to_save_file_suffix + '.csv', match_count_load_from=to_save_file + 'match_count' + to_save_file_suffix + '.pbz2', to_remove_prefix=pipe_head.get_id())
