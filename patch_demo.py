from PIL import Image
import torch
import kornia as K
import src.ncc as ncc
import src.base_modules as pipe_base
import src.miho as miho_duplex
import src.bench_utils as bench
import os
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__': 
    pipe_keynet = pipe_base.keynetaffnethardnet_module(num_features=8000, upright=True, th=0.99)
    pipe_sift = pipe_base.sift_module(num_features=8000, upright=True, th=0.95, rootsift=True)    
    pipe_miho =  miho_duplex.miho_module()
    pipe_ncc = ncc.ncc_module(also_prev=True)

    w = 10 # patch size or display

    # bench_path = '/media/bellavista/Dati1/patch_test' # '../bench_data'   
    # bench_im='imgs'  
    # save_to = 'res'    

    # benchmark_data = {
    #         'megadepth': {'name': 'megadepth', 'Name': 'MegaDepth', 'setup': bench.megadepth_bench_setup, 'is_outdoor': True, 'is_not_planar': True, 'ext': '.png', 'use_scale': True, 'also_metric': False},
    #         'scannet': {'name': 'scannet', 'Name': 'ScanNet', 'setup': bench.scannet_bench_setup, 'is_outdoor': False, 'is_not_planar': True, 'ext': '.png', 'use_scale': False, 'also_metric': False},
    #     }
    
    # b = 'megadepth' # dataset
    # i = 0  # index pair
    # b_data, _ = benchmark_data[b]['setup'](bench_path=bench_path, upright=True)
    

    # n = len(b_data['im1'])
    # ext = benchmark_data[b]['ext']
    # im_path = os.path.join(bench_im, benchmark_data[b]['name'])        
    # im1 = os.path.join(bench_path, im_path, os.path.splitext(b_data['im1'][i])[0]) + ext
    # im2 = os.path.join(bench_path, im_path, os.path.splitext(b_data['im2'][i])[0]) + ext

    im1 = 'data/demo/im1.png'
    im2 = 'data/demo/im2.png'

    pipe_data_im = {'im1': im1, 'im2': im2}
    pipe_data_keynet = pipe_keynet.run(**pipe_data_im)
    pipe_data_sift = pipe_sift.run(**pipe_data_im)
                        
    for k, v in pipe_data_im.items():
        pipe_data_keynet[k] = v
        pipe_data_sift[k] = v

    pipe_data_keynet_miho = pipe_miho.run(**pipe_data_keynet)
    pipe_data_sift_miho = pipe_miho.run(**pipe_data_sift)

    for k, v in pipe_data_im.items():
        pipe_data_keynet_miho[k] = v
        pipe_data_sift_miho[k] = v
    
    mask = pipe_data_keynet_miho['mask']
    pipe_data_keynet['Hs'] = pipe_data_keynet['Hs'][mask]
    pipe_data_keynet['kp1'] = pipe_data_keynet['kp1'][mask]
    pipe_data_keynet['kp2'] = pipe_data_keynet['kp2'][mask]
    pipe_data_keynet['pt1'] = pipe_data_keynet['pt1'][mask]
    pipe_data_keynet['pt2'] = pipe_data_keynet['pt2'][mask]
    pipe_data_keynet['val'] = pipe_data_keynet['val'][mask]

    mask = pipe_data_sift_miho['mask']
    pipe_data_sift['Hs'] = pipe_data_sift['Hs'][mask]
    pipe_data_sift['kp1'] = pipe_data_sift['kp1'][mask]
    pipe_data_sift['kp2'] = pipe_data_sift['kp2'][mask]
    pipe_data_sift['pt1'] = pipe_data_sift['pt1'][mask]
    pipe_data_sift['pt2'] = pipe_data_sift['pt2'][mask]
    pipe_data_sift['val'] = pipe_data_sift['val'][mask]

    pipe_data_keynet_miho_ncc = pipe_ncc.run(**pipe_data_keynet_miho)
    pipe_data_sift_miho_ncc = pipe_ncc.run(**pipe_data_sift_miho)

    pipe_data_keynet_ncc = pipe_ncc.run(**pipe_data_keynet)
    pipe_data_sift_ncc = pipe_ncc.run(**pipe_data_sift)

    tg = transforms.Compose([
            transforms.Grayscale(),
            transforms.PILToTensor() 
            ]) 

    t = transforms.Compose([
            transforms.PILToTensor() 
            ]) 

    im1 = Image.open(pipe_data_keynet['im1'])
    im2 = Image.open(pipe_data_keynet['im2'])

    im1c = t(im1).type(torch.float16).to(device)
    im2c = t(im2).type(torch.float16).to(device)     

    im1g = tg(im1).type(torch.float16).to(device)
    im2g = tg(im2).type(torch.float16).to(device)     

    d1 = pipe_data_sift    
    ncc.go_save_diff_patches(im1g, im2g, d1['pt1'], d1['pt2'], d1['Hs'], w, save_prefix='patch_sift_base_')
    d2 = pipe_data_sift_ncc    
    ncc.go_save_diff_patches(im1g, im2g, d2['pt1'], d2['pt2'], d2['Hs'], w, save_prefix='patch_sift_base_ncc_')

    d3 = pipe_data_sift_miho    
    ncc.go_save_diff_patches(im1g, im2g, d3['pt1'], d3['pt2'], d3['Hs'], w, save_prefix='patch_sift_miho_')
    d4 = pipe_data_sift_miho_ncc    
    ncc.go_save_diff_patches(im1g, im2g, d4['pt1'], d4['pt2'], d4['Hs'], w, save_prefix='patch_sift_miho_ncc_')
    
    d_pt1 = [d1['pt1'], d2['pt1'], d3['pt1'], d4['pt1']]
    d_pt2 = [d1['pt2'], d2['pt2'], d3['pt2'], d4['pt2']]
    d_Hs = [d1['Hs'], d2['Hs'], d3['Hs'], d4['Hs']]    
    ncc.go_save_list_diff_patches(im1g, im2g, d_pt1, d_pt2, d_Hs, w, save_prefix='patch_sift_list_')

    
    d1 = pipe_data_keynet    
    ncc.go_save_diff_patches(im1g, im2g, d1['pt1'], d1['pt2'], d1['Hs'], w, save_prefix='patch_keynet_base_')
    d2 = pipe_data_keynet_ncc    
    ncc.go_save_diff_patches(im1g, im2g, d2['pt1'], d2['pt2'], d2['Hs'], w, save_prefix='patch_keynet_base_ncc_')

    d3 = pipe_data_keynet_miho    
    ncc.go_save_diff_patches(im1g, im2g, d3['pt1'], d3['pt2'], d3['Hs'], w, save_prefix='patch_keynet_miho_')
    d4 = pipe_data_keynet_miho_ncc    
    ncc.go_save_diff_patches(im1g, im2g, d4['pt1'], d4['pt2'], d4['Hs'], w, save_prefix='patch_keynet_miho_ncc_')
    
    d_pt1 = [d1['pt1'], d2['pt1'], d3['pt1'], d4['pt1']]
    d_pt2 = [d1['pt2'], d2['pt2'], d3['pt2'], d4['pt2']]
    d_Hs = [d1['Hs'], d2['Hs'], d3['Hs'], d4['Hs']]    
    ncc.go_save_list_diff_patches(im1g, im2g, d_pt1, d_pt2, d_Hs, w, save_prefix='patch_keynet_list_')

    