import os
import src.base_modules as pipe_base
import src.miho as miho_duplex
import src.miho_other as miho_unduplex
import src.ncc as ncc
import src.GMS.gms_custom as gms
import src.OANet.learnedmatcher_custom as oanet
import src.ACNe.acne_custom as acne
import src.AdaLAM.adalam_custom as adalam
import src.DeDoDe2.dedode2_custom as dedode2
import src.bench_utils as bench
import numpy as np

# from src.DIM_modules.superpoint_lightglue_module import superpoint_lightglue_module
# from src.DIM_modules.disk_lightglue_module import disk_lightglue_module
# from src.DIM_modules.aliked_lightglue_module import aliked_lightglue_module
# from src.DIM_modules.loftr_module import loftr_module


def csv_write(lines, save_to='nameless.csv'):

    with open(save_to, 'w') as f:
        for l in lines:
            f.write(l)   


def csv_merger(csv_list):
    avg_idx = [[ 3,  6, 'F_mAA@avg_a'],
               [ 6,  9, 'E_mAA@avg_a'],
               [11, 14, 'F_mAA@avg_a'],
               [14, 17, 'E_mAA@avg_a'],
               [19, 22, 'H_mAA@avg_m'],
               [24, 27, 'F_mAA@avg_a'],
               [27, 30, 'F_mAA@avg_m'],
               [30, 33, 'E_mAA@avg_a'],
               [33, 36, 'E_mAA@avg_m'],
               ]
        
    csv_data = []
    for csv_file in csv_list:
        aux = [csv_line.split(';') for csv_line in  open(csv_file, 'r').read().splitlines()]
        to_fuse = max([idx for idx, el in enumerate([s.startswith('pipe_module') for s in aux[0]]) if el == True]) + 1

        tmp = {}
        for row in aux:
            what = ';'.join(row[:to_fuse]).replace('_outdoor_true','').replace('_outdoor_false','').replace('_fundamental_matrix','').replace('_homography','')
            tmp[what] = row[to_fuse:]

        csv_data.append(tmp)
    
    merged_csv = []
    for k in csv_data[0].keys():
        merged_csv.append([k] + [el for curr_csv in csv_data for el in curr_csv[k]])
        
    avg_csv = []
    for k, row in enumerate(merged_csv):
        if k==0:
            avg_list = [rrange[2] for rrange in avg_idx]
        else:
            avg_list = [np.mean([float(i) for i in row[rrange[0]:rrange[1]]]) for rrange in avg_idx]
        avg_csv.append(avg_list)

    fused_csv = []
    for row_base, row_avg in zip(merged_csv, avg_csv):
        row_new =  []
        for k in range(len(avg_idx) - 1, - 1, - 1):
            if k == 0:
                l = 0
            else:
                l = avg_idx[k - 1][1]
                
            if k == len(avg_idx) - 1:
                r = len(row_base)
            else:
                r = avg_idx[k][1]
                               
            row_new =  row_base[l:r] + [str(row_avg.pop())] + row_new 
        fused_csv.append(row_new)
        
    only_num_csv = [row[1:] for row in fused_csv[1:]]
    m = np.asarray(only_num_csv, dtype=float)
    sidx = np.argsort(-m, axis=0)
    sidx_ = np.argsort(sidx, axis=0)
    fused_csv_order = np.full((m.shape[0] + 1, m.shape[1] + 1), np.NaN)
    fused_csv_order[1:,1:] = sidx_

    return fused_csv, fused_csv_order


def to_latex(csv_data, csv_order, renaming_list, header_hold=None):    
    header_type = 'nmmmmmmmmmmsssssssssshhhhhhpppppppppppppppppp'
  # header_hold = 'nmm---m---mss---s---shh---hpp---p---p---p---p'
    
    if header_hold is None:
        header_hold = header_type

    use_ghost = True
    header_dict = {
        'n': '',
        'm': 'MegaDepth',
        's': 'ScanNet',
        'h': 'Planar',
        'p': 'PhotoTourism'
        }

    # removed unwanted columns
    csv_data_new = []
    csv_order_new = []
    header_type_new = ''

    for i in range(len(csv_data)):
        csv_data_new.append([csv_data[i][j] for j in range(len(header_hold)) if header_hold[j] != '-'])
        csv_order_new.append([csv_order[i][j] for j in range(len(header_hold)) if header_hold[j] != '-'])

    for i in range(len(header_hold)):
        if header_hold[i] != '-':
            header_type_new = header_type_new + header_type[i]

    csv_data = csv_data_new
    csv_order = csv_order_new
    header_type = header_type_new
    header_hold = header_hold.replace('-','') 
        
    # starting
    csv_head = csv_data[0]
    csv_head[0] = 'pipeline'
    
    csv_data = csv_data[1:]
    
    pipe_name = [row[0] for row in csv_data]
    base_index = np.argwhere(np.asarray([';;;' in row for row in pipe_name])).squeeze()
    renaming_list.append([';;;', ''])
    renaming_list.append([';;', ''])
    renaming_list.append([';', '+'])
    
    pipe_renamed = []
    for pipe in pipe_name:
        for renamed in renaming_list:
            pipe = pipe.replace(renamed[0], renamed[1])
        if pipe[-1] == '+':
            pipe = pipe[:-1]
        pipe_renamed.append(pipe)
                
        
    sort_idx = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pipe_renamed))]

    clean_pipe_renamed = []
    for i, pipe in enumerate(pipe_renamed):
        if i == base_index:
            clean_pipe_renamed.append(pipe)
        else:
            clean_pipe_renamed.append(pipe.replace(pipe_renamed[base_index], ''))
            
    clean_csv = [csv_head] + [[clean_pipe_renamed[i]] + csv_data[i][1:] for i in sort_idx]
    clean_csv_order = [csv_order[0]] + [csv_order[i + 1] for i in sort_idx]
        
    # csv_write([';'.join(csv_row) + '\n' for csv_row in clean_csv],'clean_table.csv')
    
    for i in range(1, len(clean_csv)):
        for j in range(len(clean_csv[0])):
            vv = clean_csv[i][j]
            
            try:
                v = float(vv)
            except:
                v = vv

            # numeric value            
            if isinstance(v, (int, float)):
                v = "{n:6.2f}".format(n=v*100)

                # avoid alignement issues
                if use_ghost == True:
                    for g in range(len(v)):
                        if v[g] != ' ':
                            break
                    v = "\hphantom{" + "0" * g + "}" + v[g:]

                # highlight top pipelines for each column
                c_rank = int(clean_csv_order[i][j])        
                if c_rank < 9:
                    v = '\\textcolor{C' + str(c_rank) + '}{\\contour{C' + str(c_rank) + '}{' + v + '}}'

            # text data in latex
            v = v.replace('MOP','\\textbf{MOP}')                  
            v = v.replace('NCC','\\textbf{NCC}')                  
            v = v.replace('MiHo','\\textbf{MiHo}')                  
            v = v.replace('MAGSAC^','MAGSAC$_\\uparrow$')                  
            v = v.replace('MAGSACv','MAGSAC$_\\downarrow$')                  

            # row alternate gray color    
            if (i != 0) and (j == 0) and (i % 2 == 0):
                v = '\\rowcolor{gray!15} ' + v

            # add tabs
            if (j == 0):
                v = '\t' * 4 + v
            
            clean_csv[i][j] = v
            
    # add the & separator and the \\ at the end of the row
    latex_table = [' & '.join(row) + " \\\\\n" for i, row in enumerate(clean_csv) if i > 0]

    header = [
        '\\documentclass[a4paper,landscape,10pt]{article}\n',
        '\\usepackage[empty]{fullpage}\n',        
        '\\usepackage{graphicx}\n',
        '\\usepackage{caption}\n',
        '\\usepackage{color}\n',
        '\\usepackage{adjustbox}\n',
        '\\usepackage{multirow}\n',
        '\\usepackage{booktabs}\n',
        '\\usepackage{amssymb}\n',
        '\\usepackage[table,usenames,dvipsnames]{xcolor}\n',
        '\\usepackage{amsmath}\n',
        '\\usepackage{multirow}\n',
        '\\usepackage{calc}\n',
        '\\usepackage{ulem}\n',
        '\\usepackage{nicefrac}\n',
        '\\usepackage[outline]{contour}\n',
        '\n',
        '\\newlength\\MAX\\setlength\\MAX{\\widthof{999999999}}\n',
        '\\newcommand*\\Chart[5]{\\rlap{\\textcolor{#3!#5}{\\rule[-0.5ex]{\\MAX}{3ex}}}\\rlap{\\textcolor{#3!#4}{\\rule[-0.5ex]{#2\\MAX}{3ex}}}#1}\n',
        '\n',
        '\\newcolumntype{L}[1]{>{\\raggedright\\let\\newline\\\\\\arraybackslash\\hspace{0pt}}m{#1}}\n',
        '\\newcolumntype{C}[1]{>{\\centering\\let\\newline\\\\\\arraybackslash\\hspace{0pt}}m{#1}}\n',
        '\\newcolumntype{R}[1]{>{\\raggedleft\\let\\newline\\\\\\arraybackslash\\hspace{0pt}}m{#1}}\n',
        '\n',
        # colors from https://github.com/riccardosven/tableaucolors
        '\\definecolor{C0}{HTML}{1F77B4}\n',
        '\\definecolor{C1}{HTML}{FF7F0E}\n',
        '\\definecolor{C2}{HTML}{2CA02C}\n',
        '\\definecolor{C3}{HTML}{D62728}\n',
        '\\definecolor{C4}{HTML}{9467BD}\n',
        '\\definecolor{C5}{HTML}{8C564B}\n',
        '\\definecolor{C6}{HTML}{E377C2}\n',
        '\\definecolor{C7}{HTML}{7F7F7F}\n',
        '\\definecolor{C8}{HTML}{BCBD22}\n',
        '\\definecolor{C9}{HTML}{17BECF}\n',
        '\n',        
        '\\begin{document}\n',        
        '\t\\contourlength{0.1pt}\n',
        '\t\\contournumber{10}\n',
        '\t\\begin{table}[t!]\n',
		'\t\\renewcommand{\\arraystretch}{0}\n',
        '\t\\setlength{\\tabcolsep}{4pt}\n',
        '\t\\centering\n',
        '\t\t\\resizebox{\\textwidth}{!}{\n',
        '\t\t\\begin{tabular}{l' + ('L{\\MAX}' * (len(header_type)-1)) + '}\n',
    ]
    
    # header formatting
    l=0
    header_current = header_type[0]
    header_multi = []
    header_rule = []
    header_type_ = header_type + '$'
    for i in range(1,len(header_type_)):
        if header_type_[i] != header_current:
            header_multi.append('\\multicolumn{' + str(i-l)  + '}{c}{' + header_dict[header_current] + '}')
            if l + 1 != i: header_rule.append('\\cmidrule(lr){' +  str(l + 1)  + '-' + str(i) + '}')
            l = i
            header_current = header_type_[i]
        
    header.append('\t\t\t\t' + ' & '.join(header_multi) + ' \\\\\n')
    header.append('\t\t\t\t' + ''.join(header_rule) + '\n')
    
    header_spec = []               
    for v in csv_head:
        v = v.replace('pipeline', 'Pipeline')
        v = v.replace('F_precision', 'Prec.')
        v = v.replace('F_recall', 'Recall')
        v = v.replace('H_precision', 'Prec.')
        v = v.replace('H_recall', 'Recall')
        v = v.replace('F_mAA', 'mAA$^{F}$')
        v = v.replace('E_mAA', 'mAA$^{E}$')
        v = v.replace('H_mAA', 'mAA$^{H}$')
        v = v.replace('@5', '$_{\\text{@}5}$')
        v = v.replace('@10', '$_{\\text{@}10}$')
        v = v.replace('@15', '$_{\\text{@}15}$')
        v = v.replace('@20', '$_{\\text{@}20}$')
        v = v.replace('@(5,0.5)', '$_{\\text{@}(5,.5)}$')
        v = v.replace('@(10,1)', '$_{\\text{@}(10,1)}$')
        v = v.replace('@(20,2)', '$_{\\text{@}(20,2)}$')
        v = v.replace('@avg_a', '$_\\measuredangle$')
        v = v.replace('@avg_m', '$_\\square$')
        v = v.replace('$$', '')
        header_spec.append(v)
        
    header.append('\t\t\t\t' + ' & '.join(header_spec) + ' \\\\\n')    
    header.append('\t\t\t\t\\midrule\n')

    
    footer = [
        '\t\t\t\end{tabular}\n',
        '\t\t}\n',
        '\t\t\\caption{No caption (best viewed in color and zoomed in).}\\label{none}\n',
        '\t\\end{table}\n',
        '\\end{document}\n',
    ]
    
    latex_table = header + latex_table + footer
    
    csv_write(latex_table, save_to='la.tex')
        
    return latex_table


if __name__ == '__main__':    

    pipes = [
        [ 'MAGSAC^', pipe_base.magsac_module(px_th=1.00)],
        [ 'MAGSACv', pipe_base.magsac_module(px_th=0.75)],
        [     'NCC', ncc.ncc_module(also_prev=True)],
        ['MOP+MiHo', miho_duplex.miho_module()],
        [     'MOP', miho_unduplex.miho_module()],
        [     'GMS', gms.gms_module()],
        [   'OANet', oanet.oanet_module()],
        [  'AdaLAM', adalam.adalam_module()],
        [    'ACNe', acne.acne_module()],
    ]

    pipe_heads = [
        [             'Key.Net', pipe_base.keynetaffnethardnet_module(num_features=8000, upright=True, th=0.99)],
        # [                'SIFT', pipe_base.sift_module(num_features=8000, upright=True, th=0.95, rootsift=True)],     
        # ['SuperPoint+LightGlue', pipe_base.lightglue_module(num_features=8000, upright=True, what='superpoint')],
        [    'ALIKED+LightGlue', pipe_base.lightglue_module(num_features=8000, upright=True, what='aliked')],
        # [      'DISK+LightGlue', pipe_base.lightglue_module(num_features=8000, upright=True, what='disk')],  
        # [               'LoFTr', pipe_base.loftr_module(num_features=8000, upright=True)],        
        # [             'DeDoDe2', dedode2.dedode2_module(num_features=8000, upright=True)],                
      # # ['SuperPoint+LightGlue (DIM)', superpoint_lightglue_module(nmax_keypoints=8000)],
      # # [    'ALIKED+LightGlue (DIM)', aliked_lightglue_module(nmax_keypoints=8000)],
      # # [      'DISK+LightGlue (DIM)', disk_lightglue_module(nmax_keypoints=8000)],
      # # [               'LoFTr (DIM)', loftr_module(nmax_keypoints=8000)],  
        ]
    
###

    pipe_renamed = []
    for pipe in pipes:
        new_name = pipe[0]
        old_name = pipe[1].get_id().replace('_outdoor_true','').replace('_outdoor_false','').replace('_fundamental_matrix','').replace('_homography','')
        pipe_renamed.append([old_name, new_name])

    for pipe in pipe_heads:
        new_name = pipe[0]
        old_name = pipe[1].get_id().replace('_outdoor_true','').replace('_outdoor_false','').replace('_fundamental_matrix','').replace('_homography','')
        pipe_renamed.append([old_name, new_name])

    bench_path = '../test_csv_merger'   
    save_to = 'res'
    
    benchmark_data = {
            'megadepth': {'name': 'megadepth', 'Name': 'MegaDepth', 'setup': bench.megadepth_bench_setup, 'is_outdoor': True, 'is_not_planar': True, 'ext': '.png', 'use_scale': True, 'also_metric': False},
            'scannet': {'name': 'scannet', 'Name': 'ScanNet', 'setup': bench.scannet_bench_setup, 'is_outdoor': False, 'is_not_planar': True, 'ext': '.png', 'use_scale': False, 'also_metric': False},
            'planar': {'name': 'planar', 'Name': 'Planar', 'setup': bench.planar_bench_setup, 'is_outdoor': True, 'is_not_planar': False, 'ext': '.png', 'use_scale': False, 'also_metric': False},
            'imc_phototourism': {'name': 'imc_phototourism', 'Name': 'IMC PhotoTourism', 'setup': bench.imc_phototourism_bench_setup, 'is_outdoor': True, 'is_not_planar': True, 'ext': '.jpg', 'use_scale': False, 'also_metric': True},
        }
    
###    
    
    for ip in range(len(pipe_heads)):
        csv_list = []
        pipe_head = pipe_heads[ip][1]

        for b in benchmark_data.keys():
            to_save_file =  os.path.join(bench_path, save_to, save_to + '_' + pipe_head.get_id() + '_')
            to_save_file_suffix ='_' + benchmark_data[b]['name']

            if benchmark_data[b]['is_not_planar']:
                csv_list.append(to_save_file + 'fundamental_and_essential' + to_save_file_suffix + '.csv')
            else:
                csv_list.append(to_save_file + 'homography' + to_save_file_suffix + '.csv')
                
        fused_csv, fused_csv_order = csv_merger(csv_list)
        csv_write([';'.join(csv_row) + '\n' for csv_row in fused_csv], to_save_file.replace('_outdoor_true','').replace('_outdoor_false','') + '.csv')
        
        to_latex(fused_csv, fused_csv_order, pipe_renamed)