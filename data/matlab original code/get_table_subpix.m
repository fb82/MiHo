function get_table_subpix

s=0.2;
dpath='data';
th_sac=15;

p=dir([dpath filesep '*']);
p_valid=[p.isdir];
p_valid(1:2)=0;
p=p(p_valid);
addpath('matcher');

method={...
    'keynet',          @keynet,         'Key.Net'; ...
    'keynet_upright',  @keynet_upright, 'Key.Net upright'; ...
    'hz',              @hz,             'Hz$^+$'; ...
    'hz_upright',      @hz_upright,     'Hz$^+$ upright'; ...    
    };

corr_method={...
    'lsm',        @lsm,    'LSM'; ...
    'norm_corr',  @ncorr,  'NCORR'; ...
    'fast_match', @fmatch, 'Fast Match'; ...
    };

err=11;
e=[];
for k=1:err
    if mod(k,2)
        e=[e; k*[0 1; 1 0; 0 -1; -1 0]];
    else
        e=[e; k*[1 1; 1 -1; -1 1; -1 -1]];
    end
end
ebase=sum(e.^2,2).^0.5;
[~,sidx]=sort(ebase);
ebase=ebase(sidx);

% kpt_matcher image_pair corr_method hom coarse_fine error
m_table=zeros(size(method,1),length(p),size(corr_method,1),2,2,err)+nan;
m_norm_table=m_table;
for z=1:size(method,1)
    widx=z;
    for i=1:length(p)
        ppath=[dpath filesep p(i).name filesep 'working_' method{widx,1}];    
        for j=1:size(corr_method,1)
            for hom=0:1
                for kk=1:4:size(e,1)                    
                    for k=kk:kk+3                                            
                        middle_homo_file=[ppath filesep 'matches_scale_' num2str(s) '_' method{widx,1} '_sac_' num2str(th_sac) '_err_' num2str(e(k,1)) '_' num2str(e(k,2))  '_' corr_method{j,1} '_hom_' num2str(hom) '.mat'];        
                        if exist(middle_homo_file,'file')~=2
                            disp(['not found: ' middle_homo_file]);
                        else
                            if k==kk
                                data=load(middle_homo_file);
                                data=data.data;
                            else
                                aux=load(middle_homo_file);
                                aux=aux.data;
                                data.mm1=[data.mm1; aux.mm1];
                                data.mm2=[data.mm2; aux.mm2];

                                data.time1=[data.time1; aux.time1];
                                data.time2=[data.time2; aux.time2];

                                data.err1=[data.err1; aux.err1];
                                data.err2=[data.err2; aux.err2];
                            end
                        end
                    end
                    m_table(z,i,j,hom+1,1,ceil(kk/4))=sum(data.err1<=1);
                    m_table(z,i,j,hom+1,2,ceil(kk/4))=sum(data.err2<=1);           
                    m_norm_table(z,i,j,hom+1,1,ceil(kk/4))=length(data.err1);
                    m_norm_table(z,i,j,hom+1,2,ceil(kk/4))=length(data.err2);     
                end
            end
        end
    end
end

m_table=cat(2,m_table,sum(m_table,2));
m_norm_table=cat(2,m_norm_table,sum(m_norm_table,2));

%%%

m_table=cat(6,m_table,sum(m_table,6));
m_norm_table=cat(6,m_norm_table,sum(m_norm_table,6));
%%%

m_table=m_table./m_norm_table*100;

% kpt_matcher image_pair interpol_method hom coarse_fine error
m_table_=zeros(size(method,1),length(p),3,2,2,err)+nan;
m_norm_table_=m_table_;
for z=1:size(method,1)
    widx=z;
    for i=1:length(p)
        ppath=[dpath filesep p(i).name filesep 'working_' method{widx,1}];    
        j=2;
        for hom=0:1
            for w=0:2
                for kk=1:4:size(e,1)                    
                    for k=kk:kk+3                                            
                        middle_homo_file=[ppath filesep 'matches_scale_' num2str(s) '_' method{widx,1} '_sac_' num2str(th_sac) '_err_' num2str(e(k,1)) '_' num2str(e(k,2))  '_' corr_method{j,1} '_hom_' num2str(hom) '_max_interp_' num2str(w) '.mat'];        
                        if exist(middle_homo_file,'file')~=2
                            disp(['not found: ' middle_homo_file]);
                        else
                            if k==kk
                                data=load(middle_homo_file);
                                data=data.data;
                            else
                                aux=load(middle_homo_file);
                                aux=aux.data;
                                data.mm1=[data.mm1; aux.mm1];
                                data.mm2=[data.mm2; aux.mm2];

                                data.time1=[data.time1; aux.time1];
                                data.time2=[data.time2; aux.time2];

                                data.err1=[data.err1; aux.err1];
                                data.err2=[data.err2; aux.err2];
                            end
                        end
                    end
                    m_table_(z,i,w+1,hom+1,1,ceil(kk/4))=sum(data.err1<1);
                    m_table_(z,i,w+1,hom+1,2,ceil(kk/4))=sum(data.err2<1);

                    m_norm_table_(z,i,w+1,hom+1,1,ceil(kk/4))=length(data.err1);
                    m_norm_table_(z,i,w+1,hom+1,2,ceil(kk/4))=length(data.err2);             
                end
            end
        end
    end
end

m_table_=cat(2,m_table_,sum(m_table_,2));
m_norm_table_=cat(2,m_norm_table_,sum(m_norm_table_,2));

%%%

m_table_=cat(6,m_table_,sum(m_table_,6));
m_norm_table_=cat(6,m_norm_table_,sum(m_norm_table_,6));

%%%

m_table_=m_table_./m_norm_table_*100;

doc_header=[...
    '\\documentclass{article}\n'...
    '\n'...
    '\\usepackage{booktabs}\n' ...
    '\\usepackage{subfig}\n' ...
    '\\usepackage[table]{xcolor}\n' ...
    '\\usepackage{graphicx}\n' ...
    '\\usepackage{multirow}\n' ...
    '\\usepackage{fullpage}\n' ...
    '\\usepackage{color}\n' ...
    '\\usepackage{hyperref}\n'...
    '\\usepackage{amssymb}\n' ...
    '\\usepackage{calc}\n' ...
    '\n' ...
    '\\newlength\\MAX \\setlength\\MAX{\\widthof{99.99}}\n' ...   
    '\\newcommand*\\ChartA[2]{\\rlap{\\textcolor{blue!25}{\\rule[-2.8pt]{\\MAX}{2.4ex}}}\\rlap{{\\textcolor{blue!50}{\\rule[-2.8pt]{#2\\MAX}{2.4ex}}}}#1}\n' ...        
    '\\newcommand*\\ChartB[2]{\\rlap{\\textcolor{blue!50}{\\rule[-2.8pt]{\\MAX}{2.4ex}}}\\rlap{{\\textcolor{blue!80}{\\rule[-2.8pt]{#2\\MAX}{2.4ex}}}}#1}\n' ...        
    '\\newcommand*\\cc[1]{\\multicolumn{1}{c}{#1}}\n' ...
    '\\definecolor{lightgray}{gray}{0.875}\n' ...
    '\n' ...
    '\\begin{document}\n' ...
    '\t\\pagestyle{empty}\n'...
    '\t\\listoffigures\n' ...
    '\t\\listoftables\n' ...   
    '\t\\newpage\n' ...      
    ];

doc_footer=...
    '\\end{document}\n';

opath='latex';
system(['mkdir -p ' opath]);
out_file=[opath filesep 'report_subpix.tex'];

fid=fopen(out_file,'w');
fprintf(fid,doc_header);

% for i=1:length(p)
%     fprintf(fid,to_latex_im(p(i)));
% end
% fprintf(fid,'\t\\clearpage\n');     
% fprintf(fid,'\t\\newpage\n');     

pl=[length(p)+1 1:length(p)];
for ii=1:length(pl)
    i=pl(ii);
    if i==length(p)+1
        ip='overall';
        ip_='Overall';
    else
        ip=p(i).name;
        ip_=p(i).name;
    end
    zv=[1 3 2 4];

    for zz=1:length(zv)
        z=zv(zz);
        widx=z;
        mtable=[];
        mntable=[];
        for j=1:size(corr_method,1)
            for hom=1:2
                for ss=1:2
                    mtable=[mtable; squeeze(m_table(z,i,j,hom,ss,:))'];
                    mntable=[mntable; squeeze(m_norm_table(z,i,j,hom,ss,:))'];             
                end
            end
        end
        
        c_prefix=[method{z,3} ' - ``' strrep(ip_,'_','\\_') ''''' - base matching - '];
        l_prefix=[method{z,1} '_' strrep(ip,' ','_') '_base_matching'];

        tolatex(fid,mtable,'%.2f',mtable/max(mtable,[],'all')*100,[c_prefix ' Sub-pix (\\%%)'],[l_prefix '_pct'],1);
        fprintf(fid,'\t\\newpage\n');
    end
end

%%%

pl=[length(p)+1 1:length(p)];
for ii=1:length(pl)
    i=pl(ii);
    if i==length(p)+1
        ip='overall';
        ip_='Overall';
    else
        ip=p(i).name;
        ip_=p(i).name;
    end
    zv=[1 3 2 4];

    for zz=1:length(zv)
        z=zv(zz);
        widx=z;
        mtable_=[];
        mntable_=[];
        for j=1:3
            for hom=1:2
                for ss=1:2
                    mtable_=[mtable_; squeeze(m_table_(z,i,j,hom,ss,:))'];
                    mntable_=[mntable_; squeeze(m_norm_table_(z,i,j,hom,ss,:))'];            
                end
            end
        end
        
        c_prefix=[method{z,3} ' - ``' strrep(ip_,'_','\\_') ''''' - NCC sub-pixel - '];
        l_prefix=[method{z,1} '_' strrep(ip,' ','_') '_subpixel'];

        tolatex(fid,mtable_,'%.2f',mtable_/max(mtable_,[],'all')*100,[c_prefix ' Sup-pix (\\%%)'],[l_prefix '_pct'],2);
        fprintf(fid,'\t\\newpage\n');   
    end
end

%%%

fprintf(fid,doc_footer);
fclose(fid);

system('mkdir tmp');
system(['cp ' out_file ' ' 'tmp' filesep 'aux.tex']);
system(['cp -R latex' filesep 'imgs ' 'tmp' filesep 'imgs']);
system('cd tmp; pdflatex aux.tex');
system('cd tmp; pdflatex aux.tex');
system('export LD_LIBRARY_PATH= && gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/printer -dNOPAUSE -dQUIET -dBATCH -dCompressFonts=true -dSubsetFonts=true -dColorConversionStrategy=/LeaveColorUnchanged -dPrinted=false -sOutputFile=tmp/aux_.pdf tmp/aux.pdf');
movefile(['tmp' filesep 'aux_.pdf'],[out_file(1:end-3) 'pdf']);
system('rm -R tmp');


function tolatex(fid,table_data,fmt,ntable_data,table_caption,table_label,what)

stable=arrayfun(@(x) num2str(x,fmt), table_data, 'UniformOutput', false);
lval=max(cellfun(@(x) length(x), stable),[],'all');
ntable_data=ntable_data/100;

ctable={};
for i=1:size(table_data,1)
    aux=[];
    for j=1:size(table_data,2)
        oo='';
        ll=lval-length(stable{i,j});
        if ll>0
            oo=['\\hphantom{' repmat('9',1,ll) '}'];
        end
        if ntable_data(i,j)<=1
            aux=[aux '& \\ChartA{' oo stable{i,j} '}{' num2str(ntable_data(i,j)) '} '];
        else
            aux=[aux '& \\ChartB{' oo stable{i,j} '}{' num2str(min(1,ntable_data(i,j)-1)) '} '];
        end    
    end
    ctable=[ctable; {aux}];
end

if what==1
    fprintf(fid,tabler(table_caption,table_label,ctable,lval));
else
    fprintf(fid,tabler_(table_caption,table_label,ctable,lval));
end


function s=tabler_(table_caption,table_label,ctable,lval)

s=[...
    '\t\\begin{table}[h!]\n' ...
    '\t\t\\setlength{\\MAX}{\\widthof{' repmat('9',[1 lval]) '}}\n' ...
    '\t\t\\centering\n' ...
    '\t\t\\caption{' table_caption '}\\label{' table_label '}\n' ...
    '\t\t\\resizebox{\\textwidth}{!}{\n' ...
    '\t\t\\begin{tabular}{c<{\\hspace{5pt}}c<{\\hspace{5pt}}c<{\\hspace{5pt}}l<{\\hspace{5pt}}l<{\\hspace{5pt}}l<{\\hspace{5pt}}l<{\\hspace{5pt}}l<{\\hspace{5pt}}l<{\\hspace{5pt}}l<{\\hspace{5pt}}l<{\\hspace{5pt}}l<{\\hspace{5pt}}l<{\\hspace{5pt}}l<{\\hspace{5pt}}l<{\\hspace{5pt}}}\n' ...
    '\t\t\t\\multicolumn{3}{r}{Noise offset mag. (px)} & \\cc{1} & \\cc{$2\\sqrt{2}$} & \\cc{3} & \\cc{5} & \\cc{$4\\sqrt{2}$} & \\cc{7} & \\cc{$6\\sqrt{2}$} & \\cc{9} & \\cc{11} & \\cc{$8\\sqrt{2}$} & \\cc{$10\\sqrt{2}$} & \\cc{avg.} \\\\\n' ...
    '\t\t\t\\toprule\n' ...
    '\t\t\t\\rowcolor{lightgray}\n' ...
    '\t\t\t\\cellcolor{white}{} & \\cellcolor{white}{} & $\\boxplus$ ' ctable{1} '\\\\\n' ...
    '\t\t\t\\rowcolor{white}\n' ...
    '\t\t\t\\cellcolor{white}{} & \\cellcolor{white}{\\multirow{-2}{*}{\\shortstack{no patch\\\\norm.}}} & $\\boxplus\\boxtimes$  ' ctable{2} '\\\\\n' ...
    '\t\t\t\\cmidrule{2-15}\n' ...
    '\t\t\t\\rowcolor{lightgray}\n' ...
    '\t\t\t\\cellcolor{white}{} & \\cellcolor{white}{} & $\\boxplus$ ' ctable{3} '\\\\\n' ...
    '\t\t\t\\rowcolor{white}\n' ...
    '\t\t\t\\cellcolor{white}{\\multirow{-4}{*}{\\rotatebox[origin=l]{90}{none}}} & \\cellcolor{white}{\\multirow{-2}{*}{MiHo}} & $\\boxplus\\boxtimes$ ' ctable{4} ' \\\\\n' ...
    '\t\t\t\\midrule[\\heavyrulewidth]\n' ...
    '\t\t\t\\rowcolor{lightgray}\n' ...
    '\t\t\t\\cellcolor{white}{} & \\cellcolor{white}{} & $\\boxplus$ ' ctable{5} '\\\\\n' ...
    '\t\t\t\\rowcolor{white}\n' ...
    '\t\t\t\\cellcolor{white}{} & \\cellcolor{white}{\\multirow{-2}{*}{\\shortstack{no patch\\\\norm.}}} & $\\boxplus\\boxtimes$ ' ctable{6} '\\\\\n' ...
    '\t\t\t\\cmidrule{2-15}\n' ...
    '\t\t\t\\rowcolor{lightgray}\n' ...
    '\t\t\t\\cellcolor{white}{} & \\cellcolor{white}{} & $\\boxplus$ ' ctable{7} '\\\\\n' ...
    '\t\t\t\\rowcolor{white}\n' ...
    '\t\t\t\\cellcolor{white}{\\multirow{-5}{*}{\\rotatebox[origin=l]{90}{parabolic\\hphantom{ab}}}} & \\cellcolor{white}{\\multirow{-2}{*}{MiHo}} & $\\boxplus\\boxtimes$ ' ctable{8} '\\\\\n' ...
    '\t\t\t\\midrule[\\heavyrulewidth]\n' ...
    '\t\t\t\\rowcolor{lightgray}\n' ...
    '\t\t\t\\cellcolor{white}{} & \\cellcolor{white}{} & $\\boxplus$ ' ctable{9} '\\\\\n' ...
    '\t\t\t\\cellcolor{white}{} & \\cellcolor{white}{\\multirow{-2}{*}{\\shortstack{no patch\\\\norm.}}} & $\\boxplus\\boxtimes$ ' ctable{10} '\\\\\n' ...
    '\t\t\t\\rowcolor{white}\n' ...
    '\t\t\t\\cmidrule{2-15}\n' ...
    '\t\t\t\\rowcolor{lightgray}\n' ...
    '\t\t\t\\cellcolor{white}{} & \\cellcolor{white}{} & $\\boxplus$ ' ctable{11} '\\\\\n' ...
    '\t\t\t\\rowcolor{white}\n' ...
    '\t\t\t\\cellcolor{white}{\\multirow{-4}{*}{\\rotatebox[origin=l]{90}{Taylor}}} & \\cellcolor{white}{\\multirow{-2}{*}{MiHo}} & $\\boxplus\\boxtimes$ ' ctable{12} '\\\\\n' ...
    '\t\t\t\\bottomrule\n' ...
    '\t\t\t\\end{tabular}\n' ...
    '\t\t}\n' ...
    '\t\\end{table}\n' ...
];


function s=tabler(table_caption,table_label,ctable,lval)

s=[...
    '\t\\begin{table}[h!]\n' ...
    '\t\t\\setlength{\\MAX}{\\widthof{' repmat('9',[1 lval]) '}}\n' ...
    '\t\t\\centering\n' ...
    '\t\t\\caption{' table_caption '}\\label{' table_label '}\n' ...
    '\t\t\\resizebox{\\textwidth}{!}{\n' ...
    '\t\t\\begin{tabular}{c<{\\hspace{5pt}}c<{\\hspace{5pt}}c<{\\hspace{5pt}}l<{\\hspace{5pt}}l<{\\hspace{5pt}}l<{\\hspace{5pt}}l<{\\hspace{5pt}}l<{\\hspace{5pt}}l<{\\hspace{5pt}}l<{\\hspace{5pt}}l<{\\hspace{5pt}}l<{\\hspace{5pt}}l<{\\hspace{5pt}}l<{\\hspace{5pt}}l<{\\hspace{5pt}}}\n' ...
    '\t\t\t\\multicolumn{3}{r}{Noise offset mag. (px)} & \\cc{1} & \\cc{$2\\sqrt{2}$} & \\cc{3} & \\cc{5} & \\cc{$4\\sqrt{2}$} & \\cc{7} & \\cc{$6\\sqrt{2}$} & \\cc{9} & \\cc{11} & \\cc{$8\\sqrt{2}$} & \\cc{$10\\sqrt{2}$} & \\cc{avg.} \\\\\n' ...
    '\t\t\t\\toprule\n' ...
    '\t\t\t\\rowcolor{lightgray}\n' ...
    '\t\t\t\\cellcolor{white}{} & \\cellcolor{white}{} & $\\boxplus$ ' ctable{1} '\\\\\n' ...
    '\t\t\t\\rowcolor{white}\n' ...
    '\t\t\t\\cellcolor{white}{} & \\cellcolor{white}{\\multirow{-2}{*}{\\shortstack{no patch\\\\norm.}}} & $\\boxdot\\boxplus$ ' ctable{2} '\\\\\n' ...
    '\t\t\t\\cmidrule{2-15}\n' ...
    '\t\t\t\\rowcolor{lightgray}\n' ...
    '\t\t\t\\cellcolor{white}{} & \\cellcolor{white}{} & $\\boxplus$ ' ctable{3} '\\\\\n' ...
    '\t\t\t\\rowcolor{white}\n' ...
    '\t\t\t\\cellcolor{white}{\\multirow{-4}{*}{\\rotatebox[origin=l]{90}{ALS}}} & \\cellcolor{white}{\\multirow{-2}{*}{MiHo}} & $\\boxdot\\boxplus$ ' ctable{4} ' \\\\\n' ...
    '\t\t\t\\midrule[\\heavyrulewidth]\n' ...
    '\t\t\t\\rowcolor{lightgray}\n' ...
    '\t\t\t\\cellcolor{white}{} & \\cellcolor{white}{} & $\\boxplus$ ' ctable{5} '\\\\\n' ...
    '\t\t\t\\rowcolor{white}\n' ...
    '\t\t\t\\cellcolor{white}{} & \\cellcolor{white}{\\multirow{-2}{*}{\\shortstack{no patch\\\\norm.}}} & $\\boxdot\\boxplus$ ' ctable{6} '\\\\\n' ...
    '\t\t\t\\cmidrule{2-15}\n' ...
    '\t\t\t\\rowcolor{lightgray}\n' ...
    '\t\t\t\\cellcolor{white}{} & \\cellcolor{white}{} & $\\boxplus$ ' ctable{7} '\\\\\n' ...
    '\t\t\t\\rowcolor{white}\n' ...
    '\t\t\t\\cellcolor{white}{\\multirow{-4}{*}{\\rotatebox[origin=l]{90}{NCC}}} & \\cellcolor{white}{\\multirow{-2}{*}{MiHo}} & $\\boxdot\\boxplus$ ' ctable{8} '\\\\\n' ...
    '\t\t\t\\midrule[\\heavyrulewidth]\n' ...
    '\t\t\t\\rowcolor{lightgray}\n' ...
    '\t\t\t\\cellcolor{white}{} & \\cellcolor{white}{} & $\\boxplus$ ' ctable{9} '\\\\\n' ...
    '\t\t\t\\cellcolor{white}{} & \\cellcolor{white}{\\multirow{-2}{*}{\\shortstack{no patch\\\\norm.}}} & $\\boxdot\\boxplus$ ' ctable{10} '\\\\\n' ...
    '\t\t\t\\rowcolor{white}\n' ...
    '\t\t\t\\cmidrule{2-15}\n' ...
    '\t\t\t\\rowcolor{lightgray}\n' ...
    '\t\t\t\\cellcolor{white}{} & \\cellcolor{white}{} & $\\boxplus$ ' ctable{11} '\\\\\n' ...
    '\t\t\t\\rowcolor{white}\n' ...
    '\t\t\t\\cellcolor{white}{\\multirow{-5}{*}{\\rotatebox[origin=l]{90}{FAsT-Match\\hphantom{a}}}} & \\cellcolor{white}{\\multirow{-2}{*}{MiHo}} & $\\boxdot\\boxplus$ ' ctable{12} '\\\\\n' ...
    '\t\t\t\\bottomrule\n' ...
    '\t\t\t\\end{tabular}\n' ...
    '\t\t}\n' ...
    '\t\\end{table}\n' ...
];


function latex_im=to_latex_im(p)

system('mkdir -p latex/imgs');

im1=[p.folder filesep p.name filesep 'img1.jpg'];
im=imread(im1);
sz=size(im);
sz=1024/min(sz(1:2));
im=imresize(im,sz);
img1=['latex/imgs/' p.name '_1.pdf'];
h=figure('visible','off');
h.RendererMode='manual';
h.Renderer='painters';
imshow(im,'Border','tight');
hold on;
gt=load([p.folder filesep p.name filesep 'gt.mat']);
gt=gt.gt*sz;
plot(gt(:,1),gt(:,2),'og','MarkerSize',8,'LineWidth',3);
saveas(h,'out.svg');
close(h);
system('rsvg-convert -f pdf out.svg > out_.pdf');
pdf_out=img1;
system('export LD_LIBRARY_PATH= && pdfcrop out_.pdf out__.pdf');
system(['export LD_LIBRARY_PATH= && gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/printer -dNOPAUSE -dQUIET -dBATCH -dCompressFonts=true -dSubsetFonts=true -dColorConversionStrategy=/LeaveColorUnchanged -dPrinted=false -sOutputFile=' pdf_out ' out__.pdf']);
delete('out__.pdf');
delete('out_.pdf');
delete('out.svg');

im2=[p.folder filesep p.name filesep 'img2.jpg'];
im=imread(im2);
sz=size(im);
sz=1024/min(sz(1:2));
im=imresize(im,sz);
img2=['latex/imgs/' p.name '_2.pdf'];
h=figure('visible','off');
h.RendererMode='manual';
h.Renderer='painters';
imshow(im,'Border','tight');
hold on;
gt=load([p.folder filesep p.name filesep 'gt.mat']);
gt=gt.gt*sz;
plot(gt(:,3),gt(:,4),'og','MarkerSize',8,'LineWidth',3);
saveas(h,'out.svg');
close(h);
system('rsvg-convert -f pdf out.svg > out_.pdf');
pdf_out=img2;
system('export LD_LIBRARY_PATH= && pdfcrop out_.pdf out__.pdf');
system(['export LD_LIBRARY_PATH= && gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/printer -dNOPAUSE -dQUIET -dBATCH -dCompressFonts=true -dSubsetFonts=true -dColorConversionStrategy=/LeaveColorUnchanged -dPrinted=false -sOutputFile=' pdf_out ' out__.pdf']);
delete('out__.pdf');
delete('out_.pdf');
delete('out.svg');

img1=['imgs/' p.name '_1.pdf'];
img2=['imgs/' p.name '_2.pdf'];
im_label=strrep(p.name,'','_');
im_title=['``' strrep(p.name,'_','\\_') ''''''];

latex_im=[
    '\t\\begin{figure}\n' ...
	'\t\t\\center\n' ...
	'\t\t\\subfloat[$1^{st}$ image]{\n' ...
	'\t\t\t\\includegraphics[width=0.45\\textwidth]{' img1 '}\n' ...
	'\t\t}\n' ...
	'\t\t\\hfil\n' ...
	'\t\t\\subfloat[$2^{nd}$ image]{\n' ...
	'\t\t\t\\includegraphics[width=0.45\\textwidth]{' img2 '}\n' ...
	'\t\t}\n' ...
	'\t\t\\\\\n' ...    
	'\t\t\\caption{\n' ...
    '\t\t\t\\label{' im_label '}\n' ...
	'\t\t' im_title '\n' ...
    '\t\t}\n' ...
    '\t\\end{figure}\n' ...
    ];

