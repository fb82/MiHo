function go

s=0.2;
dpath='data';
% what='keynet';
th_sac=15;
th_cf=2;

p=dir([dpath filesep '*']);
p_valid=[p.isdir];
p_valid(1:2)=0;
p=p(p_valid);
addpath('matcher');

method={...
    'keynet',          @keynet; ...
    'keynet_upright',  @keynet_upright; ...
    'hz',              @hz; ...
    'hz_upright',      @hz_upright; ...    
    };

corr_method={...
    'lsm',        @lsm; ...
    'norm_corr',  @ncorr; ...
    'fast_match', @fmatch; ...
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

for z=1:size(method,1)
    widx=z;
%   widx=find(strcmp(method(:,1),what));
    for i=1:length(p)
        bpath=[dpath filesep p(i).name];
        ppath=[dpath filesep p(i).name filesep 'working_' method{widx,1}];
        system(['mkdir -p ' ppath])
    
        im1h_name=[bpath filesep 'img1.jpg'];
        im1l_name=[ppath filesep 'img1_scale_' num2str(s) '.png'];
        if exist(im1l_name,'file')~=2
            im=imresize(imread(im1h_name),s);
            imwrite(im,im1l_name);
        end
    
        im2h_name=[bpath filesep 'img2.jpg'];    
        im2l_name=[ppath filesep 'img2_scale_' num2str(s) '.png'];
        if exist(im2l_name,'file')~=2
            im=imresize(imread(im2h_name),s);
            imwrite(im,im2l_name);
        end
    
        gt_file=[bpath filesep 'gt.mat'];
        if exist(gt_file,'file')==2
            gt=load(gt_file);
            gt=gt.gt;
        else
            gt=dlmread([bpath filesep 'gt.txt']);
            save(gt_file,'gt');
            delete([bpath filesep 'gt.txt']);        
        end
    
        match_file=[ppath filesep 'matches_scale_' num2str(s) '_' method{widx,1} '_sac_' num2str(th_sac) '.mat'];
        if exist(match_file,'file')==2
            matches=load(match_file);
            midx=matches.midx;
            matches=matches.matches;
        else
            cd('matcher');
            matches=method{widx,2}(['..' filesep im1l_name],['..' filesep im2l_name]);
            [~,midx]=fun_sac_matrix(matches(:,[1 2]),matches(:,[3 4]),th_sac);
            cd('..');    
            save(match_file,'matches','midx');    
        end
    
        gt_scaled=gt*s;
    
        mm1=pdist2(matches(:,1:2),gt_scaled(:,1:2));
        mm2=pdist2(matches(:,3:4),gt_scaled(:,3:4));
        to_remove_matches=any(mm1<th_sac*th_cf,2)|any(mm2<th_sac*th_cf,2);
        hom_matches=matches(~to_remove_matches,:);
    
        for k=1:size(e,1)
            aux=gt_scaled;
            aux(:,[3 4])=aux(:,[3 4])+repmat(e(k,:),[size(gt_scaled,1) 1]);
            all_matches=[aux; hom_matches];
    
            middle_homo_file=[ppath filesep 'matches_scale_' num2str(s) '_' method{widx,1} '_sac_' num2str(th_sac) '_err_' num2str(e(k,1)) '_' num2str(e(k,2))  '_middle_homo.mat'];
            th_out=ceil(th_sac/2);
            if exist(middle_homo_file,'file')==2
                Hdata=load(middle_homo_file);
                didx=Hdata.didx;
                Hdata=Hdata.Hdata;
            else
                [didx,Hdata]=middle_homo(im1l_name,im2l_name,all_matches,th_sac,th_out);
                save(middle_homo_file,'Hdata','didx');
            end
    
            im1=imread(im1l_name);
            im2=imread(im2l_name);
    
            to_check_matches=all_matches(1:size(gt_scaled,1),:);
            hom_data.Hdata=Hdata;
            hom_data.didx=didx(1:size(gt_scaled,1));
    
            j=2;
            for hom=0:1
                for w=0:2
                    middle_homo_file=[ppath filesep 'matches_scale_' num2str(s) '_' method{widx,1} '_sac_' num2str(th_sac) '_err_' num2str(e(k,1)) '_' num2str(e(k,2))  '_' corr_method{j,1} '_hom_' num2str(hom) '_max_interp_' num2str(w) '.mat'];        
                    if exist(middle_homo_file,'file')~=2                
                        [data.mm1,ttime1]=kpt_improver(im1,im2,to_check_matches,corr_method(j,1),th_sac,1,1,hom,hom_data,w);
                        [data.mm2,ttime2]=kpt_improver(im1,im2,to_check_matches,corr_method([j 1],1),[th_sac th_sac],[1 1],1,hom,hom_data,w);
        
                        data.time1=ttime1;
                        data.time2=ttime2;
    
                        data.err1=sum((data.mm1(:,3:4)-gt_scaled(:,[3 4])).^2,2).^0.5;
                        data.err2=sum((data.mm2(:,3:4)-gt_scaled(:,[3 4])).^2,2).^0.5;
    
                        save(middle_homo_file,'data');
                    else
                        data=load(middle_homo_file);
                        data=data.data;
                    end
                end
            end
        end
    end
end

function [didx,Hdata]=middle_homo(im1,im2,matches,th,th_out)

pt1=matches(:,[1 2]);
pt2=matches(:,[3 4]);

Hdata=get_avg_hom(pt1,pt2,th,th_out);

im1=imread(im1);
im2=imread(im2);

midx=[];
for i=1:size(Hdata,1)
    midx=[midx; Hdata{i,3}];
end
midx=midx>0;

sidx=sum(midx,2);
[~,didx]=max(repmat(sidx,[1 size(midx,2)]).*midx);

Hdata=Hdata(:,[1 2]);

function Hdata=get_avg_hom(pt1,pt2,th,th_out)

H1=eye(3);
H2=eye(3);

Hdata={};
max_iter=5;
midx=zeros(1,size(pt1,1));
tidx=zeros(1,size(pt1,1));
hc=1;

while 1
    pt1_=pt1(~midx,:);
    pt1_=[pt1_ ones(size(pt1_,1),1)]';
    pt1_=H1*pt1_;
    pt1_=pt1_(1:2,:)./repmat(pt1_(3,:),[2 1]);
    
    pt2_=pt2(~midx,:);
    pt2_=[pt2_ ones(size(pt2_,1),1)]';
    pt2_=H2*pt2_;
    pt2_=pt2_(1:2,:)./repmat(pt2_(3,:),[2 1]);
        
    [H1_,H2_,nidx,oidx]=ransac_middle(pt1_',pt2_',th,th_out);

    if (sum(nidx)<=4)
        break;
    end

    zidx=zeros(1,size(midx,2),'logical');
    zidx(~midx)=nidx;

    tidx(~midx)=tidx(~midx)|nidx;
    midx(~midx)=oidx*hc;

    H1_new=H1_*H1;
    H2_new=H2_*H2;
    for i=1:max_iter    
        pt1_=pt1(zidx,:);
        pt1_=[pt1_ ones(size(pt1_,1),1)]';
        pt1_=H1_new*pt1_;
        pt1_=pt1_./repmat(pt1_(3,:),[3 1]);
        
        pt2_=pt2(zidx,:);
        pt2_=[pt2_ ones(size(pt2_,1),1)]';
        pt2_=H2_new*pt2_;
        pt2_=pt2_./repmat(pt2_(3,:),[3 1]);

        ptm=(pt1_+pt2_)/2;
        H1_=compute_homography(pt1_,ptm);
        H2_=compute_homography(pt2_,ptm);

        H1_new=H1_*H1_new;
        H2_new=H2_*H2_new;
    end

    Hdata=[Hdata; {H1_new, H2_new, zidx}];

    hc=hc+1;
end


function [H1,H2,midx,oidx,c]=ransac_middle(pt1,pt2,th,th_out)

warning('off','MATLAB:nearlySingularMatrix');
warning('off','MATLAB:nchoosek:LargeCoefficient');

% th=3;
max_iter=10000;
min_iter=100;
p=0.9;
c=0;

n=size(pt1,1);
th=th^2;
th_out=th_out^2;

if n<4
    H1=[];
    H2=[];
    midx=zeros(1,n,'logical');
    oidx=zeros(1,n,'logical');    
    return;
end

min_iter=min(min_iter,nchoosek(n,2));

pt1=[pt1 ones(n,1)]';
pt2=[pt2 ones(n,1)]';

midx=zeros(1,n,'logical');
oidx=zeros(1,n,'logical');
Nc=inf;
for c=1:max_iter    
    sidx=randperm(n,4);
    ptm=(pt1+pt2)/2;
    [H1,eD]=compute_homography(pt1(:,sidx),ptm(:,sidx));
    if eD(end-1)<0.05
        continue;
    end    
    [H2,eD]=compute_homography(pt2(:,sidx),ptm(:,sidx));
    if eD(end-1)<0.05
        continue;
    end    

    nidx=get_hom_inliers(pt1,ptm,H1,th,sidx)&get_hom_inliers(pt2,ptm,H2,th,sidx);
    if sum(nidx)>sum(midx)
        midx=nidx;
        sidx_=sidx;
        Nc=steps(4,sum(midx)/numel(midx),p);
    end
    if (c>Nc)&&(c>min_iter)
        break;
    end
end
if any(midx)
    H1=compute_homography(pt1(:,midx),ptm(:,midx));
    H2=compute_homography(pt2(:,midx),ptm(:,midx));
    midx=get_hom_inliers(pt1,ptm,H1,th,sidx_)&get_hom_inliers(pt2,ptm,H2,th,sidx_);
    oidx=get_hom_inliers(pt1,ptm,H1,th_out,sidx_)&get_hom_inliers(pt2,ptm,H2,th_out,sidx_);
else
    H1=[];
    H2=[];
    midx=zeros(1,n,'logical');
    oidx=zeros(1,n,'logical');    
    return;
end

warning('on','MATLAB:nearlySingularMatrix');
warning('on','MATLAB:nchoosek:LargeCoefficient');


function [H,midx,c]=ransac_similarity(pt1,pt2,th)

warning('off','MATLAB:nearlySingularMatrix');
warning('off','MATLAB:nchoosek:LargeCoefficient');

% th=3;
max_iter=10000;
min_iter=100;
p=0.9;
c=0;

n=size(pt1,1);
th=th^2;

if n<2
    H=[];
    midx=zeros(1,n,'logical');
    return;
end

min_iter=min(min_iter,nchoosek(n,2));

pt1=[pt1 ones(n,1)]';
pt2=[pt2 ones(n,1)]';

midx=zeros(1,n,'logical');
H=[];
Nc=inf;
for c=1:max_iter    
    sidx=randperm(n,2);
    H=compute_similarity(pt1(:,sidx),pt2(:,sidx));  
    nidx=get_similarity_inliers(pt1,pt2,H,th);
    if sum(nidx)>sum(midx)
        midx=nidx;
        Nc=steps(2,sum(midx)/numel(midx),p);
    end
    if (c>Nc)&&(c>min_iter)
        break;
    end
end
if any(midx)
    H=compute_similarity(pt1(:,midx),pt2(:,midx));
    midx=get_similarity_inliers(pt1,pt2,H,th);
end

warning('on','MATLAB:nearlySingularMatrix');
warning('on','MATLAB:nchoosek:LargeCoefficient');


function [H,D]=compute_homography(pts1,pts2)

T1=data_normalize(pts1);
T2=data_normalize(pts2);

npts1=T1*pts1;
npts2=T2*pts2;

l=size(npts1,2);
A=[zeros(l,3) -repmat(npts2(3,:)',[1 3]).*npts1' repmat(npts2(2,:)',[1 3]).*npts1';...
    repmat(npts2(3,:)',[1 3]).*npts1' zeros(l,3) -repmat(npts2(1,:)',[1 3]).*npts1';...
    -repmat(npts2(2,:)',[1 3]).*npts1' repmat(npts2(1,:)',[1 3]).*npts1' zeros(l,3)];
[~,D,V]=svd(A);
D=diag(D);
H=reshape(V(:,9),[3 3])';
H=T2\H*T1;


function H=compute_similarity(pts1,pts2)

T1=data_normalize(pts1);
T2=data_normalize(pts2);

npts1=T1*pts1;
npts2=T2*pts2;

x=npts1(1,:)';
y=npts1(2,:)';

x_=npts2(1,:)';
y_=npts2(2,:)';

z=zeros(length(x),1);
o=ones(length(x),1);

A=[...
   x -y o z; ...
   y  x z o; ...
  ];

B=[...
    x_; ...
    y_; ...
  ];

X=pinv(A)*B;

H=[...
    X(1) -X(2) X(3); ...
    X(2) X(1)  X(4); ...
      0    0     1 ; ...
  ];

H=T2\H*T1;


function T=data_normalize(pts)

c=mean(pts,2);
s=sqrt(2)/mean(sqrt((pts(1,:)-c(1)).^2+(pts(2,:)-c(2)).^2));
T=[s 0 -c(1)*s; 0 s -c(2)*s; 0 0 1];


function [nidx,err]=get_hom_inliers(pt1,pt2,H,th,sidx)

pt2_=H*pt1;
s2_=sign(pt2_(3,:));
tmp2_=pt2_(1:2,:)./repmat(pt2_(3,:),[2 1])-pt2(1:2,:);
err2=sum(tmp2_.^2);
s2=s2_(sidx(1));
if ~all(s2_(sidx)==s2)
    nidx=zeros(1,size(pt1,2),'logical');
    err=inf;
    return;
end

pt1_=H\pt2;
s1_=sign(pt1_(3,:));
tmp1_=pt1_(1:2,:)./repmat(pt1_(3,:),[2 1])-pt1(1:2,:);
err1=sum(tmp1_.^2);

s1=s1_(sidx(1));
if ~all(s1_(sidx)==s1)
    nidx=zeros(1,size(pt1,2),'logical');
    err=inf;
    return;
end

err=max([err1; err2]);
err(~isfinite(err))=inf;
nidx=(err<th)&(s2_==s2)&(s1_==s1);


function [nidx,err]=get_similarity_inliers(pt1,pt2,H,th)

pt2_=H*pt1;
tmp2_=pt2_(1:2,:)./repmat(pt2_(3,:),[2 1])-pt2(1:2,:);
err2=sum(tmp2_.^2);

pt1_=H\pt2;
tmp1_=pt1_(1:2,:)./repmat(pt1_(3,:),[2 1])-pt1(1:2,:);
err1=sum(tmp1_.^2);

err=max([err1; err2]);
err(~isfinite(err))=inf;
nidx=(err<th);


function r=steps(pps,inl,p)

e=1-inl;
r=log(1-p)/log(1-(1-e)^pps);


function m=rpt(pt1,pt2,sz1,sz2,idx,rd)

disk=strel('disk',rd+1,8);
disk=disk.Neighborhood;

c1=zeros(sz1,'logical');
c2=zeros(sz2,'logical');

idx=find(idx);
for i=1:length(idx)
    pt1_=round(pt1(idx(i),:));
    y_tmp=[pt1_(2)-rd max(1,pt1_(2)-rd) min(sz1(1),pt1_(2)+rd) pt1_(2)+rd];
    x_tmp=[pt1_(1)-rd max(1,pt1_(1)-rd) min(sz1(2),pt1_(1)+rd) pt1_(1)+rd];
    c1(y_tmp(2):y_tmp(3),x_tmp(2):x_tmp(3))=c1(y_tmp(2):y_tmp(3),x_tmp(2):x_tmp(3))+disk(y_tmp(2)-y_tmp(1)+1:hs-(y_tmp(4)-y_tmp(3)),x_tmp(2)-x_tmp(1)+1:hs-(x_tmp(4)-x_tmp(3)));

    pt2_=round(pt2(idx(i),:));
    y_tmp=[pt2_(2)-rd max(1,pt2_(2)-rd) min(sz2(1),pt2_(2)+rd) pt2_(2)+rd];
    x_tmp=[pt2_(1)-rd max(1,pt2_(1)-rd) min(sz2(2),pt2_(1)+rd) pt2_(1)+rd];    
    c2(y_tmp(2):y_tmp(3),x_tmp(2):x_tmp(3))=c2(y_tmp(2):y_tmp(3),x_tmp(2):x_tmp(3))+disk(y_tmp(2)-y_tmp(1)+1:hs-(y_tmp(4)-y_tmp(3)),x_tmp(2)-x_tmp(1)+1:hs-(x_tmp(4)-x_tmp(3)));
end

m1=sum(c1,'all')/numel(c1);
m2=sum(c2,'all')/numel(c2);
m=[min(m1,m2) m1 m2];


function [matches_new,ttime]=kpt_improver(im1,im2,matches,what,wr,s,ref,hom,hom_data,interp_max)

% parpool setup
nthreads=10;
p=gcp('nocreate');
if isempty(p)
    parpool(nthreads);
elseif p.NumWorkers~=nthreads
    p.delete;
    parpool(nthreads);    
end

if hom~=0
    didx=hom_data.didx;
    Hdata=hom_data.Hdata;
else
    didx=ones(1,size(matches,1));
    Hdata={eye(3),eye(3)};
end

img1=double(rgb2gray(im1));
img2=double(rgb2gray(im2));

addpath(genpath('FastMatch'))    
if (exist('Configs2Affine_mex','file')~=3) ||...
    (exist('CreateList_mex','file')~=3) ||...
    (exist('CreateList_mex','file')~=3)
% Compile Mex file
    cd('FastMatch');
    CompileMex;
    cd('..');
end

matches_new=matches;
method={...
    'lsm',        @lsm; ...
    'norm_corr',  @ncorr; ...
    'fast_match', @fmatch; ...
    };

stime=tic;

T=repmat({eye(3)},[size(matches,1) 1]);
for k=1:length(what)    
    widx=find(strcmp(method(:,1),what{k}));
    wr_=wr(k);

%   for i=1:size(matches,1)    
    parfor i=1:size(matches,1)
        aux_match=matches(i,:);
        tmp_match=matches_new(i,:);
        T_=T{i};
        
        if (~ref)||(ref==1)
            [p2_new,p2_status,p2_err,p2_err_base,T2]=method{widx,2}(img1,img2,aux_match([1 2]),aux_match([3 4]),wr_,Hdata(didx(i),1:2),s(k),T_,interp_max);
        end
    
        if (~ref)||(ref==2)
            [p1_new,p1_status,p1_err,p1_err_base,T1]=method{widx,2}(img2,img1,aux_match([3 4]),aux_match([1 2]),wr_,Hdata(didx(i),2:-1:1),s(k),T_,interp_max);
        end
    
        switch ref
            case 0
                if (~p2_status)&&(p2_err<p1_err)
                    if p2_err<p2_err_base
                        tmp_match([3 4])=p2_new;
                        T_=T2\T_;
                    end
                end
                if (~p1_status)&&(p1_err<p2_err)
                    if p1_err<p1_err_base
                        tmp_match([1 2])=p1_new;
                        T_=T1*T_;
                    end        
                end
            case 1
                if (~p2_status)
                    if p2_err<p2_err_base
                        tmp_match([3 4])=p2_new;
                        T_=T2\T_;
                    end
                end
            case 2
                if (~p1_status)
                    if p1_err<p1_err_base
                        tmp_match([1 2])=p1_new;
                        T_=T1*T_;
                    end        
                end     
        end
        matches_new(i,:)=tmp_match;
        T{i}=T_;
    end
    matches=matches_new;
end

ttime=toc(stime);

function [p2_new,p2_status,p2_err,p2_err_base,T]=lsm(im1,im2,p1,p2,wr,Hs,s,T_,dummy)

max_iter=500;                     % max iteration of lsm gradient descend
alpha=0.5;                        % gradient step
px_err=1;                         % stop condition a
tm_err=10;                        % stop condition b
what=3;                           % 0 no radiometric, 1 offset
                                  % 2 scalar, 3 linear

err_count=zeros(1,tm_err+1)+inf;                                  
v=[1 0 0 0 1 0 1 0];
p2_new_old=p2;
p2_new=p2;
p2_err_old=inf;
p2_err=inf;
p2_err_base=inf;

T=[v(1:3); v(4:6); 0 0 1];
r=v(7:8);

sH=[s 0 0; 0 s 0; 0 0 1];
Hs{1}=sH*Hs{1};
Hs{2}=sH*T_*Hs{2};

[tmp_im1,p2_status]=apply_H(p1,Hs{1},eye(3),wr,im1);
if p2_status==1
    return;
end

p2_status=4;
for i=1:max_iter
    [im2_,p2_status]=apply_H(p2,Hs{2},T,wr+1,im2);
    if p2_status==1
        p2_new=p2_new_old;
        p2_err=p2_err_old;
        p2_status=2;
        break;
    end

    Hp=Hs{2}*[p2 1]';
    Hp=Hp(1:2)/Hp(3);
    
    x=Hp(1)-wr-1:Hp(1)+wr+1;
    y=Hp(2)-wr-1:Hp(2)+wr+1;
    aux1=repmat(x,[length(y) 1]);
    aux2=repmat(y',[1 length(x)]);

    x2=aux1;
    y2=aux2;

    tmp_gx=reshape(im2_(2:end-1,3:end)-(im2_(2:end-1,1:end-2))/2,[],1);
    tmp_gy=reshape(im2_(3:end,2:end-1)-(im2_(1:end-2,2:end-1))/2,[],1);
    tmp_im2=reshape(im2_(2:end-1,2:end-1),[],1);
    tmp_x2=reshape(x2(2:end-1,2:end-1),[],1);
    tmp_y2=reshape(y2(2:end-1,2:end-1),[],1);
    tmp_ones=ones(size(tmp_im2,1),1);

    switch what
        case 0
            tmp_more=[];
        case 1
            tmp_more=tmp_ones;
        case 2
            tmp_more=tmp_im2;
        case 3
            tmp_more=[tmp_im2 tmp_ones];
    end

    b=tmp_im1(:)-r(1)*tmp_im2-r(2);
    A=[r(1)*tmp_gx.*tmp_x2 r(1)*tmp_gx.*tmp_y2 r(1)*tmp_gx r(1)*tmp_gy.*tmp_x2 r(1)*tmp_gy.*tmp_y2 r(1)*tmp_gy tmp_more];
    v=(pinv(A)*b)';

    if i==1
        p2_err_base=mean(abs(b));
    end
        
    curr_err=mean(abs(b-A*v'));
    err_count=[err_count(2:end) curr_err];

    if all(abs(err_count-err_count(end))<px_err)
%       disp([i err_count(end)]);
        p2_status=0;
        break;
    end
    % close all;

    T_old=T;
    T(1,:)=T(1,:)+alpha*v(1:3);
    T(2,:)=T(2,:)+alpha*v(4:6);
    switch what
        case 1
            r(2)=r(2)+alpha*v(end);
        case 2
            r(1)=r(1)+alpha*v(end);
        case 3
            r=r+alpha*v(7:end);
    end

    if i==1
        p2_err_old=p2_err_base;
    else
        p2_err_old=p2_err;
    end
    p2_err=curr_err;
    p2_new_old=p2_new;
    p2_new=(Hs{2}\T*Hs{2}*[p2 1]')';
    p2_new=p2_new(1:2)/p2_new(3);

    if norm(p2_new-p2)>wr/s
        p2_new=p2_new_old;
        p2_err=p2_err_old;
        p2_status=3;
        T=T_old;
        break;
    end
end


function [p2_new,p2_status,p2_err,p2_err_base,T]=ncorr(im1,im2,p1,p2,wr,Hs,s,T_,interp_max)

% wr=15; % window radius
p2_new=p2;
p2_err=inf;
p2_err_base=inf;
T=eye(3);

sH=[s 0 0; 0 s 0; 0 0 1];
Hs{1}=sH*Hs{1};
Hs{2}=sH*T_*Hs{2};

[tmp1,p2_status]=apply_H(p1,Hs{1},eye(3),wr,im1);
if p2_status==1
    return;
end

[tmp2,p2_status]=apply_H(p2,Hs{2},eye(3),2*wr,im2);
if p2_status==1
    return;
end

m=normxcorr2(tmp1,tmp2);
m=m(2*wr+1:end-2*wr,2*wr+1:end-2*wr);
p2_err=max(m,[],'all');

[i,j]=find(m==p2_err);
i=i(1);
j=j(1);

switch interp_max
    case 0 % no max interpolation
        i_=i;
        j_=j;
    case 1 % parabolic max interpolation
        i_=i;
        j_=i;
        if (i>1) && (j>1) && (i<size(m,1)) &&  (j<size(m,2))
            v=m(i,j);
            vl=m(i-1,j);
            vr=m(i+1,j);
            i_=i+(-vl+vr)./(2*(-vl-vr+2*v));
            
            v=m(i,j);
            vl=m(i,j-1);
            vr=m(i,j+1);
            j_=j+(-vl+vr)./(2*(-vl-vr+2*v));
        end
    case 2 % Lowe's max interpolation
        i_=i;
        j_=j;
        for ii=1:5
            if (i_>1) && (j_>1) && (i_<size(m,1)) &&  (j_<size(m,2))               
                Dx=0.5*(m(i_,j_+1)-m(i_,j_-1));
                Dy=0.5*(m(i_+1,j_)-m(i_-1,j_));

                Dxx=m(i_,j_+1)+m(i_,j_-1)-2.0*m(i_,j_);                
                Dyy=m(i_+1,j_)+m(i_-1,j_)-2.0*m(i_,j_);                
                Dxy=0.25*(m(i_+1,j_+1)+m(i_-1,i_-1)-m(i_-1,j_+1)-m(i_+1,j_-1));

                A=[ ...
                    Dxx Dxy; ...
                    Dxy Dyy; ...
                    ];

                B=-[ ...
                    Dx; ...
                    Dy; ...
                    ];

                off=A\B;

                if any(abs(off)<0.5)
                    i_=i_+off(2);
                    j_=j_+off(1);
                    break;
                end
                
                i_=i_+round(off(2));
                j_=j_+round(off(1));
            end
        end
end

if (i_>0.5) && (j_>0.5) && (i_<=size(m,1)+0.5) &&  (j_<=size(m,2)+0.5)
    i=i_;
    j=j_;
end

i=i-wr-1;
j=j-wr-1;

Hp2=Hs{2}*[p2 1]';
Hp2=Hp2(1:2)'/Hp2(3); 

T=[1 0 j; 0 1 i; 0 0 1];

p2_new=Hs{2}\[(Hp2+[j i]) 1]';
p2_new=p2_new(1:2)'/p2_new(3);
p2_status=0;

p2_err=-abs(p2_err);


function [p1_new,p1_status,p1_err,p1_err_base,T]=fmatch(im2,im1,p2,p1,wr,Hs,s,T_,dummy)

Hs=Hs(2:-1:1);

sH=[s 0 0; 0 s 0; 0 0 1];
Hs{1}=sH*T_*Hs{1};
Hs{2}=sH*Hs{2};

p1_new=p2;
p1_err=inf;
p1_err_base=inf;
T=eye(3);

[tmp1,p1_status]=apply_H(p1,Hs{1},eye(3),2*wr,im1);
if p1_status==1
    return;
end

[tmp2,p1_status]=apply_H(p2,Hs{2},eye(3),wr,im2);
if p1_status==1
    return;
end

tmp1=double(tmp1)/255;
tmp2=double(tmp2)/255;
    
sr.minScale = 0.33;
sr.maxScale = 3;
sr.minRotation = -pi/3;
sr.maxRotation = pi/3;
sr.minTx = -wr;
sr.maxTx = wr;
sr.minTy = -wr;
sr.maxTy = wr;

% FastMatch run
[~,bestTransMat,sampledError] = FastMatch(tmp2,tmp1,[],0.5,0.75,1,sr);
% [optError,fullError,overlapError,iii1,iii2]=MatchingResult(tmp2,tmp1,bestTransMat,[],'');
% pause;

% p1_err=sampledError;
p1_status=0;
% bestTransMat=inv(bestTransMat);

Hp1=Hs{1}*[p1 1]';
Hp1=Hp1(1:2)'/Hp1(3); 

Hp2=Hs{2}*[p2 1]';
Hp2=Hp2(1:2)'/Hp2(3); 

fullH=Hs{2}\([1 0 Hp2(1); 0 1 Hp2(2); 0 0 1]*(bestTransMat\[1 0 -Hp1(1); 0 1 -Hp1(2); 0 0 1]*Hs{1}));
Hf2=fullH\[p2 1]';
Hf2=Hf2(1:2)'/Hf2(3); 

[tmp1_,p1_status]=apply_H(Hf2,fullH,eye(3),wr,im1);
if p1_status==1
    return;
end

[tmp2_,p1_status]=apply_H(p2,Hs{2},eye(3),wr,im2);
if p1_status==1
    return;
end

p1_err=mean(abs(tmp1_(:)-tmp2_(:)));

T=[1 0 Hp1(1); 0 1 Hp1(2); 0 0 1]*bestTransMat*[1 0 -Hp1(1); 0 1 -Hp1(2); 0 0 1];

p1_new=Hs{1}\[bestTransMat(1:2,3)'+Hp1 1]';
p1_new=p1_new(1:2)'/p1_new(3);

if norm(p1_new-p1)>wr/s
    p1_new=p1;
    p1_err=inf;
    p1_status=3;
end


function [im_,p_status]=apply_H(p,H,HH,wr,im)

p_status=0;

Hp=H*[p 1]';
Hp=Hp(1:2)/Hp(3);

x=Hp(1)-wr:Hp(1)+wr;
y=Hp(2)-wr:Hp(2)+wr;
aux1=repmat(x,[length(y) 1]);
aux2=repmat(y',[1 length(x)]);
sx=length(x);
sy=length(y);

aux=H\HH*[aux1(:) aux2(:) ones(numel(aux1),1)]';
x_=aux(1,:)'./aux(3,:)';
y_=aux(2,:)'./aux(3,:)';
xf=floor(x_);
yf=floor(y_);
xc=xf+1;
yc=yf+1;

if any(x_(:)<1)||any(x_(:)>size(im,2))...
    ||any(y_(:)<1)||any(y_(:)>size(im,1))
    im_=[];
    p_status=1;
    return;
end

im_=reshape(...
    im(sub2ind(size(im),yf,xf)).*(xc-x_).*(yc-y_)+...
    im(sub2ind(size(im),yc,xc)).*(x_-xf).*(y_-yf)+...            
    im(sub2ind(size(im),yf,xc)).*(x_-xf).*(yc-y_)+...                        
    im(sub2ind(size(im),yc,xf)).*(xc-x_).*(y_-yf)+...
    +0,[sy sx]);
