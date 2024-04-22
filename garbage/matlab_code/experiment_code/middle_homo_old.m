function [didx,Hdata]=middle_homo(im1,im2,matches,th,th_out)

[p,f1,~]=fileparts(im1);
[~,f2,~]=fileparts(im2);

% matches=load([p filesep f1 '_' f2 '.mat']);
% matches=matches.matches;
% % matches=matches.table;
% % mask=any(~isfinite(matches),2);
% % matches=matches(~mask,:);

pt1=matches(:,[1 2]);
pt2=matches(:,[3 4]);

% th=15;
% th_out=7;

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

for i=1:size(Hdata,1)
    figure;
    imshow(im1);
    hold on;
    plot(pt1(midx(i,:),1),pt1(midx(i,:),2),'+r');
    pause(1);

    figure;
    imshow(im2);
    hold on;
    plot(pt2(midx(i,:),1),pt2(midx(i,:),2),'+r');    
    pause;

    close all;
end

% for i=1:size(Hdata,1)
%     im1_=show_transform(im1,Hdata{i,1},im2,Hdata{i,2});
%     im2_=show_transform(im2,Hdata{i,2},im1,Hdata{i,1});
%     imwrite(im1_,['im_' num2str(i) '_a.png']);
%     imwrite(im2_,['im_' num2str(i) '_b.png']);
% end


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

    if (sum(nidx)<=4) %||(sum(nidx)-sum(tidx(nidx))<=4)
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

% set the similarity matrix
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

function im_=show_transform(im,H,im_aux,H_aux)

sz=size(im);
if length(sz)==2
    im=reshape(im,[sz 1]);
    sz=[sz 1];
end

b=[...
        0     0 1;...
    sz(2)     0 1;...
       0  sz(1) 1;...
    sz(2) sz(1) 1;...
    ]';

b_=H*b;
b_=b_';
b_=b_./repmat(b_(:,3),[1 3]);


sz_=size(im_aux);
if length(sz_)==2
    im=reshape(im,[sz_ 1]);
    sz_=[sz_ 1];
end

b_aux=[...
        0     0 1;...
    sz_(2)     0 1;...
       0  sz_(1) 1;...
    sz_(2) sz_(1) 1;...
    ]';

b_aux_=H_aux*b_aux;
b_aux_=b_aux_';
b_aux_=b_aux_./repmat(b_aux_(:,3),[1 3]);


if any(~isfinite(b_aux_(:)))
    disp('warning!');
end

b_=[b_; b_aux_];

c=[floor(min(b_)); ceil(max(b_))];
sz_=[c(2,2)-c(1,2) c(2,1)-c(1,1)];
if max(sz_)>2048
    s_=1024/max(sz_);
    disp('reduction');
else
    s_=1;
end
sz_=round(sz_*s_);
sz_=[sz_ sz(3)];

T=[s_ 0  0;...
   0  s_ 0;...
   0  0  1] * ...
  [1 0 -c(1,1);...
   0 1 -c(1,2);...
   0 0       1];

H_=T*H;
H_inv=inv(H_);

im_=zeros(sz_);
for x_=1:sz_(2)
    for y_=1:sz_(1)
        pt=H_inv*[x_ y_ 1]';
        x=pt(1)/pt(3);
        y=pt(2)/pt(3);

        xf=floor(x);
        yf=floor(y);
        xc=xf+1;
        yc=yf+1;
        
        if (xf<1)||(xc>sz(2))||(yf<1)||(yc>sz(1))
            continue;
        end
        
        for c=1:sz_(3)            
            im_(y_,x_,c)=...
                im(yf,xf,c)*(xc-x)*(yc-y)+...
                im(yc,xc,c)*(x-xf)*(y-yf)+...            
                im(yf,xc,c)*(x-xf)*(yc-y)+...                        
                im(yc,xf,c)*(xc-x)*(y-yf)+...
                0;
        end
    end
end

im_=squeeze(uint8(im_));
