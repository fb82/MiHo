function matches_new=kpt_improver(im1s,im2s,what,wr,s,ref,hom)

% wr = [7 15];
% s = [0.5 1];

[p,f1,~]=fileparts(im1s);
[~,f2,~]=fileparts(im2s);

matches=load([p filesep f1 '_' f2 '.mat']);
% matches=matches.matches;
matches=matches.table;
mask=any(~isfinite(matches),2);
matches=matches(~mask,:);
% matches=[size(imread(im1s),2)/2 size(imread(im1s),1)/2 size(imread(im2s),2)/2 size(imread(im2s),1)/2];

if hom
    [didx,Hdata]=middle_homo(im1s,im2s);
else
    didx=ones(1,size(matches,1));
    Hdata={eye(3),eye(3)};
%   Hdata={eye(3),[0 -1 256; 1 0 0; 0 0 1]};
end

nthreads=10;
show_res=0;

im1=imread(im1s);
im2=imread(im2s);

img1=double(rgb2gray(im1));
img2=double(rgb2gray(im2));

tic;

% % parpool setup
p=gcp('nocreate');
if isempty(p)
    parpool(nthreads);
elseif p.NumWorkers~=nthreads
    p.delete;
    parpool(nthreads);    
end

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

T=repmat({eye(3)},[size(matches,1) 1]);
for k=1:length(what)    
    widx=find(strcmp(method(:,1),what{k}));
    wr_=wr(k);

    parfor i=1:size(matches,1)
    % for i=1:size(matches,1)
        aux_match=matches(i,:);
        tmp_match=matches_new(i,:);
        T_=T{i};
        
        if (~ref)||(ref==1)
            [p2_new,p2_status,p2_err,p2_err_base,T2]=method{widx,2}(img1,img2,aux_match([1 2]),aux_match([3 4]),wr_,Hdata(didx(i),1:2),s(k),T_);
        end
    
        if (~ref)||(ref==2)
            [p1_new,p1_status,p1_err,p1_err_base,T1]=method{widx,2}(img2,img1,aux_match([3 4]),aux_match([1 2]),wr_,Hdata(didx(i),2:-1:1),s(k),T_);
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

    if show_res
        figure;
        imshow(im1,[]);
        hold on;
        plot(matches_new(:,1),matches_new(:,2),'+g');
        plot([matches(:,1) matches_new(:,1)]',[matches(:,2) matches_new(:,2)]','g');
        pause(2);
        
        figure;
        imshow(im2,[]);
        hold on;
        plot(matches_new(:,3),matches_new(:,4),'+g');
        plot([matches(:,3) matches_new(:,3)]',[matches(:,4) matches_new(:,4)]','g');
        pause(2);
    end

    matches=matches_new;
end

toc;

function [p2_new,p2_status,p2_err,p2_err_base,T]=lsm(im1,im2,p1,p2,wr,Hs,s,T_)

max_iter=500;                     % max iteration of lsm gradient descend
% wr=15;                          % window radius
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
% p2_box=repmat(p2,[5 1])+...
%     [...
%     -wr -wr;...
%     -wr +wr;...
%     +wr +wr;...
%     +wr -wr;...
%     -wr -wr;...
%     ];

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

%     figure;
%     imshow([im1 im2 imresize(reshape(tmp_im1,[sy-2 sx-2]),size(im1),'nearest') imresize(im2_,size(im1),'nearest')],[]);
%     hold on;
%     plot(p1(1),p1(2),'+r');
%     plot(p2_new(1)+size(im1,2),p2_new(2),'+r');
%     plot(p2(1)+size(im1,2),p2(2),'xg');
%     pp1=[x1(1,1) x1(1,end) x1(end,end) x1(end,1) x1(1,1); y1(1,1) y1(1,end) y1(end,end) y1(end,1) y1(1,1)];
%     plot(pp1(1,:),pp1(2,:),'-b');    
%     pp2=T*[p2_box'; 1 1 1 1 1];
%     plot(pp2(1,:)+size(im1,2),pp2(2,:),'-b');
%     pause;    

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


function [p2_new,p2_status,p2_err,p2_err_base,T]=ncorr(im1,im2,p1,p2,wr,Hs,s,T_)

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
i=i(1)-wr-1;
j=j(1)-wr-1;

Hp2=Hs{2}*[p2 1]';
Hp2=Hp2(1:2)'/Hp2(3); 

T=[1 0 j; 0 1 i; 0 0 1];

p2_new=Hs{2}\[(Hp2+[j i]) 1]';
p2_new=p2_new(1:2)'/p2_new(3);
p2_status=0;

p2_err=-abs(p2_err);


function [p1_new,p1_status,p1_err,p1_err_base,T]=fmatch(im2,im1,p2,p1,wr,Hs,s,T_)

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

tmp1_=apply_H(Hf2,fullH,eye(3),wr,im1);
tmp2_=apply_H(p2,Hs{2},eye(3),wr,im2);
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
    p_status=1;
    return;
end

im_=reshape(...
    im(sub2ind(size(im),yf,xf)).*(xc-x_).*(yc-y_)+...
    im(sub2ind(size(im),yc,xc)).*(x_-xf).*(y_-yf)+...            
    im(sub2ind(size(im),yf,xc)).*(x_-xf).*(yc-y_)+...                        
    im(sub2ind(size(im),yc,xf)).*(xc-x_).*(y_-yf)+...
    +0,[sy sx]);
