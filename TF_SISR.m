clear all
% dimension: true: 3D tensors, false: 2D tensors
d3 = true;
% whether the convolution has a circular boundary condition
crc_conv = false;

% scale of downsampling, integer number
scale = 2;

% 1: tooth (2D-3D), 2: dots(2D), 3: lena (2D)
image_type = 1;

% 1: gaussian, 2: from this data, 3: from dataset
psf_type = 1;

%for Gaussian PSF - these values are estimated from the data
sigma_x = 8.2/sqrt(2);
b_x = 0;
sigma_y = 7.5/sqrt(2);
b_y = 0;
sigma_z = 1.3/sqrt(2);
b_z = 0;

% is it a simulation?
simulation = false;

regularization_x = 0;%1e-0;%0;
regularization_y = 0;%1e-0;%0;
regularization_z = 0;%1e-0;%0;
eps = 0.1;

numIt = 10;

% ration of added noise
m_snr = 0;

% rank number (in 3d teeth?~500, for 2d images ~50
F = 500;

addpath(genpath('tensorlab_2016-03-28'));
% for 2D tooth image
gt_2D_tooth_path = 'gt/11b_142.png';
tr_2D_tooth_path = 'train/11b_142.png';
% foor 2D lena
lena_path = 'lena.tiff';
% for 2D PSF
PSF_2D_path = 'with_bg_psf.mat';
% for 3D
gt_folder = 'gt/';
tr_folder = 'train/';
% for known 2D PSF
PSF_3D_path = 'PSF3D.mat';
% for 2D images
if ~d3
    
    switch image_type
        case 1
            % % image slices
            gt = mat2gray(double(imread(gt_2D_tooth_path)));
            tr = imresize(mat2gray(double(imread(tr_2D_tooth_path))),1/scale);
            tr = mat2gray(tr);
        case 2
            % % dots
            gt = zeros(81,81);
            gt(randi(81*81,1,10)) = 1;
            tr = zeros(round(size(gt)/scale));
        case 3
            % % lena
            gt = imresize(mat2gray(rgb2gray(imread(lena_path))),[511,511]);
            tr = zeros(round(size(gt)/scale));
    end
    [mg, ng] = size(gt);
    [mt, nt] = size(tr);
    
    % downsampling operator
    D1 = circularing(ones(scale,1)/scale,mg,mt,crc_conv,0);
    D2 = circularing(ones(scale,1)/scale,ng,nt,crc_conv,0);
    
    % % PSF
    switch psf_type
        case 1
            %  gauss
            gauss1 = @(x,sigma) exp(-(x-0).^2/(2*sigma^2));
            psf = gauss1(-25:25,sigma_x)'*gauss1(-25:25,sigma_y);
        case 2
            % from data
            Fpsf = fftshift((fft2(tr,size(gt,1),size(gt,2))./(fft2(gt) + 1e-5)));
            hFpsf = fftshift(Fpsf.*(hann(mg)*hann(ng)'));
            psf = abs(fftshift(ifft2(hFpsf)));
            k = 25;
            psf = psf((mg-1)/2-k:(mg-1)/2+k,(ng-1)/2-k:(ng-1)/2+k).*...
                (hann(2*k+1)*hann(2*k+1)');
        case 3
            % average data
            load(PSF_2D_path)
            psf = meansmall_PSF;
    end
    [mp, np] = size(psf);
    
    % PSF in x-dir
    %this line is used to have universal solution for all PSFs.
    %i have checked, it does not influence the distribution, p1 will be the
    %same as gauss(-25:25,sigma_x)
    p1 = mat2gray(mean(psf,1))';
    P1 = circularing(p1,mg,mg,crc_conv,0);
    P1 = D1*P1;
    
    % PSF in y-dir
    p2 = mat2gray(mean(psf,2));
    P2 = circularing(p2,ng,ng,crc_conv,0);
    P2 = D2*P2;
    
    % regularizing the ill-posed matrix
    P1 = P1 + D1*regularization_x;
    P2 = P2 + D2*regularization_y;
    
    
    % simulate CBCT
    if simulation
        tr = P1*gt*P2';
    end
    
    % adding the noise
    if m_snr~=0
        tr_clear = tr;
        noise_norm = norm(tr(:))/(10^(m_snr/20));
        noise = randn(size(tr));
        noise = noise/norm(noise(:))*noise_norm;
        tr = tr + noise;
    end
    
    % initialize factorization
    U = cpd_rnd(size(gt),F);
    
    for I = 1:numIt
        U{1} = (pinv(P1)*tr)*pinv(U{2}'*P2');
        U{2} = (pinv(P2)*tr')*pinv(U{1}'*P1');
    end
    T = cpdgen(U);
    
    figure; colormap gray
    s(1) = subplot(1,3,1);
    imagesc(gt,[0,1]); colorbar; axis off; axis image
    title('ground truth')
    s(2) = subplot(1,3,2);
    imagesc(1:ng,1:mg,tr,[min(tr(:)),max(tr(:))]); colorbar; axis off; axis image
    title('corrupted')
    s(3) = subplot(1,3,3);
    imagesc(T,[min(T(:)) max(T(:))]); colorbar; axis off; axis image
    title('recovered')
    linkaxes(s,'xy')
    %%
else
% 3D-case   
    switch image_type
        case 1
            names = {'11b_', '14_', '17_', '18_', '23_', '27_', '34_', '35_', '36_',...
                '37_', '41_', '43_', '48_'};
            numbers = [ 46, 437; 0, 473; 0, 441; 30, 475; 41, 454; 17, 478; 0, 491;...
                0, 459; 33, 448; 48, 479; 95, 478; 15, 470; 36,  437];
            imnum = 1;
            % read the image volumes
            for I = numbers(imnum,1):numbers(imnum,2)
                temp = ((double(imread(strcat(gt_folder,names{imnum},...
                    num2str(I,'%03u'),'.png')))));
                gt(:,:,I-numbers(imnum,1)+1) = imresize(temp,2*(floor(size(temp)/2)));
                tr(:,:,I-numbers(imnum,1)+1) = ((double(imread(strcat(tr_folder,names{imnum},...
                    num2str(I,'%03u'),'.png')))));
            end
            [Xq, Yq, Zq] = meshgrid(...
                linspace(1,size(gt,2),size(gt,2)/scale),...
                linspace(1,size(gt,1),size(gt,1)/scale),...
                linspace(1,size(gt,3),size(gt,3)/scale));
            tr = interp3(tr,Xq,Yq,Zq);
        otherwise
            error('This image option is not implemented for 3D')
    end
    
    tr = mat2gray(tr);
    gt = mat2gray(gt);
    [mg, ng, og] = size(gt);
    [mt, nt, ot] = size(tr);
       
    % downsampling
    D1 = circularing(ones(scale,1)/scale,mg,mt,crc_conv,0);
    D2 = circularing(ones(scale,1)/scale,ng,nt,crc_conv,0);
    D3 = circularing(ones(scale,1)/scale,og,ot,crc_conv,0);
    
    % PSF
    k = 25;
    switch psf_type
        case 1
            %  gauss
            gauss1 = @(x,para) exp(-(x-para(1)).^2/(2*para(2)^2));
            psf = cpdgen({gauss1(-k:k,[b_x,sigma_x])',...
                gauss1(-k:k,[b_y,sigma_y])',...
                gauss1(-k:k,[b_z,sigma_z])'});
        case 2
            Fpsf = fftshift((fftn(tr,size(gt))./(fftn(gt) + 1e-5)));
            hFpsf = fftshift(Fpsf.*(cpdgen({hann(mg),hann(ng),hann(og)})));
            psf = abs(fftshift(ifftn(hFpsf)));
            psf = psf((mg-1)/2-k:(mg-1)/2+k,(ng-1)/2-k:(ng-1)/2+k,(og-1)/2-k:(og-1)/2+k).*...
                cpdgen({hann(2*k+1),hann(2*k+1),hann(2*k+1)});            
        otherwise
            load(PSF_3D_path);
            psf = PSF; clear PSF;
    end
    [mp np op] = size(psf);    
    
    % PSF in x-dir
    %this line is used to have universal solution for all PSFs.
    %i have checked, it does not influence the distribution, p1 will be the
    %same as gauss(-25:25,sigma_x)
    p1 = mat2gray(reshape(mean(mean(psf,2),3),[2*k+1,1,1]));
    P1 = circularing(p1,mg,mg,crc_conv,0);
    P1 = D1*P1;
    
    % PSF in y-dir
    p2 = mat2gray(reshape(mean(mean(psf,1),3),[2*k+1,1,1]));
    P2 = circularing(p2,ng,ng,crc_conv,0);
    P2 = D2*P2;
    
    % PSF in z-dir
    p3 = mat2gray(reshape(mean(mean(psf,1),2),[2*k+1,1,1]));
    P3 = circularing(p3,og,og,crc_conv,0);
    P3 = D3*P3;
    
    % regularizing the ill-posed matrix
    P1 = P1 + D1*regularization_x;
    P2 = P2 + D2*regularization_y;
    P3 = P3 + D3*regularization_z;
    
    if simulation
        tr = tmprod(gt,{P1,P2,P3},1:3);
    end
    
    tic
    U = cpd_rnd(size(gt),F);
    
    Y1 = tens2mat(tr,1);
    Y2 = tens2mat(tr,2);
    Y3 = tens2mat(tr,3);
    pP1 = tpinv(P1,eps);
    pP2 = tpinv(P2,eps);
    pP3 = tpinv(P3,eps);
    
    for I = 1:numIt
        I
        PCPB = kr(P3*U{3},P2*U{2});
        U{1} = pP1*Y1*tpinv(PCPB',eps);
        PCPA = kr(P3*U{3},P1*U{1});
        U{2} = pP2*Y2*tpinv(PCPA',eps);
        PBPA = kr(P2*U{2},P1*U{1});
        U{3} = pP3*Y3*tpinv(PBPA',eps);
    end
    
    T = cpdgen(U);
    toc
    slice_x = 150;
    slice_y = 140;
    slice_z = 300;
    
    T  = mat2gray(T);
    
    s(1,1) = subplot(3,3,1);
    imagesc(squeeze(gt(:,:,slice_z))); axis off; axis image; colormap gray
    s(1,2) = subplot(3,3,2);
    imagesc(1:ng,1:mg,squeeze(tr(:,:,round(slice_z/scale)))); axis off; axis image; colormap gray
    s(1,3) = subplot(3,3,3);
    imagesc((squeeze(T(:,:,slice_z)))); axis off; axis image; colormap gray
    s(2,1) = subplot(3,3,4);
    imagesc(squeeze(gt(:,slice_y,:))); axis off; axis image; colormap gray
    s(2,2) = subplot(3,3,5);
    imagesc(1:og,1:mg,squeeze(tr(:,round(slice_y/scale),:))); axis off; axis image; colormap gray
    s(2,3) = subplot(3,3,6);
    imagesc((squeeze(T(:,slice_y,:)))); axis off; axis image; colormap gray
    s(3,1) = subplot(3,3,7);
    imagesc(squeeze(gt(slice_x,:,:))); axis off; axis image; colormap gray
    s(3,2) = subplot(3,3,8);
    imagesc(1:og,1:ng,squeeze(tr(round(slice_x/scale),:,:))); axis off; axis image; colormap gray
    s(3,3) = subplot(3,3,9);
    imagesc((squeeze(T(slice_x,:,:)))); axis off; axis image; colormap gray
    linkaxes(s(1,:),'xy');
    linkaxes(s(2,:),'xy');
    linkaxes(s(3,:),'xy');
    
end

function B=tpinv(A,eps)

[U,S,V] = svd(A,'econ');
B = V*(S./(S.^2+eps^2))*U';

end

% 
% function X = circularing(x,kg,kt,crc_conv,regul)
% % x is vertical array
% % k size os the image (in given dimension)
% kp = length(x);
% x = padarray(x,floor((kg-kp)/2),'pre');
% x = padarray(x,ceil((kg-kp)/2),'post');
% x = x';
% x = circshift(x,ceil(-(kg)/2+1));
% % normalize
% x = (x/sum(x));
% % initialize
% X = zeros(kt,kg);
% for I = 0 : kt-1
%     % circular boundary condition
%     shift = I*round(kg/kt);
%     X(I+1,:) = circshift(x,shift);
%     if ~crc_conv
%         % zero-padded boundary condition
%         zer = [];
%         if shift < (kt+1)/2
%             zer = [kg - floor((kp+1)/2) + shift : kg];
%         elseif I > (kt+1)/2
%             zer = [1 :ceil((kp+1)/2) - kg + shift];
%         end
%         X(I+1,zer) = 0;
%     end
% end
% end

