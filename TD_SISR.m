clear all
% close all
d3 = true;
% whether the convolution has a circular boundary condition
crc_conv = true;
% scale of downsampling, integer number
scale = 2;
%for Gaussian PSF - [8.2,7.5,1.3] values are estimated from the data,
%[8,8,8] values were used for the simulation
sigma_x = 8.2/sqrt(2);%8/sqrt(2);%
sigma_y = 7.5/sqrt(2);%8/sqrt(2);%
sigma_z = 1.3/sqrt(2);%8/sqrt(2);%
% is it a simulation?
simulation = false;
% regularization for inverse of P
regularization = 1e0;%0;
% ration of added noise
m_snr = 1.25%1;%;2;%1.25;%0;%3;
% data paths
gt_folder = 'C:\Users\Janka\Dropbox\phd\tensor_code/gt/';
tr_folder = 'C:\Users\Janka\Dropbox\phd\tensor_code/train/';


[LR,HR] = read_imagevolumes(gt_folder,tr_folder,'11b_',scale);
[mg, ng, og] = size(HR);
[mt, nt, ot] = size(LR);

P = degradation_matrices(mg, ng, og, mt, nt, ot, scale, sigma_x, sigma_y, sigma_z, crc_conv, regularization);

if simulation
    LR = tmprod(HR,P,1:3);
end
if m_snr~=0
    LR_clear = LR;
    noise_norm = norm(LR(:))/(10^(m_snr));
    noise = randn(size(LR));
    noise = noise/norm(noise(:))*noise_norm;
    LR = LR_clear + noise;
end

Y = LR;
Pi = {pinv(P{1}),pinv(P{2}),pinv(P{3})};

% thresholds for the SVD truncation
th = [50 50 50];

[U,Sigma,SVn]=mlsvd(Y);
figure(1)
clf
for n = 1:3
    subplot(1,3,n);
    semilogy(SVn{n},'.-');
    xlim([1 length(SVn{n})])
    ylim([1e-2 1e4])
    hold on
    plot([th(n) th(n)],[1e-2 1e4],'r-.')
end

Utrunc{1} = U{1}(:,1:th(1));
Utrunc{2} = U{2}(:,1:th(2));
Utrunc{3} = U{3}(:,1:th(3));
Strunc = Sigma(1:th(1), 1:th(2), 1:th(3));

% denoised volume
Y = lmlragen(Utrunc,Strunc);
% deconvolution
mlsvd_Y = tmprod(Y,Pi,1:3);

% Plotting
slice_x = 160;
slice_y = 140;
slice_z = 300;

T  = mat2gray(mlsvd_Y);
figure(3)
s(1,1) = subplot(3,3,1);
imagesc(squeeze(HR(:,:,slice_z))); axis off; axis image; colormap gray
s(1,2) = subplot(3,3,2);
imagesc(1:ng,1:mg,squeeze(LR(:,:,round(slice_z/scale)))); axis off; axis image; colormap gray
s(1,3) = subplot(3,3,3);
imagesc((squeeze(T(:,:,slice_z)))); axis off; axis image; colormap gray
s(2,1) = subplot(3,3,4);
imagesc(squeeze(HR(:,slice_y,:))); axis off; axis image; colormap gray
s(2,2) = subplot(3,3,5);
imagesc(1:og,1:mg,squeeze(LR(:,round(slice_y/scale),:))); axis off; axis image; colormap gray
s(2,3) = subplot(3,3,6);
imagesc((squeeze(T(:,slice_y,:)))); axis off; axis image; colormap gray
s(3,1) = subplot(3,3,7);
imagesc(squeeze(HR(slice_x,:,:))); axis off; axis image; colormap gray
s(3,2) = subplot(3,3,8);
imagesc(1:og,1:ng,squeeze(LR(round(slice_x/scale),:,:))); axis off; axis image; colormap gray
s(3,3) = subplot(3,3,9);
imagesc((squeeze(T(slice_x,:,:)))); axis off; axis image; colormap gray
linkaxes(s(1,:),'xy');
linkaxes(s(2,:),'xy');
linkaxes(s(3,:),'xy');


function X = circularing(x,kg,kt,crc_conv,regul)
% x is vertical array
% k size os the image (in given dimension)
kp = length(x);
x = padarray(x,floor((kg-kp)/2),'pre');
x = padarray(x,ceil((kg-kp)/2),'post');
x = x';
x = circshift(x,ceil(-(kg)/2+1));
% normalize
x = (x/sum(abs(x)));
% initialize
X = zeros(kt,kg);
for I = 0 : kt-1
    % circular boundary condition
    shift = I*round(kg/kt);
    X(I+1,:) = circshift(x,shift);
    if ~crc_conv
        % zero-padded boundary condition
        zer = [];
        if shift < (kt+1)/2
            zer = [kg - floor((kp+1)/2) + shift : kg];
        elseif I > (kt+1)/2
            zer = [1 :ceil((kp+1)/2) - kg + shift];
        end
        X(I+1,zer) = 0;
    end
end
end

function [tr,gt] = read_imagevolumes(f_gt,f_tr,im_name,scale)
num_imgs = numel(dir([f_gt,im_name,'*.png']));


% read the image volumes
for I = 1:num_imgs
    temp = ((double(imread(strcat(f_gt,im_name,...
        num2str(I+45,'%03u'),'.png')))));
    gt(:,:,I) = imresize(temp,2*(floor(size(temp)/2)));
    tr(:,:,I) = ((double(imread(strcat(f_tr,im_name,...
        num2str(I+45,'%03u'),'.png')))));
end
[Xq, Yq, Zq] = meshgrid(...
    linspace(1,size(gt,2),size(gt,2)/scale),...
    linspace(1,size(gt,1),size(gt,1)/scale),...
    linspace(1,size(gt,3),size(gt,3)/scale));
tr = interp3(tr,Xq,Yq,Zq);

tr = mat2gray(tr);
gt = mat2gray(gt);

% mask for calculating the metrics
gt_mask = gt > 0.15;
% eliminating the original background noise
gt = shrinkage(gt,0.15);
gt = rescale(gt,0,1);
end

function P=degradation_matrices(mg, ng, og, mt, nt, ot, scale, sigma_x, sigma_y, sigma_z, crc_conv, regularization)
% downsampling matricies for the three modes
D1 = circularing(ones(scale,1)/scale,mg,mt,crc_conv,0);
D2 = circularing(ones(scale,1)/scale,ng,nt,crc_conv,0);
D3 = circularing(ones(scale,1)/scale,og,ot,crc_conv,0);

% PSF
% pixel-length of the PSF
k = 25;
        %  gauss
        gauss1 = @(x,para) exp(-(x-para(1)).^2/(2*para(2)^2));
        psf = cpdgen({gauss1(-k:k,[0,sigma_x])',...
            gauss1(-k:k,[0,sigma_y])',...
            gauss1(-k:k,[0,sigma_z])'});

% PSF in x-dir
%this line is used to have universal solution for all PSFs.
%i have checked, it does not influence the distribution, p1 will be the
%same as gauss(-25:25,sigma_x)
h1 = mat2gray(reshape(mean(mean(psf,2),3),[2*k+1,1,1]));
H1 = circularing(h1,mg,mg,crc_conv,0);
P1 = D1*H1;

% PSF in y-dir
h2 = mat2gray(reshape(mean(mean(psf,1),3),[2*k+1,1,1]));
H2 = circularing(h2,ng,ng,crc_conv,0);
P2 = D2*H2;

% PSF in z-dir
h3 = mat2gray(reshape(mean(mean(psf,1),2),[2*k+1,1,1]));
H3 = circularing(h3,og,og,crc_conv,0);
P3 = D3*H3;

% regularizing the ill-posed matrix
P1 = P1 + D1*regularization;
P2 = P2 + D2*regularization;
P3 = P3 + D3*regularization;

P = {P1,P2,P3};
end

function X = shrinkage(X, mu)
X = sign(X).*max(abs(X) - mu, 0);
end