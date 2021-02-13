clear all
% whether the convolution has a circular boundary condition
crc_conv = false;

% scale of downsampling, integer number
scale = 2;

% 1: gaussian, 
psf_type = 1;

%for Gaussian PSF - these values are estimated from the data
sigma_1_orig = 8.2/sqrt(2);
b_x = 0;
sigma_2_orig = 7.5/sqrt(2);
b_y = 0;
sigma_3_orig = 7.5/sqrt(2);
b_z = 0;

% is it a simulation?
simulation = false;

reg_1 = 1e-0;%0;
reg_2 = 1e-0;%0;
reg_3 = 1e-0;%0;

numIt = 5;

% rank number
F = 500;

% folder to tensorlab
addpath(genpath(''));
% folders to image volumes
gt_folder = '';
tr_folder = '';

% 3D-volumes
names = {'11b_', '14_', '17_', '18_', '23_', '27_', '34_', '35_', '36_',...
    '37_', '41_', '43_', '48_'};
numbers = [ 46, 437; 0, 473; 0, 441; 30, 475; 41, 454; 17, 478; 0, 491;...
    0, 459; 33, 448; 48, 479; 95, 478; 15, 470; 36,  437];
% this is the uploaded image volume
imnum = 1;
% read the image volumes
for I = numbers(imnum,1):numbers(imnum,2)
    temp = ((double(imread(strcat(gt_folder,names{imnum},...
        num2str(I,'%03u'),'.png')))));
    X_gt(:,:,I-numbers(imnum,1)+1) = imresize(temp,2*(floor(size(temp)/2)));
    Y(:,:,I-numbers(imnum,1)+1) = ((double(imread(strcat(tr_folder,names{imnum},...
        num2str(I,'%03u'),'.png')))));
end
[Xq, Yq, Zq] = meshgrid(...
    linspace(1,size(X_gt,2),size(X_gt,2)/scale),...
    linspace(1,size(X_gt,1),size(X_gt,1)/scale),...
    linspace(1,size(X_gt,3),size(X_gt,3)/scale));
Y = interp3(Y,Xq,Yq,Zq);


Y = mat2gray(Y);
X_gt = mat2gray(X_gt);
[mx, nx, ox] = size(X_gt);
[my, ny, oy] = size(Y);

% downsampling
D1 = circularing(ones(scale,1)/scale,mx,my,crc_conv,0); D1 = D1/scale;
D2 = circularing(ones(scale,1)/scale,nx,ny,crc_conv,0); D2 = D2/scale;
D3 = circularing(ones(scale,1)/scale,ox,oy,crc_conv,0); D3 = D3/scale;

% PSF
k = 25;

% 1D coordinates
C1 = circularing([-k:k]',mx,mx,crc_conv,k);
C2 = circularing([-k:k]',nx,nx,crc_conv,k);
C3 = circularing([-k:k]',ox,ox,crc_conv,k);
% 3D coordinates
x = [0:1:k, k*ones(1,mx-(2*k+1)), [-k:1:-1]];
y = [0:1:k, k*ones(1,nx-(2*k+1)), [-k:1:-1]];
z = [0:1:k, k*ones(1,ox-(2*k+1)), [-k:1:-1]];
[CX,CY,CZ] = meshgrid(y,x,z);

P1 = D1*(gauss1(C1,sigma_1_orig)') + D1*reg_1;
P2 = D2*(gauss1(C2,sigma_2_orig)') + D2*reg_2;
P3 = D3*(gauss1(C3,sigma_3_orig)') + D3*reg_3;

if simulation
    Y = tmprod(X_gt,{P1,P2,P3},1:3);
end

tic
U = cpd_rnd(size(X_gt),F);

Y1 = tens2mat(Y,1);
Y2 = tens2mat(Y,2);
Y3 = tens2mat(Y,3);
Y_big = tmprod(Y,{scale*D1',scale*D2',scale*D3'},1:3);
fY = fftn(Y_big);
gamma = 1e-5;

sigma_1_i = sigma_1_orig;
sigma_2_i = sigma_2_orig;
sigma_3_i = sigma_3_orig;

for I = 1:numIt
    I
    P1 = D1*(gauss1(C1,sigma_1_i)') + D1*reg_1;
    P2 = D2*(gauss1(C2,sigma_2_i)') + D2*reg_2;
    P3 = D3*(gauss1(C3,sigma_3_i)') + D3*reg_3;
    
    % update image
    PCPB = kr(P3*U{3},P2*U{2});
    U{1} = pinv(P1)*Y1*pinv(PCPB');
    PCPA = kr(P3*U{3},P1*U{1});
    U{2} = pinv(P2)*Y2*pinv(PCPA');
    PBPA = kr(P2*U{2},P1*U{1});
    U{3} = pinv(P3)*Y3*pinv(PBPA');
    
    %     update kernel
    X = cpdgen(U);
    fX = fftn(X);
    
%     figure(101)
%     clf
    sigma_1_j = sigma_1_i;
    sigma_2_j = sigma_2_i;
    sigma_3_j = sigma_3_i;
    J = 0
    difference = 1;
    while difference > 0.002
        J = J+1;
        H3 = gauss3(CX,CY,CZ,sigma_1_j,sigma_2_j,sigma_3_j);
        fH3 = fftn(H3);
        
        DF = real(ifftn(-2*conj(fX).*fY + 2*conj(fX).*fX.*fH3));
        Dh1 = dngauss3(H3,CX,sigma_1_j);
        Dh2 = dngauss3(H3,CY,sigma_2_j);
        Dh3 = dngauss3(H3,CZ,sigma_3_j);
        
        DF1 = sum(sum(sum(DF.*Dh1)));
        DF2 = sum(sum(sum(DF.*Dh2)));
        DF3 = sum(sum(sum(DF.*Dh3)));
        
        sigma_1_j = sigma_1_j - gamma*DF1;
        sigma_2_j = sigma_2_j - gamma*DF2;
        sigma_3_j = sigma_3_j - gamma*DF3;
        
        sigma_1_j = project_sigma(sigma_1_j,8.2/sqrt(2)-2,8.2/sqrt(2)+2);
        sigma_2_j = project_sigma(sigma_2_j,7.5/sqrt(2)-2,7.5/sqrt(2)+2);
        sigma_3_j = project_sigma(sigma_3_j,1.3/sqrt(2)-1,1.3/sqrt(2)+10);
        
%         hold on
%         plot(J,sigma_1_j,'r*',J,sigma_2_j,'b*',J,sigma_3_j,'k*')
%         title(num2str(I))
%         drawnow
        difference = max([abs(sigma_1_i-sigma_1_j),abs(sigma_2_i-sigma_2_j),...
            abs(sigma_3_i-sigma_3_j)])
        
        sigma_1_i = sigma_1_j;
        sigma_2_i = sigma_2_j;
        sigma_3_i = sigma_3_j;
    end
    
    
end
T = cpdgen(U);
toc
slice_x = 150;
slice_y = 140;
slice_z = 300;

T  = mat2gray(T);

s(1,1) = subplot(3,3,1);
imagesc(squeeze(X_gt(:,:,slice_z))); axis off; axis image; colormap gray
s(1,2) = subplot(3,3,2);
imagesc(1:nx,1:mx,squeeze(Y(:,:,round(slice_z/scale)))); axis off; axis image; colormap gray
s(1,3) = subplot(3,3,3);
imagesc((squeeze(T(:,:,slice_z)))); axis off; axis image; colormap gray
s(2,1) = subplot(3,3,4);
imagesc(squeeze(X_gt(:,slice_y,:))); axis off; axis image; colormap gray
s(2,2) = subplot(3,3,5);
imagesc(1:ox,1:mx,squeeze(Y(:,round(slice_y/scale),:))); axis off; axis image; colormap gray
s(2,3) = subplot(3,3,6);
imagesc((squeeze(T(:,slice_y,:)))); axis off; axis image; colormap gray
s(3,1) = subplot(3,3,7);
imagesc(squeeze(X_gt(slice_x,:,:))); axis off; axis image; colormap gray
s(3,2) = subplot(3,3,8);
imagesc(1:ox,1:nx,squeeze(Y(round(slice_x/scale),:,:))); axis off; axis image; colormap gray
s(3,3) = subplot(3,3,9);
imagesc((squeeze(T(slice_x,:,:)))); axis off; axis image; colormap gray
linkaxes(s(1,:),'xy');
linkaxes(s(2,:),'xy');
linkaxes(s(3,:),'xy');



function X = circularing(x,kg,kt,crc_conv,padval)
% x is vertical array
% kg size of the HR image (in given dimension)
% kt size of the LR image (in given dimension)
kp = length(x);
x = padarray(x,floor((kg-kp)/2),padval,'pre');
x = padarray(x,ceil((kg-kp)/2),padval,'post');
x = x';
% normalize
% x = (x/norm(x));
% padval = x(end);

x = circshift(x,ceil(-(kg)/2+1));
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
        X(I+1,zer) = padval;
    end
end
end



function H = gauss1(x,para)

H = 1/(para*sqrt(2*pi))*exp(-(x).^2/(2*para^2));

end

function DH = dgauss(x,para)

DH = (x.^2 - para^2).*exp(-(x).^2/(2*para^2))./(para^4*sqrt(2*pi));

end

function H3 = gauss3(X,Y,Z,s1,s2,s3)

H3 = gauss1(X,s1).*gauss1(Y,s2).*gauss1(Z,s3);

end

function DnH3 = dngauss3(H,N,sn)

DnH3 = H.*(N.^2-sn^2)/sn^3;

end

function sigma = project_sigma(st,s0,s1)

if st < s0
    sigma = s0;
elseif st > s1
    sigma = s1;
else
    sigma = st;
end

end
