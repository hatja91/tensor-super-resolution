function X = circularing(x,kg,kt,crc_conv,regul)
% x is vertical array
% k size os the image (in given dimension)
kp = length(x);
x = padarray(x,floor((kg-kp)/2),'pre');
x = padarray(x,ceil((kg-kp)/2),'post');
x = x';
x = circshift(x,ceil(-(kg)/2+1));
% normalize
x = (x/sum(x));
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

