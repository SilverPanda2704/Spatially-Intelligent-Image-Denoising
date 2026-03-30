img = imread('cameraman.tif');
img = im2double(img);
noisy = imnoise(img,'gaussian',0,0.01);
denoised = wiener2(noisy,[5 5]);
classMap = randi([1 5], size(img));

% now run your plotting code

figure('Position', [100 100 1000 300]);

if isempty(img) || isempty(noisy) || isempty(denoised) || isempty(classMap)
    error('One or more variables are empty. Run full pipeline first.');
end

% ORIGINAL
subplot(1,4,1);
imshow(img, []);
title('Original');
axis image off;

% NOISY
subplot(1,4,2);
imshow(noisy, []);
title('Noisy');
axis image off;

% CLASS MAP
subplot(1,4,3);
imagesc(classMap);
axis image off;
colormap(gca, jet);
colorbar;
title('SVM Classification Map');

% DENOISED
subplot(1,4,4);
imshow(denoised, []);
title('Denoised');
axis image off;

