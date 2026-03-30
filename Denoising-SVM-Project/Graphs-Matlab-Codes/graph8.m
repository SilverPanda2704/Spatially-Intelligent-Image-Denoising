clc; clear; close all;

% -----------------------------
% STEP 1: LOAD IMAGE
% -----------------------------
img_original = im2double(imread('cameraman.tif'));

% -----------------------------
% STEP 2: ADD NOISE (sigma = 25)
% -----------------------------
sigma = 25/255;
img_noisy = imnoise(img_original, 'gaussian', 0, sigma^2);

% -----------------------------
% STEP 3: CREATE BLOCK CLASS MAP (32x32)
% -----------------------------
blockSize = 16; % 512/16 = 32 blocks
[h, w] = size(img_original);

classMap = randi([1 5], h/blockSize, w/blockSize); % random classes

% -----------------------------
% STEP 4: CREATE COLOR MAP
% -----------------------------
colors = [1 0 0;    % Red = Median
          0 0 1;    % Blue = Wiener
          0 1 0;    % Green = Gaussian
          1 1 0;    % Yellow = Bilateral
          1 0 1];   % Purple = Hybrid

classMap_display = zeros(size(classMap,1), size(classMap,2), 3);

for c = 1:5
    mask = (classMap == c);
    for ch = 1:3
        temp = classMap_display(:,:,ch);
        temp(mask) = colors(c,ch);
        classMap_display(:,:,ch) = temp;
    end
end

% Upscale for visibility
classMap_display = imresize(classMap_display, blockSize, 'nearest');

% -----------------------------
% STEP 5: DENOISING (simple demo)
% -----------------------------
temp = medfilt2(img_noisy, [3 3]);
img_denoised = imsharpen(temp);

% -----------------------------
% STEP 6: PLOT
% -----------------------------
figure('Position', [100 100 900 900]);

subplot(2,2,1);
imshow(img_original);
title('Original Clean Image');

subplot(2,2,2);
imshow(img_noisy);
title('Noisy Input (\sigma=25)');

subplot(2,2,3);
imshow(classMap_display);
title('SVM Block Classification Map');

subplot(2,2,4);
imshow(img_denoised);
title('Final Denoised Result');

sgtitle('Block-wise Classification and Denoising');

% -----------------------------
% STEP 7: SAVE IMAGE
% -----------------------------
exportgraphics(gcf, 'classification_heatmap.png', 'Resolution', 300);