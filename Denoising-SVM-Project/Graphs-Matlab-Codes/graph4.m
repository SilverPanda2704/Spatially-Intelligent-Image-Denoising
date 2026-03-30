% Read image
img = imread('cameraman.tif'); % you can replace with your own image
img = im2double(img);

% Create noisy image
noisy = imnoise(img, 'gaussian', 0, 0.01);

% Denoising (block-wise / approximation using median filter)
denoised = medfilt2(noisy, [3 3]);

% Edge-aware sharpening (post-processing)
sharpened = imsharpen(denoised, 'Radius', 2, 'Amount', 1);

% Edge maps (Canny)
edge_orig = edge(img, 'canny');
edge_noisy = edge(noisy, 'canny');
edge_denoised = edge(denoised, 'canny');
edge_final = edge(sharpened, 'canny');

% ---- PLOTTING ----
figure('Position', [100 100 1200 600]);

% Row 1: Images
subplot(2,4,1);
imshow(img);
title('Original Image');

subplot(2,4,2);
imshow(noisy);
title('Noisy Image');

subplot(2,4,3);
imshow(denoised);
title('Denoised');

subplot(2,4,4);
imshow(sharpened);
title('Final Output');

% Row 2: Edge Maps
subplot(2,4,5);
imshow(edge_orig);
title('Edges (Original)');

subplot(2,4,6);
imshow(edge_noisy);
title('Edges (Noisy)');

subplot(2,4,7);
imshow(edge_denoised);
title('Edges (Denoised)');

subplot(2,4,8);
imshow(edge_final);
title('Edges (Final)');

sgtitle('Edge Preservation Before and After Post-Processing');

% ---- SAVE HIGH QUALITY ----
print(gcf, 'edge_preservation_comparison', '-dpng', '-r300');