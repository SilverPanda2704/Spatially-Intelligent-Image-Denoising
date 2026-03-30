clc; clear; close all;

% -----------------------------
% STEP 1: LOAD IMAGE (use same image 3 times for demo)
% -----------------------------
img = im2double(imread('cameraman.tif'));

imgs = {img, img, img}; % replace later with Barbara, fingerprint, etc.

figure('Position',[100 100 1200 900]);

titles = {'Original', 'Noisy (\sigma=25)', 'NLM', 'Proposed'};

for r = 1:3
    
    img_clean = imgs{r};
    
    % -----------------------------
    % STEP 2: ADD NOISE (sigma = 25)
    % -----------------------------
    sigma = 25/255;
    img_noisy = imnoise(img_clean, 'gaussian', 0, sigma^2);
    
    % -----------------------------
    % STEP 3: NLM FILTER
    % -----------------------------
    img_nlm = imnlmfilt(img_noisy);
    
    % -----------------------------
    % STEP 4: PROPOSED METHOD (simple hybrid)
    % -----------------------------
    temp = medfilt2(img_noisy, [3 3]); % denoise
    img_proposed = imsharpen(temp, 'Radius', 1.5, 'Amount', 1); % edge restore
    
    % -----------------------------
    % STEP 5: PLOT ROW
    % -----------------------------
    images = {img_clean, img_noisy, img_nlm, img_proposed};
    
    for c = 1:4
        subplot(3,4,(r-1)*4 + c);
        imshow(images{c});
        
        if r == 1
            title(titles{c});
        end
    end
end

sgtitle('Visual Comparison of Denoising Methods');

% -----------------------------
% STEP 6: SAVE IMAGE FOR LATEX
% -----------------------------
exportgraphics(gcf, 'visual_comparison.png', 'Resolution', 300);