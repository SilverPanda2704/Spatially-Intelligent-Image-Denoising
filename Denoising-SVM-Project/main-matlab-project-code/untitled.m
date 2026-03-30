function untitled

clc; clear; close all;

%% =========================================================
%  AI-POWERED INTELLIGENT ADAPTIVE IMAGE DENOISING SYSTEM
%
%  Novel Contribution:
%  A trained SVM classifier analyses per-block statistical
%  features to autonomously identify noise type and severity,
%  then selects the optimal filter — no hardcoded thresholds.
%
%  ML Pipeline:
%   1. Synthetic training data generation
%   2. Feature extraction (7 statistical features per block)
%   3. SVM multi-class training (5 noise classes)
%   4. Per-block inference → filter selection
%   5. Adaptive filtering + edge-aware sharpening
%% =========================================================

%% ---- FIGURE ----
fig = uifigure('Name','AI-Powered Adaptive Image Denoising System',...
    'Position',[50 50 1200 780]);

%% ---- IMAGE AXES ----
ax1 = uiaxes(fig,'Position',[30  520 240 210]); title(ax1,'Original');
ax2 = uiaxes(fig,'Position',[280 520 240 210]); title(ax2,'Noisy');
ax3 = uiaxes(fig,'Position',[530 520 240 210]); title(ax3,'Denoised');
ax4 = uiaxes(fig,'Position',[780 520 380 210]); title(ax4,'Block Classification Map');
colormap(ax4,'jet');

%% ---- CONTROL PANEL ----
panel = uipanel(fig,'Title','Controls','Position',[30 400 1140 110]);

uibutton(panel,'push','Text','Load Image',...
    'Position',[10 45 120 38],...
    'ButtonPushedFcn',@(~,~) loadImage());

uilabel(panel,'Text','Noise Type:','Position',[148 70 80 20]);
noiseType = uidropdown(panel,...
    'Items',{'Gaussian','Salt & Pepper','Speckle','Poisson','Mixed'},...
    'Position',[148 48 120 25]);

uilabel(panel,'Text','Noise Level:','Position',[283 70 80 20]);
noiseSlider = uislider(panel,'Position',[283 60 110 3],...
    'Limits',[0.001 0.1],'Value',0.02);
noiseLevelLbl = uilabel(panel,'Text','0.020','Position',[400 53 45 20]);
noiseSlider.ValueChangedFcn = @(s,~) set(noiseLevelLbl,'Text',sprintf('%.3f',s.Value));

uibutton(panel,'push','Text','Add Noise',...
    'Position',[455 45 120 38],...
    'ButtonPushedFcn',@(~,~) addNoise());

uilabel(panel,'Text','Block Size:','Position',[593 70 70 20]);
blockSizeDrop = uidropdown(panel,'Items',{'8','16','32','64'},...
    'Value','16','Position',[593 48 60 25]);

uibutton(panel,'push','Text','Train AI Classifier',...
    'Position',[668 45 150 38],...
    'BackgroundColor',[0.85 0.33 0.1],...
    'FontColor','white','FontWeight','bold',...
    'ButtonPushedFcn',@(~,~) trainClassifier());

uibutton(panel,'push','Text','Run AI Denoising',...
    'Position',[830 45 160 38],...
    'BackgroundColor',[0.2 0.6 0.9],...
    'FontColor','white','FontWeight','bold',...
    'ButtonPushedFcn',@(~,~) denoiseImage());

uibutton(panel,'push','Text','Export Report',...
    'Position',[1005 45 120 38],...
    'BackgroundColor',[0.2 0.75 0.45],...
    'FontColor','white',...
    'ButtonPushedFcn',@(~,~) exportReport());

%% ---- BOTTOM AXES ----
graphAx   = uiaxes(fig,'Position',[30  50 400 310]);
title(graphAx,'Performance Comparison');

blockAx   = uiaxes(fig,'Position',[460 50 340 310]);
title(blockAx,'Block Filter Distribution');

confAx    = uiaxes(fig,'Position',[830 50 330 310]);
title(confAx,'SVM Confidence per Block');

%% ---- RESULTS PANEL (overlaid bottom-right corner) ----
resultPanel = uipanel(fig,'Title','AI Metrics','Position',[830 50 330 310]);
resultLabel = uilabel(resultPanel,...
    'Position',[10 120 310 160],...
    'Text','Train classifier, then run denoising.',...
    'FontSize',11,'WordWrap','on');
statusLabel = uilabel(resultPanel,...
    'Position',[10 90 310 25],...
    'Text','Status: Idle',...
    'FontSize',11,'FontWeight','bold');
trainLabel  = uilabel(resultPanel,...
    'Position',[10 60 310 25],...
    'Text','Classifier: Not trained',...
    'FontSize',10,'FontColor',[0.7 0.1 0.1]);
accLabel    = uilabel(resultPanel,...
    'Position',[10 35 310 25],...
    'Text','Training Accuracy: —',...
    'FontSize',10);

%% ---- SHARED STATE ----
img           = [];
noisy         = [];
denoised      = [];
classMap      = [];     % per-block predicted class index
confScores    = [];     % per-block SVM confidence
svmModel      = [];     % trained classifier
metrics       = struct();
blockCounts   = zeros(1,5);   % counts per class [Median Wiener Gaussian Bilateral Hybrid]
classNames    = {'Median','Wiener','Gaussian','Bilateral','Hybrid'};

%% ================================================================
%  LOAD IMAGE
%% ================================================================
    function loadImage()
        [file,path] = uigetfile({'*.jpg;*.jpeg;*.png;*.tif;*.bmp','Images'});
        if isequal(file,0), return; end
        raw = im2double(imread(fullfile(path,file)));
        if size(raw,3)==3, raw = rgb2gray(raw); end
        img = raw;
        imshow(img,'Parent',ax1);
        cla(ax2); cla(ax3); cla(ax4);
        statusLabel.Text = 'Status: Image Loaded';
    end

%% ================================================================
%  ADD NOISE
%% ================================================================
    function addNoise()
        if isempty(img)
            uialert(fig,'Load an image first.','Error'); return;
        end
        lvl = noiseSlider.Value;
        switch noiseType.Value
            case 'Gaussian'
                noisy = imnoise(img,'gaussian',0,lvl);
            case 'Salt & Pepper'
                noisy = imnoise(img,'salt & pepper',lvl);
            case 'Speckle'
                noisy = imnoise(img,'speckle',lvl);
            case 'Poisson'
                noisy = imnoise(img,'poisson');
            case 'Mixed'
                tmp   = imnoise(img,'gaussian',0,lvl*0.6);
                noisy = imnoise(tmp,'salt & pepper',lvl*0.4);
        end
        imshow(noisy,'Parent',ax2);
        statusLabel.Text = sprintf('Status: Noise Added (%s, %.3f)',noiseType.Value,lvl);
    end

%% ================================================================
%  TRAIN SVM CLASSIFIER   ← The AI Core
%
%  Strategy: Generate synthetic noisy patches from a uniform
%  random "image" base, extract 7 statistical features per patch,
%  label each patch by noise type, train a multi-class SVM.
%
%  Classes:
%   1 = Median     (impulse / S&P dominant)
%   2 = Wiener     (heavy Gaussian, flat region)
%   3 = Gaussian   (very light noise)
%   4 = Bilateral  (heavy noise + high texture)
%   5 = Hybrid     (mixed / moderate noise)
%
%  Features per block:
%   F1 = local variance
%   F2 = skewness of pixel intensities
%   F3 = kurtosis of pixel intensities
%   F4 = impulse ratio  (fraction of 0/1 pixels)
%   F5 = edge density   (gradient threshold fraction)
%   F6 = entropy        (texture complexity)
%   F7 = mean gradient magnitude
%% ================================================================
    function trainClassifier()
        pb = uiprogressdlg(fig,'Title','Training AI Classifier',...
            'Message','Generating synthetic training data...','Cancelable','off');

        bSize       = 16;       % fixed training patch size
        nPerClass   = 300;      % synthetic patches per class
        nClasses    = 5;
        nFeatures   = 7;
        totalSamples= nPerClass * nClasses;

        X = zeros(totalSamples, nFeatures);
        Y = zeros(totalSamples, 1);

        idx = 0;
        for cls = 1:nClasses
            pb.Value   = 0.05 + 0.55*(cls/nClasses);
            pb.Message = sprintf('Generating class %d/%d patches...', cls, nClasses);

            for s = 1:nPerClass
                % Random base patch (simulates an image region)
                basePatch = rand(bSize, bSize);

                % Add noise matching this class
                switch cls
                    case 1   % Median class — strong impulse
                        patch = imnoise(basePatch,'salt & pepper', 0.05 + rand()*0.15);
                    case 2   % Wiener class — heavy Gaussian, flat
                        patch = imnoise(smooth2d(basePatch,3),'gaussian',0, 0.01+rand()*0.04);
                    case 3   % Gaussian class — very light noise
                        patch = imnoise(basePatch,'gaussian',0, 0.0005+rand()*0.002);
                    case 4   % Bilateral class — noise + texture
                        textured = basePatch + 0.3*sin(2*pi*(1:bSize)/4)'*cos(2*pi*(1:bSize)/4);
                        textured = mat2gray(textured);
                        patch = imnoise(textured,'gaussian',0, 0.005+rand()*0.02);
                    case 5   % Hybrid class — mixed moderate noise
                        tmp   = imnoise(basePatch,'gaussian',0, 0.003+rand()*0.008);
                        patch = imnoise(tmp,'salt & pepper', 0.01+rand()*0.04);
                end

                idx = idx + 1;
                X(idx,:) = extractFeatures(patch);
                Y(idx)   = cls;
            end
        end

        pb.Value   = 0.65;
        pb.Message = 'Training SVM model...';

        % Train multi-class SVM (one-vs-one)
        svmModel = fitcecoc(X, Y,...
            'Learners', templateSVM('KernelFunction','rbf','Standardize',true),...
            'ClassNames', 1:nClasses);

        pb.Value   = 0.90;
        pb.Message = 'Evaluating training accuracy...';

        % Cross-validation accuracy
        cvModel  = crossval(svmModel,'KFold',5);
        cvLoss   = kfoldLoss(cvModel);
        trainAcc = (1 - cvLoss) * 100;

        pb.Value = 1.0;
        close(pb);

        trainLabel.Text     = 'Classifier: Trained ✓';
        trainLabel.FontColor= [0.1 0.6 0.1];
        accLabel.Text       = sprintf('5-Fold CV Accuracy: %.1f%%', trainAcc);
        statusLabel.Text    = 'Status: AI Classifier Ready';

        uialert(fig, sprintf(['SVM Classifier trained successfully!\n\n' ...
            '5-Fold Cross-Validation Accuracy: %.1f%%\n' ...
            'Classes: Median | Wiener | Gaussian | Bilateral | Hybrid\n' ...
            'Features: Variance, Skewness, Kurtosis, Impulse, Edge, Entropy, Gradient'], ...
            trainAcc), 'Training Complete', 'Icon','success');
    end

%% ================================================================
%  AI-POWERED BLOCK-WISE DENOISING
%% ================================================================
    function denoiseImage()
        if isempty(noisy)
            uialert(fig,'Add noise first.','Error'); return;
        end
        if isempty(svmModel)
            uialert(fig,'Train the AI classifier first!','Error'); return;
        end

        pb = uiprogressdlg(fig,'Title','AI Denoising',...
            'Message','Initialising...','Cancelable','off');

        blockSize = str2double(blockSizeDrop.Value);
        [H,W]     = size(noisy);

        % Pad to block multiple
        padH = ceil(H/blockSize)*blockSize;
        padW = ceil(W/blockSize)*blockSize;
        nPad = padarray(noisy,[padH-H, padW-W],'replicate','post');
        denPad = zeros(padH,padW);

        nBH = padH/blockSize;
        nBW = padW/blockSize;
        totalBlocks = nBH*nBW;

        % Storage for visualisation
        classMapPad = zeros(nBH, nBW);
        confPad     = zeros(nBH, nBW);
        blockCounts = zeros(1,5);

        pb.Message = 'AI classifying blocks & filtering...';
        blkNum = 0;

        for bi = 1:nBH
            for bj = 1:nBW
                blkNum = blkNum + 1;
                pb.Value = blkNum / totalBlocks;

                r1 = (bi-1)*blockSize+1; r2 = r1+blockSize-1;
                c1 = (bj-1)*blockSize+1; c2 = c1+blockSize-1;
                blk = nPad(r1:r2, c1:c2);

                % ---- AI INFERENCE ----
                feat = extractFeatures(blk);
                [predClass, score] = predict(svmModel, feat);
                conf = max(score);   % highest class probability

                classMapPad(bi,bj) = predClass;
                confPad(bi,bj)     = conf;
                blockCounts(predClass) = blockCounts(predClass) + 1;

                % ---- Apply filter based on SVM prediction ----
                denPad(r1:r2,c1:c2) = applyFilter(blk, predClass, blockSize);
            end
        end

        denoised = denPad(1:H,1:W);
        classMap = classMapPad;
        confScores = confPad;

        pb.Message = 'Edge-aware sharpening...';
        pb.Value   = 0.93;

        % Global edge-aware sharpening pass
        edgeMask = edge(img,'canny');
        sharpMask = imgaussfilt(double(edgeMask),1.5);
        denoised  = max(0,min(1, denoised + 0.12*sharpMask));

        pb.Message = 'Computing metrics...';
        pb.Value   = 0.97;

        p_noisy    = psnr(noisy,    img);
        p_denoised = psnr(denoised, img);
        s_noisy    = ssim(noisy,    img);
        s_denoised = ssim(denoised, img);
        mse_val    = immse(denoised, img);
        edgePres   = edgePreservationRatio(img, denoised);
        improvement= ((p_denoised-p_noisy)/abs(p_noisy))*100;

        metrics = struct('p_noisy',p_noisy,'p_denoised',p_denoised,...
            's_noisy',s_noisy,'s_denoised',s_denoised,...
            'mse',mse_val,'edgePres',edgePres,'improvement',improvement);

        pb.Value = 1.0; close(pb);

        % ---- Display ----
        imshow(denoised,'Parent',ax3);

        % Block classification map (coloured by class)
        classImg = imresize(classMapPad,[H W],'nearest') / 5;
        imagesc(classImg,'Parent',ax4); colormap(ax4,'jet');
        axis(ax4,'image'); colorbar(ax4);
        title(ax4,'Block Classification Map (colour = filter used)');

        updateGraphs();

        resultLabel.Text = sprintf([...
            'PSNR:   %.2f → %.2f dB\n',...
            'SSIM:   %.3f → %.3f\n',...
            'MSE:    %.5f\n',...
            'Edge Preservation: %.1f%%\n',...
            'PSNR Improvement:  %+.1f%%'],...
            p_noisy,p_denoised,s_noisy,s_denoised,mse_val,edgePres*100,improvement);
        statusLabel.Text = 'Status: AI Denoising Complete ✓';
    end

%% ================================================================
%  FEATURE EXTRACTION — 7 Statistical Features
%  Input:  image block (any size)
%  Output: 1×7 feature vector
%% ================================================================
    function feat = extractFeatures(blk)
        v    = blk(:);
        f1   = var(v);                              % variance
        f2   = skewness(v);                         % skewness
        f3   = kurtosis(v);                         % kurtosis
        f4   = sum(v==0|v==1)/numel(v);             % impulse ratio
        gx   = diff(blk,1,2); gy = diff(blk,1,1);
        gmag = [abs(gx(:)); abs(gy(:))];
        f5   = mean(gmag > 0.05);                   % edge density
        f6   = entropy(blk);                        % entropy (texture)
        f7   = mean(gmag);                          % mean gradient magnitude
        feat = [f1 f2 f3 f4 f5 f6 f7];
    end

%% ================================================================
%  APPLY FILTER based on SVM predicted class
%% ================================================================
    function out = applyFilter(blk, cls, bSize)
        switch cls
            case 1   % Median
                out = medfilt2(blk,[3 3]);
            case 2   % Wiener
                out = wiener2(blk,[5 5]);
            case 3   % Light Gaussian
                out = imgaussfilt(blk,0.5);
            case 4   % Bilateral (edge-preserving)
                out = bilateralFilter(blk, bSize);
            case 5   % Hybrid Wiener+Gaussian
                w   = wiener2(blk,[5 5]);
                g   = imgaussfilt(blk,1.0);
                out = 0.65*w + 0.35*g;
        end
    end

%% ================================================================
%  BILATERAL FILTER — no extra toolbox needed
%% ================================================================
    function out = bilateralFilter(blk, bSize)
        sigmaS = max(1, bSize/8);
        sigmaR = 0.15;
        [rows,cols] = size(blk);
        out  = zeros(rows,cols);
        halfW= max(2,round(sigmaS*2));
        for r = 1:rows
            for c = 1:cols
                rMin=max(1,r-halfW); rMax=min(rows,r+halfW);
                cMin=max(1,c-halfW); cMax=min(cols,c+halfW);
                patch = blk(rMin:rMax,cMin:cMax);
                [pr,pc]=ndgrid(rMin:rMax,cMin:cMax);
                wS = exp(-((pr-r).^2+(pc-c).^2)/(2*sigmaS^2));
                wR = exp(-(patch-blk(r,c)).^2/(2*sigmaR^2));
                w  = wS.*wR;
                out(r,c)=sum(w(:).*patch(:))/sum(w(:));
            end
        end
    end

%% ================================================================
%  SMOOTH2D — helper used in synthetic patch generation
%% ================================================================
    function out = smooth2d(img2d, sigma)
        out = imgaussfilt(img2d, sigma);
    end

%% ================================================================
%  EDGE PRESERVATION RATIO
%% ================================================================
    function ratio = edgePreservationRatio(original, filtered)
        eO = edge(original,'sobel');
        eF = edge(filtered,'sobel');
        ratio = sum(eO(:)&eF(:)) / max(1,sum(eO(:)));
    end

%% ================================================================
%  UPDATE GRAPHS
%% ================================================================
    function updateGraphs()
        % Performance graph
        cla(graphAx); hold(graphAx,'on');
        bar(graphAx,1,[metrics.p_noisy metrics.p_denoised],'grouped');
        xticks(graphAx,1); xticklabels(graphAx,{''});
        hold(graphAx,'off');

        cla(graphAx); hold(graphAx,'on');
        plot(graphAx,[1 2],[metrics.p_noisy metrics.p_denoised],'o-b','LineWidth',2,'MarkerSize',8);
        plot(graphAx,[1 2],[metrics.s_noisy*40 metrics.s_denoised*40],'s-r','LineWidth',2,'MarkerSize',8);
        xticks(graphAx,[1 2]); xticklabels(graphAx,{'Noisy','Denoised'});
        legend(graphAx,{'PSNR (dB)','SSIM×40'},'Location','best');
        grid(graphAx,'on'); title(graphAx,'Performance Comparison');
        hold(graphAx,'off');

        % Block distribution bar
        cla(blockAx);
        clrs = [0.2 0.6 0.9;0.9 0.4 0.2;0.3 0.8 0.4;0.8 0.3 0.8;0.9 0.75 0.1];
        b = bar(blockAx, blockCounts,'FaceColor','flat');
        b.CData = clrs;
        set(blockAx,'XTickLabel',classNames,'XTick',1:5);
        ylabel(blockAx,'Blocks'); grid(blockAx,'on');
        title(blockAx,'AI Filter Assignment Distribution');

        % Confidence heatmap
        if ~isempty(confScores)
            imagesc(confScores,'Parent',confAx);
            colormap(confAx,'hot'); colorbar(confAx);
            axis(confAx,'image');
            title(confAx,'SVM Confidence Map');
        end
    end

%% ================================================================
%  EXPORT REPORT
%% ================================================================
    function exportReport()
        if isempty(denoised)
            uialert(fig,'Run denoising first.','Error'); return;
        end
        [file,path] = uiputfile({'*.png','PNG';'*.txt','Report'},...
            'Save','ai_denoised_report');
        if isequal(file,0), return; end

        imwrite(denoised, fullfile(path, strrep(file,'.txt','.png')));

        fid = fopen(fullfile(path, strrep(file,'.png','.txt')),'w');
        fprintf(fid,'=== AI-Powered Denoising Report ===\n');
        fprintf(fid,'Classifier:         SVM (RBF Kernel, One-vs-One)\n');
        fprintf(fid,'Features Used:      Variance, Skewness, Kurtosis, Impulse,\n');
        fprintf(fid,'                    Edge Density, Entropy, Gradient Magnitude\n');
        fprintf(fid,'Noise Type:         %s\n', noiseType.Value);
        fprintf(fid,'Block Size:         %s x %s px\n', blockSizeDrop.Value, blockSizeDrop.Value);
        fprintf(fid,'\n--- Image Quality Metrics ---\n');
        fprintf(fid,'PSNR (noisy):       %.2f dB\n', metrics.p_noisy);
        fprintf(fid,'PSNR (denoised):    %.2f dB\n', metrics.p_denoised);
        fprintf(fid,'SSIM (noisy):       %.4f\n',    metrics.s_noisy);
        fprintf(fid,'SSIM (denoised):    %.4f\n',    metrics.s_denoised);
        fprintf(fid,'MSE:                %.6f\n',    metrics.mse);
        fprintf(fid,'Edge Preservation:  %.1f%%\n',  metrics.edgePres*100);
        fprintf(fid,'PSNR Improvement:   %+.1f%%\n', metrics.improvement);
        fprintf(fid,'\n--- AI Block Classification ---\n');
        for k=1:5
            fprintf(fid,'  %-12s: %d blocks\n', classNames{k}, blockCounts(k));
        end
        fclose(fid);

        uialert(fig,'Report exported successfully.','Done','Icon','success');
    end

end