clc; clear; close all;

% -----------------------------
% STEP 1: CREATE CLEAN SYNTHETIC DATA
% -----------------------------

classNames = {'Median','Wiener','Gaussian','Bilateral','Hybrid'};
numClasses = 5;
samplesPerClass = 50;

X = [];
Y = [];

for c = 1:numClasses
    
    for i = 1:samplesPerClass
        
        % 7 features with slight class separation
        X = [X;
            randn + c, ...
            randn*0.5 + c*0.8, ...
            abs(randn) + c*0.3, ...
            rand + c*0.2, ...
            randn*0.3 + c, ...
            randn*0.2 + c*0.5, ...
            abs(randn)*0.5 + c*0.1
        ];
        
        Y = [Y; c];
    end
end

Y = categorical(Y);

% -----------------------------
% STEP 2: PLOT (SAFE VERSION)
% -----------------------------

figure('Position',[100 100 1000 700]);

for feat = 1:7
    
    subplot(2,4,feat);
    
    data = X(:,feat);
    
    % safe normalization (prevents divide-by-zero issues)
    data = (data - mean(data)) / (std(data) + eps);
    
    boxplot(data, Y);
    grid on;
    
    title(['Feature F_' num2str(feat)]);
    ylabel('Normalized Value');
    
end

sgtitle('Feature Distribution Across 5 Classes');

set(gca,'XTickLabel',classNames);