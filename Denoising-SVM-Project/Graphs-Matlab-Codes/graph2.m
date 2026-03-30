clc; clear; close all;

% --- SAMPLE DATA (REMOVE THIS IF YOU ALREADY HAVE DATA) ---
% Replace these with your actual labels
Y_true = [1 2 3 4 5 1 2 3 4 5];
Y_pred = [1 2 2 4 5 1 3 3 4 5];

classNames = {'Median','Wiener','Gaussian','Bilateral','Hybrid'};

% --- CONFUSION MATRIX ---
confmatrix = confusionmat(Y_true, Y_pred);

% --- NORMALIZE ROW-WISE ---
confmatrix_norm = confmatrix ./ sum(confmatrix, 2);

% Handle divide-by-zero (important)
confmatrix_norm(isnan(confmatrix_norm)) = 0;

% --- PLOT HEATMAP ---
figure;
h = heatmap(classNames, classNames, confmatrix_norm);

h.Title = 'SVM Classification Confusion Matrix';
h.XLabel = 'Predicted Class';
h.YLabel = 'True Class';

colormap hot;
caxis([0 1]);