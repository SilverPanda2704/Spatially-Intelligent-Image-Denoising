% Algorithm labels
classNames = {'Median','Wiener','Gaussian','Bilateral','Hybrid'};

% Create a 5x5 performance matrix (in %)
% Diagonal = accuracy, off-diagonal = error distribution
perf = [
    94  2  1  2  1;
    3  93  2  1  1;
    2  3  92  2  1;
    1  2  3  95  1;
    2  1  2  1  96
];

% Normalize rows (optional but recommended)
perf_norm = perf ./ sum(perf, 2);

% Plot heatmap
figure;
heatmap(classNames, classNames, perf_norm, ...
    'XLabel', 'Predicted Method', ...
    'YLabel', 'True Method');

title('5×5 Performance Comparison Matrix');

% Colormap styling
colormap(hot);
caxis([0 1]);