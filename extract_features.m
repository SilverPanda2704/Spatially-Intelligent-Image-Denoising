function feat = extract_features(blk)

v    = blk(:);
f1   = var(v);
f2   = skewness(v);
f3   = kurtosis(v);
f4   = sum(v==0 | v==1)/numel(v);

gx   = diff(blk,1,2); 
gy   = diff(blk,1,1);
gmag = [abs(gx(:)); abs(gy(:))];

f5   = mean(gmag > 0.05);
f6   = entropy(blk);
f7   = mean(gmag);

feat = [f1 f2 f3 f4 f5 f6 f7];
end