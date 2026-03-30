function out = apply_adaptive_filtering(blk, cls, bSize)

switch cls
    case 1
        out = medfilt2(blk,[3 3]);
    case 2
        out = wiener2(blk,[5 5]);
    case 3
        out = imgaussfilt(blk,0.5);
    case 4
        out = imbilatfilt(blk);
    case 5
        w = wiener2(blk,[5 5]);
        g = imgaussfilt(blk,1);
        out = 0.6*w + 0.4*g;
end
end