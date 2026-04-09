function mean_map = local_mean(data, window_size, ignore_nan, min_valid)
% 局部平均 (支持NaN)

[rows, cols] = size(data);
half_win = floor(window_size / 2);
mean_map = nan(size(data), 'single');

for i = 1:rows
    for j = 1:cols
        row_start = max(1, i - half_win);
        row_end = min(rows, i + half_win);
        col_start = max(1, j - half_win);
        col_end = min(cols, j + half_win);
        
        window = data(row_start:row_end, col_start:col_end);
        
        if ignore_nan
            valid = window(~isnan(window) & ~isinf(window));
        else
            valid = window(:);
        end
        
        if length(valid) >= min_valid
            mean_map(i, j) = mean(valid);
        end
    end
end

end