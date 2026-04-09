function median_map = local_median(data, window_size, min_valid)
% 局部中值 (支持NaN)

[rows, cols] = size(data);
half_win = floor(window_size / 2);
median_map = nan(size(data), 'single');

for i = 1:rows
    for j = 1:cols
        row_start = max(1, i - half_win);
        row_end = min(rows, i + half_win);
        col_start = max(1, j - half_win);
        col_end = min(cols, j + half_win);
        
        window = data(row_start:row_end, col_start:col_end);
        valid = window(~isnan(window) & ~isinf(window));
        
        if length(valid) >= min_valid
            median_map(i, j) = median(valid);
        end
    end
end

end