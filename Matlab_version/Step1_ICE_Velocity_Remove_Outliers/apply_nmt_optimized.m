function [filtered_data, outlier_mask, norm_residual, stats] = ...
    apply_nmt_optimized(data, window_size, threshold, epsilon, min_segment)
% 优化的归一化中值检验
%
% 使用nlfilter或im2col进行滑动窗口操作
% 使用矢量化操作提高速度
fprintf("====================================================== \n")
fprintf("====================== start NMT  ==================== \n")
fprintf("====================================================== \n")

[rows, cols] = size(data);
half_win = floor(window_size / 2);

fprintf('  计算局部中值 (使用优化算法)...\n');

% 方法1: 使用移动窗口中值滤波
% 注意: medfilt2不能排除中心点，所以需要特殊处理
% 这里使用自定义函数

% 初始化
local_median = nan(size(data), 'single');
residual_median = nan(size(data), 'single');

% 步骤1: 计算局部中值 U_m
% 使用padarray避免边界问题
data_padded = padarray(data, [half_win, half_win], NaN, 'both');

for i = 1:rows
    for j = 1:cols
        % 提取邻域
        neighborhood = data_padded(i:i+2*half_win, j:j+2*half_win);
        
        % 排除中心点
        neighborhood(half_win+1, half_win+1) = NaN;
        
        % 计算中值
        valid_neighbors = neighborhood(~isnan(neighborhood));
        if length(valid_neighbors) >= window_size
            local_median(i, j) = median(valid_neighbors, 'omitnan');
        end
    end
end

fprintf('  计算残差中值...\n');

% 步骤2: 计算残差中值 R_m
for i = 1:rows
    for j = 1:cols
        u_m = local_median(i, j);
        
        if isnan(u_m)
            continue;
        end
        
        % 提取邻域
        neighborhood = data_padded(i:i+2*half_win, j:j+2*half_win);
        
        % 计算残差
        residuals = abs(neighborhood - u_m);
        residuals(half_win+1, half_win+1) = NaN;
        
        % 计算残差中值
        valid_residuals = residuals(~isnan(residuals));
        if length(valid_residuals) >= window_size
            residual_median(i, j) = median(valid_residuals, 'omitnan');
        end
    end
end

% 步骤3: 归一化残差 (矢量化)
norm_residual = abs(data - local_median) ./ (residual_median + epsilon);

% 步骤4: 异常值检测 (矢量化)
outlier_mask = (norm_residual > threshold) & ~isnan(data);

% 滤波
filtered_data = data;
filtered_data(outlier_mask) = NaN;

% 步骤5: 移除小连通域 (使用bwconncomp)
fprintf('  移除小连通域...\n');
valid_mask = ~isnan(filtered_data);
cc = bwconncomp(valid_mask);
num_pixels = cellfun(@numel, cc.PixelIdxList);
small_idx = num_pixels < min_segment;

n_small = 0;
for k = find(small_idx)
    filtered_data(cc.PixelIdxList{k}) = NaN;
    n_small = n_small + num_pixels(k);
end

% 统计
n_valid = sum(~isnan(data(:)));
n_outliers = sum(outlier_mask(:));

stats = struct();
stats.n_valid = n_valid;
stats.n_outliers = n_outliers;
stats.outlier_pct = n_outliers / n_valid * 100;
stats.n_small = n_small;
stats.window_size = window_size;
stats.threshold = threshold;

end
