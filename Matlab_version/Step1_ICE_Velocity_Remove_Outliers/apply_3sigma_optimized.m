function [filtered_data, outlier_mask, norm_residual, stats] = ...
    apply_3sigma_optimized(data, sigma_times, min_segment)
% 优化的3-sigma异常值剔除
%
% 使用矢量化操作提高速度
fprintf("====================================================== \n")
fprintf("======================= start N-Sigma ================ \n")
fprintf("====================================================== \n")

% 计算统计量 (使用内置函数)
valid_data = data(~isnan(data));
n_valid = length(valid_data);

mu = mean(valid_data);
sigma = std(valid_data, 1);  % 使用样本标准差

% N-sigma边界
lower_bound = mu - sigma_times * sigma;
upper_bound = mu + sigma_times * sigma;

% 检测异常值 (矢量化)
outlier_mask = (data < lower_bound | data > upper_bound) & ~isnan(data);

% 滤波
filtered_data = data;
filtered_data(outlier_mask) = NaN;

% 移除小连通域 (使用bwconncomp优化)
valid_mask = ~isnan(filtered_data);
cc = bwconncomp(valid_mask);
num_pixels = cellfun(@numel, cc.PixelIdxList);
small_idx = num_pixels < min_segment;

n_small = 0;
for k = find(small_idx)
    filtered_data(cc.PixelIdxList{k}) = NaN;
    n_small = n_small + num_pixels(k);
end

% 归一化残差 (为了可视化)
norm_residual = (data - mu) / sigma;

% 统计
n_outliers = sum(outlier_mask(:));

stats = struct();
stats.n_valid = n_valid;
stats.n_outliers = n_outliers;
stats.outlier_pct = n_outliers / n_valid * 100;
stats.n_small = n_small;
stats.mu = mu;
stats.sigma = sigma;
stats.lower = lower_bound;
stats.upper = upper_bound;
stats.sigma_times = sigma_times;

% disp(stats)

end