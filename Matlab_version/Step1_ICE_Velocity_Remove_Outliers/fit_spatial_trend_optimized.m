function [trend_surface, detrend, params, stats] = ...
    fit_spatial_trend_optimized(data, method, use_weights)
% 优化的空间趋势面拟合
%
% 使用lscov进行加权最小二乘拟合
% 使用meshgrid矢量化操作
fprintf("====================================================== \n")
fprintf("====================== start deramp ================== \n")
fprintf("====================================================== \n")

[rows, cols] = size(data);

% 创建坐标网格 (矢量化)
[X, Y] = meshgrid(1:cols, 1:rows);

% 展平
x_flat = X(:);
y_flat = Y(:);
z_flat = data(:);

% 移除NaN
valid_idx = ~isnan(z_flat);
x_valid = x_flat(valid_idx);
y_valid = y_flat(valid_idx);
z_valid = z_flat(valid_idx);

n_valid = length(x_valid);

if n_valid < 3
    trend_surface = zeros(size(data));
    detrend = data;
    params = [];
    stats = struct('r2', 0, 'rms_before', 0, 'rms_after', 0, 'improvement', 0);
    return;
end

% 构建设计矩阵
if strcmp(method, 'plane')
    % 一次曲面: z = a*x + b*y + c
    A = [x_valid, y_valid, ones(n_valid, 1)];
else  % quadratic
    % 二次曲面: z = a*x² + b*y² + c*xy + d*x + e*y + f
    A = [x_valid.^2, y_valid.^2, x_valid.*y_valid, ...
         x_valid, y_valid, ones(n_valid, 1)];
end

% 求解 (使用lscov或普通最小二乘)
if use_weights
    % 加权最小二乘: 数据质量越好权重越高
    % 这里简单使用数据绝对值的倒数作为权重
    weights = 1 ./ (abs(z_valid) + 1);  % 避免除零
    [params, ~, mse] = lscov(A, z_valid, weights);
    %[params, ~, mse] = lscov(A, z_valid);
else
    % 普通最小二乘
    params = A \ z_valid;
    mse = nan;
end

% 计算趋势面 (矢量化)
if strcmp(method, 'plane')
    trend_surface = params(1)*X + params(2)*Y + params(3);
else strcmp(method, 'plane')
    trend_surface = params(1)*X.^2 + params(2)*Y.^2 + params(3)*X.*Y + ...
                   params(4)*X + params(5)*Y + params(6);
end

% 计算残差
detrend = data - trend_surface;

% 统计 (使用矢量化操作)
observed = data(valid_idx);
predicted = trend_surface(valid_idx);

% R² 使用corrcoef快速计算
r = corrcoef(observed, predicted);
r2 = r(1,2)^2;

% 或传统方法
% tss = sum((observed - mean(observed)).^2);
% rss = sum((observed - predicted).^2);
% r2 = 1 - rss/tss;

% RMS
rms_before = sqrt(mean(observed.^2));
rms_after = sqrt(mean(detrend(valid_idx).^2));

stats = struct();
stats.method = method;
stats.r2 = r2;
stats.rms_before = rms_before;
stats.rms_after = rms_after;
stats.improvement = (1 - rms_after/rms_before) * 100;
stats.params = params;
stats.use_weights = use_weights;
if use_weights
    stats.mse = mse;
end

end
