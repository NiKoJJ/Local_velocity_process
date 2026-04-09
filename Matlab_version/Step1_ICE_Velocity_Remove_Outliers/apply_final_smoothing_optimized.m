function [smoothed_data, std_map, stats] = apply_final_smoothing_optimized(data, window_size, options)
% 最终平滑和误差估计 (优化版)
% local_std local_mean local_median
% 使用MATLAB内置函数加速:
%   - stdfilt: 局部标准差滤波
%   - imboxfilt: 盒式滤波器 (快速平均)
%   - medfilt2: 

%% 参数处理
if nargin < 2 || isempty(window_size)
    window_size = 5;
end

if nargin < 3
    options = struct();
end

if ~isfield(options, 'ignore_nan'), options.ignore_nan = true; end
if ~isfield(options, 'min_valid'), options.min_valid = 15; end
if ~isfield(options, 'method'), options.method = 'mean'; end

fprintf('最终平滑和误差估计 (优化版)\n');
fprintf('  窗口: %d×%d, 方法: %s\n', window_size, window_size, options.method);

%% 步骤1: 计算局部标准差 (使用stdfilt或自定义)
fprintf('  步骤1: 计算局部标准差...\n');

fprintf('using local_std function to smooth: \n')
std_map = local_std(data, window_size, options.ignore_nan, options.min_valid);
% std_map = stdfilt(data, ones(window_size));

% if exist('stdfilt', 'file')
%     fprintf('using stdfilt function to smooth: \n')
%     std_map = stdfilt(data, ones(window_size));
% 
% else
%     fprintf('using local_std function to smooth: \n')
%     std_map = local_std(data, window_size, options.ignore_nan, options.min_valid);
% end

% 统计标准差
valid_std = std_map(~isnan(std_map) & ~isinf(std_map));
if ~isempty(valid_std)
    mean_std = mean(valid_std);
    median_std = median(valid_std);
    max_std = max(valid_std);
    
    fprintf('    平均标准差: %.4f\n', mean_std);
    fprintf('    中值标准差: %.4f\n', median_std);
    fprintf('    最大标准差: %.4f\n', max_std);
else
    mean_std = NaN;
    median_std = NaN;
    max_std = NaN;
end

%% 步骤2: 最终平滑
fprintf('  步骤2: 最终平滑...\n');

if strcmp(options.method, 'mean')
    fprintf('using mean method to smooth: \n')
    % 平均滤波
    fprintf('using local_mean function to smooth: \n')
        smoothed_data = local_mean(data, window_size, options.ignore_nan, options.min_valid);

    % if exist('imboxfilt', 'file')
    %     fprintf('using imboxfilt function to smooth: \n')
    %     smoothed_data = imboxfilt(data, window_size);
    % else
    %     fprintf('using local_mean function to smooth: \n')
    %     smoothed_data = local_mean(data, window_size, options.ignore_nan, options.min_valid);
    % end
    
elseif strcmp(options.method, 'median')
    fprintf('using median method to smooth: \n')
    % 中值滤波
    fprintf('  using local_median function to smooth: \n')
    smoothed_data = local_median(data, window_size, options.min_valid);
    % smoothed_data = medfilt2(data, [window_size, window_size], 'symmetric');
end

% 统计
n_valid_original = sum(~isnan(data(:)) & ~isinf(data(:)));
n_valid_smoothed = sum(~isnan(smoothed_data(:)) & ~isinf(smoothed_data(:)));

fprintf('    原始有效: %d, 平滑后: %d\n', n_valid_original, n_valid_smoothed);

% RMS变化
valid_both = ~isnan(data) & ~isnan(smoothed_data);
if sum(valid_both(:)) > 0
    rms_original = sqrt(mean(data(valid_both).^2));
    rms_smoothed = sqrt(mean(smoothed_data(valid_both).^2));
    rms_change = abs(rms_smoothed - rms_original) / rms_original * 100;
    
    fprintf('    RMS: %.4f → %.4f (%.2f%%)\n', rms_original, rms_smoothed, rms_change);
else
    rms_original = NaN;
    rms_smoothed = NaN;
    rms_change = NaN;
end

%% 统计
stats = struct();
stats.window_size = window_size;
stats.method = options.method;
stats.n_valid_original = n_valid_original;
stats.n_valid_smoothed = n_valid_smoothed;
stats.mean_std = mean_std;
stats.median_std = median_std;
stats.max_std = max_std;
stats.rms_original = rms_original;
stats.rms_smoothed = rms_smoothed;
stats.rms_change = rms_change;

fprintf('✓ 完成\n');

end