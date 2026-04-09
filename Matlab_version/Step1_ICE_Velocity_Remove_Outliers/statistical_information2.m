function statistical_information2(data_dict, output_file)
% 统计信息输出 (支持文件保存)
%
% 输入:
%   data_dict   - 数据字典结构体
%   output_file - 输出文件路径 (可选)
%                 如果不提供，只输出到控制台
%                 如果提供，同时输出到控制台和文件
%
% 用法:
%   % 只输出到控制台
%   statistical_information(data_dict);
%
%   % 输出到控制台和文件
%   statistical_information(data_dict, 'statistics.txt');
%
%   % 自动命名
%   statistical_information(data_dict, 'auto');

% 处理输出文件参数
if nargin < 2
    output_file = '';  % 只输出到控制台
end

% 自动命名
if strcmp(output_file, 'auto')
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    output_file = sprintf('statistics_%s.txt', timestamp);
end

% 打开文件 (如果需要)
if ~isempty(output_file)
    fid = fopen(output_file, 'w');
    if fid == -1
        warning('无法打开文件 %s，只输出到控制台', output_file);
        fid = [];
    else
        fprintf('✓ 统计信息将保存到: %s\n\n', output_file);
    end
else
    fid = [];  % 只输出到控制台
end

% 开始输出
output_line(fid, '\n========================================');
output_line(fid, '=============== 统计信息报告 ===============');
output_line(fid, '========================================');
output_line(fid, sprintf('生成时间: %s', datestr(now, 'yyyy-mm-dd HH:MM:SS')));
output_line(fid, '========================================');

output_line(fid, '\n【1. RAW数据 - deformation from range/azimuth tif】');
print_detailed_stats_to_file(data_dict.raw, fid);
output_line(fid, '');

output_line(fid, '\n【2. NMT滤波 - NMT Filtered】');
print_detailed_stats_to_file(data_dict.filtered_nmt, fid);
output_line(fid, '');

output_line(fid, '\n【3. 平滑后数据 - Smoothed】');
print_detailed_stats_to_file(data_dict.smoothed_data, fid);
output_line(fid, '');

output_line(fid, '\n【4. 标准差图 - Std Map】');
print_detailed_stats_to_file(data_dict.std_map, fid);
output_line(fid, '');

output_line(fid, '\n========================================');
output_line(fid, '=============== 处理效果对比 ===============');
output_line(fid, '========================================');

output_line(fid, '\n【对比1: RAW → NMT】');
compare_before_after_to_file(data_dict.raw, data_dict.filtered_nmt, fid);
output_line(fid, '');

output_line(fid, '\n【对比2: NMT → Smooth】');
compare_before_after_to_file(data_dict.filtered_nmt, data_dict.smoothed_data, fid);
output_line(fid, '');

output_line(fid, '\n【对比3: RAW → Smooth (总体效果)】');
compare_before_after_to_file(data_dict.raw, data_dict.smoothed_data, fid);
output_line(fid, '');

output_line(fid, '========================================');
output_line(fid, '=============== 报告结束 ===============');
output_line(fid, '========================================');

% 关闭文件
if ~isempty(fid)
    fclose(fid);
    fprintf('\n✓ 统计信息已保存到: %s\n', output_file);
end

end


%% ========================================================================
%% 核心函数: 输出一行文本 (同时到控制台和文件)
%% ========================================================================

function output_line(fid, text)
% 输出一行文本到控制台和文件
%
% 输入:
%   fid  - 文件标识符 (如果为空，只输出到控制台)
%   text - 要输出的文本

% 输出到控制台
fprintf('%s\n', text);

% 输出到文件
if ~isempty(fid)
    fprintf(fid, '%s\n', text);
end

end


%% ========================================================================
%% 详细统计 (输出到文件版本)
%% ========================================================================

function print_detailed_stats_to_file(data, fid)
% 打印详细统计信息到文件

valid_data = data(~isnan(data) & ~isinf(data));

if isempty(valid_data)
    output_line(fid, '  ⚠ 无有效数据');
    return;
end

n = length(valid_data);
mean_val = mean(valid_data);
median_val = median(valid_data);
std_val = std(valid_data);
min_val = min(valid_data);
max_val = max(valid_data);

q25 = prctile(valid_data, 25);
q50 = prctile(valid_data, 50);
q75 = prctile(valid_data, 75);

iqr = q75 - q25;

output_line(fid, '  ┌─ 基本统计 ─────────────────────');
output_line(fid, sprintf('  │ 有效像素: %d', n));
output_line(fid, sprintf('  │ 均值 (μ): %.6f', mean_val));
output_line(fid, sprintf('  │ 中值: %.6f', median_val));
output_line(fid, sprintf('  │ 标准差 (σ): %.6f', std_val));
output_line(fid, sprintf('  │ 最小值: %.6f', min_val));
output_line(fid, sprintf('  │ 最大值: %.6f', max_val));
output_line(fid, sprintf('  │ 范围: %.6f', max_val - min_val));
output_line(fid, '  └────────────────────────────────');

output_line(fid, '  ┌─ 分位数 ───────────────────────');
output_line(fid, sprintf('  │ Q1 (25%%): %.6f', q25));
output_line(fid, sprintf('  │ Q2 (50%%): %.6f', q50));
output_line(fid, sprintf('  │ Q3 (75%%): %.6f', q75));
output_line(fid, sprintf('  │ IQR (Q3-Q1): %.6f', iqr));
output_line(fid, '  └────────────────────────────────');

output_line(fid, '  ┌─ 置信区间 ─────────────────────');
output_line(fid, sprintf('  │ μ ± 1σ: [%.6f, %.6f] (68.3%%)', ...
                        mean_val - std_val, mean_val + std_val));
output_line(fid, sprintf('  │ μ ± 2σ: [%.6f, %.6f] (95.4%%)', ...
                        mean_val - 2*std_val, mean_val + 2*std_val));
output_line(fid, sprintf('  │ μ ± 3σ: [%.6f, %.6f] (99.7%%)', ...
                        mean_val - 3*std_val, mean_val + 3*std_val));
output_line(fid, '  └────────────────────────────────');

end


%% ========================================================================
%% 对比分析 (输出到文件版本)
%% ========================================================================

function compare_before_after_to_file(data_before, data_after, fid)
% 对比平滑前后的变化并输出到文件

valid_before = data_before(~isnan(data_before) & ~isinf(data_before));
valid_after = data_after(~isnan(data_after) & ~isinf(data_after));

if isempty(valid_before) || isempty(valid_after)
    output_line(fid, '  ⚠ 无法对比 (数据不足)');
    return;
end

% 统计量变化
mean_before = mean(valid_before);
mean_after = mean(valid_after);
std_before = std(valid_before);
std_after = std(valid_after);

mean_change = abs(mean_after - mean_before);
mean_change_pct = abs(mean_after - mean_before) / abs(mean_before) * 100;
std_change = abs(std_after - std_before);
std_reduction = (1 - std_after/std_before) * 100;

output_line(fid, '  ┌─ 统计量变化 ───────────────────');
output_line(fid, sprintf('  │ 均值: %.6f → %.6f', mean_before, mean_after));
output_line(fid, sprintf('  │   变化: Δ = %.6f (%.2f%%)', mean_change, mean_change_pct));
output_line(fid, '  │');
output_line(fid, sprintf('  │ 标准差: %.6f → %.6f', std_before, std_after));
output_line(fid, sprintf('  │   变化: Δ = %.6f (%.2f%%)', std_change, abs(std_reduction)));
if std_reduction > 0
    output_line(fid, sprintf('  │   ✓ 降低了 %.2f%%', std_reduction));
else
    output_line(fid, sprintf('  │   ⚠ 增加了 %.2f%%', -std_reduction));
end
output_line(fid, '  └────────────────────────────────');

% 有效像素变化
n_before = length(valid_before);
n_after = length(valid_after);
n_change = n_after - n_before;
n_change_pct = n_change / n_before * 100;

output_line(fid, '  ┌─ 有效像素变化 ─────────────────');
output_line(fid, sprintf('  │ 数量: %d → %d', n_before, n_after));
output_line(fid, sprintf('  │ 变化: Δ = %d (%.2f%%)', n_change, n_change_pct));
output_line(fid, '  └────────────────────────────────');

% RMS
rms_before = sqrt(mean(valid_before.^2));
rms_after = sqrt(mean(valid_after.^2));
rms_change = (1 - rms_after/rms_before) * 100;

output_line(fid, '  ┌─ RMS 分析 ─────────────────────');
output_line(fid, sprintf('  │ RMS: %.6f → %.6f', rms_before, rms_after));
if rms_change > 0
    output_line(fid, sprintf('  │ ✓ 降低了 %.2f%%', rms_change));
else
    output_line(fid, sprintf('  │ ⚠ 增加了 %.2f%%', -rms_change));
end
output_line(fid, '  └────────────────────────────────');

% 相关性 (如果有重叠数据)
both_valid = ~isnan(data_before) & ~isnan(data_after);
if sum(both_valid(:)) > 100
    corr_val = corr(data_before(both_valid), data_after(both_valid));
    output_line(fid, '  ┌─ 相关性分析 ───────────────────');
    output_line(fid, sprintf('  │ 相关系数: %.6f', corr_val));
    if corr_val > 0.95
        output_line(fid, '  │ ✓ 高度相关 (数据一致性好)');
    elseif corr_val > 0.80
        output_line(fid, '  │ ○ 中度相关');
    else
        output_line(fid, '  │ ⚠ 低相关 (可能过度处理)');
    end
    output_line(fid, '  └────────────────────────────────');
end

end