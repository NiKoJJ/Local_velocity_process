% =========================================================================
%  预处理：统一文件夹下所有 V/Vx/Vy tif 文件的维度
%
%  策略：以文件夹中出现最多的尺寸为目标尺寸（众数尺寸）
%        对不一致的文件用 imresize 裁剪/插值到目标尺寸
%        原文件备份到 original_backup/ 子目录
%
%  运行完成后再执行主程序 timeseries_weighted_avg_indep.m
% =========================================================================
clear; clc;

%% ========== 用户参数 ==========
% data_dir   = '/data2/Phd_Work1/ICE_Velocity_Process/GAMMA_POT_Output2/Cook_80m/Tidal_IBE_Flexure_Correction/Vx_Vy_V';
data_dir   = '/data2/Phd_Work1/ICE_Velocity_Process/GAMMA_POT_Output/Cook_120m/Remove_outliers_Correction/velocity';
no_data_val = 0;

% 目标尺寸策略:
%   'mode'  → 使用出现最多的尺寸（推荐）
%   'first' → 使用第一个文件的尺寸
%   'manual'→ 手动指定（需填写 target_rows / target_cols）
size_strategy = 'mode';
target_rows = [];   % 仅 manual 模式下填写，如 2989
target_cols = [];   % 仅 manual 模式下填写，如 2410
%% ================================

%% 1. 找到所有 tif 文件（V / Vx / Vy）
all_files = [dir(fullfile(data_dir, '*-V.tif'));
             dir(fullfile(data_dir, '*-Vx.tif'));
             dir(fullfile(data_dir, '*-Vy.tif'))];
[~, idx] = sort({all_files.name});
all_files = all_files(idx);
Nf = length(all_files);
fprintf('共找到 %d 个 tif 文件\n', Nf);

%% 2. 读取所有文件的尺寸
fprintf('正在扫描所有文件尺寸...\n');
file_rows = zeros(Nf, 1);
file_cols = zeros(Nf, 1);
R_cell    = cell(Nf, 1);   % 保存空间参考

for i = 1:Nf
    fpath = fullfile(data_dir, all_files(i).name);
    info  = georasterinfo(fpath);
    file_rows(i) = info.RasterSize(1);
    file_cols(i) = info.RasterSize(2);
    if i == 1 || (file_rows(i) == file_rows(1) && file_cols(i) == file_cols(1))
        % 只读第一个和与第一个不同的文件的空间参考（节省时间）
    end
    if mod(i, 50) == 0 || i == Nf
        fprintf('  已扫描 %d/%d\n', i, Nf);
    end
end

%% 3. 统计尺寸分布
size_pairs = [file_rows, file_cols];
[unique_sizes, ~, ic] = unique(size_pairs, 'rows');
counts = accumarray(ic, 1);

fprintf('\n---- 尺寸统计 ----\n');
for s = 1:size(unique_sizes, 1)
    fprintf('  %d × %d : %d 个文件\n', unique_sizes(s,1), unique_sizes(s,2), counts(s));
end

%% 4. 确定目标尺寸
switch size_strategy
    case 'mode'
        [~, max_idx] = max(counts);
        target_rows = unique_sizes(max_idx, 1);
        target_cols = unique_sizes(max_idx, 2);
        fprintf('\n目标尺寸（众数）: %d × %d（共 %d 个文件符合）\n', ...
            target_rows, target_cols, counts(max_idx));
    case 'first'
        target_rows = file_rows(1);
        target_cols = file_cols(1);
        fprintf('\n目标尺寸（第一个文件）: %d × %d\n', target_rows, target_cols);
    case 'manual'
        assert(~isempty(target_rows) && ~isempty(target_cols), ...
            '请在参数区填写 target_rows 和 target_cols');
        fprintf('\n目标尺寸（手动指定）: %d × %d\n', target_rows, target_cols);
end

%% 5. 找出需要调整的文件
need_fix = (file_rows ~= target_rows) | (file_cols ~= target_cols);
n_fix = sum(need_fix);
fprintf('\n需要调整的文件数: %d\n', n_fix);

if n_fix == 0
    fprintf('所有文件尺寸一致，无需处理。\n');
    return;
end

%% 6. 读取一个目标尺寸文件的空间参考（用于统一写出）
ref_idx = find(~need_fix, 1, 'first');
[~, R_target] = readgeoraster(fullfile(data_dir, all_files(ref_idx).name));
epsg = 3031;   % 南极极地立体投影 WGS84

%% 7. 备份 + 修正
backup_dir = fullfile(data_dir, 'original_backup');
mkdir(backup_dir);
fprintf('\n原始文件将备份到: %s\n', backup_dir);
fprintf('开始修正...\n\n');

fix_log = fullfile(data_dir, 'resize_log.txt');
fid = fopen(fix_log, 'w');
fprintf(fid, '目标尺寸: %d × %d\n', target_rows, target_cols);
fprintf(fid, '处理时间: %s\n', datestr(now));
fprintf(fid, '%s\n', repmat('-', 1, 60));

fix_list = find(need_fix);
for k = 1:length(fix_list)
    i     = fix_list(k);
    fname = all_files(i).name;
    fpath = fullfile(data_dir, fname);

    orig_r = file_rows(i);
    orig_c = file_cols(i);

    fprintf('[%3d/%3d] %-40s  %d×%d → %d×%d\n', ...
        k, n_fix, fname, orig_r, orig_c, target_rows, target_cols);

    % 备份原始文件
    copyfile(fpath, fullfile(backup_dir, fname));

    % 读取原始数据
    data = single(readgeoraster(fpath));

    % 处理无效值：先记录NaN位置，resize后恢复
    nan_mask = isnan(data) | (data == no_data_val);

    % resize（对于速度图用双线性插值）
    data_fixed = imresize(data, [target_rows, target_cols], 'bilinear');

    % 对应位置的NaN也要resize（用最近邻）
    nan_resized = imresize(single(nan_mask), [target_rows, target_cols], 'nearest');
    data_fixed(nan_resized > 0.5) = NaN;

    % 覆盖写回（使用目标空间参考，指定 EPSG:3031 南极极地立体投影）
    geotiffwrite(fpath, data_fixed, R_target, 'CoordRefSysCode', epsg);

    % 写日志
    fprintf(fid, '%s: %d×%d → %d×%d\n', fname, orig_r, orig_c, target_rows, target_cols);
end

fclose(fid);
fprintf('\n=== 完成 ===\n');
fprintf('修正文件数: %d\n', n_fix);
fprintf('原始备份位置: %s\n', backup_dir);
fprintf('修正日志: %s\n', fix_log);
fprintf('\n现在可以运行主程序 timeseries_weighted_avg_v2_final.m\n');
