% =========================================================================
%  Mertz/Cook冰架 Sentinel-1 POT 时序加权平均（V/Vx/Vy 独立计算版）
%
%  与上一版区别：V 不由 Vx Vy 合成，而是直接对 V 文件独立加权平均
%  输出：Vx/Vy/V 各自的最终速度 + 各自的不确定性 + 各自的 neff
%
%  文件命名格式: yyyyMMdd-yyyyMMdd-V.tif / -Vx.tif / -Vy.tif
%  依赖函数: weighted_average_m2.m, local_std.m
% =========================================================================
clear; clc;

%% ========== 用户参数（只需修改此区域）==========
data_dir   = '/data2/Phd_Work1/ICE_Velocity_Process/V_test_2017';
output_dir = '/data2/Phd_Work1/ICE_Velocity_Process/V_test_2017/Weighted_Average_IndepV';

has_sigma   = false;    % true=有sigma文件 / false=用local_std估计
window_size = 5;        % local_std窗口大小
min_valid   = 4;        % local_std最小有效像素数
ignore_nan  = true;
no_data_val = 0;        % tif中无效值标识

% 选择时间模式
% mode = 'yearly';
mode = 'seasonal';
% mode = 'yearly';
% mode = 'fixed_6';
% mode = 'fixed_12';
% mode = 'fixed_18';
% mode = 'fixed_30';
% mode = 'fixed_60';
%% ===============================================

mkdir(output_dir);

%% 1. 查找所有 Vx 文件并解析日期（以Vx为索引查找对应V/Vy）
vx_files = dir(fullfile(data_dir, '*-Vx.tif'));
[~, idx] = sort({vx_files.name});
vx_files = vx_files(idx);
N = length(vx_files);
fprintf('共找到 %d 个影像对\n', N);

% 解析 date1 / date2
dates     = NaT(N, 1);
date1_str = cell(N, 1);
date2_str = cell(N, 1);
for i = 1:N
    fname = vx_files(i).name;   % 20160730-20160811-Vx.tif
    date1_str{i} = fname(1:8);
    date2_str{i} = fname(10:17);
    dates(i) = datetime(fname(1:8), 'InputFormat', 'yyyyMMdd');
end

%% 2. 读取空间参考
[tmp, R] = readgeoraster(fullfile(data_dir, vx_files(1).name));
[rows, cols] = size(tmp);
clear tmp;
epsg = 3031;   % 南极极地立体投影 WGS84 EPSG:3031
fprintf('图像尺寸: %d × %d\n\n', rows, cols);

%% 3. 生成分组标签
group_labels  = assign_groups(dates, mode);      % each tif have a group_labels
unique_groups = unique(group_labels, 'stable');  % keep time-series using stable
G = length(unique_groups);
fprintf('时间模式: [%s]，共 %d 组\n\n', mode, G);

%% 4. 写分组日志 txt
log_file = fullfile(output_dir, sprintf('group_info_%s.txt', mode));
fid = fopen(log_file, 'w');
fprintf(fid, '时间模式: %s\n', mode);
fprintf(fid, '数据目录: %s\n', data_dir);
fprintf(fid, '总影像对数: %d\n', N);
fprintf(fid, '总分组数: %d\n', G);
fprintf(fid, '生成时间: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf(fid, '%s\n\n', repmat('=', 1, 70));
for g = 1:G
    gname   = unique_groups{g};  % eg: '20200315'
    in_grp  = strcmp(group_labels, gname);  % reture 0 or 1
    n_grp   = sum(in_grp);
    grp_idx = find(in_grp);  % location index in dates
    fprintf(fid, '【组 %d/%d】%s（共 %d 个影像对）\n', g, G, gname, n_grp);
    for k = 1:n_grp
        ii = grp_idx(k);
        fprintf(fid, '  %2d. %s-%s  [%s → %s]\n', ...
            k, date1_str{ii}, date2_str{ii}, date1_str{ii}, date2_str{ii});
    end
    fprintf(fid, '\n');
end
fclose(fid);
fprintf('分组日志已写入: %s\n\n', log_file);

%% 5. 定义三个分量（循环处理，结构完全一致）
components = {'Vx', 'Vy', 'V'};   % 三个分量独立处理

%% 6. 逐组 × 逐分量 加权平均
fprintf('开始处理...\n');
t_total = tic;

for g = 1:G
    gname   = unique_groups{g};
    in_grp  = strcmp(group_labels, gname);
    n_grp   = sum(in_grp);
    grp_idx = find(in_grp);

    out_sub = fullfile(output_dir, gname);
    mkdir(out_sub);

    fprintf('[%3d/%3d] 组 %-18s: %d 个影像对\n', g, G, gname, n_grp);

    for c = 1:length(components)
        comp = components{c};   % 'Vx' / 'Vy' / 'V'
        t_comp = tic;

        %% 6a. 读取该分量的所有影像 + sigma
        V_stack = nan(rows, cols, n_grp, 'single');
        S_stack = nan(rows, cols, n_grp, 'single');

        for k = 1:n_grp
            ii = grp_idx(k);

            % 速度文件名：把 -Vx.tif 替换为对应分量
            v_name = strrep(vx_files(ii).name, '-Vx.tif', ['-' comp '.tif']);
            v = single(readgeoraster(fullfile(data_dir, v_name)));
            v(v == no_data_val | isinf(v)) = NaN;

            if has_sigma
                s_name = strrep(vx_files(ii).name, '-Vx.tif', ['-' comp '_std.tif']);
                s = single(readgeoraster(fullfile(data_dir, s_name)));
                s(s <= 0 | isinf(s)) = NaN;
            else
                s = local_std(v, window_size, ignore_nan, min_valid);
            end

            % 联合掩膜：速度或sigma任一无效则都置NaN
            bad = isnan(v) | isnan(s) | s < 1e-8;
            v(bad) = NaN;
            s(bad) = NaN;

            V_stack(:,:,k) = v;
            S_stack(:,:,k) = s;
        end

        %% 6b. 加权平均
        if n_grp == 1
            V_final    = V_stack(:,:,1);
            sigma_final= S_stack(:,:,1);
            neff_map   = ones(rows, cols, 'single');
            neff_map(isnan(V_final)) = NaN;
        else
            [V_final, sigma_final, neff_map] = weighted_average_m2(V_stack, S_stack);
        end

        %% 6c. 保存（文件名含组名+分量）
        geotiffwrite(fullfile(out_sub, [gname '-' comp '.tif']),       single(V_final),     R, 'CoordRefSysCode', epsg);
        geotiffwrite(fullfile(out_sub, [gname '-sigma_' comp '.tif']), single(sigma_final), R, 'CoordRefSysCode', epsg);
        geotiffwrite(fullfile(out_sub, [gname '-neff_'  comp '.tif']), single(neff_map),    R, 'CoordRefSysCode', epsg);

        fprintf('       %s 完成 (%.1f 秒)\n', comp, toc(t_comp));
    end
end

fprintf('\n=== 全部完成，总耗时 %.1f 秒 ===\n', toc(t_total));
fprintf('输出目录: %s\n', output_dir);
fprintf('分组日志: %s\n', log_file);
fprintf('\n输出文件说明:\n');
fprintf('  每个组文件夹内包含 9 个文件:\n');
fprintf('  组名-Vx.tif        组名-Vy.tif        组名-V.tif\n');
fprintf('  组名-sigma_Vx.tif  组名-sigma_Vy.tif  组名-sigma_V.tif\n');
fprintf('  组名-neff_Vx.tif   组名-neff_Vy.tif   组名-neff_V.tif\n');


% =========================================================================
%  分组函数（南极四季版）
% =========================================================================
function labels = assign_groups(dates, mode)
    N = length(dates);
    labels = cell(N, 1);

    switch mode

        case 'monthly'
            % 自然月，date1 落在哪个月就归哪个月
            for i = 1:N
                labels{i} = datestr(dates(i), 'yyyy-mm');
            end

        case 'seasonal'
            % 南极四季（与北半球相反）：
            %   Summer(夏): 12, 1, 2  月
            %   Autumn(秋):  3, 4, 5  月
            %   Winter(冬):  6, 7, 8  月
            %   Spring(春):  9,10,11  月
            % 夏季跨年处理：12月归当年，1-2月归上一年
            for i = 1:N
                m = month(dates(i));
                y = year(dates(i));
                if m == 12
                    labels{i} = sprintf('%d-%d-Summer', y, y+1);
                elseif m == 1 || m == 2
                    labels{i} = sprintf('%d-%d-Summer', y-1, y);
                elseif m >= 3 && m <= 5
                    labels{i} = sprintf('%d-Autumn', y);
                elseif m >= 6 && m <= 8
                    labels{i} = sprintf('%d-Winter', y);
                else   % 9, 10, 11
                    labels{i} = sprintf('%d-Spring', y);
                end
            end

        case 'yearly'
            % 自然年：date1 落在哪一年就归哪一年（1月～12月全部纳入）
            for i = 1:N
                labels{i} = sprintf('%d', year(dates(i)));
            end

        otherwise
            % fixed_N 天固定窗口
            win_days = str2double(strrep(mode, 'fixed_', ''));
            if isnan(win_days) || win_days <= 0
                error('未知模式: %s\n可选: monthly/seasonal/yearly/fixed_6/fixed_12/fixed_18/fixed_30/fixed_60', mode);
            end
            t0 = min(dates);
            for i = 1:N
                bin       = floor(days(dates(i) - t0) / win_days);
                bin_start = t0 + days(bin * win_days);
                labels{i} = datestr(bin_start, 'yyyy-mm-dd');
            end
    end
end
