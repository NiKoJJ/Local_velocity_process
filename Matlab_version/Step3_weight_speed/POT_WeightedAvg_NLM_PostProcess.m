% =========================================================================
%  方案A：先加权平均，再对结果做 NLM 滤波（后处理）
%
%  处理流程:
%    读取 → 加权平均 → NLM滤波 → 保存
%
%  每组输出 12 个文件:
%    原始加权平均: Vx / Vy / V / sigma_Vx / sigma_Vy / sigma_V / neff_Vx / neff_Vy / neff_V
%    NLM后处理:   Vx_NLM / Vy_NLM / V_NLM
%
%  文件命名格式: yyyyMMdd-yyyyMMdd-Vx.tif
%  南极四季定义（与北半球相反）:
%    夏季(Summer): 12,1,2月
%    秋季(Autumn): 3,4,5月
%    冬季(Winter): 6,7,8月
%    春季(Spring): 9,10,11月
%
%  依赖函数: weighted_average_m2.m / local_std.m / apply_nlm.m
% =========================================================================
clear; clc;

%% ========== 用户参数（只需修改此区域）==========
% data_dir   = '/data2/Phd_Work1/ICE_Velocity_Process/GAMMA_POT_Output2/Mertz_100m/Tidal_IBE_Flexure_Correction/Vx_Vy_V';
% output_dir = '/data2/Phd_Work1/ICE_Velocity_Process/GAMMA_POT_Output2/Mertz_100m/Tidal_IBE_Flexure_Correction/Vx_Vy_V_Weighted_Average/yearly';

data_dir   = '/data2/Phd_Work1/ICE_Velocity_Process/GAMMA_POT_Output2/Mertz_100m/Tidal_IBE_Flexure_Correction/Vx_Vy_V';
output_dir = '/data2/Phd_Work1/ICE_Velocity_Process/GAMMA_POT_Output2/Mertz_100m/Tidal_IBE_Flexure_Correction/Weighted_Average_PostNLM/yearly';

% --- 加权平均参数 ---
has_sigma   = false;
window_size = 5;
min_valid   = 4;
ignore_nan  = true;
no_data_val = 0;
epsg        = 3031;   % 南极极地立体投影

% --- NLM 参数 ---
nlm_h        = 10;   % 滤波强度（DegreeOfSmoothing），与噪声水平正相关
nlm_temp_win = 7;    % Patch 大小（ComparisonWindowSize，奇数）
nlm_srch_win = 21;   % 搜索窗口大小（SearchWindowSize，奇数）

% --- 时间分组模式 ---
% mode = 'monthly';
% mode = 'seasonal';
mode = 'yearly';
% mode = 'fixed_6';
% mode = 'fixed_12';
% mode = 'fixed_18';
% mode = 'fixed_30';
% mode = 'fixed_60';
%% ===============================================

mkdir(output_dir);

%% 1. 查找所有 Vx 文件并解析日期
vx_files = dir(fullfile(data_dir, '*-Vx.tif'));
[~, idx] = sort({vx_files.name});
vx_files = vx_files(idx);
N = length(vx_files);
fprintf('共找到 %d 个影像对\n', N);

dates     = NaT(N, 1);
date1_str = cell(N, 1);
date2_str = cell(N, 1);
for i = 1:N
    fname = vx_files(i).name;
    date1_str{i} = fname(1:8);
    date2_str{i} = fname(10:17);
    dates(i) = datetime(fname(1:8), 'InputFormat', 'yyyyMMdd');
end

%% 2. 读取空间参考
[tmp, R] = readgeoraster(fullfile(data_dir, vx_files(1).name));
[rows, cols] = size(tmp);
clear tmp;
fprintf('图像尺寸: %d × %d\n\n', rows, cols);

%% 3. 生成分组标签
group_labels  = assign_groups(dates, mode);
unique_groups = unique(group_labels, 'stable');
G = length(unique_groups);
fprintf('时间模式: [%s]，共 %d 组\n', mode, G);
fprintf('NLM参数: h=%.1f, patch=%d×%d, 搜索=%d×%d\n\n', ...
    nlm_h, nlm_temp_win, nlm_temp_win, nlm_srch_win, nlm_srch_win);

%% 4. 写分组日志 txt
log_file = fullfile(output_dir, sprintf('group_info_%s_PostNLM.txt', mode));
fid = fopen(log_file, 'w');
fprintf(fid, '方案: A（先加权平均，后 NLM 滤波）\n');
fprintf(fid, '时间模式: %s\n', mode);
fprintf(fid, '数据目录: %s\n', data_dir);
fprintf(fid, '总影像对数: %d\n', N);
fprintf(fid, '总分组数: %d\n', G);
fprintf(fid, 'NLM参数: h=%.1f, patch=%d, 搜索窗口=%d\n', nlm_h, nlm_temp_win, nlm_srch_win);
fprintf(fid, '生成时间: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf(fid, '%s\n\n', repmat('=', 1, 70));
for g = 1:G
    gname   = unique_groups{g};
    in_grp  = strcmp(group_labels, gname);
    n_grp   = sum(in_grp);
    grp_idx = find(in_grp);
    fprintf(fid, '【组 %d/%d】%s（共 %d 个影像对）\n', g, G, gname, n_grp);
    for k = 1:n_grp
        ii = grp_idx(k);
        fprintf(fid, '  %2d. %s-%s\n', k, date1_str{ii}, date2_str{ii});
    end
    fprintf(fid, '\n');
end
fclose(fid);
fprintf('分组日志已写入: %s\n\n', log_file);

%% 5. 逐组处理
fprintf('开始处理...\n');
t_total = tic;

for g = 1:G
    gname   = unique_groups{g};
    in_grp  = strcmp(group_labels, gname);
    n_grp   = sum(in_grp);
    grp_idx = find(in_grp);

    fprintf('[%3d/%3d] 组 %-18s: %d 个影像对\n', g, G, gname, n_grp);

    out_sub = fullfile(output_dir, gname);
    mkdir(out_sub);

    %% 5a. 初始化三个分量的堆栈
    Vx_stack  = nan(rows, cols, n_grp, 'single');
    Vy_stack  = nan(rows, cols, n_grp, 'single');
    V_stack   = nan(rows, cols, n_grp, 'single');
    sVx_stack = nan(rows, cols, n_grp, 'single');
    sVy_stack = nan(rows, cols, n_grp, 'single');
    sV_stack  = nan(rows, cols, n_grp, 'single');

    %% 5b. 读取该组所有影像
    for k = 1:n_grp
        ii = grp_idx(k);
        vx_name = vx_files(ii).name;
        vy_name = strrep(vx_name, '-Vx.tif', '-Vy.tif');
        v_name  = strrep(vx_name, '-Vx.tif', '-V.tif');

        % --- 读取速度 ---
        vx = single(readgeoraster(fullfile(data_dir, vx_name)));
        vy = single(readgeoraster(fullfile(data_dir, vy_name)));
        v  = single(readgeoraster(fullfile(data_dir, v_name)));

        vx(vx == no_data_val | isinf(vx)) = NaN;
        vy(vy == no_data_val | isinf(vy)) = NaN;
        v (v  == no_data_val | isinf(v))  = NaN;

        % --- 获取 sigma ---
        if has_sigma
            svx_name = strrep(vx_name, '-Vx.tif', '-Vx_std.tif');
            svy_name = strrep(vx_name, '-Vx.tif', '-Vy_std.tif');
            sv_name  = strrep(vx_name, '-Vx.tif', '-V_std.tif');
            svx = single(readgeoraster(fullfile(data_dir, svx_name)));
            svy = single(readgeoraster(fullfile(data_dir, svy_name)));
            sv  = single(readgeoraster(fullfile(data_dir, sv_name)));
            svx(svx <= 0 | isinf(svx)) = NaN;
            svy(svy <= 0 | isinf(svy)) = NaN;
            sv (sv  <= 0 | isinf(sv))  = NaN;
        else
            svx = local_std(vx, window_size, ignore_nan, min_valid);
            svy = local_std(vy, window_size, ignore_nan, min_valid);
            sv  = local_std(v,  window_size, ignore_nan, min_valid);
        end

        % --- 联合掩膜 ---
        bad_x = isnan(vx) | isnan(svx) | svx < 1e-8;
        bad_y = isnan(vy) | isnan(svy) | svy < 1e-8;
        bad_v = isnan(v)  | isnan(sv)  | sv  < 1e-8;
        vx(bad_x) = NaN;  svx(bad_x) = NaN;
        vy(bad_y) = NaN;  svy(bad_y) = NaN;
        v (bad_v) = NaN;  sv (bad_v) = NaN;

        Vx_stack(:,:,k)  = vx;
        Vy_stack(:,:,k)  = vy;
        V_stack(:,:,k)   = v;
        sVx_stack(:,:,k) = svx;
        sVy_stack(:,:,k) = svy;
        sV_stack(:,:,k)  = sv;
    end

    %% 5c. 加权平均（三个分量独立）
    t_wa = tic;
    if n_grp == 1
        Vx_final  = Vx_stack(:,:,1);
        Vy_final  = Vy_stack(:,:,1);
        V_final   = V_stack(:,:,1);
        sVx_final = sVx_stack(:,:,1);
        sVy_final = sVy_stack(:,:,1);
        sV_final  = sV_stack(:,:,1);
        neff_Vx   = ones(rows, cols, 'single');  neff_Vx(isnan(Vx_final)) = NaN;
        neff_Vy   = ones(rows, cols, 'single');  neff_Vy(isnan(Vy_final)) = NaN;
        neff_V    = ones(rows, cols, 'single');  neff_V (isnan(V_final))  = NaN;
    else
        [Vx_final, sVx_final, neff_Vx] = weighted_average_m2(Vx_stack, sVx_stack);
        [Vy_final, sVy_final, neff_Vy] = weighted_average_m2(Vy_stack, sVy_stack);
        [V_final,  sV_final,  neff_V ] = weighted_average_m2(V_stack,  sV_stack);
    end
    fprintf('       加权平均完成 (%.1f s)\n', toc(t_wa));

    %% 5d. 【方案A核心】对加权平均结果做 NLM 后处理
    t_nlm = tic;
    Vx_nlm = apply_nlm(Vx_final, nlm_h, nlm_temp_win, nlm_srch_win);
    Vy_nlm = apply_nlm(Vy_final, nlm_h, nlm_temp_win, nlm_srch_win);
    V_nlm  = apply_nlm(V_final,  nlm_h, nlm_temp_win, nlm_srch_win);
    fprintf('       NLM 后处理完成 (%.1f s)\n', toc(t_nlm));

    %% 5e. 保存（原始加权平均 9 个 + NLM后处理 3 个，共 12 个文件）
    % -- 原始加权平均结果 --
    geotiffwrite(fullfile(out_sub, [gname '-Vx.tif']),       single(Vx_final),  R, 'CoordRefSysCode', epsg);
    geotiffwrite(fullfile(out_sub, [gname '-Vy.tif']),       single(Vy_final),  R, 'CoordRefSysCode', epsg);
    geotiffwrite(fullfile(out_sub, [gname '-V.tif']),        single(V_final),   R, 'CoordRefSysCode', epsg);
    geotiffwrite(fullfile(out_sub, [gname '-sigma_Vx.tif']), single(sVx_final), R, 'CoordRefSysCode', epsg);
    geotiffwrite(fullfile(out_sub, [gname '-sigma_Vy.tif']), single(sVy_final), R, 'CoordRefSysCode', epsg);
    geotiffwrite(fullfile(out_sub, [gname '-sigma_V.tif']),  single(sV_final),  R, 'CoordRefSysCode', epsg);
    geotiffwrite(fullfile(out_sub, [gname '-neff_Vx.tif']),  single(neff_Vx),   R, 'CoordRefSysCode', epsg);
    geotiffwrite(fullfile(out_sub, [gname '-neff_Vy.tif']),  single(neff_Vy),   R, 'CoordRefSysCode', epsg);
    geotiffwrite(fullfile(out_sub, [gname '-neff_V.tif']),   single(neff_V),    R, 'CoordRefSysCode', epsg);
    % -- NLM 后处理结果 --
    geotiffwrite(fullfile(out_sub, [gname '-Vx_NLM.tif']),   single(Vx_nlm),    R, 'CoordRefSysCode', epsg);
    geotiffwrite(fullfile(out_sub, [gname '-Vy_NLM.tif']),   single(Vy_nlm),    R, 'CoordRefSysCode', epsg);
    geotiffwrite(fullfile(out_sub, [gname '-V_NLM.tif']),    single(V_nlm),     R, 'CoordRefSysCode', epsg);

    fprintf('       保存完成 → %s\n', out_sub);
end

fprintf('\n=== 方案A 全部完成，总耗时 %.1f 秒 ===\n', toc(t_total));
fprintf('输出目录: %s\n', output_dir);
fprintf('\n每组输出 12 个文件:\n');
fprintf('  原始加权平均(9): Vx/Vy/V + sigma_Vx/Vy/V + neff_Vx/Vy/V\n');
fprintf('  NLM后处理  (3): Vx_NLM / Vy_NLM / V_NLM\n');


% =========================================================================
%  分组函数（南极四季版）
% =========================================================================
function labels = assign_groups(dates, mode)
    N = length(dates);
    labels = cell(N, 1);
    switch mode
        case 'monthly'
            for i = 1:N
                labels{i} = datestr(dates(i), 'yyyy-mm');
            end
        case 'seasonal'
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
                else
                    labels{i} = sprintf('%d-Spring', y);
                end
            end
        case 'yearly'
            for i = 1:N
                labels{i} = sprintf('%d', year(dates(i)));
            end
        otherwise
            win_days = str2double(strrep(mode, 'fixed_', ''));
            if isnan(win_days) || win_days <= 0
                error('未知模式: %s\n可选: monthly/seasonal/yearly/fixed_N', mode);
            end
            t0 = min(dates);
            for i = 1:N
                bin       = floor(days(dates(i) - t0) / win_days);
                bin_start = t0 + days(bin * win_days);
                labels{i} = datestr(bin_start, 'yyyy-mm-dd');
            end
    end
end
