% =========================================================================
%  Mertz/Cook Sentinel-1 POT 加权平均（含空间滤波）
%  流程：读取 → 空间MAD滤波（逐对） → 加权平均 → 输出
%
%  空间滤波参考：
%    Lee et al. (2023) Remote Sensing 15(12), 3079
%    5x5滑动窗口，双准则剔除误差offset：
%      (1) |V_center - median| > MAD  → 剔除
%      (2) std(window) > std_thresh   → 剔除
% =========================================================================
%% ========== 用户设置 ==========
data_dir   = '/data2/Phd_Work1/ICE_Velocity_Process/V_test_2017';
output_dir = '/data2/Phd_Work1/ICE_Velocity_Process/V_test_2017/Vx_Vy_V_Weighted_Average/seasonal';

has_sigma   = false;
window_size = 7;
min_valid   = 4;
ignore_nan  = true;
no_data_val = 0;
epsg        = 3031;

% ---- 自适应空间MAD滤波参数 ---------------------------
do_spatial_filter = true;

% 窗口大小（像元）：根据局部速度梯度自适应选择
sf_win_small  = 3;      % 梯度大区域（冰流边缘、接地线附近）
sf_win_large  = 7;      % 梯度小区域（内陆冰盖、平缓冰架）
grad_thr      = 0.15;     % 速度梯度阈值 (m/a/pixel)，高于此用小窗口

% 动态std阈值：基于局部速度量级的相对阈值
% std_thr = max(std_thr_abs_min, alpha × |local_median_V|)
% 物理含义：允许偏差为当地速度的alpha倍，同时设置绝对下限
alpha         = 0.3;    % 相对比例系数（速度的30%）
std_thr_min   = 0.3;     % 绝对下限 (m/a)，防止慢速区阈值过小
std_thr_max   = 1.5;    % 绝对上限 (m/a)，防止快速区阈值过大
% -----------------------------------------------------

% ---- NLM滤波参数（作用于加权平均后结果）-------------
do_nlm      = true;
nlm_h       = 0.2;       % 滤波强度 (m/a)，建议 = 1~2 × 年均sigma_V
nlm_patch_r = 2;        % patch半径（→ 5×5 patch）
nlm_search_r= 10;       % 搜索窗口半径（→ 21×21搜索域）
nlm_norm    = false;    % false: 幅值+方向共同决定相似度
% -----------------------------------------------------

% mode = 'monthly';
mode = 'seasonal';
% mode = 'yearly';
%% ==============================

mkdir(output_dir);

%% 1. 收集文件并解析日期
vx_files = dir(fullfile(data_dir, '*-Vx.tif'));
[~, idx] = sort({vx_files.name});
vx_files = vx_files(idx);
N = length(vx_files);
fprintf('找到 %d 个速度对\n', N);

dates     = NaT(N, 1);
date1_str = cell(N, 1);
date2_str = cell(N, 1);
for i = 1:N
    fname        = vx_files(i).name;
    date1_str{i} = fname(1:8);
    date2_str{i} = fname(10:17);
    dates(i)     = datetime(fname(1:8), 'InputFormat', 'yyyyMMdd');
end

%% 2. 读取参考信息
[tmp, R] = readgeoraster(fullfile(data_dir, vx_files(1).name));
[rows, cols] = size(tmp);
clear tmp;
fprintf('栅格大小: %d x %d\n\n', rows, cols);

%% 3. 分组
group_labels  = assign_groups(dates, mode);
unique_groups = unique(group_labels, 'stable');
G = length(unique_groups);
fprintf('分组模式 [%s]：共 %d 组\n\n', mode, G);

%% 4. 写日志
log_file = fullfile(output_dir, sprintf('group_info_%s.txt', mode));
fid = fopen(log_file, 'w');
fprintf(fid, '分组模式: %s\n', mode);
fprintf(fid, '数据路径: %s\n', data_dir);
fprintf(fid, '总对数: %d\n', N);
fprintf(fid, '总组数: %d\n', G);
if do_spatial_filter
    fprintf(fid, '空间滤波: 开启  窗口=%d  std阈值=%.0f m/a\n', sf_win, sf_std_thr);
else
    fprintf(fid, '空间滤波: 关闭\n');
end
fprintf(fid, '生成时间: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf(fid, '%s\n\n', repmat('=', 1, 70));
for g = 1:G
    gname   = unique_groups{g};
    in_grp  = strcmp(group_labels, gname);
    n_grp   = sum(in_grp);
    grp_idx = find(in_grp);
    fprintf(fid, '分组 %d/%d  [%s]  共 %d 对\n', g, G, gname, n_grp);
    for k = 1:n_grp
        ii = grp_idx(k);
        fprintf(fid, '  %2d. %s-%s\n', k, date1_str{ii}, date2_str{ii});
    end
    fprintf(fid, '\n');
end
fclose(fid);
fprintf('日志已写入: %s\n\n', log_file);

%% 5. 逐组处理
fprintf('开始处理...\n');
t_total = tic;

for g = 1:G
    gname   = unique_groups{g};
    in_grp  = strcmp(group_labels, gname);
    n_grp   = sum(in_grp);
    grp_idx = find(in_grp);

    fprintf('[%3d/%3d] 组 %-18s: %d 对\n', g, G, gname, n_grp);

    out_sub = fullfile(output_dir, gname);
    mkdir(out_sub);

    %% 5a. 预分配
    Vx_stack  = nan(rows, cols, n_grp, 'single');
    Vy_stack  = nan(rows, cols, n_grp, 'single');
    V_stack   = nan(rows, cols, n_grp, 'single');
    sVx_stack = nan(rows, cols, n_grp, 'single');
    sVy_stack = nan(rows, cols, n_grp, 'single');
    sV_stack  = nan(rows, cols, n_grp, 'single');

    n_filtered_vx = 0;
    n_filtered_vy = 0;

    %% 5b. 读取 + 空间滤波
    for k = 1:n_grp
        ii      = grp_idx(k);
        vx_name = vx_files(ii).name;
        vy_name = strrep(vx_name, '-Vx.tif', '-Vy.tif');
        v_name  = strrep(vx_name, '-Vx.tif', '-V.tif');

        vx = single(readgeoraster(fullfile(data_dir, vx_name)));
        vy = single(readgeoraster(fullfile(data_dir, vy_name)));
        v  = single(readgeoraster(fullfile(data_dir, v_name)));

        vx(vx == no_data_val | isinf(vx)) = NaN;
        vy(vy == no_data_val | isinf(vy)) = NaN;
        v (v  == no_data_val | isinf(v))  = NaN;

        % ---- 空间MAD滤波 ----------------------------------------
        if do_spatial_filter
            valid_before_vx = sum(~isnan(vx(:)));
            valid_before_vy = sum(~isnan(vy(:)));

            vx = spatial_mad_filter(vx, sf_win, sf_std_thr);
            vy = spatial_mad_filter(vy, sf_win, sf_std_thr);
            % 由滤波后的Vx/Vy重新计算V，保持三者一致
            v  = sqrt(vx.^2 + vy.^2);
            v(isnan(vx) | isnan(vy)) = NaN;

            n_filtered_vx = n_filtered_vx + (valid_before_vx - sum(~isnan(vx(:))));
            n_filtered_vy = n_filtered_vy + (valid_before_vy - sum(~isnan(vy(:))));
        end
        % ---------------------------------------------------------

        % ---- sigma（局部std或外部文件）
        if has_sigma
            svx = single(readgeoraster(fullfile(data_dir, strrep(vx_name, '-Vx.tif', '-Vx_std.tif'))));
            svy = single(readgeoraster(fullfile(data_dir, strrep(vx_name, '-Vx.tif', '-Vy_std.tif'))));
            sv  = single(readgeoraster(fullfile(data_dir, strrep(vx_name, '-Vx.tif', '-V_std.tif'))));
            svx(svx <= 0 | isinf(svx)) = NaN;
            svy(svy <= 0 | isinf(svy)) = NaN;
            sv (sv  <= 0 | isinf(sv))  = NaN;
        else
            svx = local_std(vx, window_size, ignore_nan, min_valid);
            svy = local_std(vy, window_size, ignore_nan, min_valid);
            sv  = local_std(v,  window_size, ignore_nan, min_valid);
        end

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

    if do_spatial_filter
        fprintf('       空间滤波剔除像元: Vx=%d  Vy=%d\n', n_filtered_vx, n_filtered_vy);
    end

    %% 5c. 加权平均
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
        [Vx_final,  sVx_final, neff_Vx] = weighted_average_m2(Vx_stack,  sVx_stack);
        [Vy_final,  sVy_final, neff_Vy] = weighted_average_m2(Vy_stack,  sVy_stack);
        [V_final,   sV_final,  neff_V ] = weighted_average_m2(V_stack,   sV_stack);
    end
    fprintf('       加权平均完成 (%.1f 秒)\n', toc(t_wa));

    %% 5d. 写出结果（9个主要文件）
    geotiffwrite(fullfile(out_sub, [gname '-Vx.tif']),       single(Vx_final),  R, 'CoordRefSysCode', epsg);
    geotiffwrite(fullfile(out_sub, [gname '-Vy.tif']),       single(Vy_final),  R, 'CoordRefSysCode', epsg);
    geotiffwrite(fullfile(out_sub, [gname '-V.tif']),        single(V_final),   R, 'CoordRefSysCode', epsg);
    geotiffwrite(fullfile(out_sub, [gname '-sigma_Vx.tif']), single(sVx_final), R, 'CoordRefSysCode', epsg);
    geotiffwrite(fullfile(out_sub, [gname '-sigma_Vy.tif']), single(sVy_final), R, 'CoordRefSysCode', epsg);
    geotiffwrite(fullfile(out_sub, [gname '-sigma_V.tif']),  single(sV_final),  R, 'CoordRefSysCode', epsg);
    geotiffwrite(fullfile(out_sub, [gname '-neff_Vx.tif']),  single(neff_Vx),   R, 'CoordRefSysCode', epsg);
    geotiffwrite(fullfile(out_sub, [gname '-neff_Vy.tif']),  single(neff_Vy),   R, 'CoordRefSysCode', epsg);
    geotiffwrite(fullfile(out_sub, [gname '-neff_V.tif']),   single(neff_V),    R, 'CoordRefSysCode', epsg);

    %% 5e. 由加权Vx/Vy重新计算V和sigma_V
    V_final2  = sqrt(Vx_final.^2 + Vy_final.^2);
    safe_V    = V_final2;
    safe_V(safe_V < 1e-8) = NaN;
    sV_final2 = sqrt((Vx_final ./ safe_V .* sVx_final).^2 + ...
                     (Vy_final ./ safe_V .* sVy_final).^2);
    sV_final2(isnan(safe_V)) = NaN;

    geotiffwrite(fullfile(out_sub, [gname '-V-2.tif']),       single(V_final2),  R, 'CoordRefSysCode', epsg);
    geotiffwrite(fullfile(out_sub, [gname '-sigma_V-2.tif']), single(sV_final2), R, 'CoordRefSysCode', epsg);
    geotiffwrite(fullfile(out_sub, [gname '-neff_V-2.tif']),  single(neff_V),    R, 'CoordRefSysCode', epsg);

    fprintf('       结果已写入: %s\n', out_sub);
end

fprintf('\n=== 全部完成，总耗时 %.1f 秒 ===\n', toc(t_total));
fprintf('输出目录: %s\n', output_dir);
fprintf('日志文件: %s\n', log_file);


% =========================================================================
%  空间MAD滤波（Lee et al. 2023, Section 3.2）
%  输入:
%    data     - 2D速度场 (single, NaN为无效)
%    win      - 滑动窗口边长 (奇数, e.g. 5 代表5x5窗口)
%    std_thr  - 窗口内std阈值 (m/a)
%  输出:
%    out      - 滤波后速度场 (异常像元→NaN)
%
%  双准则:
%    (1) |V_center - median(window)| > MAD(window)   → NaN
%    (2) std(window) > std_thr                        → NaN
% =========================================================================
function out = spatial_mad_filter(data, win, std_thr)
    out  = data;
    half = floor(win / 2);
    [nr, nc] = size(data);

    % 镜像padding避免边缘问题
    pad = padarray(data, [half half], NaN, 'symmetric');

    med_map = nan(nr, nc, 'single');
    mad_map = nan(nr, nc, 'single');
    std_map = nan(nr, nc, 'single');

    for r = 1:nr
        for c = 1:nc
            patch = pad(r : r+win-1, c : c+win-1);
            vals  = patch(~isnan(patch));
            if numel(vals) < 3
                % 有效样本不足，直接标记剔除
                std_map(r,c) = Inf;
                continue
            end
            m            = median(vals);
            med_map(r,c) = m;
            mad_map(r,c) = median(abs(vals - m));
            std_map(r,c) = std(vals);
        end
    end

    % 准则1: 偏离窗口中值超过MAD
    crit1 = abs(data - med_map) > mad_map;
    % 准则2: 窗口内std超过阈值
    crit2 = std_map > std_thr;

    out(crit1 | crit2) = NaN;
end


% =========================================================================
%  分组函数
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
                error('未知模式: %s', mode);
            end
            t0 = min(dates);
            for i = 1:N
                bin       = floor(days(dates(i) - t0) / win_days);
                bin_start = t0 + days(bin * win_days);
                labels{i} = datestr(datetime(bin_start), 'yyyy-mm-dd');
            end
    end
end
