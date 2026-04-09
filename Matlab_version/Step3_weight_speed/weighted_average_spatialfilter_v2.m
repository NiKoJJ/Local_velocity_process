% =========================================================================
%  Sentinel-1 POT 加权平均 v2
%  改进：
%   (1) 空间滤波：自适应窗口 + 动态MAD阈值（基于局部速度梯度）
%   (2) 加权平均后施加NLM滤波
%
%  流程：
%   读取 → 自适应空间MAD滤波（逐对）→ 加权平均 → NLM滤波 → 输出
% =========================================================================
clear; clc;

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
fprintf('栅格大小: %d × %d\n\n', rows, cols);

%% 3. 分组
group_labels  = assign_groups(dates, mode);
unique_groups = unique(group_labels, 'stable');
G = length(unique_groups);
fprintf('分组模式 [%s]：共 %d 组\n\n', mode, G);

%% 4. 写日志
log_file = fullfile(output_dir, sprintf('group_info_%s.txt', mode));
fid = fopen(log_file, 'w');
fprintf(fid, '分组模式     : %s\n', mode);
fprintf(fid, '数据路径     : %s\n', data_dir);
fprintf(fid, '总对数       : %d\n', N);
fprintf(fid, '总组数       : %d\n', G);
fprintf(fid, '\n--- 空间滤波 ---\n');
fprintf(fid, '开关         : %s\n', mat2str(do_spatial_filter));
fprintf(fid, '窗口(小/大)  : %d / %d 像元\n', sf_win_small, sf_win_large);
fprintf(fid, '梯度切换阈值 : %.0f m/a/pixel\n', grad_thr);
fprintf(fid, '动态阈值     : max(%.0f, %.2f×|V_median|), 上限%.0f\n', ...
    std_thr_min, alpha, std_thr_max);
fprintf(fid, '\n--- NLM滤波 ---\n');
fprintf(fid, '开关         : %s\n', mat2str(do_nlm));
fprintf(fid, 'h            : %.0f m/a\n', nlm_h);
fprintf(fid, 'patch半径    : %d 像元\n', nlm_patch_r);
fprintf(fid, '搜索半径     : %d 像元\n', nlm_search_r);
fprintf(fid, '\n生成时间     : %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
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

    n_filt_vx = 0;
    n_filt_vy = 0;

    %% 5b. 读取 + 自适应空间MAD滤波
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

        if do_spatial_filter
            nb_vx = sum(~isnan(vx(:)));
            nb_vy = sum(~isnan(vy(:)));

            % ---- 计算速度场梯度（用于自适应窗口选择）----
            % 以V幅值的梯度为依据，V不可用时退化用Vx
            V_ref = v;
            % V_ref(isnan(V_ref)) = vx(~isnan(vx) & isnan(V_ref));
            fill_mask = isnan(V_ref) & ~isnan(vx);
            V_ref(fill_mask) = abs(vx(fill_mask));

            [gx, gy] = gradient(V_ref);
            grad_mag = sqrt(gx.^2 + gy.^2);  % m/a/pixel
            % NaN区域梯度置为0（不影响阈值选择）
            grad_mag(isnan(grad_mag)) = 0;

            % ---- 动态std阈值图（基于局部速度量级）----
            % 在sf_win_large窗口内计算局部中值速度
            V_med = local_median(abs(v), sf_win_large);
            std_thr_map = min(std_thr_max, ...
                          max(std_thr_min, alpha .* V_med));
            std_thr_map(isnan(std_thr_map)) = std_thr_min;

            % ---- 自适应窗口：梯度大→小窗口，梯度小→大窗口 ----
            win_map = ones(rows, cols, 'uint8') * sf_win_large;
            win_map(grad_mag > grad_thr) = sf_win_small;

            % ---- 施加自适应滤波 ----
            vx = adaptive_mad_filter(vx, win_map, std_thr_map);
            vy = adaptive_mad_filter(vy, win_map, std_thr_map);

            % V 由滤波后 Vx/Vy 重新计算，保持一致
            v  = sqrt(vx.^2 + vy.^2);
            v(isnan(vx) | isnan(vy)) = NaN;

            n_filt_vx = n_filt_vx + (nb_vx - sum(~isnan(vx(:))));
            n_filt_vy = n_filt_vy + (nb_vy - sum(~isnan(vy(:))));
        end

        % ---- sigma ----
        if has_sigma
            svx = single(readgeoraster(fullfile(data_dir, strrep(vx_name,'-Vx.tif','-Vx_std.tif'))));
            svy = single(readgeoraster(fullfile(data_dir, strrep(vx_name,'-Vx.tif','-Vy_std.tif'))));
            sv  = single(readgeoraster(fullfile(data_dir, strrep(vx_name,'-Vx.tif','-V_std.tif'))));
            svx(svx<=0|isinf(svx)) = NaN;
            svy(svy<=0|isinf(svy)) = NaN;
            sv (sv <=0|isinf(sv))  = NaN;
        else
            svx = local_std(vx, window_size, ignore_nan, min_valid);
            svy = local_std(vy, window_size, ignore_nan, min_valid);
            sv  = local_std(v,  window_size, ignore_nan, min_valid);
        end

        bad_x = isnan(vx)|isnan(svx)|svx<1e-8;
        bad_y = isnan(vy)|isnan(svy)|svy<1e-8;
        bad_v = isnan(v) |isnan(sv) |sv <1e-8;
        vx(bad_x)=NaN; svx(bad_x)=NaN;
        vy(bad_y)=NaN; svy(bad_y)=NaN;
        v (bad_v)=NaN; sv (bad_v)=NaN;

        Vx_stack(:,:,k)  = vx;
        Vy_stack(:,:,k)  = vy;
        V_stack(:,:,k)   = v;
        sVx_stack(:,:,k) = svx;
        sVy_stack(:,:,k) = svy;
        sV_stack(:,:,k)  = sv;
    end

    if do_spatial_filter
        fprintf('       自适应空间滤波剔除: Vx=%d  Vy=%d 像元·对\n', ...
            n_filt_vx, n_filt_vy);
    end

    %% 5c. 加权平均
    t_wa = tic;
    if n_grp == 1
        Vx_avg = Vx_stack(:,:,1); sVx_avg = sVx_stack(:,:,1);
        Vy_avg = Vy_stack(:,:,1); sVy_avg = sVy_stack(:,:,1);
        V_avg  = V_stack(:,:,1);  sV_avg  = sV_stack(:,:,1);
        neff_Vx = single(~isnan(Vx_avg)); neff_Vx(isnan(Vx_avg)) = NaN;
        neff_Vy = single(~isnan(Vy_avg)); neff_Vy(isnan(Vy_avg)) = NaN;
        neff_V  = single(~isnan(V_avg));  neff_V (isnan(V_avg))  = NaN;
    else
        [Vx_avg, sVx_avg, neff_Vx] = weighted_average_m2(Vx_stack, sVx_stack);
        [Vy_avg, sVy_avg, neff_Vy] = weighted_average_m2(Vy_stack, sVy_stack);
        [V_avg,  sV_avg,  neff_V ] = weighted_average_m2(V_stack,  sV_stack);
    end
    fprintf('       加权平均完成 (%.1f 秒)\n', toc(t_wa));

    %% 5d. NLM 滤波（作用于加权平均后的年均场）
    if do_nlm
        t_nlm = tic;
        fprintf('       NLM滤波中...\n');
        [Vx_final, Vy_final] = nlm_vector_filter( ...
            Vx_avg, Vy_avg, ...
            'h',        nlm_h, ...
            'patch_r',  nlm_patch_r, ...
            'search_r', nlm_search_r, ...
            'normalize',nlm_norm);

        % 重新计算V和sigma_V
        V_final   = sqrt(Vx_final.^2 + Vy_final.^2);
        V_final(isnan(Vx_final)|isnan(Vy_final)) = NaN;
        safe_V    = V_final; safe_V(safe_V<1e-8) = NaN;
        sVx_final = sVx_avg;   % sigma 保留加权平均结果（NLM不改变不确定度）
        sVy_final = sVy_avg;
        sV_final  = sqrt((Vx_final./safe_V.*sVx_avg).^2 + ...
                         (Vy_final./safe_V.*sVy_avg).^2);
        sV_final(isnan(safe_V)) = NaN;
        fprintf('       NLM完成 (%.1f 秒)\n', toc(t_nlm));
    else
        Vx_final  = Vx_avg;  Vy_final  = Vy_avg;
        sVx_final = sVx_avg; sVy_final = sVy_avg;
        V_final   = V_avg;   sV_final  = sV_avg;
    end

    %% 5e. 写出结果
    % --- 加权平均结果（未经NLM）---
    wa_sub = fullfile(out_sub, 'weighted_avg');
    mkdir(wa_sub);
    geotiffwrite(fullfile(wa_sub,[gname '-Vx.tif']),      single(Vx_avg),  R,'CoordRefSysCode',epsg);
    geotiffwrite(fullfile(wa_sub,[gname '-Vy.tif']),      single(Vy_avg),  R,'CoordRefSysCode',epsg);
    geotiffwrite(fullfile(wa_sub,[gname '-V.tif']),       single(V_avg),   R,'CoordRefSysCode',epsg);
    geotiffwrite(fullfile(wa_sub,[gname '-sigma_Vx.tif']),single(sVx_avg), R,'CoordRefSysCode',epsg);
    geotiffwrite(fullfile(wa_sub,[gname '-sigma_Vy.tif']),single(sVy_avg), R,'CoordRefSysCode',epsg);
    geotiffwrite(fullfile(wa_sub,[gname '-sigma_V.tif']), single(sV_avg),  R,'CoordRefSysCode',epsg);
    geotiffwrite(fullfile(wa_sub,[gname '-neff_Vx.tif']), single(neff_Vx), R,'CoordRefSysCode',epsg);
    geotiffwrite(fullfile(wa_sub,[gname '-neff_Vy.tif']), single(neff_Vy), R,'CoordRefSysCode',epsg);
    geotiffwrite(fullfile(wa_sub,[gname '-neff_V.tif']),  single(neff_V),  R,'CoordRefSysCode',epsg);

    % --- 最终结果（NLM后，或与加权平均相同）---
    geotiffwrite(fullfile(out_sub,[gname '-Vx.tif']),      single(Vx_final),  R,'CoordRefSysCode',epsg);
    geotiffwrite(fullfile(out_sub,[gname '-Vy.tif']),      single(Vy_final),  R,'CoordRefSysCode',epsg);
    geotiffwrite(fullfile(out_sub,[gname '-V.tif']),       single(V_final),   R,'CoordRefSysCode',epsg);
    geotiffwrite(fullfile(out_sub,[gname '-sigma_Vx.tif']),single(sVx_final), R,'CoordRefSysCode',epsg);
    geotiffwrite(fullfile(out_sub,[gname '-sigma_Vy.tif']),single(sVy_final), R,'CoordRefSysCode',epsg);
    geotiffwrite(fullfile(out_sub,[gname '-sigma_V.tif']), single(sV_final),  R,'CoordRefSysCode',epsg);
    geotiffwrite(fullfile(out_sub,[gname '-neff_Vx.tif']), single(neff_Vx),   R,'CoordRefSysCode',epsg);
    geotiffwrite(fullfile(out_sub,[gname '-neff_Vy.tif']), single(neff_Vy),   R,'CoordRefSysCode',epsg);
    geotiffwrite(fullfile(out_sub,[gname '-neff_V.tif']),  single(neff_V),    R,'CoordRefSysCode',epsg);

    fprintf('       结果写入: %s\n', out_sub);
end

fprintf('\n=== 全部完成，总耗时 %.1f 秒 ===\n', toc(t_total));
fprintf('输出目录: %s\n', output_dir);


% =========================================================================
%  自适应MAD空间滤波
%  每个像元使用独立的窗口大小（win_map）和std阈值（thr_map）
%
%  双准则：
%   (1) |V_center - median(window)| > MAD(window)    → NaN
%   (2) std(window) > thr_map(r,c)                   → NaN
% =========================================================================
function out = adaptive_mad_filter(data, win_map, thr_map)
    out  = data;
    [nr, nc] = size(data);

    % 预计算两种窗口下的统计量
    wins = unique(win_map(:))';
    med_maps = containers.Map('KeyType','int32','ValueType','any');
    mad_maps = containers.Map('KeyType','int32','ValueType','any');
    std_maps = containers.Map('KeyType','int32','ValueType','any');

    for w = wins
        half = floor(double(w)/2);
        pad  = padarray(data, [half half], NaN, 'symmetric');
        med_m = nan(nr, nc, 'single');
        mad_m = nan(nr, nc, 'single');
        std_m = nan(nr, nc, 'single');
        for r = 1:nr
            for c = 1:nc
                patch = pad(r:r+double(w)-1, c:c+double(w)-1);
                vals  = patch(~isnan(patch));
                if numel(vals) < 3
                    std_m(r,c) = Inf;
                    continue
                end
                m = median(vals);
                med_m(r,c) = m;
                mad_m(r,c) = median(abs(vals - m));
                std_m(r,c) = std(vals);
            end
        end
        med_maps(int32(w)) = med_m;
        mad_maps(int32(w)) = mad_m;
        std_maps(int32(w)) = std_m;
    end

    % 按像元选取对应统计量，应用双准则
    for r = 1:nr
        for c = 1:nc
            if isnan(data(r,c)); continue; end
            w   = int32(win_map(r,c));
            med = med_maps(w);
            mad = mad_maps(w);
            sd  = std_maps(w);
            thr = thr_map(r,c);
            if abs(data(r,c) - med(r,c)) > mad(r,c) || sd(r,c) > thr
                out(r,c) = NaN;
            end
        end
    end
end


% =========================================================================
%  局部中值（用于计算动态阈值图）
% =========================================================================
function out = local_median(data, win)
    half = floor(win/2);
    pad  = padarray(data, [half half], NaN, 'symmetric');
    [nr, nc] = size(data);
    out  = nan(nr, nc, 'single');
    for r = 1:nr
        for c = 1:nc
            patch = pad(r:r+win-1, c:c+win-1);
            vals  = patch(~isnan(patch));
            if ~isempty(vals)
                out(r,c) = median(vals);
            end
        end
    end
end


% =========================================================================
%  NLM 向量滤波（作用于加权平均后的年均Vx/Vy）
%
%  参数:
%   h         滤波强度 (m/a)
%   patch_r   patch半径 (像元)
%   search_r  搜索窗口半径 (像元)
%   normalize true: 用单位向量计算patch距离（流向主导）
% =========================================================================
function [Vx_out, Vy_out] = nlm_vector_filter(Vx, Vy, varargin)
    p = inputParser;
    addParameter(p, 'h',         50,    @isnumeric);
    addParameter(p, 'patch_r',   2,     @isnumeric);
    addParameter(p, 'search_r',  10,    @isnumeric);
    addParameter(p, 'normalize', false, @islogical);
    parse(p, varargin{:});
    h   = single(p.Results.h);
    pr  = p.Results.patch_r;
    sr  = p.Results.search_r;
    do_norm = p.Results.normalize;

    Vx = single(Vx);  Vy = single(Vy);
    [nr, nc] = size(Vx);

    % 归一化（可选）
    if do_norm
        Vmag = sqrt(Vx.^2 + Vy.^2);
        Vmag(Vmag < 1e-8) = NaN;
        Vxs = Vx ./ Vmag;  Vys = Vy ./ Vmag;
    else
        Vxs = Vx;  Vys = Vy;
    end

    pad   = pr + sr;
    Vxs_p = padarray(Vxs, [pad pad], NaN, 'symmetric');
    Vys_p = padarray(Vys, [pad pad], NaN, 'symmetric');
    Vx_p  = padarray(Vx,  [pad pad], NaN, 'symmetric');
    Vy_p  = padarray(Vy,  [pad pad], NaN, 'symmetric');

    pw       = 2*pr + 1;
    h2       = h * h;
    nf       = single(pw * pw * 2);
    min_p    = ceil(pw*pw * 0.5);
    min_both = ceil(pw*pw * 0.3);

    Vx_out = nan(nr, nc, 'single');
    Vy_out = nan(nr, nc, 'single');

    [dR, dC] = meshgrid(-sr:sr, -sr:sr);
    dR = dR(:);  dC = dC(:);
    nc_sr = numel(dR);

    fprintf('    NLM: %d×%d 像元, patch=%d×%d, 搜索域=%d×%d, h=%.0f\n', ...
        nr, nc, pw, pw, 2*sr+1, 2*sr+1, h);
    t0 = tic;

    for r = 1:nr
        if mod(r, max(1,floor(nr/20))) == 0
            fprintf('    NLM 进度 %4d/%d (%.0fs)\n', r, nr, toc(t0));
        end
        rp = r + pad;

        for c = 1:nc
            cp = c + pad;
            if isnan(Vx(r,c)) && isnan(Vy(r,c)); continue; end

            pc_x = Vxs_p(rp-pr:rp+pr, cp-pr:cp+pr);
            pc_y = Vys_p(rp-pr:rp+pr, cp-pr:cp+pr);
            vc   = ~isnan(pc_x) & ~isnan(pc_y);
            if sum(vc(:)) < min_p
                % patch样本不足，保留原值
                Vx_out(r,c) = Vx(r,c);
                Vy_out(r,c) = Vy(r,c);
                continue
            end

            sw_x = single(0); sw_y = single(0); sw = single(0);

            for k = 1:nc_sr
                rq = rp + dR(k);  cq = cp + dC(k);
                if isnan(Vx_p(rq,cq)) && isnan(Vy_p(rq,cq)); continue; end

                pq_x = Vxs_p(rq-pr:rq+pr, cq-pr:cq+pr);
                pq_y = Vys_p(rq-pr:rq+pr, cq-pr:cq+pr);
                vb   = vc & ~isnan(pq_x) & ~isnan(pq_y);
                if sum(vb(:)) < min_both; continue; end

                dx = pc_x(vb) - pq_x(vb);
                dy = pc_y(vb) - pq_y(vb);
                d2 = (sum(dx.*dx) + sum(dy.*dy)) / nf;
                w  = exp(-d2 / h2);

                sw_x = sw_x + w * Vx_p(rq,cq);
                sw_y = sw_y + w * Vy_p(rq,cq);
                sw   = sw   + w;
            end

            if sw > 1e-10
                Vx_out(r,c) = sw_x / sw;
                Vy_out(r,c) = sw_y / sw;
            else
                Vx_out(r,c) = Vx(r,c);
                Vy_out(r,c) = Vy(r,c);
            end
        end
    end
    fprintf('    NLM 完成 (%.1f 秒)\n', toc(t0));
end


% =========================================================================
%  分组函数
% =========================================================================
function labels = assign_groups(dates, mode)
    N = length(dates);
    labels = cell(N,1);
    switch mode
        case 'monthly'
            for i=1:N; labels{i}=datestr(dates(i),'yyyy-mm'); end
        case 'seasonal'
            for i=1:N
                m=month(dates(i)); y=year(dates(i));
                if     m==12;          labels{i}=sprintf('%d-%d-Summer',y,y+1);
                elseif m==1||m==2;     labels{i}=sprintf('%d-%d-Summer',y-1,y);
                elseif m>=3&&m<=5;     labels{i}=sprintf('%d-Autumn',y);
                elseif m>=6&&m<=8;     labels{i}=sprintf('%d-Winter',y);
                else;                  labels{i}=sprintf('%d-Spring',y);
                end
            end
        case 'yearly'
            for i=1:N; labels{i}=sprintf('%d',year(dates(i))); end
        otherwise
            wd=str2double(strrep(mode,'fixed_',''));
            if isnan(wd)||wd<=0; error('未知模式: %s',mode); end
            t0=min(dates);
            for i=1:N
                bin=floor(days(dates(i)-t0)/wd);
                labels{i}=datestr(datetime(t0+days(bin*wd)),'yyyy-mm-dd');
            end
    end
end
