% =========================================================================
%  冰速场可视化脚本（插值连续版）
%  修复：剖面误差带含 NaN 时先插值补全再绘制，保证误差带始终连续
%
%  输出：
%   1. *_profile.png       — 剖面速度 + 连续误差带（V / Vx / Vy）
%   2. *_velocity_map.png  — 速度场彩图 + 流向箭头 + 剖面线
%   3. *_sigma_map.png     — 不确定性三联图
% =========================================================================
clear; clc; close all

%% ========== 用户设置 ==========
result_dir = '/data2/Phd_Work1/ICE_Velocity_Process/GAMMA_POT_Output/Mertz_100m/Tidal_IBE_Flexure_Correction/Vx_Vy_V_Weighted_Average/yearly/2025';
group_name = '2025';

fig_dir = fullfile(result_dir, 'figures');
mkdir(fig_dir);

% 剖面端点坐标（EPSG:3031，单位 m）
profile_A     = [1433081, -1966321];
profile_B     = [1425616, -2076528];
n_profile_pts = 150;

% 插值方法：
%  'linear' — 线性，速度快，适合数据连续性好的情况
%  'pchip'  — 保形分段三次，不产生振荡，推荐冰速剖面
%  'spline' — 三次样条，最平滑但大空洞两端可能过拟合
interp_method = 'pchip';

% 速度场显示参数
quiver_step   = 50;     % 每隔多少像元画一个箭头
speed_clim    = [0 1];  % 速度色标范围 (m/day)

% 箭头缩放参数
% quiver_ref_speed : 参考速度 (m/day)，该速度下的箭头长度 = quiver_step 个像元
%   建议设为研究区最大速度的 50~70%，让快速区箭头醒目但不重叠
%   设为 [] 则自动取剖面有效速度的 95 百分位
quiver_ref_speed = [1.5];   % m/day，[] = 自动
quiver_max_scale = 1.2;  % 最大箭头长度上限（相对于 quiver_step 像元）
quiver_min_scale = 0.05; % 最小箭头长度下限（避免极慢区箭头消失）

dpi = 300;
%% ================================

%% 1. 读取数据
fprintf('读取数据...\n');
[Vx,       R] = readgeoraster(fullfile(result_dir, [group_name '-Vx.tif']));
[Vy,       ~] = readgeoraster(fullfile(result_dir, [group_name '-Vy.tif']));
[V,        ~] = readgeoraster(fullfile(result_dir, [group_name '-V.tif']));
[sigma_Vx, ~] = readgeoraster(fullfile(result_dir, [group_name '-sigma_Vx.tif']));
[sigma_Vy, ~] = readgeoraster(fullfile(result_dir, [group_name '-sigma_Vy.tif']));
[sigma_V,  ~] = readgeoraster(fullfile(result_dir, [group_name '-sigma_V.tif']));

Vx       = double(Vx);       Vy       = double(Vy);       V        = double(V);
sigma_Vx = double(sigma_Vx); sigma_Vy = double(sigma_Vy); sigma_V  = double(sigma_V);

Vx(Vx == 0 | isinf(Vx)) = NaN;
Vy(Vy == 0 | isinf(Vy)) = NaN;
V (V  == 0 | isinf(V))  = NaN;
sigma_Vx(sigma_Vx <= 0 | isinf(sigma_Vx)) = NaN;
sigma_Vy(sigma_Vy <= 0 | isinf(sigma_Vy)) = NaN;
sigma_V (sigma_V  <= 0 | isinf(sigma_V))  = NaN;

[rows, cols] = size(V);
x_vec = linspace(R.XWorldLimits(1), R.XWorldLimits(2), cols);
y_vec = linspace(R.YWorldLimits(2), R.YWorldLimits(1), rows);
[X, Y] = meshgrid(x_vec, y_vec);

fprintf('栅格大小: %d x %d，分辨率: %.0f m\n\n', rows, cols, R.SampleSpacingInWorldX);


% =========================================================================
%  图1：剖面速度 + 误差带
% =========================================================================
fprintf('绘制剖面图...\n');

%% 2. 剖面采样
px      = linspace(profile_A(1), profile_B(1), n_profile_pts);
py      = linspace(profile_A(2), profile_B(2), n_profile_pts);
dist_km = sqrt((px-px(1)).^2 + (py-py(1)).^2) / 1000;

V_prof        = interp2(X, Y, V,        px, py, 'linear', NaN);
sigma_V_prof  = interp2(X, Y, sigma_V,  px, py, 'linear', NaN);
Vx_prof       = interp2(X, Y, Vx,       px, py, 'linear', NaN);
sigma_Vx_prof = interp2(X, Y, sigma_Vx, px, py, 'linear', NaN);
Vy_prof       = interp2(X, Y, Vy,       px, py, 'linear', NaN);
sigma_Vy_prof = interp2(X, Y, sigma_Vy, px, py, 'linear', NaN);

fprintf('剖面 NaN 数量: V=%d  Vx=%d  Vy=%d / %d 总点数\n', ...
    sum(isnan(V_prof)), sum(isnan(Vx_prof)), sum(isnan(Vy_prof)), n_profile_pts);
fprintf('插值方法: %s\n\n', interp_method);

%% 3. 绘制剖面图
fig1 = figure('Units','centimeters','Position',[2 2 72 60],'Color','w');

ax1 = subplot(3,1,1);
plot_band_interp(ax1, dist_km, V_prof, sigma_V_prof, ...
    [0.7 0.85 1.0], 'b', 'V (speed)', interp_method);
xlabel('Distance along profile (km)','FontSize',24);
ylabel('Speed (m/day)','FontSize',24);
title(sprintf('%s  Speed Profile', group_name),'FontSize',24,'FontWeight','bold');
legend(ax1,'Location','best','FontSize',24);
grid on; box on; xlim([0 max(dist_km)]);
set(ax1,'FontSize',24);

ax2 = subplot(3,1,2);
plot_band_interp(ax2, dist_km, Vx_prof, sigma_Vx_prof, ...
    [1.0 0.8 0.8], 'r', 'Vx', interp_method);
yline(0,'k--','LineWidth',0.8,'HandleVisibility','off');
xlabel('Distance along profile (km)','FontSize',24);
ylabel('Vx (m/day)','FontSize',24);
title('Vx Profile','FontSize',24,'FontWeight','bold');
legend(ax2,'Location','best','FontSize',24);
grid on; box on; xlim([0 max(dist_km)]);
set(ax2,'FontSize',24);

ax3 = subplot(3,1,3);
plot_band_interp(ax3, dist_km, Vy_prof, sigma_Vy_prof, ...
    [0.8 1.0 0.8], [0 0.6 0], 'Vy', interp_method);
yline(0,'k--','LineWidth',0.8,'HandleVisibility','off');
xlabel('Distance along profile (km)','FontSize',24);
ylabel('Vy (m/day)','FontSize',24);
title('Vy Profile','FontSize',24,'FontWeight','bold');
legend(ax3,'Location','best','FontSize',24);
grid on; box on; xlim([0 max(dist_km)]);
set(ax3,'FontSize',24);
% format_fig_V2(fig1);

out_profile = fullfile(fig_dir, [group_name '_profile.png']);
exportgraphics(fig1, out_profile, 'Resolution', dpi);
fprintf('剖面图已保存: %s\n', out_profile);


% =========================================================================
%  图2：速度场彩图 + 流向箭头 + 剖面线
% =========================================================================
fprintf('绘制速度场图...\n');

fig2 = figure('Units','centimeters','Position',[2 2 84 72],'Color','w');
ax_main = axes('Position',[0.08 0.08 0.82 0.84]);

h_img = imagesc(x_vec/1e3, y_vec/1e3, V);
set(h_img,'AlphaData', ~isnan(V));
axis xy;
colormap(ax_main, speed_colormap());
clim(speed_clim);
cb = colorbar('Location','eastoutside');
cb.Label.String = 'Ice Speed (m/day)';
cb.Label.FontSize = 24;
hold on;

row_idx = round(quiver_step/2) : quiver_step : rows;
col_idx = round(quiver_step/2) : quiver_step : cols;
Xq  = X(row_idx, col_idx) / 1e3;
Yq  = Y(row_idx, col_idx) / 1e3;
Vxq = Vx(row_idx, col_idx);
Vyq = Vy(row_idx, col_idx);
Vq  = V(row_idx, col_idx);

% ---- 按实际速度缩放箭头长度 ----
% 参考速度：quiver_ref_speed 对应 1 个 quiver_step 单元长度的箭头
if isempty(quiver_ref_speed)
    vq_valid = Vq(~isnan(Vq));
    if isempty(vq_valid)
        quiver_ref_speed = 1;
    else
        quiver_ref_speed = prctile(vq_valid, 95);   % 自动取 95 百分位
    end
    fprintf('quiver_ref_speed 自动设定为 %.3f m/day (V 的 95%%分位)\n', quiver_ref_speed);
end

% 每个箭头的基础像元单位长度（km）
unit_len = quiver_step * R.SampleSpacingInWorldX / 1e3;

% 缩放系数 = 速度 / 参考速度，限制在 [min, max] 范围内
Vnorm_raw = sqrt(Vxq.^2 + Vyq.^2);   % 真实速度幅值
ratio     = Vnorm_raw / quiver_ref_speed;
ratio     = max(quiver_min_scale, min(quiver_max_scale, ratio));  % 限幅

% 方向单位向量（归一化）
Vnorm_nz = Vnorm_raw;
Vnorm_nz(Vnorm_nz < 1e-6) = NaN;
Ux = Vxq ./ Vnorm_nz;
Uy = Vyq ./ Vnorm_nz;

% 最终箭头分量 = 单位长度 × 速度缩放比 × 方向
Ax = Ux .* ratio .* unit_len;
Ay = Uy .* ratio .* unit_len;

valid_q = ~isnan(Ax) & ~isnan(Ay);
quiver(Xq(valid_q), Yq(valid_q), Ax(valid_q), -Ay(valid_q), 0, ...
       'Color',[0.1 0.1 0.1], 'LineWidth',0.8, 'MaxHeadSize',0.4, ...
       'DisplayName','Flow direction');

% 箭头图例（标注参考速度对应箭头长度）
legend_str = sprintf('Flow (ref: %.2f m/day)', quiver_ref_speed);
quiver(NaN, NaN, NaN, NaN, 0, 'Color',[0.1 0.1 0.1], ...
       'LineWidth',0.8, 'MaxHeadSize',0.4, 'DisplayName',legend_str);

plot([profile_A(1) profile_B(1)]/1e3, [profile_A(2) profile_B(2)]/1e3, ...
     'w-','LineWidth',2.5,'DisplayName','Profile');
plot([profile_A(1) profile_B(1)]/1e3, [profile_A(2) profile_B(2)]/1e3, ...
     'k--','LineWidth',1.5,'HandleVisibility','off');
plot(profile_A(1)/1e3, profile_A(2)/1e3,'ws','MarkerSize',8,'MarkerFaceColor','k');
plot(profile_B(1)/1e3, profile_B(2)/1e3,'w^','MarkerSize',8,'MarkerFaceColor','k');
text(profile_A(1)/1e3, profile_A(2)/1e3,'  A','Color','w','FontSize',24,'FontWeight','bold');
text(profile_B(1)/1e3, profile_B(2)/1e3,'  B','Color','w','FontSize',24,'FontWeight','bold');
hold off;

xlabel('Easting (km, EPSG:3031)','FontSize',24);
ylabel('Northing (km, EPSG:3031)','FontSize',24);
title(sprintf('%s  Ice Velocity Field', group_name),'FontSize',24,'FontWeight','bold');
legend('Location','northwest','FontSize',24,'TextColor','w','Color',[0.4 0.4 0.4]);
set(ax_main,'FontSize',24,'DataAspectRatio',[1 1 1],'Color',[0.85 0.90 0.95]);


out_vector = fullfile(fig_dir, [group_name '_velocity_map.png']);
exportgraphics(fig2, out_vector, 'Resolution', dpi);
fprintf('速度场图已保存: %s\n', out_vector);


% =========================================================================
%  图3：不确定性三联图
% =========================================================================
fprintf('绘制不确定性图...\n');

fig3 = figure('Units','centimeters','Position',[2 2 98 42],'Color','w');
sigma_fields = {sigma_V,  sigma_Vx,  sigma_Vy};
sigma_titles = {'\sigma_V (m/day)','\sigma_{Vx} (m/day)','\sigma_{Vy} (m/day)'};
for k = 1:3
    sf = sigma_fields{k};
    subplot(1,3,k);
    imagesc(x_vec/1e3, y_vec/1e3, sf);
    set(gca,'YDir','normal'); axis equal tight;
    colormap(gca, hot(256)); colorbar;
    valid_sf = sf(~isnan(sf));
    if ~isempty(valid_sf); clim([0 prctile(valid_sf,95)]); end
    title(sigma_titles{k},'FontSize',24);
    xlabel('km'); ylabel('km'); set(gca,'FontSize',24);
end
sgtitle(sprintf('%s  Velocity Uncertainty', group_name),'FontSize',24,'FontWeight','bold');
% format_fig_V2(fig3);

out_sigma = fullfile(fig_dir, [group_name '_sigma_map.png']);
exportgraphics(fig3, out_sigma, 'Resolution', dpi);
fprintf('不确定性图已保存: %s\n\n', out_sigma);
fprintf('=== 全部完成 ===\n');


% =========================================================================
%  plot_band_interp — 插值补 NaN 后绘制连续误差带
%
%  处理策略（三层）：
%   1. 内插区（首末有效点之间的 NaN）：interp1 插值，绘制连续带
%   2. 外推区（首有效点之前 / 末有效点之后）：保留 NaN，不绘制
%   3. 插值填补区域：叠加灰色半透明竖条 + 图例提示，明确告知读者
%
%  参数：
%   ax         — 目标坐标轴
%   x          — x轴（剖面距离 km）
%   val        — 中心线数值（可含 NaN）
%   sigma      — 不确定性（可含 NaN）
%   face_color — 误差带颜色 [r g b]
%   line_color — 中心线颜色
%   label      — 图例中心线标签
%   method     — 插值方法 ('linear'/'pchip'/'spline')
% =========================================================================
function plot_band_interp(ax, x, val, sigma, face_color, line_color, label, method)

    % ---- 1. 记录原始 NaN 位置 ----
    nan_orig  = isnan(val) | isnan(sigma);
    valid_idx = find(~nan_orig);

    if numel(valid_idx) < 2
        warning('plot_band_interp: 有效点不足 2 个，跳过 [%s]', label);
        return;
    end

    i_first = valid_idx(1);
    i_last  = valid_idx(end);

    % ---- 2. 内插：只在首末有效点范围内插值，不做外推 ----
    val_plot   = val;
    sigma_plot = sigma;

    inner = i_first : i_last;
    nan_inner = nan_orig(inner);

    if any(nan_inner)
        x_valid   = x(~nan_orig);
        val_valid = val(~nan_orig);
        sig_valid = sigma(~nan_orig);

        x_inner             = x(inner);
        val_plot(inner)     = interp1(x_valid, val_valid, x_inner, method);
        sigma_plot(inner)   = interp1(x_valid, sig_valid, x_inner, method);
        sigma_plot          = max(sigma_plot, 0);   % 防止插值产生负值
    end

    % ---- 3. 绘制连续误差带（内插范围内）----
    x_draw = x(inner);
    hi     = val_plot(inner) + sigma_plot(inner);
    lo     = val_plot(inner) - sigma_plot(inner);

    fill(ax, [x_draw, fliplr(x_draw)], [hi, fliplr(lo)], ...
         face_color, 'EdgeColor','none', 'FaceAlpha',0.6, ...
         'DisplayName', '±1\sigma uncertainty');
    hold(ax, 'on');

    % ---- 4. 绘制连续中心线 ----
    plot(ax, x_draw, val_plot(inner), ...
         '-', 'Color',line_color, 'LineWidth',1.5, 'DisplayName',label);

    % ---- 5. 标注插值填补区域（灰色竖条）----
    if any(nan_inner)
        % 自动获取当前 y 轴范围（需先绘制才有范围）
        drawnow;
        yl = ylim(ax);

        % 找连续的插值段
        d_nan = diff([false; nan_inner(:); false]);
        seg_s = find(d_nan ==  1);
        seg_e = find(d_nan == -1) - 1;

        for s = 1:numel(seg_s)
            x1 = x(inner(seg_s(s)));
            x2 = x(inner(seg_e(s)));
            patch(ax, [x1 x2 x2 x1], [yl(1) yl(1) yl(2) yl(2)], ...
                  [0.7 0.7 0.7], 'EdgeColor','none', 'FaceAlpha',0.3, ...
                  'HandleVisibility','off');
        end

        % 图例只加一次
        patch(ax, [NaN NaN NaN NaN], [NaN NaN NaN NaN], ...
              [0.7 0.7 0.7], 'EdgeColor','none', 'FaceAlpha',0.3, ...
              'DisplayName','Interpolated gap');
    end
end


% =========================================================================
%  speed_colormap — 仿 ITS_LIVE 风格配色
% =========================================================================
function cmap = speed_colormap()
    colors = [
        0.00  0.20  0.60;
        0.00  0.50  0.90;
        0.00  0.80  0.80;
        0.20  0.90  0.40;
        1.00  0.90  0.00;
        1.00  0.50  0.00;
        0.85  0.00  0.00;
    ];
    n      = 256;
    x_orig = linspace(0, 1, size(colors,1));
    x_new  = linspace(0, 1, n);
    cmap   = max(0, min(1, interp1(x_orig, colors, x_new)));
end
