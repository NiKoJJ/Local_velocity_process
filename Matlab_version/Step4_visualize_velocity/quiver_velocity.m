clc; clear; close all;

%% ========== 参数设置 ==========
result_dir = '/data2/Phd_Work1/ICE_Velocity_Process/GAMMA_POT_Output2/Mertz_100m/Tidal_IBE_Flexure_Correction/Vx_Vy_V_Weighted_Average/yearly/2025';
group_name = '2025';

output_dir = '/data2/Phd_Work1/ICE_Velocity_Process/Step4_visualize_velocity';
fig_dir    = fullfile(output_dir, 'figures');
if ~exist(fig_dir, 'dir'); mkdir(fig_dir); end

% 剖面端点（EPSG:3031，单位 m）
profile_A = [1427812, -1992753];
profile_B = [1426582, -2031743];

% 速度场显示参数（单位 m/yr）
speed_clim    = [0 1000];   % 色标范围 m/yr

% 箭头参数
quiver_step      = 50;    % 每隔多少像元画一个箭头
quiver_ref_speed = 700;    % m/yr，[] = 自动取 95 百分位
quiver_max_scale = 1.2;   % 最大箭头长度上限（相对 quiver_step 像元）
quiver_min_scale = 0.05;  % 最小箭头长度下限

dpi = 300;

%% ========== 1. 读取数据 ==========
fprintf('读取数据...\n');
[Vx, R] = readgeoraster(fullfile(result_dir, [group_name '-Vx.tif']));
[Vy, ~] = readgeoraster(fullfile(result_dir, [group_name '-Vy.tif']));
[V,  ~] = readgeoraster(fullfile(result_dir, [group_name '-V.tif']));

Vx = double(Vx);
Vy = double(Vy);
V  = double(V);

% 无效值清除
Vx(Vx == 0 | isinf(Vx)) = NaN;
Vy(Vy == 0 | isinf(Vy)) = NaN;
V (V  == 0 | isinf(V))  = NaN;

% 坐标网格
[rows, cols] = size(V);
x_vec = linspace(R.XWorldLimits(1), R.XWorldLimits(2), cols);
y_vec = linspace(R.YWorldLimits(2), R.YWorldLimits(1), rows);
[X, Y] = meshgrid(x_vec, y_vec);

fprintf('栅格大小: %d x %d，分辨率: %.0f m\n\n', rows, cols, R.SampleSpacingInWorldX);

%% ========== 2. 绘制速度场 + 矢量箭头 ==========
fprintf('绘制速度场图...\n');

fig = figure('Units','centimeters','Position',[2 2 28 24],'Color','w');
ax  = axes('Position',[0.08 0.08 0.82 0.84]);

% --- 底图：速度场 ---
h_img = imagesc(x_vec/1e3, y_vec/1e3, V);
set(h_img, 'AlphaData', ~isnan(V));
axis xy;
colormap(ax, speed_colormap());
clim(speed_clim);
cb = colorbar('Location','eastoutside');
cb.Label.String   = 'Ice Speed (m/yr)';
cb.Label.FontSize = 11;
hold on;

% --- 矢量箭头（按速度大小缩放）---
row_idx = round(quiver_step/2) : quiver_step : rows;
col_idx = round(quiver_step/2) : quiver_step : cols;

Xq  = X(row_idx, col_idx) / 1e3;
Yq  = Y(row_idx, col_idx) / 1e3;
Vxq = Vx(row_idx, col_idx);
Vyq = Vy(row_idx, col_idx);
Vq  = V(row_idx, col_idx);

% 自动 / 手动设定参考速度
if isempty(quiver_ref_speed)
    vq_valid = Vq(~isnan(Vq));
    if isempty(vq_valid)
        quiver_ref_speed = 500;
    else
        quiver_ref_speed = prctile(vq_valid, 95);
    end
    fprintf('quiver_ref_speed 自动设定为 %.1f m/yr (V 的 95%%分位)\n', quiver_ref_speed);
end

% 每个箭头基础长度（km）
unit_len = quiver_step * R.SampleSpacingInWorldX / 1e3;

% 速度幅值 → 缩放比，限制在 [min, max]
Vnorm_raw = sqrt(Vxq.^2 + Vyq.^2);
ratio     = Vnorm_raw / quiver_ref_speed;
ratio     = max(quiver_min_scale, min(quiver_max_scale, ratio));

% 方向单位向量
Vnorm_nz = Vnorm_raw;
Vnorm_nz(Vnorm_nz < 1e-6) = NaN;
Ux = Vxq ./ Vnorm_nz;
Uy = Vyq ./ Vnorm_nz;

% 最终箭头分量
Ax =  Ux .* ratio .* unit_len;
Ay = -Uy .* ratio .* unit_len;   % Y 轴方向翻转

valid_q = ~isnan(Ax) & ~isnan(Ay);

quiver(Xq(valid_q), Yq(valid_q), Ax(valid_q), Ay(valid_q), 0, ...
       'Color',       [0.1 0.1 0.1], ...
       'LineWidth',   0.8, ...
       'MaxHeadSize', 0.4, ...
       'DisplayName', sprintf('Flow (ref: %.0f m/yr)', quiver_ref_speed));

% --- 剖面线 A→B ---
% plot([profile_A(1) profile_B(1)]/1e3, [profile_A(2) profile_B(2)]/1e3, ...
%      'w-',  'LineWidth', 2.5, 'DisplayName', 'Profile A-B');
% plot([profile_A(1) profile_B(1)]/1e3, [profile_A(2) profile_B(2)]/1e3, ...
%      'k--', 'LineWidth', 1.5, 'HandleVisibility','off');
% plot(profile_A(1)/1e3, profile_A(2)/1e3, 'ws','MarkerSize',8,'MarkerFaceColor','k');
% plot(profile_B(1)/1e3, profile_B(2)/1e3, 'w^','MarkerSize',8,'MarkerFaceColor','k');
% text(profile_A(1)/1e3, profile_A(2)/1e3, '  A','Color','w','FontSize',10,'FontWeight','bold');
% text(profile_B(1)/1e3, profile_B(2)/1e3, '  B','Color','w','FontSize',10,'FontWeight','bold');

hold off;

xlabel('Easting (km, EPSG:3031)',  'FontSize',11);
ylabel('Northing (km, EPSG:3031)', 'FontSize',11);
title(sprintf('%s  —  Ice Velocity Field', group_name),'FontSize',12,'FontWeight','bold');
legend('Location','northwest','FontSize',9,'TextColor','w','Color',[0.2 0.2 0.2]);
set(ax,'FontSize',10,'DataAspectRatio',[1 1 1],'Color',[0.85 0.90 0.95]);

%% ========== 3. 保存 ==========
out_fig = fullfile(fig_dir, [group_name '_velocity_map.png']);
exportgraphics(fig, out_fig, 'Resolution', dpi);
fprintf('速度场图已保存: %s\n', out_fig);

%% ========== 自定义色表（仿 ITS-LIVE 风格）==========
function cmap = speed_colormap()
    colors = [
        0.00  0.20  0.60
        0.00  0.50  0.90
        0.00  0.80  0.80
        0.20  0.90  0.40
        1.00  0.90  0.00
        1.00  0.50  0.00
        0.85  0.00  0.00
    ];
    x_orig = linspace(0, 1, size(colors,1));
    x_new  = linspace(0, 1, 256);
    cmap   = max(0, min(1, interp1(x_orig, colors, x_new)));
end