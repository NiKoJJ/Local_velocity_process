% =========================================================================
%  冰流速可视化：误差棒剖面图 + 流速矢量图
%
%  输入：加权平均后的 Vx / Vy / V / sigma_Vx / sigma_Vy / sigma_V tif
%  输出：两种专业可视化图
% =========================================================================
clear; clc; close all

%% ========== 用户参数 ==========
% 输入文件（某一时间组，如月度结果）
result_dir = '/data2/Phd_Work1/ICE_Velocity_Process/GAMMA_POT_Output/Mertz_100m/Tidal_IBE_Flexure_Correction/Vx_Vy_V_Weighted_Average/yearly/2025';
group_name = '2025';

% 输出图像目录
fig_dir = fullfile(result_dir, 'figures');
mkdir(fig_dir);

% 剖面线定义（极地立体坐标，单位：米）
% 从点A到点B画一条剖面线
profile_A = [1427812,  -1992753];   % [x_start, y_start]  ← 根据你的区域修改
profile_B = [1426582,  -2031743];   % [x_end,   y_end]
n_profile_pts = 200;               % 剖面采样点数

% 矢量图参数
quiver_step   = 50;     % 每隔多少像素画一个箭头（越大越稀疏）
quiver_scale  = 0.8;    % 箭头长度缩放因子（越小箭头越短）
speed_clim    = [0 1]; % 速度色标范围（m/day），根据实际调整

% 图像分辨率
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

Vx = double(Vx);  Vy = double(Vy);  V = double(V);
sigma_Vx = double(sigma_Vx);
sigma_Vy = double(sigma_Vy);
sigma_V  = double(sigma_V);

% 无效值处理
Vx(Vx == 0 | isinf(Vx)) = NaN;
Vy(Vy == 0 | isinf(Vy)) = NaN;
V (V  == 0 | isinf(V))  = NaN;
sigma_Vx(sigma_Vx <= 0 | isinf(sigma_Vx)) = NaN;
sigma_Vy(sigma_Vy <= 0 | isinf(sigma_Vy)) = NaN;
sigma_V (sigma_V  <= 0 | isinf(sigma_V))  = NaN;

% 构建坐标网格
[rows, cols] = size(V);
x_vec = linspace(R.XWorldLimits(1), R.XWorldLimits(2), cols);
y_vec = linspace(R.YWorldLimits(2), R.YWorldLimits(1), rows);
[X, Y] = meshgrid(x_vec, y_vec);

fprintf('数据尺寸: %d × %d，分辨率: %.0f m\n\n', rows, cols, R.SampleSpacingInWorldX);

% =========================================================================
%  图一：带误差棒的剖面图
% =========================================================================
fprintf('绘制剖面图...\n');

%% 2. 沿剖面线采样
px = linspace(profile_A(1), profile_B(1), n_profile_pts);
py = linspace(profile_A(2), profile_B(2), n_profile_pts);

% 计算剖面距离（km）
dist_m = sqrt((px - px(1)).^2 + (py - py(1)).^2);
dist_km = dist_m / 1000;

% 双线性插值采样
V_prof        = interp2(X, Y, V,        px, py, 'linear', NaN);
sigma_V_prof  = interp2(X, Y, sigma_V,  px, py, 'linear', NaN);
Vx_prof       = interp2(X, Y, Vx,       px, py, 'linear', NaN);
sigma_Vx_prof = interp2(X, Y, sigma_Vx, px, py, 'linear', NaN);
Vy_prof       = interp2(X, Y, Vy,       px, py, 'linear', NaN);
sigma_Vy_prof = interp2(X, Y, sigma_Vy, px, py, 'linear', NaN);

%% 3. 绘图
fig1 = figure('Units','centimeters','Position',[2 2 100 80],'Color','w');

% --- 子图1：V 速度剖面 + 误差棒 ---
ax1 = subplot(3,1,1);
hold on;
% 误差带（填充区域，比离散误差棒更专业）
fill([dist_km, fliplr(dist_km)], ...
     [V_prof + sigma_V_prof, fliplr(V_prof - sigma_V_prof)], ...
     [0.7 0.85 1.0], 'EdgeColor', 'none', 'FaceAlpha', 0.6, ...
     'DisplayName', '±1σ uncertainty');
% 速度主线
plot(dist_km, V_prof, 'b-', 'LineWidth', 1.5, 'DisplayName', 'V (speed)');
hold off;
xlabel('Distance along profile (km)', 'FontSize', 10);
ylabel('Speed (m/day)', 'FontSize', 10);
title(sprintf('%s  —  Speed Profile', group_name), 'FontSize', 11, 'FontWeight','bold');
legend('Location','best','FontSize',9);
grid on; box on;
xlim([0, max(dist_km)]);
set(ax1,'FontSize',9);

% --- 子图2：Vx 剖面 ---
ax2 = subplot(3,1,2);
hold on;
fill([dist_km, fliplr(dist_km)], ...
     [Vx_prof + sigma_Vx_prof, fliplr(Vx_prof - sigma_Vx_prof)], ...
     [1.0 0.8 0.8], 'EdgeColor','none','FaceAlpha',0.6,'DisplayName','±1σ');
plot(dist_km, Vx_prof, 'r-', 'LineWidth',1.5,'DisplayName','Vx');
yline(0,'k--','LineWidth',0.8);
hold off;
xlabel('Distance along profile (km)','FontSize',10);
ylabel('Vx (m/day)','FontSize',10);
title('Vx Profile','FontSize',11,'FontWeight','bold');
legend('Location','best','FontSize',9);
grid on; box on;
xlim([0, max(dist_km)]);
set(ax2,'FontSize',9);

% --- 子图3：Vy 剖面 ---
ax3 = subplot(3,1,3);
hold on;
fill([dist_km, fliplr(dist_km)], ...
     [Vy_prof + sigma_Vy_prof, fliplr(Vy_prof - sigma_Vy_prof)], ...
     [0.8 1.0 0.8],'EdgeColor','none','FaceAlpha',0.6,'DisplayName','±1σ');
plot(dist_km, Vy_prof, 'Color',[0 0.6 0],'LineWidth',1.5,'DisplayName','Vy');
yline(0,'k--','LineWidth',0.8);
hold off;
xlabel('Distance along profile (km)','FontSize',10);
ylabel('Vy (m/day)','FontSize',10);
title('Vy Profile','FontSize',11,'FontWeight','bold');
legend('Location','best','FontSize',9);
grid on; box on;
xlim([0, max(dist_km)]);
set(ax3,'FontSize',9);
format_fig_V2(fig1);

% 保存
out_profile = fullfile(fig_dir, [group_name '_profile.png']);
exportgraphics(fig1, out_profile, 'Resolution', dpi);
fprintf('剖面图已保存: %s\n', out_profile);


% =========================================================================
%  图二：流速矢量图（速度色图 + 方向箭头）
% =========================================================================
fprintf('绘制矢量图...\n');

fig2 = figure('Units','centimeters','Position',[2 2 120 100],'Color','w');

%% 4. 速度底图
ax_main = axes('Position',[0.08 0.08 0.82 0.84]);

% 速度色图（用 imagesc 配合 pcolor 风格）
h_img = imagesc(x_vec/1e3, y_vec/1e3, V);
set(h_img,'AlphaData', ~isnan(V));   % NaN区域透明
axis xy;                              % y轴正方向向上
colormap(ax_main, speed_colormap());  % 自定义色表（见底部函数）
clim(speed_clim);
cb = colorbar('Location','eastoutside');
cb.Label.String = 'Ice Speed (m/day)';
cb.Label.FontSize = 11;
hold on;

%% 5. 矢量箭头（子采样）
% 取每 quiver_step 个像素一个点
row_idx = round(quiver_step/2) : quiver_step : rows;
col_idx = round(quiver_step/2) : quiver_step : cols;

Xq  = X(row_idx, col_idx) / 1e3;
Yq  = Y(row_idx, col_idx) / 1e3;
Vxq = Vx(row_idx, col_idx);
Vyq = Vy(row_idx, col_idx);
Vq  = V(row_idx, col_idx);

% 归一化箭头（只显示方向，不反映大小）
Vnorm = sqrt(Vxq.^2 + Vyq.^2);
Vnorm(Vnorm < 0.1) = NaN;
Ux = Vxq ./ Vnorm;   % 单位方向向量
Uy = Vyq ./ Vnorm;

% 过滤NaN点
valid_q = ~isnan(Ux) & ~isnan(Uy) & ~isnan(Vq);
scale = quiver_scale * quiver_step * R.SampleSpacingInWorldX / 1e3;

quiver(Xq(valid_q), Yq(valid_q), ...
       Ux(valid_q) * scale, -Uy(valid_q) * scale, ...
       0, ...                          % 0=不自动缩放
       'Color', [0.1 0.1 0.1], ...
       'LineWidth', 1, ...
       'MaxHeadSize', 1, ...
       'DisplayName', 'Flow direction');

%% 6. 在矢量图上叠加剖面线位置
plot([profile_A(1), profile_B(1)]/1e3, [profile_A(2), profile_B(2)]/1e3, ...
     'w-', 'LineWidth', 2.5, 'DisplayName', 'Profile');
plot([profile_A(1), profile_B(1)]/1e3, [profile_A(2), profile_B(2)]/1e3, ...
     'k--', 'LineWidth', 1.5, 'HandleVisibility','off');
plot(profile_A(1)/1e3, profile_A(2)/1e3, 'ws','MarkerSize',8,'MarkerFaceColor','k');
plot(profile_B(1)/1e3, profile_B(2)/1e3, 'w^','MarkerSize',8,'MarkerFaceColor','k');
text(profile_A(1)/1e3, profile_A(2)/1e3, '  A','Color','w','FontSize',10,'FontWeight','bold');
text(profile_B(1)/1e3, profile_B(2)/1e3, '  B','Color','w','FontSize',10,'FontWeight','bold');

hold off;
xlabel('Easting (km, EPSG:3031)','FontSize',11);
ylabel('Northing (km, EPSG:3031)','FontSize',11);
title(sprintf('%s  —  Ice Velocity Field', group_name),'FontSize',12,'FontWeight','bold');
legend('Location','northwest','FontSize',9,'TextColor','w','Color',[0.2 0.2 0.2]);
set(ax_main,'FontSize',10,'DataAspectRatio',[1 1 1],'Color',[0.85 0.90 0.95]);
grid off;
format_fig_V2(fig2);

% 保存
out_vector = fullfile(fig_dir, [group_name '_velocity_map.png']);
exportgraphics(fig2, out_vector, 'Resolution', dpi);
fprintf('矢量图已保存: %s\n', out_vector);

%% 7. 附加：不确定性分布图
fig3 = figure('Units','centimeters','Position',[2 2 120 40],'Color','w');

subplot(1,3,1);
imagesc(x_vec/1e3, y_vec/1e3, sigma_V);
set(gca,'YDir','normal'); axis equal tight;
colormap(gca, hot(256)); colorbar;
clim([0 prctile(sigma_V(~isnan(sigma_V)), 95)]);
title('\sigma_V (m/day)','FontSize',11); xlabel('km'); ylabel('km');

subplot(1,3,2);
imagesc(x_vec/1e3, y_vec/1e3, sigma_Vx);
set(gca,'YDir','normal'); axis equal tight;
colormap(gca, hot(256)); colorbar;
clim([0 prctile(sigma_Vx(~isnan(sigma_Vx)), 95)]);
title('\sigma_{Vx} (m/day)','FontSize',11); xlabel('km'); ylabel('km');

subplot(1,3,3);
imagesc(x_vec/1e3, y_vec/1e3, sigma_Vy);
set(gca,'YDir','normal'); axis equal tight;
colormap(gca, hot(256)); colorbar;
clim([0 prctile(sigma_Vy(~isnan(sigma_Vy)), 95)]);
title('\sigma_{Vy} (m/day)','FontSize',11); xlabel('km'); ylabel('km');

sgtitle(sprintf('%s  —  Velocity Uncertainty', group_name),'FontSize',12,'FontWeight','bold');
format_fig_V2(fig3);

out_sigma = fullfile(fig_dir, [group_name '_sigma_map.png']);
exportgraphics(fig3, out_sigma, 'Resolution', dpi);
fprintf('不确定性图已保存: %s\n\n', out_sigma);
fprintf('=== 全部完成 ===\n');


% =========================================================================
%  自定义速度色表（蓝→青→绿→黄→红，冰川学常用）
% =========================================================================
function cmap = speed_colormap()
    % 参考 ITS_LIVE / NSIDC 常用配色
    colors = [
        0.00  0.20  0.60;   % 深蓝（低速）
        0.00  0.50  0.90;   % 蓝
        0.00  0.80  0.80;   % 青
        0.20  0.90  0.40;   % 绿
        1.00  0.90  0.00;   % 黄
        1.00  0.50  0.00;   % 橙
        0.85  0.00  0.00;   % 红（高速）
    ];
    n = 256;
    x_orig = linspace(0, 1, size(colors,1));
    x_new  = linspace(0, 1, n);
    cmap   = interp1(x_orig, colors, x_new);
    cmap   = max(0, min(1, cmap));
end
