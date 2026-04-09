clc; clear; close all;

%% ===== 1. 输入数据路径 =====
result_dir  = '/data2/Phd_Work1/ICE_Velocity_Process/GAMMA_POT_Output2/Mertz_100m/Tidal_IBE_Flexure_Correction/Vx_Vy_V_Weighted_Average/ALL';
group_name  = 'ALL';
istlive_dir = '/data2/Phd_Work1/ICE_Velocity_Process';

%% ===== 2. 读取数据（各自保留空间参考 R）=====
[V_gamma,         R_gamma]    = readgeoraster(fullfile(result_dir,   [group_name '-V.tif']));
[V_sigma_gamma,   ~]          = readgeoraster(fullfile(result_dir,   [group_name '-sigma_V.tif']));

[V_istlive,       R_istlive]  = readgeoraster(fullfile(istlive_dir,  'Mertz_ITS_LIVE_velocity_120m_RGI19A_0000_v02_v.tif'));
[V_sigma_istlive, ~]          = readgeoraster(fullfile(istlive_dir,  'Mertz_ITS_LIVE_velocity_120m_RGI19A_0000_v02_v_error.tif'));

% 转为 double
V_gamma         = double(V_gamma);
V_sigma_gamma   = double(V_sigma_gamma);
V_istlive       = double(V_istlive);
V_sigma_istlive = double(V_sigma_istlive);

%% ===== 3. 输出目录 =====
output_dir = '/data2/Phd_Work1/ICE_Velocity_Process/Step4_visualize_velocity';
fig_dir    = fullfile(output_dir, 'figures');
if ~exist(fig_dir, 'dir'); mkdir(fig_dir); end

%% ===== 4. 剖面端点（EPSG:3031，单位 m）=====
profile_A      = [1427812, -1992753];   % 起点 [x, y]
profile_B      = [1426582, -2031743];   % 终点 [x, y]
n_profile_pts  = 200;

%% ===== 5. 清除 Inf 值（对各自数据集单独处理）=====
% GAMMA
bad_g               = isinf(V_gamma);
V_gamma(bad_g)      = NaN;
V_sigma_gamma(bad_g)= NaN;


% ITS-LIVE
bad_i                  = isinf(V_istlive);
V_istlive(bad_i)       = NaN;
V_sigma_istlive(bad_i) = NaN;

%% ===== 6. 分别建立各自的坐标网格 =====
% --- GAMMA (100 m) ---
[rows_g, cols_g] = size(V_gamma);
x_vec_g = linspace(R_gamma.XWorldLimits(1),   R_gamma.XWorldLimits(2),   cols_g);
y_vec_g = linspace(R_gamma.YWorldLimits(2),   R_gamma.YWorldLimits(1),   rows_g);  % 注意 Y 从大到小
[X_g, Y_g] = meshgrid(x_vec_g, y_vec_g);
fprintf('GAMMA   grid : %d x %d，分辨率 %.0f m\n', rows_g, cols_g, R_gamma.SampleSpacingInWorldX);

% --- ITS-LIVE (120 m) ---
[rows_i, cols_i] = size(V_istlive);
x_vec_i = linspace(R_istlive.XWorldLimits(1), R_istlive.XWorldLimits(2), cols_i);
y_vec_i = linspace(R_istlive.YWorldLimits(2), R_istlive.YWorldLimits(1), rows_i);
[X_i, Y_i] = meshgrid(x_vec_i, y_vec_i);
fprintf('ITS-LIVE grid: %d x %d，分辨率 %.0f m\n\n', rows_i, cols_i, R_istlive.CellExtentInWorldX);

%% ===== 7. 生成剖面采样点 =====
fprintf('开始绘制剖面...\n');
px = linspace(profile_A(1), profile_B(1), n_profile_pts);
py = linspace(profile_A(2), profile_B(2), n_profile_pts);

% 沿剖面距离（km）
dist_km = sqrt((px - px(1)).^2 + (py - py(1)).^2) / 1000;

%% ===== 8. 沿剖面插值（各自用各自的网格）=====
V_prof_gamma         = interp2(X_g, Y_g, V_gamma,         px, py, 'linear', NaN);
V_sigma_prof_gamma   = interp2(X_g, Y_g, V_sigma_gamma,   px, py, 'linear', NaN);

V_prof_istlive       = interp2(X_i, Y_i, V_istlive,       px, py, 'linear', NaN);
V_sigma_prof_istlive = interp2(X_i, Y_i, V_sigma_istlive, px, py, 'linear', NaN);

%% ===== 9. 绘图 =====
fig1 = figure('Units','centimeters','Position',[2 2 24 18],'Color','w');

% ---- 子图1：GAMMA 速度剖面 ----
ax1 = subplot(2,1,1);
hold on;
fill([dist_km, fliplr(dist_km)], ...
     [V_prof_gamma + V_sigma_prof_gamma, fliplr(V_prof_gamma - V_sigma_prof_gamma)], ...
     [0.7 0.85 1.0], 'EdgeColor','none','FaceAlpha',0.6, ...
     'DisplayName','GAMMA ±1σ uncertainty');
plot(dist_km, V_prof_gamma, 'b-', 'LineWidth', 1.5, 'DisplayName', 'GAMMA Speed');
hold off;
xlabel('Distance along profile (km)', 'FontSize', 10);
ylabel('Speed (m/day)',               'FontSize', 10);
title(sprintf('%s — GAMMA Speed Profile', group_name), 'FontSize', 11, 'FontWeight','bold');
legend('Location','best','FontSize',9);
xlim([0, max(dist_km)]);
grid on; box on;
set(ax1,'FontSize',9);

% ---- 子图2：两者对比 ----
ax2 = subplot(2,1,2);
hold on;
% GAMMA 不确定性带
fill([dist_km, fliplr(dist_km)], ...
     [V_prof_gamma + V_sigma_prof_gamma, fliplr(V_prof_gamma - V_sigma_prof_gamma)], ...
     [0.7 0.85 1.0], 'EdgeColor','none','FaceAlpha',0.5, ...
     'DisplayName','GAMMA ±1σ');
% ITS-LIVE 不确定性带
fill([dist_km, fliplr(dist_km)], ...
     [V_prof_istlive + V_sigma_prof_istlive, fliplr(V_prof_istlive - V_sigma_prof_istlive)], ...
     [1.0 0.85 0.7], 'EdgeColor','none','FaceAlpha',0.5, ...
     'DisplayName','ITS-LIVE ±1σ');
% 速度曲线
plot(dist_km, V_prof_gamma,   'b-',  'LineWidth', 1, 'DisplayName', 'GAMMA');
plot(dist_km, V_prof_istlive, 'r--', 'LineWidth', 1, 'DisplayName', 'ITS-LIVE');
hold off;
xlabel('Distance along profile (km)', 'FontSize', 10);
ylabel('Speed (m/day)',               'FontSize', 10);
title('GAMMA vs ITS-LIVE Speed Profile Comparison', 'FontSize', 11, 'FontWeight','bold');
legend('Location','best','FontSize',9);
xlim([0, max(dist_km)]);
grid on; box on;
set(ax2,'FontSize',9);

%% ===== 10. 保存图像 =====
out_fig = fullfile(fig_dir, sprintf('%s_speed_profile_comparison.png', group_name));
exportgraphics(fig1, out_fig, 'Resolution', 300);
fprintf('图像已保存至：%s\n', out_fig);