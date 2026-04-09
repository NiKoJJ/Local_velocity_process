clc;clear;close all

%% input data
% gamma pot velocity: V V-sigma
result_dir = '/data2/Phd_Work1/ICE_Velocity_Process/GAMMA_POT_Output2/Mertz_100m/Tidal_IBE_Flexure_Correction/Vx_Vy_V_Weighted_Average/ALL';
group_name = 'ALL';

[V_gamma, R]       = readgeoraster(fullfile(result_dir,[group_name '-V.tif']));
[V_sigma_gamma, ~] = readgeoraster(fullfile(result_dir,[group_name '-sigma_V.tif']));

% ITSLIVE velocity: V V-error
istlive_dir = '/data2/Phd_Work1/ICE_Velocity_Process';

[V_istlive, ~] = readgeoraster(fullfile(istlive_dir,'Mertz_ITS_LIVE_velocity_120m_RGI19A_0000_v02_v.tif'));
[V_sigma_istlive, ~] = readgeoraster(fullfile(istlive_dir,'Mertz_ITS_LIVE_velocity_120m_RGI19A_0000_v02_v_error.tif'));

%% output 
output_dir = '/data2/Phd_Work1/ICE_Velocity_Process/Step4_visualize_velocity';
fig_dir = fullfile(output_dir, 'figures');
mkdir(fig_dir);

%% set A ... G points along profile
% EPSG: 3031  meter
profile_A = [1427812,  -1992753];   % [x_start, y_start]  ← 根据你的区域修改
profile_B = [1426582,  -2031743];   % [x_end,   y_end]
n_profile_pts = 200;               % 剖面采样点数


%% start

V_gamma(isinf(double(V_gamma))) = NaN;
V_sigma_gamma(isinf(double(V_gamma))) = NaN;

V_istlive(isinf(double(V_gamma))) = NaN;
V_sigma_istlive(isinf(double(V_gamma))) = NaN;

[rows, cols] = size(V_gamma);
x_vec = linspace(R.XWorldLimits(1), R.XWorldLimits(2), cols);
y_vec = linspace(R.YWorldLimits(2), R.YWorldLimits(1), rows);
[X, Y] = meshgrid(x_vec, y_vec);
fprintf('数据尺寸: %d × %d，分辨率: %.0f m\n\n', rows, cols, R.SampleSpacingInWorldX);


% draw 
fprintf("Start draw profile......")
px = linspace(profile_A(1), profile_B(1),n_profile_pts);
py = linspace(profile_A(2), profile_B(2),n_profile_pts);

% distance（km）
dist_m = sqrt((px - px(1)).^2 + (py - py(1)).^2);
dist_km = dist_m / 1000;

% 双线性插值采样
V_prof_gamma          = interp2(X, Y, V_gamma,        px, py, 'linear', NaN);
V_prof_istlive        = interp2(X, Y, V_istlive,      px, py, 'linear', NaN);

V_sigma_prof_gamma    = interp2(X, Y, V_sigma_gamma,        px, py, 'linear', NaN);
V_sigma_prof_istlive  = interp2(X, Y, V_sigma_istlive,      px, py, 'linear', NaN);

%% 3. 绘图
fig1 = figure('Units','centimeters','Position',[2 2 100 80],'Color','w');

% --- 子图1：V 速度剖面 + 误差棒 ---
ax1 = subplot(2,1,1);
hold on;
% 误差带（填充区域，比离散误差棒更专业）
fill([dist_km, fliplr(dist_km)], ...
     [V_prof_gamma + V_sigma_prof_gamma , fliplr(V_prof_gamma - V_sigma_prof_gamma )], ...
     [0.7 0.85 1.0], 'EdgeColor', 'none', 'FaceAlpha', 0.6, ...
     'DisplayName', '±1σ uncertainty');
% 速度主线
plot(dist_km, V_prof_gamma, 'b-', 'LineWidth', 1.5, 'DisplayName', 'V (speed)');
hold off;
xlabel('Distance along profile (km)', 'FontSize', 10);
ylabel('Speed (m/day)', 'FontSize', 10);
title(sprintf('%s  —  Speed Profile', group_name), 'FontSize', 11, 'FontWeight','bold');
legend('Location','best','FontSize',9);
grid on; box on;
xlim([0, max(dist_km)]);
set(ax1,'FontSize',9);