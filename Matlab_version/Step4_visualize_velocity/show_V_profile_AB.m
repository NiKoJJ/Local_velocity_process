clc;clear;close all


V_dir = '/data2/Phd_Work1/ICE_Velocity_Process/GAMMA_POT_Input/input/velocity';
tiff_name = '2020_v.tif';
tiff_error_name = '2020_v_err.tif';

% 输出图像目录
fig_dir = fullfile(V_dir, 'figures');
if ~fig_dir
    mkdir(fig_dir);
end



% 从点A到点B画一条剖面线  1418479.962  -2067398.747
profile_A = [1027490,  -2050289];   % [x_start, y_start]  ← 根据你的区域修改
profile_B = [1086456,  -2113113];   % [x_end,   y_end]
n_profile_pts = 200;               % 剖面采样点数


fprintf(" ....  read the tiff data .... \n");

[V, R] = readgeoraster(fullfile(V_dir,tiff_name));
[V_error, ~] = readgeoraster(fullfile(V_dir,tiff_error_name));

V = flipud(V);  % !!!!!!!!!! imagesc leftup is o (small), leftdown is increase(biggest)
v_error = flipud(V_error);

V = double(V);
V(isinf(V)| isnan(V)) = NaN;

[rows,cols] = size(V);

% x_vec = linspace(R.XWorldLimits(1), R.XWorldLimits(2),cols);
% y_vec = linspace(R.YWorldLimits(1), R.YWorldLimits(2),rows);

x_vec = linspace(R.XWorldLimits(1), R.XWorldLimits(2),cols);
y_vec = linspace(R.YWorldLimits(1), R.YWorldLimits(2),rows);
[X, Y] = meshgrid(x_vec, y_vec);

fprintf(' .... the size of Velocity: %d x %d \n',rows, cols);
fprintf(' .... the resolution of Velocity: %.0f m \n',R.CellExtentInWorldX);

%% get value along profile

% get points
fprintf("\n ....  resample points along profile line: %d points .... \n", n_profile_pts);

px = linspace(profile_A(1), profile_B(1), n_profile_pts);
py = linspace(profile_A(2), profile_B(2), n_profile_pts);

distance_m = sqrt((px - px(1)).^2 + (py - py(1)).^2);
distance_km = distance_m / 1000;

% get points value
V_profile       = interp2(X, Y, V, px, py, "linear", NaN);
V_error_profile = interp2(X, Y, V_error, px, py, "linear", NaN);

%% plot
fig1 = figure('Units','centimeters','Position',[2 2 60 30],'Color','w');

t = tiledlayout(1,2);
% 
nexttile
fill([distance_km, fliplr(distance_km)], ...
     [V_profile+ V_error_profile, fliplr(V_profile - V_error_profile)], ...
     [1.0, 0.7, 0.7], 'EdgeColor', 'none', 'FaceAlpha', 0.6, ...
     'DisplayName', '±1σ uncertainty');
hold on

plot(distance_km, V_profile,'b-', 'LineWidth', 1.5, 'DisplayName', 'V (speed)');
title(sprintf('Speed Profile'), 'FontSize', 20, 'FontWeight','bold');
xlabel("distance along the profile",'FontSize',20)
ylabel("velocity along the profile",'FontSize',20)
ylim([0, 1000]);
legend('Location','northeast','FontSize',18);
grid on; box on;
xlim([0, max(distance_km)]);
set(gca,'FontSize',18);

% 
nexttile
imagesc(x_vec,y_vec,V);
set(gca, 'YDir', 'normal');
hold on;
colormap(jet);
colorbar;
clim([0, 1500]);

plot(profile_A(1), profile_A(2),'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r', 'DisplayName', 'Point A'); 
plot(profile_B(1), profile_B(2),'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r', 'DisplayName', 'Point B');
plot(px, py, 'k-', 'LineWidth', 1.5, 'DisplayName', 'Profile Line');

text(profile_A(1)+1000, profile_A(2)-1000, 'A', 'Color', 'k', 'FontSize', 22, 'FontWeight', 'bold')
text(profile_B(1)+1000, profile_B(2)-1000, 'B', 'Color', 'k', 'FontSize', 22, 'FontWeight', 'bold')

title('Velocity Field with Profile Line', 'FontSize', 24, 'FontWeight','bold');
xlabel('X (m)', 'FontSize', 20);
ylabel('Y (m)', 'FontSize', 20);
legend('Location','northeast','FontSize',18);
% grid on; box on;
axis equal;
set(gca, 'FontSize', 18);

% 保存
out_profile = fullfile(fig_dir, 'velocity_profile.png');
exportgraphics(fig1, out_profile, 'Resolution', 150);
fprintf('剖面图已保存: %s\n', out_profile);