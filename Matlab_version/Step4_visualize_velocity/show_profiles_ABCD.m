clc; clear; close all

V_dir = '/data2/Phd_Work1/ICE_Velocity_Process/GAMMA_POT_Input/input/velocity';
tiff_name = '2025_v.tif';
tiff_error_name = '2025_v_err.tif';

% 输出目录
fig_dir = fullfile(V_dir, 'figures');
if ~exist(fig_dir, 'dir'); mkdir(fig_dir); end

%% ========== 1. 定义剖面线 (支持多点折线) ==========
profile_definitions = {
    struct('name', 'L1', 'points', [1027490, -2050289; 1086456, -2113113]);
    struct('name', 'L2', 'points', [1072863, -2076007; 1087007, -2096397; 1099131, -2109439]);
    struct('name', 'L3', 'points', [1089212, -2074537; 1109602, -2101357]);
};

n_profiles = length(profile_definitions);
n_profile_pts = 200; 

%% ========== 2. 数据读取与坐标处理 ==========
fprintf(" ....  reading the tiff data .... \n");

[V, R] = readgeoraster(fullfile(V_dir, tiff_name));
[V_error, ~] = readgeoraster(fullfile(V_dir, tiff_error_name));

% 坐标修正：GeoTIFF 通常需要翻转以配合 imagesc 的 'normal' 方向
V = flipud(double(V));
V_error = flipud(double(V_error));

V(isinf(V)| isnan(V)) = NaN;
V_error(isinf(V_error)| isnan(V_error)) = NaN;

[rows, cols] = size(V);
x_vec = linspace(R.XWorldLimits(1), R.XWorldLimits(2), cols);
y_vec = linspace(R.YWorldLimits(1), R.YWorldLimits(2), rows);
[X, Y] = meshgrid(x_vec, y_vec);

%% ========== 3. 采样与插值 (核心逻辑) ==========
fprintf("\n ....  resampling %d profile lines .... \n", n_profiles);
profile_data = struct();

for i = 1:n_profiles
    pts = profile_definitions{i}.points;
    
    % 调用辅助函数进行多段线均匀采样
    [px, py, d_m] = sample_along_polyline(pts, n_profile_pts);
    
    % 执行空间插值
    v_prof = interp2(X, Y, V, px, py, "linear", NaN);
    e_prof = interp2(X, Y, V_error, px, py, "linear", NaN);

    profile_data(i).name = profile_definitions{i}.name;
    profile_data(i).points = pts;
    profile_data(i).px = px;
    profile_data(i).py = py;
    profile_data(i).distance_km = d_m(:) / 1000; % 确保是列向量
    profile_data(i).V_profile = v_prof(:);
    profile_data(i).V_error_profile = e_prof(:);
    profile_data(i).total_length_km = d_m(end) / 1000;
end

%% ========== 4. 绘图展示 ==========
fig1 = figure('Units','centimeters','Position',[2 2 80 24],'Color','w');
tlo = tiledlayout(1, 2, 'TileSpacing', 'compact');

% ----- 子图1: 速度剖面 + 误差带 -----
nexttile
hold on;
colors = lines(n_profiles); 

for i = 1:n_profiles
    dist = profile_data(i).distance_km;
    v    = profile_data(i).V_profile;
    err  = profile_data(i).V_error_profile;
    
    % 剔除 NaN，否则 fill 函数无法正常闭合渲染
    valid = ~isnan(v) & ~isnan(err);
    if any(valid)
        vv = v(valid);
        dd = dist(valid);
        ee = err(valid);
        
        % 构造 fill 的顶点：[x, flip(x)], [y_upper, flip(y_lower)]
        fill_x = [dd; flipud(dd)];
        fill_y = [vv + ee; flipud(vv - ee)];
        
        % 绘制误差带：颜色匹配线段，EdgeColor 设为 none 更美观
        fill(fill_x, fill_y, colors(i,:), ...
             'EdgeColor', 'none', 'FaceAlpha', 0.4, 'HandleVisibility', 'off');
    end
    
    % 绘制剖面主线
    plot(dist, v, '-', 'Color', colors(i,:), 'LineWidth', 2.5, ...
         'DisplayName', sprintf('%s (%.1f km)', profile_data(i).name, profile_data(i).total_length_km));
end

% 添加统一的误差说明到图例
plot(NaN, NaN, 's', 'MarkerFaceColor', [0.8 0.8 0.8], 'MarkerEdgeColor', 'none', ...
     'DisplayName', '\pm1\sigma uncertainty');

title('Ice Velocity Profiles with Uncertainty', 'FontSize', 22, 'FontWeight','bold');
xlabel("Distance along profile (km)", 'FontSize', 18);
ylabel("Velocity (m/year)", 'FontSize', 18);
grid on; box on; ylim([0, 1500]);
xlim([0, max([profile_data.total_length_km])]);
legend('Location','northwest','FontSize',14);
set(gca, 'FontSize', 16);

% ----- 子图2: 采样位置图 -----
nexttile
imagesc(x_vec, y_vec, V);
set(gca, 'YDir', 'normal');
hold on;
colormap(jet); colorbar; caxis([0, 1500]);

for i = 1:n_profiles
    pts = profile_data(i).points;
    plot(pts(:,1), pts(:,2), '--', 'Color', colors(i,:), 'LineWidth', 2, 'HandleVisibility','off');
    plot(pts(:,1), pts(:,2), 'o', 'MarkerSize', 8, 'MarkerFaceColor', colors(i,:), 'MarkerEdgeColor', 'w');
    
    text(pts(1,1), pts(1,2), [' ' profile_data(i).name], 'Color', 'w', 'FontWeight', 'bold', 'FontSize', 14);
end

title('Velocity Field & Profile Locations', 'FontSize', 22, 'FontWeight','bold');
xlabel('X (m)'); ylabel('Y (m)'); axis equal;
set(gca, 'FontSize', 16);

% 保存结果
saveas(fig1, fullfile(fig_dir, 'velocity_profiles_final.png'));
fprintf('\n .... Figure saved to: %s\n', fullfile(fig_dir, 'velocity_profiles_final.png'));

%% ========== 辅助函数：多段线采样 ==========
function [px, py, distances] = sample_along_polyline(points, n_samples)
    % 计算各段长度
    dx = diff(points(:, 1));
    dy = diff(points(:, 2));
    seg_lens = sqrt(dx.^2 + dy.^2);
    cum_dist = [0; cumsum(seg_lens)];
    
    % 生成均匀的累计距离采样点
    total_len = cum_dist(end);
    sample_d = linspace(0, total_len, n_samples)';
    
    % 利用 interp1 在路径上插值坐标
    px = interp1(cum_dist, points(:, 1), sample_d, 'linear');
    py = interp1(cum_dist, points(:, 2), sample_d, 'linear');
    distances = sample_d;
end