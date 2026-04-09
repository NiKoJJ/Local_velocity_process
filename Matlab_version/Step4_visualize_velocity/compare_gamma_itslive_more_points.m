clc; clear; close all;

%% ===== 1. 输入数据路径 =====
result_dir  = '/data2/Phd_Work1/ICE_Velocity_Process/GAMMA_POT_Output2/Mertz_100m/Tidal_IBE_Flexure_Correction/Vx_Vy_V_Weighted_Average/ALL';
group_name  = 'ALL';
istlive_dir = '/data2/Phd_Work1/ICE_Velocity_Process';

%% ===== 2. 读取数据 =====
[V_gamma,         R_gamma]   = readgeoraster(fullfile(result_dir,  [group_name '-V.tif']));
[V_sigma_gamma,   ~]         = readgeoraster(fullfile(result_dir,  [group_name '-sigma_V.tif']));
[V_istlive,       R_istlive] = readgeoraster(fullfile(istlive_dir, 'Mertz_ITS_LIVE_velocity_120m_RGI19A_0000_v02_v.tif'));
[V_sigma_istlive, ~]         = readgeoraster(fullfile(istlive_dir, 'Mertz_ITS_LIVE_velocity_120m_RGI19A_0000_v02_v_error.tif'));

V_gamma         = double(V_gamma);
V_sigma_gamma   = double(V_sigma_gamma);
V_istlive       = double(V_istlive);
V_sigma_istlive = double(V_sigma_istlive);

%% ===== 3. 输出目录 =====
output_dir = '/data2/Phd_Work1/ICE_Velocity_Process/Step4_visualize_velocity';
fig_dir    = fullfile(output_dir, 'figures');
if ~exist(fig_dir,'dir'); mkdir(fig_dir); end

%% ===== 4. ★ 多点剖面定义（核心改动）=====
% 按顺序填入所有控制点坐标（EPSG:3031，单位 m）
% 每行 [x, y]，程序自动连接 A→B→C→... 并计算累计距离
profile_pts = [
    1464864.87	-1933163.7
1454047.54	-1943981.03
1435259.54	-1963338.36
1427288.87	-1986111.7
1426150.21	-2019133.03
1427858.21	-2061263.7
1429566.21	-2077774.36
];
% 控制点标签（与 profile_pts 行数一致）
profile_labels = {'A', 'B', 'C', 'D', 'E','F','G'};

% 每段的采样点数（标量 = 全段统一；向量 = 每段单独指定，长度 = 段数）
n_pts_per_seg = 100;   % 每两个控制点之间的插值点数

%% ===== 5. 清除无效值 =====
bad_g = isinf(V_gamma);
V_gamma(bad_g)       = NaN;
V_sigma_gamma(bad_g) = NaN;

bad_i = isinf(V_istlive);
V_istlive(bad_i)       = NaN;
V_sigma_istlive(bad_i) = NaN;

%% ===== 6. 坐标网格 =====
[rows_g, cols_g] = size(V_gamma);
x_vec_g = linspace(R_gamma.XWorldLimits(1),   R_gamma.XWorldLimits(2),   cols_g);
y_vec_g = linspace(R_gamma.YWorldLimits(2),   R_gamma.YWorldLimits(1),   rows_g);
[X_g, Y_g] = meshgrid(x_vec_g, y_vec_g);

[rows_i, cols_i] = size(V_istlive);
x_vec_i = linspace(R_istlive.XWorldLimits(1), R_istlive.XWorldLimits(2), cols_i);
y_vec_i = linspace(R_istlive.YWorldLimits(2), R_istlive.YWorldLimits(1), rows_i);
[X_i, Y_i] = meshgrid(x_vec_i, y_vec_i);

fprintf('GAMMA   grid: %d x %d, %.0f m\n', rows_g, cols_g, R_gamma.SampleSpacingInWorldX);
fprintf('ITS-LIVE grid: %d x %d, %.0f m\n\n', rows_i, cols_i, R_istlive.CellExtentInWorldX);

%% ===== 7. ★ 分段生成折线剖面采样点 =====
n_segs = size(profile_pts, 1) - 1;

% 每段采样点数处理（支持标量或向量）
if isscalar(n_pts_per_seg)
    n_pts_per_seg = repmat(n_pts_per_seg, 1, n_segs);
end

px = [];  py = [];
seg_start_idx = zeros(1, n_segs+1);   % 记录每个控制点在 px/py 中的位置

for s = 1:n_segs
    x1 = profile_pts(s,   1);  y1 = profile_pts(s,   2);
    x2 = profile_pts(s+1, 1);  y2 = profile_pts(s+1, 2);
    n  = n_pts_per_seg(s);

    % 避免控制点重复（除第一段外，跳过段首）
    if s == 1
        seg_x = linspace(x1, x2, n);
        seg_y = linspace(y1, y2, n);
        seg_start_idx(s) = 1;
    else
        seg_x = linspace(x1, x2, n);
        seg_y = linspace(y1, y2, n);
        seg_x = seg_x(2:end);   % 去掉首点（上段末点）
        seg_y = seg_y(2:end);
        seg_start_idx(s) = numel(px) + 1;
    end
    px = [px, seg_x];
    py = [py, seg_y];
end
seg_start_idx(end) = numel(px);   % 最后一个控制点位置

% 累计距离（km）
dx_seg    = diff(px);
dy_seg    = diff(py);
dist_km   = [0, cumsum(sqrt(dx_seg.^2 + dy_seg.^2))] / 1000;

% 各控制点对应的累计距离
ctrl_dist_km = dist_km(seg_start_idx);

fprintf('剖面总长度: %.2f km，共 %d 段，%d 个采样点\n\n', ...
    max(dist_km), n_segs, numel(px));

%% ===== 8. 沿剖面插值 =====
V_prof_gamma         = interp2(X_g, Y_g, V_gamma,         px, py, 'linear', NaN);
V_sigma_prof_gamma   = interp2(X_g, Y_g, V_sigma_gamma,   px, py, 'linear', NaN);
V_prof_istlive       = interp2(X_i, Y_i, V_istlive,       px, py, 'linear', NaN);
V_sigma_prof_istlive = interp2(X_i, Y_i, V_sigma_istlive, px, py, 'linear', NaN);

%% ===== 9. 绘制剖面对比图 =====
fig1 = figure('Units','centimeters','Position',[2 2 26 18],'Color','w');

% ---- 子图1：GAMMA 剖面 ----
ax1 = subplot(2,1,1);
hold on;
fill([dist_km, fliplr(dist_km)], ...
     [V_prof_gamma + V_sigma_prof_gamma, fliplr(V_prof_gamma - V_sigma_prof_gamma)], ...
     [0.7 0.85 1.0], 'EdgeColor','none','FaceAlpha',0.6,'DisplayName','GAMMA ±1σ');
plot(dist_km, V_prof_gamma, 'b-', 'LineWidth',1.5, 'DisplayName','GAMMA Speed');
% ★ 控制点垂直标注线
add_ctrl_labels(ax1, ctrl_dist_km, profile_labels);
hold off;
xlabel('Distance along profile (km)','FontSize',10);
ylabel('Speed (m/day)','FontSize',10);
title(sprintf('%s — GAMMA Speed Profile', group_name),'FontSize',11,'FontWeight','bold');
legend('Location','best','FontSize',9);
xlim([0, max(dist_km)]); grid on; box on; set(ax1,'FontSize',9);

% ---- 子图2：GAMMA vs ITS-LIVE 对比 ----
ax2 = subplot(2,1,2);
hold on;
fill([dist_km, fliplr(dist_km)], ...
     [V_prof_gamma + V_sigma_prof_gamma, fliplr(V_prof_gamma - V_sigma_prof_gamma)], ...
     [0.7 0.85 1.0], 'EdgeColor','none','FaceAlpha',0.5,'DisplayName','GAMMA ±1σ');
fill([dist_km, fliplr(dist_km)], ...
     [V_prof_istlive + V_sigma_prof_istlive, fliplr(V_prof_istlive - V_sigma_prof_istlive)], ...
     [1.0 0.85 0.7], 'EdgeColor','none','FaceAlpha',0.5,'DisplayName','ITS-LIVE ±1σ');
plot(dist_km, V_prof_gamma,   'b-',  'LineWidth',1.5, 'DisplayName','GAMMA');
plot(dist_km, V_prof_istlive, 'r--', 'LineWidth',1.5, 'DisplayName','ITS-LIVE');
add_ctrl_labels(ax2, ctrl_dist_km, profile_labels);
hold off;
xlabel('Distance along profile (km)','FontSize',10);
ylabel('Speed (m/day)','FontSize',10);
title('GAMMA vs ITS-LIVE Speed Profile Comparison','FontSize',11,'FontWeight','bold');
legend('Location','best','FontSize',9);
xlim([0, max(dist_km)]); grid on; box on; set(ax2,'FontSize',9);

out_fig = fullfile(fig_dir, sprintf('%s_speed_profile_comparison.png', group_name));
exportgraphics(fig1, out_fig, 'Resolution',300);
fprintf('剖面图已保存: %s\n', out_fig);

%% ===== 10. 速度场底图 + 折线剖面标注 =====
fig2 = figure('Units','centimeters','Position',[2 2 20 18],'Color','w');
ax_map = axes;

% 速度底图（用 GAMMA）
imagesc(x_vec_g/1e3, y_vec_g/1e3, V_gamma);
set(gca,'YDir','normal'); axis equal tight;
colormap(ax_map, speed_colormap());
clim([0 max(V_gamma(:), [], 'omitnan') * 0.8]);
cb = colorbar; cb.Label.String = 'Ice Speed (m/day)'; cb.Label.FontSize=10;
hold on;

% 绘制折线剖面
plot(px/1e3, py/1e3, 'w-',  'LineWidth',2.5, 'DisplayName','Profile');
plot(px/1e3, py/1e3, 'k--', 'LineWidth',1.2, 'HandleVisibility','off');

% 控制点标记与标签
marker_styles = {'ws','wo','w^','wd','wp'};   % 可扩展
for k = 1:size(profile_pts,1)
    mk = marker_styles{min(k, numel(marker_styles))};
    plot(profile_pts(k,1)/1e3, profile_pts(k,2)/1e3, mk, ...
         'MarkerSize',8, 'MarkerFaceColor','k');
    text(profile_pts(k,1)/1e3, profile_pts(k,2)/1e3, ...
         ['  ' profile_labels{k}], ...
         'Color','w','FontSize',10,'FontWeight','bold');
end
hold off;
xlabel('Easting (km, EPSG:3031)','FontSize',11);
ylabel('Northing (km, EPSG:3031)','FontSize',11);
title(sprintf('%s — Ice Velocity & Profile', group_name),'FontSize',12,'FontWeight','bold');
set(ax_map,'FontSize',10,'DataAspectRatio',[1 1 1]);

out_map = fullfile(fig_dir, sprintf('%s_velocity_map_profile.png', group_name));
exportgraphics(fig2, out_map, 'Resolution',300);
fprintf('速度场图已保存: %s\n', out_map);

%% ===== 辅助函数 =====

% 在剖面图上添加控制点标注竖线
function add_ctrl_labels(ax, ctrl_dist_km, labels)
    yl = ylim(ax);
    for k = 1:numel(ctrl_dist_km)
        xline(ax, ctrl_dist_km(k), '--', 'Color',[0.5 0.5 0.5], ...
              'LineWidth',0.8, 'HandleVisibility','off');
        text(ax, ctrl_dist_km(k), yl(2)*0.97, labels{k}, ...
             'FontSize',9, 'Color',[0.3 0.3 0.3], ...
             'HorizontalAlignment','center', 'FontWeight','bold');
    end
end

% 仿 ITS-LIVE 色表
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