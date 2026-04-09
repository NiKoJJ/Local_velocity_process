clc; clear; close all;

%% ===== 1. 输入数据路径 =====
result_dir  = '/data2/Phd_Work1/ICE_Velocity_Process/GAMMA_POT_Output2/Mertz_100m/Tidal_IBE_Flexure_Correction/Vx_Vy_V_Weighted_Average/ALL';
group_name  = 'ALL';
istlive_dir = '/data2/Phd_Work1/ICE_Velocity_Process';

output_dir  = '/data2/Phd_Work1/ICE_Velocity_Process/Step4_visualize_velocity';
fig_dir     = fullfile(output_dir, 'figures');
if ~exist(fig_dir,'dir'); mkdir(fig_dir); end

n_pts_per_seg = 100;   % 每段采样点数
dpi           = 300;

%% ===== 2. 读取数据 =====
fprintf('读取数据...\n');
[V_gamma,         R_gamma]   = readgeoraster(fullfile(result_dir,  [group_name '-V_filled_nanmean.tif']));
[V_sigma_gamma,   ~]         = readgeoraster(fullfile(result_dir,  [group_name '-sigma_V.tif']));

[V_istlive,       R_istlive] = readgeoraster(fullfile(istlive_dir, 'Mertz_ITS_LIVE_velocity_120m_RGI19A_0000_v02_1_v.tif'));
[V_sigma_istlive, ~]         = readgeoraster(fullfile(istlive_dir, 'Mertz_ITS_LIVE_velocity_120m_RGI19A_0000_v02_v_error.tif'));

V_gamma         = double(V_gamma);
V_sigma_gamma   = double(V_sigma_gamma);
V_istlive       = double(V_istlive);
V_sigma_istlive = double(V_sigma_istlive);

bad_g = isinf(V_gamma);   V_gamma(bad_g)   = NaN;  V_sigma_gamma(bad_g)   = NaN;
bad_i = isinf(V_istlive); V_istlive(bad_i) = NaN;  V_sigma_istlive(bad_i) = NaN;

%% ===== 3. 坐标网格 =====
[rows_g, cols_g] = size(V_gamma);
x_vec_g = linspace(R_gamma.XWorldLimits(1),   R_gamma.XWorldLimits(2),   cols_g);
y_vec_g = linspace(R_gamma.YWorldLimits(2),   R_gamma.YWorldLimits(1),   rows_g); % carefully
[X_g, Y_g] = meshgrid(x_vec_g, y_vec_g);

[rows_i, cols_i] = size(V_istlive);
x_vec_i = linspace(R_istlive.XWorldLimits(1), R_istlive.XWorldLimits(2), cols_i);
y_vec_i = linspace(R_istlive.YWorldLimits(2), R_istlive.YWorldLimits(1), rows_i);
[X_i, Y_i] = meshgrid(x_vec_i, y_vec_i);

fprintf('GAMMA   grid: %d x %d, %.0f m\n',   rows_g, cols_g, R_gamma.SampleSpacingInWorldX);
fprintf('ITS-LIVE grid: %d x %d, %.0f m\n\n', rows_i, cols_i, R_istlive.CellExtentInWorldX);

%% ===== 4. ★ 鼠标交互选点 =====
fig_pick = figure('Units','centimeters','Position',[2 2 120 100], ...
                  'Color','w','Name','选点窗口');
ax_pick  = axes(fig_pick);

h_img = imagesc(ax_pick, x_vec_g/1e3, y_vec_g/1e3, V_gamma);
set(h_img,'AlphaData', ~isnan(V_gamma));
set(ax_pick,'YDir','normal');
axis(ax_pick,'equal','tight');
colormap(ax_pick, speed_colormap());
clim(ax_pick, [0, prctile(V_gamma(~isnan(V_gamma)), 98)]);
cb = colorbar(ax_pick);
cb.Label.String  = 'Ice Speed (m/day)';
cb.Label.FontSize = 10;
hold(ax_pick,'on');
title(ax_pick, '左键选点  |  右键/Delete 撤销  |  双击/Enter 确认完成', ...
      'FontSize',11,'FontWeight','bold','Color',[0.05 0.15 0.75]);
xlabel(ax_pick,'Easting (km, EPSG:3031)','FontSize',10);
ylabel(ax_pick,'Northing (km, EPSG:3031)','FontSize',10);

% ---- 初始化所有共享变量 ----
pts_km      = zeros(0,2);
h_pts       = gobjects(0);
h_labels    = gobjects(0);
h_line      = gobjects(0);
done        = false;
label_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';

% ---- 键盘回调（仅记录按键，不调用嵌套函数）----
set(fig_pick,'KeyPressFcn', @(~,e) setappdata(fig_pick,'key',e.Key));
setappdata(fig_pick,'key','');

fprintf('\n=== 开始选点 ===\n');
fprintf('  左键        : 添加控制点\n');
fprintf('  右键/Delete : 撤销最后一个点\n');
fprintf('  双击左键    : 确认完成\n');
fprintf('  Enter       : 确认完成\n\n');

% ---- 主交互循环 ----
while ~done && ishandle(fig_pick)

    setappdata(fig_pick,'key','');

    try
        [cx, cy, btn] = ginput(1);
    catch
        break;
    end

    if ~ishandle(fig_pick); break; end

    % 
    if isempty(btn)
        done = true; break;
    end
    btn = btn(1);   % 

    % 检查键盘（ginput 期间按键）
    key_now = getappdata(fig_pick,'key');
    if ismember(key_now,{'return','escape'})
        done = true; break;
    end
    if ismember(key_now,{'delete','backspace'})
        btn = -1;
    end

    % ========================
    %  左键：添加点
    % ========================
    if btn == 1

        % 双击检测（与上一点距离 < 0.5 km 视为双击）
        if ~isempty(pts_km) && ...
           sqrt((cx - pts_km(end,1))^2 + (cy - pts_km(end,2))^2) < 0.5
            done = true; break;
        end

        pts_km(end+1,:) = [cx, cy];
        n   = size(pts_km,1);
        lbl = label_chars(min(n, numel(label_chars)));

        % 画标记点
        hp = plot(ax_pick, cx, cy, 'wp', ...
                  'MarkerSize',12, 'MarkerFaceColor','k', 'LineWidth',0.7);
        ht = text(ax_pick, cx+0.3, cy, lbl, ...
                  'Color','w','FontSize',18,'FontWeight','bold');
        h_pts(end+1)    = hp;
        h_labels(end+1) = ht;

        % 重绘折线（内联，不依赖嵌套函数）
        if ~isempty(h_line) && all(isvalid(h_line)); delete(h_line); end
        h_line = gobjects(0);
        if size(pts_km,1) >= 2
            h_line = plot(ax_pick, pts_km(:,1), pts_km(:,2), ...
                          'w-','LineWidth',2,'HandleVisibility','off');
        end

        title(ax_pick, sprintf('已选 %d 个点  |  右键/Delete撤销  |  双击/Enter完成', n), ...
              'FontSize',10,'FontWeight','bold','Color',[0.05 0.55 0.05]);
        fprintf('  点 %s : (%.2f, %.2f) km  [共 %d 个]\n', lbl, cx, cy, n);

    % ========================
    %  右键 或 Delete：撤销
    % ========================
    elseif btn == 3 || btn == -1

        if isempty(pts_km)
            fprintf('  无点可撤销\n');
            continue;
        end

        pts_km(end,:) = [];

        if ~isempty(h_pts)    && isvalid(h_pts(end));    delete(h_pts(end));    h_pts(end)    = []; end
        if ~isempty(h_labels) && isvalid(h_labels(end)); delete(h_labels(end)); h_labels(end) = []; end

        % 重绘折线（内联）
        if ~isempty(h_line) && all(isvalid(h_line)); delete(h_line); end
        h_line = gobjects(0);
        if size(pts_km,1) >= 2
            h_line = plot(ax_pick, pts_km(:,1), pts_km(:,2), ...
                          'w-','LineWidth',2,'HandleVisibility','off');
        end

        n = size(pts_km,1);
        title(ax_pick, sprintf('已选 %d 个点  |  右键/Delete撤销  |  双击/Enter完成', n), ...
              'FontSize',10,'FontWeight','bold','Color',[0.75 0.15 0.05]);
        fprintf('  ← 撤销，当前 %d 个点\n', n);
    end
end

% ---- 检查有效点数 ----
if ~ishandle(fig_pick) || size(pts_km,1) < 2
    warning('选点不足 2 个或窗口已关闭，退出。');
    return;
end

n_ctrl         = size(pts_km,1);
profile_pts    = pts_km * 1e3;                           % km → m
profile_labels = cellstr(label_chars(1:n_ctrl)')';

fprintf('\n=== 选点完成，共 %d 个控制点 ===\n', n_ctrl);

%% ===== 5. 分段生成折线剖面采样点 =====
n_segs = n_ctrl - 1;
if isscalar(n_pts_per_seg)
    n_pts_per_seg = repmat(n_pts_per_seg, 1, n_segs);
end

px = [];  py = [];
seg_start_idx = zeros(1, n_ctrl);

for s = 1:n_segs
    x1 = profile_pts(s,1);   y1 = profile_pts(s,2);
    x2 = profile_pts(s+1,1); y2 = profile_pts(s+1,2);
    n  = n_pts_per_seg(s);
    sx = linspace(x1, x2, n);
    sy = linspace(y1, y2, n);
    if s > 1; sx = sx(2:end); sy = sy(2:end); end
    seg_start_idx(s) = numel(px) + 1;
    px = [px, sx]; %#ok<AGROW>
    py = [py, sy]; %#ok<AGROW>
end
seg_start_idx(end) = numel(px);

dist_km      = [0, cumsum(sqrt(diff(px).^2 + diff(py).^2))] / 1000;
ctrl_dist_km = dist_km(seg_start_idx);

fprintf('剖面总长度: %.2f km，%d 段，%d 个采样点\n\n', ...
    max(dist_km), n_segs, numel(px));

%% ===== 6. 沿剖面插值 =====
V_prof_gamma         = interp2(X_g, Y_g, V_gamma,         px, py, 'linear', NaN);
V_sigma_prof_gamma   = interp2(X_g, Y_g, V_sigma_gamma,   px, py, 'linear', NaN);
V_prof_istlive       = interp2(X_i, Y_i, V_istlive,       px, py, 'linear', NaN);
V_sigma_prof_istlive = interp2(X_i, Y_i, V_sigma_istlive, px, py, 'linear', NaN);

%% ===== 7. 更新选点图：叠加最终采样折线并保存 =====
plot(ax_pick, px/1e3, py/1e3, 'c-','LineWidth',0.8,'HandleVisibility','off');
title(ax_pick, sprintf('已选 %d 个控制点  —  剖面总长 %.1f km', n_ctrl, max(dist_km)), ...
      'FontSize',15,'FontWeight','bold','Color',[0 0.5 0]);
drawnow;

out_pick = fullfile(fig_dir, [group_name '_picked_profile_map.png']);
exportgraphics(fig_pick, out_pick, 'Resolution', dpi);
fprintf('选点地图已保存: %s\n', out_pick);

%% ===== 8. 绘制剖面对比图 =====
fig2 = figure('Units','centimeters','Position',[2 2 26 18],'Color','w');

% ---- 子图1：GAMMA 剖面 ----
ax1 = subplot(2,1,1);
hold on;
fill([dist_km, fliplr(dist_km)], ...
     [V_prof_gamma + V_sigma_prof_gamma, fliplr(V_prof_gamma - V_sigma_prof_gamma)], ...
     [0.7 0.85 1.0],'EdgeColor','none','FaceAlpha',0.6,'DisplayName','GAMMA ±1σ');
plot(dist_km, V_prof_gamma,'b-','LineWidth',1.5,'DisplayName','GAMMA Speed');
add_ctrl_labels(ax1, ctrl_dist_km, profile_labels);
hold off;
xlabel('Distance along profile (km)','FontSize',10);
ylabel('Speed (m/day)','FontSize',10);
title(sprintf('%s — GAMMA Speed Profile', group_name),'FontSize',11,'FontWeight','bold');
legend('Location','best','FontSize',9);
xlim([0 max(dist_km)]); grid on; box on; set(ax1,'FontSize',9);

% ---- 子图2：GAMMA vs ITS-LIVE 对比 ----
ax2 = subplot(2,1,2);
hold on;
fill([dist_km, fliplr(dist_km)], ...
     [V_prof_gamma + V_sigma_prof_gamma, fliplr(V_prof_gamma - V_sigma_prof_gamma)], ...
     [0.7 0.85 1.0],'EdgeColor','none','FaceAlpha',0.5,'DisplayName','GAMMA ±1σ');
fill([dist_km, fliplr(dist_km)], ...
     [V_prof_istlive + V_sigma_prof_istlive, fliplr(V_prof_istlive - V_sigma_prof_istlive)], ...
     [1.0 0.85 0.7],'EdgeColor','none','FaceAlpha',0.5,'DisplayName','ITS-LIVE ±1σ');
plot(dist_km, V_prof_gamma,   'b-', 'LineWidth',1.5,'DisplayName','GAMMA');
plot(dist_km, V_prof_istlive, 'r--','LineWidth',1.5,'DisplayName','ITS-LIVE');
add_ctrl_labels(ax2, ctrl_dist_km, profile_labels);
hold off;
xlabel('Distance along profile (km)','FontSize',10);
ylabel('Speed (m/day)','FontSize',10);
title('GAMMA vs ITS-LIVE Speed Profile Comparison','FontSize',11,'FontWeight','bold');
legend('Location','best','FontSize',9);
xlim([0 max(dist_km)]); grid on; box on; set(ax2,'FontSize',9);

out_fig = fullfile(fig_dir, sprintf('%s_speed_profile_comparison.png', group_name));
exportgraphics(fig2, out_fig, 'Resolution', dpi);
fprintf('剖面对比图已保存: %s\n', out_fig);

%% ===== 9. 保存控制点坐标到 CSV =====
out_csv = fullfile(fig_dir, [group_name '_profile_points.csv']);
fid = fopen(out_csv,'w');
fprintf(fid,'Label,X_m,Y_m,X_km,Y_km,Dist_km\n');
for k = 1:n_ctrl
    fprintf(fid,'%s,%.2f,%.2f,%.4f,%.4f,%.4f\n', ...
        profile_labels{k}, ...
        profile_pts(k,1), profile_pts(k,2), ...
        profile_pts(k,1)/1e3, profile_pts(k,2)/1e3, ...
        ctrl_dist_km(k));
end
fclose(fid);
fprintf('控制点坐标已保存: %s\n', out_csv);

fprintf('\n=== 全部完成 ===\n');

%% =========================================================================
%%  辅助函数
%% =========================================================================

% 在剖面图上添加控制点竖线 + 字母标注
function add_ctrl_labels(ax, ctrl_dist_km, labels)
    yl = ylim(ax);
    for k = 1:numel(ctrl_dist_km)
        xline(ax, ctrl_dist_km(k), '--', ...
              'Color',[0.5 0.5 0.5],'LineWidth',0.8,'HandleVisibility','off');
        text(ax, ctrl_dist_km(k), yl(2) - 0.04*diff(yl), labels{k}, ...
             'FontSize',9,'Color',[0.2 0.2 0.2], ...
             'HorizontalAlignment','center','FontWeight','bold');
    end
end

% 仿 ITS-LIVE 风格色表
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
