function plot_monthly_profiles(data_dir, varargin)
% PLOT_MONTHLY_PROFILES  Monthly ice velocity profiles along a transect.
%
% USAGE
%   % Points in WGS-84 lat/lon
%   plot_monthly_profiles('./monthly_V_Error', ...
%       'points_latlon', [158.5,-68.2; 160.0,-67.8; 162.0,-67.5], ...
%       'labels',        {'A','B','C'}, ...
%       'mode',          'all', ...
%       'smooth',        'savgol', ...
%       'smooth_window', 15, ...
%       'output',        'profile.png');
%
%   % Points in EPSG:3031 (metres)
%   plot_monthly_profiles('./monthly_V_Error', ...
%       'points_3031', [974000,-2100000; 1050000,-2050000], ...
%       'mode', 'hovmoller');
%
% MODES
%   hovmoller       Time x distance heatmap  (cleanest, recommended)
%   mean_anomaly    Mean �� std + seasonal anomaly panel
%   small_multiples One sub-panel per month
%   all             Save all three
%
% SMOOTHING
%   none       raw values
%   moving_avg boxcar rolling mean
%   savgol     Savitzky-Golay (preserves peaks, recommended)
%   gaussian   Gaussian kernel

%% ���� Parse inputs ��������������������������������������������������������������������������������������������������������������������
p = inputParser;
addRequired(p, 'data_dir', @(x) ischar(x)||isstring(x));
addParameter(p, 'points_latlon', [],  @isnumeric);  % Nx2 [lon lat]
addParameter(p, 'points_3031',   [],  @isnumeric);  % Nx2 [x y]
addParameter(p, 'labels',        {}, @iscell);
addParameter(p, 'n_pts',         200, @isnumeric);
addParameter(p, 'output',        'profile_monthly.png', @(x) ischar(x)||isstring(x));
addParameter(p, 'unit',          'm/yr',  @(x) ischar(x)||isstring(x));
addParameter(p, 'vmin',          [],  @isnumeric);
addParameter(p, 'vmax',          [],  @isnumeric);
addParameter(p, 'ncols',         4,   @isnumeric);
addParameter(p, 'mode',          'hovmoller', ...
    @(x) ismember(x,{'hovmoller','mean_anomaly','small_multiples','all'}));
addParameter(p, 'smooth',        'savgol', ...
    @(x) ismember(x,{'none','moving_avg','savgol','gaussian'}));
addParameter(p, 'smooth_window', 11, @isnumeric);
parse(p, data_dir, varargin{:});
opt = p.Results;
data_dir = char(opt.data_dir);

%% ���� Control points ����������������������������������������������������������������������������������������������������������������
if ~isempty(opt.points_latlon)
    ctrl_xy = latlon_to_3031(opt.points_latlon(:,1), opt.points_latlon(:,2));
    fprintf('Converted %d lat/lon points to EPSG:3031\n', size(ctrl_xy,1));
elseif ~isempty(opt.points_3031)
    ctrl_xy = opt.points_3031;
else
    error('Provide points_latlon or points_3031.');
end

n_ctrl = size(ctrl_xy, 1);
if n_ctrl < 2; error('At least 2 control points required.'); end

if isempty(opt.labels)
    ctrl_labels = arrayfun(@(i) char(64+i), 1:n_ctrl, 'UniformOutput', false);
else
    ctrl_labels = opt.labels(1:n_ctrl);
end

%% ���� Build profile ������������������������������������������������������������������������������������������������������������������
[px, py, dist_km, ctrl_dist_km] = build_profile(ctrl_xy, opt.n_pts);
fprintf('Profile: %d pts,  %.2f km\n', numel(px), max(dist_km));

%% ���� Discover monthly TIF pairs ����������������������������������������������������������������������������������������
pairs = discover_pairs(data_dir);
if isempty(pairs)
    error('No monthly v/v_err TIF pairs found in: %s', data_dir);
end
fprintf('Found %d monthly pairs (%s �C %s)\n', ...
    numel(pairs), pairs(1).label, pairs(end).label);

%% ���� Sample rasters ����������������������������������������������������������������������������������������������������������������
profiles(numel(pairs)) = struct('label','','v',[],'err',[]);
for k = 1:numel(pairs)
    fprintf('  Sampling %s ...', pairs(k).label);
    v   = sample_tif(pairs(k).v_path,   px, py);
    err = sample_tif(pairs(k).err_path, px, py);

    % Mask only truly invalid errors (keep small positives)
    err(~isfinite(err)) = NaN;
    err(err < 0)        = NaN;
    err(err == 0)       = NaN;

    pct_v   = sum(isfinite(v))   / numel(v)   * 100;
    pct_err = sum(isfinite(err)) / numel(err) * 100;
    fprintf('  v=%.0f%%  err=%.0f%%\n', pct_v, pct_err);
    profiles(k).label = pairs(k).label;
    profiles(k).v     = v;
    profiles(k).err   = err;
end

%% ���� Smooth ��������������������������������������������������������������������������������������������������������������������������������
if ~strcmp(opt.smooth, 'none')
    fprintf('Smoothing: method=%s  window=%d\n', opt.smooth, opt.smooth_window);
    for k = 1:numel(profiles)
        profiles(k).v   = smooth_profile(profiles(k).v,   opt.smooth, opt.smooth_window);
        profiles(k).err = smooth_profile(profiles(k).err, opt.smooth, opt.smooth_window);
    end
end

%% ���� Plot ������������������������������������������������������������������������������������������������������������������������������������
[out_dir, stem, ext] = fileparts(opt.output);
if isempty(out_dir); out_dir = '.'; end
if isempty(ext);     ext     = '.png'; end

modes = {'hovmoller','mean_anomaly','small_multiples'};
if ~strcmp(opt.mode, 'all'); modes = {opt.mode}; end

for mi = 1:numel(modes)
    mode = modes{mi};
    if strcmp(opt.mode, 'all')
        out_path = fullfile(out_dir, [stem '_' mode ext]);
    else
        out_path = opt.output;
    end

    switch mode
        case 'hovmoller'
            fig_hovmoller(profiles, dist_km, ctrl_dist_km, ctrl_labels, ...
                           out_path, opt);
        case 'mean_anomaly'
            fig_mean_anomaly(profiles, dist_km, ctrl_dist_km, ctrl_labels, ...
                              out_path, opt);
        case 'small_multiples'
            fig_small_multiples(profiles, dist_km, ctrl_dist_km, ctrl_labels, ...
                                 out_path, opt);
    end
end
end


%% =========================================================================
%%  FIGURE FUNCTIONS
%% =========================================================================

function fig_hovmoller(profiles, dist_km, ctrl_dist_km, ctrl_labels, out_path, opt)
% Time �� distance heatmap + mean error panel.

n_t = numel(profiles);
n_d = numel(dist_km);
V_mat   = nan(n_t, n_d);
Err_mat = nan(n_t, n_d);
for k = 1:n_t
    V_mat(k,:)   = profiles(k).v;
    Err_mat(k,:) = profiles(k).err;
end

vlo = pct_finite(V_mat(:), 2);   vhi = pct_finite(V_mat(:), 98);
if ~isempty(opt.vmin); vlo = opt.vmin; end
if ~isempty(opt.vmax); vhi = opt.vmax; end

fig = figure('Visible','off','Color','k','Position',[0 0 1400 700]);
t   = tiledlayout(fig, 4, 1, 'TileSpacing','compact','Padding','compact');

% ���� heatmap ������������������������������������������������������������������������������������������������������������������������������
ax1 = nexttile(t, [3 1]);
imagesc(ax1, dist_km, 1:n_t, V_mat, [vlo vhi]);
set(ax1, 'YDir','normal', 'Color','k', 'XColor','#999', 'YColor','#999', ...
    'FontSize', 9, 'TickDir','out');
yticks(ax1, 1:n_t);
yticklabels(ax1, {profiles.label});
ylabel(ax1, 'Month', 'Color','#ccc', 'FontSize', 11);
title(ax1, 'Hovm?ller diagram �� monthly ice velocity', 'Color','w', 'FontSize', 12);
colormap(ax1, rdylbu_r(256));
cb = colorbar(ax1, 'Color','#999');
cb.Label.String   = sprintf('Speed (%s)', opt.unit);
cb.Label.Color    = '#ccc';
cb.Label.FontSize = 10;
clim(ax1, [vlo vhi]);
hold(ax1, 'on');
for k = 1:numel(ctrl_dist_km)
    xline(ax1, ctrl_dist_km(k), '--w', 'LineWidth', 0.8, 'Alpha', 0.5);
    text(ax1, ctrl_dist_km(k)+0.3, 0.3, ctrl_labels{k}, ...
        'Color','w','FontSize',8,'FontWeight','bold');
end
hold(ax1, 'off');
ax1.XTickLabel = {};

% ���� mean error bar ����������������������������������������������������������������������������������������������������������������
ax2 = nexttile(t, [1 1]);
mean_err = mean(Err_mat, 1, 'omitnan');
q25_err  = prctile(Err_mat, 25, 1);
q75_err  = prctile(Err_mat, 75, 1);
set(ax2,'Color','k','XColor','#999','YColor','#999','FontSize',9);
hold(ax2,'on');
fill([dist_km, fliplr(dist_km)], [q25_err, fliplr(q75_err)], ...
    [0.91 0.55 0.18], 'FaceAlpha',0.3, 'EdgeColor','none');
plot(ax2, dist_km, mean_err, 'Color',[0.91 0.55 0.18], 'LineWidth',1.5);
hold(ax2,'off');
ylabel(ax2, sprintf('\x3c3 (%s)', opt.unit), 'Color','#ccc', 'FontSize',9);
xlabel(ax2, 'Distance along profile (km)', 'Color','#ccc', 'FontSize',11);
xlim(ax2, [0 max(dist_km)]);
grid(ax2,'on'); ax2.GridColor = '#2a2a2a';
for k = 1:numel(ctrl_dist_km)
    xline(ax2, ctrl_dist_km(k), '--', 'Color','#888', 'LineWidth',0.6, 'Alpha',0.4);
end

save_fig(fig, out_path);
fprintf('Saved (hovmoller) -> %s\n', out_path);
end


function fig_mean_anomaly(profiles, dist_km, ctrl_dist_km, ctrl_labels, out_path, opt)
n_t   = numel(profiles);
V_mat = nan(n_t, numel(dist_km));
for k = 1:n_t; V_mat(k,:) = profiles(k).v; end
mean_v = mean(V_mat, 1, 'omitnan');
std_v  = std(V_mat, 0, 1, 'omitnan');

fig = figure('Visible','off','Color','k','Position',[0 0 1400 700]);
t   = tiledlayout(fig, 3, 1, 'TileSpacing','compact','Padding','compact');

% ���� mean �� std ������������������������������������������������������������������������������������������������������������������������
ax1 = nexttile(t, [2 1]);
set(ax1,'Color','k','XColor','#999','YColor','#999','FontSize',9);
hold(ax1,'on');
fill([dist_km, fliplr(dist_km)], ...
     [mean_v+std_v, fliplr(mean_v-std_v)], ...
     [0.27 0.51 0.71], 'FaceAlpha',0.22, 'EdgeColor','none');
for k = 1:n_t
    mo  = profiles(k).label(6:7);
    col = season_color(mo);
    ok  = isfinite(profiles(k).v);
    plot(ax1, dist_km(ok), profiles(k).v(ok), 'Color', [col 0.55], 'LineWidth', 0.8);
end
plot(ax1, dist_km, mean_v, 'w', 'LineWidth', 2.2);
hold(ax1,'off');
ylabel(ax1, sprintf('Ice speed (%s)', opt.unit), 'Color','#ccc','FontSize',11);
title(ax1, 'Monthly velocity: mean �� variability + seasonal anomaly', ...
    'Color','w','FontSize',12);
if ~isempty(opt.vmin); ylim(ax1, [opt.vmin, ax1.YLim(2)]); end
if ~isempty(opt.vmax); ylim(ax1, [ax1.YLim(1), opt.vmax]); end
grid(ax1,'on'); ax1.GridColor = '#2a2a2a';
ax1.XTickLabel = {};

% Season legend
legend_patches(ax1);
add_ctrl_lines(ax1, ctrl_dist_km, ctrl_labels, ax1.YLim(2));

% ���� anomaly ������������������������������������������������������������������������������������������������������������������������������
ax2 = nexttile(t, [1 1]);
set(ax2,'Color','k','XColor','#999','YColor','#999','FontSize',9);
hold(ax2,'on');
fill([dist_km, fliplr(dist_km)], ...
     [-std_v, fliplr(std_v)], ...
     [0.27 0.51 0.71], 'FaceAlpha',0.12, 'EdgeColor','none');
for k = 1:n_t
    mo   = profiles(k).label(6:7);
    col  = season_color(mo);
    anom = profiles(k).v - mean_v;
    ok   = isfinite(anom);
    plot(ax2, dist_km(ok), anom(ok), 'Color',[col 0.75], 'LineWidth',0.9);
end
yline(ax2, 0, '--w', 'LineWidth',0.8, 'Alpha',0.5);
hold(ax2,'off');
ylabel(ax2, sprintf('Anomaly (%s)', opt.unit), 'Color','#ccc','FontSize',10);
xlabel(ax2, 'Distance along profile (km)', 'Color','#ccc','FontSize',11);
xlim(ax2, [0, max(dist_km)]);
xlim(ax1, [0, max(dist_km)]);
grid(ax2,'on'); ax2.GridColor = '#2a2a2a';
add_ctrl_lines(ax2, ctrl_dist_km, {}, []);

save_fig(fig, out_path);
fprintf('Saved (mean_anomaly) -> %s\n', out_path);
end


function fig_small_multiples(profiles, dist_km, ctrl_dist_km, ctrl_labels, out_path, opt)
n_months = numel(profiles);
ncols    = min(opt.ncols, n_months);
nrows    = ceil(n_months / ncols);

all_v = cellfun(@(p) p.v, num2cell(profiles), 'UniformOutput', false);
all_v = cat(1, all_v{:});
vlo   = pct_finite(all_v(:), 1);   vhi = pct_finite(all_v(:), 99);
if ~isempty(opt.vmin); vlo = opt.vmin; end
if ~isempty(opt.vmax); vhi = opt.vmax; end

fig = figure('Visible','off','Color','k', ...
    'Position',[0 0 ncols*350 nrows*250]);
t   = tiledlayout(fig, nrows, ncols, 'TileSpacing','compact','Padding','compact');

for k = 1:n_months
    ax  = nexttile(t);
    set(ax,'Color','k','XColor','#999','YColor','#999','FontSize',8);
    mo  = profiles(k).label(6:7);
    col = season_color(mo);

    draw_band_line(ax, dist_km, profiles(k).v, profiles(k).err, col, 0.35, 1.3);

    for j = 1:numel(ctrl_dist_km)
        xline(ax, ctrl_dist_km(j), '--', 'Color','#444', 'LineWidth',0.6);
        text(ax, ctrl_dist_km(j), vhi*0.97, ctrl_labels{j}, ...
            'HorizontalAlignment','center','VerticalAlignment','top',...
            'FontSize',6,'Color','#777');
    end

    title(ax, profiles(k).label, 'Color','w', 'FontSize',9);
    ylim(ax, [vlo vhi]);
    xlim(ax, [0, max(dist_km)]);
    grid(ax,'on'); ax.GridColor = '#2a2a2a';
end

% Hide unused tiles
for k = n_months+1 : nrows*ncols
    ax = nexttile(t); axis(ax,'off');
end

% Shared axis labels
annotation(fig,'textbox',[0.5 0 0.5 0.03],'String',...
    'Distance along profile (km)','Color','#ccc','FontSize',11,...
    'HorizontalAlignment','center','EdgeColor','none','FitBoxToText','off');
annotation(fig,'textbox',[0 0.3 0.03 0.4],'String',...
    sprintf('Ice speed (%s)', opt.unit),'Color','#ccc','FontSize',11,...
    'HorizontalAlignment','center','Rotation',90,'EdgeColor','none');
title(t, sprintf('Monthly ice velocity profiles  (shading = \xb11\x3c3)'), ...
    'Color','w','FontSize',13);

save_fig(fig, out_path);
fprintf('Saved (small_multiples) -> %s\n', out_path);
end


%% =========================================================================
%%  LOCAL HELPERS
%% =========================================================================

function pairs = discover_pairs(data_dir)
% Find all YYYY-MM_v.tif + YYYY-MM_v_err.tif pairs.
files = dir(fullfile(data_dir, '*_v.tif'));
pairs = struct('label',{},'v_path',{},'err_path',{});
for i = 1:numel(files)
    fname = files(i).name;
    tok   = regexp(fname, '^(\d{4}-\d{2})_v\.tif$', 'tokens', 'once');
    if isempty(tok); continue; end
    label    = tok{1};
    err_path = fullfile(data_dir, [label '_v_err.tif']);
    if ~isfile(err_path)
        fprintf('  [warn] no error file for %s\n', label); continue;
    end
    pairs(end+1).label    = label;          %#ok<AGROW>
    pairs(end).v_path     = fullfile(data_dir, fname);
    pairs(end).err_path   = err_path;
end
[~, ord] = sort({pairs.label});
pairs = pairs(ord);
end


function ctrl_xy = latlon_to_3031(lons, lats)
% Convert WGS-84 lon/lat to EPSG:3031 using MATLAB projcrs.
proj    = projcrs(3031);
[xs,ys] = projfwd(proj, lats(:), lons(:));
ctrl_xy = [xs(:), ys(:)];
end


function [px, py, dist_km, ctrl_dist_km] = build_profile(ctrl_xy, n_pts)
% Piecewise-linear profile between control points.
n_segs   = size(ctrl_xy, 1) - 1;
px = []; py = []; ctrl_idx = 1;
for s = 1:n_segs
    sx = linspace(ctrl_xy(s,1), ctrl_xy(s+1,1), n_pts);
    sy = linspace(ctrl_xy(s,2), ctrl_xy(s+1,2), n_pts);
    if s > 1; sx = sx(2:end); sy = sy(2:end); end
    px = [px, sx]; py = [py, sy]; %#ok<AGROW>
    ctrl_idx(end+1) = numel(px); %#ok<AGROW>
end
dist_km = [0, cumsum(sqrt(diff(px).^2 + diff(py).^2))] / 1e3;
ctrl_dist_km = dist_km(ctrl_idx);
end


function data = sample_tif(tif_path, px, py)
% Read TIF and bilinear-interpolate onto profile points.
[raw, R] = readgeoraster(tif_path);
raw      = double(raw);
nd       = georasterinfo(tif_path).MissingDataIndicator;
if ~isempty(nd); raw(raw == nd) = NaN; end
raw(~isfinite(raw)) = NaN;

% Build pixel-centre coordinate vectors (handles both ref types)
[nr, nc] = size(raw);
if isprop(R,'CellExtentInWorldX')
    dx = R.CellExtentInWorldX; dy = R.CellExtentInWorldY;
    xs = linspace(R.XWorldLimits(1)+dx/2, R.XWorldLimits(2)-dx/2, nc);
    ys = linspace(R.YWorldLimits(2)-dy/2, R.YWorldLimits(1)+dy/2, nr);
else
    xs = linspace(R.XWorldLimits(1), R.XWorldLimits(2), nc);
    ys = linspace(R.YWorldLimits(2), R.YWorldLimits(1), nr);
end

% interp2 expects Y increasing; raw is north-up (Y decreasing row-wise)
% ys is already in descending order �� flip
raw = flipud(raw);
ys  = fliplr(ys);

data = single(interp2(xs, ys(:), raw, px(:)', py(:)', 'linear', NaN));
end


function out = smooth_profile(arr, method, window)
% 1-D profile smoothing with NaN interpolation pre/post fill.
out = arr;
valid = isfinite(arr);
if sum(valid) < 4 || strcmp(method,'none'); return; end

x_all  = 1:numel(arr);
filled = double(interp1(x_all(valid), double(arr(valid)), x_all, 'linear', 'extrap'));
filled = filled(:)';   % ensure row vector

switch method
    case 'moving_avg'
        k        = ones(1, window) / window;
        smoothed = conv(filled, k, 'same');
        norm_k   = conv(ones(1, numel(filled)), k, 'same');
        smoothed = smoothed ./ norm_k;

    case 'savgol'
        w       = window + mod(window+1, 2);   % must be odd
        polyord = min(3, w - 1);
        smoothed = sgolayfilt(filled, polyord, w);

    case 'gaussian'
        sigma    = window / 4;
        smoothed = imgaussfilt(filled, sigma);   % row vector treated as 1-D

    otherwise
        return;
end

out(valid) = single(smoothed(valid));
end


function draw_band_line(ax, dist_km, v, err, col, alpha_fill, lw)
% Draw error band and velocity line independently.
ok_v   = isfinite(v);
ok_err = isfinite(v) & isfinite(err) & (err > 0);

hold(ax,'on');
if sum(ok_err) >= 2
    d  = dist_km; vv = v; ee = err;
    vv(~ok_err) = NaN; ee(~ok_err) = NaN;
    fill(ax, [d, fliplr(d)], [vv+ee, fliplr(vv-ee)], col, ...
        'FaceAlpha', alpha_fill, 'EdgeColor','none');
end
if sum(ok_v) >= 2
    plot(ax, dist_km(ok_v), v(ok_v), 'Color', col, 'LineWidth', lw);
end
hold(ax,'off');
end


function add_ctrl_lines(ax, ctrl_dist_km, ctrl_labels, top_y)
hold(ax,'on');
for k = 1:numel(ctrl_dist_km)
    xline(ax, ctrl_dist_km(k), '--', 'Color','#888', 'LineWidth',0.7, 'Alpha',0.5);
    if ~isempty(ctrl_labels) && ~isempty(top_y)
        text(ax, ctrl_dist_km(k), top_y, [' ' ctrl_labels{k}], ...
            'Color','#aaa','FontSize',9,'FontWeight','bold',...
            'VerticalAlignment','top','HorizontalAlignment','left');
    end
end
hold(ax,'off');
end


function col = season_color(month_str)
switch month_str
    case {'12','01','02'}; col = [0.91 0.30 0.24];   % Summer DJF  red
    case {'03','04','05'}; col = [0.90 0.50 0.13];   % Autumn MAM  orange
    case {'06','07','08'}; col = [0.20 0.60 0.86];   % Winter JJA  blue
    otherwise;             col = [0.18 0.80 0.44];   % Spring SON  green
end
end


function legend_patches(ax)
hold(ax,'on');
h = [
    patch(ax,NaN,NaN,[0.91 0.30 0.24],'DisplayName','Summer DJF')
    patch(ax,NaN,NaN,[0.90 0.50 0.13],'DisplayName','Autumn MAM')
    patch(ax,NaN,NaN,[0.20 0.60 0.86],'DisplayName','Winter JJA')
    patch(ax,NaN,NaN,[0.18 0.80 0.44],'DisplayName','Spring SON')
    plot(ax,NaN,NaN,'w','LineWidth',2,'DisplayName','Mean')
    patch(ax,NaN,NaN,[0.27 0.51 0.71],'FaceAlpha',0.4,'DisplayName','\pm1\sigma range')
];
legend(h,'FontSize',8,'Location','northwest','TextColor','#ccc');
hold(ax,'off');
end


function v = pct_finite(arr, pct)
arr = arr(isfinite(arr));
if isempty(arr); v = 0; return; end
v = prctile(arr, pct);
end


function cmap = rdylbu_r(n)
% Red-Yellow-Blue colormap reversed (warm = fast).
cmap = flipud(colormap_rdylbu(n));
end

function cmap = colormap_rdylbu(n)
% Approximate RdYlBu from matplotlib.
stops = [
    0.647  0.000  0.149
    0.843  0.188  0.153
    0.957  0.427  0.263
    0.992  0.682  0.380
    0.996  0.878  0.565
    1.000  1.000  0.749
    0.878  0.953  0.973
    0.671  0.851  0.914
    0.455  0.678  0.820
    0.271  0.459  0.706
    0.192  0.212  0.584
];
x  = linspace(0,1,size(stops,1));
xn = linspace(0,1,n);
cmap = interp1(x, stops, xn);
end


function save_fig(fig, out_path)
% Export figure to PNG with tight bounding box.
[d,~,~] = fileparts(out_path);
if ~isempty(d) && ~isfolder(d); mkdir(d); end
exportgraphics(fig, out_path, 'Resolution', 200, 'BackgroundColor','k');
close(fig);
end