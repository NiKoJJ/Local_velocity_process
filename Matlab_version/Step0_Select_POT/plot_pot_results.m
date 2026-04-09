% =========================================================
% Plot POT Results: Range / Azimuth / Histograms
% 2x2 layout per pair: range | azimuth | range_hist | azimuth_hist
% =========================================================

clear; clc;

% ---- USER SETTINGS -------------------------------------------------------
base_dir   = '/data2/Phd_Work1/ICE_Velocity_Process/GAMMA_POT_Input/Mertz_100m';
out_dir    = fullfile(base_dir, 'POT_Figures');
range_dir  = fullfile(base_dir, 'range');
azi_dir    = fullfile(base_dir, 'azimuth');

% Color limits derived from data (2nd–98th percentile, robust to outliers)
clim_pct   = [2 98];       % percentile range, adjust if needed (e.g. [5 95])
hist_bins  = 200;
font_size  = 24;
fig_dpi    = 200;           % dpi for export (high res but fast)
% -------------------------------------------------------------------------

if ~exist(out_dir, 'dir'); mkdir(out_dir); end

range_files = dir(fullfile(range_dir, '*.range.tif'));
n = numel(range_files);
fprintf('Found %d pairs. Starting ...\n', n);

set(0, 'DefaultFigureVisible', 'off');   % no display

for i = 1 : n
    rfile = fullfile(range_dir, range_files(i).name);

    % Derive matching azimuth filename
    pair_id = strrep(range_files(i).name, '.range.tif', '');
    afile   = fullfile(azi_dir, [pair_id, '.azi.tif']);

    if ~exist(afile, 'file')
        fprintf('[%d/%d] Skipping %s  (no azi file)\n', i, n, pair_id);
        continue
    end

    % --- Read data ---
    R = double(imread(rfile));
    A = double(imread(afile));

    % Common NaN mask (0 often used as nodata in SAR offset products)
    R(R == 0) = NaN;
    A(A == 0) = NaN;

    % --- Figure ---
    fig = figure('Units','pixels','Position',[0 0 1600 1200]);

    % --- Compute per-image percentile color limits -----------------------
    rv = R(~isnan(R));
    av = A(~isnan(A));
    if ~isempty(rv)
        clim_range = prctile(rv, clim_pct);
        if diff(clim_range) == 0; clim_range = clim_range + [-1 1]; end
    else
        clim_range = [-1 1];
    end
    if ~isempty(av)
        clim_azi = prctile(av, clim_pct);
        if diff(clim_azi) == 0; clim_azi = clim_azi + [-1 1]; end
    else
        clim_azi = [-1 1];
    end

    % ---- Subplot 1: Range ------------------------------------------------
    ax1 = subplot(2,2,1);
    imagesc(R, 'AlphaData', ~isnan(R));
    set(ax1, 'Color', [0.5 0.5 0.5]);   % grey background for NaN
    caxis(clim_range);
    colormap(ax1, jet);
    cb = colorbar; cb.FontSize = font_size;
    ylabel(cb, 'Range Offset (m)', 'FontSize', font_size);
    axis image off;
    title(['Range: ' strrep(pair_id,'-',' \rightarrow ')], ...
          'FontSize', font_size, 'Interpreter','tex');
    set(ax1, 'FontSize', font_size);

    % ---- Subplot 2: Azimuth ----------------------------------------------
    ax2 = subplot(2,2,2);
    imagesc(A, 'AlphaData', ~isnan(A));
    set(ax2, 'Color', [0.5 0.5 0.5]);
    caxis(clim_azi);
    colormap(ax2, jet);
    cb2 = colorbar; cb2.FontSize = font_size;
    ylabel(cb2, 'Azimuth Offset (m)', 'FontSize', font_size);
    axis image off;
    title(['Azimuth: ' strrep(pair_id,'-',' \rightarrow ')], ...
          'FontSize', font_size, 'Interpreter','tex');
    set(ax2, 'FontSize', font_size);

    % ---- Subplot 3: Range Histogram ---------------------------------------
    ax3 = subplot(2,2,3);
    if ~isempty(rv)
        histogram(rv, hist_bins, 'FaceColor',[0.2 0.5 0.8], ...
                  'EdgeColor','none', 'Normalization','count');
    end
    xlabel('Range Offset (m)', 'FontSize', font_size);
    ylabel('Count',           'FontSize', font_size);
    title('Range Histogram',  'FontSize', font_size);
    set(ax3, 'FontSize', font_size, 'Box','on');
    grid on;

    % ---- Subplot 4: Azimuth Histogram ------------------------------------
    ax4 = subplot(2,2,4);
    if ~isempty(av)
        histogram(av, hist_bins, 'FaceColor',[0.8 0.3 0.2], ...
                  'EdgeColor','none', 'Normalization','count');
    end
    xlabel('Azimuth Offset (m)', 'FontSize', font_size);
    ylabel('Count',              'FontSize', font_size);
    title('Azimuth Histogram',   'FontSize', font_size);
    set(ax4, 'FontSize', font_size, 'Box','on');
    grid on;

    % ---- Save ------------------------------------------------------------
    out_file = fullfile(out_dir, [pair_id, '.png']);
    exportgraphics(fig, out_file, 'Resolution', fig_dpi);
    close(fig);

    if mod(i,10) == 0 || i == n
        fprintf('[%d/%d] Done: %s\n', i, n, pair_id);
    end
end

fprintf('\nAll figures saved to: %s\n', out_dir);
