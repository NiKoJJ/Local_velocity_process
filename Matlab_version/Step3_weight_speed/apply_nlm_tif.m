function result = apply_nlm_tif(input_tif, output_dir, varargin)
% APPLY_NLM_TIF  对单幅 GeoTIFF 做 NLM 滤波并输出对比图
%
% -------------------------------------------------------------------------
% 语法:
%   result = apply_nlm_tif(input_tif, output_dir)
%   result = apply_nlm_tif(input_tif, output_dir, Name, Value, ...)
%
% -------------------------------------------------------------------------
% 必选输入:
%   input_tif   - 输入 GeoTIFF 路径 (字符串)
%   output_dir  - 输出目录路径 (字符串)
%
% -------------------------------------------------------------------------
% 可选参数 (Name-Value):
%   'H'           - NLM 滤波强度 DegreeOfSmoothing，默认 10
%                   弱噪声 5~10 / 中等 10~20 / 强噪声 20~40
%   'PatchSize'   - Patch 大小 ComparisonWindowSize（奇数），默认 7
%   'SearchSize'  - 搜索窗口 SearchWindowSize（奇数），默认 21
%   'NoDataVal'   - 输入 NoData 值，默认 0
%   'EPSG'        - 输出投影代码，默认 3031
%   'Colormap'    - 显示色表，默认 'turbo'
%   'ClimAuto'    - 自动色阶（2%~98%分位），默认 true
%   'ClimRange'   - 手动色阶 [min max]，ClimAuto=false 时生效
%   'SaveFig'     - 是否保存对比图，默认 true
%   'FigFormat'   - 图像格式 'png'（默认）/ 'pdf' / 'tif'
%   'FigDPI'      - 图像分辨率，默认 200
%
% -------------------------------------------------------------------------
% 输出 result 结构体:
%   .img_raw      - 原始图像 (single)
%   .img_nlm      - NLM 滤波后图像 (single)
%   .out_tif      - 输出 tif 路径
%   .out_fig      - 输出图像路径
%   .R            - 空间参考对象
%
% -------------------------------------------------------------------------
% 示例:
%   result = apply_nlm_tif('2016-V.tif', './output/');
%
  % result = apply_nlm_tif('2016-Vx.tif', './output/', ...
  %     'H',          15,       ...
  %     'PatchSize',  7,        ...
  %     'SearchSize', 21,       ...
  %     'Colormap',   'turbo',  ...
  %     'ClimRange',  [0 800],  ...
  %     'ClimAuto',   false,    ...
  %     'FigDPI',     300);
% -------------------------------------------------------------------------

    %% ---- 解析参数 ----
    p = inputParser;
    addRequired(p,  'input_tif',   @ischar);
    addRequired(p,  'output_dir',  @ischar);
    addParameter(p, 'H',           10,      @isnumeric);
    addParameter(p, 'PatchSize',   7,       @isnumeric);
    addParameter(p, 'SearchSize',  21,      @isnumeric);
    addParameter(p, 'NoDataVal',   0,       @isnumeric);
    addParameter(p, 'EPSG',        3031,    @isnumeric);
    addParameter(p, 'Colormap',    'turbo', @ischar);
    addParameter(p, 'ClimAuto',    true,    @islogical);
    addParameter(p, 'ClimRange',   [],      @isnumeric);
    addParameter(p, 'SaveFig',     true,    @islogical);
    addParameter(p, 'FigFormat',   'png',   @ischar);
    addParameter(p, 'FigDPI',      200,     @isnumeric);
    parse(p, input_tif, output_dir, varargin{:});
    opt = p.Results;

    mkdir(output_dir);

    %% ---- 读取 ----
    fprintf('\n========== apply_nlm_tif ==========\n');
    fprintf('输入: %s\n', input_tif);

    [img_raw, R] = readgeoraster(input_tif);
    img_raw = single(img_raw);
    img_raw(img_raw == opt.NoDataVal | ~isfinite(img_raw)) = NaN;

    [rows, cols] = size(img_raw);
    nan_mask = isnan(img_raw);
    fprintf('图像尺寸  : %d × %d\n', rows, cols);
    fprintf('NaN 像素  : %d (%.1f%%)\n', sum(nan_mask(:)), 100*mean(nan_mask(:)));
    fprintf('NLM 参数  : h=%g, patch=%d×%d, 搜索=%d×%d\n', ...
        opt.H, opt.PatchSize, opt.PatchSize, opt.SearchSize, opt.SearchSize);

    %% ---- NLM 滤波 ----
    t0 = tic;
    img_nlm = do_nlm(img_raw, opt);
    fprintf('NLM 耗时  : %.1f 秒\n', toc(t0));

    %% ---- 保存 tif ----
    [~, fname] = fileparts(input_tif);
    out_tif = fullfile(output_dir, sprintf('%s_NLM_h%g.tif', fname, opt.H));
    geotiffwrite(out_tif, single(img_nlm), R, 'CoordRefSysCode', opt.EPSG);
    fprintf('已保存 TIF: %s\n', out_tif);

    %% ---- 绘图 ----
    out_fig = plot_compare(img_raw, img_nlm, fname, opt, output_dir);

    %% ---- 返回 ----
    result.img_raw  = img_raw;
    result.img_nlm  = img_nlm;
    result.out_tif  = out_tif;
    result.out_fig  = out_fig;
    result.R        = R;
    fprintf('====================================\n\n');
end


% =========================================================================
%  内部：NLM 滤波（保留 NaN 掩膜）
% =========================================================================
function out = do_nlm(img, opt)
    nan_mask  = isnan(img);
    img_fill  = img;
    img_fill(nan_mask) = 0;   % NaN 临时填 0，不影响有效区域滤波

    out = imnlmfilt(img_fill, ...
        'DegreeOfSmoothing',    opt.H,          ...
        'ComparisonWindowSize', opt.PatchSize,   ...
        'SearchWindowSize',     opt.SearchSize);

    out(nan_mask) = NaN;   % 恢复 NaN 掩膜
    out = single(out);
end


% =========================================================================
%  内部：绘制对比图（3 子图）
% =========================================================================
function out_fig = plot_compare(img_raw, img_nlm, fname, opt, output_dir)

    % 色阶
    valid = img_raw(~isnan(img_raw));
    if opt.ClimAuto || isempty(opt.ClimRange)
        lo = double(prctile(valid, 2));
        hi = double(prctile(valid, 98));
    else
        lo = opt.ClimRange(1);
        hi = opt.ClimRange(2);
    end
    clim_rng = [lo, hi];

    % 差值图
    diff_img = img_nlm - img_raw;

    fig = figure('Visible', 'off', 'Color', 'w', ...
                 'Units', 'pixels', 'Position', [100 100 1400 440]);

    %% 子图1: 原始
    ax1 = subplot(1, 3, 1);
    imagesc(ax1, img_raw, clim_rng);
    colormap(ax1, opt.Colormap);
    cb = colorbar(ax1); cb.Label.String = 'm yr^{-1}';
    axis(ax1, 'equal', 'tight', 'off');
    title(ax1, '原始图', 'FontSize', 11, 'FontWeight', 'bold');
    add_stats(ax1, img_raw);

    %% 子图2: NLM 滤波后
    ax2 = subplot(1, 3, 2);
    imagesc(ax2, img_nlm, clim_rng);
    colormap(ax2, opt.Colormap);
    cb = colorbar(ax2); cb.Label.String = 'm yr^{-1}';
    axis(ax2, 'equal', 'tight', 'off');
    title(ax2, sprintf('NLM 滤波后\nh=%g, patch=%d, search=%d', ...
        opt.H, opt.PatchSize, opt.SearchSize), ...
        'FontSize', 11, 'FontWeight', 'bold');
    add_stats(ax2, img_nlm);

    %% 子图3: 差值图（滤波后 - 原始）
    ax3 = subplot(1, 3, 3);
    diff_vals = diff_img(~isnan(diff_img));
    d_abs     = max(abs(prctile(diff_vals, [2 98])));
    imagesc(ax3, diff_img, [-d_abs, d_abs]);
    % colormap(ax3, 'rdbu_r');   % 蓝负红正
    % try
    %     colormap(ax3, customcolormap_preset('rdbu'));
    % catch
    %     colormap(ax3, 'cool');
    % end
    n = 128;
    cmap_div = [linspace(0,1,n)', linspace(0,1,n)', ones(n,1); ...
                ones(n,1), linspace(1,0,n)', linspace(1,0,n)'];
    colormap(ax3, cmap_div);

    cb = colorbar(ax3); cb.Label.String = '\Delta (m yr^{-1})';
    axis(ax3, 'equal', 'tight', 'off');
    title(ax3, sprintf('差值图 (NLM - 原始)\n均值: %.2f  std: %.2f', ...
        mean(diff_vals, 'omitnan'), std(diff_vals, 'omitnan')), ...
        'FontSize', 11, 'FontWeight', 'bold');

    sgtitle(sprintf('NLM 滤波对比  |  %s', strrep(fname, '_', '\_')), ...
        'FontSize', 13, 'FontWeight', 'bold');

    %% 保存
    out_fig = '';
    if opt.SaveFig
        out_fig = fullfile(output_dir, ...
            sprintf('%s_NLM_h%g_compare.%s', fname, opt.H, opt.FigFormat));
        exportgraphics(fig, out_fig, 'Resolution', opt.FigDPI);
        fprintf('已保存对比图: %s\n', out_fig);
    end
    set(fig, 'Visible', 'on');
end


% =========================================================================
%  内部：在图像左下角显示统计信息
% =========================================================================
function add_stats(ax, img)
    vals = img(~isnan(img));
    if isempty(vals), return; end
    str = sprintf('min %.1f\nmax %.1f\nmean %.1f\nstd %.1f', ...
        min(vals), max(vals), mean(vals), std(vals));
    text(ax, 0.02, 0.02, str, ...
        'Units', 'normalized', ...
        'FontSize', 7, ...
        'Color', 'w', ...
        'BackgroundColor', [0 0 0 0.4], ...
        'VerticalAlignment', 'bottom', ...
        'Interpreter', 'none');
end
