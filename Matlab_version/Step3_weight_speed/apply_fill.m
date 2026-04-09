function result = apply_fill(input_tif, output_dir, varargin)
% APPLY_FILL  对单幅冰速 GeoTIFF 进行空洞填补，并输出对比图
%
% -------------------------------------------------------------------------
% 语法:
%   result = apply_fill(input_tif, output_dir)
%   result = apply_fill(input_tif, output_dir, Name, Value, ...)
%
% -------------------------------------------------------------------------
% 必选输入:
%   input_tif   - 输入 GeoTIFF 文件路径 (字符串)
%                 例: '/data/2017-01-Vx.tif'
%   output_dir  - 输出目录路径 (字符串)
%                 例: '/data/output/'
%
% -------------------------------------------------------------------------
% 可选参数 (Name-Value):
%   'Method'        - 填补方法
%                     'inpaint'（默认）: inpaintCoherent，保边，适合流场
%                     'region'         : regionfill，拉普拉斯，速度快
%                     'nanmean'        : NaN均值滤波迭代，适合椒盐小孔洞
%
%   'SmoothFactor'  - inpaintCoherent 平滑强度 [0~1]，默认 0.7
%                     越大填补区域越平滑
%
%   'NanMeanWin'    - nanmean 方法的滤波窗口大小（奇数），默认 5
%
%   'NanMeanIter'   - nanmean 方法的迭代次数，默认 3
%
%   'EPSG'          - 输出 GeoTIFF 投影代码，默认 3031（南极极地立体）
%
%   'Colormap'      - 图像显示色表，默认 'jet'
%                     可选: 'turbo' / 'hot' / 'parula' / 'redblue' 等
%
%   'ClimAuto'      - 自动色阶 true（默认）/ false（使用ClimRange）
%
%   'ClimRange'     - 手动色阶范围 [min max]，ClimAuto=false 时生效
%                     例: [0 500]（单位与输入数据一致，如 m/yr）
%
%   'SaveFig'       - 是否保存对比图 true（默认）/ false
%
%   'FigFormat'     - 图像格式 'png'（默认）/ 'pdf' / 'tif'
%
%   'FigDPI'        - 图像分辨率 DPI，默认 200
%
%   'Title'         - 图像标题前缀（默认使用文件名）
%
% -------------------------------------------------------------------------
% 输出:
%   result  - 结构体，包含以下字段:
%     .img_raw       - 原始图像 (single)
%     .img_filled    - 填补后图像 (single)
%     .nan_mask      - 空洞掩膜 (logical)
%     .n_nan_in      - 填补前空洞像素数
%     .n_nan_out     - 填补后残余空洞数
%     .fill_ratio    - 填补率 (%)
%     .out_tif       - 输出 tif 文件路径
%     .out_fig       - 输出图像文件路径（SaveFig=true 时）
%     .R             - 空间参考对象
%
% -------------------------------------------------------------------------
% 使用示例:
%   % 最简用法
%   result = apply_fill('/data/2017-01-Vx.tif', '/data/output/');
%
%   % 指定方法和参数
%   result = apply_fill('/data/2017-01-Vx.tif', '/data/output/', ...
%       'Method',       'inpaint',  ...
%       'SmoothFactor', 0.8,        ...
%       'Colormap',     'turbo',    ...
%       'ClimAuto',     false,      ...
%       'ClimRange',    [0 800],    ...
%       'FigDPI',       300);
%
%   % nanmean 方法（适合散点小孔洞）
%   result = apply_fill('/data/2017-01-V.tif', '/data/output/', ...
%       'Method',      'nanmean', ...
%       'NanMeanWin',  5,         ...
%       'NanMeanIter', 5);
% -------------------------------------------------------------------------

    %% ---- 解析参数 ----
    p = inputParser;
    addRequired(p,  'input_tif',   @ischar);
    addRequired(p,  'output_dir',  @ischar);
    addParameter(p, 'Method',       'inpaint',  @ischar);
    addParameter(p, 'SmoothFactor', 0.7,        @isnumeric);
    addParameter(p, 'NanMeanWin',   5,          @isnumeric);
    addParameter(p, 'NanMeanIter',  3,          @isnumeric);
    addParameter(p, 'EPSG',         3031,       @isnumeric);
    addParameter(p, 'Colormap',     'jet',      @ischar);
    addParameter(p, 'ClimAuto',     true,       @islogical);
    addParameter(p, 'ClimRange',    [],         @isnumeric);
    addParameter(p, 'SaveFig',      true,       @islogical);
    addParameter(p, 'FigFormat',    'png',      @ischar);
    addParameter(p, 'FigDPI',       200,        @isnumeric);
    addParameter(p, 'Title',        '',         @ischar);
    parse(p, input_tif, output_dir, varargin{:});
    opt = p.Results;

    mkdir(output_dir);

    %% ---- 读取数据 ----
    fprintf('\n========== apply_fill ==========\n');
    fprintf('输入文件: %s\n', input_tif);

    [img_raw, R] = readgeoraster(input_tif);
    img_raw = single(img_raw);

    % 0、Inf、-Inf、NaN 统一视为无效
    img_raw(img_raw == 0 | ~isfinite(img_raw)) = NaN;

    [rows, cols] = size(img_raw);
    nan_mask     = isnan(img_raw);
    n_nan_in     = sum(nan_mask(:));
    n_total      = rows * cols;

    [~, fname, ~] = fileparts(input_tif);
    title_str = opt.Title;
    if isempty(title_str)
        title_str = strrep(fname, '_', '\_');
    end

    fprintf('图像尺寸: %d × %d\n', rows, cols);
    fprintf('空洞像素: %d / %d (%.2f%%)\n', n_nan_in, n_total, 100*n_nan_in/n_total);
    fprintf('填补方法: %s\n', opt.Method);

    %% ---- 执行填补 ----
    t_fill = tic;
    img_filled = do_fill(img_raw, nan_mask, opt);
    fprintf('填补耗时: %.2f 秒\n', toc(t_fill));

    n_nan_out  = sum(isnan(img_filled(:)));
    fill_ratio = 100 * (n_nan_in - n_nan_out) / max(n_nan_in, 1);
    fprintf('填补后残余空洞: %d 像素\n', n_nan_out);
    fprintf('填补率: %.1f%%\n', fill_ratio);

    %% ---- 保存填补后的 GeoTIFF ----
    out_tif = fullfile(output_dir, [fname '_filled_' lower(opt.Method) '.tif']);
    geotiffwrite(out_tif, single(img_filled), R, 'CoordRefSysCode', opt.EPSG);
    fprintf('已保存 TIF → %s\n', out_tif);

    %% ---- 绘制对比图 ----
    out_fig = '';
    if opt.SaveFig || nargout > 0
        out_fig = plot_comparison(img_raw, img_filled, nan_mask, ...
            title_str, opt, output_dir, fname);
    end

    %% ---- 返回结构体 ----
    result.img_raw    = img_raw;
    result.img_filled = single(img_filled);
    result.nan_mask   = nan_mask;
    result.n_nan_in   = n_nan_in;
    result.n_nan_out  = n_nan_out;
    result.fill_ratio = fill_ratio;
    result.out_tif    = out_tif;
    result.out_fig    = out_fig;
    result.R          = R;

    fprintf('================================\n\n');
end


% =========================================================================
%  内部：执行填补
% =========================================================================
function out = do_fill(img, nan_mask, opt)
    switch lower(opt.Method)

        case 'inpaint'
            % inpaintCoherent 要求所有非掩膜像素必须是有限值
            % 将残余 Inf/-Inf 也纳入掩膜一并填补
            inf_mask      = ~isfinite(img);           % 找出 Inf / -Inf
            full_mask     = nan_mask | inf_mask;      % 合并掩膜
            img_clean     = img;
            img_clean(full_mask) = 0;                 % 临时填 0（值不影响结果）
            out = inpaintCoherent(img_clean, full_mask, ...
                'SmoothingFactor', opt.SmoothFactor);
            % 恢复原始有效像素（不改动原始非空洞区）
            out(~full_mask) = img(~full_mask);
            % 更新 nan_mask 统计（Inf 区域也视为被填补）
            if any(inf_mask(:))
                fprintf('  [inpaint] 额外处理 Inf 像素: %d 个\n', sum(inf_mask(:)));
            end

        case 'region'
            out = regionfill(img, nan_mask);

        case 'nanmean'
            out = img;
            for iter = 1:opt.NanMeanIter
                n_before = sum(isnan(out(:)));
                out      = nanmean_fill(out, opt.NanMeanWin);
                n_after  = sum(isnan(out(:)));
                fprintf('  nanmean 第%d次: NaN %d → %d\n', iter, n_before, n_after);
                if n_after == 0, break; end
            end

        otherwise
            error('未知方法: %s，可选: inpaint / region / nanmean', opt.Method);
    end
    out = single(out);
end


% =========================================================================
%  内部：NaN均值滤波（单次）
% =========================================================================
function out = nanmean_fill(img, win)
    half   = floor(win / 2);
    padded = padarray(double(img), [half half], NaN, 'both');
    [rows, cols] = size(img);

    sum_map   = zeros(rows, cols);
    count_map = zeros(rows, cols);

    for di = -half:half
        for dj = -half:half
            slice = padded(half+1+di : half+rows+di, ...
                           half+1+dj : half+cols+dj);
            valid = ~isnan(slice);
            sum_map(valid)   = sum_map(valid)   + slice(valid);
            count_map(valid) = count_map(valid) + 1;
        end
    end

    out = single(sum_map ./ count_map);
    out(count_map == 0) = NaN;
end


% =========================================================================
%  内部：绘制对比图（4子图）
% =========================================================================
function out_fig = plot_comparison(img_raw, img_filled, nan_mask, ...
                                    title_str, opt, output_dir, fname)

    % 色阶范围
    valid_vals = img_raw(~nan_mask);
    if opt.ClimAuto || isempty(opt.ClimRange)
        lo = double(prctile(valid_vals, 2));
        hi = double(prctile(valid_vals, 98));
    else
        lo = opt.ClimRange(1);
        hi = opt.ClimRange(2);
    end
    clim_range = [lo, hi];

    % 差值图（填补区域的填入值）
    diff_img        = nan(size(img_raw), 'single');
    diff_img(nan_mask) = img_filled(nan_mask);

    fig = figure('Visible', 'off', 'Color', 'w', ...
                 'Units', 'pixels', 'Position', [100 100 1400 420]);

    %% 子图1: 原始图
    ax1 = subplot(1, 4, 1);
    imagesc(ax1, img_raw, clim_range);
    colormap(ax1, opt.Colormap);
    cb1 = colorbar(ax1);
    cb1.Label.String = 'm yr^{-1}';
    axis(ax1, 'equal', 'tight', 'off');
    title(ax1, sprintf('原始图\n(%s)', title_str), ...
        'Interpreter', 'tex', 'FontSize', 10, 'FontWeight', 'bold');
    set_axes_style(ax1);

    %% 子图2: 空洞掩膜
    ax2 = subplot(1, 4, 2);
    mask_rgb = ones(size(img_raw, 1), size(img_raw, 2), 3, 'single');  % 白色背景
    mask_rgb(:,:,1) = mask_rgb(:,:,1) .* single(~nan_mask);            % 有效区域灰色
    mask_rgb(:,:,2) = mask_rgb(:,:,2) .* single(~nan_mask);
    mask_rgb(:,:,3) = mask_rgb(:,:,3) .* single(~nan_mask);
    image(ax2, mask_rgb);
    hold(ax2, 'on');
    % 空洞用红色高亮
    hole_layer = zeros(size(img_raw,1), size(img_raw,2), 3, 'single');
    hole_layer(:,:,1) = single(nan_mask);
    image(ax2, hole_layer, 'AlphaData', single(nan_mask) * 0.85);
    hold(ax2, 'off');
    axis(ax2, 'equal', 'tight', 'off');
    title(ax2, sprintf('空洞分布（红色）\n共 %d 像素 (%.1f%%)', ...
        sum(nan_mask(:)), 100*mean(nan_mask(:))), ...
        'FontSize', 10, 'FontWeight', 'bold');
    set_axes_style(ax2);

    %% 子图3: 填补后图
    ax3 = subplot(1, 4, 3);
    imagesc(ax3, img_filled, clim_range);
    colormap(ax3, opt.Colormap);
    cb3 = colorbar(ax3);
    cb3.Label.String = 'm yr^{-1}';
    axis(ax3, 'equal', 'tight', 'off');
    n_nan_out = sum(isnan(img_filled(:)));
    title(ax3, sprintf('填补后（%s）\n残余空洞: %d 像素', opt.Method, n_nan_out), ...
        'FontSize', 10, 'FontWeight', 'bold');
    set_axes_style(ax3);

    %% 子图4: 填入值分布（仅空洞区域）
    ax4 = subplot(1, 4, 4);
    filled_vals = double(diff_img(~isnan(diff_img)));
    if ~isempty(filled_vals)
        histogram(ax4, filled_vals, 60, ...
            'FaceColor', [0.2 0.5 0.9], 'EdgeColor', 'none', 'FaceAlpha', 0.8);
        xlabel(ax4, '填入速度值 (m yr^{-1})', 'FontSize', 9);
        ylabel(ax4, '像素数', 'FontSize', 9);
        title(ax4, sprintf('填入值分布\n均值: %.1f  std: %.1f', ...
            mean(filled_vals), std(filled_vals)), ...
            'FontSize', 10, 'FontWeight', 'bold');
        grid(ax4, 'on');
        box(ax4, 'off');
    else
        text(ax4, 0.5, 0.5, '无填补区域', 'HorizontalAlignment', 'center', ...
            'Units', 'normalized', 'FontSize', 12);
        axis(ax4, 'off');
    end

    % 总标题
    sgtitle(sprintf('空洞填补对比  |  方法: %s  |  %s', opt.Method, fname), ...
        'FontSize', 12, 'FontWeight', 'bold', 'Interpreter', 'none');

    %% 保存图像
    out_fig = '';
    if opt.SaveFig
        out_fig = fullfile(output_dir, sprintf('%s_fill_compare_%s.%s', fname, lower(opt.Method), opt.FigFormat));
        exportgraphics(fig, out_fig, 'Resolution', opt.FigDPI);
        fprintf('已保存对比图 → %s\n', out_fig);
    end

    % 同时显示到屏幕
    set(fig, 'Visible', 'on');
end


% =========================================================================
%  内部：统一坐标轴风格
% =========================================================================
function set_axes_style(ax)
    ax.FontSize  = 8;
    ax.LineWidth = 0.8;
    ax.Box       = 'off';
end
