function plot_batch_results2(data_dict, cfg, output_path, title_str)
    % 优化版绘图脚本
    % 改进点：
    % 1. 使用 tiledlayout 替代 subplot，布局更紧凑
    % 2. 区分数据类型：位移场使用 turbo/parula，掩膜使用黑白灰度
    % 3. 统一处理 NaN 值的透明度
    % 4. 优化字体和背景颜色
    
    % 设置高清画布，背景设为白色
    fig = figure('Position', [100, 100, 1000, 800], 'Visible', 'off'); 
    
    % 使用 tiledlayout 进行紧凑布局 (需要 R2019b+)
    % 如果是老版本 MATLAB，请保留 subplot 但需手动调整 position
    t = tiledlayout(cfg.row, cfg.col, 'TileSpacing', 'compact', 'Padding', 'compact');
    
    title(t, title_str, 'FontSize', 22, 'FontWeight', 'bold', 'Interpreter', 'none');

    %% 定义绘图辅助函数 (简化重复代码)
    function plot_deformation(data, v_range, tit_str, ax_idx)
        nexttile(ax_idx);
        % 使用 imagesc 绘图，设置 AlphaData 使 NaN 透明
        im = imagesc(data); 
        set(im, 'AlphaData', ~isnan(data)); 
        
        axis image; axis off; % 保持比例并隐藏坐标轴
        colormap(gca, turbo); % 使用对比度更好的 turbo 或 parula
        if ~isempty(v_range)
            clim(v_range);
        % else
        %     % 自动范围 (比如 Detrended)
        %     max_val = max(abs(data(:)), [], 'omitnan');
        %     if max_val == 0, max_val = 1; end
        %     clim([-max_val, max_val]);
        end
        
        % 美化标题
       title(tit_str, 'FontSize', 18, 'Color', [0.3 0.3 0.3]);
        
        % 美化 Colorbar
        cb = colorbar;
        cb.TickLabelInterpreter = 'none';
        cb.FontSize = 14;
        % 可选：设置背景色以突显 NaN (默认为白色)
        set(gca, 'Color', [0.9 0.9 0.9]); 
    end

    function plot_mask(data, tit_str, ax_idx)
        nexttile(ax_idx);
        im = imagesc(data);
        set(im, 'AlphaData', ~isnan(data)); 
        
        axis image; axis off;
        % 掩膜使用黑白配色：0(背景)=白色, 1(掩膜)=黑色/深色
        colormap(gca, flipud(gray)); 
        clim([0, 1]); 
        
        title(tit_str, 'FontSize', 18, 'Color', [0.3 0.3 0.3]);
        % Mask 通常不需要 colorbar，或者只需要简化的
        cb = colorbar;
        cb.TickLabelInterpreter = 'none';
        cb.FontSize = 14;
        set(gca, 'Color', [0.9 0.9 0.9]);
    end

    %% --- 1. 原始位移 (Raw) ---
    plot_deformation(data_dict.raw, [cfg.vmin, cfg.vmax], '1. Raw Deformation', 1);

    %% --- 6. 拟合趋势面 (Ramp) ---
    % 动态计算 Ramp 范围
    % ramp_std = max(data_dict.ramp(:), 'omitnan');
    plot_deformation(data_dict.ramp, [], ['2. Estimated Ramp (', cfg.detrend_method, ')'], 2);

    %% --- 2. 去趋势结果 (Detrended) ---
    % Detrend 结果通常以 0 为中心，建议使用发散色条范围
    plot_deformation(data_dict.detrended, [cfg.vmin, cfg.vmax], '3. Detrended (Residual)', 3);

    %% --- 3. N-Sigma滤波 ---
    plot_deformation(data_dict.filtered_sigma, [cfg.vmin, cfg.vmax], ...
        ['4. N-Sigma Filter (N=', num2str(cfg.sigma_times), ')'], 4);

    %% --- 4. NMT after deramp ---
    plot_deformation(data_dict.filtered_nmt, [cfg.vmin, cfg.vmax], ...
        ['5. NMT after deramp (T=', num2str(cfg.nmt_threshold), ')'], 5);

    %% --- 5. NMT using Raw ---
    plot_deformation(data_dict.filtered_nmt_raw, [cfg.vmin, cfg.vmax], ...
        ['6. NMT using raw (T=', num2str(cfg.nmt_threshold), ')'], 6);

    %% --- 7-9. Masks (使用专用 Mask 绘图函数) ---
    plot_mask(data_dict.outlier_mask_sigma, '7. Mask: N-Sigma', 7);

    plot_mask(data_dict.outlier_mask_nmt, '8. Mask: NMT-Deramp', 8);

    plot_mask(data_dict.outlier_mask_nmt_raw, '9. Mask: NMT-Raw', 9);

    %% 保存图片
    % 使用 exportgraphics (R2020a+) 能够获得更好的裁剪效果和分辨率
    % 如果版本较低，回退到 saveas
    try
        exportgraphics(fig, output_path, 'Resolution', 300);
    catch
        saveas(fig, output_path);
    end
    
    close(fig);
end