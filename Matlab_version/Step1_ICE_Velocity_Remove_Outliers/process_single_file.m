function process_single_file(tiff_data, output_dir, cfg)
    % 提取文件名信息
    [~, name, ~] = fileparts(tiff_data);
   
    %% --- 1. 读取数据 ---
    [deformation, R] = readgeoraster(tiff_data);
    valid_mask = (deformation < cfg.data_max) & (deformation > cfg.data_min) & ~isnan(deformation);
    if sum(valid_mask(:)) < 100, error('有效像元太少'); end

    
    %% --- 2. 空间去趋势 ---
    % deformation(~valid_mask) = NaN;
    [~, detrended, params, stats] = fit_spatial_trend_optimized(deformation, cfg.detrend_method, cfg.use_weights);
    model_ramp = deformation - detrended;

    % 统计分析
    deformation_valid = deformation(valid_mask);
    detrended_valid = detrended(valid_mask);
    ramp_valid = model_ramp(valid_mask);
    
    rms_before = sqrt(mean(deformation_valid.^2));
    rms_after = sqrt(mean(detrended_valid.^2));
    improvement = (1 - rms_after/rms_before) * 100;
    ramp_ratio = std(ramp_valid) / std(deformation_valid) * 100;
    
    fprintf('=========== step1: deramp =========== \n')
    fprintf('  方法: %s\n', cfg.detrend_method);
    fprintf('  拟合公式: %.4f*X + %.4f*Y + %.4f\n',params(1),params(2),params(3));
    fprintf('  R²: %.4f\n', stats.r2);
    fprintf('  RMS改善: %.4f → %.4f (%.1f%%)\n', rms_before, rms_after, improvement);
    fprintf('  趋势面占比: %.1f%% of total signal\n', ramp_ratio);
    fprintf(' \n');

    % 保存去趋势结果
    detrended(~valid_mask) = NaN;
    geotiffwrite(fullfile(output_dir, [name, '-detrend.tif']), detrended, R, 'CoordRefSysCode', cfg.epsg_code);
    geotiffwrite(fullfile(output_dir, [name, '-ramp.tif']), model_ramp, R, 'CoordRefSysCode', cfg.epsg_code);

    
    %% --- 3. N-sigma异常值剔除 ---
    [filtered_sigma, outlier_mask_sigma, norm_residual_sigma, stats_sigma] = apply_3sigma_optimized(detrended, cfg.sigma_times, cfg.min_segment_sigma);
    geotiffwrite(fullfile(output_dir, [name, '-sigma_filtered.tif']), filtered_sigma, R, 'CoordRefSysCode', cfg.epsg_code);
    % geotiffwrite(fullfile(output_dir, [name, '-sigma_outlier_mask.tif']), outlier_mask_sigma, R, 'CoordRefSysCode', cfg.epsg_code);

    fprintf('=========== Remove Outliers: N-Sigma after deramp =========== \n')
    fprintf('  sigma倍数: %.1f\n', stats_sigma.sigma_times);
    fprintf('  有效像素: %d\n', stats_sigma.n_valid);
    fprintf('  异常值: %d (%.2f%%)\n', stats_sigma.n_outliers, stats_sigma.outlier_pct);
    fprintf('  小区域: %d\n', stats_sigma.n_small);
    fprintf('  均值: %.4f\n', stats_sigma.mu);
    fprintf('  标准差: %.4f\n', stats_sigma.sigma);
    fprintf('  范围: [%.4f, %.4f]\n', stats_sigma.lower, stats_sigma.upper);
    % fprintf('  归一化残差: %.4f\n', norm_residual_sigma)
    fprintf(' \n');

    
    %% --- 4. NMT,normalized median test方法 (deramp) ---
    [filtered_nmt, outlier_mask_nmt, norm_residual_nmt, stats_nmt] = apply_nmt_optimized(detrended, cfg.nmt_window, cfg.nmt_threshold, cfg.nmt_epsilon, cfg.min_segment_nmt);
    
    filtered_nmt(~valid_mask) = NaN;
    geotiffwrite(fullfile(output_dir, [name, '-nmt_filtered.tif']), filtered_nmt, R, 'CoordRefSysCode', cfg.epsg_code);
    % geotiffwrite(fullfile(output_dir, [name, '-nmt_outlier_mask.tif']), outlier_mask_nmt, R, 'CoordRefSysCode', cfg.epsg_code);

    fprintf('=========== Remove Outliers: Normalized Median Test after deramp =========== \n')
    fprintf('  窗口: %d×%d\n', stats_nmt.window_size, stats_nmt.window_size);
    fprintf('  阈值: %.1f\n', stats_nmt.threshold);
    fprintf('  有效像素: %d\n', stats_nmt.n_valid);
    fprintf('  异常值: %d (%.2f%%)\n', stats_nmt.n_outliers, stats_nmt.outlier_pct);
    fprintf('  小区域: %d\n', stats_nmt.n_small);
    % fprintf('  归一化残差: %.4f\n', norm_residual_nmt)
    fprintf(' \n');

    
    %% --- 5. NMT,normalized median test方法 ( raw) ---
    [filtered_nmt_raw, outlier_mask_nmt_raw, norm_residual_nmt, stats_nmt] = apply_nmt_optimized(deformation, cfg.nmt_window, cfg.nmt_threshold, cfg.nmt_epsilon, cfg.min_segment_nmt);
    
    filtered_nmt_raw(~valid_mask) = NaN;
    geotiffwrite(fullfile(output_dir, [name, '-nmt_filtered_raw.tif']), filtered_nmt_raw, R, 'CoordRefSysCode', cfg.epsg_code);
    % geotiffwrite(fullfile(output_dir, [name, '-nmt_outlier_mask_raw.tif']), outlier_mask_nmt_raw, R, 'CoordRefSysCode', cfg.epsg_code);

    fprintf('=========== Remove Outliers: Normalized Median Test using Raw =========== \n')
    fprintf('  窗口: %d×%d\n', stats_nmt.window_size, stats_nmt.window_size);
    fprintf('  阈值: %.1f\n', stats_nmt.threshold);
    fprintf('  有效像素: %d\n', stats_nmt.n_valid);
    fprintf('  异常值: %d (%.2f%%)\n', stats_nmt.n_outliers, stats_nmt.outlier_pct);
    fprintf('  小区域: %d\n', stats_nmt.n_small);
    % fprintf('  归一化残差: %.4f\n', norm_residual_nmt)
    fprintf(' \n');

    
    %% --- 6. 中值滤波 ---
    % filtered_median = medfilt2(detrended, [cfg.medfilt2.window, cfg.medfilt2.window], 'symmetric');
    % geotiffwrite(fullfile(output_dir, [name, '-median_filtered.tif']), filtered_median, R, 'CoordRefSysCode', cfg.epsg_code);

    
    %% --- 7. smooth filter
    fprintf('=========== smooth filter and save as GTiff =========== \n')
    [smoothed_data, std_map, stats] = apply_final_smoothing_optimized(filtered_nmt, cfg.window_size, cfg);
    geotiffwrite(fullfile(output_dir, [name, '-smooth.tif']), smoothed_data, R, 'CoordRefSysCode', cfg.epsg_code);
    geotiffwrite(fullfile(output_dir, [name, '-std.tif']), std_map, R, 'CoordRefSysCode', cfg.epsg_code);

    
    %% --- 8. Visualization
    fprintf("====================================================== \n")
    fprintf("==================== Visualization Mode ============== \n")
    fprintf("====================================================== \n")

    % --- 调用可视化函数 ---
    [deformation, ~] = readgeoraster(tiff_data);

    vis_data.raw                  =  deformation;
    vis_data.ramp                 =  model_ramp;
    vis_data.detrended            =  detrended;

    vis_data.filtered_sigma       =  filtered_sigma;
    vis_data.outlier_mask_sigma   =  outlier_mask_sigma;

    vis_data.filtered_nmt         =  filtered_nmt;
    vis_data.outlier_mask_nmt     =  outlier_mask_nmt;

    vis_data.filtered_nmt_raw     =  filtered_nmt_raw;
    vis_data.outlier_mask_nmt_raw =  outlier_mask_nmt_raw;

    vis_data.smoothed_data        =  smoothed_data;
    vis_data.std_map              =  std_map;


    png_name = fullfile(output_dir, [name, '_summary.png']);
    plot_batch_results2(vis_data, cfg, png_name, ['File: ', name]);
    fprintf("** Save1: summary png, done! ** \n")

    smooth_name = fullfile(output_dir, [name, '_smooth_analysis.png']);
    plot_smooth_function(vis_data.filtered_nmt, cfg, name, smooth_name)
    fprintf("** save2: smooth_analysis png, done! ** \n")

   
    %% --- 9. Statistic
    % statistical_information(vis_data);

    output_file = fullfile(output_dir, [name, '_statistical_information.txt']);
    statistical_information2(vis_data, output_file)

end