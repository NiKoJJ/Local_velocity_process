function plot_smooth_function(data, cfg, name, output_path)
fprintf("====================================================== \n")
fprintf("==================== start smooth filter ============= \n")
fprintf("====================================================== \n")

row = 2;
col = 3;

fprintf('\n平滑处理参数:\n');
fprintf('  窗口大小: %d×%d\n', cfg.window_size, cfg.window_size);
fprintf('  方法: %s\n', cfg.method);
fprintf('  最小有效像素: %d\n', cfg.min_valid);

%% 执行平滑处理
tic;
[smoothed_data, std_map, stats] = apply_final_smoothing_optimized(data, cfg.window_size, cfg);
processing_time = toc;

fprintf('  处理时间: %.2f秒\n', processing_time);

%% 可视化 - 2行3列布局
fig = figure('Position', [50, 50, 2000, 1100], 'Color', 'w', 'Visible', 'off'); 
t = tiledlayout(row, col, 'TileSpacing', 'compact', 'Padding', 'compact');

% 总标题
title_str = sprintf('Final Smoothing Analysis | %s | Window: %d×%d | Method: %s', ...
                    name, cfg.window_size, cfg.window_size, upper(cfg.method));
title(t, title_str, 'FontSize', 24, 'FontWeight', 'bold', 'Interpreter', 'none');

%% 第一行: 图像显示
% 子图1: 原始数据
plot_image(data, '(a) NMT Filtered Data', [cfg.vmin, cfg.vmax], 1);
plot_image(smoothed_data, sprintf('(b) Smoothed Data (%s)', upper(cfg.method)), [cfg.vmin, cfg.vmax], 2);
plot_image(std_map, '(c) Local Std Deviation', [0, 2], 3);

%% 第二行: 直方图
plot_enhanced_histogram(data, '(d) NMT Filtered - Histogram', [cfg.vmin, cfg.vmax], 4);
plot_enhanced_histogram(smoothed_data, '(e) Smoothed - Histogram', [cfg.vmin, cfg.vmax], 5);
plot_enhanced_histogram(std_map, '(f) Std Deviation - Histogram', [0, 2], 6);

%% 打印详细统计
fprintf('\n========================================\n');
fprintf('详细统计分析\n');
fprintf('========================================\n');

fprintf('\n【NMT数据 - NMT Filtered】\n');
print_detailed_stats(data);

fprintf('\n【平滑后数据 - Smoothed】\n');
print_detailed_stats(smoothed_data);

fprintf('\n【标准差图 - Std Map】\n');
print_detailed_stats(std_map);

fprintf('\n【处理效果】\n');
compare_before_after(data, smoothed_data);

fprintf('========================================\n');

%% 保存图像
% output_name = sprintf('%s_smooth_analysis.png', name);
% saveas(fig, output_name, 'png');
saveas(fig, output_path);
fprintf('\n✓ 图像已保存: %s\n',output_path);

end