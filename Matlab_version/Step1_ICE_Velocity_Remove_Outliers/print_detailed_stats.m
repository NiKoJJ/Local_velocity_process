function print_detailed_stats(data)
% 打印详细统计信息
    valid_data = data(~isnan(data) & ~isinf(data));
    
    if isempty(valid_data)
        fprintf('  无有效数据\n');
        return;
    end
    
    n = length(valid_data);
    mean_val = mean(valid_data);
    median_val = median(valid_data);
    std_val = std(valid_data);
    min_val = min(valid_data);
    max_val = max(valid_data);
    
    q25 = prctile(valid_data, 25);
    q50 = prctile(valid_data, 50);
    q75 = prctile(valid_data, 75);
    
    iqr = q75 - q25;
    
    fprintf('  基本统计:\n');
    fprintf('    有效像素: %d\n', n);
    fprintf('    均值 (μ): %.6f\n', mean_val);
    fprintf('    中值: %.6f\n', median_val);
    fprintf('    标准差 (σ): %.6f\n', std_val);
    fprintf('    最小值: %.6f\n', min_val);
    fprintf('    最大值: %.6f\n', max_val);
    fprintf('    范围: %.6f\n', max_val - min_val);
    
    fprintf('  分位数:\n');
    fprintf('    Q1 (25%%): %.6f\n', q25);
    fprintf('    Q2 (50%%): %.6f\n', q50);
    fprintf('    Q3 (75%%): %.6f\n', q75);
    fprintf('    IQR (Q3-Q1): %.6f\n', iqr);
    
    fprintf('  置信区间:\n');
    fprintf('    μ ± 1σ: [%.6f, %.6f] (68.3%%)\n', mean_val - std_val, mean_val + std_val);
    fprintf('    μ ± 2σ: [%.6f, %.6f] (95.4%%)\n', mean_val - 2*std_val, mean_val + 2*std_val);
    fprintf('    μ ± 3σ: [%.6f, %.6f] (99.7%%)\n', mean_val - 3*std_val, mean_val + 3*std_val);
end