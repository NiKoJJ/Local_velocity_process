function compare_before_after(data_before, data_after)
% 对比平滑前后的变化
    valid_before = data_before(~isnan(data_before) & ~isinf(data_before));
    valid_after = data_after(~isnan(data_after) & ~isinf(data_after));
    
    if isempty(valid_before) || isempty(valid_after)
        fprintf('  无法对比\n');
        return;
    end
    
    % 统计量变化
    mean_before = mean(valid_before);
    mean_after = mean(valid_after);
    std_before = std(valid_before);
    std_after = std(valid_after);
    
    mean_change = abs(mean_after - mean_before);
    std_change = abs(std_after - std_before);
    std_reduction = (1 - std_after/std_before) * 100;
    
    fprintf('  均值变化: %.6f → %.6f (Δ = %.6f)\n', ...
            mean_before, mean_after, mean_change);
    fprintf('  标准差变化: %.6f → %.6f (Δ = %.6f, 降低 %.2f%%)\n', ...
            std_before, std_after, std_change, std_reduction);
    
    % 有效像素变化
    n_before = length(valid_before);
    n_after = length(valid_after);
    n_change = n_after - n_before;
    
    fprintf('  有效像素: %d → %d (Δ = %d)\n', n_before, n_after, n_change);
    
    % 相关性
    both_valid = ~isnan(data_before) & ~isnan(data_after);
    if sum(both_valid(:)) > 100
        corr_val = corr(data_before(both_valid), data_after(both_valid));
        fprintf('  相关系数: %.6f\n', corr_val);
    end
    
    % RMS
    rms_before = sqrt(mean(valid_before.^2));
    rms_after = sqrt(mean(valid_after.^2));
    rms_change = (1 - rms_after/rms_before) * 100;
    
    fprintf('  RMS: %.6f → %.6f (变化 %.2f%%)\n', ...
            rms_before, rms_after, rms_change);
end
