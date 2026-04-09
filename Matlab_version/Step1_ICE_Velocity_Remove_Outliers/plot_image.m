function plot_image(data, title_str, range, index)
    nexttile(index);
% 绘制图像 (增强版)
    im = imagesc(data);
    set(im, 'AlphaData', ~isnan(data));
    axis image; axis off;
    colormap(gca, turbo);
    
    if ~isempty(range)
        clim(range);
    end
    
    % 标题
    title(title_str, 'FontSize', 20, 'FontWeight', 'bold', 'Color', [0.15 0.15 0.15]);
    
    % Colorbar
    cb = colorbar('FontSize', 18);
    cb.Label.String = 'deformation (m)';
    cb.Label.FontSize = 18;
    
    % 背景
    set(gca, 'Color', [0.92 0.92 0.92]); 
    
    % 添加统计信息
    valid_data = data(~isnan(data) & ~isinf(data));
    if ~isempty(valid_data)
        stats_str = sprintf('N=%d \n μ=%.2f \n σ=%.2f', ...
                           length(valid_data), mean(valid_data), std(valid_data));
        text(0.02, 0.98, stats_str, ...
             'Units', 'normalized', ...
             'VerticalAlignment', 'top', ...
             'FontSize', 18, ...
             'BackgroundColor', [1 1 1 0.8], ...
             'EdgeColor', [0.5 0.5 0.5], ...
             'Margin', 3);
    end
end
