function M = calculate_vaughan_flexure_mask(xx, yy, thickness, ground_mask, E, nu, rho_sw, g)
% 计算Vaughan (1995)的一维线性弹性弯曲掩膜
%
% 公式 (Text S5, Equation 6):
%   M(x) = 1 - exp(-β·x) · [cos(β·x) + sin(β·x)]
%   β = [3·ρ_sw·g·(1-ν²) / (E·H³)]^(1/4)
%
% 其中:
%   x - 到接地线的正交距离 [m]
%   H - 冰厚度 [m]
%   E - 杨氏模量 [Pa]
%   ν - 泊松比
%   ρ_sw - 海水密度 [kg/m³]
%   g - 重力加速度 [m/s²]

    % 计算到接地线的距离
    pixel_size = abs(xx(1,2) - xx(1,1));
    dist_to_GL = double(bwdist(ground_mask)) * pixel_size;  % [m
    figure("Name","distance  to GL",'Visible','on')
    imagesc(dist_to_GL);title("distance  to GL");colorbar
    format_fig(gcf, './', 'distance2GL');
    
    % 计算特征参数β [1/m]
    beta = (3 * rho_sw * g * (1 - nu^2) ./ (E .* thickness.^3)).^(1/4);
    
    % 处理无效值
    beta(isinf(beta) | isnan(beta)) = 0;
    % beta(thickness < 5) = 0;  % 太薄的冰不计算弯曲
    
    % 计算弯曲掩膜
    M_raw = 1 - exp(-beta .* dist_to_GL) .* ...
            (cos(beta .* dist_to_GL) + sin(beta .* dist_to_GL));
    
    % 掩膜处理
    M_raw(ground_mask == 1) = 0;      % 接地区域M=0（不响应潮汐）
    M_raw(isnan(M_raw)) = 1;          % 无数据区域M=1（完全响应）
    M_raw(isinf(M_raw)) = 1;
    
    % 低通滤波（平滑到4km）
    try
        M = filt2(M_raw, pixel_size, 2*2000, 'lp');
    catch
        % 如果没有filt2函数，使用高斯平滑
        sigma = 4000 / pixel_size / 2.355;  % 4km FWHM
        M = imgaussfilt(M_raw, sigma);
    end
    
    % 确保物理范围 [0, 1]
    % M(M < 0) = 0;
    % M(M > 1) = 1;
end
