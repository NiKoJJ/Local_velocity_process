% =========================================================================
%  batch_convert_mday_to_myear.m
%  批量将 V / Vx / Vy GeoTIFF 单位从 m/day 转换为 m/year
%
%  转换系数: × 365.25
%  目标文件: *V_final.tif / *Vx_final.tif / *Vy_final.tif
% =========================================================================
clear; clc;

%% ========== 用户参数 ==========
input_dir  = '/data2/Phd_Work1/ICE_Velocity_Process/GAMMA_POT_Output/Mertz_100m/Tidal_IBE_Flexure_Correction/Vx_Vy_V_Weighted_Average/ALL';
output_dir = '/data2/Phd_Work1/ICE_Velocity_Process/GAMMA_POT_Output/Mertz_100m/Tidal_IBE_Flexure_Correction/Vx_Vy_V_Weighted_Average/ALL/myear';

epsg        = 3031;    % 南极极地立体投影
no_data_in  = 0;       % 输入 NoData 值
suffix_out  = '_myear'; % 输出文件名后缀
%% ==============================

DAYS_PER_YEAR = 365.25;
mkdir(output_dir);

%% 只处理 V / Vx / Vy 三个分量（排除 sigma / neff）
targets = {'V_final.tif', 'Vx_final.tif', 'Vy_final.tif',...
   'sigma_V_final.tif','sigma_Vx_final.tif', 'sigma_Vy_final.tif'  };

fprintf('输入目录: %s\n', input_dir);
fprintf('输出目录: %s\n', output_dir);
fprintf('转换系数: × %.2f (m/day → m/year)\n\n', DAYS_PER_YEAR);

t_total = tic;
n_done  = 0;

for t = 1:length(targets)
    pattern   = fullfile(input_dir, ['*' targets{t}]);
    tif_files = dir(pattern);

    if isempty(tif_files)
        fprintf('[跳过] 未找到匹配文件: *%s\n\n', targets{t});
        continue;
    end

    for f = 1:length(tif_files)
        in_path = fullfile(tif_files(f).folder, tif_files(f).name);
        [~, fn] = fileparts(tif_files(f).name);
        out_name = [fn suffix_out '.tif'];
        out_path = fullfile(output_dir, out_name);

        fprintf('[%s]\n', tif_files(f).name);

        % 读取
        [img, R] = readgeoraster(in_path);
        img = single(img);
        img(img == no_data_in | ~isfinite(img)) = NaN;

        % 转换
        img_year = img * DAYS_PER_YEAR;

        % 统计
        fprintf('  m/day  范围: %8.4f ~ %8.4f\n', ...
            min(img(:),      [], 'omitnan'), max(img(:),      [], 'omitnan'));
        fprintf('  m/year 范围: %8.2f ~ %8.2f\n', ...
            min(img_year(:), [], 'omitnan'), max(img_year(:), [], 'omitnan'));

        % 保存
        geotiffwrite(out_path, single(img_year), R, 'CoordRefSysCode', epsg);
        fprintf('  已保存 → %s\n\n', out_name);
        n_done = n_done + 1;
    end
end

fprintf('=== 完成！共转换 %d 个文件，耗时 %.1f 秒 ===\n', n_done, toc(t_total));
fprintf('输出目录: %s\n', output_dir);
