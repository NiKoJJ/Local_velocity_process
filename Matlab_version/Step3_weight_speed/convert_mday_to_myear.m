% =========================================================================
%  convert_mday_to_myear.m
%  将冰速 GeoTIFF 单位从 m/day 转换为 m/year
%
%  转换系数: 1 m/day × 365.25 = 365.25 m/year
% =========================================================================
clear; clc;

%% ========== 用户参数 ==========
input_tif  = '/data2/Phd_Work1/ICE_Velocity_Process/GAMMA_POT_Output2/Mertz_100m/2016-V.tif';
output_tif = '/data2/Phd_Work1/ICE_Velocity_Process/GAMMA_POT_Output2/Mertz_100m/2016-V_myear.tif';

epsg       = 3031;       % 投影，南极极地立体
no_data_in = 0;          % 输入文件的 NoData 值（0 视为无效）
%% ==============================

%% 1. 读取
fprintf('读取: %s\n', input_tif);
[img, R] = readgeoraster(input_tif);
img = single(img);

% 无效值处理
img(img == no_data_in | ~isfinite(img)) = NaN;

fprintf('图像尺寸: %d × %d\n', size(img, 1), size(img, 2));
fprintf('有效像素范围: %.4f ~ %.4f m/day\n', ...
    min(img(:), [], 'omitnan'), max(img(:), [], 'omitnan'));

%% 2. 单位转换
DAYS_PER_YEAR = 365.25;
img_year = img * DAYS_PER_YEAR;

fprintf('转换后范围:   %.2f ~ %.2f m/year\n', ...
    min(img_year(:), [], 'omitnan'), max(img_year(:), [], 'omitnan'));

%% 3. 保存
% 确保输出目录存在
out_dir = fileparts(output_tif);
if ~isempty(out_dir)
    mkdir(out_dir);
end

geotiffwrite(output_tif, single(img_year), R, 'CoordRefSysCode', epsg);
fprintf('已保存: %s\n', output_tif);
fprintf('单位转换完成: m/day → m/year (× %.2f)\n', DAYS_PER_YEAR);
