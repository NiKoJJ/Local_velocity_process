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

tif_data = '/data2/Phd_Work1/ICE_Velocity_Process/GAMMA_POT_Output2/Mertz_100m/Tidal_IBE_Flexure_Correction/Vx_Vy_V_Weighted_Average/ALL/ALL-V.tif';
output = '/data2/Phd_Work1/ICE_Velocity_Process/GAMMA_POT_Output2/Mertz_100m/Tidal_IBE_Flexure_Correction/Vx_Vy_V_Weighted_Average/ALL';

result = apply_fill(tif_data,output, ...
      'Method',       'inpaint',  ...
      'SmoothFactor', 0.8,        ...
      'Colormap',     'turbo',    ...
      'ClimAuto',     false,      ...
      'ClimRange',    [0 800],    ...
      'FigDPI',       300);