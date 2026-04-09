% =========================================================================
%  情况一：无不确定性文件，用局部空间标准差估计 σ
%  依赖：weighted_average_m2.m, local_std.m（用户自备）
% =========================================================================
clear; clc;

%% ========== 用户参数区 ==========
data_dir   = '/data2/Phd_Work1/ICE_Velocity_Process/GAMMA_POT_Output2/Cook_80m/Tidal_IBE_Flexure_Correction/Vx_Vy_V';
output_dir = '/data2/Phd_Work1/ICE_Velocity_Process/GAMMA_POT_Output2/Cook_80m/Tidal_IBE_Flexure_Correction/Vx_Vy_V_Weighted_Average/ALL';

mkdir(output_dir);
epsg = 3031;

% local_std 参数
window_size = 5;     % 局部窗口大小（像素），建议奇数，5或7
min_valid   = 4;     % 窗口内至少多少个有效像素才计算std
ignore_nan  = true;

% 无效值阈值（根据你的数据调整）
no_data_val = 0;     % 原始数据中表示无效的值（如0或-9999）
%% ================================

%% 1. 查找所有 Vx / Vy 文件
vx_files = dir(fullfile(data_dir, '*-Vx.tif'));
vy_files = dir(fullfile(data_dir, '*-Vy.tif'));

% 按文件名排序（确保Vx和Vy顺序一致）
[~, idx] = sort({vx_files.name});  vx_files = vx_files(idx);
[~, idx] = sort({vy_files.name});  vy_files = vy_files(idx);

N = length(vx_files);
assert(N == length(vy_files), 'Vx和Vy文件数量不一致！');
fprintf('共找到 %d 个影像对\n', N);

%% 2. 读取第一幅图获取尺寸和空间参考
[vx_tmp, R] = readgeoraster(fullfile(data_dir, vx_files(1).name));
[rows, cols] = size(vx_tmp);

%% 3. 预估内存，提示用户
mem_gb = rows * cols * N * 4 / 1e9;
fprintf('预计需要内存：%.2f GB（×4个堆栈）= %.2f GB\n', mem_gb, mem_gb*4);
fprintf('若内存不足，请减少N或改用逐批处理。\n\n');

%% 4. 初始化堆栈
Vx_stack       = nan(rows, cols, N, 'single');
Vy_stack       = nan(rows, cols, N, 'single');
sigma_Vx_stack = nan(rows, cols, N, 'single');
sigma_Vy_stack = nan(rows, cols, N, 'single');

%% 5. 逐一读取文件，计算局部std作为σ
fprintf('正在读取文件并计算局部标准差（σ估计）...\n');
t0 = tic;

for i = 1:N
    if mod(i,10)==0 || i==1 || i==N
        fprintf('  [%3d/%3d] %s\n', i, N, vx_files(i).name);
    end
    
    % 读取
    vx = single(readgeoraster(fullfile(data_dir, vx_files(i).name)));
    vy = single(readgeoraster(fullfile(data_dir, vy_files(i).name)));
    
    % 无效值处理
    vx(vx == no_data_val | isinf(vx)) = NaN;
    vy(vy == no_data_val | isinf(vy)) = NaN;
    
    % 用局部空间std估计σ
    % 注意：这里的σ是局部空间变异性，是测量不确定性的近似
    sigma_vx = local_std(vx, window_size, ignore_nan, min_valid);
    sigma_vy = local_std(vy, window_size, ignore_nan, min_valid);
    
    % 存入堆栈
    Vx_stack(:,:,i)       = vx;
    Vy_stack(:,:,i)       = vy;
    sigma_Vx_stack(:,:,i) = sigma_vx;
    sigma_Vy_stack(:,:,i) = sigma_vy;
end
fprintf('读取完成，耗时 %.1f 秒\n\n', toc(t0));

%% 6. 加权平均（方法二-严谨版）
fprintf('正在进行加权平均（方法二）...\n');
t1 = tic;

[Vx_final, sigma_Vx_final, neff_Vx] = weighted_average_m2(Vx_stack, sigma_Vx_stack);
fprintf('  Vx 完成\n');
[Vy_final, sigma_Vy_final, neff_Vy] = weighted_average_m2(Vy_stack, sigma_Vy_stack);
fprintf('  Vy 完成，耗时 %.1f 秒\n\n', toc(t1));

%% 7. 由 Vx_final, Vy_final 计算 V_final（不直接平均V）
% V = sqrt(Vx² + Vy²)，误差传播：σ_V = sqrt((Vx/V·σ_Vx)² + (Vy/V·σ_Vy)²)
V_final = sqrt(Vx_final.^2 + Vy_final.^2);

safe_V = V_final;
safe_V(safe_V < 1e-8) = NaN;  % 避免除以零

sigma_V_final = sqrt((Vx_final ./ safe_V .* sigma_Vx_final).^2 + ...
                     (Vy_final ./ safe_V .* sigma_Vy_final).^2);
sigma_V_final(isnan(safe_V)) = NaN;

%% 8. 保存结果
fprintf('正在保存结果...\n');
geotiffwrite(fullfile(output_dir, 'ALL-Vx.tif'),       single(Vx_final),       R, 'CoordRefSysCode', epsg);
geotiffwrite(fullfile(output_dir, 'ALL-Vy.tif'),       single(Vy_final),       R, 'CoordRefSysCode', epsg);
geotiffwrite(fullfile(output_dir, 'ALL-V.tif'),        single(V_final),        R, 'CoordRefSysCode', epsg);
geotiffwrite(fullfile(output_dir, 'ALL-sigma_Vx.tif'), single(sigma_Vx_final), R, 'CoordRefSysCode', epsg);
geotiffwrite(fullfile(output_dir, 'ALL-sigma_Vy.tif'), single(sigma_Vy_final), R, 'CoordRefSysCode', epsg);
geotiffwrite(fullfile(output_dir, 'ALL-sigma_V.tif'),  single(sigma_V_final),  R, 'CoordRefSysCode', epsg);
geotiffwrite(fullfile(output_dir, 'ALL-neff_Vx.tif'),        single(neff_Vx),        R, 'CoordRefSysCode', epsg);  % 诊断图
geotiffwrite(fullfile(output_dir, 'ALL-neff_Vy.tif'),        single(neff_Vy),        R, 'CoordRefSysCode', epsg);

fprintf('\n=== 完成 ===\n');
fprintf('输出文件位于：%s\n', output_dir);
fprintf('输出说明：\n');
fprintf('  Vx/Vy/V_final.tif       最终融合速度\n');
fprintf('  sigma_Vx/Vy/V_final.tif 不确定性\n');
fprintf('  neff_Vx/Vy.tif          有效样本数（诊断用，越大越可靠）\n');

%% ---- 注意事项 ----
% 1. 用局部空间std作为σ是近似做法：
%    - 优点：不需要额外文件，实现简单
%    - 缺点：空间速度梯度大的区域（如剪切边缘）σ偏大，导致权重偏小
%      即这些区域的真实速度测量被"惩罚"了
%
% 2. window_size 建议取 5~9（太小噪声大，太大会模糊边界）
%
% 3. local_std 的循环较慢，若速度太慢可考虑用下方快速版本：
%    sigma_vx = movstd(vx, window_size, 0, 'omitnan');  % 仅1D，不等价
%    或使用 Image Processing Toolbox 的 stdfilt（不支持NaN）