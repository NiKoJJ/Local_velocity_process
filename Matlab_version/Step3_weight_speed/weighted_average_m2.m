function [v_hat, sigma_hat, n_eff_map] = weighted_average_m2(V_stack, sigma_stack)
% WEIGHTED_AVERAGE_M2  方法二（严谨版）加权平均
%
% 输入：
%   V_stack     [rows, cols, N]  N幅速度图（单分量，如Vx）
%   sigma_stack [rows, cols, N]  对应的不确定性图
%
% 输出：
%   v_hat       [rows, cols]     加权平均速度
%   sigma_hat   [rows, cols]     最终不确定性
%   n_eff_map   [rows, cols]     有效样本数（诊断用）
%
% 公式来源：Bevington (1969), Joughin et al. (2021)

[rows, cols, N] = size(V_stack);

%% 屏蔽无效值：速度或sigma为NaN/Inf/零sigma 的位置权重置NaN
sigma_stack(sigma_stack < 1e-8) = NaN;   % 避免除以零
w_stack = 1 ./ (sigma_stack .^ 2);       % 权重 w_n = 1/σ_n²

% 速度无效则对应权重也无效
invalid = isnan(V_stack) | isinf(V_stack) | isnan(w_stack);
w_stack(invalid) = NaN;
V_stack(invalid) = NaN;

%% ---- 第一步：加权平均 ----
sum_w  = sum(w_stack, 3, 'omitnan');          % Σ w_n        [rows,cols]
sum_wv = sum(w_stack .* V_stack, 3, 'omitnan'); % Σ w_n*v_n  [rows,cols]

v_hat = sum_wv ./ sum_w;
v_hat(sum_w == 0) = NaN;

%% ---- 第二步：有效样本数 n_eff ----
% n_eff = (Σw_n)² / Σ(w_n²)
% 若所有图权重相等，n_eff = N；若某幅图权重极大，n_eff → 1
sum_w2   = sum(w_stack .^ 2, 3, 'omitnan');  % Σ w_n²
n_eff_map = (sum_w .^ 2) ./ sum_w2;

%% ---- 第三步：加权方差（考虑实际残差）----
% 将 v_hat 扩展到 [rows,cols,N] 以便广播相减
v_hat_3d = repmat(v_hat, 1, 1, N);

residuals  = V_stack - v_hat_3d;              % v_n - v̄
w_resid2   = sum(w_stack .* residuals.^2, 3, 'omitnan');  % Σ w_n*(v_n-v̄)²

% 加权方差（含自由度修正）
var_wtd = (w_resid2 ./ sum_w) .* (n_eff_map ./ max(n_eff_map - 1, eps));

%% ---- 第四步：最终不确定性 ----
sigma_hat = sqrt(var_wtd ./ n_eff_map);

% 有效观测不足时（n_eff≤1）结果不可信，置NaN
sigma_hat(n_eff_map <= 1 | sum_w == 0) = NaN;
v_hat    (sum_w == 0)                  = NaN;

end