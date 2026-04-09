function out = apply_nlm(img, h, temp_win, srch_win)
% apply_nlm - 对单幅 single 图像做 NLM 滤波，自动处理 NaN 掩膜
%
% 输入:
%   img      - 输入图像 (single, 含 NaN)
%   h        - 滤波强度（DegreeOfSmoothing）
%   temp_win - patch 大小（ComparisonWindowSize，奇数）
%   srch_win - 搜索窗口大小（SearchWindowSize，奇数）
%
% 输出:
%   out      - NLM 滤波后图像（NaN 位置保持 NaN）

    nan_mask = isnan(img);

    % imnlmfilt 不接受 NaN，用 0 临时填充
    img_fill = img;
    img_fill(nan_mask) = 0;

    % 转 uint16 再滤波（imnlmfilt 对浮点直接支持）
    out = imnlmfilt(img_fill, ...
        'DegreeOfSmoothing',   h, ...
        'ComparisonWindowSize', temp_win, ...
        'SearchWindowSize',     srch_win);

    % 恢复原始 NaN 掩膜
    out(nan_mask) = NaN;
end