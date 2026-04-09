% function [ground_mask, grounded_ratio, ice_bottom_dynamic] = detect_dynamic_grounding_line(...
%     ice_bottom, bed_elevation, tidal_height, IBE, varargin)
% % DETECT_DYNAMIC_GROUNDING_LINE 检测考虑潮汐和IBE的动态接地线
% %
% % 这是原始 Tidal_Flexure_correction.m 中的核心创新！
% % 接地线位置会随潮汐和IBE动态变化
% %
% % 语法:
% %   [ground_mask, grounded_ratio] = detect_dynamic_grounding_line(...
% %       ice_bottom, bed_elevation, tidal_height, IBE)
% %
% %   [...] = detect_dynamic_grounding_line(..., 'threshold', 1.0)
% %   [...] = detect_dynamic_grounding_line(..., 'fill_holes', true)
% %
% % 输入:
% %   ice_bottom      - 静态冰底高程 [m] (相对EGM2008大地水准面)
% %   bed_elevation   - 海床高程 [m] (相对EGM2008大地水准面)
% %   tidal_height    - 潮汐高度变化 [m] (date2 - date1)
% %   IBE             - 逆气压效应 [m] (date2 - date1)
% %
% % 可选参数（Name-Value pairs）:
% %   'threshold'     - 接地判据阈值 [m]，默认 1.0
% %   'fill_holes'    - 是否填充孤立点，默认 true
% %   'verbose'       - 是否输出详细信息，默认 false
% %
% % 输出:
% %   ground_mask          - 接地掩膜 [logical]
% %                          1 = 接地 (grounded)
% %                          0 = 浮动 (floating)
% %   grounded_ratio       - 接地区域比例 [%]
% %   ice_bottom_dynamic   - 动态冰底高程 [m] (考虑潮汐+IBE)
% %
% % 原理:
% %   动态冰底 = 静态冰底 + 潮汐高度 + IBE
% %   间隙 = 动态冰底 - 海床高程
% %   
% %   如果 间隙 > threshold:
% %       → 冰底明显高于海床 → 浮动
% %   否则:
% %       → 冰底接触或低于海床 → 接地
% %
% % 物理意义:
% %   高潮时: 海面升高 → 冰底抬升 → 接地线后退（向内陆）
% %   低潮时: 海面下降 → 冰底下降 → 接地线前进（向海洋）
% %
% % 示例:
% %   % 基本用法
% %   [ground_mask, ratio] = detect_dynamic_grounding_line(...
% %       ice_bottom, bed, tidal_height, IBE);
% %   
% %   % 使用更严格的阈值
% %   [ground_mask, ratio] = detect_dynamic_grounding_line(...
% %       ice_bottom, bed, tidal_height, IBE, 'threshold', 0.5);
% %   
% %   % 输出详细信息
% %   [ground_mask, ratio] = detect_dynamic_grounding_line(...
% %       ice_bottom, bed, tidal_height, IBE, 'verbose', true);
% %
% % 参考:
% %   Based on Tidal_Flexure_correction.m (原始实现)
% %
% % 作者: Claude AI
% % 日期: 2026-03-11
% 
% %% 解析输入参数
% p = inputParser;
% addRequired(p, 'ice_bottom', @isnumeric);
% addRequired(p, 'bed_elevation', @isnumeric);
% addRequired(p, 'tidal_height', @isnumeric);
% addRequired(p, 'IBE', @isnumeric);
% addParameter(p, 'threshold', 0.1, @(x) isnumeric(x) && isscalar(x) && x > 0);
% addParameter(p, 'fill_holes', true, @islogical);
% addParameter(p, 'verbose', false, @islogical);
% parse(p, ice_bottom, bed_elevation, tidal_height, IBE, varargin{:});
% 
% threshold = p.Results.threshold;
% fill_holes = p.Results.fill_holes;
% verbose = p.Results.verbose;
% 
% %% 检查输入维度
% if ~isequal(size(ice_bottom), size(bed_elevation), size(tidal_height), size(IBE))
%     error('所有输入数组必须维度一致！');
% end
% 
% %% 步骤1：计算动态冰底高程
% % ⭐ 关键创新：考虑潮汐和IBE的影响
% ice_bottom_dynamic = ice_bottom + tidal_height + IBE;
% 
% if verbose
%     fprintf('动态接地线检测:\n');
%     fprintf('  静态冰底: %.2f ± %.2f m\n', mean(ice_bottom(:),'omitnan'), std(ice_bottom(:),'omitnan'));
%     fprintf('  潮汐高度: %.3f ± %.3f m\n', mean(tidal_height(:),'omitnan'), std(tidal_height(:),'omitnan'));
%     fprintf('  IBE效应: %.4f ± %.4f m\n', mean(IBE(:),'omitnan'), std(IBE(:),'omitnan'));
%     fprintf('  动态冰底: %.2f ± %.2f m\n', mean(ice_bottom_dynamic(:),'omitnan'), std(ice_bottom_dynamic(:),'omitnan'));
% end
% 
% %% 步骤2：计算冰底-海床间隙
% A = double(ice_bottom_dynamic - bed_elevation);
% 
% if verbose
%     fprintf('  间隙范围: %.2f 到 %.2f m\n', min(A(:)), max(A(:)));
% end
% 
% %% 步骤3：初步接地判据
% % 修正：使用 A > threshold 而不是 abs(A) > threshold
% % 原始代码用 abs(A) 有问题：会将 A<-1 的接地区域误判为浮动
% Mask = zeros(size(A));
% Mask(abs(A) > threshold) = 1;  % 冰底高于海床 > threshold → 浮动
% 
% initial_floating_ratio = 100 * mean(Mask(:));
% if verbose
%     fprintf('  初步浮动比例: %.2f%%\n', initial_floating_ratio);
% end
% 
% %% 步骤4：填充孤立点（两次imfill，原始代码的精髓）
% if fill_holes
%     % 第一次填充：去除浮动区域内的孤立接地点
%     ground = imfill(Mask, 'holes');
% 
%     % 反转标签（准备第二次填充）
%     ground(ground == 0) = 2;
%     ground(ground == 1) = 0;
% 
%     % 第二次填充：去除接地区域内的孤立浮动点
%     ground_new = imfill(ground, 'holes');
% 
%     % 恢复标签
%     ground_new(ground_new == 0) = 0;  % 浮动
%     ground_new(ground_new == 2) = 1;  % 接地
% 
%     ground_mask = logical(ground_new);
% 
%     if verbose
%         fprintf('  填充后接地比例: %.2f%%\n', 100 * mean(ground_mask(:)));
%     end
% else
%     % 不填充，直接使用
%     ground_mask = logical(~Mask);  % 反转：0→接地，1→浮动
% end
% 
% %% 步骤5：计算接地比例
% grounded_ratio = 100 * sum(ground_mask(:)) / numel(ground_mask);
% 
% if verbose
%     fprintf('  最终接地比例: %.2f%%\n', grounded_ratio);
%     fprintf('  最终浮动比例: %.2f%%\n', 100 - grounded_ratio);
% end
% 
% %% 步骤6：质量检查
% if grounded_ratio < 5 || grounded_ratio > 95
%     warning('接地比例异常（%.1f%%），请检查输入数据！', grounded_ratio);
% end
% 
% % 检查极端值
% if any(abs(A(:)) > 1000)
%     warning('冰底-海床间隙存在极端值（>1000m），可能有数据错误！');
% end
% 
% end



function [ground_mask, floating_ratio, ice_bottom_dynamic] = detect_dynamic_grounding_line(...
    ice_bottom, bed_elevation, tidal_height, IBE, varargin)
% DETECT_DYNAMIC_GROUNDING_LINE ����������IBE��������������
% ������0 = ���� (Grounded), 1 = ���� (Floating)

%% 1. ��������
p = inputParser;
addRequired(p, 'ice_bottom', @isnumeric);
addRequired(p, 'bed_elevation', @isnumeric);
addRequired(p, 'tidal_height', @isnumeric);
addRequired(p, 'IBE', @isnumeric);
addParameter(p, 'threshold', 1.0, @(x) isnumeric(x) && isscalar(x));
addParameter(p, 'fill_holes', false, @islogical);
addParameter(p, 'verbose', false, @islogical);
parse(p, ice_bottom, bed_elevation, tidal_height, IBE, varargin{:});

threshold = p.Results.threshold;
fill_holes = p.Results.fill_holes;
verbose = p.Results.verbose;

%% 2. ������������
% ��������������������������IBE������������������ [cite: 8]
ice_bottom_dynamic = ice_bottom + tidal_height + IBE;

% �������������������� A [cite: 8, 29]
A = double(ice_bottom_dynamic - bed_elevation);

%% 3. ������������ (0=����, 1=����)
% �������������������������������� 1 (����)�������� 0 (����) [cite: 19]
Mask = zeros(size(A));
Mask(abs(A) > threshold) = 1; 

%% 4. ������������������
if fill_holes
    % --- �������� ---
    % A. ����������(1)����������
    temp_mask = imfill(logical(Mask), 'holes');
    
    % B. ����������(0)����������
    % ����������(0��1) -> ���� -> ����������
    temp_inv = ~temp_mask; 
    temp_inv = imfill(temp_inv, 'holes');
    
    % ����������0=����, 1=����
    ground_mask = ~temp_inv; 
else
    % --- �������������������������������� ---
    % ������Mask ������ 1=����, 0=����������������������
    ground_mask = logical(Mask); 
end

%% 5. ����������
floating_ratio = 100 * mean(ground_mask(:)); % ������������

if verbose
    fprintf('--- ������������������ ---\n');
    fprintf('  ��������: %.2f m\n', threshold);
    fprintf('  ��������: BedMachine v4.1 [cite: 1, 2]\n');
    fprintf('  ��������(1)����: %.2f%%\n', floating_ratio);
    fprintf('  ��������(0)����: %.2f%%\n', 100 - floating_ratio);
end

% ����������������
ground_mask = logical(ground_mask);

end