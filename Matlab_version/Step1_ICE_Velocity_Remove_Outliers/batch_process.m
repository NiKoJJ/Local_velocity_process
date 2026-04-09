clc; clear; close all;

%% ========================================================================
%% 1. 配置信息
%% ========================================================================
% 设置输入根目录  range azimuth
input_dir = '/data2/Phd_Work1/ICE_Velocity_Process/GAMMA_POT_Input/Cook_120m'; 
folders = {'range'}; 

output_dir = {'/data2/Phd_Work1/ICE_Velocity_Process/GAMMA_POT_Output/Cook_120m/Remove_outliers_Correction'};

%% ========================================================================
%% 处理参数
%% ========================================================================

% data min max
cfg.data_max  = 40;
cfg.data_min  = -40;

% deramp
cfg.detrend_method = 'plane';
cfg.use_weights = true;

% N-sigma
cfg.sigma_times = 3;
cfg.min_segment_sigma = 100;

% NMT
cfg.nmt_window = 15;  % 15
cfg.nmt_threshold = 2.5; % 3.0
cfg.nmt_epsilon = 0.1;
cfg.min_segment_nmt = 100; % 75

cfg.medfilt2.window = 7;
cfg.epsg_code = 3031;

% smooth
cfg.ignore_nan = true;
cfg.min_valid = 15;
cfg.method = 'mean';  % mean median
cfg.window_size = 7;

% show
cfg.vmin = -15;
cfg.vmax = 15;

% cfg.mask_vmin = 0;
% cfg.mask_vmax = 1;

cfg.row = 3;
cfg.col = 3;

%% ========================================================================
%% 2. 批量循环处理
%% ========================================================================

for f = 1:length(folders)
    current_folder = folders{f};
    output_base = output_dir{f};
    input_path = fullfile(input_dir, current_folder);
    output_path = fullfile(output_base, current_folder);
    
    if ~exist(output_path, 'dir'), mkdir(output_path); end
    
    % 获取所有 .tif 文件
    file_list = dir(fullfile(input_path, '*.tif'));
    fprintf('\n开始处理目录: %s (共 %d 个文件)\n', current_folder, length(file_list));
    
    for i = 1:length(file_list)
        tiff_data = fullfile(input_path, file_list(i).name);
        [~, name, ~] = fileparts(file_list(i).name);
        
        fprintf('\n[进度 %d/%d] 正在处理: %s\n', i, length(file_list), file_list(i).name);
        % process_single_file(tiff_data, output_path, cfg);
        try
            % --- 调用处理函数 ---
            process_single_file(tiff_data, output_path, cfg);
            fprintf('✓ 文件 %s 处理成功\n', name);
        catch ME
            fprintf('× 文件 %s 处理失败! 错误原因: %s\n', name, ME.message);
            continue; % 跳过错误文件，处理下一个
        end
        % ============ 关键：每次都清理图形 ============
        close all force;
        
        % ============ 每10个文件深度清理 ============
        if mod(i, 50) == 0
            fprintf('  [内存清理] 已处理%d个文件...\n', i);
            
            % 保留必要变量，清理其他
            clearvars -except file_list i current_folder output_base ...
                             input_path output_path cfg f folders ...
                             output_dir input_dir total_start ...
                             success_count fail_count; 
            % Java垃圾回收
            java.lang.System.gc();
            
            % 给系统时间回收
            pause(0.5);
            
            % 显示内存使用（可选）
            try
                mem_info = memory;
                fprintf('  内存使用: %.2f GB\n', mem_info.MemUsedMATLAB/1024^3);
            catch
                % 某些系统不支持memory函数
            end
            
            fprintf('  ✓ 清理完成\n');
        end
        
        % ============ 每50个文件额外暂停 ============
        if mod(i, 50) == 0
            fprintf('  [休息] 已处理%d个文件，暂停2秒...\n', i);
            pause(2);
        end
    end
end
fprintf('\n========================================\n批量处理全部完成！\n');