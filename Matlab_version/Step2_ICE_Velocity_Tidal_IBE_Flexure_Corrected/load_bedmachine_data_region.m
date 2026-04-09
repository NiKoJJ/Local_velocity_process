function [thickness, bed, ground_mask] = load_bedmachine_data_region(lat, lon)
% 加载BedMachine数据的子区域

    % [lat, lon] = ps2ll(xx, yy);
    % 检查输入维度
    if ~isequal(size(lat), size(lon))
        error('lat和lon必须维度一致！');
    end
    
    % 加载BedMachine数据
    fprintf('加载BedMachine数据 (reference: %s)...\n', 'WGS84');
    
    try
        % 方法1: 使用BedMachine工具箱
        thickness = bedmachine_interp('thickness', lat, lon, 'geoid');
        bed = bedmachine_interp('bed', lat, lon, 'geoid');
        mask_bm = bedmachine_interp('mask', lat, lon, 'geoid');
        
        % 接地掩膜: mask值 0=ocean, 1=ice-free land, 2=grounded ice, 3=floating, 4=lake
        ground_mask = ismember(mask_bm, [1, 2]);  % 1和2为接地
        
        % 数据清理
        thickness(thickness < 0) = 0;
        thickness(thickness > 5000) = NaN;
        
    catch ME
        error('BedMachine工具箱调用失败: %s\n建议安装: https://github.com/chadagreene/BedMachine', ME.message);
        % % 方法2: 如果无法访问BedMachine，使用默认值
        % warning('无法加载BedMachine数据，使用默认值: %s', ME.message);
        % thickness = 500 * ones(size(xx));  % 默认500m厚度
        % bed = zeros(size(xx));
        % ground_mask = zeros(size(xx));     % 假设全部浮动
    end
end