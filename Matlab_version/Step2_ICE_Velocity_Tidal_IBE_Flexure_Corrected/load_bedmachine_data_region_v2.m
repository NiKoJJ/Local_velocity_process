function [thickness, bed, surface, ice_bottom] = ...
    load_bedmachine_data_region_v2(lat, lon)

% 检查输入维度
if ~isequal(size(lat), size(lon))
    error('lat和lon必须维度一致！');
end

% 加载BedMachine数据
fprintf('加载BedMachine数据 (reference: %s)...\n', 'WGS84');

try
% 使用BedMachine工具箱插值
    thickness = bedmachine_interp('thickness', lat, lon, 'geoid');
    bed = bedmachine_interp('bed', lat, lon, 'geoid');
    surface = bedmachine_interp('surface', lat, lon, 'geoid');

    fprintf('  使用BedMachine工具箱成功\n');

catch ME
    error('BedMachine工具箱调用失败: %s\n建议安装: https://github.com/chadagreene/BedMachine', ME.message);
end

% 计算静态冰底高程
ice_bottom = surface - thickness;

fprintf('  数据统计:\n');
fprintf('    冰面高程: %.1f ± %.1f m\n', mean(surface(:),'omitnan'), std(surface(:),'omitnan'));
fprintf('    冰厚度: %.1f ± %.1f m\n', mean(thickness(:),'omitnan'), std(thickness(:),'omitnan'));
fprintf('    海床高程: %.1f ± %.1f m\n', mean(bed(:),'omitnan'), std(bed(:),'omitnan'));
fprintf('    静态冰底: %.1f ± %.1f m\n', mean(ice_bottom(:),'omitnan'), std(ice_bottom(:),'omitnan'));

end
