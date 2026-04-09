clc; clear;

% =========================================================================
% 1. ����������������������������������
% =========================================================================
lv_phi_tif   = './input/Mertz_path54/20251116.lv_phi_100m.tif';
azimuth_tif  = './input/Mertz_path54/20251122-20251128.azi2-100m.tif';
range_tif    = './input/Mertz_path54/20251122-20251128.range2-100m.tif';  % ��������������������

epsg_code = 3031;   % �������� EPSG:3031
unit      = 'year';  % ����������'day' �� 'year'

% =========================================================================
% 2. ��������������������������������
% =========================================================================
fprintf('���� %s ...\n', range_tif);
[range_data, R_range] = readgeoraster(range_tif);

fprintf('���� %s ...\n', azimuth_tif);
[azimuth_data, R_az] = readgeoraster(azimuth_tif);

fprintf('���� %s ...\n', lv_phi_tif);
[lv_phi_orig, R_phi] = readgeoraster(lv_phi_tif);

% ������������
fprintf('\n��������������\n');
fprintf('range_data  : %d �� %d\n', size(range_data,1), size(range_data,2));
fprintf('azimuth_data: %d �� %d\n', size(azimuth_data,1), size(azimuth_data,2));
fprintf('lv_phi_orig : %d �� %d\n', size(lv_phi_orig,1), size(lv_phi_orig,2));

% =========================================================================
% 3. ���� range �� azimuth ������������������������
% =========================================================================
if ~isequal(size(range_data), size(azimuth_data))
    error('range_data �� azimuth_data ����������������������');
end

% =========================================================================
% 4. ���� lv_phi ������ range �������������������������� interp2��
% =========================================================================
target_size = size(range_data);  % [����, ����]

if ~isequal(size(lv_phi_orig), target_size)
    fprintf('\n?? lv_phi ����������������������������...\n');
    
    % 4.1 ����������������������������Xq, Yq��
    % ������������������
    xlim_range = R_range.XWorldLimits;
    ylim_range = R_range.YWorldLimits;
    cols_range = target_size(2);
    rows_range = target_size(1);
    
    % ��������������X ��������������Y ������������������������ R ��������������
    % �������������� Y �������������������������� i ������Y ������
    % ������������������������������
    Xq = linspace(xlim_range(1) + R_range.SampleSpacingInWorldX/2, ...
                  xlim_range(2) - R_range.SampleSpacingInWorldX/2, cols_range);
    Yq = linspace(ylim_range(2) - R_range.SampleSpacingInWorldY/2, ...  % ������Y ��������
                  ylim_range(1) + R_range.SampleSpacingInWorldY/2, rows_range);
    % ���� Yq ����������interp2 ���� Yq ����������Xq ����������
    Xq = reshape(Xq, 1, []);   % ������
    Yq = reshape(Yq, [], 1);   % ������
    
    % 4.2 �������� lv_phi ����������������X_phi, Y_phi��
    xlim_phi = R_phi.XWorldLimits;
    ylim_phi = R_phi.YWorldLimits;
    cols_phi = size(lv_phi_orig,2);
    rows_phi = size(lv_phi_orig,1);
    
    X_phi = linspace(xlim_phi(1) + R_phi.SampleSpacingInWorldX/2, ...
                     xlim_phi(2) - R_phi.SampleSpacingInWorldX/2, cols_phi);
    Y_phi = linspace(ylim_phi(2) - R_phi.SampleSpacingInWorldY/2, ...  % ��������
                     ylim_phi(1) + R_phi.SampleSpacingInWorldY/2, rows_phi);
    X_phi = reshape(X_phi, 1, []);   % ������
    Y_phi = reshape(Y_phi, [], 1);   % ������
    
    % 4.3 ���� interp2 ���������������������������� NaN
    lv_phi = interp2(X_phi, Y_phi, double(lv_phi_orig), Xq, Yq, 'linear');
    % ������interp2 ���������������� NaN������������������������ NaN
    
    % ������������������ NaN����������
    if any(isnan(lv_phi(:)))
        fprintf('?? �������������� NaN��������������������������������������������\n');
    end
    
    fprintf('? ��������������������%d �� %d\n', size(lv_phi,1), size(lv_phi,2));
else
    lv_phi = double(lv_phi_orig);
    fprintf('\n? lv_phi ����������������������������\n');
end

% ������������������������
if ~isequal(size(range_data), size(azimuth_data), size(lv_phi))
    error('������������������range %s, azimuth %s, lv_phi %s', ...
        mat2str(size(range_data)), mat2str(size(azimuth_data)), mat2str(size(lv_phi)));
end

% =========================================================================
% 5. ������������������������������������������
% =========================================================================
[~, name, ~] = fileparts(range_tif);
% ������������'20251122-20251128.range2-100m'
date1_str = name(1:8);   % '20251122'
date2_str = name(10:17); % '20251128'

year1  = str2double(date1_str(1:4));
month1 = str2double(date1_str(5:6));
day1   = str2double(date1_str(7:8));

year2  = str2double(date2_str(1:4));
month2 = str2double(date2_str(5:6));
day2   = str2double(date2_str(7:8));

t1 = datetime(year1, month1, day1);
t2 = datetime(year2, month2, day2);

day_gap  = days(t2 - t1);
year_gap = years(t2 - t1);

fprintf('\n=== �������� ===\n');
fprintf('��������: %s\n', char(t1));
fprintf('��������: %s\n', char(t2));
fprintf('��������: %.2f �� = %.4f ��\n', day_gap, year_gap);

% =========================================================================
% 6. ��������������������������������
% =========================================================================
if strcmp(unit, 'year')
    Vgr = range_data  / year_gap;
    Vaz = azimuth_data / year_gap;
    unit_str = 'm/year';
elseif strcmp(unit, 'day')
    Vgr = range_data  / day_gap;
    Vaz = azimuth_data / day_gap;
    unit_str = 'm/day';
else
    error('���������� ''day'' �� ''year''');
end

% ������ double ��������������������������
Vgr = double(Vgr);
Vaz = double(Vaz);
lv_phi = double(lv_phi);

% =========================================================================
% 7. ��������
% =========================================================================
% ������������ lv_phi ������������
% ���� lv_phi ������������������������������������
% lv_phi = deg2rad(lv_phi);

fprintf('\n=== �������� ===\n');
u = Vgr .* cos(lv_phi) - Vaz .* sin(lv_phi);   % ��������������������
v = Vgr .* sin(lv_phi) + Vaz .* cos(lv_phi);   % ��������������������
V = hypot(u, v);                               % ����������

% =========================================================================
% 8. ���������� GeoTIFF
% =========================================================================
output_dir = './velocity';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
    fprintf('������������: %s\n', output_dir);
end

% ������������������������������������������
out_prefix = [date1_str '_' date2_str];

% ����������
out_file_V = fullfile(output_dir, [out_prefix, '_velocity_mag.tif']);
geotiffwrite(out_file_V, V, R_range, 'CoordRefSysCode', epsg_code);
fprintf('? ��������������: %s\n', out_file_V);

% ������������ u
out_file_u = fullfile(output_dir, [out_prefix, '_velocity_u.tif']);
geotiffwrite(out_file_u, u, R_range, 'CoordRefSysCode', epsg_code);
fprintf('? ������������ u ��������: %s\n', out_file_u);

% ������������ v
out_file_v = fullfile(output_dir, [out_prefix, '_velocity_v.tif']);
geotiffwrite(out_file_v, v, R_range, 'CoordRefSysCode', epsg_code);
fprintf('? ������������ v ��������: %s\n', out_file_v);

fprintf('\n?? ��������������\n');