clc; clear;

% =========================================================================
% 1. Input file paths and parameters
% =========================================================================
lv_phi_tif   = './input/Mertz_path54/20251116.lv_phi_100m.tif';
azimuth_tif  = './input/Mertz_path54/20251122-20251128.azi2-100m.tif';
range_tif    = './input/Mertz_path54/20251122-20251128.range2-100m.tif';  % Dates are parsed from this filename

epsg_code = 3031;   % Output coordinate reference system: EPSG:3031
unit      = 'year';  % Velocity output unit: 'day' or 'year'

% =========================================================================
% 2. Read input rasters
% =========================================================================
fprintf('Reading %s ...\n', range_tif);
[range_data, R_range] = readgeoraster(range_tif);

fprintf('Reading %s ...\n', azimuth_tif);
[azimuth_data, R_az] = readgeoraster(azimuth_tif);

fprintf('Reading %s ...\n', lv_phi_tif);
[lv_phi_orig, R_phi] = readgeoraster(lv_phi_tif);

% Print grid dimensions for verification
fprintf('\nInput raster dimensions:\n');
fprintf('range_data  : %d rows x %d cols\n', size(range_data,1), size(range_data,2));
fprintf('azimuth_data: %d rows x %d cols\n', size(azimuth_data,1), size(azimuth_data,2));
fprintf('lv_phi_orig : %d rows x %d cols\n', size(lv_phi_orig,1), size(lv_phi_orig,2));

% =========================================================================
% 3. Verify that range and azimuth grids have the same dimensions
% =========================================================================
if ~isequal(size(range_data), size(azimuth_data))
    error('range_data and azimuth_data must have identical dimensions.');
end

% =========================================================================
% 4. Resample lv_phi to match the range grid if dimensions differ (using interp2)
% =========================================================================
target_size = size(range_data);  % [rows, cols]

if ~isequal(size(lv_phi_orig), target_size)
    fprintf('\nlv_phi grid differs from target �� resampling to match range grid ...\n');

    % 4.1 Build query coordinate vectors (Xq, Yq) for the target (range) grid
    xlim_range = R_range.XWorldLimits;
    ylim_range = R_range.YWorldLimits;
    cols_range = target_size(2);
    rows_range = target_size(1);

    % X increases left��right; Y decreases top��bottom (north-up raster).
    % Pixel centres are offset by half a pixel from the world limits.
    Xq = linspace(xlim_range(1) + R_range.SampleSpacingInWorldX/2, ...
                  xlim_range(2) - R_range.SampleSpacingInWorldX/2, cols_range);
    Yq = linspace(ylim_range(2) - R_range.SampleSpacingInWorldY/2, ...  % Y starts from top (max)
                  ylim_range(1) + R_range.SampleSpacingInWorldY/2, rows_range);
    % Shape vectors so interp2 interprets Xq as column indices and Yq as row indices
    Xq = reshape(Xq, 1, []);   % row vector
    Yq = reshape(Yq, [], 1);   % column vector

    % 4.2 Build source coordinate vectors for lv_phi
    xlim_phi = R_phi.XWorldLimits;
    ylim_phi = R_phi.YWorldLimits;
    cols_phi = size(lv_phi_orig,2);
    rows_phi = size(lv_phi_orig,1);

    X_phi = linspace(xlim_phi(1) + R_phi.SampleSpacingInWorldX/2, ...
                     xlim_phi(2) - R_phi.SampleSpacingInWorldX/2, cols_phi);
    Y_phi = linspace(ylim_phi(2) - R_phi.SampleSpacingInWorldY/2, ...  % Y starts from top
                     ylim_phi(1) + R_phi.SampleSpacingInWorldY/2, rows_phi);
    X_phi = reshape(X_phi, 1, []);   % row vector
    Y_phi = reshape(Y_phi, [], 1);   % column vector

    % 4.3 Resample using bilinear interpolation; query points outside source
    %     extent are filled with NaN automatically
    lv_phi = interp2(X_phi, Y_phi, double(lv_phi_orig), Xq, Yq, 'linear');

    % Warn if any NaN values appear in the resampled result
    if any(isnan(lv_phi(:)))
        fprintf('Warning: NaN values present in resampled lv_phi �� query points may lie outside the source extent.\n');
    end

    fprintf('Resampling complete �� output size: %d rows x %d cols\n', size(lv_phi,1), size(lv_phi,2));
else
    lv_phi = double(lv_phi_orig);
    fprintf('\nlv_phi already matches the target grid �� no resampling needed.\n');
end

% Final dimension check
if ~isequal(size(range_data), size(azimuth_data), size(lv_phi))
    error('Grid size mismatch after resampling: range %s, azimuth %s, lv_phi %s', ...
        mat2str(size(range_data)), mat2str(size(azimuth_data)), mat2str(size(lv_phi)));
end

% =========================================================================
% 5. Parse acquisition dates from the range filename and compute time span
% =========================================================================
[~, name, ~] = fileparts(range_tif);
% Expected filename format: '20251122-20251128.range2-100m'
date1_str = name(1:8);   % e.g. '20251122'
date2_str = name(10:17); % e.g. '20251128'

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

fprintf('\n=== Time span ===\n');
fprintf('Start date : %s\n', char(t1));
fprintf('End date   : %s\n', char(t2));
fprintf('Time span  : %.2f days = %.4f years\n', day_gap, year_gap);

% =========================================================================
% 6. Convert displacement to velocity
% =========================================================================
if strcmp(unit, 'year')
    Vgr = range_data   / year_gap;
    Vaz = azimuth_data / year_gap;
    unit_str = 'm/year';
elseif strcmp(unit, 'day')
    Vgr = range_data   / day_gap;
    Vaz = azimuth_data / day_gap;
    unit_str = 'm/day';
else
    error('Unknown unit must be ''day'' or ''year''.');
end

% Cast to double for subsequent arithmetic
Vgr    = double(Vgr);
Vaz    = double(Vaz);
lv_phi = double(lv_phi);

% =========================================================================
% 7. Decompose into map-projected (East/North) velocity components
% =========================================================================
% lv_phi is the SAR look-vector azimuth angle (heading angle of the
% ground-range direction projected onto the horizontal plane).
% Uncomment the line below if lv_phi is stored in degrees rather than radians:
% lv_phi = deg2rad(lv_phi);

fprintf('\n=== Velocity decomposition ===\n');
u = Vgr .* cos(lv_phi) - Vaz .* sin(lv_phi);   % East component  (Vx)
v = Vgr .* sin(lv_phi) + Vaz .* cos(lv_phi);   % North component (Vy)
V = hypot(u, v);                                 % Speed magnitude

% =========================================================================
% 8. Write output GeoTIFFs
% =========================================================================
output_dir = './velocity';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
    fprintf('Created output directory: %s\n', output_dir);
end

% Build output filename prefix from the two acquisition dates
out_prefix = [date1_str '_' date2_str];

% Speed magnitude
out_file_V = fullfile(output_dir, [out_prefix, '_velocity_mag.tif']);
geotiffwrite(out_file_V, V, R_range, 'CoordRefSysCode', epsg_code);
fprintf('Saved speed magnitude : %s\n', out_file_V);

% East component (u)
out_file_u = fullfile(output_dir, [out_prefix, '_velocity_u.tif']);
geotiffwrite(out_file_u, u, R_range, 'CoordRefSysCode', epsg_code);
fprintf('Saved East component  : %s\n', out_file_u);

% North component (v)
out_file_v = fullfile(output_dir, [out_prefix, '_velocity_v.tif']);
geotiffwrite(out_file_v, v, R_range, 'CoordRefSysCode', epsg_code);
fprintf('Saved North component : %s\n', out_file_v);

fprintf('\nDone.\n');
