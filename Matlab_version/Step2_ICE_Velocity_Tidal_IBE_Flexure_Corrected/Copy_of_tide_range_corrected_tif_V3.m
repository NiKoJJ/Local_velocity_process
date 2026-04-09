%% InSAR Glacier Velocity Processing with Tidal & IBE Correction
% Workflow:
%   1. Load range/azimuth displacement and geometry files
%   2. Tidal correction (CATS2008)
%   3. Inverse Barometer Effect (IBE) correction from ERA5
%      3a. Read ERA5 coordinate axes, crop to velocity grid extent
%      3b. Read only the cropped region pressure data
%      3c. Compute IBE, interpolate to velocity grid
%   4. Flexure mask calculation
%   5. Velocity decomposition (Vgr/Vaz -> Vx/Vy)
%   6. Save outputs

clc; clear; close all;

%% =========================================================
%  1. Configuration
% %% =========================================================
% cfg.tiff_range   = './range_tif/20170511-20170523.range-nmt_filtered.tif';
% cfg.tiff_azimuth = './range_tif/20170511-20170523.azi-nmt_filtered.tif';
% cfg.tiff_lv_phi  = './range_tif/20170511.lv_phi.tif';
% cfg.tiff_inc     = './range_tif/20170511.inc.tif';

cfg.tiff_range   = '/data2/Phd_Work1/ICE_Velocity_Process/GAMMA_POT_Output2/Cook_80m/Remove_outliers_Correction/range/20200413-20200419.range-nmt_filtered.tif';
cfg.tiff_azimuth = '/data2/Phd_Work1/ICE_Velocity_Process/GAMMA_POT_Output2/Cook_80m/Remove_outliers_Correction/azimuth/20200413-20200419.azi-nmt_filtered.tif';
cfg.tiff_lv_phi  = '/data2/Phd_Work1/ICE_Velocity_Process/GAMMA_POT_Output2/Cook_80m/Remove_outliers_Correction/geometry/20200407.inc.tif';
cfg.tiff_inc     = '/data2/Phd_Work1/ICE_Velocity_Process/GAMMA_POT_Output2/Cook_80m/Remove_outliers_Correction/geometry/20200407.lv_phi.tif';

cfg.nc_tide      = 'CATS2008_v2023.nc';
cfg.nc_era5      = 'ERA5-2015-2025.nc';

cfg.res          = 80;      % Grid resolution [m]
cfg.epsg_code    = 3031;    % Output CRS (Antarctic Polar Stereographic)
cfg.unit         = 'year';   % Velocity unit: 'day' or 'year'
cfg.IBE_warn_thr = 5;       % Warning threshold [m]
cfg.era5_buf     = 0.5;     % ERA5 crop buffer [degrees]

cfg.E       = 0.88e9;   % Young's modulus [Pa]
cfg.nu      = 0.3;      % Poisson's ratio
cfg.rho_sw  = 1027;     % Seawater density [kg/m3]
cfg.g       = 9.81;     % Gravitational acceleration [m/s2]

cfg.out_vel  = './Corrected_Velocity';
cfg.out_vxvy = './Vx_Vy_V';

cfg.range_suffix = '.range';

%% =========================================================
%  2. Load Displacement & Geometry Data
%% =========================================================
fprintf('=== Loading Data ===\n');

[a, name_ext, c] = fileparts(cfg.tiff_range);
name = extractBefore(name_ext, cfg.range_suffix); 

[range_data, R]    = readgeoraster(cfg.tiff_range);
[azimuth_data, ~]  = readgeoraster(cfg.tiff_azimuth);
[lv_phi, ~]        = readgeoraster(cfg.tiff_lv_phi);
[inc, ~]           = readgeoraster(cfg.tiff_inc);

%% =========================================================
%  3. Build Coordinate Grid
%% =========================================================
fprintf('=== Building Grid ===\n');

X = R.XWorldLimits(1) : cfg.res : R.XWorldLimits(2);
Y = R.YWorldLimits(2) : -cfg.res : R.YWorldLimits(1);
[x, y] = meshgrid(X, Y);
[lat, lon] = ps2ll(x, y);

fprintf('  Latitude  range: %.4f ~ %.4f\n', min(lat(:)), max(lat(:)));
fprintf('  Longitude range: %.4f ~ %.4f\n', min(lon(:)), max(lon(:)));

%% =========================================================
%  4. Parse Dates from Filename
%% =========================================================
fprintf('=== Parsing Dates ===\n');

t1 = datetime(str2double(name(1:4)),  str2double(name(5:6)),  str2double(name(7:8)));
t2 = datetime(str2double(name(10:13)),str2double(name(14:15)),str2double(name(16:17)));

day_gap  = days(t2 - t1);
year_gap = years(t2 - t1);

t_acq = datetime(t1.Year,t1.Month,t1.Day,10,7,32) : day_gap : ...
        datetime(t2.Year,t2.Month,t2.Day,10,7,32);

fprintf('  Start: %s | End: %s | Gap: %.0f days (%.4f yr)\n', ...
        char(t1), char(t2), day_gap, year_gap);

%% =========================================================
%  5. Tidal Correction (CATS2008)
%% =========================================================
fprintf('=== Tidal Correction ===\n');

Z = tmd_predict(cfg.nc_tide, lat, lon, t_acq, 'h', 'coasts', 'nan');
fprintf('  Tidal grid size: %d x %d x %d\n', size(Z,1), size(Z,2), size(Z,3));

Tidal_height = Z(:,:,2) - Z(:,:,1);   % h(t2) - h(t1) [m]
Tidal_height(isnan(Tidal_height)) = 0;

save_geotiff(cfg.out_vel, [name, '-Tidal_height.tif'], Tidal_height, R, cfg.epsg_code);
%% =========================================================
%  6. IBE Correction  (with ERA5 spatial crop)
%% =========================================================
fprintf('=== IBE Correction ===\n');

% ---------------------------------------------------------
%  6a. Read ERA5 coordinate axes only (very fast, no pressure data yet)
%      sp dimension order: (longitude, latitude, valid_time)
%      latitude: stored_direction = 'decreasing' (north to south)
% ---------------------------------------------------------
lat_ERA5_all = ncread(cfg.nc_era5, 'latitude');    % (nlat_all x 1)
lon_ERA5_all = ncread(cfg.nc_era5, 'longitude');   % (nlon_all x 1)
time_era5    = ncread(cfg.nc_era5, 'valid_time');  % (nt x 1) seconds since 1970-01-01

fprintf('  ERA5 full extent: lon [%.2f, %.2f]  lat [%.2f, %.2f]\n', ...
        min(lon_ERA5_all), max(lon_ERA5_all), ...
        min(lat_ERA5_all), max(lat_ERA5_all));

% ---------------------------------------------------------
%  6b. Compute crop indices based on velocity grid extent + buffer
% ---------------------------------------------------------
lon_min_q = min(lon(:)) - cfg.era5_buf;
lon_max_q = max(lon(:)) + cfg.era5_buf;
lat_min_q = min(lat(:)) - cfg.era5_buf;
lat_max_q = max(lat(:)) + cfg.era5_buf;

lon_idx = find(lon_ERA5_all >= lon_min_q & lon_ERA5_all <= lon_max_q);
lat_idx = find(lat_ERA5_all >= lat_min_q & lat_ERA5_all <= lat_max_q);

if isempty(lon_idx) || isempty(lat_idx)
    error(['ERA5 coverage does not overlap with the velocity grid.\n' ...
           'ERA5 lon: %.2f~%.2f  lat: %.2f~%.2f\n' ...
           'Required lon: %.2f~%.2f  lat: %.2f~%.2f'], ...
           min(lon_ERA5_all), max(lon_ERA5_all), ...
           min(lat_ERA5_all), max(lat_ERA5_all), ...
           lon_min_q, lon_max_q, lat_min_q, lat_max_q);
end

lon_start = min(lon_idx);   lon_count = numel(lon_idx);
lat_start = min(lat_idx);   lat_count = numel(lat_idx);

lat_ERA5 = lat_ERA5_all(lat_idx);   % (nlat_crop x 1)
lon_ERA5 = lon_ERA5_all(lon_idx);   % (nlon_crop x 1)

fprintf('  Cropped ERA5: %d lon x %d lat gridpoints\n', lon_count, lat_count);
fprintf('  Crop extent : lon [%.2f, %.2f]  lat [%.2f, %.2f]\n', ...
        min(lon_ERA5), max(lon_ERA5), min(lat_ERA5), max(lat_ERA5));

% ---------------------------------------------------------
%  6c. Find time indices matching acquisition hours (10:00-11:00 UTC)
%      then read ONLY the cropped spatial region at those times
%
%  ncread(file, var, start, count)
%    start = [lon_start, lat_start, t_start]   (1-based indices)
%    count = [lon_count, lat_count, t_count]
%    output shape: (nlon_crop, nlat_crop, t_count)
% ---------------------------------------------------------
date_era5 = double(time_era5) / (24*60*60) + datetime('1970-01-01');

t_idx1 = find(date_era5 >= datetime(t1.Year,t1.Month,t1.Day,9,0,0) & ...
              date_era5 <= datetime(t1.Year,t1.Month,t1.Day,11,0,0));
t_idx2 = find(date_era5 >= datetime(t2.Year,t2.Month,t2.Day,9,0,0) & ...
              date_era5 <=  datetime(t2.Year,t2.Month,t2.Day,11,0,0));

if isempty(t_idx1)
    error('No ERA5 timestamp found for t1 = %s (10:00-11:00 UTC).', char(t1));
end
if isempty(t_idx2)
    error('No ERA5 timestamp found for t2 = %s (10:00-11:00 UTC).', char(t2));
end

fprintf('  ERA5 time match: t1 -> %d epoch(s), t2 -> %d epoch(s)\n', ...
        numel(t_idx1), numel(t_idx2));

% Read cropped pressure [Pa]: shape (nlon_crop, nlat_crop, n_t)
sp_t1 = ncread(cfg.nc_era5, 'sp', ...
               [lon_start, lat_start, t_idx1(1)], ...
               [lon_count, lat_count, numel(t_idx1)]);

sp_t2 = ncread(cfg.nc_era5, 'sp', ...
               [lon_start, lat_start, t_idx2(1)], ...
               [lon_count, lat_count, numel(t_idx2)]);

% ---------------------------------------------------------
%  6d. Compute IBE
%
%  Pa -> hPa: * 0.01
%  delta_h_IBE = delta_P [hPa] * (-0.01) [m/hPa]
%  Physics: 1 hPa pressure rise -> ~1 cm sea surface drop
%  Sign check: P_t2 > P_t1 -> delta_P > 0 -> delta_h < 0 (surface drops) OK
% ---------------------------------------------------------
sp1_mean = mean(sp_t1, 3) * 0.01;   % (nlon_crop, nlat_crop) [hPa]
sp2_mean = mean(sp_t2, 3) * 0.01;   % (nlon_crop, nlat_crop) [hPa]

ibe_raw = (sp2_mean - sp1_mean) * (-0.01);   % [m], shape (nlon_crop, nlat_crop)
ibe_raw(~isfinite(ibe_raw)) = 0;

% ---------------------------------------------------------
%  6e. Interpolate IBE onto velocity grid
%
%  interp2 requires V of shape (nlat, nlon) -> must transpose ibe_raw
%  lat_ERA5 is decreasing (north->south); meshgrid preserves this order
%  -> rows of y_ERA5 also decrease -> no fliplr needed
% ---------------------------------------------------------
[x_ERA5, y_ERA5] = meshgrid(lon_ERA5, lat_ERA5);   % (nlat_crop, nlon_crop)
IBE = interp2(x_ERA5, y_ERA5, ibe_raw', lon, lat); % ibe_raw' -> (nlat_crop, nlon_crop)
IBE(~isfinite(IBE)) = 0;

fprintf('  IBE stats: mean=%.4f m | std=%.4f m | max|x|=%.4f m\n', ...
        mean(IBE(:),'omitnan'), std(IBE(:),'omitnan'), max(abs(IBE(:))));

% Diagnostic plot
figure('Name','IBE Diagnostic','Position',[100 100 1000 400]);
subplot(1,2,1); imagesc(lon_ERA5, lat_ERA5, ibe_raw');set(gca, 'YDir', 'normal');
axis image; colorbar; title('IBE on cropped ERA5 grid [m]');
xlabel('Longitude'); ylabel('Latitude');
subplot(1,2,2); imagesc(IBE);
axis image; colorbar; title('IBE interpolated to velocity grid [m]');
format_fig(gcf, cfg.out_vel, [name, '-IBE_diagnostic']);

R_era5 = georasterref( ...
    'RasterSize',      [lat_count, lon_count], ...
    'LatitudeLimits',  [min(lat_ERA5), max(lat_ERA5)], ...
    'LongitudeLimits', [min(lon_ERA5), max(lon_ERA5)]);

save_geotiff(cfg.out_vel, [name, '-IBE_RAW.tif'], ibe_raw, R_era5, 4326);
save_geotiff(cfg.out_vel, [name, '-IBE.tif'], IBE, R, cfg.epsg_code);

%% =========================================================
%  7. Flexure Mask
%% =========================================================
fprintf('=== Flexure Mask ===\n');

% % [thickness, bed, ground_mask] = load_bedmachine_data_region(lat, lon);

% % M = calculate_vaughan_flexure_mask(x, y, thickness, ground_mask, ...
% %                                    cfg.E, cfg.nu, cfg.rho_sw, cfg.g);
% % 
% % fprintf('  Grounded ice fraction: %.1f%%\n', ...
% %         100 * sum(ground_mask(:)==1) / numel(ground_mask));
% % 
% % figure('Name','BedMachine','Position',[100 100 1600 400]);
% % subplot(1,4,1); imagesc(range_data); colorbar; title('Range');
% % subplot(1,4,2); imagesc(thickness); colorbar; title('Ice Thickness');
% % subplot(1,4,3); imagesc(bed); colorbar; title('Bed Elevation');
% % subplot(1,4,4); imagesc(ground_mask); colorbar; title('Ground Mask');
% % format_fig(gcf, cfg.out_vel, [name, '-BedMachine']); 
% % 
% % figure('Name','Flexure Mask','Position',[100 100 1800 1600]);
% % subplot(2,2,1); imagesc(M); colorbar; title('Flexure Mask M', 'FontSize',16);

% new 2026.03.11
[thickness, bed, surface, ice_bottom] = ...
    load_bedmachine_data_region_v2(lat, lon);

ice_bottom = surface - thickness;
ice_bottom_dynamic = ice_bottom + Tidal_height + IBE;

A = double(ice_bottom_dynamic - bed);

Mask = zeros(size(A));
Mask(abs(A) > 1) = 1;   %float

ground = imfill(Mask,'holes');
ground(ground==0)=2;
ground(ground==1)=0;
ground_new = imfill(ground,'holes');
ground_new(ground_new==0)=1;
ground_new(ground_new==2)=0; % jiedi

dst2ground = double(bwdist(ground))*80;  % ground_new  to ground
rho_sw = 1027; 
gravity = 9.81; 
nu = 0.3; 
beta = (3*rho_sw * gravity*((1-nu^2)./(8.8e9.*thickness.^3))).^(1/4);
flextmp = 1 - exp(-beta.*dst2ground) .* (cos(beta.*dst2ground)+sin(beta.*dst2ground));
% flextmp(ground_new==1) = 0; 
% flextmp(isnan(flextmp))=1; 
flexure = filt2(flextmp,500,2*2000,'lp'); % lowpass filter to 4 km before interpolating to 100 km grid

M = flexure;

% [ground_mask_dynamic, grounded_ratio] = detect_dynamic_grounding_line(...
%     ice_bottom, bed, Tidal_height, IBE, 'verbose', true);
% 
% M = calculate_vaughan_flexure_mask(...
%     x, y, thickness, ground_mask_dynamic, cfg.E, cfg.nu, cfg.rho_sw, cfg.g);


figure('Name','BedMachine','Position',[100 100 1600 400]);
subplot(2,4,1); imagesc(ice_bottom); colorbar; title('ice_bottom');
subplot(2,4,2); imagesc(Mask); colorbar; title('Mask');
subplot(2,4,3); imagesc(bed); colorbar; title('Bed Elevation');
subplot(2,4,4); imagesc(ground); colorbar; title('ground');
subplot(2,4,5); imagesc(ice_bottom_dynamic); colorbar; title('ice bottom dynamic');
subplot(2,4,6); imagesc(A); colorbar; title('A');
subplot(2,4,7); imagesc(ground_new); colorbar; title('ground_new');
subplot(2,4,8); imagesc(dst2ground); colorbar; title('dst2ground');

format_fig(gcf, cfg.out_vel, [name, '-BedMachine']); 

figure('Name','Flexure Mask','Position',[100 100 1800 1600]);
subplot(2,2,1); imagesc(M); colorbar; title('Flexure Mask M', 'FontSize',16);
%% =========================================================
%  8. Combined Tidal + IBE Vertical Correction
%% =========================================================
fprintf('=== Applying Tidal + IBE Correction ===\n');

tidal_IBE_vertical = ((Tidal_height + IBE) .* M) ./ tan(inc);
tidal_IBE_vertical(~isfinite(tidal_IBE_vertical)) = 0;

fprintf('  Correction stats: mean=%.3f m | std=%.3f m | max|x|=%.3f m\n', ...
        mean(tidal_IBE_vertical(:),'omitnan'), ...
        std(tidal_IBE_vertical(:),'omitnan'), ...
        max(abs(tidal_IBE_vertical(:))));

if max(abs(tidal_IBE_vertical(:))) > cfg.IBE_warn_thr
    warning('Tidal+IBE correction exceeds %.0f m -- please verify inputs.', cfg.IBE_warn_thr);
end

subplot(2,2,2); imagesc(Tidal_height); colorbar;title('Tidal_height [m]','FontSize',16);
subplot(2,2,3); imagesc(IBE); colorbar;title('IBE [m]','FontSize',16);
subplot(2,2,4); imagesc(tidal_IBE_vertical); colorbar;title('Tidal+IBE Vertical [m]','FontSize',16);

format_fig(gcf, cfg.out_vel, [name, '-FlexureMask']);
%% =========================================================
%  9. Compute Velocity (Vgr, Vaz)
%% =========================================================
fprintf('=== Computing Velocity ===\n');

switch cfg.unit
    case 'year';  time_factor = year_gap;
    case 'day';   time_factor = day_gap;
    otherwise;    error('Unknown unit "%s". Use ''year'' or ''day''.', cfg.unit);
end

Vgr = (range_data + tidal_IBE_vertical) / time_factor;
Vaz =  azimuth_data                      / time_factor;

save_geotiff(cfg.out_vel, [name, '-range.tif'], (range_data + tidal_IBE_vertical), R, cfg.epsg_code);
save_geotiff(cfg.out_vel, [name, '-azimuth.tif'], azimuth_data, R, cfg.epsg_code);
%% =========================================================
%  10. Save Corrected Velocity GeoTIFFs
%% =========================================================
save_geotiff(cfg.out_vel, [name, '-Vgr.tif'], Vgr, R, cfg.epsg_code);
save_geotiff(cfg.out_vel, [name, '-Vaz.tif'], Vaz, R, cfg.epsg_code);

%% =========================================================
%  11. Decompose to Cartesian Velocity (Vx, Vy, V)
%% =========================================================
fprintf('=== Velocity Decomposition ===\n');

u = Vgr .* cos(lv_phi) - Vaz .* sin(lv_phi);
v = Vgr .* sin(lv_phi) + Vaz .* cos(lv_phi);

[Vx, Vy] = uv2vxvy(x, y, u, v);
V  = hypot(Vx, Vy);

figure('Name','Velocity Components','Position',[100 100 1500 900]);
subplot(2,3,1); imagesc(Vx); colorbar; title('Vx');
subplot(2,3,2); imagesc(Vy); colorbar; title('Vy');
subplot(2,3,3); imagesc(V);  colorbar; title('Speed |V|');
subplot(2,3,4); plot_enhanced_histogram2(Vx, 'Vx', [min(Vx(:)), max(Vx(:))]);
subplot(2,3,5); plot_enhanced_histogram2(Vy, 'Vy', [min(Vy(:)), max(Vy(:))]);
subplot(2,3,6); plot_enhanced_histogram2(V,  '|V|',[min(V(:)),  max(V(:))]);
format_fig(gcf, cfg.out_vxvy, [name, '-Velocity']);
%% =========================================================
%  12. Save Vx / Vy / V GeoTIFFs
%% =========================================================
save_geotiff(cfg.out_vxvy, [name, '-Vx.tif'], Vx, R, cfg.epsg_code);
save_geotiff(cfg.out_vxvy, [name, '-Vy.tif'], Vy, R, cfg.epsg_code);
save_geotiff(cfg.out_vxvy, [name, '-V.tif'],  V,  R, cfg.epsg_code);

fprintf('=== Done ===\n');

%% =========================================================
%  Helper Function
%% =========================================================
function save_geotiff(out_dir, filename, data, R, epsg)
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
        fprintf('  Created directory: %s\n', out_dir);
    end
    geotiffwrite(fullfile(out_dir, filename), data, R, 'CoordRefSysCode', epsg);
    fprintf('  Saved: %s\n', fullfile(out_dir, filename));
end