function gamma_pot_to_velocity(range_path, azimuth_path, geom_dir, varargin)
% GAMMA_POT_TO_VELOCITY  Convert GAMMA POT range/azimuth offsets to
%   map-projected velocity components (V, Vx, Vy).
%
%   Handles both single-pair and batch (directory) mode automatically.
%
% USAGE
%   % Single pair — full file paths
%   gamma_pot_to_velocity(range_tif, azimuth_tif, geom_dir)
%
%   % Batch mode — pass directories + suffixes; all matching pairs processed
%   gamma_pot_to_velocity(range_dir, azi_dir, geom_dir, ...
%       'range_suffix', '.range2', 'azi_suffix', '.azi')
%
% REQUIRED INPUTS
%   range_path   – full path to a range TIF  OR  directory containing range TIFs
%   azimuth_path – full path to an azimuth TIF  OR  directory containing azimuth TIFs
%   geom_dir     – directory containing  <date>.lv_phi.tif  [and <date>.inc.tif]
%                  Geometry is matched by date1 of each pair; falls back to
%                  any *.lv_phi.tif in the directory.
%
% OPTIONAL NAME-VALUE PAIRS
%   'range_suffix'  – file suffix after 'date1-date2', e.g. '.range2',
%                     '.range-100m', '.range-nmt_filtered'
%                     Required when range_path is a directory.
%   'azi_suffix'    – same for azimuth (default: auto-derived from range_suffix
%                     by replacing 'range' with 'azi')
%   'output_dir'    – output folder           (default: './velocity')
%   'unit'          – 'year' or 'day'         (default: 'year')
%   'epsg'          – output EPSG code        (default: 3031)
%   'resolution'    – output pixel size (m);  [] = native  (default: [])
%   'interp_method' – 'linear','nearest','cubic','spline'   (default: 'linear')
%   'lv_phi_in_deg' – true if lv_phi is in degrees (auto-detected) (default: false)
%   'lv_phi_fallback'– explicit lv_phi file to use when no date-matched file exists;
%                      may be a full path or a bare filename inside geom_dir
%                      e.g. '20220308.lv_phi.tif'  (default: '', auto-scan)
%
% SUPPORTED FILENAME PATTERNS
%   date1-date2.range.tif        date1-date2.azi.tif
%   date1-date2.range2.tif       date1-date2.azi2.tif
%   date1-date2.range-100m.tif   date1-date2.azi-100m.tif
%   date1-date2.range-nmt_filtered.tif  ...  (any suffix)
%
% OUTPUTS  (written to output_dir)
%   <date1>_<date2>_V.tif    speed magnitude
%   <date1>_<date2>_Vx.tif   East  component
%   <date1>_<date2>_Vy.tif   North component
%
% EXAMPLES
%   % Single pair
%   gamma_pot_to_velocity( ...
%       './range/20251122-20251128.range2.tif', ...
%       './azimuth/20251122-20251128.azi2.tif', ...
%       './geometry');
%
%   % Batch – process every pair in the directories
%   gamma_pot_to_velocity( ...
%       './input/Mertz_path54/range', ...
%       './input/Mertz_path54/azimuth', ...
%       './input/Mertz_path54/geometry', ...
%       'range_suffix', '.range2', ...
%       'azi_suffix',   '.azi',    ...
%       'output_dir',   './velocity', ...
%       'resolution',   120);

% ── parse options ─────────────────────────────────────────────────────────
p = inputParser;
addRequired(p,  'range_path',    @(x) ischar(x)||isstring(x));
addRequired(p,  'azimuth_path',  @(x) ischar(x)||isstring(x));
addRequired(p,  'geom_dir',      @(x) ischar(x)||isstring(x));
addParameter(p, 'range_suffix',  '',          @(x) ischar(x)||isstring(x));
addParameter(p, 'azi_suffix',    '',          @(x) ischar(x)||isstring(x));
addParameter(p, 'output_dir',    './velocity',@(x) ischar(x)||isstring(x));
addParameter(p, 'unit',          'year',      @(x) ismember(x,{'year','day'}));
addParameter(p, 'epsg',          3031,        @isnumeric);
addParameter(p, 'resolution',    [],          @(x) isempty(x)||isnumeric(x));
addParameter(p, 'interp_method', 'linear',    @ischar);
addParameter(p, 'lv_phi_in_deg', false,       @islogical);
addParameter(p, 'lv_phi_fallback', '',        @(x) ischar(x)||isstring(x));
parse(p, range_path, azimuth_path, geom_dir, varargin{:});
opt = p.Results;

range_path   = char(opt.range_path);
azimuth_path = char(opt.azimuth_path);
geom_dir     = char(opt.geom_dir);

% Normalise suffixes (add leading dot if missing)
rsuf = normalise_suffix(char(opt.range_suffix));
asuf = char(opt.azi_suffix);
if isempty(asuf) && ~isempty(rsuf)
    % Auto-derive azi suffix: replace 'range' with 'azi' in the range suffix
    asuf = regexprep(rsuf, 'range', 'azi');
    fprintf('[gamma_pot_to_velocity] azi_suffix auto-set to "%s"\n', asuf);
end
asuf = normalise_suffix(asuf);

% ── discover file list ────────────────────────────────────────────────────
range_files = collect_files(range_path, rsuf, 'range');
azi_files   = collect_files(azimuth_path, asuf, 'azimuth');

% Match range and azimuth files by date pair
[range_files, azi_files] = match_pairs(range_files, azi_files);

n = numel(range_files);
fprintf('[gamma_pot_to_velocity] %d pair(s) to process\n\n', n);

if ~exist(opt.output_dir, 'dir'); mkdir(opt.output_dir); end

% ── loop over all pairs ───────────────────────────────────────────────────
for k = 1:n
    fprintf('--- [%d/%d] %s\n', k, n, range_files{k});
    try
        process_one_pair(range_files{k}, azi_files{k}, geom_dir, opt);
    catch ME
        fprintf('  ERROR: %s\n', ME.message);
    end
    fprintf('\n');
end

fprintf('[gamma_pot_to_velocity] All done -> %s\n', opt.output_dir);
end


% =========================================================================
%  SINGLE-PAIR PROCESSING
% =========================================================================

function process_one_pair(range_file, azi_file, geom_dir, opt)

% ── dates & time span ─────────────────────────────────────────────────────
[date1_str, date2_str] = parse_dates(range_file);
t1 = datetime(date1_str, 'InputFormat','yyyyMMdd');
t2 = datetime(date2_str, 'InputFormat','yyyyMMdd');
day_gap  = days(t2 - t1);
year_gap = years(t2 - t1);
fprintf('  Dates : %s -> %s  (%.0f days)\n', date1_str, date2_str, day_gap);

% ── read offsets ──────────────────────────────────────────────────────────
[range_data, R_ref] = readgeoraster(range_file);
range_data = apply_nodata(double(range_data));

[az_data, ~] = readgeoraster(azi_file);
az_data = apply_nodata(double(az_data));

if ~isequal(size(range_data), size(az_data))
    error('Range and azimuth grids have different dimensions (%s vs %s).', ...
        mat2str(size(range_data)), mat2str(size(az_data)));
end

% ── geometry ──────────────────────────────────────────────────────────────
lv_phi_path = find_geom_file(geom_dir, date1_str, 'lv_phi', char(opt.lv_phi_fallback));
fprintf('  lv_phi: %s\n', lv_phi_path);

[lv_phi_raw, R_phi] = readgeoraster(lv_phi_path);
lv_phi_raw = apply_nodata(double(lv_phi_raw));

if opt.lv_phi_in_deg || max(abs(lv_phi_raw(:)),[],'omitnan') > 2*pi
    fprintf('  lv_phi in degrees -> converting to radians\n');
    lv_phi_raw = deg2rad(lv_phi_raw);
end

lv_phi = resample_to_grid(lv_phi_raw, R_phi, R_ref, ...
                           size(range_data), opt.interp_method);

% ── optional resolution change ────────────────────────────────────────────
if ~isempty(opt.resolution) && ...
        abs(opt.resolution - R_ref.SampleSpacingInWorldX) > 1e-3
    fprintf('  Resampling to %.0f m ...\n', opt.resolution);
    [range_data, R_out] = resample_raster(range_data, R_ref, opt.resolution, opt.interp_method);
    az_data = resample_raster(az_data, R_ref, opt.resolution, opt.interp_method);
    lv_phi  = resample_raster(lv_phi,  R_ref, opt.resolution, opt.interp_method);
else
    R_out = R_ref;
end

% ── displacement -> velocity ──────────────────────────────────────────────
if strcmp(opt.unit, 'year')
    scale = year_gap;
else
    scale = day_gap;
end

Vgr = range_data / scale;
Vaz = az_data    / scale;

% East / North decomposition
Vx = Vgr .* cos(lv_phi) - Vaz .* sin(lv_phi);
Vy = Vgr .* sin(lv_phi) + Vaz .* cos(lv_phi);
V  = hypot(Vx, Vy);

fprintf('  Speed (m/%s): mean=%.1f  max=%.1f\n', opt.unit, ...
    mean(V(:),'omitnan'), max(V(:),[],'omitnan'));

% ── save ──────────────────────────────────────────────────────────────────
prefix = sprintf('%s_%s', date1_str, date2_str);
save_tif(fullfile(opt.output_dir, [prefix '_V.tif']),  single(V),  R_out, opt.epsg);
save_tif(fullfile(opt.output_dir, [prefix '_Vx.tif']), single(Vx), R_out, opt.epsg);
save_tif(fullfile(opt.output_dir, [prefix '_Vy.tif']), single(Vy), R_out, opt.epsg);
end


% =========================================================================
%  LOCAL HELPERS
% =========================================================================

function suf = normalise_suffix(suf)
% Ensure suffix starts with '.' unless it is empty.
if ~isempty(suf) && suf(1) ~= '.'
    suf = ['.' suf];
end
end


function files = collect_files(path_in, suffix, label)
% Return a sorted cell array of full file paths.
%
%  path_in is a FILE  -> return {path_in} (suffix ignored if already a file)
%  path_in is a DIR   -> scan for files matching  *<suffix>.tif
%                        suffix must be non-empty in this case

if isfile(path_in)
    % Single explicit file
    files = {path_in};
    return;
end

% Resolve directory: path_in itself or its parent
if isfolder(path_in)
    search_dir = path_in;
elseif isfolder(fileparts(path_in))
    search_dir = fileparts(path_in);
else
    error('"%s" is neither an existing file nor a folder.', path_in);
end

if isempty(suffix)
    error(['%s path "%s" is a directory but no suffix was provided.\n' ...
           'Use ''range_suffix'' / ''azi_suffix'' to specify the file type.'], ...
          label, path_in);
end

hits = dir(fullfile(search_dir, ['*' suffix '.tif']));
% Keep only files whose name starts with yyyyMMdd-yyyyMMdd
valid = arrayfun(@(h) ~isempty(regexp(h.name,'^(\d{8})-(\d{8})\.','once')), hits);
hits  = hits(valid);

if isempty(hits)
    error('No files matching "*%s.tif" with date-pair prefix found in:\n  %s', ...
          suffix, search_dir);
end

files = sort(fullfile({hits.folder}, {hits.name})');
fprintf('[gamma_pot_to_velocity] Found %d %s file(s) in %s\n', ...
    numel(files), label, search_dir);
end


function [rf, af] = match_pairs(range_files, azi_files)
% Match range and azimuth files by their date-pair prefix.
% Warns about unmatched files.

get_prefix = @(f) regexp(f, '(\d{8}-\d{8})(?=\.)', 'match', 'once');

r_pfx = cellfun(get_prefix, range_files, 'UniformOutput', false);
a_pfx = cellfun(get_prefix, azi_files,   'UniformOutput', false);

common = intersect(r_pfx, a_pfx);
if isempty(common)
    error('No matching date pairs between range and azimuth directories.');
end

only_r = setdiff(r_pfx, a_pfx);
only_a = setdiff(a_pfx, r_pfx);
if ~isempty(only_r)
    fprintf('  Warning: range pairs with no matching azimuth: %s\n', ...
        strjoin(only_r, ', '));
end
if ~isempty(only_a)
    fprintf('  Warning: azimuth pairs with no matching range: %s\n', ...
        strjoin(only_a, ', '));
end

rf = range_files(ismember(r_pfx, common));
af = azi_files  (ismember(a_pfx, common));
end


function [d1, d2] = parse_dates(filepath)
[~, fname, ~] = fileparts(filepath);
tok = regexp(fname, '^(\d{8})-(\d{8})\.', 'tokens', 'once');
if isempty(tok)
    error('Cannot parse dates from filename: "%s"', fname);
end
d1 = tok{1};  d2 = tok{2};
end


function fpath = find_geom_file(geom_dir, date1_str, field_name, fallback_path)
% fallback_path (optional): explicit file to use when no date-matched file exists.
%   May be a full path OR a bare filename (resolved relative to geom_dir).
if nargin < 4; fallback_path = ''; end

exact = fullfile(geom_dir, sprintf('%s.%s.tif', date1_str, field_name));
if isfile(exact)
    fpath = exact;  return;
end

% ── user-supplied fallback ────────────────────────────────────────────────
if ~isempty(fallback_path)
    % Accept bare filename: resolve against geom_dir
    if ~isfile(fallback_path)
        candidate = fullfile(geom_dir, fallback_path);
        if isfile(candidate)
            fallback_path = candidate;
        else
            error('lv_phi_fallback "%s" not found (tried as absolute path and inside geom_dir).', ...
                  fallback_path);
        end
    end
    fprintf('  [geom] No %s for date %s, using lv_phi_fallback: %s\n', ...
        field_name, date1_str, fallback_path);
    fpath = fallback_path;  return;
end

% ── auto-scan fallback (original behaviour) ───────────────────────────────
hits = dir(fullfile(geom_dir, sprintf('*.%s.tif', field_name)));
if isempty(hits)
    if strcmp(field_name, 'lv_phi')
        error('No *.%s.tif file found in: %s', field_name, geom_dir);
    else
        fpath = '';  return;
    end
end
fpath = fullfile(hits(1).folder, hits(1).name);
fprintf('  [geom] No %s for date %s, using fallback: %s\n', ...
    field_name, date1_str, hits(1).name);
end


function out = resample_to_grid(data, R_src, R_dst, target_size, method)
if isequal(size(data), target_size) && ...
        abs(R_src.SampleSpacingInWorldX - R_dst.SampleSpacingInWorldX) < 1e-3
    out = data;  return;
end
[Xq, Yq] = meshgrid(px_cx(R_dst, target_size(2)), ...
                     px_cy(R_dst, target_size(1)));
out = interp2(px_cx(R_src,size(data,2)), px_cy(R_src,size(data,1)), ...
              data, Xq, Yq, method);
end


function [data_out, R_out] = resample_raster(data, R_in, res_out, method)
xl = R_in.XWorldLimits;  yl = R_in.YWorldLimits;
W  = max(1, round((xl(2)-xl(1))/res_out));
H  = max(1, round((yl(2)-yl(1))/res_out));
Xd = linspace(xl(1)+res_out/2, xl(2)-res_out/2, W);
Yd = linspace(yl(2)-res_out/2, yl(1)+res_out/2, H);
[Xq,Yq] = meshgrid(Xd, Yd);
data_out = interp2(px_cx(R_in,size(data,2)), px_cy(R_in,size(data,1)), ...
                   data, Xq, Yq, method);
R_out = maprefcells(xl, yl, H, W, 'ColumnsStartFrom','north');
try; R_out.ProjectedCRS = R_in.ProjectedCRS; catch; end
end


function xc = px_cx(R, n)
xc = reshape(linspace(R.XWorldLimits(1)+R.SampleSpacingInWorldX/2, ...
                       R.XWorldLimits(2)-R.SampleSpacingInWorldX/2, n), 1, []);
end

function yc = px_cy(R, n)
yc = reshape(linspace(R.YWorldLimits(2)-R.SampleSpacingInWorldY/2, ...
                       R.YWorldLimits(1)+R.SampleSpacingInWorldY/2, n), [], 1);
end

function data = apply_nodata(data)
data(data == 0)       = NaN;
data(data == -9999)   = NaN;
data(~isfinite(data)) = NaN;
end

function save_tif(filepath, data, R, epsg)
geotiffwrite(filepath, data, R, 'CoordRefSysCode', epsg);
fprintf('    -> %s\n', filepath);
end
