function [hice,hbed,hwater] = bedmap3_profile(lati_or_xi,loni_or_yi,varargin)
% bedmap3_profile plots a 2D profile of ice, water, and rock elevations
% along any path in Antarctica. The data are from Bedmap3.
% 
%% Requirements 
% You'll need two things! 
% 
% 1. The bedmap3.nc file, found here: https://doi.org/10.5285/2d0e4791-8e20-46a3-80e4-f5f6716025d2
% 2. Antarctic Mapping Tools: https://github.com/chadagreene/Antarctic-Mapping-Tools
% 
%% Syntax
% 
%  bedmap3_profile(lati,loni)
%  bedmap3_profile(xi,yi)
%  bedmap3_profile(...,'horiz',HorizontalAxisValues)
%  bedmap3_profile(...,PatchProperty,PatchValue)
%  [hice,hbed,hwater] = bedmap3_profile(...)
% 
%% Description
% 
% bedmap3_profile(lati,loni) plots a side-view profile along a path given by
% geo coordinates lat,lon. lat and lon must be 1D arrays of equal length. If only two
% points are entered, an equally-spaced 1000-point line is created between those points. 
% 
% bedmap3_profile(xi,yi) as above, but for polar stereographic meters xi,yi
% in the polar stereographic projection. 
%
% bedmap3_profile(...,'horiz',HorizontalAxisValues) specifies horizontal axis
% values where HorizontalAxisValues is a 1D monotonically-increasing or decreasing
% array of dimensions corresponding to lat and lon. By default,
% HorizontalAxisValues are calculated as pathdistps. If you prefer to
% plot profiles with respect to some other values such as latitude of a north/south
% transect, use bedmap3_profile(lat,lon,'horiz',lat). 
%
% bedmap3_profile(...,PatchProperty,PatchValue) specifies edge line width, face
% color, and edge color of ice, water, or bed. The following properties may
% be specified: 
% 
%     * 'IceFace',ColorSpec
%     * 'IceEdge',ColorSpec
%     * 'IceEdgeWidth',LineWidth
%     * 'WaterFace',ColorSpec
%     * 'WaterEdge',ColorSpec
%     * 'WaterEdgeWidth',LineWidth
%     * 'BedFace',ColorSpec
%     * 'BedEdge',ColorSpec
%     * 'BedEdgeWidth',LineWidth
%     * 'Sky',ColorSpec
% 
% [hice,hbed,hwater] = bedmap3_profile(...) returns handles of ice, bed,
% and water patch objects. 
% 
%% Example 
% A crude profile of Pine Island Glacier: 
% 
%  lat = [-76.31 -74.81]; 
%  lon = [-91.95 -102.26];
% 
%  figure
%  bedmap3_profile(lat,lon);
%
%% Citations
% If you use BedMachine data, please cite the Morlighem paper listed below. 
% And if this function is useful for you, please do me a kindness and cite 
% my Antarctic Mapping Tools paper. 
%  
% Morlighem, M., E. Rignot, T. Binder, D. D. Blankenship, R. Drews, G. Eagles, 
% O. Eisen, F. Ferraccioli, R. Forsberg, P. Fretwell, V. Goel, J. S. Greenbaum,
% H. Gudmundsson, J. Guo, V. Helm, C. Hofstede, I. Howat, A. Humbert, W. Jokat,
% N. B. Karlsson, W. Lee, K. Matsuoka, R. Millan, J. Mouginot, J. Paden, F. Pattyn,
% J. L. Roberts, S. Rosier, A. Ruppel, H. Seroussi, E. C. Smith, D. Steinhage, 
% B. Sun, M. R. van den Broeke, T. van Ommen, M. van Wessem, and D. A. Young. 2019. 
% Deep glacial troughs and stabilizing ridges unveiled beneath the margins of the
% Antarctic ice sheet, Nature Geoscience. https://doi.org/10.1038/s41561-019-0510-8
% 
% Greene, C. A., Gwyther, D. E., & Blankenship, D. D. Antarctic Mapping Tools  
% for Matlab. Computers & Geosciences. 104 (2017) pp.151-157. 
% http://dx.doi.org/10.1016/j.cageo.2016.08.003
% 
%% Citations
% If you use Bedmap3 data, please cite the Pritchard paper listed below. 
% And if this function is useful for you, please do me a kindness and cite 
% my Antarctic Mapping Tools paper. 
% 
% Pritchard, H.D., Fretwell, P.T., Fremand, A.C. et al. Bedmap3 updated ice bed, 
% surface and thickness gridded datasets for Antarctica. Sci Data 12, 414 (2025). 
% https://doi.org/10.1038/s41597-025-04672-y
% 
% Greene, C. A., Gwyther, D. E., & Blankenship, D. D. Antarctic Mapping Tools  
% for Matlab. Computers & Geosciences. 104 (2017) pp.151-157. 
% http://dx.doi.org/10.1016/j.cageo.2016.08.003
% 
%% Author Info
% This function was adapted from bedmachine_profile by Chad A. Greene, April 2025. 
% 
% See also bedmap3_interp and bedmap3_data. 

%% Error checks: 

assert(isvector(lati_or_xi),'Profile plotting requires that input coordinates in the form of a vector.'); 
assert(isvector(loni_or_yi),'Profile plotting requires that input coordinates in the form of a vector.'); 
assert(~isscalar(lati_or_xi),'You must enter more than one point for a profile.') 
assert(numel(lati_or_xi)==numel(loni_or_yi),'Input lat vector must be the same size as input lon.');

%% Set defaults: 

distax = true; % km on X axis

% Colors: 
iceface = [0.8431    1.0000    0.9961]; 
iceedge = [0.5451, 0, 0];  % red

waterface = [0, 0.4314, 0.7255]; 
wateredge = [0.4902, 0.1490, 0.8039]; % purple

bedface = [0.7373, 0.6431, 0.5412];    
bededge = [1, 0.9569, 0.3098]; %[0.1137 0.0078      0];  yellow
sky = 'w'; 

% Line widths:
iceedgewidth = 1; 
wateredgewidth = 1; 
bededgewidth = 1; 

%% Parse user inputs: 

% Are inputs georeferenced coordinates or polar stereographic?
if islatlon(lati_or_xi,loni_or_yi)
   [xi,yi] = ll2ps(lati_or_xi,loni_or_yi); % The ll2ps function is in the Antarctic Mapping Tools package.
else
   xi = lati_or_xi;
   yi = loni_or_yi;    
end
    
if nargin>2
    
    % Horizontal Axis Values: 
    tmp = strncmpi(varargin,'horizontalaxis',3); 
    if any(tmp)
        horizax = varargin{find(tmp)+1}; 
        distax = false; % don't use track distance as the horizontal axis. 
        
        % Columnate horizontal axis to ensure consistent behavior: 
        horizax = horizax(:); 
        assert(isnumeric(horizax)==1,'horizontal axis values must be numeric.')
        assert(numel(horizax)==numel(lati_or_xi),'If you enter values for a horizontal axis, they must correspond to input lat,lon values. It looks like you have entered a horizontal axis array that does not match the size of lat and lon.') 
    end
    
    % All other inputs are colors or Line widths:
    tmp = strcmpi(varargin,'iceface');
    if any(tmp)
        iceface = varargin{find(tmp)+1}; 
    end
    
    tmp = strcmpi(varargin,'bedface');
    if any(tmp)
        bedface = varargin{find(tmp)+1}; 
    end
    
    tmp = strcmpi(varargin,'waterface');
    if any(tmp)
        waterface = varargin{find(tmp)+1}; 
    end
    
    tmp = strcmpi(varargin,'iceedge');
    if any(tmp)
        iceedge = varargin{find(tmp)+1}; 
    end
    
    tmp = strcmpi(varargin,'bededge');
    if any(tmp)
        bededge = varargin{find(tmp)+1}; 
    end
    
    tmp = strcmpi(varargin,'wateredge');
    if any(tmp)
        wateredge = varargin{find(tmp)+1}; 
    end
    
    tmp = strncmpi(varargin,'iceedgewidth',8);
    if any(tmp)
        iceedgewidth = varargin{find(tmp)+1}; 
        assert(isscalar(iceedgewidth),'iceedgewidth value must be scalar.')
    end
    
    tmp = strncmpi(varargin,'bededgewidth',8);
    if any(tmp)
        bededgewidth = varargin{find(tmp)+1}; 
        assert(isscalar(bededgewidth),'iceedgewidth value must be scalar.')
    end
    
    tmp = strncmpi(varargin,'wateredgewidth',10);
    if any(tmp)
        wateredgewidth = varargin{find(tmp)+1}; 
        assert(isscalar(wateredgewidth),'iceedgewidth value must be scalar.')
    end
    
    tmp = strcmpi(varargin,'sky');
    if any(tmp)
        sky = varargin{find(tmp)+1}; 
    end
end

%% Reformat coordinates as needed:  

% Get rid of nans: 
isf = isfinite(xi) & isfinite(yi); 
xi = xi(isf); 
yi = yi(isf); 

% If only two input points, turn them into a 1000 point line: 
if numel(xi)==2
    xi = linspace(xi(1),xi(2),1000); 
    yi = linspace(yi(1),yi(2),1000); 
end

% Columnate for consistency: 
xi = xi(:); 
yi = yi(:); 

if distax

  horizax = pathdistps(xi,yi,'km'); 
   
else
   horizax = horizax(isf); % ensures same elements as input coordinates 
end
   

%% Get data

% Get bedmap3 data: 
sfz = bedmap3_interp('surface',xi,yi); 
bed = bedmap3_interp('bed',xi,yi); 
thck = bedmap3_interp('thickness',xi,yi); 

sfz(isnan(sfz))=0; 
thck(isnan(thck)) = 0; 
icebottom = sfz-thck;

% Indices of non-ocean data: 
% nreal = isfinite(icebottom);

% Some vertical limits: 
maxsfz = max(sfz(isfinite(sfz))); 
if isempty(maxsfz)
    maxsfz = 0; 
end
minbed = min(bed(isfinite(bed))); 
padding = (maxsfz-minbed)/20; 

%% Generate plot 

% Draw water:
hwater=fill([horizax(1);horizax(end);horizax(end);horizax(1)],[0;0;minbed-padding;minbed-padding],waterface);
set(hwater,'edgecolor',wateredge,'linewidth',wateredgewidth)
hold on;

% Draw bed:
realbed = find(isfinite(bed)); % 结果为索引值
hbed = fill([horizax(realbed);horizax(realbed(length(realbed)));horizax(realbed(1))],[bed(realbed);minbed-padding;minbed-padding],bedface);
set(hbed,'edgecolor',bededge,'linewidth',bededgewidth)

% Draw ice:
hice = patch([horizax;flipud(horizax)],[sfz;flipud(icebottom)],iceface); 
set(hice,'edgecolor',iceedge,'linewidth',iceedgewidth);

% Format axes:
axis tight; 
box off;
   ylabel('elevation (m)')

% Only label x axis if it's a distance axis: 
if distax
    xlabel('distance along profile (km)')
end

% Sky color
set(gca,'color',sky)
set(gcf,'color',sky)

%% Clean Up

if nargout==0
    clear hice
end
