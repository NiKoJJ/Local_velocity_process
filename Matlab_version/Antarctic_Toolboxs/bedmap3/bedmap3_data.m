function [Z,x,y] = bedmap3_data(variable,lati_or_xi,loni_or_yi,buffer_km) 
% bedmap3_data loads gridded data from Bedmap3.
% 
%% Requirements 
% You'll need two things! 
% 
% 1. The bedmap3.nc file, found here: https://doi.org/10.5285/2d0e4791-8e20-46a3-80e4-f5f6716025d2
% 2. Antarctic Mapping Tools: https://github.com/chadagreene/Antarctic-Mapping-Tools
% 
%% Syntax 
% 
%  Z = bedmap3_data(variable)
%  Z = bedmap3_data(variable,lati,loni)
%  Z = bedmap3_data(variable,xi,yi) 
%  Z = bedmap3_data(...,buffer_km)
%  [Z,x,y] = bedmap3_data(...)
% 
%% Description 
% 
% Z = bedmap3_data(variable) loads a specified variable for the whole ice 
% sheet. The variable can be: 
%    * 'surface'         meters relative to gl04c geoid. 
%    * 'bed'             meters relative to gl04c geoid.
%    * 'bed error'       meters
%    * 'thickness'       meters total thickness (including firn)
%    * 'thickness error' meters
%    * 'mask'            0 = ocean, 1 = grounded ice, 2 = transiently grounded ice shelf, 3 = floating ice shef, 4 = rock
%    * 'count'           Number of survey data points used to calculate ice thickness.
%    * 'ice'             true for all ice grid cells
%    * 'rock'            true for all rock grid cells
%    * 'base'            meters base of the ice sheet (bottom of ice shelves, but same as bed over grounded ice.) 
%    * 'wct'             meters water column thickness (derived, not an official Bedmap3 product.) 
%    * 'head'            meters freshwater equivalent, static pressure head (derived, not an official Bedmap3 product.)  
%
% Z = bedmap3_data(variable,lati,loni) returns only enough Bedmap3 data to fully
% encompass a set of points given by geo coordinates lati,loni. This is a
% good way to save computer memory, only loading, analyzing, and plotting the
% data you need to work in a region of interest. 
% 
% Z = bedmap3_data(variable,xi,yi) As above, but for polar stereographic coordinates
% xi, yi in meters (ps71 for Antarctica). The function automatically 
% determines whether input coordinates are geo or polar stereographic via the 
% islatlon function. 
% 
% Z = bedmap3_data(...,buffer_km) as above, but adds a buffer around the input
% coordinates. This option is useful for loading only the data in your region
% of interest, plus a little extra around the sides for good measure. 
% 
% [Z,x,y] = bedmap3_data(...) returns x and y coordinates in ps71 meters. 
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
% This function was written by Chad A. Greene, April 2025. 
% 
% See also bedmap3_interp and bedmap3_profile. 

%% Initial error checks: 

narginchk(1,Inf)
assert(~isnumeric(variable),'Error: variable must be a string, e.g. ''bed'', ''surface'', ''mask'', etc') 

%% Set defaults: 

subset    = false; % use whole data set (not a regional subset) by default 
rho_ice   = 917;   % ice density for thickness above flotation and head calculation
rho_fw    = 1000;  % freshwater density for head calculation 

if nargin>3
     assert(numel(buffer_km)<3,'Error: buffer must be one or two elements, in kilometers.') 
else 
    buffer_km = 0; 
end

%% Parse inputs: 

variable = lower(variable); 
switch variable 
    case {'thick','thickness','ice thickness'}
        variable = 'ice_thickness'; 
    case {'surf','surface','surface topography'}
        variable = 'surface_topography'; 
    case {'count'}
        variable = 'thickness_survey_count'; 
    case {'bed','bed topography'}
        variable = 'bed_topography'; 
    case {'bed uncertainty','bed error','bed_error'}
        variable = 'bed_uncertainty'; 
    case {'thickness uncertainty','thickness error'}
        variable = 'thickness_uncertainty'; 
    case {'bed_topography','bed_uncertainty','ice_thickness','mask','surface_topography','thickness_survey_count','thickness_uncertainty'}
        % do nothing bc these are the official variable names 
    case {'ground','grounded','rock','wct','base','head','ice'}
        % do nothing bc these are derived products
    otherwise
        error(['Unrecognized variable ',variable,'.'])
end

if nargin>1
   
    subset = true; 

    % Are inputs georeferenced coordinates or polar stereographic?
    if islatlon(lati_or_xi,loni_or_yi)
        [xi,yi] = ll2ps(lati_or_xi,loni_or_yi); % The ll2ps function is in the Antarctic Mapping Tools package.
    else 
        xi = lati_or_xi;
        yi = loni_or_yi;    
    end


end

%% Define filename: 

filename = 'bedmap3.nc';
      
assert(exist(filename,'file')==2,['Error: cannot find ',filename,'. If you have already downloaded the data, make sure Matlab can find it. If you do not have the data, get it here: https://nsidc.org/data/idbmg4']) 

%% Load data 

x = ncread(filename,'x'); 
y = ncread(filename,'y'); 

if subset
   
   if isscalar(buffer_km)
      buffer_km = [buffer_km buffer_km]; 
   end
   
    % A crude manual fix for when a single xi,yi lies between pixels: 
    if isscalar(xi)
          buffer_km = [max([buffer_km(1) 1]) max([buffer_km(2) 1])]; 
    end
    
    % Get xlimits (xl) and ylimits (yl) of input coordinates + buffer:
    xl = [min(xi(:))-buffer_km(1)*1000 max(xi(:))+buffer_km(1)*1000];
    yl = [min(yi(:))-buffer_km(2)*1000 max(yi(:))+buffer_km(2)*1000];
    
    % Region of rows and columns of pixels to read: 
    ci=find((y>=yl(1))&(y<=yl(2)));
    ri=find((x>=xl(1))&(x<=xl(2)));

    x = x(ri); 
    y = y(ci); 

else
    ci = 1:length(y); 
    ri = 1:length(x); 
end

% Load data: 
switch lower(variable)
   case 'base'
      th = ncread(filename,'ice_thickness',[ri(1) ci(1)],[length(ri) length(ci)]);
      sfz = ncread(filename,'surface_topography',[ri(1) ci(1)],[length(ri) length(ci)]);
      Z = sfz-th; 
      
   case 'wct'
      bed = ncread(filename,'bed_topography',[ri(1) ci(1)],[length(ri) length(ci)]);
      th = ncread(filename,'ice_thickness',[ri(1) ci(1)],[length(ri) length(ci)]);
      sfz = ncread(filename,'surface_topography',[ri(1) ci(1)],[length(ri) length(ci)]);
      Z = sfz-th-bed; 
      
   case 'head' 
      bed = ncread(filename,'bed_topography',[ri(1) ci(1)],[length(ri) length(ci)]);
      th = ncread(filename,'ice_thickness',[ri(1) ci(1)],[length(ri) length(ci)]);
      mask = ncread(filename,'mask',[ri(1) ci(1)],[length(ri) length(ci)]); 

      % Calculate head: 
      Z = th.*rho_ice./rho_fw + bed; 

      Z(mask==0) = NaN; % ocean 
      Z(mask==2) = NaN; % transiently floating ice 
      Z(mask==3) = NaN; % floating ice 

    case {'ground','grounded'}
        Z = ncread(filename,'mask',[ri(1) ci(1)],[length(ri) length(ci)])==1; 

    case 'rock'
        Z = ncread(filename,'mask',[ri(1) ci(1)],[length(ri) length(ci)])==4; 

    case 'ice'
        Z = ismember(ncread(filename,'mask',[ri(1) ci(1)],[length(ri) length(ci)]),[1 2 3]); 

   otherwise 
        Z = ncread(filename,variable,[ri(1) ci(1)],[length(ri) length(ci)]);
end
   
%% Final adjustments: 

if strcmpi(variable,'mask')
    Z = uint8(Z); 
end

Z = Z'; 

end