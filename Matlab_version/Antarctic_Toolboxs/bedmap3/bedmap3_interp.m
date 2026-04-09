function zi = bedmap3_interp(variable,lati_or_xi,loni_or_yi,method)
% bedmap3_interp interpolates Bedmap3 gridded data to any coordinates.
%
%% Requirements 
% You'll need two things! 
% 
% 1. The bedmap3.nc file, found here: https://doi.org/10.5285/2d0e4791-8e20-46a3-80e4-f5f6716025d2
% 2. Antarctic Mapping Tools: https://github.com/chadagreene/Antarctic-Mapping-Tools
% 
%% Syntax 
% 
%  zi = bedmap3_interp(variable,lati,loni)
%  zi = bedmap3_interp(variable,xi,yi)
%  zi = bedmap3_interp(...,method)
% 
%% Description
% 
% zi = bedmap3_interp(variable,lati,loni) returns the specified variable at
% given geographic coordinates. The variable can be 
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
% zi = bedmap3_interp(variable,xi,yi) returns the specified variable at the
% given polar stereographic coordinates. 
%
% zi = bedmap3_interp(...,method) specifies a method of 2d interpolation.
% Default method for masks is 'nearest' and default method for all other
% fields is 'linear'. 
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
% 

%% Input checks 

narginchk(3,4) 
extrapval = NaN; % Default extrapolation value

if nargin==3
    if ismember(lower(variable),{'mask','ground','grounded','rock','ice'})
        method = 'nearest'; 
        extrapval = 0; 
    else
        method = 'linear'; 
    end
else 
    assert(~isnumeric(method),'Input error: Interpolation method must be a string.')
end

assert(~isnumeric(variable),'Input error: First input of bedmap3_interp must be a string (a variable name).')
assert(isequal(size(lati_or_xi),size(loni_or_yi)),'Input error: Dimensions of query coordinates must match.')

if islatlon(lati_or_xi,loni_or_yi) % islatlon is in Antarctic Mapping Tools for Matlab. 
    [xi,yi] = ll2ps(lati_or_xi,loni_or_yi); % The ll2ps function is in the Antarctic Mapping Tools package.
else 
   xi = lati_or_xi;
   yi = loni_or_yi;    
end

%% Load data 

[Z,x,y] = bedmap3_data(variable,xi,yi,3); 

zi = interp2(x,y,Z,xi,yi,method,extrapval); 

end