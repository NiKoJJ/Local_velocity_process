function h = bedmap3(variable,varargin)
% bedmap3 loads and plots bedmap3 data on a polar stereographic map. 
% 
%% Requirements 
% You'll need two things! 
% 
% 1. The bedmap3.nc file, found here: https://doi.org/10.5285/2d0e4791-8e20-46a3-80e4-f5f6716025d2
% 2. Antarctic Mapping Tools: https://github.com/chadagreene/Antarctic-Mapping-Tools
% 
%% Syntax
% 
%  bedmap3
%  bedmap3(variable) 
%  bedmap3(variable,'contour',PropertyName,PropertyValue,...) 
%  bedmap3(outline,LineProperty,LineValue,...)
%  h = bedmap3(...) 
% 
%% Description 
% 
% bedmap3 plots a gray grounding line and coast line from bedmap3. 
% 
% bedmap3(variable) plots any Bedmap3 variable as an imagesc plot.
% The variable can be: 
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
% bedmap3(variable,'contour',PropertyName,PropertyValue,...) plots any 
% BedMachine variable as a contour plot. 
%
% bedmap3(outline,LineProperty,LineValue,...) plots any of the following, 
% with optional line formatting: 
%    * 'gl'      grounding line
%    * 'coast'   coast line
% 
% h = bedmap3(...) returns a handle h of the plotted object(s). 
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
% This function was adapted from the bedmachine function by Chad A. Greene, April 2025. 
% 
% See also: bedmap3_data and bedmap3_interp. 

%% Shortcut to a simple map: 

if nargin==0
   h(1) = bedmap3('coast','color',[0.5725    0.5843    0.5686]); 
   h(2) = bedmap3('gl','color',[0.5725    0.5843    0.5686]);
   if nargout==0
      clear h 
   end
   return
end

%% Set defaults

outline = true; 
contourplot = false; 

%% Parse inputs

if nargin>0
   if ~ismember(lower(variable),{'gl','coast'})
      outline = false; 
   end
   
   tmp = strncmpi(varargin,'contour',3); 
   if any(tmp)
      contourplot = true; 
      varargin = varargin(~tmp); 
   end
   
end

%% Get initial conditions: 

da = daspect; 
da = [1 1 da(3)]; 
hld = ishold; 
hold on

mapwasopen = ~isequal(axis,[0 1 0 1]); 
ax = axis; 

%%

if outline
   
   % Load bedmap3-derived outlines: 
   switch variable
       case 'gl'
          B3 = load('bedmap3_gl_and_coastline.mat','gl'); 
          x = B3.gl.Vertices(:,1);          
          y = B3.gl.Vertices(:,2);

       case 'coast'
          B3 = load('bedmap3_gl_and_coastline.mat','coast'); 
          x = B3.coast.Vertices(:,1);          
          y = B3.coast.Vertices(:,2);
   end

   if mapwasopen
      OutOfBounds = x<ax(1) | x>ax(2) | y<ax(3) | y>ax(4); 
      x(OutOfBounds) = NaN; % trims away everything outside current map extents while keeping the nans that separate different sections of the outline 
      y(OutOfBounds) = NaN; 
   else
      axis([min(x) max(x) min(y) max(y)])
   end
   
   h = plot(x,y,varargin{:}); 
else
   

   if mapwasopen
      [Z,x,y] = bedmap3_data(variable,ax(1:2),ax(3:4)); 
   else
      [Z,x,y] = bedmap3_data(variable); 
   end
   
   if contourplot
      [~,h] = contour(x,y,Z,varargin{:}); 
   else
      
      h = imagesc(x,y,Z); 
      set(h,'alphadata',isfinite(Z)); 
      axis xy; 
   end
   
end

%% Put things back the way we found them: 

daspect(da)
if ~hld
   hold off
end

if mapwasopen
   axis(ax); 
else 
   axis([min(x) max(x) min(y) max(y)]); 
end

%% Clean up: 

if nargout==0 
   clear h
end

end