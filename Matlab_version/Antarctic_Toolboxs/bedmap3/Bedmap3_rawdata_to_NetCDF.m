% This script compiles all Bedmap3 raw airborne data into a single NetCDF
% file. Written by Chad A. Greene, NASA/JPL Oct 25, 2023. 
% 
% The final file is called Bedmap_compiled_thickness_bed_surface.nc. 

% Get a list of all the csv files in this folder: 
d = dir('*.csv');

warning off % bc the readtable dislikes the csv headers. 

% Preallocate some variables: 
lon = []; 
lat = []; 
date = []; 
surface_wgs84 = []; 
thickness = []; 
bed_wgs84 = []; 
id = []; 
bedmap_version = []; 

% Read each csv file: 
for k = 1:length(d)

    tmp = readtable(d(k).name);

    lon = [lon;tmp.longitude_degree_east_];
    lat = [lat;tmp.latitude_degree_north_];
    date = [date;datenum(tmp.date)];
    surface_wgs84 = [surface_wgs84;tmp.surface_altitude_m_];
    thickness = [thickness;tmp.land_ice_thickness_m_];
    bed_wgs84 = [bed_wgs84;tmp.bedrock_altitude_m_];
    id = [id;k*ones(size(tmp,1),1)];

    if contains(d(k).name,'BM1')
        bedmap_version = [bedmap_version;ones(size(tmp,1),1)];
    elseif contains(d(k).name,'BM2')
        bedmap_version = [bedmap_version;2*ones(size(tmp,1),1)];
    elseif contains(d(k).name,'BM3')
        bedmap_version = [bedmap_version;3*ones(size(tmp,1),1)];
    end

    k
end

return


% Convert to decimal year for ease of use: 
year = doy(date,'decimalyear'); 

% Estimate the dates where BEDMAP3 neglected to do so or used inconsistent
% date formatting: 
year(date<1900 & id==find(strcmp({d(:).name},'BAS_2007_Lake-Ellsworth_GRN_BM3.csv'))) = 2007;
year(date<1900 & id==find(strcmp({d(:).name},'BAS_2007_Rutford_GRN_BM3.csv'))) =2007;
year(date<1900 & id==find(strcmp({d(:).name},'BAS_2008_Lake-Ellsworth_GRN_BM3.csv'))) = 2008;
year(date<1900 & id==find(strcmp({d(:).name},'BAS_2010_IMAFI_AIR_BM3.csv'))) = 2010;
year(date<1900 & id==find(strcmp({d(:).name},'BAS_2011_Adelaide_AIR_BM3.csv'))) = 2011;
year(date<datenum(1900,1,1) & id==find(strcmp({d(:).name},'INGV_1997_Talos-Dome_AIR_BM3.csv'))) = 1997;
year(date<datenum(1900,1,1) & id==find(strcmp({d(:).name},'INGV_1999_Talos-Dome_AIR_BM3.csv'))) = 1999;
year(date<datenum(1900,1,1) & id==find(strcmp({d(:).name},'INGV_2001_Talos-Dome_AIR_BM3.csv'))) = 2001;
year(date<datenum(1900,1,1) & id==find(strcmp({d(:).name},'INGV_2003_Talos-Dome_AIR_BM3.csv'))) = 2003;
year(date<1900 & id==find(strcmp({d(:).name},'PRIC_2015_CHA1_AIR_BM3.csv'))) = 2015;
year(date<1900 & id==find(strcmp({d(:).name},'PRIC_2016_CHA2_AIR_BM3.csv'))) = 2016;
year(date<1900 & id==find(strcmp({d(:).name},'PRIC_2017_CHA3_AIR_BM3.csv'))) = 2017;
year(date<1900 & id==find(strcmp({d(:).name},'PRIC_2018_CHA4_AIR_BM3.csv'))) = 2018;
year(date<1900 & id==find(strcmp({d(:).name},'RNRF_1971_Lambert-Amery_SEI_BM3.csv'))) = 1971;
year(date<1900 & id==find(strcmp({d(:).name},'RNRF_1975_Filchner-Ronne_SEI_BM3.csv'))) = 1975;
year(date<1900 & id==find(strcmp({d(:).name},'RNRF_1975_Lazarev_SEI_BM3.csv'))) = 1975;
year(date<1900 & id==find(strcmp({d(:).name},'RNRF_2004_Mirny-Vostok_AIR_BM3.csv'))) = 2004;
year(date<1900 & id==find(strcmp({d(:).name},'RNRF_2006_RAEap5_AIR_BM3.csv'))) = 2006;
year(date<1900 & id==find(strcmp({d(:).name},'RNRF_2009_RAEap5_AIR_BM3.csv'))) = 2009;
year(isnan(date) & id==find(strcmp({d(:).name},'RNRF_2010_RAE_AIR_BM3.csv'))) = 2010; 
year(isnan(date) & id==find(strcmp({d(:).name},'RNRF_2011_RAE_AIR_BM3.csv'))) = 2011; 
year(isnan(date) & id==find(strcmp({d(:).name},'RNRF_2013_RAE_AIR_BM3.csv'))) = 2013; 
year(date<1900 & id==find(strcmp({d(:).name},'STANFORD_1971_SPRI-NSF-TUD_AIR_BM3.csv'))) = 1978;

year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'AWI_1994_DML1_AIR_BM2.csv'))) = 1994;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'AWI_1995_DML2_AIR_BM2.csv'))) = 1995;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'AWI_1996_DML3_AIR_BM2.csv'))) = 1996;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'AWI_1997_DML4_AIR_BM2.csv'))) = 1997;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'AWI_1998_DML5_AIR_BM2.csv'))) = 1998;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'AWI_2000_DML6_AIR_BM2.csv'))) = 2000;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'AWI_2001_DML7_AIR_BM2.csv'))) = 2001;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'AWI_2002_DML8_AIR_BM2.csv'))) = 2002;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'AWI_2003_DML9_AIR_BM2.csv'))) = 2003;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'AWI_2004_DML10_AIR_BM2.csv'))) = 2004;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'AWI_2005_ANTSYSO_AIR_BM2.csv'))) = 2005;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'AWI_2007_ANTR_AIR_BM2.csv'))) = 2007;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'BAS_1994_Evans_AIR_BM2.csv'))) = 1994;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'BAS_1998_Dufek_AIR_BM2.csv'))) = 1998;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'BAS_2001_Bailey-Slessor_AIR_BM2.csv'))) = 2001;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'BAS_2004_BBAS_AIR_BM2.csv'))) = 2004;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'BAS_2005_WISE-ISODYN_AIR_BM2.csv'))) = 2005;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'BAS_2007_Lake-Ellsworth_GRN_BM3.csv'))) = 2007;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'BAS_2007_Rutford_GRN_BM3.csv.csv'))) = 2007;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'BAS_2008_Lake-Ellsworth_GRN_BM3.csv'))) = 2008;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'BAS_2010_IMAFI_AIR_BM3.csv'))) = 2010;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'BAS_2010_PIG_AIR_BM2.csv'))) = 2010;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'BAS_2011_Adelaide_AIR_BM3.csv'))) = 2011;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'BGR_1999_GANOVEX-VIII-Matusevich_AIR_BM2.csv'))) = 1999;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'BGR_1999_GANOVEX-VIII-Mertz_AIR_BM2.csv'))) = 1999 ;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'BGR_2002_PCMEGA_AIR_BM2.csv'))) = 2002;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'INGV_1997_ITASE_AIR_BM2.csv'))) = 1997;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'LDEO_2007_AGAP-GAMBIT_AIR_BM2.csv'))) = 2007;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'LDEO_2007_Recovery-Lakes_AIR_BM2.csv'))) = 2007;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'NASA_2002_ICEBRIDGE_AIR_BM2.csv'))) = 2002;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'NASA_2004_ICEBRIDGE_AIR_BM2.csv'))) = 2004;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'NASA_2009_ICEBRIDGE_AIR_BM2.csv'))) = 2009;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'NASA_2010_ICEBRIDGE_AIR_BM2.csv'))) = 2010;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'NIPR_1999_JARE40_GRN_BM2.csv'))) = 1999;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'NIPR_2007_JARE49_GRN_BM2.csv'))) = 2007;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'NIPR_2007_JASE_GRN_BM2.csv'))) = 2007;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'NPI_2008_BELISSIMA_GRN_BM2.csv'))) = 2008;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'NPI_2010_SRM_AIR_BM2.csv'))) = 2010;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'PRIC_2004_CHINARE-21_GRN_BM2.csv'))) = 2004;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'PRIC_2007_CHINARE-24_GRN_BM2.csv'))) = 2007;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'PRIC_2015_CHA1_AIR_BM3.csv.csv'))) = 2015;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'PRIC_2016_CHA2_AIR_BM3.csv'))) = 2015;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'PRIC_2017_CHA3_AIR_BM3.csv'))) = 2017;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'PRIC_2018_CHA4_AIR_BM3.csv'))) = 2018;

year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'BAS_2001_MAMOG_AIR_BM2.csv'))) = 2001;
year(year<1950 & id==find(strcmp({d(:).name},'BAS_2001_MAMOG_AIR_BM2.csv'))) = 2001;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'RNRF_2003_48RAEap5_AIR_BM2.csv'))) = 2003;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'RNRF_2004_49RAEap5_AIR_BM2.csv'))) = 2004;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'RNRF_2005_50RAEap5_AIR_BM2.csv'))) = 2005;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'RNRF_2006_51RAEap5_AIR_BM2.csv'))) = 2006;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'RNRF_2006_KV1-area_AIR_BM2.csv'))) = 2006;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'RNRF_2007_52RAEap5_AIR_BM2.csv'))) = 2007;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'RNRF_2007_Mirny-Vostok_AIR_BM2.csv'))) = 2007;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'RNRF_2008_53RAEap5_AIR_BM2.csv'))) = 2008;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'RNRF_2008_Vostok-Subglacial-Lake_AIR_BM2.csv'))) = 2008;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'RNRF_2013_Vostok-Progress_AIR_BM2.csv'))) = 2013;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'STOLAF_1994_Siple-Dome_GRN_BM2.csv'))) = 1994;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'STOLAF_2001_ITASE-Byrd-Ellsworth_GRN_BM2.csv'))) = 2001;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'STOLAF_2001_ITASE-Ellsworth_GRN_BM2.csv'))) = 2001;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'STOLAF_2002_ITASE-Byrd-South-Pole_GRN_BM2.csv'))) = 2002;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'STOLAF_2002_ITASE-Hercules-Dome_GRN_BM2.csv'))) = 2002;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'UCANTERBURY_2008_Darwin-Hatherton_GRN_BM2.csv'))) = 2008;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'UTIG_1991_CASERTZ_AIR_BM2.csv'))) = 1991;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'UTIG_1998_West-Marie-Byrd-Land_AIR_BM2.csv'))) = 1998;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'UTIG_1999_SOAR-LVS-WLK_AIR_BM2.csv'))) = 1999;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'UTIG_2000_Robb-Glacier_AIR_BM2.csv'))) = 2000;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'UTIG_2004_AGASEA_AIR_BM2.csv'))) = 2004;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'UTIG_2008_ICECAP_AIR_BM2.csv'))) = 2008;
year((date<=0 | isnan(date)) & id==find(strcmp({d(:).name},'BEDMAP1_1966-2000_AIR_BM1.csv'))) = 0;

%%

% Convert to projected coordinates: 
[x,y] = ll2ps(lat,lon); 

% Geoid-referenced data for convenience: 
bed_geoid = wgs2gl04c(x,y,bed_wgs84); 
surface_geoid = wgs2gl04c(x,y,surface_wgs84); 

% Write no-data values: 
bed_geoid(bed_wgs84==-9999) = -9999; 
surface_geoid(surface_wgs84==-9999) = -9999; 

% Eliminate spots where all data are missing: 
bad = thickness==-9999 & bed_geoid==-9999 & surface_geoid==-9999;
x(bad) = []; 
y(bad) = [];
lat(bad) = [];
lon(bad) = [];
year(bad) = [];
thickness(bad) = [];
surface_wgs84(bad) = [];
surface_geoid(bad) = [];
bed_wgs84(bad) = [];
bed_geoid(bad) = [];
id(bad) = [];
bedmap_version(bad) = []; 

clear bad d

%% Write the netcdf 

% 1. Create netCDF file handle and Attributes
mode = netcdf.getConstant('NETCDF4');
%mode = bitor(mode,netcdf.getConstant('CLASSIC_MODEL'));
ncid=netcdf.create('Bedmap_compiled_thickness_bed_surface.nc',mode);
netcdf.putAtt(ncid,netcdf.getConstant('NC_GLOBAL'),'Conventions','CF-1.8');
netcdf.putAtt(ncid,netcdf.getConstant('NC_GLOBAL'),'Title','BEDMAP3 - Ice thickness, bed and surface elevation for Antarctica - standardised data points');
netcdf.putAtt(ncid,netcdf.getConstant('NC_GLOBAL'),'Description','All Bedmap3 data compiled into a single NetCDF.');
netcdf.putAtt(ncid,netcdf.getConstant('NC_GLOBAL'),'creation_date',datestr(now,'yyyy-mm-dd'));
netcdf.putAtt(ncid,netcdf.getConstant('NC_GLOBAL'),'NetCDF_conversion','Chad A. Greene');
netcdf.putAtt(ncid,netcdf.getConstant('NC_GLOBAL'),'doi','BEDMAP1 = https://doi.org/jg6q, BEDMAP2 = https://doi.org/jg6r, BEDMAP3 = https://doi.org/jg6n');
netcdf.putAtt(ncid,netcdf.getConstant('NC_GLOBAL'),'Literature_citation','Frémand, A. C., Fretwell, P., Bodart, J. A., Pritchard, H. D., Aitken, A., Bamber, J. L., Bell, R., Bianchi, C., Bingham, R. G., Blankenship, D. D., Casassa, G., Catania, G., Christianson, K., Conway, H., Corr, H. F. J., Cui, X., Damaske, D., Damm, V., Drews, R., Eagles, G., Eisen, O., Eisermann, H., Ferraccioli, F., Field, E., Forsberg, R., Franke, S., Fujita, S., Gim, Y., Goel, V., Gogineni, S. P., Greenbaum, J., Hills, B., Hindmarsh, R. C. A., Hoffman, A. O., Holmlund, P., Holschuh, N., Holt, J. W., Horlings, A. N., Humbert, A., Jacobel, R. W., Jansen, D., Jenkins, A., Jokat, W., Jordan, T., King, E., Kohler, J., Krabill, W., Kusk Gillespie, M., Langley, K., Lee, J., Leitchenkov, G., Leuschen, C., Luyendyk, B., MacGregor, J., MacKie, E., Matsuoka, K., Morlighem, M., Mouginot, J., Nitsche, F. O., Nogi, Y., Nost, O. A., Paden, J., Pattyn, F., Popov, S. V., Rignot, E., Rippin, D. M., Rivera, A., Roberts, J., Ross, N., Ruppel, A., Schroeder, D. M., Siegert, M. J., Smith, A. M., Steinhage, D., Studinger, M., Sun, B., Tabacco, I., Tinto, K., Urbini, S., Vaughan, D., Welch, B. C., Wilson, D. S., Young, D. A., and Zirizzotti, A.: Antarctic Bedmap data: Findable, Accessible, Interoperable, and Reusable (FAIR) sharing of 60 years of ice bed, surface, and thickness data, Earth Syst. Sci. Data, 15, 2695–2710, https://doi.org/10.5194/essd-15-2695-2023, 2023. ');

% 2. Define dimensions
% Define mapping variable
mapping_var_id= netcdf.defVar(ncid,'mapping','NC_CHAR',[]);
netcdf.putAtt(ncid,mapping_var_id,'grid_mapping_name', 'polar_stereographic');
netcdf.putAtt(ncid,mapping_var_id,'latitude_of_projection_origin',-90.);
netcdf.putAtt(ncid,mapping_var_id,'standard_parallel',-71.);
netcdf.putAtt(ncid,mapping_var_id,'false_easting',0.);
netcdf.putAtt(ncid,mapping_var_id,'false_northing',0.); 

% Define x: 
x_id     = netcdf.defDim(ncid,'x',length(x));
x_var_id = netcdf.defVar(ncid,'x','NC_FLOAT',x_id);
netcdf.putAtt(ncid,x_var_id,'standard_name','projection_x_coordinate');
netcdf.putAtt(ncid,x_var_id,'units',        'meter');

% Define y: 
%y_id     = netcdf.defDim(ncid,'y',length(y));
y_var_id = netcdf.defVar(ncid,'y','NC_FLOAT',x_id);
netcdf.putAtt(ncid,y_var_id,'standard_name','projection_y_coordinate');
netcdf.putAtt(ncid,y_var_id,'units',        'meter');

% Define lat
%lat_id     = netcdf.defDim(ncid,'lat',length(lat));
lat_var_id = netcdf.defVar(ncid,'lat','NC_FLOAT',x_id);
netcdf.putAtt(ncid,lat_var_id,'standard_name','latitude');
netcdf.putAtt(ncid,lat_var_id,'units',        'degree');

% Define lon
%lon_id     = netcdf.defDim(ncid,'lon',length(lon));
lon_var_id = netcdf.defVar(ncid,'lon','NC_FLOAT',x_id);
netcdf.putAtt(ncid,lon_var_id,'standard_name','longitude');
netcdf.putAtt(ncid,lon_var_id,'units',        'degree');

% Define year
%year_id     = netcdf.defDim(ncid,'year',length(year));
year_var_id = netcdf.defVar(ncid,'year','NC_FLOAT',x_id);
netcdf.putAtt(ncid,year_var_id,'standard_name','year');

% Define thickness
%thickness_id     = netcdf.defDim(ncid,'thickness',length(thickness));
thickness_var_id = netcdf.defVar(ncid,'thickness','NC_FLOAT',x_id);
netcdf.putAtt(ncid,thickness_var_id,'standard_name','ice thickness');
netcdf.putAtt(ncid,thickness_var_id,'_FillValue',single(-9999));

% Define surface_wgs84
%surface_wgs84_id     = netcdf.defDim(ncid,'surface_wgs84',length(surface_wgs84));
surface_wgs84_var_id = netcdf.defVar(ncid,'surface_wgs84','NC_FLOAT',x_id);
netcdf.putAtt(ncid,surface_wgs84_var_id,'standard_name','surface_wgs84');
netcdf.putAtt(ncid,surface_wgs84_var_id,'description','surface elevation relative to the WGS84 ellipsoid');
netcdf.putAtt(ncid,surface_wgs84_var_id,'_FillValue',single(-9999));

% Define surface_geoid
%surface_geoid_id     = netcdf.defDim(ncid,'surface_geoid',length(surface_geoid));
surface_geoid_var_id = netcdf.defVar(ncid,'surface_geoid','NC_FLOAT',x_id);
netcdf.putAtt(ncid,surface_geoid_var_id,'standard_name','surface_geoid');
netcdf.putAtt(ncid,surface_geoid_var_id,'description','surface elevation relative to the GL04C geoid (https://doi.org/10.1007/s00190-007-0183-8)');
netcdf.putAtt(ncid,surface_geoid_var_id,'_FillValue',single(-9999));

% Define bed_wgs84
%bed_wgs84_id     = netcdf.defDim(ncid,'bed_wgs84',length(bed_wgs84));
bed_wgs84_var_id = netcdf.defVar(ncid,'bed_wgs84','NC_FLOAT',x_id);
netcdf.putAtt(ncid,bed_wgs84_var_id,'standard_name','bed_wgs84');
netcdf.putAtt(ncid,bed_wgs84_var_id,'description','bed elevation relative to the WGS84 ellipsoid');
netcdf.putAtt(ncid,bed_wgs84_var_id,'_FillValue',single(-9999));

% Define bed_geoid
%bed_geoid_id     = netcdf.defDim(ncid,'bed_geoid',length(bed_geoid));
bed_geoid_var_id = netcdf.defVar(ncid,'bed_geoid','NC_FLOAT',x_id);
netcdf.putAtt(ncid,bed_geoid_var_id,'standard_name','bed_geoid');
netcdf.putAtt(ncid,bed_geoid_var_id,'description','bed elevation relative to the GL04C geoid (https://doi.org/10.1007/s00190-007-0183-8)');
netcdf.putAtt(ncid,bed_geoid_var_id,'_FillValue',single(-9999));

% Define id
%id_id     = netcdf.defDim(ncid,'id',length(id));
id_var_id = netcdf.defVar(ncid,'id','NC_UBYTE',x_id);
netcdf.putAtt(ncid,id_var_id,'standard_name','id');
netcdf.putAtt(ncid,id_var_id,'description','Data file indentification corresponding to list in Bedmap3_source_data.xlsx') 

%bedmap_id     = netcdf.defDim(ncid,'bedmap_version',length(id));
bedmap_var_id = netcdf.defVar(ncid,'bedmap_version','NC_UBYTE',x_id);
netcdf.putAtt(ncid,bedmap_var_id,'description','Bedmap version. 1 = BEDMAP1 (https://doi.org/jg6q), 2 = BEDMAP2 (https://doi.org/jg6r), 3 = BEDMAP3 (https://doi.org/jg6n)');

netcdf.endDef(ncid);

%3. Place data
netcdf.putVar(ncid,x_var_id,single(x));
netcdf.putVar(ncid,y_var_id,single(y));
netcdf.putVar(ncid,lat_var_id,single(lat));
netcdf.putVar(ncid,lon_var_id,single(lon));
netcdf.putVar(ncid,year_var_id,single(year));
netcdf.putVar(ncid,thickness_var_id,single(thickness));
netcdf.putVar(ncid,surface_wgs84_var_id,single(surface_wgs84));
netcdf.putVar(ncid,surface_geoid_var_id,single(surface_geoid));
netcdf.putVar(ncid,bed_wgs84_var_id,single(bed_wgs84));
netcdf.putVar(ncid,bed_geoid_var_id,single(bed_geoid));
netcdf.putVar(ncid,id_var_id,uint8(id));
netcdf.putVar(ncid,bedmap_var_id,uint8(bedmap_version));

%4. Close file 
netcdf.close(ncid)