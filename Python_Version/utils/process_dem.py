from osgeo import gdal

def crop_local_dem(src_path, bounds, out_path='dem.tif'):
    """
    Clip SPS_0120m_h.tif, then reproject to EPSG:4326
    
    src_path : '/data1/itslive_params/SPS_0120m_h.tif'
    bounds   : [lon_min, lat_min, lon_max, lat_max]  WGS84 
    out_path : path/dem.tif
    """
    in_ds = gdal.Open(src_path, gdal.GA_ReadOnly)
    warp_options = gdal.WarpOptions(
        format='GTIFF',
        outputType=gdal.GDT_Int16,
        resampleAlg='cubic',
        xRes=0.001,
        yRes=0.001,
        dstSRS='EPSG:4326',
        dstNodata=0,
        outputBounds=bounds,
    )
    gdal.Warp(out_path, in_ds, options=warp_options)
    in_ds = None
    print(f'Done: {out_path}')

# Cook 
bounds = [ 145, -71, 160, -66]  # [lon_min, lat_min, lon_max, lat_max]
crop_local_dem('/data1/itslive_params/SPS_0120m_h.tif', bounds, 'dem_wgs84.tif')
