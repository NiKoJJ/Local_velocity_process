import xarray as xr

ds = xr.open_dataset("itslive_sentinel1_output\S1C_IW_SLC__1SSH_20251219T105633_20251219T105654_005519_00AFFE_96DF_X_S1A_IW_SLC__1SSH_20251225T105734_20251225T105755_062470_07D3B6_FA5F_G0120V02_P085.nc")
for var in ["v", "vx", "vy", "v_error"]:
    if var in ds:
        units = ds[var].attrs.get("units", "未标注")
        desc  = ds[var].attrs.get("long_name", "")
        print(f"{var}: units={units}  ({desc})")
# ```

# 通常输出会是：
# ```
# v:       units=m/yr  (velocity magnitude)
# vx:      units=m/yr  (velocity component in x direction)
# vy:      units=m/yr  (velocity component in y direction)
# v_error: units=m/yr  (velocity error)