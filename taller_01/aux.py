import xarray as xr
import numpy as np
from datacube.utils.masking import make_mask, mask_invalid_data, describe_variable_flags

def process_image(img: xr.Dataset, sensor: str = "landsat8"):
    
    if sensor in ["landsat8", "landsat9"]:
        refl_b_ = ["coastal", "blue", "green", "red", "nir08", "swir16", "swir22"]
        quality_band = 'qa_pixel'
        cloud_free_mask = (
            # make_mask(img[quality_band], snow='high_confidence') | 
            make_mask(img[quality_band], cloud="high_confidence") |
            make_mask(img[quality_band], cirrus="high_confidence") |
            make_mask(img[quality_band], cloud_shadow="high_confidence") |
            make_mask(img[quality_band], nodata=True)
        )
    
    elif sensor in ["sentinel2"]:
        refl_b_ = ["coastal", "blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "nir08", "nir09", "swir16", "swir22"]
        quality_band = "scl"
        cloud_free_mask = (
            make_mask(img[quality_band], qa="no data") | 
            make_mask(img[quality_band], qa="cloud shadows") | 
            make_mask(img[quality_band], qa="cloud medium probability") |
            make_mask(img[quality_band], qa="cloud high probability") |
            make_mask(img[quality_band], qa="thin cirrus")
        )
    
    bands = [b for b in refl_b_ if b in img.data_vars]
    dsf = xr.where(cloud_free_mask, np.nan, img[bands])  
    
    if sensor in ['sentinel2']:
        dsf.update(dsf * 0.00001)
    else:
        dsf.update(dsf * 0.0000275 + -0.2)
    
    dsf.update(dsf.where(dsf >= 0).where(dsf <= 1))
    
    return dsf