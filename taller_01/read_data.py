%matplotlib inline

import os

os.environ['USE_PYGEOS'] = '0'

import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import datacube
import matplotlib.pyplot as plt
import matplotlib as mpl
import xarray as xr
import rioxarray
import xrspatial as xrs
from datacube.utils.cog import write_cog
#from datacube.helpers import write_geotiff  # deprecated
from datacube.utils.masking import make_mask, mask_invalid_data, describe_variable_flags
from datacube.utils.rio import configure_s3_access
from dask.distributed import Client, LocalCluster

cluster = LocalCluster()
client = Client(cluster)

configure_s3_access(aws_unsigned=False, requester_pays=True, client=client)

dc = datacube.Datacube(app='Taller 1') 

#--------------------- Lectura vectores ------------------
catveg = gpd.read_file("vectores/CatVegeAdeRamón.shp")
catveg_cols = ['USO_TIERRA', 'USO', 'SUBUSO', 'ESTRUCTURA',
       'COBERTURA', 'ALTURA', 'TIPO_FORES', 'SUBTIPOFOR', 'ESPECI1_CI',
       'ESPECI2_CI', 'ESPECI3_CI', 'ESPECI4_CI', 'ESPECI5_CI', 'ESPECI6_CI',
       'ESPECI1_CO', 'ESPECI2_CO', 'ESPECI3_CO', 'ESPECI4_CO', 'ESPECI5_CO',
       'ESPECI6_CO', 'ESPCC1', 'ESPCC2', 'geometry']
cuenca = gpd.read_file("vectores/CuencaAdeRamón.shp")
muestreo = gpd.read_file("vectores/puntos_muestreo.shp")

## Unión espacial (Spatial Join) entre los puntos de muestreo y la cobertura de categoría vegetacional
v1 = muestreo.sjoin(catveg[catveg_cols]).reset_index(drop=True)



#----------- Rasters de Variables Funcionales
rasters_names = ["Cummulative Greenness", "EFT", "Minimum Greenness", "Season Length", "Seasonal Variation"]

ras = []
for r in rasters_names:
    r_ = xr.open_dataset(f"raster/{r}.tif", engine="rasterio").squeeze(drop=True)
    r_ = r_.rename({"band_data": r.lower().replace(" ", "_")})
    ras.append(r_)
    
vf = xr.merge(ras)

#----------- ¿Cómo se ven con los puntos de muestreo?
fig, ax = plt.subplots(figsize=(15, 10))
vf.season_length.plot(ax=ax)
cuenca.plot(ax=ax, edgecolor='red', color='none')


## Extraer valor de variables funcionales para los puntos de muestreo
x_ = xr.DataArray(v1.geometry.x)
y_ = xr.DataArray(v1.geometry.y)
valores = vf.sel(x=x_, y=y_, method="nearest").to_array().values
vfv = pd.DataFrame(valores.T, columns=vf.keys(), index=v1.id).reset_index()

v2 = pd.merge(v1, vfv, on="id")


#------- Buscando imágenes: DEM, Landsat-8 y Sentinel-2 -----------
x0, y0, x1, y1 = cuenca.to_crs(4326).bounds.values[0]

query = {
    "y": (y0, y1), 
    "x": (x0, x1),
    "output_crs": "EPSG:32719",
    "resolution": (-30, 30),
    "dask_chunks": {"time": 1, 'x':2048, 'y':2048},
    "group_by": "solar_day"
}

# ---------- DEM -------
query["product"] = "copernicus_dem_30"
dem = dc.load(**query).squeeze()
dem.elevation.plot()

slope = xrs.slope(dem.elevation).to_dataset()  # en grados
aspect = xrs.aspect(dem.elevation).to_dataset() 

topo = xr.merge([dem, slope, aspect])
valores = topo.sel(x=x_, y=y_, method="nearest").to_array().values
topov = pd.DataFrame(valores.T, columns=topo.keys(), index=v1.id).reset_index()

v3 = pd.merge(v2, topov, on="id")


#--------- LANDSAT 8 ---------
## Importar una librería para simplificar procesamiento de Landsat y Sentinel.
## Más adelante lo revisaremos con mayor detalle
from aux import process_image

query["product"] = "landsat8_c2l2_sr"
query["time"] = ("2019-01-01", "2019-12-31")
l8 = dc.load(**query)

l8f = process_image(l8, "landsat8")
# Cálculo del índice vegetacional NDVI
ndvi_l8_all = (l8f["nir08"] - l8f["red"]) / (l8f["nir08"] + l8f["red"])
# Cálculo de una mediana anual de NDVI
ndvi_l8 = ndvi_l8_all.median(dim="time").compute()

ndvi_l8.attrs = l8.attrs
write_cog(ndvi_l8, "ndvi_l8.tif")


#-------- Sentinel-2
query["product"] = "s2_l2a"
query["resolution"] = (-10, 10)
s2 = dc.load(**query)

s2f = process_image(s2, "sentinel2")
ndvi_s2_all = (s2f["nir"] - s2f["red"]) / (s2f["nir"] + s2f["red"])
ndvi_s2 = ndvi_s2_all.median(dim="time").compute()

ndvi_s2.attrs = s2.attrs
write_cog(ndvi_s2, "ndvi_s2.tif")


#---------- Data Final
## Extraer los valores de NDVI de Landsat-8 y Sentinel-2 para los puntos de muestreo.
valores1 = ndvi_l8.sel(x=x_, y=y_, method="nearest").values
valores2 = ndvi_s2.sel(x=x_, y=y_, method="nearest").values
inds = pd.DataFrame([valores1, valores2], columns=v1.id, index=["ndvi_l8", "ndvi_s2"]).T.reset_index()

v4 = pd.merge(v3, inds)

## Exportando a shapefile y CSV
v4.to_file("puntos_muestreo_data.shp")
v4.drop(columns="geometry").to_csv("puntos_muestreo_data.csv", index=False)


#-------- Estadísticas generales variables funcionales
from xrspatial.zonal import stats

zonas = []
zon = vf.eft.fillna(0) # no pueden haber nan
for b in vf.data_vars:
    if b != 'eft':
        _ = stats(zones=zon, values=vf[b])
        _["vf"] = b
        _["area_km2"] = (np.abs(vf.rio.resolution()).prod() * _["count"] / 10**6).round(2)
        zonas.append(_)

out = pd.concat(zonas).set_index("vf").reset_index()
out.rename(columns={"zone": "eft"}, inplace=True)
out.to_csv("efts_stats.csv", index=False)


client.close()
cluster.close()
