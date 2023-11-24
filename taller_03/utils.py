import folium
import itertools    
import math
import numpy as np
import pyproj
import xarray as xr
import pandas as pd

import joblib
import dask.array as da
from datacube.utils.geometry import assign_crs
from dask_ml.wrappers import ParallelPostFit

def extract_data(array, points, include_xy=False):
    x_ = xr.DataArray(points.geometry.x, dims=["id"], coords={"id": points.id})
    y_ = xr.DataArray(points.geometry.y, dims=["id"], coords={"id": points.id})
    valores = array.sel(x=x_, y=y_, method="nearest").to_array()
    _ = valores.to_pandas().T
    if include_xy:
        return pd.DataFrame({"x": points.geometry.x.values, "y": points.geometry.y.values}, index=points.id).join(_)
    else:
        return _

def coordinate_converter(x, y, source_crs = "epsg: 32719", target_crs = "epsg:4326"):
    transformer = pyproj.Transformer.from_crs(source_crs, target_crs)
    return transformer.transform(x, y)

def _degree_to_zoom_level(l1, l2, margin = 0.0):
    
    degree = abs(l1 - l2) * (1 + margin)
    zoom_level_int = 0
    if degree != 0:
        zoom_level_float = math.log(360/degree)/math.log(2)
        zoom_level_int = int(zoom_level_float)
    else:
        zoom_level_int = 18
    return zoom_level_int



def metricas(cm):
    FP = cm.sum(axis=0) - np.diag(cm)  # Falsos positivos
    FN = cm.sum(axis=1) - np.diag(cm)  # Falsos negativos
    TP = np.diag(cm)  # Verdaderos positivos
    TN = cm.sum() - (FP + FN + TP)  # Verdaderos negativos
    FNR = FN / (TP + FN)
    
    Po = TP / cm.sum() 
    Pe = (cm.sum(axis=0) * cm.sum(axis=1)) / cm.sum() ** 2 
    k = (Po - Pe) / (1 - Pe)
    
    mtr = {
        "FP": FP,
        "FN": FN,
        "TP": TP,
        "TN": TN,
        "TPR": TP / (TP + FN), # Sensitivity, hit rate, recall, or true positive rate 
        "TNR": TN / (TN + FP), # Specificity or true negative rate 
        "PPV": TP / (TP + FP), # Precision or positive predictive value or consumer accuracy 
        "NPV": TN / (TN + FN), # Negative predictive value 
        "FPR": FP / (FP + TN), # Fall out or false positive rate 
        "FNR": FNR, # False negative rate 
        "FDR": FP / (TP + FP), # False discovery rate 
        "PA": 1 - FNR,  # Producer's accuracy
        "F1": 2*TP / (2*TP + FP + FN), # F1 Score
        "Kappa": (Po - Pe) / (1 - Pe),
        "Kappa_score": (Po.sum() - Pe.sum()) / (1 - Pe.sum())
    }    
    
    return mtr



def display_map(latitude = None, longitude = None, resolution = None):
    """ Generates a folium map with a lat-lon bounded rectangle drawn on it. Folium maps can be 
    
    Args:
        latitude   (float,float): a tuple of latitude bounds in (min,max) format
        longitude  ((float, float)): a tuple of longitude bounds in (min,max) format
        resolution ((float, float)): tuple in (lat,lon) format used to draw a grid on your map. Values denote   
                                     spacing of latitude and longitude lines.  Gridding starts at top left 
                                     corner. Default displays no grid at all.  

    Returns:
        folium.Map: A map centered on the lat lon bounds. A rectangle is drawn on this map detailing the
        perimeter of the lat,lon bounds.  A zoom level is calculated such that the resulting viewport is the
        closest it can possibly get to the centered bounding rectangle without clipping it. An 
        optional grid can be overlaid with primitive interpolation.  

    .. _Folium
        https://github.com/python-visualization/folium

    """
    
    assert latitude is not None
    assert longitude is not None

    ###### ###### ######   CALC ZOOM LEVEL     ###### ###### ######

    margin = -0.5
    zoom_bias = 0
    
    lat_zoom_level = _degree_to_zoom_level(*latitude, margin = margin) + zoom_bias
    lon_zoom_level = _degree_to_zoom_level(*longitude, margin = margin) + zoom_bias
    zoom_level = min(lat_zoom_level, lon_zoom_level) 

    ###### ###### ######   CENTER POINT        ###### ###### ######
    
    center = [np.mean(latitude), np.mean(longitude)]

    ###### ###### ######   CREATE MAP         ###### ###### ######
    
    map_hybrid = folium.Map(
        location=center,
        zoom_start=zoom_level, 
        tiles=" http://mt1.google.com/vt/lyrs=y&z={z}&x={x}&y={y}",
        attr="Google"
    )
    
    ###### ###### ######   RESOLUTION GRID    ###### ###### ######
    
    if resolution is not None:
        res_lat, res_lon = resolution

        lats = np.arange(*latitude, abs(res_lat))
        lons = np.arange(*longitude, abs(res_lon))

        vertical_grid   = map(lambda x :([x[0][0],x[1]],[x[0][1],x[1]]),itertools.product([latitude],lons))
        horizontal_grid = map(lambda x :([x[1],x[0][0]],[x[1],x[0][1]]),itertools.product([longitude],lats))

        for segment in vertical_grid:
            folium.features.PolyLine(segment, color = 'white', opacity = 0.3).add_to(map_hybrid)    
        
        for segment in horizontal_grid:
            folium.features.PolyLine(segment, color = 'white', opacity = 0.3).add_to(map_hybrid)   
    
    ###### ###### ######     BOUNDING BOX     ###### ###### ######
    
    line_segments = [(latitude[0],longitude[0]),
                     (latitude[0],longitude[1]),
                     (latitude[1],longitude[1]),
                     (latitude[1],longitude[0]),
                     (latitude[0],longitude[0])
                    ]
    
    
    
    map_hybrid.add_child(
        folium.features.PolyLine(
            locations=line_segments,
            color='red',
            opacity=0.8)
    )

    map_hybrid.add_child(folium.features.LatLngPopup())        

    return map_hybrid



# extracted from: https://gist.github.com/cbur24/9a52c14698b6a9324a62f5449972cf7f
def predict_xr(model,
               input_xr,
               chunk_size=None,
               persist=True,
               proba=False,
               clean=False,
               return_input=False):
    """
    Using dask-ml ParallelPostfit(), runs  the parallel
    predict and predict_proba methods of sklearn
    estimators. Useful for running predictions
    on a larger-than-RAM datasets.
    
    Last modified: September 2020

    Parameters
    ----------
    model : scikit-learn model or compatible object
        Must have a .predict() method that takes numpy arrays.
    input_xr : xarray.DataArray or xarray.Dataset. 
        Must have dimensions 'x' and 'y'
    chunk_size : int
        The dask chunk size to use on the flattened array. If this
        is left as None, then the chunks size is inferred from the
        .chunks method on the `input_xr`
    persist : bool
        If True, and proba=True, then 'input_xr' data will be
        loaded into distributed memory. This will ensure data
        is not loaded twice for the prediction of probabilities,
        but this will only work if the data is not larger than RAM.
    proba : bool
        If True, predict probabilities
    clean : bool
        If True, remove Infs and NaNs from input and output arrays
    return_input : bool
        If True, then the data variables in the 'input_xr' dataset will
        be appended to the output xarray dataset.
    
    Returns
    ----------
    output_xr : xarray.Dataset 
        An xarray.Dataset containing the prediction output from model 
        with input_xr as input, if proba=True then dataset will also contain
        the prediciton probabilities. Has the same spatiotemporal structure 
        as input_xr.

    """
    if chunk_size is None:
        chunk_size=int(input_xr.chunks['x'][0])*int(input_xr.chunks['y'][0])
    
    #convert model to dask predict
    model=ParallelPostFit(model)   
    
    with joblib.parallel_backend('dask'):
        x, y, crs = input_xr.x, input_xr.y, input_xr.geobox.crs

        input_data = []

        for var_name in input_xr.data_vars:
            input_data.append(input_xr[var_name])

        input_data_flattened = []

        for data in input_data:
            data = data.data.flatten().rechunk(chunk_size)
            input_data_flattened.append(data)

        # reshape for prediction
        input_data_flattened = da.array(input_data_flattened).transpose()

        if clean==True:        
            input_data_flattened = da.where(da.isfinite(input_data_flattened),
                                            input_data_flattened, 0)

        if (proba==True) & (persist==True):
            #persisting data so we don't require loading all the data twice
            input_data_flattened=input_data_flattened.persist()

        #apply the classification
        print('   predicting...') 
        out_class = model.predict(input_data_flattened)

        # Mask out NaN or Inf values in results
        if clean==True:        
            out_class = da.where(da.isfinite(out_class),out_class, 0)

        # Reshape when writing out
        out_class = out_class.reshape(len(y), len(x))

        # stack back into xarray
        output_xr = xr.DataArray(out_class, coords={
                        "x": x,
                        "y": y},
                        dims=["y", "x"])

        output_xr = output_xr.to_dataset(name='Predictions')
        
        if proba == True:
            print("   probabilities...")
            out_proba = model.predict_proba(input_data_flattened)

            #convert to %
            out_proba = da.max(out_proba, axis=1) * 100.0

            if clean==True:
                out_proba = da.where(da.isfinite(out_proba), out_proba, 0)

            out_proba = out_proba.reshape(len(y), len(x))

            out_proba = xr.DataArray(out_proba, coords={"x": x,"y": y}, dims=["y", "x"])
            output_xr['Probabilities'] = out_proba
        
        if return_input==True:
            print("   input features...")            
            # unflatten the input_data_flattened array and append
            # to the output_xr containin the predictions
            arr = input_xr.to_array()
            stacked = arr.stack(z=['x', 'y'])

            # handle multivariable output
            output_px_shape = ()
            if len(input_data_flattened.shape[1:]):
                output_px_shape = input_data_flattened.shape[1:]

            output_features = input_data_flattened.reshape((len(stacked.z), *output_px_shape))

            # set the stacked coordinate to match the input
            output_features = xr.DataArray(output_features, coords={'z': stacked['z']},
                                     dims=['z', *['output_dim_' + str(idx) for
                                                  idx in range(len(output_px_shape))]]).unstack()

            #convert to dataset and rename arrays
            output_features = output_features.to_dataset(dim='output_dim_0')
            data_vars = list(input_xr.data_vars)
            output_features = output_features.rename({i:j for i,j in zip(output_features.data_vars, data_vars)})
            
            #merge with predictions
            output_xr = xr.merge([output_xr, output_features], compat='override')

        return assign_crs(output_xr, str(crs))