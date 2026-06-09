!pip install geopandas kaggle


import os
import requests
import math

import geopandas as gpd

from flask import Flask, request, jsonify, Response


# Return the Universal Transverse Mercater (UTM) Coordinate Reference System (CRS) for for the latitude 
# and longitude given.  UTM coordinate systems don't distort due to latitude
def get_utm_crs(lat, lng):
    if abs(lat) > 90.0:
        print(f"Not a valid latitude: {lat}!")
        return

    if abs(lng) > 180.0:
        print(f"Not a valid longitude: {lng}!")

    zone_is_south = lat < 0.0
    zone = math.ceil((lng + 180.0) / 6.0)

    crs = "+proj=utm +zone="
    crs = crs + str(zone)
    if lat < 0.0:
        crs = crs + " +south"

    crs = crs + " +datum=WGS84 +units=m +no_defs"

    return crs


import json
from shapely.geometry import shape

# We access the MAPBOX_TOKEN variable differently on Kaggle
def running_in_kaggle() -> bool:
    """
    Heuristics that are true in Kaggle notebooks:
    - Special directories exist (/kaggle/input, /kaggle/working)
    - Env var KAGGLE_KERNEL_RUN_TYPE is set
    - The kaggle_secrets module is available
    """
    try:
        if os.path.isdir("/kaggle/input") and os.path.isdir("/kaggle/working"):
            return True
        if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
            return True
        import kaggle_secrets  # noqa: F401  (only exists in Kaggle)
        return True
    except Exception:
        return False

def read_mapbox_directions(o_lat, o_lng, d_lat, d_lng):

    if running_in_kaggle():
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        mapbox_token = user_secrets.get_secret("MAPBOX_TOKEN")
    else:
        mapbox_token = os.environ['MAPBOX_TOKEN']
        
    url = f"https://api.mapbox.com/directions/v5/mapbox/driving/{o_lng}%2C{o_lat}%3B{d_lng}%2C{d_lat}?alternatives=false&annotations=maxspeed&geometries=geojson&language=en&overview=full&steps=true&access_token={mapbox_token}"

    # Store the full Directions response for further processing later.
    response = requests.get(url)
    mapbox_data = response.json()

    # Retrieve the GEOJSON portion of the directions, which describes every geographic point along the route.
    # Then, create a GeoDataFrame of the route, which will enable measuring changes in heading, or curviness.
    subdata = mapbox_data['routes'][0]['geometry']

    crs = get_utm_crs(o_lat, o_lng)
    geom = shape(subdata)
    geo_df = gpd.GeoDataFrame([1], geometry=[geom], crs="EPSG:4326")

    # Convert coordinates into local UTM so latitude isn't distorted
    geo_df = geo_df.to_crs(crs)

    return mapbox_data, geo_df


dfw_data, dfw_df = read_mapbox_directions(32.759295, -97.503393, 32.903862, -96.973375)
pikes_data, pikes_df = read_mapbox_directions(38.900889, -104.973437, 38.840776, -105.043467)
idaho_data, idaho_df = read_mapbox_directions(45.817674, -116.251336, 45.81141, -116.258672)


dfw_df.plot(figsize=(8,8))
print("DFW")


pikes_df.plot(figsize=(8,8))
print("Pikes Peak Highway")


idaho_df.plot(figsize=(8,8))
print("Idaho")


import math
import numpy as np

def angle_p1p2p3(p1, p2, p3):
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]], dtype=float)
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]], dtype=float)

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0.0 or n2 == 0.0:
        return 0.0  # undefined; (perhaps use value of None instead?)
        
    # robust atan2 using cross and dot
    cross = v1[0]*v2[1] - v1[1]*v2[0]           # scalar "2D cross"
    dot   = float(np.dot(v1, v2))
    theta = math.degrees(math.atan2(abs(cross), dot))

    return theta



import numpy as np
import pandas as pd

def calculate_linestring_curvature(linestring):
    """
    Calculates the curvature for each vertex of a Shapely LineString.
    Returns a list of curvature values. The first and last points
    have a curvature of 180 as they cannot form a triplet.
    """
    coords = np.array(linestring.coords)
    if len(coords) < 3:
        return [180] * len(coords)

    curvatures = [180]
    for i in range(1, len(coords) - 1):
        p1 = coords[i - 1]
        p2 = coords[i]
        p3 = coords[i + 1]

        ang = angle_p1p2p3(p1, p2, p3)

        curvatures.append(ang)

    curvatures.append(180)
    return curvatures



dfw_curvs = dfw_df['geometry'].apply(calculate_linestring_curvature)
pikes_curvs = pikes_df['geometry'].apply(calculate_linestring_curvature)
idaho_curvs = idaho_df['geometry'].apply(calculate_linestring_curvature)


dfw_curvs_ser = pd.Series(dfw_curvs[0])
pikes_curvs_ser = pd.Series(pikes_curvs[0])
idaho_curvs_ser = pd.Series(idaho_curvs[0])


dfw_curvs_ser.describe()


pikes_curvs_ser.describe()


idaho_curvs_ser.describe()


def calculate_curviness(geo_df):
    geo_curvs = geo_df['geometry'].apply(calculate_linestring_curvature)
    geo_curvs_ser = pd.Series(geo_curvs[0])
    raw_curviness = geo_curvs_ser.mean()

    raw_curviness = max(raw_curviness, 160.0)
    curviness = (180.0 - raw_curviness) / (180.0 - 160.0)

    return curviness


# Testing...

print("DFW route curviness        :",  calculate_curviness(dfw_df))
print("Pike's Peak route curviness:",  calculate_curviness(pikes_df))
print("Idaho route curviness      :",  calculate_curviness(idaho_df))




