import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import great_circle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from shapely.geometry import Point


def subsample_neighbors(meta: pd.DataFrame, location_i: list, delta_lat_lon: float, idx_i: int =-1) -> pd.DataFrame:
    """
    Subsamples tiles in the vicinity of the observed location. 
    Look within +- one delta_lat_lon from the specified loc_i, and return a subframe containing only the points in that region.


    -------
    :param meta: Initial pandas dataframe of Core-DEM. It has four numerical coluns -- min_val (altitude in meters), max_val (altitude in meters), centre_lat (decimal latitude), and centre_lon (decimal longitude)
    :param location_i: Tuple of latitude and logitude of the observed location
    :param delta_lat_lon: Radius in degrees that we will inspect for subsampling neighbors. It shoudl be bigger than the distance of neighboring tiles.
    :param idx_i: Index of the entry of 'meta' whose location overlaps with 'location_i'.

    -------
    :returns: Subframe of 'meta' that contains only the tiles within specified vicinity of loc_i
    
    """
    subframe = meta[meta.centre_lat>location_i[0]-delta_lat_lon]
    subframe = subframe[subframe.centre_lat<location_i[0]+delta_lat_lon]
    subframe = subframe[subframe.centre_lon>location_i[1]-delta_lat_lon]
    subframe = subframe[subframe.centre_lon<location_i[1]+delta_lat_lon]
    subframe = subframe[subframe.index!=idx_i]
    return subframe

def optimal_neighbor(meta: pd.DataFrame, location_i:list, altitude_i:float, idx_i:int, delta_lat_lon:float, destination_location:list, distance:float, coefficients:np.array, prior_indices:list=[], destintaion_idx:int=-1) -> tuple[int,list,float,float,float]:
    """
    Finds the optimal neighbor, i.e., the one that has an optimal tradeoff between the distance reduction to the destination point, and the altitude change. 
    
    -------
    :param meta: Initial pandas dataframe of Core-DEM. It has four numerical coluns -- min_val (altitude in meters), max_val (altitude in meters), centre_lat (decimal latitude), and centre_lon (decimal longitude)
    :param location_i: Tuple of latitude and logitude of the observed location
    :param altitude_i: Altitude in meters for 'location_i'
    :param idx_i: Index of the entry of 'meta' whose location overlaps with 'location_i'.
    :param delta_lat_lon: Radius in degrees that we will inspect for subsampling neighbors. It shoudl be bigger than the distance of neighboring tiles.
    :param destination_location: Tuple of latitude and logitude of the destination location
    :param distance: Straight-line distance between 'location_i' and 'destination_location'
    :param coefficients: Vector of optimization coefficients. First one giving weight for distance reduction, and the second one for the altitude change, when hopping from location_i to the neighboring tile.
    :param prior_indices: List of all the indices (from 'meta') that were already assigned to this path
    :param destintaion_idx: Index of the entry of 'meta' whose location overlaps with 'destination_location'.

    -------
    :returns: s_idx - index of the selectd optimal neighbor, locations[idx] - its geo-coordinates, subframe.min_val[s_idx] - its altitude, distances[idx] - straight line distance to destination, and alt_differences[idx] - altitude change between location_i and a selected neighbor.
    
    """
    subframe = subsample_neighbors(meta, location_i, delta_lat_lon, idx_i)
    for idx in prior_indices:
        subframe = subframe[subframe.index!=idx]
    if destintaion_idx in subframe.index: # this means, if the destination point is within the neighborhood
        return destintaion_idx, destination_location, subframe.min_val[destintaion_idx], distance, np.abs(altitude_i-subframe['min_val'][destintaion_idx])
    alt_differences = np.abs(altitude_i-subframe['min_val'].values)
    locations = subframe[['centre_lat','centre_lon']].values
    distances = [great_circle(destination_location, locations[i]).km for i in range(len(locations))]
    dist_differences = [distance-d for d in distances]
    search_values = np.array((dist_differences, alt_differences))
    idx = np.argmin(coefficients.dot(search_values))
    s_idx = subframe.index[idx]
    return s_idx, locations[idx], subframe.min_val[s_idx], distances[idx], alt_differences[idx]

def find_path(meta:pd.DataFrame, idx1:int, idx2:int, location1:list, location2:list, altitude1:float, delta_lat_lon:float, coefficients:np.array) -> tuple[bool, list, list, list]:
    """
    Find the path between two locations. In case the destination is not reached after 100 steps, search terminates, returning corresponding flag.

    -------
    :param meta: Initial pandas dataframe of Core-DEM. It has four numerical coluns -- min_val (altitude in meters), max_val (altitude in meters), centre_lat (decimal latitude), and centre_lon (decimal longitude)
    :param idx1: Index of the entry of 'meta' whose location overlaps with 'location1', i.e., the initial location.
    :param idx2: Index of the entry of 'meta' whose location overlaps with 'location2', i.e., the destination location.
    :param location1: Geo-coordinates of the initial locaiton
    :param location2: Geo-coordinates of the destination locaiton
    :param altitude1: Altitude in meters at 'location1'
    :param delta_lat_lon: Radius in degrees that we will inspect for subsampling neighbors. It shoudl be bigger than the distance of neighboring tiles.
    :param coefficients: Vector of optimization coefficients. First one giving weight for distance reduction, and the second one for the altitude change, when hopping from location_i to the neighboring tile.
    
    -------
    :returns: arrived - flag indicating if the destination is reached within 100 steps. paths - list of locations forming the path between the initial and destination locaton, including the two. distances2d - list of the legths of each part of the path (sequential values in 'path'). alt_differences - list of altitude differences between the sequential location at the path. 
        
    """
    prior_indices = [idx1]
    paths = [location1]
    distances2d = [0]
    altitudes = [altitude1]
    alt_differences = [0]

    distance = great_circle(paths[-1], location2).km

    arrived = False
    counter = 0
    while not arrived:
        if idx1 == idx2: # If these are the same site, we are already there
            arrived = True
        elif counter > 100:
            break

        else:
            # find new site
            idx1, location, altitude, distance, alt_diff = optimal_neighbor(meta, paths[-1], altitudes[-1], idx1, delta_lat_lon, location2, distance, coefficients, prior_indices, idx2)
            prior_indices.append(idx1)
            # compute the traversed path:
            d = great_circle(location, paths[-1]).km
            distances2d.append(d)
            altitudes.append(altitude)
            paths.append(location)
            alt_differences.append(alt_diff)

            counter += 1

    return arrived, paths, distances2d, alt_differences

def load_main_rivers(AOI_BBOX, AOI_CRS, DATASETS):
    '''
    Loading the rivers data within a specified boundingbox.
    See https://www.kaggle.com/code/ceeluna/checkpoint-2-pixels-palms-and-the-past#%F0%9F%8C%BF-Pixels,-Palms,-and-the-Past:-A-Hunt-for-Acre%E2%80%99s-Hidden-Sites
    '''
    RIVERS_PATH = DATASETS["hydrorivers"]["source"]
    
    rivers = gpd.read_file(RIVERS_PATH, bbox=(AOI_BBOX[0], AOI_BBOX[1], AOI_BBOX[2], AOI_BBOX[3]))
    rivers = rivers.to_crs(AOI_CRS)
    
    main_rivers = rivers[rivers["ORD_FLOW"].astype(float) <= 5].copy()

    return main_rivers


def optimal_neighbor_river(rivers_coords:np.array, meta:pd.DataFrame, location_i:list, altitude_i:float, idx_i:int, delta_lat_lon:float, distance:float, coefficients:np.array, prior_indices:list=[])->tuple[int,np.array,float,float,float]:
    """
    Finds an easy neighbor, iteratively, until arrive close enough to some river.
        
    -------
    :param rivers_coords: Array of geo-coordinates for river points 
    :param meta: Initial pandas dataframe of Core-DEM. It has four numerical coluns -- min_val (altitude in meters), max_val (altitude in meters), centre_lat (decimal latitude), and centre_lon (decimal longitude)
    :param location_i: Tuple of latitude and logitude of the observed location
    :param altitude_i: Altitude in meters for 'location_i'
    :param idx_i: Index of the entry of 'meta' whose location overlaps with 'location_i'.
    :param delta_lat_lon: Radius in degrees that we will inspect for subsampling neighbors. It shoudl be bigger than the distance of neighboring tiles.
    :param distance: straight-line distance from location_i to the nearest river point
    :param coefficients: Vector of optimization coefficients. First one giving weight for distance reduction, and the second one for the altitude change, when hopping from location_i to the neighboring tile.
    :param prior_indices: List of all the indices (from 'meta') that were already assigned to this path

    -------
    :returns: s_idx - index of the selectd optimal neighbor, locations[idx] - its geo-coordinates, subframe.min_val[s_idx] - its altitude, distances[idx] - straight line distance to the nearest river point, and alt_differences[idx] - altitude change between location_i and a selected neighbor.
    
    """

    subframe = meta[meta.centre_lat>location_i[0]-delta_lat_lon]
    subframe = subframe[subframe.centre_lat<location_i[0]+delta_lat_lon]
    subframe = subframe[subframe.centre_lon>location_i[1]-delta_lat_lon]
    subframe = subframe[subframe.centre_lon<location_i[1]+delta_lat_lon]
    subframe = subframe[subframe.index!=idx_i]
    for idx in prior_indices:
        subframe = subframe[subframe.index!=idx]

    # Extend the radius until some river is there
    river_flag = False
    multiplier = 2
    while not river_flag:
        subrivers = rivers_coords[rivers_coords[:,0]>location_i[0]-multiplier*delta_lat_lon]
        subrivers = subrivers[subrivers[:,0]<location_i[0]+multiplier*delta_lat_lon]
        subrivers = subrivers[subrivers[:,1]>location_i[1]-multiplier*delta_lat_lon]
        subrivers = subrivers[subrivers[:,1]<location_i[1]+multiplier*delta_lat_lon]
        multiplier += 1
        if len(subrivers):
            river_flag = True

    alt_differences = np.abs(altitude_i-subframe['min_val'].values)
    locations = subframe[['centre_lat','centre_lon']].values

    distances = [np.min([great_circle(locations[i], river).km for river in subrivers]) for i in range(len(locations))]
    dist_differences = [distance-d for d in distances]
    search_values = np.array((dist_differences, alt_differences))
    idx = np.argmin(coefficients.dot(search_values))
    s_idx = subframe.index[idx]

    return s_idx, locations[idx], subframe.min_val[s_idx], distances[idx], alt_differences[idx]

def path_to_river(index:int, locations:np.array, altitudes:list, indices:list, rivers_coords:np.array, meta:pd.DataFrame, delta_lat_lon:float, coefficients:np.array) -> tuple[list, list, list]:
    '''
    Get the easiest path to the nearby river.

    -------
    :param index: Index of the intial loaciton (absolute index of settlement, not the one from meta dataframe)     
    :param locations: Array of geo-coordinates for settlements
    :param altitudes: List of altitudes for corresponding locaiton in 'locations' 
    :param indices: List of indices of meta tiles that correspond to the locations of settlements   
    :param rivers_coords: Array of geo-coordinates for river points     
    :param meta: Initial pandas dataframe of Core-DEM. It has four numerical coluns -- min_val (altitude in meters), max_val (altitude in meters), centre_lat (decimal latitude), and centre_lon (decimal longitude)
    :param delta_lat_lon: Radius in degrees that we will inspect for subsampling neighbors. It shoudl be bigger than the distance of neighboring tiles.
    :param coefficients: Vector of optimization coefficients. First one giving weight for distance reduction, and the second one for the altitude change, when hopping from location_i to the neighboring tile.

    -------
    :returns: paths - list of locations forming the path between the initial and destination locaton (river), including the two. distances2d - list of the legths of each part of the path (sequential values in 'path'). alt_differences - list of altitude differences between the sequential location at the path. 
    '''
    distances2d = []
    prior_indices = []
    altitude_differences = []
    location_i = locations[index]
    path = [location_i]
    altitude_i = altitudes[index]
    idx_i = indices[index]
    distance = np.min([great_circle(location_i, river).km for river in rivers_coords])

    # Subsample nearby rivers
    river_flag = False
    while not river_flag:
        subrivers = rivers_coords[rivers_coords[:,0]>location_i[0]-delta_lat_lon]
        subrivers = subrivers[subrivers[:,0]<location_i[0]+delta_lat_lon]
        subrivers = subrivers[subrivers[:,1]>location_i[1]-delta_lat_lon]
        subrivers = subrivers[subrivers[:,1]<location_i[1]+delta_lat_lon]
        
        # If any of them is within the reach, no need to search further
        if len(subrivers):
            river_flag = True
            distances = [great_circle(location_i, river).km for river in subrivers]
            idx = np.argmin(distances)
            path.append(subrivers[idx])
            distances2d.append(distances[idx])
    
        else:
            prior_indices.append(idx_i)
            idx_i, location_i, altitude_i, distance, alt_diff = optimal_neighbor_river(rivers_coords, meta, location_i, altitude_i, idx_i, delta_lat_lon, distance, coefficients, prior_indices)
            path.append(location_i)
            distances2d.append(distance)
            altitude_differences.append(alt_diff)

    return path, distances2d, altitude_differences


DELTA_LAT_LON  = 0.2  # This is the value that would respect Core-DEM tile distancing constraints
DIST_DIFF_COEF = -1   # This one should be negative. It is safe to keep it -1, and only tune the value of ALT_DIFF_COEF (if needed)
ALT_DIFF_COEF  = 0.01 # This should stay positive, but optimal value might vary depending on the graph
COEFFICIENTS   = np.array([DIST_DIFF_COEF, ALT_DIFF_COEF])


DATASETS = {
    "hydrorivers": {
        "type": "file",
        "source": "/kaggle/input/hydrorivers-dataset/HydroRIVERS.gdb",
        "description": (
            " HydroRivers database. Lehner, B., Grill G. (2013): Global river hydrography and network routing: baseline data and new approaches to study the world’s large river systems. Hydrological Processes, 27(15): 2171–2186. Data is available at www.hydrosheds.org."
        )
    },}


# Latitudes and longitudes extracted in the previous notebook
locations = np.load(r'/kaggle/input/perudata2/Peru_locations_matrix.npy')
(max_lat, max_lon) = locations.max(0)
(min_lat, min_lon) = locations.min(0)
max_lat += 1
min_lat -= 1
max_lon += 1
min_lon -= 1

print('Maximum and minimum longitude and latitudes of the collected settlements:')
print('locations max: ',locations.max(0))
print('locations min: ',locations.min(0))

# Graph adjecency matrix, constructed in hte previous notebook
adjecency_matrix = np.load(r'/kaggle/input/perudata2/Peru_adjecency_matrix.npy')

# Elevation data from TOM
file_path = r"/kaggle/input/perudata2/metadata.parquet"
meta = pd.read_parquet(file_path,columns=['min_val','max_val','centre_lat','centre_lon'])
meta = meta[meta.max_val>0]
meta = meta[meta.centre_lat<max_lat]
meta = meta[meta.centre_lat>min_lat]
meta = meta[meta.centre_lon<max_lon]
meta = meta[meta.centre_lon>min_lon]
print('\n Dataframe of the Core-DEM satelite imagery. The selected columns contain min and max altitude of each tile, \nas well as the geo-coodinates of the middle of the image')
print(meta.head())
print(meta.shape)



alphas = meta['min_val']/meta['min_val'].max()
alphas[alphas<0]=0
fig = plt.figure(figsize=[16, 7])
gs = fig.add_gridspec(1, 2)

ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
ax2 = fig.add_subplot(gs[0, 1])
ax1.add_feature(cfeature.COASTLINE)
ax1.add_feature(cfeature.OCEAN)
ax1.add_feature(cfeature.RIVERS)

ax1.scatter(meta['centre_lon'],meta['centre_lat'],alpha=alphas,s=4,color='k', transform=ccrs.Geodetic())
ax1.scatter(locations[:,1],locations[:,0],s=5,color='r',edgecolor='r', transform=ccrs.Geodetic())
order = np.argsort(adjecency_matrix.mean(1))
ax2.imshow(adjecency_matrix[order][:,order],cmap='binary')
ax1.spines[['right', 'top','left','bottom']].set_visible(False)
ax1.set_xlim(min_lon,max_lon)
ax1.set_ylim(min_lat,max_lat)
ax2.set_xticks([])
ax2.set_yticks([])
ax1.set_title("Land Elevation", fontsize=12)
ax2.set_title("Adjacency Matrix", fontsize=12)
plt.show()


# Assign each camp to the nearest DEM pixel, and take the corresponding lat, lon, and altitude
locations_nearest = 0.*locations
altitudes = []
indices = []
for i in range(len(locations)):
    loc_i = locations[i]
    subframe = subsample_neighbors(meta, loc_i, DELTA_LAT_LON)
    locs = subframe[['centre_lat','centre_lon']].values
    distances = [great_circle(loc, loc_i).km for loc in locs]
    idx = np.argmin(distances)
    locations_nearest[i] += locs[idx]
    altitudes.append(subframe['min_val'].values[idx])
    indices.append(subframe.index[idx])

np.save('Peru_altitudes.npy',np.array(altitudes))


paths_dict = {}
dist2d_matrix = 0*adjecency_matrix
alt_matrix = 0*adjecency_matrix
for i in range(len(adjecency_matrix)):
    for j in range(i):
        if adjecency_matrix[i,j]:
            idx1, idx2 =  indices[i],indices[j]
            if idx1 == idx2:
                dist2d_matrix[i,j] = 1
                dist2d_matrix[j,i] = 1
            else:
                location1 = locations_nearest[i]
                location2 = locations_nearest[j]
                altitude1, altitude2 = altitudes[i], altitudes[j]
                arrived, paths, distances2d, alt_differences = find_path(meta, idx1, idx2, location1, location2, altitude1, DELTA_LAT_LON, COEFFICIENTS)
                if arrived:
                    paths_dict[(j,i)] = np.array(paths)
                    alt_matrix[i,j] = np.sum(alt_differences)
                    alt_matrix[j,i] = alt_matrix[i,j]
                    dist2d_matrix[i,j] = 1+np.sum(distances2d)
                    dist2d_matrix[j,i] = dist2d_matrix[i,j]

np.save('Peru_distance_matrix.npy',dist2d_matrix)
np.save('Peru_alt_diff_matrix.npy',alt_matrix)


max_distance = np.max(dist2d_matrix)
dist2d_matrix[dist2d_matrix>0] = (max_distance - dist2d_matrix[dist2d_matrix>0]+1)/max_distance
fig, ax = plt.subplots(figsize=(7,7))
order = np.argsort(dist2d_matrix.sum(1))
plt.imshow(dist2d_matrix[order][:,order],cmap='Reds')
ax.spines[['right', 'top','left','bottom']].set_visible(False)
plt.title("Distances matrix")
plt.xticks([])
plt.yticks([])
plt.show()


i,j = 108,171
plt.figure(figsize=(5,5))
plt.scatter(locations_nearest[i,1],locations_nearest[i,0],color='r')
plt.scatter(locations_nearest[j,1], locations_nearest[j,0],color='r')
plt.plot(paths_dict[(i,j)][:,1],paths_dict[(i,j)][:,0],color='k',linewidth=0.5)
plt.show()


fig, ax = plt.subplots(figsize=(7,7))
legend_flag = True
xlimits = (-78,-70)
ylimits = (-17,-8)
subframe = meta[meta.centre_lon>xlimits[0]]
subframe = subframe[subframe.centre_lon<xlimits[1]]
subframe = subframe[subframe.centre_lat<ylimits[1]]
subframe = subframe[subframe.centre_lat>ylimits[0]]
alphas1 = subframe['min_val']/subframe['min_val'].max()
alphas1[alphas1<0]=0
plt.scatter(subframe['centre_lon'],subframe['centre_lat'],alpha=alphas1,s=9,color='g',label='Terrain')
for _ in range(100):
    i = np.random.choice(len(dist2d_matrix))
    j = np.random.choice(np.where(adjecency_matrix[i]>0)[0])
    if i > j:
        j,i=i,j
    if (i,j) in paths_dict.keys():
        plt.scatter(locations_nearest[i,1],locations_nearest[i,0],color='r')
        if legend_flag:
            plt.scatter(locations_nearest[j,1], locations_nearest[j,0],color='r',label='Site')
            plt.plot(paths_dict[(i,j)][:,1],paths_dict[(i,j)][:,0],color='k',linewidth=0.5,label='Path')
            legend_flag = False
        else:
            plt.scatter(locations_nearest[j,1], locations_nearest[j,0],color='r')
            plt.plot(paths_dict[(i,j)][:,1],paths_dict[(i,j)][:,0],color='k',linewidth=0.5)
ax.set_xlim(xlimits)
ax.set_ylim(ylimits)
ax.spines[['right', 'top','left','bottom']].set_visible(False)
plt.title("Word-distances")
plt.xticks([])
plt.yticks([])
plt.legend()
plt.show()



AOI_CRS = "EPSG:4326"
AOI_BBOX = [locations_nearest.min(0)[1], locations_nearest.min(0)[0], locations_nearest.max(0)[1], locations_nearest.max(0)[0]]   # xmin, ymin   # xmax, ymax

main_rivers = load_main_rivers(AOI_BBOX, AOI_CRS, DATASETS)

fig, ax = plt.subplots(figsize=(10, 10))
plt.scatter(meta['centre_lon'], meta['centre_lat'], alpha=alphas, s=9, color='g')
main_rivers.plot(ax=ax, linewidth=0.5, color="k")
ax.set_xlim(AOI_BBOX[0],AOI_BBOX[2])
ax.set_ylim(AOI_BBOX[1],AOI_BBOX[3])
plt.xticks([])
plt.yticks([])
plt.title('Main Rivers')
plt.show()


all_coords = []

for geom in main_rivers.geometry:
    if geom.geom_type == "MultiLineString":
        for line in geom.geoms:
            coords = list(line.coords)
            all_coords.extend(coords)
    elif geom.geom_type == "LineString":
        coords = list(geom.coords)
        all_coords.extend(coords)

rivers_coords = []
min_distances = []
for coords in all_coords:
    coords = tuple(reversed(coords))
    all_dist = [great_circle(coords, loc).km for loc in locations_nearest]
    if np.min(all_dist) < 100:
        rivers_coords.append(coords)
rivers_coords = np.array(rivers_coords)



plt.figure(figsize=(7,7))
plt.scatter(rivers_coords[:,1],rivers_coords[:,0],s=.01, label='River')
legend_flag = True
for _ in range(70):
    index = np.random.randint(len(indices))
    path, distances2d, altitude_differences = path_to_river(index, locations_nearest, altitudes, indices, rivers_coords, meta, DELTA_LAT_LON, COEFFICIENTS)
    path = np.array(path)
    if legend_flag:
        plt.scatter(path[0,1],path[0,0],color='r', label='Site')
        plt.plot(path[:,1],path[:,0],color='k',linewidth=0.5, label='Path')
    else:
        plt.scatter(path[0,1],path[0,0],color='r')
        plt.plot(path[:,1],path[:,0],color='k',linewidth=0.5)
    legend_flag = False
plt.xlim(-78,-70)
plt.ylim(-16,-11)
plt.xticks([])
plt.yticks([])
plt.legend()
plt.show()


river_distances = []
river_alt = []
for index in range(len(indices)):
    path, distances2d, altitude_differences = path_to_river(index, locations_nearest, altitudes, indices, rivers_coords, meta, DELTA_LAT_LON, COEFFICIENTS)
    path = np.array(path)
    river_distances.append(1+np.sum(distances2d))
    river_alt.append(np.sum(altitude_differences))

river_distances = np.array(river_distances)
plt.figure(figsize=(10,3))
plt.hist(river_distances, bins=100)
plt.title('Distances to river')
plt.ylabel('Count')
plt.xlabel('Distance in km')
plt.show()
np.save('Peru_river_distances.npy', river_distances)
river_alt = np.array(river_alt)
np.save('Peru_river_alt.npy', river_alt)

