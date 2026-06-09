# FOR IMPORTING IN OTHER SCRIPTS, EXIT IF IMPORTED
import sys
if "downloadLibraries" in globals().keys(): exit(0)
!pip install earthengine-api
!pip install geopandas
!pip install openai
!pip install typing
!pip install geemap


import ee
import geemap
import geopandas as gpd

import openai

from typing import Optional, Tuple, List, Union, Dict
# from dotenv import load_dotenv  # For Local .env
from kaggle_secrets import UserSecretsClient  # For Kaggle
import json
import os


def stripZDimension(geometry):
    """Remove altitude (Z-dimension) from a GeoJSON Polygon geometry."""
    coords = geometry['coordinates']
    flat_coords = [[[lon, lat] for lon, lat, *_ in ring] for ring in coords]
    return ee.Geometry.Polygon(flat_coords)


def convertDMS(latDMS, lonDMS):
    """Convert DMS format coordinates (like 2Â°11'19"S) to decimal degrees."""
    def parse(dms):
        degrees, minutes, seconds, direction = dms
        decimal = degrees + minutes / 60 + seconds / 3600
        return -decimal if direction in ('S', 'W') else decimal

    return parse(latDMS), parse(lonDMS)

def verbosity(verbose=1, min=1, max=None) -> bool:
    """Set verbosity level for debugging."""
    return verbose >= min and (verbose <= max if max is not None else True)

def loadSecretImport(secret, kaggle):
    # For Kaggle
    if kaggle:
        return UserSecretsClient().get_secret(secret)

    # For Local
    load_dotenv(override=True)
    return os.getenv(secret)


class InvalidParameter(Exception):
    def __init__(self, message="", warning=False):
        super().__init__(f"Invalid Parameter {'Warning' if warning else 'Exception'}:", message)

class OpenAIExc(Exception):
    def __init__(self, message="", warning=False):
        super().__init__(f"OpenAI {'Warning' if warning else 'Exception'}:", message)

class GeoDataExc(Exception):
    def __init__(self, message="", warning=False):
        super().__init__(f"GeoData {'Warning' if warning else 'Exception'}:", message)


def throwError(condition, error, warning=False):
    if condition:
        if warning: 
            print(str(error))
        else:
            raise error


class GeoData:
    """
    A general-purpose handler for Earth Engine satellite/geospatial datasets.
    Allows loading AOI via GeoJSON file or lat/lon + box, loading image collections,
    and generating composites with selected bands. \n
    Accepted Initial Parameters:
    - initialVerbose: Value to control verbosity during initialization, Defaults to 1. Possible values: 0 (silent), 1 (slight-verbose), 2 (verbose). Will be clipped to 0-2, if any else provided
    - errorPriority: Bool to control whether to raise an error or print a warning, Defaults to False. If True, will raise an error, if False, will print a warning
    - geoJSON: path to a GeoJSON file defining the Area of Interest (AOI), Defaults to None
    - aoiCenter: tuple of (latitude, longitude) to define a center point for AOI, Defaults to None
    - aoiSizeKM: tuple of (widthKM, heightKM) to define the size of the AOI box, Defaults to (5km x 5km)
    - dateRange: tuple of (start_date, end_date) in format YYYY-MM-DD to filter collections by date, Defaults to ("2023-01-01", "2023-12-31"). Will only filter is AOI is set initially.
    - loadDefaultImages: Bool to control whether to load default collections and composites after AOI is set, Defaults to True. If False, no default collections/composites will be loaded, regardless of initial AOI Set.

    If `geoJSON` or `aoicCenter` == None, then no AOI is set initially. After initialization, you can set the AOI using `.loadAoi()` method.
    If both `geoJSON` and `aoiCenter` are provided, the `geoJSON` will be used to set the AOI.

    Will load following Collections by default, if AOI set initially:
    - COPERNICUS/S1_GRD (Sentinel-1: collectionName)
    - COPERNICUS/S2_SR_HARMONIZED (Sentinel-2: collectionName)
    - LARSE/GEDI/GEDI04_A_002 (GEDI: collectionName)
    - NASA/NASADEM_HGT/001 (NASADEM: collectionName)
    - JAXA/ALOS/AW3D30/V3_2 (DSM: collectionName) \n

    Will load following composites by default, if AOI set initially:
    - Sentinel-1 Composite (bands: 'VV', 'VH') (from Sentinel-1)
    - Sentinel-2 Composite (bands: 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12') (from Sentinel-2)
    - VV-VH Composite (bands: 'VV', 'VH' -> (VV - VH)) (from Sentinel-1)
    - VV/VH Composite (bands: 'VV', 'VH' -> (VV / VH)) (from Sentinel-1)
    - RGB Composite (bands: 'B4', 'B3', 'B2') (from Sentinel-2)
    - SWIR Composite (bands: 'B12', 'B8', 'B4') (from Sentinel-2)
    - NDVI Composite (bands: 'B8', 'B4' -> ((B8 - B4) / (B8 + B4))) (from Sentinel-2) \n


    Can access existing:
    - Collections via `.getCollection(collectionName)`
    - Composites via `.getComposite(compositeName)`
    """

    def __init__(self, **kwargs):
        # Initializations
        self.aoi: Optional[ee.Geometry] = None
        self.collections = {}
        self.composites = {}
        self.errorPriority = kwargs.get('errorPriority', False)  # If True, raise error, if False, print warning

        initialVerbose = kwargs.get('initialVerbose', 1)

        geoJSON = kwargs.get('geoJSON', None)
        aoiCenter = kwargs.get('aoiCenter', None)
        aoiSizeKM = kwargs.get('aoiSizeKM', (5, 5))  # Default to 5km x 5km
        dateRange = kwargs.get('dateRange', ("2023-01-01", "2023-12-31"))
        loadDefaultImages = kwargs.get('loadDefaultImages', True)

        if geoJSON:
            if verbosity(initialVerbose, 2):
                print(f"Loading AOI from GeoJSON: {geoJSON}")
            self.aoi = self.loadAoi(geoJSONPath=geoJSON)

        if not self.aoi:
            if aoiCenter:
                if verbosity(initialVerbose, 2):
                    print(f"Loading AOI from center: {aoiCenter} with size {aoiSizeKM} km")
                self.aoi = self.loadAoi(aoiCenter=aoiCenter, aoiSizeKm=aoiSizeKM)
            else:
                if verbosity(initialVerbose, 1):
                    print("No AOI set initially. Use loadAoi() to set it later.")
    
        if self.aoi and loadDefaultImages:
            if verbosity(initialVerbose, 1):
                print("Loading default collections for AOI...")
            self.loadCollection("COPERNICUS/S1_GRD", "Sentinel-1", dateRange, bands=["VV", "VH"], filters=[ee.Filter.eq('instrumentMode', 'IW'), ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'), ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')])
            self.loadCollection("COPERNICUS/S2_SR_HARMONIZED", "Sentinel-2", dateRange, bands=["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"], filters=[ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10)])
            self.loadCollection("LARSE/GEDI/GEDI04_A_002", "GEDI", dateRange)
            self.loadCollection("NASA/NASADEM_HGT/001", "NASADEM", dateRange)
            self.loadCollection("JAXA/ALOS/AW3D30/V3_2", "DSM", dateRange) 


            sent1Comp = self.loadComposite("Sentinel-1", "Sentinel-1 Composite", reducer='median')
            sent2Comp = self.loadComposite("Sentinel-2", "Sentinel-2 Composite", reducer='median')

            self.loadComposite("Sentinel-2", "RGB Composite", bands=['B4', 'B3', 'B2'], reducer='median')
            self.loadComposite("Sentinel-2", "SWIR Composite", bands=['B12', 'B8', 'B4'], reducer='median')

            self.composites['VV-VH Composite'] = sent1Comp.select('VV').subtract(sent1Comp.select('VH')).rename('VV-VH Composite')
            self.composites['VV/VH Composite'] = sent1Comp.select('VV').divide(sent1Comp.select('VH')).rename('VV/VH Composite')
            self.composites['NDVI Composite'] = self.getComposite("Sentinel-2 Composite").normalizedDifference(['B8', 'B4']).rename('NDVI')
            
            
            

    def loadAoi(self, *, geoJSONPath: Optional[str] = None, aoiCenter: Optional[Tuple[float, float]] = None, aoiSizeKm: Tuple[float, float] = (5, 5)) -> Union[ee.Geometry, None]:
        """
        Load AOI either from a GeoJSON file or from a center point with width/height.

        If none given, returns None
        If both given, geoJSONPath will be used to load AOI, if valid else Center/Size, if valid, else returns None.

        Stores the AOI in class itself.

        Parameters:
        - geoJSONPath: path to GeoJSON file, Defaults to None
        - center: (lat, lon) tuple, Defaults to None
        - sizeKm: (width_km, height_km) of box around center, Defaults to (5, 5)

        Returns:
        - ee.Geometry.Rectangle object
        - None, if Error
        """
        loadByGeoJSON = True

        # ~ Error Handling
        validateChoice = geoJSONPath is not None or aoiCenter is not None
        if not validateChoice:
            throwError(True, InvalidParameter("Can't load AOI. Either geoJSONPath or aoiCenter must be provided.", warning = not self.errorPriority), warning = not self.errorPriority)
            return None
        
        validateGeoJSONType = type(geoJSONPath) in [str, bytes, os.PathLike, int]
        validateCenter = type(aoiCenter) == tuple and len(aoiCenter) == 2 and all(isinstance(coord, (int, float)) for coord in aoiCenter)
        if not validateGeoJSONType and not validateCenter:
            throwError(True, InvalidParameter("Can't load AOI. Either geoJSONPath (File Path) or aoiCenter (Tuple(float, float)) must be valid.", warning = not self.errorPriority), warning = not self.errorPriority)
            return None
        
        if not validateGeoJSONType: loadByGeoJSON = False
        else: 
            validateGeoJSONPath = os.path.exists(geoJSONPath)
            if not validateGeoJSONPath: loadByGeoJSON = False

        if not loadByGeoJSON:
            validateSize = type(aoiSizeKm) == tuple and len(aoiSizeKm) == 2 and all(isinstance(dim, (int, float)) for dim in aoiSizeKm)
            if not validateSize:
                throwError(True, InvalidParameter("Can't load AOI. geoJSONPath no exists or aoiSizeKm must be a tuple of (width_km, height_km) with valid numbers.", warning = not self.errorPriority), warning = not self.errorPriority)
                return None
        

        try:
            if loadByGeoJSON:
                with open(geoJSONPath) as geoJSONFile:
                    geoJSON = json.load(geoJSONFile)
                self.aoi = ee.Geometry(geoJSON['features'][0]['geometry'])

            else:
                lat, lon = aoiCenter
                halfH = aoiSizeKm[1] * 500 / 110540  # degrees lat
                halfW = aoiSizeKm[0] * 500 / 111320  # degrees lon
                self.aoi = ee.Geometry.Rectangle([
                    lon - halfW, lat - halfH,
                    lon + halfW, lat + halfH
                ])
        except Exception as e:
            throwError(True, GeoDataExc(f"Failed to load AOI: {str(e)}", warning = not self.errorPriority), warning = not self.errorPriority)

        return self.aoi


    def loadCollection(self, source: str, collectionName: str, dateRange: Tuple[str, str], bands: Optional[List[str]] = None, filters: Optional[List[ee.Filter]] = None) -> Union[ee.Image, None]:
        """
        Load an image collection from Earth Engine.
        Stores the collection in .collections[collectionName] attribute.
        To get the collection later, use `.getCollection(collectionName)` method.

        Parameters:
        - source: collection ID (e.g., 'COPERNICUS/S2_SR_HARMONIZED')
        - dateRange: tuple of (start_date, end_date) in format YYYY-MM-DD
        - bands: optional list of bands to select
        - filters: optional list of ee.Filter objects

        Returns:
        - ee.ImageCollection
        - None, if Error
        """
        if not self.aoi:
            throwError(True, GeoDataExc("AOI must be set, using `.loadAoi()` before loading collections.", warning = not self.errorPriority), warning = not self.errorPriority)

        try: 
            collection = ee.ImageCollection(source)
        except Exception as e:
            throwError(True, GeoDataExc(f"Collection Load Failed: {source} not exist", warning = not self.errorPriority), warning = not self.errorPriority)
            return None
        
        
        try:
            collection.filterBounds(self.aoi).filterDate(*dateRange) # .clip(self.aoi)
        except Exception as e:
            throwError(True, GeoDataExc(f"Collection Load Failed: Either AOI (ee.Geometry) or Date Range (startDate: YYYY/MM/DD, endDate: YYYY/MM/DD) is invalid", warning = not self.errorPriority), warning = not self.errorPriority)
            return None

        try:
            if filters:
                for f in filters:
                    collection = collection.filter(f)
        except Exception as e:
            throwError(True, GeoDataExc(f"Collection Load Failed: Invalid filter(s) provided: {str(e)}", warning = not self.errorPriority), warning = not self.errorPriority)
            return None

        try: 
            if bands:
                collection = collection.select(bands)
        except Exception as e:
            validBands = ee.Image(collection.first()).bandNames().getInfo()
            throwError(True, GeoDataExc(f"Collection Load Failed: Invalid bands provided. Correct Bands: {validBands}", warning = not self.errorPriority), warning = not self.errorPriority)
            return None
        
        self.collections[collectionName] = collection

        return collection
    

    def getCollection(self, collectionName: str) -> Union[ee.ImageCollection, None]:        
        """
        Get an image collection by name.

        Parameters:
        - collectionName: name of the collection to retrieve

        Returns:
        - ee.ImageCollection
        - None, if collection not found
        """
        if collectionName not in self.collections:
            throwError(True, GeoDataExc(f"Collection '{collectionName}' not found. Load it first using `.loadCollection()`.", warning = not self.errorPriority), warning = not self.errorPriority)
            return None
        return self.collections[collectionName]


    def loadComposite(self, collection: Union[ee.ImageCollection, str], compositeName: str, bands: list = None, reducer: str = 'median') -> Union[ee.Image, None]:
        """
        Load composite image from collection.

        Parameters:
        - collection: Either ee image collection or a collection name (string) from self.collections. Collection Name must be loaded before, if passed.
        - compositeName: name of the composite to be saved in self.composites and renamed
        - bands: list of bands to include in the composite. Defaults to None, which means all bands will be used.
        - reducer: 'median', 'mean', 'mosaic', etc. Defaults to 'median'.

        Returns:
        - ee.Image
        - None, if Error
        """
        # Error Handling
        validReducer = reducer in ['median', 'mean', 'mosaic']
        if not validReducer:
            throwError(True, GeoDataExc(f"Invalid reducer: {reducer}. Supported reducers: 'median', 'mean', 'mosaic'.", warning = not self.errorPriority), warning = not self.errorPriority)
            return None
        
        validCollectionType = type(collection) in [ee.ImageCollection, str]
        if not validCollectionType: 
            throwError(True, GeoDataExc(f"Invalid collection type: {type(collection)}. Must be ee.ImageCollection or a string (collection name).", warning = not self.errorPriority), warning = not self.errorPriority)
            return None
        if isinstance(collection, str):
            if collection not in self.collections:
                throwError(True, GeoDataExc(f"Collection Name '{collection}' not found. Load it first using `.loadCollection()`, or directly pass a Collection.", warning = not self.errorPriority), warning = not self.errorPriority)
                return None
            collection = self.collections[collection]

        validBandsType = type(bands) in [list, type(None)]
        if not validBandsType:
            throwError(True, GeoDataExc(f"Invalid bands type: {type(bands)}. Must be a list of strings or None.", warning = not self.errorPriority), warning = not self.errorPriority)
            return None
        validBands = ee.Image(collection.first()).bandNames().getInfo()
        bands = bands if bands else validBands
        for band in bands:
            if band not in validBands:
                throwError(True, GeoDataExc(f"Invalid band '{band}' in bands list. Valid bands: {validBands}", warning = not self.errorPriority), warning = not self.errorPriority)
                return None

        if reducer == 'median':
            composite = collection.median().select(bands).clip(self.aoi)
        elif reducer == 'mean':
            composite = collection.mean().select(bands).clip(self.aoi)
        elif reducer == 'mosaic':
            composite = collection.mosaic().select(bands).clip(self.aoi)

        self.composites[compositeName] = composite
        return composite
        
    
    def getComposite(self, compositeName: str) -> Union[ee.Image, None]:
        """
        Get a composite image by name.

        Parameters:
        - compositeName: name of the composite to retrieve

        Returns:
        - ee.Image
        - None, if composite not found
        """
        if compositeName not in self.composites:
            throwError(True, GeoDataExc(f"Composite '{compositeName}' not found. Load it first using `.loadComposite()`.", warning = not self.errorPriority), warning = not self.errorPriority)
            return None
        return self.composites[compositeName]

    def mapComposite(self, composites: Union[ee.Image, str, List[ee.Image], List[str]], bands: Union[List[str], List[List[str]]] = None, layerNames: str = "Composite X", minV: Union[float, List[float]] = [0], maxV: Union[float, List[float]] = [0], palettes = None, static: bool = False):
        """
        Maps a Composite
        Note: Need to use `static=True` in Function Call for Kaggle Notebook

        Parameters:
        - composite: Either ee.Image or a composite name (string) from self.composites, or List of any 1 or mixed. Composite Names must be loaded before, if passed.
        - bands: optional list of list of bands to select for plotting, where each sub-list is for each Composite. If None, will use all bands in the composite.
        - layerNames: Names of each of the composites/layers to be displayed in the map
        - minV: minimum value for visualization, can be a single value or a list of values for each composite
        - maxV: maximum value for visualization, can be a single value or a list of values for each composite
        - palettes: optional list of palettes for each composite, if None, will use default palette
        - static: bool to control whether to use geemap.foliumap (True) or geemap (False). Defaults to False. If True, will use geemap.foliumap, if False, will use geemap.
        """
        if static: import geemap.foliumap as geemap
        else: import geemap

        compositeMap = geemap.Map(add_google_map=False)
        compositeMap.centerObject(self.aoi, zoom=10)

        if type(composites) != list: composites = [composites]  # Ensure composite is a list 
        if type(bands) != list: bands = [bands]  # Ensure bands is a list 
        if type(palettes) != list: palettes = [palettes]  # Ensure palette is a list 
        if type(minV) != list: minV = [minV]  # Ensure minV is a list 
        if type(maxV) != list: maxV = [maxV]  # Ensure maxV is a list 
        if type(layerNames) != list: layerNames = [layerNames]  # Ensure palette is a list 

        if len(composites) != len(bands) != len(palettes) != len(minV) != len(maxV) != len(layerNames):
            throwError(True, GeoDataExc(f"Length mismatch: composites ({len(composites)}), bands ({len(bands)}), palettes ({len(palettes)}), minV ({len(minV)}), maxV ({len(maxV)})", warning = not self.errorPriority), warning = not self.errorPriority)
            return None

        for composite, band, palette, miV, maV, layerName in zip(composites, bands, palettes, minV, maxV, layerNames):
            if type(composite) == str:
                if not self.getComposite(composite): continue
                composite = self.getComposite(composite)

            if not bands: bands = [composite.bandNames().getInfo()]

            if palette: compositeMap.addLayer(composite, {'bands': band, 'min': miV, 'max': maV, 'palette': palette}, layerName)
            else: compositeMap.addLayer(composite, {'bands': band, 'min': miV, 'max': maV}, layerName)
 
        return compositeMap

    def plotCompositeGraph(
        self,
        composites: Union[ee.Image, str, List[Union[ee.Image, str]]],
        bands: Optional[Union[List[str], List[List[str]]]] = None,
        layerNames: Union[str, List[str]] = "Composite X",
        minV: Union[float, List[float]] = 0,
        maxV: Union[float, List[float]] = 3000,
        palettes: Optional[Union[str, List[str]]] = None,
        scale: int = 10,
        saveFile: Optional[str] = None
    ):
        """
        Plots 2D graph of composite images (single or multi-band) using Matplotlib.

        Parameters:
        - composites: List or single ee.Image or composite name (from self.composites).
        - bands: List of bands or list of list of bands for each composite.
        - layerNames: Name(s) of layer(s) to display on plots.
        - minV: Minimum display value(s).
        - maxV: Maximum display value(s).
        - palettes: Optional palette(s), ignored in multiband (RGB) mode.
        - scale: Target resolution in meters (applies reprojection before sampling).
        - saveFile: Name of File to save in Plot. Default None. If none, dont save
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if not isinstance(composites, list): composites = [composites]
        if not isinstance(bands, list): bands = [bands] * len(composites)
        if not isinstance(layerNames, list): layerNames = [layerNames] * len(composites)
        if not isinstance(minV, list): minV = [minV] * len(composites)
        if not isinstance(maxV, list): maxV = [maxV] * len(composites)
        if palettes and not isinstance(palettes, list): palettes = [palettes] * len(composites)
        elif not palettes: palettes = [None] * len(composites)

        for comp, bnd, name, miV, maV, pal in zip(composites, bands, layerNames, minV, maxV, palettes):
            if isinstance(comp, str): comp = self.getComposite(comp)
            image = comp.select(bnd) if bnd else comp

            # Apply projection rescaling
            proj = image.select(0).projection().atScale(scale)
            image = image.reproject(proj)

            sampled = image.sampleRectangle(region=self.aoi)

            if isinstance(bnd, list) and len(bnd) == 3:
                arr = np.stack([np.array(sampled.get(b).getInfo()) for b in bnd], axis=-1)
            else:
                arr = np.array(sampled.get(bnd if isinstance(bnd, str) else bnd[0]).getInfo())

            coords = self.aoi.bounds().coordinates().getInfo()[0]
            lon = np.linspace(coords[0][0], coords[2][0], arr.shape[1])
            lat = np.linspace(coords[2][1], coords[0][1], arr.shape[0])
            extent = [lon[0], lon[-1], lat[-1], lat[0]]

            plt.figure(figsize=(8, 6))
            if isinstance(bnd, list) and len(bnd) == 3:
                plt.imshow(np.clip(arr / maV, 0, 1), extent=extent, origin='lower')
            else:
                plt.imshow(arr, cmap=pal or 'viridis', vmin=miV, vmax=maV, extent=extent, origin='lower')
                plt.colorbar(label="Value")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.title(name)
            plt.grid(False)
            if saveFile: plt.savefig(f"{saveFile}.png", dpi=300, bbox_inches='tight')
            plt.show()
        
    def exportImage(self, image: ee.Image, exportName: str, folder: str, scale: int = 10):
        """
        Export an EE image to Google Drive.

        Parameters:
        - image: ee.Image to export
        - exportName: export task name and filename prefix
        - folder: Google Drive folder
        - scale: spatial resolution in meters
        """
        if not self.aoi:
            raise RuntimeError("AOI must be set before export.")

        task = ee.batch.Export.image.toDrive(
            image=image,
            description=exportName,
            folder=folder,
            fileNamePrefix=exportName,
            region=self.aoi,
            scale=scale,
            maxPixels=1e13
        )
        task.start()
        print(f"ðŸš€ Export started: {exportName}")


class OpenAI:
    """
    Wrapper for OpenAI GPT interactions.
    Handles both authentication and prompting.
    """

    def __init__(self, apiKey):
        self.client = openai.OpenAI(api_key=apiKey)

    def prompt(self, role: str = "user", prompt: str = "Hi!", images: List[str] = [], model="gpt-4o"):
        """Send contextual chat messages to GPT model."""
        if not isinstance(role, str) or not isinstance(prompt, str):
            raise InvalidParameter("Roles and prompt must be Strings.")
        if not type(images) in [list]:
            raise InvalidParameter("Images must be a list of strings.")
        
        return self.client.chat.completions.create(
            model=model,
            messages=[{'role': role, 
                       'content': [
                           {"type": "text", "text": prompt},
                           *([{"type": "image_url", "image_url": {"url": img}} for img in images])
                        ]}]
        ).choices[0].message.content



# FOR IMPORTING IN OTHER SCRIPTS, EXIT IF IMPORTED
if "isImported" in globals().keys(): sys.exit(0)


# Load Secrets
kaggle = True # True if kaggle environment else false (for local)

def loadSecret(secret):
    # For Kaggle
    if kaggle:
        return UserSecretsClient().get_secret(secret)

    # For Local
    load_dotenv(override=True)
    return os.getenv(secret)

openAPIKey = loadSecret('OpenAIKey')
googleDriveFolder = loadSecret('GDriveFolder')
eeProjectName = loadSecret('eeProjectName')


# Authenticate EE
scopes = [
    "https://www.googleapis.com/auth/earthengine",
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/devstorage.full_control",
    "https://www.googleapis.com/auth/cloud-platform"
]

ee.Authenticate(scopes=scopes)
ee.Initialize(project=eeProjectName)


openAI = OpenAI(openAPIKey)  # Initialize OpenAI wrapper


openAI.prompt() # Send 'Hi!' (default) to ChatGPT Model 4o, with role 'user' (default)
# Should respond with "Hello! How can I assist you today?" or similar greeting.


openAI.prompt("system", "You are a GeoAnalyzer", model="gpt-3.5-turbo-0125") 
# Send 'You are a GeoAnalyzer' to ChatGPT Model 3.5, with role 'system'


# Test GeoData Area with Random Data
geoData = GeoData(
    initialVerbose=2,  # Set verbosity level for debugging
    errorPriority=True,  # Raise errors instead of warnings
    # geoJSON="path/to/your/aoi.geojson",  # Path to GeoJSON file
    aoiCenter=(-0.9500, -66.6333),  # Example center point (latitude, longitude)
    aoiSizeKM=(5, 5),  # Size of AOI box in kilometers
    dateRange=("2023-01-01", "2023-12-31")  # Date range for collections
)

geoData.getCollection("Sentinel-2") # Get Sentinel-2 collection
rgbComposite = geoData.getComposite('RGB Composite') # Build a test Composite - RGB Composite from Sentinel-2
ndviComposite = geoData.getComposite('NDVI Composite') # Build a test Composite - NDVI Composite from Sentinel-2

# geoData.exportImage(rgbComposite, "Test_Validate_Export_RGBComposite", googleDriveFolder, scale=10)  # Export the composite image to Google Drive
# Will Print "ðŸš€ Export started: Test_Validate_Export_RGBComposite" and start the export task. Check the Success of Task in Task Manager of Earth Engine


# Test Mapping RGB & NDVI Composite
geoData.mapComposite([rgbComposite, ndviComposite], bands=[None, None], minV=[0, 0], maxV=[3000, 1], layerNames=["Test RGB Composite", "Test NDVI Composite"], palettes=[None, ["white", "green"]], static=True)


# Test Plotting NDVI Composite
geoData.plotCompositeGraph(ndviComposite, bands=["NDVI"], layerNames="Test NDVI Composite", minV=0, maxV=1, palettes="Greens", scale=10)

