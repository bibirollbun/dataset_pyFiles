%%capture
# To remove uncessary logs

isImported = True # For Importing: Can set to any value

%run /kaggle/usr/lib/digital_archaeology_general_code/__notebook__.ipynb
 
# Will take time to import: Will give Exception `SystemExit: 0`, which means, it successfully Imported!


# Now you can use the 2 Classes!
testGeoData = GeoData()
testOpenAI = OpenAI("")


# Imports
import ee
import base64

ifKaggle = True # If the Environment is Kaggle
   
# Secrets
openAPIKey = loadSecretImport("openAPIKey", ifKaggle)
googleDriveFolder = loadSecretImport("googleDriveFolder", ifKaggle)
eeProjectName = loadSecretImport("eeProjectName", ifKaggle)


# Authenticate EE
scopes = [
    "https://www.googleapis.com/auth/earthengine",
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/devstorage.full_control",
    "https://www.googleapis.com/auth/cloud-platform"
]

ee.Authenticate(scopes=scopes)
ee.Initialize(project=eeProjectName)


# Initialize GeoData Area with Random Data
geoData = GeoData(
    initialVerbose=2,  # Set verbosity level for debugging
    errorPriority=True,  # Raise errors instead of warnings
    # geoJSON="path/to/your/aoi.geojson",  # Path to GeoJSON file
    aoiCenter=(-0.9500, -66.6333),  # Example center point (latitude, longitude)
    aoiSizeKM=(5, 5),  # Size of AOI box in kilometers
    dateRange=("2023-01-01", "2023-12-31")  # Date range for collections
)


# Get Existing Collections
sentinel1 = geoData.getCollection("Sentinel-1") # Get Sentinel-1 collection
sentinel2 = geoData.getCollection("Sentinel-2") # Get Sentinel-2 collection
dem = geoData.getCollection("NASADEM") # Get DEM (Digital Elevation Model - from NASA) collection
# ... And More, like GEDI, DSM

# Get Existing Composites
rgbComposite = geoData.getComposite('RGB Composite') # Get a RGB Composite from Sentinel-2
ndviComposite = geoData.getComposite('NDVI Composite') # Get a NDVI Composite from Sentinel-2
vv_vhComposite = geoData.getComposite('VV-VH Composite') # Get a VV-VH Composite from Sentinel-1
# ... And More, like VV/VH, SWIR

# Get a new Collection - (SourceName, NameOfCollectionToStoreIn, DataRange)
geoData.loadCollection("NASA/ECOSTRESS/LST", "EcoStress", ("2021-01-01", "2021-12-31")) # For Example, ECOSTRESS (High-res Thermal Data)

# Plot a Composite in 2d Graph
geoData.plotCompositeGraph(ndviComposite, bands=["NDVI"], layerNames="Test NDVI Composite", minV=0, maxV=1, palettes="Greens", scale=10)

# Map the Composite on a Map
geoData.mapComposite([rgbComposite, ndviComposite], bands=[None, None], minV=[0, 0], maxV=[3000, 1], layerNames=["Test RGB Composite", "Test NDVI Composite"], palettes=[None, ["white", "green"]], static=True)

# Export Composite
geoData.exportImage(rgbComposite, "Test_Validate_Export_RGBComposite", googleDriveFolder, scale=10)  # Export the composite image to Google Drive
# Will Print "ðŸš€ Export started: Test_Validate_Export_RGBComposite" and start the export task. Check the Success of Task in Task Manager of Earth Engine


# Initialize the OpenAI wrapper with API Key
openAI = OpenAI(openAPIKey)

print("Default Prompt: ", openAI.prompt()) # Send 'Hi!' to ChatGPT Model 4o, with no Images, and Role (user) (default values)
# Should respond with "Hello! How can I assist you today?" or similar greeting.

# Send a Prompt (You are a GeoAnalyzer) to a Model (GPT 3.5) with Images (None), and Role (system)
openAI.prompt("system", "You are a GeoAnalyzer", model="gpt-3.5-turbo-0125") 



# Sentinel-2 Scene ID (of NHamini Regions)
sentinel2SceneID = "20240229T142711_20240229T142951_T20MRC"
# Parse details from Scene ID
mgrsTile = "20MRC"
startDate = "2024-02-29"
endDate = "2024-02-30"  # Usually just Â±1 day for safety
sceneAoiCenter = (-2.3024350249980539, -59.809799014369805)  # SceneAOI
sceneAoiSizeKM = (20, 20)

# Initialize GeoData on Scene
sceneGeoData = GeoData(
    initialVerbose=2,
    errorPriority=True,
    aoiCenter=sceneAoiCenter,
    aoiSizeKM=sceneAoiSizeKM,
    dateRange=(startDate, endDate)
)


# Get Collection of this SceneID
sceneCollection = sceneGeoData.loadCollection("COPERNICUS/S2_SR_HARMONIZED", "SceneSent2Collection", dateRange=(startDate, endDate), filters=[ee.Filter.eq("MGRS_TILE", mgrsTile), ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10)])
sceneComposite = sceneGeoData.loadComposite(sceneCollection, "Scene Composite", reducer='median')
# Load Scene NDVI
sceneNDVI = sceneComposite.normalizedDifference(['B8', 'B4']).rename('NDVI')

# Plot Scene NDVI
sceneGeoData.plotCompositeGraph(
    composites=sceneNDVI,
    bands=["NDVI"],
    layerNames="Scene NDVI",
    minV=0,
    maxV=1,
    scale=100,
    palettes="Greens",
    saveFile="/kaggle/working/cp1-scene-ndvi" # Save File
)


# Ask Gpt about this NDVI
prompt = "Describe this surface feature from Sentinel-2 in plain English. What might explain its shape?"
role = "user"
model = "gpt-4o"

imageBase64 = base64.b64encode(open("cp1-scene-ndvi.png", "rb").read()).decode("utf-8")
imageURL = f"data:image/png;base64,{imageBase64}"

openAI.prompt(role=role, prompt=prompt, images=[imageURL], model="gpt-4o")


# Final Print
print("Scene ID:", sentinel2SceneID)
print("Model:", model)


# AOI (Area of Interest) Latitude/Longitude with Width/Height
aoiCenter = (-2.06456 , -60.14787) 
aoiSizeKM = (40, 40)

# Initialize GeoData on AOI
geoData = GeoData(
    initialVerbose=2,  # Set verbosity level for debugging
    errorPriority=True,  # Raise errors instead of warnings
    aoiCenter=aoiCenter,  # Example center point (latitude, longitude)
    aoiSizeKM=aoiSizeKM,  # Size of AOI box in kilometers
    dateRange=("2024-01-01", "2024-12-31")  # Date range for collections
)

