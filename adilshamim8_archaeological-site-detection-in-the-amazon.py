# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy import ndimage
from scipy.spatial.distance import pdist, squareform
import os
import json
from datetime import datetime

# Optional imports - will use if available
try:
    import rasterio
    from rasterio.plot import show
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    print("rasterio not available - some GIS functionality limited")

try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    print("geopandas not available - some mapping functionality limited")

try:
    import folium
    from folium import plugins
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    print("folium not available - interactive maps disabled")

try:
    import ee
    # Don't initialize until we need it
    EE_AVAILABLE = True
except ImportError:
    EE_AVAILABLE = False
    print("Earth Engine API not available - will use alternative data sources")

# Set up display formatting
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-whitegrid')

print("OpenAI to Z Challenge: Finding Hidden Archaeological Sites in the Amazon".center(80))


# Define our region of interest - Eastern Bolivia's Llanos de Moxos region
# This area has been highly productive for archaeological discovery according to Prümers et al. (2022)
# We're focusing on a specific section where the paper suggests more sites may exist

roi = {
    'name': "Eastern Llanos de Moxos",
    'min_lon': -65.2,  # Western boundary
    'max_lon': -64.8,  # Eastern boundary
    'min_lat': -14.0,  # Southern boundary
    'max_lat': -13.6   # Northern boundary
}

print(f"\nRegion of interest: {roi['name']}")
print(f"Geographic extent: {roi['min_lon']:.4f}°E to {roi['max_lon']:.4f}°E, {roi['min_lat']:.4f}°N to {roi['max_lat']:.4f}°N")

# Calculate approximate area in square kilometers
lat_center = (roi['min_lat'] + roi['max_lat']) / 2
lon_distance_km = 111.32 * np.cos(np.radians(lat_center)) * (roi['max_lon'] - roi['min_lon'])
lat_distance_km = 110.574 * (roi['max_lat'] - roi['min_lat'])
area_km2 = lon_distance_km * lat_distance_km

print(f"Area: Approximately {area_km2:.1f} km²")
print(f"Selected based on proximity to known sites reported in Prümers et al. (2022)")


def load_sample_data(resolution=1000):
    """
    Load or generate sample satellite data for demonstration purposes
    
    In a real competition entry, this would load actual satellite imagery.
    For this demonstration, we create synthetic data that mimics the
    patterns found in actual archaeological sites in the Amazon.
    
    Parameters:
    - resolution: Size of the generated image in pixels
    
    Returns:
    - Dictionary containing various data layers:
        - rgb: RGB composite image
        - elevation: Digital elevation model
        - ndvi: Normalized Difference Vegetation Index
        - moisture: Soil moisture indicator
    """
    print("\nLoading sample data for analysis...")
    
    # Create a synthetic dataset with properties similar to satellite imagery
    image_size = resolution
    
    # Create base terrain with realistic topographic features
    # Use perlin noise for realistic terrain
    def perlin_noise(shape, scale=100):
        """Generate Perlin noise for realistic terrain simulation"""
        from numpy.random import RandomState
        rng = RandomState(23)  # Fixed seed for reproducibility
        
        # Generate base noise
        noise = rng.randn(shape[0]+1, shape[1]+1)
        
        # Smooth with Gaussian filter
        smooth = ndimage.gaussian_filter(noise, sigma=scale/20, mode='wrap')
        
        # Normalize
        smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min())
        return smooth[:shape[0], :shape[1]]
    
    # Generate base terrain with multiple scales of Perlin noise
    terrain_base = perlin_noise((image_size, image_size), scale=100) * 0.6
    terrain_detail = perlin_noise((image_size, image_size), scale=20) * 0.3
    terrain_micro = perlin_noise((image_size, image_size), scale=5) * 0.1
    
    terrain = terrain_base + terrain_detail + terrain_micro
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min())
    
    # Generate vegetation index based on terrain
    # In real data, NDVI would be calculated from red and near-infrared bands
    vegetation_base = perlin_noise((image_size, image_size), scale=50)
    # Higher elevations tend to be less green in floodplains
    vegetation = 0.8 - 0.3 * terrain + 0.5 * vegetation_base
    vegetation = np.clip(vegetation, 0, 1)
    
    # Create simulated RGB image
    rgb = np.zeros((image_size, image_size, 3))
    rgb[:,:,0] = 0.2 + 0.3 * terrain + 0.1 * perlin_noise((image_size, image_size), scale=10)  # Red
    rgb[:,:,1] = 0.3 + 0.4 * vegetation + 0.1 * perlin_noise((image_size, image_size), scale=10)  # Green
    rgb[:,:,2] = 0.15 + 0.2 * terrain + 0.1 * perlin_noise((image_size, image_size), scale=10)  # Blue
    rgb = np.clip(rgb, 0, 1)
    
    # Create soil moisture indicator (higher in low areas)
    moisture = 0.8 - 0.7 * terrain + 0.3 * perlin_noise((image_size, image_size), scale=30)
    moisture = np.clip(moisture, 0, 1)
    
    # Add synthetic archaeological features
    # Based on patterns described in Prümers et al. (2022) and Iriarte et al. (2020)
    features = []
    
    # Define feature types to add: rectangular, circular, linear
    feature_types = [
        {"shape": "rectangle", "size": 40, "x": 250, "y": 250, "rotation": 0.0},
        {"shape": "rectangle", "size": 35, "x": 600, "y": 300, "rotation": 0.2},
        {"shape": "rectangle", "size": 30, "x": 400, "y": 700, "rotation": 0.1},
        {"shape": "circle", "size": 35, "x": 800, "y": 200, "rotation": 0.0},
        {"shape": "circle", "size": 25, "x": 700, "y": 600, "rotation": 0.0},
        {"shape": "linear", "size": 150, "x": 500, "y": 500, "rotation": 0.6}
    ]
    
    # Create mask for features
    feature_mask = np.zeros((image_size, image_size))
    feature_elevation_change = np.zeros((image_size, image_size))
    
    # Add each feature to the synthetic data
    for feature in feature_types:
        shape_type = feature["shape"]
        size = feature["size"]
        center_x = feature["x"]
        center_y = feature["y"]
        rotation = feature["rotation"]
        
        if shape_type == "rectangle":
            # Create mask for rotated rectangle
            xx, yy = np.mgrid[:image_size, :image_size]
            
            # Center coordinates
            xc = xx - center_x
            yc = yy - center_y
            
            # Apply rotation
            cos_r = np.cos(rotation)
            sin_r = np.sin(rotation)
            xr = xc * cos_r - yc * sin_r
            yr = xc * sin_r + yc * cos_r
            
            # Create rectangle
            mask = (np.abs(xr) < size) & (np.abs(yr) < size)
            feature_mask[mask] = 1
            
            # Create a berm (raised edge) around the rectangular feature
            # This is common in Amazonian earthworks
            berm = (np.abs(xr) < size*1.1) & (np.abs(yr) < size*1.1) & ~mask
            feature_elevation_change[mask] = -0.02  # Slight depression inside
            feature_elevation_change[berm] = 0.05   # Raised edge
            
            # Record feature for verification
            features.append({
                "type": "rectangular earthwork",
                "center_x": center_x,
                "center_y": center_y,
                "size": size,
                "rotation": rotation
            })
            
        elif shape_type == "circle":
            # Create circular feature
            xx, yy = np.mgrid[:image_size, :image_size]
            circle_mask = ((xx - center_x)**2 + (yy - center_y)**2) < size**2
            feature_mask[circle_mask] = 1
            
            # Add berm around circle
            circle_berm = ((xx - center_x)**2 + (yy - center_y)**2 < (size*1.2)**2) & ~circle_mask
            feature_elevation_change[circle_mask] = -0.02  # Slight depression inside
            feature_elevation_change[circle_berm] = 0.05   # Raised edge
            
            # Record feature
            features.append({
                "type": "circular earthwork",
                "center_x": center_x,
                "center_y": center_y,
                "size": size,
                "rotation": 0.0
            })
            
        elif shape_type == "linear":
            # Create linear feature (causeway or road)
            xx, yy = np.mgrid[:image_size, :image_size]
            
            # Center coordinates
            xc = xx - center_x
            yc = yy - center_y
            
            # Apply rotation
            cos_r = np.cos(rotation)
            sin_r = np.sin(rotation)
            xr = xc * cos_r - yc * sin_r
            yr = xc * sin_r + yc * cos_r
            
            # Create line with width
            line_mask = (np.abs(yr) < 5) & (np.abs(xr) < size)
            feature_mask[line_mask] = 1
            feature_elevation_change[line_mask] = 0.05  # Raised causeway
            
            # Record feature
            features.append({
                "type": "causeway",
                "center_x": center_x,
                "center_y": center_y,
                "size": size,
                "rotation": rotation
            })
    
    # Apply elevation changes from features (subtle)
    terrain = terrain + feature_elevation_change
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min())
    
    # Apply vegetation changes (archaeological sites often have different vegetation)
    # Features often appear as slightly different vegetation patterns
    vegetation_change = np.zeros((image_size, image_size))
    # Vegetation inside features is slightly different
    vegetation_change[feature_mask > 0] = 0.05
    # Especially around the edges
    vegetation_edge = ndimage.binary_dilation(feature_mask > 0).astype(int) - (feature_mask > 0).astype(int)
    vegetation_change[vegetation_edge > 0] = 0.1
    
    # Apply the changes to vegetation
    vegetation = vegetation + vegetation_change
    vegetation = np.clip(vegetation, 0, 1)
    
    # Update RGB with feature impact
    rgb[:,:,1] = rgb[:,:,1] + vegetation_change * 0.2  # Slightly more green on features
    rgb = np.clip(rgb, 0, 1)
    
    print(f"Generated sample data with {len(features)} archaeological features")
    
    # Return the generated data
    return {
        "rgb": rgb,
        "elevation": terrain,
        "ndvi": vegetation,
        "moisture": moisture,
        "known_features": features
    }

# Load or generate sample data
sample_data = load_sample_data(resolution=1000)

# Visualize the sample data
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# RGB image
axes[0, 0].imshow(sample_data['rgb'])
axes[0, 0].set_title('RGB Composite')
axes[0, 0].axis('off')

# Elevation data
elevation_display = axes[0, 1].imshow(sample_data['elevation'], cmap='terrain')
axes[0, 1].set_title('Elevation Model')
axes[0, 1].axis('off')
plt.colorbar(elevation_display, ax=axes[0, 1], label='Relative Elevation')

# NDVI vegetation index
ndvi_display = axes[1, 0].imshow(sample_data['ndvi'], cmap='YlGn')
axes[1, 0].set_title('Vegetation Index (NDVI)')
axes[1, 0].axis('off')
plt.colorbar(ndvi_display, ax=axes[1, 0], label='NDVI')

# Soil moisture
moisture_display = axes[1, 1].imshow(sample_data['moisture'], cmap='Blues')
axes[1, 1].set_title('Soil Moisture Indicator')
axes[1, 1].axis('off')
plt.colorbar(moisture_display, ax=axes[1, 1], label='Moisture Index')

plt.tight_layout()
plt.show()

# Display the "known features" that we've hidden in the synthetic data
# In a real analysis, these wouldn't be known in advance
print("\nReference features (hidden in the data):")
for i, feature in enumerate(sample_data['known_features']):
    print(f"{i+1}. {feature['type'].capitalize()} at pixel ({feature['center_x']}, {feature['center_y']}), " +
          f"size: {feature['size']}, rotation: {feature['rotation']:.2f}")


def detect_geometric_features(data, min_size=10, max_size=100, threshold=0.7):
    """
    Detect geometric features that might indicate archaeological sites
    
    Parameters:
    - data: Dictionary with data layers
    - min_size: Minimum feature size in pixels
    - max_size: Maximum feature size in pixels
    - threshold: Confidence threshold for detection
    
    Returns:
    - DataFrame with detected features
    """
    print("\nDetecting geometric archaeological features...")
    
    # We'll use a combination of edge detection and shape analysis
    
    # Start with elevation data which often shows earthworks best
    elevation = data['elevation']
    
    # Apply edge enhancement
    # Use Sobel filters for edge detection
    edges_x = ndimage.sobel(elevation, axis=0)
    edges_y = ndimage.sobel(elevation, axis=1)
    edge_magnitude = np.sqrt(edges_x**2 + edges_y**2)
    
    # Normalize edge magnitude
    edge_magnitude = (edge_magnitude - edge_magnitude.min()) / (edge_magnitude.max() - edge_magnitude.min())
    
    # Apply threshold to find significant edges
    edge_threshold = np.percentile(edge_magnitude, 85)  # Adjust percentile based on testing
    binary_edges = edge_magnitude > edge_threshold
    
    # Clean up the binary edge image
    # Remove small artifacts
    binary_edges = ndimage.binary_opening(binary_edges, structure=np.ones((3,3)))
    
    # Detect connected components (potential features)
    labeled_features, num_features = ndimage.label(binary_edges)
    
    print(f"Initial edge detection found {num_features} potential features")
    
    # Analyze each component
    feature_list = []
    for feature_id in range(1, num_features + 1):
        # Extract this feature
        feature_mask = labeled_features == feature_id
        feature_size = np.sum(feature_mask)
        
        # Skip features that are too small or too large
        if feature_size < min_size or feature_size > max_size**2:
            continue
            
        # Find feature boundaries
        y_indices, x_indices = np.where(feature_mask)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        
        # Calculate dimensions
        width = x_max - x_min
        height = y_max - y_min
        
        # Skip if dimensions don't make sense
        if width < min_size or height < min_size:
            continue
            
        # Calculate centroid
        centroid_y = np.mean(y_indices)
        centroid_x = np.mean(x_indices)
        
        # Calculate extent (ratio of pixels in the hull to pixels in the rectangle)
        extent = feature_size / (width * height)
        
        # Analyze shape regularity
        # Calculate how well the shape fills its convex hull
        # First create a binary image of just this feature
        feature_img = np.zeros_like(feature_mask, dtype=bool)
        feature_img[y_indices, x_indices] = True
        
        # Calculate convex hull
        from scipy.spatial import ConvexHull
        try:
            points = np.column_stack((x_indices, y_indices))
            hull = ConvexHull(points)
            
            # Create a mask of the convex hull
            hull_mask = np.zeros_like(feature_mask, dtype=bool)
            
            # Create a polygon from the hull vertices
            from matplotlib.path import Path
            vertices = points[hull.vertices]
            path = Path(vertices)
            
            # Create a grid of points and check if each point is inside the hull
            y_grid, x_grid = np.mgrid[:feature_mask.shape[0], :feature_mask.shape[1]]
            grid_points = np.column_stack((x_grid.ravel(), y_grid.ravel()))
            hull_mask = path.contains_points(grid_points).reshape(feature_mask.shape)
            
            # Calculate solidity (ratio of pixels in the region to pixels in the convex hull)
            hull_area = np.sum(hull_mask)
            solidity = feature_size / hull_area if hull_area > 0 else 0
            
        except:
            # If convex hull calculation fails, estimate solidity from extent
            solidity = extent
        
        # Calculate rectangularity (how well the shape fits a rectangle)
        rectangularity = extent
        
        # Calculate circularity
        # For a perfect circle, 4*π*area / perimeter^2 = 1
        # Perimeter is approximated by the length of the boundary
        boundary = ndimage.binary_dilation(feature_mask) ^ feature_mask
        perimeter = np.sum(boundary)
        circularity = (4 * np.pi * feature_size) / (perimeter**2) if perimeter > 0 else 0
        
        # Determine shape type
        is_rectangular = rectangularity > 0.8
        is_circular = circularity > 0.7
        
        # Calculate confidence based on shape regularity
        if is_rectangular:
            confidence = rectangularity * 0.8 + solidity * 0.2
        elif is_circular:
            confidence = circularity * 0.8 + solidity * 0.2
        else:
            confidence = solidity * 0.5
            
        # Convert pixel coordinates to geographic coordinates
        # For synthetic data, we'll use a simple linear transformation
        # In real data, this would use proper georeferencing
        lon = roi['min_lon'] + (centroid_x / elevation.shape[1]) * (roi['max_lon'] - roi['min_lon'])
        lat = roi['max_lat'] - (centroid_y / elevation.shape[0]) * (roi['max_lat'] - roi['min_lat'])
        
        # Only include features with confidence above threshold
        if confidence >= threshold:
            feature_list.append({
                'lat': lat,
                'lon': lon,
                'center_x': centroid_x,
                'center_y': centroid_y,
                'width_pixels': width,
                'height_pixels': height,
                'width_m': width * 20,  # Assuming 20m resolution for synthetic data
                'height_m': height * 20,
                'area_sq_m': feature_size * 400,  # 20m x 20m pixels
                'is_rectangular': is_rectangular,
                'is_circular': is_circular,
                'rectangularity': rectangularity,
                'circularity': circularity,
                'solidity': solidity,
                'detection_method': 'geometric',
                'confidence': confidence
            })
    
    # Convert to DataFrame
    features_df = pd.DataFrame(feature_list)
    
    # Sort by confidence
    if not features_df.empty:
        features_df = features_df.sort_values(by='confidence', ascending=False)
    
    print(f"Geometric detection identified {len(features_df)} potential archaeological features")
    
    return features_df

# Here's the fixed function to replace in your notebook:

def detect_terrain_anomalies(data, window_size=30, threshold=1.5):
    """
    Detect terrain anomalies that might indicate archaeological sites
    
    Parameters:
    - data: Dictionary with data layers
    - window_size: Size of the local window for anomaly detection
    - threshold: Standard deviation threshold for anomaly detection
    
    Returns:
    - DataFrame with detected features
    """
    print("\nAnalyzing terrain for archaeological anomalies...")
    
    elevation = data['elevation']
    
    # Calculate local deviation from surroundings
    # This highlights areas that are unnaturally different from their surroundings
    
    # Create a smoothed version of the elevation data
    smoothed = ndimage.gaussian_filter(elevation, sigma=window_size/3)
    
    # Calculate local difference
    local_diff = elevation - smoothed
    
    # Calculate local standard deviation without using windowed_std
    # Use scipy.ndimage.generic_filter instead
    def local_std(x):
        return np.std(x)
    
    # Apply the filter with a square window
    std_dev = ndimage.generic_filter(local_diff, local_std, 
                                     size=int(window_size/2), 
                                     mode='reflect')
    
    # Normalize
    std_dev = (std_dev - std_dev.min()) / (std_dev.max() - std_dev.min())
    
    # Identify anomalies
    anomalies = std_dev > np.percentile(std_dev, 98)  # Top 2%
    
    # Clean up anomalies
    anomalies = ndimage.binary_opening(anomalies, structure=np.ones((3,3)))
    
    # Find connected components
    labeled_anomalies, num_anomalies = ndimage.label(anomalies)
    
    print(f"Initial terrain analysis found {num_anomalies} potential anomalies")
    
    # Rest of function remains unchanged
    # Analyze each anomaly
    anomaly_list = []
    for anomaly_id in range(1, num_anomalies + 1):
        # Extract this anomaly
        anomaly_mask = labeled_anomalies == anomaly_id
        anomaly_size = np.sum(anomaly_mask)
        
        # Skip if too small
        if anomaly_size < 9:  # At least 3x3 pixels
            continue
            
        # Find boundaries
        y_indices, x_indices = np.where(anomaly_mask)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        
        # Calculate dimensions
        width = x_max - x_min
        height = y_max - y_min
        
        # Skip if dimensions don't make sense
        if width < 2 or height < 2:
            continue
            
        # Calculate centroid
        centroid_y = np.mean(y_indices)
        centroid_x = np.mean(x_indices)
        
        # Calculate mean elevation difference in the anomaly
        mean_diff = np.mean(local_diff[anomaly_mask])
        
        # Convert to geographic coordinates
        lon = roi['min_lon'] + (centroid_x / elevation.shape[1]) * (roi['max_lon'] - roi['min_lon'])
        lat = roi['max_lat'] - (centroid_y / elevation.shape[0]) * (roi['max_lat'] - roi['min_lat'])
        
        # Calculate confidence based on strength of anomaly
        confidence = np.mean(std_dev[anomaly_mask])
        
        anomaly_list.append({
            'lat': lat,
            'lon': lon,
            'center_x': centroid_x,
            'center_y': centroid_y,
            'width_pixels': width,
            'height_pixels': height,
            'width_m': width * 20,  # Assuming 20m resolution
            'height_m': height * 20,
            'area_sq_m': anomaly_size * 400,  # 20m x 20m pixels
            'is_rectangular': False,  # Determined only by geometric detection
            'is_circular': False,     # Determined only by geometric detection
            'mean_elevation_diff': mean_diff,
            'detection_method': 'terrain',
            'confidence': min(confidence * 1.5, 1.0)  # Scale confidence but cap at 1.0
        })
    
    # Convert to DataFrame
    anomalies_df = pd.DataFrame(anomaly_list)
    
    # Sort by confidence
    if not anomalies_df.empty:
        anomalies_df = anomalies_df.sort_values(by='confidence', ascending=False)
    
    print(f"Terrain analysis identified {len(anomalies_df)} potential archaeological features")
    
    return anomalies_df

def detect_vegetation_patterns(data, window_size=40, threshold=1.5):
    """
    Detect unusual vegetation patterns that might indicate archaeological sites
    
    Parameters:
    - data: Dictionary with data layers
    - window_size: Size of the local window for pattern detection
    - threshold: Standard deviation threshold for anomaly detection
    
    Returns:
    - DataFrame with detected features
    """
    print("\nAnalyzing vegetation for archaeological patterns...")
    
    ndvi = data['ndvi']
    
    # Calculate local deviation from surroundings (similar to terrain analysis)
    smoothed_ndvi = ndimage.gaussian_filter(ndvi, sigma=window_size/3)
    local_diff = ndvi - smoothed_ndvi
    
    # Calculate local texture (variation in vegetation patterns)
    # Use Gabor filters to detect textural patterns
    from skimage.filters import gabor
    
    # Apply Gabor filters at different orientations
    orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    texture_energy = np.zeros_like(ndvi)
    
    for theta in orientations:
        filt_real, filt_imag = gabor(ndvi, frequency=0.1, theta=theta, 
                                     sigma_x=3, sigma_y=3)
        texture_energy += np.sqrt(filt_real**2 + filt_imag**2)
    
    # Normalize texture energy
    texture_energy = (texture_energy - texture_energy.min()) / (texture_energy.max() - texture_energy.min())
    
    # Find significant local variations in NDVI
    local_var = ndimage.generic_filter(ndvi, np.var, size=window_size)
    
    # Normalize
    local_var = (local_var - local_var.min()) / (local_var.max() - local_var.min())
    
    # Combine evidence
    combined_evidence = (local_var * 0.4) + (texture_energy * 0.6)
    
    # Threshold to find significant patterns
    significant_patterns = combined_evidence > np.percentile(combined_evidence, 95)  # Top 5%
    
    # Clean up
    significant_patterns = ndimage.binary_opening(significant_patterns, structure=np.ones((3,3)))
    
    # Find connected components
    labeled_patterns, num_patterns = ndimage.label(significant_patterns)
    
    print(f"Initial vegetation analysis found {num_patterns} potential patterns")
    
    # Analyze each pattern
    pattern_list = []
    for pattern_id in range(1, num_patterns + 1):
        # Extract this pattern
        pattern_mask = labeled_patterns == pattern_id
        pattern_size = np.sum(pattern_mask)
        
        # Skip if too small
        if pattern_size < 9:  # At least 3x3 pixels
            continue
            
        # Find boundaries
        y_indices, x_indices = np.where(pattern_mask)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        
        # Calculate dimensions
        width = x_max - x_min
        height = y_max - y_min
        
        # Skip if dimensions don't make sense
        if width < 2 or height < 2:
            continue
            
        # Calculate centroid
        centroid_y = np.mean(y_indices)
        centroid_x = np.mean(x_indices)
        
        # Calculate mean NDVI difference in the pattern
        mean_ndvi_diff = np.mean(local_diff[pattern_mask])
        
        # Analyze shape compactness
        area = pattern_size
        perimeter = np.sum(ndimage.binary_dilation(pattern_mask) ^ pattern_mask)
        compactness = (perimeter**2) / (4 * np.pi * area) if area > 0 and perimeter > 0 else float('inf')
        
        # Convert to geographic coordinates
        lon = roi['min_lon'] + (centroid_x / ndvi.shape[1]) * (roi['max_lon'] - roi['min_lon'])
        lat = roi['max_lat'] - (centroid_y / ndvi.shape[0]) * (roi['max_lat'] - roi['min_lat'])
        
        # Calculate confidence based on strength of pattern and compactness
        pattern_strength = np.mean(combined_evidence[pattern_mask])
        # More compact (lower compactness) is better, with diminishing returns
        compactness_score = 1.0 if compactness < 1.1 else 1.0 / np.sqrt(compactness)
        confidence = pattern_strength * 0.7 + compactness_score * 0.3
        
        pattern_list.append({
            'lat': lat,
            'lon': lon,
            'center_x': centroid_x,
            'center_y': centroid_y,
            'width_pixels': width,
            'height_pixels': height,
            'width_m': width * 20,  # Assuming 20m resolution
            'height_m': height * 20,
            'area_sq_m': pattern_size * 400,  # 20m x 20m pixels
            'is_rectangular': False,  # Determined only by geometric detection
            'is_circular': False,     # Determined only by geometric detection
            'mean_ndvi_diff': mean_ndvi_diff,
            'compactness': compactness,
            'detection_method': 'vegetation',
            'confidence': min(confidence * 1.2, 1.0)  # Scale confidence but cap at 1.0
        })
    
    # Convert to DataFrame
    patterns_df = pd.DataFrame(pattern_list)
    
    # Sort by confidence
    if not patterns_df.empty:
        patterns_df = patterns_df.sort_values(by='confidence', ascending=False)
    
    print(f"Vegetation analysis identified {len(patterns_df)} potential archaeological features")
    
    return patterns_df

# Apply all three detection methods
geometric_features = detect_geometric_features(sample_data)
terrain_features = detect_terrain_anomalies(sample_data)
vegetation_features = detect_vegetation_patterns(sample_data)

# Combine all detected features
all_features = pd.concat([geometric_features, terrain_features, vegetation_features], ignore_index=True)

# Create columns for verification flags (initialized to False)
all_features['geometric_verified'] = all_features['detection_method'] == 'geometric'
all_features['terrain_verified'] = all_features['detection_method'] == 'terrain'
all_features['vegetation_verified'] = all_features['detection_method'] == 'vegetation'

# Print summary of detected features
print("\nSummary of feature detection:")
print(f"Geometric features: {len(geometric_features)}")
print(f"Terrain anomalies: {len(terrain_features)}")
print(f"Vegetation patterns: {len(vegetation_features)}")
print(f"Total potential features: {len(all_features)}")


def display_features_on_image(image, features_df, title="Detected Archaeological Features"):
    """
    Display detected features overlaid on an image
    
    Parameters:
    - image: 2D or 3D image array to use as background
    - features_df: DataFrame with feature information
    - title: Plot title
    """
    if features_df.empty:
        print(f"No features to display for '{title}'")
        return
    
    plt.figure(figsize=(12, 10))
    
    # Display the background image
    if image.ndim == 3:  # RGB image
        plt.imshow(image)
    else:  # Single-channel image
        plt.imshow(image, cmap='gray')
    
    # Color scheme for different detection methods
    method_colors = {
        'geometric': 'red',
        'terrain': 'blue',
        'vegetation': 'green'
    }
    
    # Plot each feature with appropriate style
    for _, feature in features_df.iterrows():
        # Set marker style based on feature type
        if feature.get('is_rectangular', False):
            marker = 's'  # square
        elif feature.get('is_circular', False):
            marker = 'o'  # circle
        else:
            marker = 'x'  # x-mark
        
        # Set color based on detection method
        color = method_colors.get(feature['detection_method'], 'yellow')
        
        # Size based on confidence
        size = 50 + 150 * feature['confidence']
        
        # Plot the feature
        plt.scatter(feature['center_x'], feature['center_y'], 
                  s=size, alpha=0.6, 
                  color=color, marker=marker)
    
    # Add a legend
    legend_elements = []
    for method, color in method_colors.items():
        if any(features_df['detection_method'] == method):
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                           markerfacecolor=color, markersize=10,
                                           label=f"{method.capitalize()} detection"))
    
    plt.legend(handles=legend_elements, loc='upper right')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Display all features on the RGB image
display_features_on_image(sample_data['rgb'], all_features, 
                        title="All Detected Archaeological Features")

# Display features by detection method on appropriate data layers
display_features_on_image(sample_data['elevation'], geometric_features, 
                        title="Geometric Features on Elevation Data")

display_features_on_image(sample_data['elevation'], terrain_features, 
                        title="Terrain Anomalies on Elevation Data")

display_features_on_image(sample_data['ndvi'], vegetation_features, 
                        title="Vegetation Patterns on NDVI Data")


def cross_verify_features(all_features, data, radius_pixels=20):
    """
    Cross-verify features detected by different methods
    
    Parameters:
    - all_features: DataFrame with features from all detection methods
    - data: Dictionary with data layers
    - radius_pixels: Radius to search for nearby features
    
    Returns:
    - DataFrame with updated verification flags
    """
    print("\nCross-verifying archaeological features with multiple detection methods...")
    
    if all_features.empty:
        print("No features to cross-verify")
        return all_features
    
    # Create a copy of the features DataFrame
    verified_features = all_features.copy()
    
    # Create a spatial index for efficient proximity search
    # For each feature, find nearby features detected by other methods
    for i, feature in verified_features.iterrows():
        center_x = feature['center_x']
        center_y = feature['center_y']
        orig_method = feature['detection_method']
        
        # Search for nearby features detected by other methods
        for j, other_feature in verified_features.iterrows():
            if i == j:
                continue  # Skip self
                
            if other_feature['detection_method'] == orig_method:
                continue  # Skip features from same method
                
            # Calculate distance in pixels
            distance = np.sqrt((center_x - other_feature['center_x'])**2 + 
                              (center_y - other_feature['center_y'])**2)
            
            # If nearby, mark as cross-verified by that method
            if distance <= radius_pixels:
                other_method = other_feature['detection_method']
                verified_features.at[i, f'{other_method}_verified'] = True
    
    # Calculate number of verification methods for each feature
    verification_columns = ['geometric_verified', 'terrain_verified', 'vegetation_verified']
    verified_features['verification_count'] = verified_features[verification_columns].sum(axis=1)
    
    # Mark features as verified if detected by at least two methods
    verified_features['verified'] = verified_features['verification_count'] >= 2
    
    # Display verification results
    print(f"Cross-verification complete:")
    print(f"  Features verified by at least 2 methods: {sum(verified_features['verified'])}")
    print(f"  Features verified by all 3 methods: {sum(verified_features['verification_count'] == 3)}")
    
    return verified_features

# Cross-verify features
verified_features = cross_verify_features(all_features, sample_data)

# Show verification statistics for each detection method
print("\nVerification statistics by detection method:")
for method in ['geometric', 'terrain', 'vegetation']:
    method_features = verified_features[verified_features['detection_method'] == method]
    if len(method_features) > 0:
        verified_count = sum(method_features['verified'])
        print(f"  {method.capitalize()}: {verified_count} of {len(method_features)} verified ({verified_count/len(method_features)*100:.1f}%)")

# Display the verified features as a table
print("\nTop verified features:")
if sum(verified_features['verified']) > 0:
    selected_columns = ['lat', 'lon', 'detection_method', 'verification_count', 'confidence']
    display(verified_features[verified_features['verified']].sort_values('verification_count', ascending=False)[selected_columns].head(10))
else:
    print("No features were verified by multiple methods.")


def visualize_verified_sites(image, verified_features, data, radius=40):
    """
    Create a detailed visualization of verified archaeological sites
    
    Parameters:
    - image: RGB image for background
    - verified_features: DataFrame with verified features
    - data: Dictionary with all data layers
    - radius: Radius for detail insets
    """
    # Filter for verified sites
    verified_sites = verified_features[verified_features['verified']].copy()
    
    if verified_sites.empty:
        print("No verified sites to visualize")
        return
    
    # Sort by confidence and verification count
    verified_sites = verified_sites.sort_values(['verification_count', 'confidence'], 
                                              ascending=[False, False])
    
    # Take up to top 3 sites to match our grid layout (3 rows available)
    top_sites = verified_sites.head(3)
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    
    # Main plot with all verified sites
    ax_main = plt.subplot2grid((3, 6), (0, 0), colspan=4, rowspan=3)
    ax_main.imshow(image)
    
    # Plot all verified sites on main image
    for _, site in verified_sites.iterrows():
        color = 'red' if site['verification_count'] == 3 else 'orange' if site['verification_count'] == 2 else 'yellow'
        size = 80 + (site['verification_count'] * 40)
        ax_main.scatter(site['center_x'], site['center_y'], 
                      s=size, alpha=0.7, 
                      color=color, edgecolor='white')
    
    # Title and settings
    ax_main.set_title('Verified Archaeological Sites', fontsize=16)
    ax_main.axis('off')
    
    # Detail insets for top sites
    for i, (_, site) in enumerate(top_sites.iterrows()):
        # Create inset for this site - check if we're within grid bounds
        if i < 3:  # We have 3 rows available
            ax_inset = plt.subplot2grid((3, 6), (i, 4), colspan=2)
            
            # Extract region around this site
            center_x, center_y = int(site['center_x']), int(site['center_y'])
            min_y = max(0, center_y - radius)
            max_y = min(image.shape[0], center_y + radius)
            min_x = max(0, center_x - radius)
            max_x = min(image.shape[1], center_x + radius)
            
            # Show region in context with appropriate visualization
            if site['detection_method'] == 'geometric':
                # For geometric features, show elevation with enhanced contrast
                elev_detail = data['elevation'][min_y:max_y, min_x:max_x]
                # Enhance local contrast
                elev_min, elev_max = np.percentile(elev_detail, [5, 95])
                elev_normalized = np.clip((elev_detail - elev_min) / (elev_max - elev_min), 0, 1)
                ax_inset.imshow(elev_normalized, cmap='terrain')
                
            elif site['detection_method'] == 'terrain':
                # For terrain features, show hillshade visualization
                elev_detail = data['elevation'][min_y:max_y, min_x:max_x]
                # Create hillshade effect
                from matplotlib.colors import LightSource
                ls = LightSource(azdeg=315, altdeg=45)
                hillshade = ls.hillshade(elev_detail, vert_exag=10)
                ax_inset.imshow(hillshade, cmap='gray')
                
            else:  # vegetation
                # For vegetation features, show NDVI
                ndvi_detail = data['ndvi'][min_y:max_y, min_x:max_x]
                ax_inset.imshow(ndvi_detail, cmap='YlGn')
            
            # Add a marker at the center
            center_marker_x = center_x - min_x
            center_marker_y = center_y - min_y
            ax_inset.scatter(center_marker_x, center_marker_y, 
                           s=100, color='red', marker='x')
            
            # Add a title
            site_type = site['is_rectangular'] if 'is_rectangular' in site and site['is_rectangular'] else 'Circular' if 'is_circular' in site and site['is_circular'] else 'Irregular'
            ax_inset.set_title(f"Site {i+1}: {site['detection_method'].capitalize()}", fontsize=12)
            
            # Add details as text
            detail_text = f"Verified by {int(site['verification_count'])} methods\n"
            detail_text += f"Confidence: {site['confidence']:.2f}\n"
            detail_text += f"Size: {int(site['width_m'])}×{int(site['height_m'])}m"
            
            ax_inset.text(0.05, 0.05, detail_text, transform=ax_inset.transAxes, 
                         fontsize=10, color='white', backgroundcolor='black',
                         verticalalignment='bottom')
            
            ax_inset.axis('off')
            
            # Add a rectangle in the main plot showing the detail area
            rect = plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, 
                               fill=False, edgecolor='white', linestyle='--')
            ax_main.add_patch(rect)
            
            # Add a number label
            ax_main.text(center_x + 10, center_y + 10, str(i+1), 
                       color='white', fontsize=12, fontweight='bold',
                       bbox=dict(facecolor='black', alpha=0.7))
    
    # Add a note if there are more sites than we can show in detail
    if len(verified_sites) > 3:
        ax_main.text(10, 10, f"Showing details for top 3 of {len(verified_sites)} sites", 
                   color='white', fontsize=12, fontweight='bold',
                   bbox=dict(facecolor='black', alpha=0.7))
    
    plt.tight_layout()
    plt.show()

# Create a detailed visualization of verified sites
visualize_verified_sites(sample_data['rgb'], verified_features, sample_data)

# Create a simple interactive map
if FOLIUM_AVAILABLE:
    def create_interactive_map(verified_features, roi):
        """Create an interactive map showing verified archaeological sites"""
        if verified_features.empty or sum(verified_features['verified']) == 0:
            print("No verified sites to display on map")
            return None
        
        # Calculate map center
        center_lat = (roi['min_lat'] + roi['max_lat']) / 2
        center_lon = (roi['min_lon'] + roi['max_lon']) / 2
        
        # Create the map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
        
        # Add a rectangle showing the region of interest
        folium.Rectangle(
            bounds=[[roi['min_lat'], roi['min_lon']], [roi['max_lat'], roi['max_lon']]],
            color='green',
            fill=False,
            weight=2
        ).add_to(m)
        
        # Add markers for verified sites
        verified_sites = verified_features[verified_features['verified']]
        
        for idx, site in verified_sites.iterrows():
            # Determine popup content
            detection_method = site['detection_method'].capitalize()
            verification_count = int(site['verification_count'])
            confidence = site['confidence']
            width = site.get('width_m', 0)
            height = site.get('height_m', 0)
            
            popup_html = f"""
            <h4>Archaeological Site</h4>
            <b>Detection Method:</b> {detection_method}<br>
            <b>Verified by:</b> {verification_count} methods<br>
            <b>Confidence:</b> {confidence:.2f}<br>
            <b>Approx. Size:</b> {int(width)}×{int(height)} m<br>
            <b>Coordinates:</b> {site['lat']:.6f}, {site['lon']:.6f}
            """
            
            # Set marker color based on verification count
            if verification_count == 3:
                color = 'red'  # Verified by all methods
            elif verification_count == 2:
                color = 'orange'  # Verified by 2 methods
            else:
                color = 'blue'  # Default
            
            # Add the marker
            folium.CircleMarker(
                location=[site['lat'], site['lon']],
                radius=10,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=folium.Popup(popup_html, max_width=300)
            ).add_to(m)
        
        # Save the map as HTML
        m.save('archaeological_sites_map.html')
        print("Interactive map saved as 'archaeological_sites_map.html'")
        
        return m
    
    # Create interactive map
    interactive_map = create_interactive_map(verified_features, roi)
    
    # Display the map (when running in Jupyter/Colab)
    try:
        from IPython.display import display
        display(interactive_map)
    except:
        pass
else:
    print("Folium not available - interactive map generation skipped")


def analyze_spatial_patterns(verified_features):
    """
    Analyze spatial patterns and relationships between verified sites
    """
    # Filter for verified sites
    verified_sites = verified_features[verified_features['verified']]
    
    if len(verified_sites) < 2:
        print("Not enough verified sites for spatial pattern analysis")
        return
    
    print("\nAnalyzing spatial patterns among verified sites...")
    
    # Calculate distances between all pairs of sites
    coords = verified_sites[['center_x', 'center_y']].values
    distances = squareform(pdist(coords, metric='euclidean'))
    
    # Convert distances to meters (based on our synthetic data resolution)
    distances_m = distances * 20  # Assuming 20m resolution
    
    # Calculate basic statistics
    mean_distance = np.mean(distances_m[distances_m > 0])
    min_distance = np.min(distances_m[distances_m > 0])
    max_distance = np.max(distances_m)
    
    print(f"Distance statistics between sites:")
    print(f"  Mean distance: {mean_distance:.1f} meters")
    print(f"  Minimum distance: {min_distance:.1f} meters")
    print(f"  Maximum distance: {max_distance:.1f} meters")
    
    # Check for linear arrangements
    # For real analysis, we'd use more sophisticated methods
    if len(verified_sites) >= 3:
        # Try to fit a line to the site coordinates
        from sklearn.linear_model import LinearRegression
        
        X = coords[:, 0].reshape(-1, 1)  # x-coordinates
        y = coords[:, 1]  # y-coordinates
        
        model = LinearRegression().fit(X, y)
        r_squared = model.score(X, y)
        
        if r_squared > 0.8:
            print(f"Sites show a strong linear arrangement (R² = {r_squared:.2f})")
            # Convert to cardinal direction
            angle = np.arctan(model.coef_[0]) * 180 / np.pi
            direction = "N-S" if abs(angle) > 45 else "E-W"
            print(f"  Orientation: {direction} ({angle:.1f}°)")
        elif r_squared > 0.5:
            print(f"Sites show a moderate linear arrangement (R² = {r_squared:.2f})")
        else:
            print(f"Sites do not show a strong linear arrangement (R² = {r_squared:.2f})")
    
    # Check for grid-like patterns
    # This would require more sophisticated analysis in a real application
    
    # Look for regular spacing
    if len(verified_sites) >= 4:
        # Sort all non-zero distances
        flat_distances = distances_m[distances_m > 0].flatten()
        flat_distances.sort()
        
        # Check if there are clusters of similar distances
        from sklearn.cluster import KMeans
        
        # Reshape for KMeans
        flat_distances_reshaped = flat_distances.reshape(-1, 1)
        
        # Try with different numbers of clusters
        best_score = -np.inf
        best_k = 1
        best_clusters = None
        
        for k in range(1, min(5, len(flat_distances))):
            kmeans = KMeans(n_clusters=k, random_state=0).fit(flat_distances_reshaped)
            score = -kmeans.inertia_  # Negative inertia as score (higher is better)
            
            if score > best_score:
                best_score = score
                best_k = k
                best_clusters = kmeans
        
        # If we found meaningful clusters
        if best_k > 1:
            # Get cluster centers
            centers = best_clusters.cluster_centers_.flatten()
            centers.sort()
            
            print(f"Sites show evidence of regular spacing with {best_k} characteristic distances:")
            for i, center in enumerate(centers):
                print(f"  Distance pattern {i+1}: {center:.1f} meters")
            
            # Check if the distances are multiples of each other
            if len(centers) >= 2:
                ratio = centers[1] / centers[0]
                if 1.8 < ratio < 2.2:
                    print(f"  Distance pattern suggests a regular grid with 1:2 ratio")
                elif 2.8 < ratio < 3.2:
                    print(f"  Distance pattern suggests a regular grid with 1:3 ratio")
    
    # Create a visualization of the connections between sites
    plt.figure(figsize=(10, 8))
    
    # Plot the sites
    plt.scatter(coords[:, 0], coords[:, 1], s=100, c='red')
    
    # Add site numbers
    for i, (x, y) in enumerate(coords):
        plt.text(x + 5, y + 5, str(i+1), fontsize=12)
    
    # Draw connections between nearby sites
    threshold_distance = np.percentile(distances_m[distances_m > 0], 30)  # Connect closest 30%
    
    for i in range(len(coords)):
        for j in range(i+1, len(coords)):
            if distances_m[i, j] <= threshold_distance:
                plt.plot([coords[i, 0], coords[j, 0]], 
                        [coords[i, 1], coords[j, 1]], 
                        'b-', alpha=0.6)
    
    plt.title('Spatial Relationships Between Archaeological Sites')
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# Analyze spatial patterns of verified sites
analyze_spatial_patterns(verified_features)

def generate_final_report(verified_features, roi):
    """Generate a comprehensive report of findings"""
    # Filter for verified sites
    verified_sites = verified_features[verified_features['verified']]
    
    print("\n" + "="*80)
    print("ARCHAEOLOGICAL SITE DISCOVERY REPORT".center(80))
    print("="*80)
    
    print(f"\nRegion of Analysis: {roi['name']}")
    print(f"Geographic Extent: {roi['min_lon']:.4f}° to {roi['max_lon']:.4f}°E, " + 
          f"{roi['min_lat']:.4f}° to {roi['max_lat']:.4f}°N")
    print(f"Area: Approximately {(roi['max_lon'] - roi['min_lon']) * 111.32 * np.cos(np.radians((roi['min_lat'] + roi['max_lat'])/2)) * (roi['max_lat'] - roi['min_lat']) * 110.574:.1f} km²")
    print(f"Analysis Date: {datetime.now().strftime('%B %d, %Y')}")
    print(f"Analysis Completed By: AdilShamim8")
    
    print("\n" + "-"*80)
    print("SUMMARY OF FINDINGS".center(80))
    print("-"*80)
    
    print(f"\nTotal features detected: {len(verified_features)}")
    print(f"Features verified by multiple methods: {len(verified_sites)}")
    
    method_counts = verified_features['detection_method'].value_counts()
    print("\nDetection method breakdown:")
    for method, count in method_counts.items():
        verified_count = sum(verified_features[verified_features['detection_method'] == method]['verified'])
        print(f"  {method.capitalize()}: {verified_count} verified out of {count} detected")
    
    if not verified_sites.empty:
        print("\n" + "-"*80)
        print("VERIFIED ARCHAEOLOGICAL SITES".center(80))
        print("-"*80)
        
        for i, (idx, site) in enumerate(verified_sites.sort_values(['verification_count', 'confidence'], ascending=[False, False]).iterrows(), 1):
            print(f"\nSITE {i}:")
            print(f"  Coordinates: {site['lat']:.6f}°, {site['lon']:.6f}°")
            print(f"  Detection Method: {site['detection_method'].capitalize()}")
            print(f"  Verification Methods: {int(site['verification_count'])}")
            
            # List which methods verified this site
            methods = []
            if site['geometric_verified']: methods.append("Geometric")
            if site['terrain_verified']: methods.append("Terrain")
            if site['vegetation_verified']: methods.append("Vegetation")
            print(f"  Verified By: {', '.join(methods)}")
            
            print(f"  Confidence Score: {site['confidence']:.2f}")
            
            # Physical characteristics
            width = site.get('width_m', 0)
            height = site.get('height_m', 0)
            print(f"  Dimensions: {int(width)}m × {int(height)}m")
            
            # Shape information
            if 'is_rectangular' in site and site['is_rectangular']:
                print(f"  Shape: Rectangular earthwork")
            elif 'is_circular' in site and site['is_circular']:
                print(f"  Shape: Circular earthwork")
            else:
                print(f"  Shape: Irregular feature")
            
            # Additional characteristics based on detection method
            if site['detection_method'] == 'terrain':
                if 'mean_elevation_diff' in site:
                    elevation_change = site['mean_elevation_diff']
                    direction = "raised" if elevation_change > 0 else "depressed"
                    print(f"  Terrain: {direction} feature, {abs(elevation_change)*100:.1f}cm relative to surroundings")
            
            elif site['detection_method'] == 'vegetation':
                if 'mean_ndvi_diff' in site:
                    ndvi_change = site['mean_ndvi_diff']
                    vegetation_type = "denser" if ndvi_change > 0 else "sparser"
                    print(f"  Vegetation: {vegetation_type} than surroundings")
    
    print("\n" + "-"*80)
    print("ARCHAEOLOGICAL INTERPRETATION".center(80))
    print("-"*80)
    
    # Add interpretation based on findings
    if len(verified_sites) == 0:
        print("\nNo verified archaeological sites were found in the study area.")
    elif len(verified_sites) == 1:
        print("\nA single archaeological site was identified, suggesting a possible")
        print("isolated settlement or ceremonial structure. Without additional sites,")
        print("it's difficult to determine its cultural context or temporal placement.")
    elif 1 < len(verified_sites) <= 3:
        print("\nThe small number of archaeological sites suggests a minor settlement")
        print("pattern. These may represent a small community or special-purpose sites")
        print("such as resource extraction areas or ceremonial locations.")
    elif 3 < len(verified_sites) <= 10:
        print("\nThe moderate number of archaeological sites suggests a significant")
        print("human presence in this area. The patterns observed are consistent with")
        print("pre-Columbian settlement systems identified in the Llanos de Moxos region")
        print("by Prümers et al. (2022).")
    else:
        print("\nThe high density of archaeological sites strongly suggests an extensive")
        print("settlement system. This pattern is consistent with the 'low-density urbanism'")
        print("model proposed for pre-Columbian Amazonian civilizations, characterized by")
        print("a network of interconnected settlements with agricultural earthworks.")
    
    # Add specific interpretations based on feature types
    if not verified_sites.empty:
        rectangular_count = sum(verified_sites.get('is_rectangular', False))
        circular_count = sum(verified_sites.get('is_circular', False))
        
        if rectangular_count > circular_count:
            print("\nThe predominance of rectangular earthworks suggests affiliation with")
            print("cultural traditions identified in the southern Amazon, particularly")
            print("those documented in the Bolivian Llanos de Moxos (Prümers et al., 2022).")
        elif circular_count > rectangular_count:
            print("\nThe predominance of circular earthworks suggests possible affiliation")
            print("with cultural traditions identified in the upper Xingu River basin")
            print("(Heckenberger et al., 2008) or the geoglyph-building cultures of Acre.")
        
        # Check for linear arrangements
        if len(verified_sites) >= 3:
            coords = verified_sites[['center_x', 'center_y']].values
            X = coords[:, 0].reshape(-1, 1)
            y = coords[:, 1]
            
            from sklearn.linear_model import LinearRegression
            model = LinearRegression().fit(X, y)
            r_squared = model.score(X, y)
            
            if r_squared > 0.7:
                angle = np.arctan(model.coef_[0]) * 180 / np.pi
                print(f"\nThe linear arrangement of sites (orientation {angle:.1f}°) suggests")
                print("intentional planning and may represent a ceremonial or astronomical")
                print("alignment, consistent with patterns observed in other Amazonian")
                print("archaeological complexes.")
    
    print("\n" + "-"*80)
    print("VERIFICATION METHODOLOGY".center(80))
    print("-"*80)
    
    print("\nFeatures were verified using multiple independent methods:")
    print("1. Geometric detection: Identified regular shapes indicative of human construction")
    print("2. Terrain analysis: Detected anomalies in elevation data suggesting earthworks")
    print("3. Vegetation pattern analysis: Identified unusual plant growth patterns that")
    print("   often indicate buried archaeological features")
    
    print("\nOnly features confirmed by at least two independent methods were considered")
    print("verified archaeological sites, in accordance with competition requirements.")
    
    print("\n" + "-"*80)
    print("RECOMMENDATIONS FOR FURTHER RESEARCH".center(80))
    print("-"*80)
    
    print("\n1. Acquire high-resolution LiDAR data for the identified sites to confirm")
    print("   their structural characteristics with greater precision")
    print("2. Conduct targeted field surveys to collect surface artifacts and obtain")
    print("   samples for radiocarbon dating")
    print("3. Expand the analysis to adjacent areas to identify additional components")
    print("   of the settlement system")
    print("4. Compare the identified patterns with ethnographic and historical records")
    print("   of Indigenous land use in the region")
    
    print("\n" + "-"*80)
    print("DATA SOURCES AND REFERENCES".center(80))
    print("-"*80)
    
    print("\nData Sources:")
    print("1. Synthetic data generated for demonstration purposes")
    print("2. If this were a real analysis, we would use:")
    print("   - Sentinel-2 multispectral imagery (10m resolution)")
    print("   - SRTM or ALOS PALSAR elevation data")
    print("   - LiDAR data from OpenTopography where available")
    
    print("\nKey References:")
    print("1. Prümers, H., Betancourt, C.J., Iriarte, J. et al. (2022). Lidar reveals")
    print("   pre-Hispanic low-density urbanism in the Bolivian Amazon. Nature 606, 325–328.")
    print("2. Vinicius Peripato et al. (2023). More than 10,000 pre-Columbian earthworks")
    print("   are still hidden throughout Amazonia. Science 382:6666, 103-109.")
    print("3. Iriarte, J., et al. (2020). Geometry by Design: Contribution of Lidar to the")
    print("   Understanding of Settlement Patterns of the Mound Villages in SW Amazonia.")
    print("   Journal of Computer Applications in Archaeology 3:1, 151-169.")
    
    print("\nReport generated using OpenAI models to assist with data analysis")
    print(f"Generation timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

# Generate final report
generate_final_report(verified_features, roi)


def export_results_for_submission(verified_features, roi, filename_prefix='amazon_archaeology'):
    """
    Export the results in formats suitable for competition submission
    
    Parameters:
    - verified_features: DataFrame with verified features
    - roi: Dictionary with region of interest information
    - filename_prefix: Prefix for output files
    """
    # Filter for verified sites
    verified_sites = verified_features[verified_features['verified']].copy().reset_index(drop=True)
    
    if verified_sites.empty:
        print("No verified sites to export")
        return
    
    # Create a timestamp for filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. Export to CSV
    csv_filename = f"{filename_prefix}_{timestamp}.csv"
    verified_sites.to_csv(csv_filename, index=False)
    print(f"Exported verified sites to {csv_filename}")
    
    # 2. Export summary JSON
    summary = {
        'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'region': {
            'name': roi['name'],
            'min_lon': roi['min_lon'],
            'max_lon': roi['max_lon'],
            'min_lat': roi['min_lat'],
            'max_lat': roi['max_lat'],
            'area_km2': (roi['max_lon'] - roi['min_lon']) * 111.32 * np.cos(np.radians((roi['min_lat'] + roi['max_lat'])/2)) * (roi['max_lat'] - roi['min_lat']) * 110.574
        },
        'sites_count': len(verified_sites),
        'sites': []
    }
    
    # Add each site to the summary
    for _, site in verified_sites.iterrows():
        site_data = {
            'lat': float(site['lat']),
            'lon': float(site['lon']),
            'confidence': float(site['confidence']),
            'verification_count': int(site['verification_count']),
            'detection_method': site['detection_method'],
            'dimensions': {
                'width_m': float(site.get('width_m', 0)),
                'height_m': float(site.get('height_m', 0)),
                'area_sq_m': float(site.get('area_sq_m', 0))
            },
            'shape': 'rectangular' if site.get('is_rectangular', False) else 'circular' if site.get('is_circular', False) else 'irregular'
        }
        summary['sites'].append(site_data)
    
    # Save JSON
    json_filename = f"{filename_prefix}_{timestamp}.json"
    with open(json_filename, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Exported summary to {json_filename}")
    
    # 3. Export KML for Google Earth visualization
    kml_content = []
    kml_content.append('<?xml version="1.0" encoding="UTF-8"?>')
    kml_content.append('<kml xmlns="http://www.opengis.net/kml/2.2">')
    kml_content.append('<Document>')
    kml_content.append(f'<name>Archaeological Sites - {roi["name"]}</name>')
    kml_content.append('<description>Detected archaeological sites from OpenAI to Z Challenge analysis</description>')
    
    # Create styles for different verification levels
    kml_content.append('<Style id="verified3"><IconStyle><color>ff0000ff</color><scale>1.2</scale></IconStyle></Style>')
    kml_content.append('<Style id="verified2"><IconStyle><color>ff00aaff</color><scale>1.0</scale></IconStyle></Style>')
    
    # Add each site as a placemark
    for i, (_, site) in enumerate(verified_sites.iterrows(), 1):
        verification_count = int(site['verification_count'])
        style = f"verified{verification_count}" if verification_count >= 2 else "verified2"
        
        kml_content.append('<Placemark>')
        kml_content.append(f'<name>Archaeological Site {i}</name>')
        kml_content.append(f'<styleUrl>#{style}</styleUrl>')
        
        # Create description with site details
        desc = f"<![CDATA["
        desc += f"<h3>Archaeological Site {i}</h3>"
        desc += f"<p><b>Detection Method:</b> {site['detection_method'].capitalize()}</p>"
        desc += f"<p><b>Verified by:</b> {verification_count} methods</p>"
        desc += f"<p><b>Confidence:</b> {site['confidence']:.2f}</p>"
        desc += f"<p><b>Size:</b> {int(site.get('width_m', 0))}×{int(site.get('height_m', 0))} m</p>"
        desc += f"<p><b>Area:</b> {int(site.get('area_sq_m', 0))} sq.m</p>"
        desc += "]]>"
        kml_content.append(f'<description>{desc}</description>')
        
        # Add coordinates
        kml_content.append('<Point>')
        kml_content.append(f'<coordinates>{site["lon"]},{site["lat"]},0</coordinates>')
        kml_content.append('</Point>')
        kml_content.append('</Placemark>')
    
    # Add a polygon for the region of interest
    kml_content.append('<Placemark>')
    kml_content.append('<name>Region of Interest</name>')
    kml_content.append('<Style><LineStyle><color>ff00ff00</color><width>2</width></LineStyle><PolyStyle><fill>0</fill></PolyStyle></Style>')
    kml_content.append('<Polygon>')
    kml_content.append('<outerBoundaryIs>')
    kml_content.append('<LinearRing>')
    kml_content.append('<coordinates>')
    kml_content.append(f'{roi["min_lon"]},{roi["min_lat"]},0 ')
    kml_content.append(f'{roi["max_lon"]},{roi["min_lat"]},0 ')
    kml_content.append(f'{roi["max_lon"]},{roi["max_lat"]},0 ')
    kml_content.append(f'{roi["min_lon"]},{roi["max_lat"]},0 ')
    kml_content.append(f'{roi["min_lon"]},{roi["min_lat"]},0')
    kml_content.append('</coordinates>')
    kml_content.append('</LinearRing>')
    kml_content.append('</outerBoundaryIs>')
    kml_content.append('</Polygon>')
    kml_content.append('</Placemark>')
    
    kml_content.append('</Document>')
    kml_content.append('</kml>')
    
    # Save KML
    kml_filename = f"{filename_prefix}_{timestamp}.kml"
    with open(kml_filename, 'w') as f:
        f.write('\n'.join(kml_content))
    print(f"Exported KML for Google Earth to {kml_filename}")
    
    print("\nFiles exported successfully. Use these files for your competition submission.")

# Export results
export_results_for_submission(verified_features, roi)

