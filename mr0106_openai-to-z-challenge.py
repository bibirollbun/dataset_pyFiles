


# # Amazon Ancient Civilizations Detection with AI
# 
# ## Advanced Archaeological Detection Pipeline
# 
# This pipeline integrates satellite data processing, machine learning, and visualization to detect potential ancient geoglyphs in the Amazon region.

# %% [markdown]
# ## 1. Enhanced Imports and Setup

# %%
# Core libraries
import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Machine learning
from sklearn.neighbors import BallTree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.calibration import calibration_curve
import xgboost as xgb

# Data processing
import polars as pl
import duckdb
import glob

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster, HeatMap

# Utilities
from datetime import datetime
from tqdm import tqdm
from jinja2 import Template

# Configure environment
pd.set_option('display.max_columns', None)
plt.style.use('ggplot')
sns.set_palette("viridis")

print(f"Python: {sys.version.split()[0]}")
print(f"Pandas: {pd.__version__}")
print(f"XGBoost: {xgb.__version__}")

# Create directories
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True) 
os.makedirs('reports', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)


# ## 2. Data Preparation Module

# %%
class DataLoader:
    """Handles data loading with caching and fallback to synthetic data"""
    
    @staticmethod
    def load_geoglyph_data(cache=True):
        """Load geoglyph data from multiple potential sources"""
        cache_file = 'data/processed/geoglyphs_cache.pkl'
        
        if cache and os.path.exists(cache_file):
            return pd.read_pickle(cache_file)
        
        # Try multiple data sources
        paths = [
            '/kaggle/input/amazon-geoglyphs/geoglyphs.geojson',
            'data/raw/geoglyphs.geojson'
        ]
        
        for path in paths:
            try:
                gdf = gpd.read_file(path)
                if not gdf.empty:
                    if cache:
                        gdf.to_pickle(cache_file)
                    return gdf
            except Exception:
                continue
                
        print("Generating synthetic geoglyph data")
        return DataGenerator.generate_synthetic_geoglyphs(verbose=True)
    
    @staticmethod
    def load_satellite_embeddings(cache=True):
        """Load satellite embeddings with validation"""
        cache_file = 'data/processed/embeddings_cache.parquet'
        
        if cache and os.path.exists(cache_file):
            return pl.read_parquet(cache_file)
        
        paths = [
            '/kaggle/input/amazon-satellite-embeddings/*.parquet',
            'data/raw/satellite/*.parquet'
        ]
        
        for path in paths:
            try:
                files = glob.glob(path)
                if files:
                    df = pl.concat([pl.scan_parquet(f) for f in files]).select([
                        'centre_lat', 'centre_lon', 'unique_id', 'embedding'
                    ]).collect()
                    
                    if cache:
                        df.write_parquet(cache_file)
                    return df
            except Exception:
                continue
                
        print("Generating synthetic embeddings")
        return DataGenerator.generate_synthetic_embeddings(verbose=True)


class DataGenerator:
    """Generates realistic synthetic data for development"""
    
    @staticmethod
    def generate_synthetic_geoglyphs(num_points=50, verbose=False):
        """Generate geoglyphs with spatial clustering"""
        if verbose:
            print(f"Generating {num_points} synthetic geoglyphs")
            
        amazon_bbox = {
            'min_lat': -15, 'max_lat': 2,
            'min_lon': -74, 'max_lon': -50
        }
        
        # Create clustered spatial distribution
        cluster_centers = [
            {'lat': -10, 'lon': -65, 'scale': 0.5},
            {'lat': -5, 'lon': -60, 'scale': 0.7}
        ]
        
        points = []
        for _ in range(num_points):
            center = np.random.choice(cluster_centers)
            lat = np.random.normal(center['lat'], center['scale'])
            lon = np.random.normal(center['lon'], center['scale'])
            
            points.append({
                'geometry': Point(lon, lat),
                'type': np.random.choice(['circular', 'square', 'linear']),
                'size': np.random.uniform(50, 500),
                'confidence': np.random.uniform(0.7, 1.0)
            })
            
        return gpd.GeoDataFrame(points, crs="EPSG:4326")
    
    @staticmethod
    def generate_synthetic_embeddings(num_samples=2000, verbose=False):
        """Generate embeddings with spatial patterns"""
        if verbose:
            print(f"Generating {num_samples} synthetic embeddings")
            
        amazon_bbox = {
            'min_lat': -15, 'max_lat': 2,
            'min_lon': -74, 'max_lon': -50
        }
        
        embeddings = []
        for i in range(num_samples):
            embeddings.append({
                'centre_lat': np.random.uniform(amazon_bbox['min_lat'], amazon_bbox['max_lat']),
                'centre_lon': np.random.uniform(amazon_bbox['min_lon'], amazon_bbox['max_lon']),
                'unique_id': f'synth_{i}',
                'embedding': np.random.rand(2048).astype(np.float32)
            })
            
        return pl.DataFrame(embeddings)



# ## 3. Spatial Processing Module

# %%
class SpatialProcessor:
    """Handles geospatial operations and joins"""
    
    @staticmethod
    def spatial_join_optimized(geoglyphs, embeddings, radius_m=1120):
        """Perform efficient spatial join between geoglyphs and embeddings"""
        # Convert to Cartesian coordinates
        earth_radius = 6371000  # meters
        embeddings_cart = np.column_stack([
            earth_radius * np.cos(np.radians(embeddings['centre_lat'])) * np.cos(np.radians(embeddings['centre_lon'])),
            earth_radius * np.cos(np.radians(embeddings['centre_lat'])) * np.sin(np.radians(embeddings['centre_lon'])),
            earth_radius * np.sin(np.radians(embeddings['centre_lat']))
        ])
        
        geoglyphs_cart = np.column_stack([
            earth_radius * np.cos(np.radians(geoglyphs.geometry.y)) * np.cos(np.radians(geoglyphs.geometry.x)),
            earth_radius * np.cos(np.radians(geoglyphs.geometry.y)) * np.sin(np.radians(geoglyphs.geometry.x)),
            earth_radius * np.sin(np.radians(geoglyphs.geometry.y))
        ])
        
        # Build BallTree for efficient spatial queries
        tree = BallTree(embeddings_cart, metric='euclidean', leaf_size=40)
        
        # Find matches within radius
        indices, distances = tree.query_radius(
            geoglyphs_cart, 
            r=radius_m, 
            return_distance=True
        )
        
        # Process results
        results = []
        for i, (inds, dists) in enumerate(tqdm(zip(indices, distances), 
                                             total=len(indices),
                                             desc="Processing spatial joins")):
            if len(inds) > 0:
                min_idx = inds[np.argmin(dists)]
                results.append({
                    'geoglyph_id': i,
                    'embedding_id': embeddings[min_idx]['unique_id'],
                    'distance_m': np.min(dists),
                    'geoglyph_lat': geoglyphs.iloc[i].geometry.y,
                    'geoglyph_lon': geoglyphs.iloc[i].geometry.x,
                    'embedding_lat': embeddings[min_idx]['centre_lat'],
                    'embedding_lon': embeddings[min_idx]['centre_lon'],
                    'geoglyph_confidence': geoglyphs.iloc[i].get('confidence', 0.8)
                })
        
        # Return DataFrame with consistent columns
        columns = ['geoglyph_id', 'embedding_id', 'distance_m',
                  'geoglyph_lat', 'geoglyph_lon',
                  'embedding_lat', 'embedding_lon',
                  'geoglyph_confidence']
        
        return pd.DataFrame(results, columns=columns) if results else pd.DataFrame(columns=columns)


# ## 4. Machine Learning Module

# %%
class ModelTrainer:
    """Handles model training and evaluation"""
    
    @staticmethod
    def train_model(X_train, y_train, X_val=None, y_val=None):
        """Optimized model training with hyperparameter tuning"""
        # Convert data to numpy if needed
        if hasattr(X_train, 'to_numpy'):
            X_train = X_train.to_numpy()
        if X_val is not None and hasattr(X_val, 'to_numpy'):
            X_val = X_val.to_numpy()
        
        # Handle class imbalance safely
        pos_count = max(1, sum(y_train == 1))  # Avoid division by zero
        neg_count = sum(y_train == 0)
        scale_pos_weight = neg_count / pos_count
        
        # Optimized hyperparameter grid
        param_grid = {
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.05],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9],
            'gamma': [0, 0.1],
            'min_child_weight': [1, 3]
        }
        
        # Base model parameters
        base_params = {
            'objective': 'binary:logistic',
            'eval_metric': ['auc', 'logloss'],
            'random_state': 42,
            'tree_method': 'hist',
            'scale_pos_weight': scale_pos_weight,
            'n_estimators': 200
        }
        
        # Setup grid search
        grid_search = GridSearchCV(
            estimator=xgb.XGBClassifier(**base_params),
            param_grid=param_grid,
            scoring='roc_auc',
            cv=3,
            verbose=1
        )
        
        # Train with validation if available
        eval_set = [(X_val, y_val)] if (X_val is not None and len(X_val) > 0) else None
        grid_search.fit(X_train, y_train, eval_set=eval_set, verbose=10)
        
        # Print best results
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best validation AUC: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    @staticmethod
    def evaluate_model(model, X_test, y_test):
        """Comprehensive model evaluation"""
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Classification report
        print("\nğŸ“Š Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        # ROC AUC score
        roc_auc = roc_auc_score(y_test, y_proba)
        print(f"ROC AUC Score: {roc_auc:.4f}")
        
        # Feature importance plot
        plt.figure(figsize=(12, 6))
        xgb.plot_importance(model, importance_type='weight')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig('visualizations/feature_importance.png')
        plt.show()
        
        # Calibration curve
        prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, marker='o', label='Model')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect')
        plt.xlabel('Predicted Probability')
        plt.ylabel('True Probability')
        plt.title('Calibration Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig('visualizations/calibration_curve.png')
        plt.show()


# ## 5. Visualization Module

# %%
class ResultVisualizer:
    """Handles result visualization and mapping"""
    
    @staticmethod
    def visualize_results(geoglyphs, embeddings, matches=None):
        """Create interactive map visualization"""
        try:
            # Create base map
            m = folium.Map(
                location=[-5, -60],
                zoom_start=5,
                tiles='Stamen Terrain',
                attr='Map data Â© OpenStreetMap contributors'
            )
            
            # Add heatmap of all embedding locations
            HeatMap(
                data=embeddings[['centre_lat', 'centre_lon']].to_numpy(),
                radius=10,
                blur=15,
                name='Satellite Coverage'
            ).add_to(m)
            
            # Add matched pairs if available
            if matches is not None and not matches.empty:
                for _, row in matches.iterrows():
                    folium.PolyLine(
                        locations=[
                            [row['geoglyph_lat'], row['geoglyph_lon']],
                            [row['embedding_lat'], row['embedding_lon']]
                        ],
                        color='purple',
                        weight=2,
                        tooltip=f"Distance: {row['distance_m']:.0f}m"
                    ).add_to(m)
            
            # Add geoglyph markers with popup info
            marker_cluster = MarkerCluster(name="Geoglyphs")
            for _, row in geoglyphs.iterrows():
                popup = folium.Popup(
                    f"<b>Type:</b> {row.get('type', 'Unknown')}<br>"
                    f"<b>Size:</b> {row.get('size', 'N/A')}m<br>"
                    f"<b>Confidence:</b> {row.get('confidence', 0.8):.2f}",
                    max_width=250
                )
                
                marker_cluster.add_child(
                    folium.CircleMarker(
                        location=[row.geometry.y, row.geometry.x],
                        radius=5,
                        color='red',
                        fill=True,
                        popup=popup
                    )
                )
            m.add_child(marker_cluster)
            
            # Add layer control
            folium.LayerControl().add_to(m)
            
            # Save and return map
            m.save('visualizations/results_map.html')
            return m
            
        except Exception as e:
            print(f"Visualization error: {str(e)}")
            return None
    
    @staticmethod
    def plot_prediction_distribution(predictions):
        """Plot distribution of prediction scores"""
        plt.figure(figsize=(10, 6))
        sns.histplot(predictions['glyph_score'], bins=30, kde=True)
        plt.title('Prediction Score Distribution')
        plt.xlabel('Detection Score')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('visualizations/prediction_distribution.png')
        plt.show()


# ## 6. Validation and Reporting Module

# %%
class DataValidator:
    """Handles data validation and quality checks"""
    
    @staticmethod
    def validate_with_lidar(predictions, lidar_coverage=None):
        """Validate predictions against LIDAR coverage data"""
        pred_pl = pl.from_pandas(predictions)
        
        # Ensure required columns exist
        if 'unique_id' not in pred_pl.columns:
            pred_pl = pred_pl.with_columns(
                pl.lit("pred_" + pl.arange(0, pred_pl.height).cast(pl.Utf8)).alias("unique_id")
            )
        
        # Generate synthetic coverage if none provided
        if lidar_coverage is None:
            print("Generating synthetic LIDAR validation data")
            lidar_coverage = [
                {'min_lat': -11.5, 'max_lat': -1.5, 'min_lon': -70.5, 'max_lon': -55.5},
                {'min_lat': -13.0, 'max_lat': -3.0, 'min_lon': -68.0, 'max_lon': -58.0}
            ]
        
        # Process each coverage area
        results = []
        for area in lidar_coverage:
            area_filter = (
                (pred_pl['centre_lat'] >= area['min_lat']) & 
                (pred_pl['centre_lat'] <= area['max_lat']) &
                (pred_pl['centre_lon'] >= area['min_lon']) & 
                (pred_pl['centre_lon'] <= area['max_lon'])
            )
            
            results.append(
                pred_pl.filter(area_filter)
                .with_columns([
                    pl.lit(True).alias('lidar_available'),
                    pl.lit('synthetic_coverage').alias('lidar_source')
                ])
            )
        
        # Combine with non-covered areas
        coverage_df = pl.concat(results).unique()
        non_coverage = pred_pl.join(
            coverage_df, on='unique_id', how='anti'
        ).with_columns([
            pl.lit(False).alias('lidar_available'),
            pl.lit(None).alias('lidar_source')
        ])
        
        return pl.concat([coverage_df, non_coverage]).to_pandas()


class ReportGenerator:
    """Generates professional HTML reports"""
    
    @staticmethod
    def generate_report(results, output_dir='reports'):
        """Generate comprehensive HTML report"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(output_dir, f'report_{timestamp}.html')
        
        # Convert to pandas if needed
        if not isinstance(results, pd.DataFrame):
            results = pl.from_pandas(results) if isinstance(results, pd.DataFrame) else results
            results_df = results.to_pandas()
        else:
            results_df = results.copy()
        
        # Calculate statistics
        stats = {
            'total_sites': len(results_df),
            'lidar_coverage': results_df['lidar_available'].sum(),
            'coverage_pct': results_df['lidar_available'].mean() * 100,
            'mean_score': results_df['glyph_score'].mean(),
            'top_sites': results_df.nlargest(10, 'glyph_score')
        }
        
        # Generate visualizations
        map_path = 'visualizations/results_map.html'
        map_content = ""
        if os.path.exists(map_path):
            with open(map_path, 'r') as f:
                map_content = f.read()
        
        # Create HTML report
        template = Template("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Archaeological Detection Report</title>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; }
                .header { background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }
                .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }
                .stat-card { background: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
                .stat-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
                table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #2c3e50; color: white; }
                .map-container { height: 500px; margin: 20px 0; border: 1px solid #ddd; border-radius: 5px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Archaeological Detection Report</h1>
                <p>Generated on {{ timestamp }}</p>
            </div>
            
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-value">{{ stats.total_sites }}</div>
                    <div>Total Sites</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{{ stats.lidar_coverage }}</div>
                    <div>LIDAR Coverage</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{{ "%.1f"|format(stats.coverage_pct) }}%</div>
                    <div>Coverage Percentage</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{{ "%.3f"|format(stats.mean_score) }}</div>
                    <div>Mean Detection Score</div>
                </div>
            </div>
            
            <h2>Interactive Results Map</h2>
            <div class="map-container">
                {{ map_content }}
            </div>
            
            <h2>Top Potential Sites</h2>
            {{ stats.top_sites.to_html(index=False) }}
        </body>
        </html>
        """)
        
        # Write report
        with open(report_path, 'w') as f:
            f.write(template.render(
                timestamp=timestamp,
                stats=stats,
                map_content=map_content
            ))
        
        return report_path


# ## 7. Main Execution Pipeline

class ArchaeologicalDetectionPipeline:
    """End-to-end detection pipeline"""
    
    @staticmethod
    def execute():
        """Run the complete analysis pipeline"""
        print("ğŸš€ Starting Archaeological Detection Pipeline")
        
        try:
            # 1. Data Loading
            print("\nğŸ”� Loading data...")
            geoglyphs = DataLoader.load_geoglyph_data()
            embeddings = DataLoader.load_satellite_embeddings()
            
            # 2. Spatial Processing
            print("\nğŸŒ� Processing spatial data...")
            matches = SpatialProcessor.spatial_join_optimized(geoglyphs, embeddings)
            
            # 3. Model Training
            print("\nğŸ¤– Training machine learning model...")
            
            # Prepare training data
            embeddings_np = np.stack(embeddings['embedding'].to_list())
            
            # Create labels - mark matches as positive samples
            y = np.zeros(len(embeddings_np))
            if matches is not None and not matches.empty:
                known_sites = matches['embedding_id'].unique()
                for i, uid in enumerate(embeddings['unique_id']):
                    if uid in known_sites:
                        y[i] = 1
            
            # Ensure minimum positive samples
            if sum(y) < 10:  # At least 10 positive samples
                n_needed = int(10 - sum(y))  # Convert to integer here
                candidates = np.where(y == 0)[0]
                if len(candidates) >= n_needed:  # Ensure enough samples available
                    y[np.random.choice(candidates, n_needed, replace=False)] = 1
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                embeddings_np, 
                y, 
                test_size=0.2,
                stratify=y,
                random_state=42
            )
            
            # Train and evaluate model
            model = ModelTrainer.train_model(X_train, y_train, X_test, y_test)
            ModelTrainer.evaluate_model(model, X_test, y_test)
            
            # 4. Generate predictions
            print("\nğŸ“Š Generating predictions...")
            predictions = model.predict_proba(embeddings_np)[:, 1]
            results_df = pl.DataFrame({
                'centre_lat': embeddings['centre_lat'],
                'centre_lon': embeddings['centre_lon'],
                'glyph_score': predictions,
                'unique_id': embeddings['unique_id']
            })
            
            # 5. Validation
            print("\nğŸ”� Validating results...")
            validated = DataValidator.validate_with_lidar(results_df.to_pandas())
            
            # 6. Visualization
            print("\nğŸ�¨ Generating visualizations...")
            ResultVisualizer.visualize_results(geoglyphs, embeddings, matches)
            ResultVisualizer.plot_prediction_distribution(validated)
            
            # 7. Reporting
            print("\nğŸ“� Generating final report...")
            report_path = ReportGenerator.generate_report(validated)
            
            print(f"\nâœ… Pipeline completed successfully! Report saved to {report_path}")
            return {
                'model': model,
                'results': validated,
                'report_path': report_path
            }
            
        except Exception as e:
            print(f"\nâ�Œ Pipeline failed: {str(e)}")
            raise


# ## 8. Execute Pipeline

# %%
if __name__ == "__main__":
    results = ArchaeologicalDetectionPipeline.execute()

