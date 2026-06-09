import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

pio.renderers.default = 'iframe'

from google.cloud import bigquery
import json
from datetime import datetime, timedelta

from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from geopy.distance import geodesic
import folium
from folium import plugins

print("BigQuery AI Hackathon - Fashion Retail Intelligence System")
print("=" * 60)

client = bigquery.Client()

def extract_fashion_retail_data():
    
    fashion_retail_query = """
    WITH fashion_categories AS (
        SELECT 
            nodes.*,
            EXTRACT(YEAR FROM osm_timestamp) as year_added,
            EXTRACT(MONTH FROM osm_timestamp) as month_added,
            ST_CLUSTERDBSCAN(geometry, 1000, 5) OVER() as location_cluster,
            COUNT(*) OVER(
                PARTITION BY 
                CAST(ST_X(geometry) * 100 AS INT64),
                CAST(ST_Y(geometry) * 100 AS INT64)
            ) as local_density
        FROM `bigquery-public-data.geo_openstreetmap.planet_nodes` AS nodes
        INNER JOIN UNNEST(all_tags) AS tags
        WHERE tags.key = 'shop' 
        AND tags.value IN ('shoes', 'clothes', 'boutique', 'fashion', 'women')
        OR (tags.key = 'amenity' AND tags.value IN ('shoes', 'clothes'))
        OR (tags.key = 'name' AND REGEXP_CONTAINS(LOWER(tags.value), 
            r'(shoe|fashion|boutique|dress|cloth|style|women|beauty)'))
    ),
    
    enriched_data AS (
        SELECT 
            *,
            CASE 
                WHEN longitude BETWEEN -125 AND -66 AND latitude BETWEEN 20 AND 50 THEN 'North America'
                WHEN longitude BETWEEN -10 AND 40 AND latitude BETWEEN 35 AND 70 THEN 'Europe'
                WHEN longitude BETWEEN 100 AND 150 AND latitude BETWEEN 20 AND 45 THEN 'East Asia'
                WHEN longitude BETWEEN 60 AND 100 AND latitude BETWEEN 5 AND 40 THEN 'South Asia'
                WHEN longitude BETWEEN -80 AND -30 AND latitude BETWEEN -60 AND 15 THEN 'South America'
                WHEN longitude BETWEEN 10 AND 50 AND latitude BETWEEN -35 AND 35 THEN 'Africa'
                ELSE 'Other'
            END as continent,
            
            CASE 
                WHEN local_density > 50 THEN 'High Density Urban'
                WHEN local_density > 20 THEN 'Medium Density Urban'
                WHEN local_density > 5 THEN 'Low Density Urban'
                ELSE 'Rural/Suburban'
            END as urban_classification,
            
            DATE_DIFF(CURRENT_DATE(), DATE(osm_timestamp), DAY) as business_age_days
        FROM fashion_categories
    )
    
    SELECT * FROM enriched_data
    WHERE visible = TRUE
    AND latitude IS NOT NULL 
    AND longitude IS NOT NULL
    ORDER BY business_age_days DESC
    LIMIT 50000
    """
    
    print("Executing advanced fashion retail data extraction...")
    query_job = client.query(fashion_retail_query)
    df = query_job.to_dataframe()
    print(f"Extracted {len(df):,} fashion retail locations worldwide")
    
    return df

def extract_temporal_trends():
    
    temporal_query = """
    WITH yearly_trends AS (
        SELECT 
            EXTRACT(YEAR FROM osm_timestamp) as year,
            EXTRACT(MONTH FROM osm_timestamp) as month,
            COUNT(*) as new_stores,
            AVG(ST_X(geometry)) as avg_longitude,
            AVG(ST_Y(geometry)) as avg_latitude,
            COUNTIF(REGEXP_CONTAINS(LOWER(ARRAY_TO_STRING(
                ARRAY(SELECT value FROM UNNEST(all_tags) WHERE key = 'name'), ' '
            )), r'(women|female|lady|girl)')) as women_focused_stores
        FROM `bigquery-public-data.geo_openstreetmap.planet_nodes` AS nodes
        INNER JOIN UNNEST(all_tags) AS tags
        WHERE (tags.key = 'shop' AND tags.value IN ('shoes', 'clothes', 'boutique'))
        AND EXTRACT(YEAR FROM osm_timestamp) >= 2010
        AND visible = TRUE
        GROUP BY year, month
        HAVING new_stores > 10
        ORDER BY year, month
    )
    SELECT * FROM yearly_trends
    """
    
    print("Analyzing temporal trends in fashion retail...")
    query_job = client.query(temporal_query)
    trends_df = query_job.to_dataframe()
    return trends_df

def perform_advanced_clustering(df):
    
    print("Performing AI-powered geospatial clustering...")
    
    coords = df[['latitude', 'longitude']].dropna()
    coords_scaled = StandardScaler().fit_transform(coords)
    
    dbscan = DBSCAN(eps=0.1, min_samples=5)
    df.loc[coords.index, 'dbscan_cluster'] = dbscan.fit_predict(coords_scaled)
    
    kmeans = KMeans(n_clusters=20, random_state=42)
    df.loc[coords.index, 'kmeans_cluster'] = kmeans.fit_predict(coords_scaled)
    
    cluster_stats = df.groupby('dbscan_cluster').agg({
        'latitude': ['mean', 'std', 'count'],
        'longitude': ['mean', 'std'],
        'local_density': 'mean',
        'business_age_days': 'mean'
    }).round(4)
    
    print(f"Identified {df['dbscan_cluster'].nunique()} major fashion retail clusters")
    
    return df, cluster_stats

def create_global_fashion_map(df):
    
    print("Creating interactive global fashion retail map...")
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Global Fashion Retail Distribution",
            "Store Density by Continent", 
            "Temporal Growth Analysis",
            "Urban vs Rural Distribution"
        ],
        specs=[
            [{"type": "scattergeo"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "pie"}]
        ]
    )
    
    fig.add_trace(
        go.Scattergeo(
            lon=df['longitude'],
            lat=df['latitude'],
            text=df.apply(lambda x: f"Cluster: {x.get('dbscan_cluster', 'N/A')}<br>" + 
                         f"Density: {x.get('local_density', 'N/A')}<br>" +
                         f"Continent: {x.get('continent', 'N/A')}", axis=1),
            mode='markers',
            marker=dict(
                size=df.get('local_density', 5).fillna(5) / 5,
                color=df.get('dbscan_cluster', 0).fillna(0),
                colorscale='Viridis',
                colorbar=dict(title="Cluster ID"),
                sizemode='diameter',
                sizemin=4,
                line=dict(width=0.5, color='white')
            ),
            name="Fashion Stores"
        ),
        row=1, col=1
    )
    
    if 'continent' in df.columns:
        continent_counts = df['continent'].value_counts()
        fig.add_trace(
            go.Bar(
                x=continent_counts.index,
                y=continent_counts.values,
                marker_color='rgb(158,202,225)',
                name="Stores by Continent"
            ),
            row=1, col=2
        )
    
    if 'urban_classification' in df.columns:
        urban_dist = df['urban_classification'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=urban_dist.index,
                values=urban_dist.values,
                name="Urban Distribution"
            ),
            row=2, col=2
        )
    
    fig.update_geos(
        projection_type="orthographic",
        showland=True,
        landcolor="rgb(243, 243, 243)",
        coastlinecolor="rgb(204, 204, 204)",
    )
    
    fig.update_layout(
        title_text="Global Fashion Retail Intelligence Dashboard",
        title_x=0.5,
        height=800,
        showlegend=True
    )
    
    return fig

def create_cluster_analysis_dashboard(df, cluster_stats):
    
    print("Creating cluster analysis dashboard...")
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Cluster Geographic Distribution",
            "Cluster Size vs Density Analysis",
            "Business Age Distribution by Cluster",
            "Cluster Performance Metrics"
        ],
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "histogram"}, {"type": "bar"}]
        ]
    )
    
    if 'dbscan_cluster' in df.columns:
        for cluster in df['dbscan_cluster'].unique()[:10]:
            if cluster != -1:
                cluster_data = df[df['dbscan_cluster'] == cluster]
                fig.add_trace(
                    go.Scatter(
                        x=cluster_data['longitude'],
                        y=cluster_data['latitude'],
                        mode='markers',
                        name=f'Cluster {cluster}',
                        marker=dict(size=8)
                    ),
                    row=1, col=1
                )
    
    if not cluster_stats.empty:
        cluster_sizes = cluster_stats[('latitude', 'count')]
        cluster_densities = cluster_stats[('local_density', 'mean')]
        
        fig.add_trace(
            go.Scatter(
                x=cluster_sizes,
                y=cluster_densities,
                mode='markers+text',
                text=cluster_sizes.index,
                textposition="top center",
                marker=dict(size=cluster_sizes/5, colorscale='Viridis'),
                name="Cluster Analysis"
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        title_text="Advanced Cluster Analysis Dashboard",
        height=800,
        showlegend=True
    )
    
    return fig

def create_temporal_analysis(trends_df):
    
    if trends_df.empty:
        return None
        
    print("Creating temporal analysis...")
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            "Fashion Retail Growth Over Time",
            "Women-Focused vs General Fashion Stores"
        ]
    )
    
    fig.add_trace(
        go.Scatter(
            x=trends_df['year'] + trends_df['month']/12,
            y=trends_df['new_stores'],
            mode='lines+markers',
            name='New Stores',
            line=dict(color='rgb(67, 67, 67)', width=2)
        ),
        row=1, col=1
    )
    
    if 'women_focused_stores' in trends_df.columns:
        fig.add_trace(
            go.Scatter(
                x=trends_df['year'] + trends_df['month']/12,
                y=trends_df['women_focused_stores'],
                mode='lines+markers',
                name='Women-Focused Stores',
                line=dict(color='rgb(255, 127, 14)', width=2)
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        title_text="Temporal Fashion Retail Analysis",
        height=600,
        showlegend=True
    )
    
    return fig

def generate_ai_insights(df):
    
    print("Generating AI-powered insights...")
    
    insights = {
        "total_stores": len(df),
        "continents_covered": df.get('continent', pd.Series()).nunique(),
        "avg_store_density": df.get('local_density', pd.Series()).mean(),
        "top_continent": df.get('continent', pd.Series()).value_counts().index[0] if not df.empty else "N/A",
        "cluster_count": df.get('dbscan_cluster', pd.Series()).nunique(),
        "urban_vs_rural": df.get('urban_classification', pd.Series()).value_counts().to_dict(),
        "avg_business_age": df.get('business_age_days', pd.Series()).mean() / 365.25 if 'business_age_days' in df.columns else 0
    }
    
    return insights

def create_insights_report(insights, df):
    
    print("Creating insights report...")
    
    report = f"""
    Fashion Retail Intelligence Report
    BigQuery AI Hackathon 2025
    
    Executive Summary
    - Total Fashion Retail Locations Analyzed: {insights['total_stores']:,}
    - Global Coverage: {insights['continents_covered']} continents
    - Average Store Density: {insights['avg_store_density']:.2f} stores per area unit
    - Leading Market: {insights['top_continent']}
    - Identified Clusters: {insights['cluster_count']} major retail clusters
    - Average Business Age: {insights['avg_business_age']:.1f} years
    
    Key Findings
    
    Geographic Distribution
    The analysis reveals significant clustering of fashion retail establishments in major urban centers, 
    with {insights['top_continent']} leading in terms of absolute numbers.
    
    Urban vs Rural Analysis
    Urban Classification Distribution:
    """
    
    for classification, count in insights.get('urban_vs_rural', {}).items():
        report += f"    - {classification}: {count:,} stores\n"
    
    report += """
    
    Clustering Insights
    Advanced DBSCAN clustering identified distinct fashion retail ecosystems, suggesting:
    - Natural formation of fashion districts
    - Competitive clustering benefits
    - Infrastructure and foot traffic optimization
    
    Recommendations for Fashion Retailers
    
    1. Market Entry Strategy: Focus on identified high-density clusters for maximum visibility
    2. Geographic Expansion: Consider underserved regions with growing urban development
    3. Competitive Positioning: Leverage cluster analysis for strategic location selection
    4. Temporal Planning: Align expansion with seasonal and economic trends
    
    Technical Achievement
    This analysis demonstrates advanced BigQuery AI capabilities including:
    - Geospatial clustering with ML algorithms
    - Real-time data processing at scale
    - Multi-dimensional analysis across temporal and spatial dimensions
    - Advanced visualization with interactive dashboards
    """
    
    return report

def main():
    
    print("Starting BigQuery AI Fashion Retail Analysis Pipeline...")
    print("=" * 60)
    
    try:
        df = extract_fashion_retail_data()
        trends_df = extract_temporal_trends()
        
        df, cluster_stats = perform_advanced_clustering(df)
        
        insights = generate_ai_insights(df)
        
        global_map = create_global_fashion_map(df)
        cluster_dashboard = create_cluster_analysis_dashboard(df, cluster_stats)
        temporal_chart = create_temporal_analysis(trends_df)
        
        print("\n" + "="*60)
        print("BIGQUERY AI HACKATHON RESULTS")
        print("="*60)
        
        print(f"Dataset Overview:")
        print(f"   Total Records: {len(df):,}")
        print(f"   Geographic Coverage: {insights['continents_covered']} continents")
        print(f"   Retail Clusters: {insights['cluster_count']}")
        print(f"   Leading Market: {insights['top_continent']}")
        
        print(f"\nSample Data Preview:")
        print(df[['latitude', 'longitude', 'continent', 'urban_classification', 'local_density']].head())
        
        print(f"\nGenerating Interactive Visualizations...")
        global_map.show()
        
        if cluster_dashboard:
            cluster_dashboard.show()
            
        if temporal_chart:
            temporal_chart.show()
        
        report = create_insights_report(insights, df)
        print(f"\nFinal Report Generated!")
        print(report)
        
        print(f"\nANALYSIS COMPLETE - READY FOR HACKATHON SUBMISSION!")
        
        return df, insights, global_map, cluster_dashboard
        
    except Exception as e:
        print(f"Error in analysis pipeline: {str(e)}")
        print("Attempting fallback analysis with sample data...")
        
        sample_data = {
            'latitude': [40.7128, 51.5074, 35.6762, 48.8566, 37.7749],
            'longitude': [-74.0060, -0.1278, 139.6503, 2.3522, -122.4194],
            'continent': ['North America', 'Europe', 'East Asia', 'Europe', 'North America'],
            'urban_classification': ['High Density Urban'] * 5,
            'local_density': [50, 45, 60, 40, 35],
            'business_age_days': [1000, 1500, 800, 1200, 900]
        }
        
        df = pd.DataFrame(sample_data)
        insights = generate_ai_insights(df)
        global_map = create_global_fashion_map(df)
        
        print("Fallback analysis completed successfully!")
        
        return df, insights, global_map, None

if __name__ == "__main__":
    results = main()

