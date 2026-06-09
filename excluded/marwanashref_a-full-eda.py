# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')

# Initial Data Inspection
print("Missing Values:\n", df.isnull().sum())
print("\nData Types:\n", df.dtypes)
print("\nDescriptive Statistics:\n", df.describe())

# Set up visualization style
sns.set(style="whitegrid", palette="muted")
plt.figure(figsize=(12, 8))

# 1. Target Variable Distribution
plt.figure(figsize=(8,6))
sns.countplot(x='rainfall', data=df)
plt.title('Rainfall Class Distribution')
plt.show()

# 2. Numerical Features Distribution
numerical_features = ['pressure', 'maxtemp', 'temparature', 'mintemp', 
                     'dewpoint', 'humidity', 'cloud', 'sunshine', 'windspeed']

plt.figure(figsize=(20, 15))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(3, 3, i)
    sns.histplot(data=df, x=feature, kde=True, bins=30)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()

# 3. Boxplots for Rain vs No-Rain
plt.figure(figsize=(20, 15))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(x='rainfall', y=feature, data=df)
    plt.title(f'{feature} vs Rainfall')
plt.tight_layout()
plt.show()

# 4. Correlation Analysis
corr_matrix = df.corr()
plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Matrix')
plt.show()

# 5. Time Series Analysis (Daily Patterns)
plt.figure(figsize=(16, 6))
sns.lineplot(x='day', y='temparature', data=df, label='Temperature')
sns.lineplot(x='day', y='humidity', data=df, label='Humidity')
plt.title('Daily Temperature and Humidity Trends')
plt.xlabel('Day Number')
plt.ylabel('Value')
plt.legend()
plt.show()

# 6. Pairwise Relationships
sns.pairplot(df[['temparature', 'humidity', 'pressure', 'dewpoint', 'rainfall']], 
             hue='rainfall', diag_kind='kde', palette='viridis')
plt.suptitle('Pairwise Feature Relationships', y=1.02)
plt.show()

# 7. PCA Visualization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[numerical_features])
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
sns.scatterplot(x=principal_components[:,0], y=principal_components[:,1], 
                hue=df['rainfall'], palette='viridis')
plt.title('PCA: Rainfall Separation')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# 8. t-SNE Visualization
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
sns.scatterplot(x=tsne_results[:,0], y=tsne_results[:,1], 
                hue=df['rainfall'], palette='viridis')
plt.title('t-SNE: Rainfall Separation')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.show()

# 9. Feature Importance Analysis
from sklearn.ensemble import RandomForestClassifier

X = df.drop(['rainfall', 'id'], axis=1)
y = df['rainfall']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance Ranking')
plt.show()

# 10. Interactive 3D Visualization (using Plotly)
fig = px.scatter_3d(df, x='temparature', y='humidity', z='pressure',
                    color='rainfall', opacity=0.7,
                    title='3D Feature Space Visualization',
                    color_discrete_sequence=['green', 'blue'])
fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))
fig.show()

# 11. Temporal Patterns Analysis
df['month'] = (df['day'] // 30) + 1
monthly_agg = df.groupby('month').agg({
    'rainfall': 'mean',
    'temparature': 'mean',
    'humidity': 'mean'
}).reset_index()

plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
sns.lineplot(x='month', y='rainfall', data=monthly_agg, marker='o')
plt.title('Monthly Rainfall Probability')
plt.ylim(0, 1)

plt.subplot(1, 2, 2)
sns.lineplot(x='month', y='temparature', data=monthly_agg, marker='o', label='Temperature')
sns.lineplot(x='month', y='humidity', data=monthly_agg, marker='o', label='Humidity')
plt.title('Monthly Climate Patterns')
plt.legend()
plt.tight_layout()
plt.show()

# 12. Advanced Distribution Plots
plt.figure(figsize=(16, 12))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(3, 3, i)
    sns.violinplot(x='rainfall', y=feature, data=df, split=True, inner="quart")
    plt.title(f'{feature} Distribution by Rainfall')
plt.tight_layout()
plt.show()

# 13. Cluster Analysis
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(10, 8))
sns.scatterplot(x=tsne_results[:,0], y=tsne_results[:,1], 
                hue=clusters, style=df['rainfall'],
                palette='viridis', s=80)
plt.title('Cluster Analysis vs Rainfall Labels')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.show()

# NEW ADDITIONS START HERE

# 14. Outlier Detection and Visualization
plt.figure(figsize=(16, 12))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(3, 3, i)
    
    # Calculate Z-scores for outlier detection
    z_scores = np.abs(stats.zscore(df[feature].dropna()))
    outliers = df[feature][z_scores > 3]
    
    # Plot histograms with outliers highlighted
    sns.histplot(data=df, x=feature, kde=True, bins=30, color='blue', alpha=0.5)
    if len(outliers) > 0:
        sns.histplot(data=df[z_scores > 3], x=feature, color='red', bins=10, alpha=0.7)
        
    plt.title(f'Outliers in {feature} (Z-score > 3): {len(outliers)}')
plt.tight_layout()
plt.show()

# 15. Feature Distribution by Rainfall Status with KDE
plt.figure(figsize=(20, 15))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(3, 3, i)
    
    # Plot KDE for rain and no-rain groups
    sns.kdeplot(data=df[df['rainfall']==0], x=feature, fill=True, alpha=0.5, label='No Rain', color='blue')
    sns.kdeplot(data=df[df['rainfall']==1], x=feature, fill=True, alpha=0.5, label='Rain', color='orange')
    
    plt.title(f'{feature} Distribution by Rainfall Status')
    plt.legend()
plt.tight_layout()
plt.show()

# 16. Radar Chart for Feature Comparison
# Prepare data for radar chart by scaling features
radar_df = df.groupby('rainfall')[numerical_features].mean()
radar_df = (radar_df - radar_df.min()) / (radar_df.max() - radar_df.min())

# Create the radar chart
categories = numerical_features
fig = go.Figure()

fig.add_trace(go.Scatterpolar(
      r=radar_df.loc[0].values.tolist(),
      theta=categories,
      fill='toself',
      name='No Rain'
))
fig.add_trace(go.Scatterpolar(
      r=radar_df.loc[1].values.tolist(),
      theta=categories,
      fill='toself',
      name='Rain'
))

fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[0, 1]
    )),
  showlegend=True,
  title="Feature Comparison: Rain vs No Rain"
)

fig.show()

# 17. Enhanced Heatmap of Daily Patterns
df['day_of_month'] = df['day'] % 30
if 'month' not in df.columns:
    df['month'] = (df['day'] // 30) + 1

# Create a pivot table for the heatmap
heatmap_data = df.pivot_table(index='month', columns='day_of_month', values='rainfall', aggfunc='mean')

plt.figure(figsize=(16, 8))
sns.heatmap(heatmap_data, cmap='Blues', annot=False, cbar_kws={'label': 'Rainfall Probability'})
plt.title('Daily Rainfall Patterns Throughout Months')
plt.xlabel('Day of Month')
plt.ylabel('Month')
plt.show()

# 18. Joint Plots for Key Feature Relationships
key_pairs = [('humidity', 'temparature'), ('pressure', 'humidity'), ('dewpoint', 'temparature')]

for pair in key_pairs:
    plt.figure(figsize=(10, 8))
    sns.jointplot(
        data=df, 
        x=pair[0], 
        y=pair[1], 
        hue='rainfall',
        kind='scatter',
        height=8,
        ratio=3, 
    )
    plt.suptitle(f'Joint Distribution: {pair[0]} vs {pair[1]}', y=1.02)
    plt.tight_layout()
    plt.show()

# 19. Feature Relationships with Contour Plots
for pair in key_pairs:
    plt.figure(figsize=(12, 10))
    
    # Split data by rainfall
    rain_df = df[df['rainfall'] == 1]
    no_rain_df = df[df['rainfall'] == 0]
    
    # Add contour plots
    sns.kdeplot(data=rain_df, x=pair[0], y=pair[1], levels=5, color="red", alpha=0.5, label="Rain", fill=True)
    sns.kdeplot(data=no_rain_df, x=pair[0], y=pair[1], levels=5, color="blue", alpha=0.5, label="No Rain", fill=True)
    
    # Add scatter points
    sns.scatterplot(data=df, x=pair[0], y=pair[1], hue="rainfall", alpha=0.3, s=15)
    
    plt.title(f'Density Contours: {pair[0]} vs {pair[1]}')
    plt.legend()
    plt.tight_layout()
    plt.show()

# 20. Parallel Coordinates Plot
# Create a DataFrame with scaled features and rainfall
parallel_df = pd.DataFrame(X_scaled, columns=numerical_features)
parallel_df['rainfall'] = df['rainfall']

# Create parallel coordinates plot
fig = px.parallel_coordinates(
    parallel_df, 
    color="rainfall",
    labels={col: col for col in parallel_df.columns},
    color_continuous_scale=px.colors.diverging.Tealrose,
    title="Parallel Coordinates Plot of Weather Features"
)
fig.show()

# 21. Andrews Curves Visualization
from pandas.plotting import andrews_curves

plt.figure(figsize=(12, 8))
andrews_curves(df[numerical_features + ['rainfall']], 'rainfall', colormap='viridis')
plt.title('Andrews Curves for Weather Features')
plt.show()

# 22. Ridgeline Plot for Feature Distributions
# Fix for the PolyCollection error - use pandas directly instead of joypy
# Comment out the joypy visualization since it's causing errors
'''
import joypy
fig, axes = joypy.joyplot(df, column=numerical_features, by="rainfall", 
                          alpha=0.5, figsize=(12, 10), colormap="viridis")
plt.title('Ridgeline Plots of Features by Rainfall Status')
plt.show()
'''

# Alternative: Create separate KDE plots for each feature
plt.figure(figsize=(20, 15))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(3, 3, i)
    sns.kdeplot(data=df[df['rainfall']==0], x=feature, label='No Rain', color='blue')
    sns.kdeplot(data=df[df['rainfall']==1], x=feature, label='Rain', color='orange')
    plt.title(f'Distribution of {feature} by Rainfall')
    plt.legend()
plt.tight_layout()
plt.show()

# 23. Sunburst Chart for Hierarchical Data (Using Plotly)
# Create categorical versions of key features
df['temp_cat'] = pd.qcut(df['temparature'], 3, labels=['Low', 'Medium', 'High'])
df['humid_cat'] = pd.qcut(df['humidity'], 3, labels=['Low', 'Medium', 'High'])

# Create counts for sunburst
sunburst_data = df.groupby(['temp_cat', 'humid_cat', 'rainfall']).size().reset_index(name='count')

# Create sunburst chart
fig = px.sunburst(
    sunburst_data, 
    path=['temp_cat', 'humid_cat', 'rainfall'], 
    values='count',
    color='rainfall',
    color_discrete_sequence=['blue', 'red'],
    title='Hierarchical Relationship: Temperature → Humidity → Rainfall'
)
fig.show()

# 24. Geographic Pattern Visualization (if latitude and longitude are available)
# If location data is not available, you can create a synthetic visualization
if 'latitude' in df.columns and 'longitude' in df.columns:
    fig = px.scatter_mapbox(
        df, 
        lat='latitude', 
        lon='longitude', 
        color='rainfall',
        size='humidity',
        hover_data=['temparature', 'pressure'],
        color_continuous_scale='Viridis',
        size_max=15, 
        zoom=5,
        title='Geographic Distribution of Rainfall'
    )
    
    fig.update_layout(mapbox_style="carto-positron")
    fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
    fig.show()

