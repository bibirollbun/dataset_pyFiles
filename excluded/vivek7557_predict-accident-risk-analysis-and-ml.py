# ==========================================================
# Full EDA + Modeling notebook for 'playground-series-s5e10'
# Predict accident_risk
# ==========================================================

import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Paths
INPUT_DIR = Path("/kaggle/input/playground-series-s5e10")
OUT_DIR = Path("/kaggle/working/eda_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# 0. Imports
# -------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import gc
%matplotlib inline
import plotly.graph_objects as go
import plotly.offline as pyo
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import lightgbm as lgb

print("Libraries loaded")

# Set style for attractive visualizations
sns.set_style("darkgrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# -------------------------
# Load data
# -------------------------
train_path = INPUT_DIR / "train.csv"
test_path = INPUT_DIR / "test.csv"

train = pd.read_csv(train_path)
print("Loaded train:", train.shape)

if test_path.exists():
    test = pd.read_csv(test_path)
    print("Loaded test:", test.shape)
else:
    test = None
    print("Warning: test.csv not found â€” submission will not be created.")


df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")

print("âœ… Dataset Generated Successfully!")
print(f"Shape: {df.shape}")
print("\nColumn Information:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())


test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

print("âœ… Dataset Generated Successfully!")
print(f"Shape: {df.shape}")
print("\nColumn Information:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())


print(f"Dataset Shape: {df.shape}")
print(f"Total Observations: {len(df)}")
print(f"Number of Features: {df.shape[1]}")


print("\nMissing Values:")
print(df.isnull().sum())


print("\nData Types:")
print(df.dtypes)


# Summary statistics for numerical columns
numerical_cols = ['id', 'num_lanes','speed_limit','curvature','num_reported_accidents','accident_risk']
print("\nSummary Statistics (Numerical Features):")
print(df[numerical_cols].describe())


# ============================================================================
#  TARGET VARIABLE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print(" TARGET VARIABLE ANALYSIS (Accident Risk)")
print("="*80)

print(f"\nğŸ“ˆ Accident Risk Statistics:")
print(f"  Mean: {df['accident_risk'].mean():.4f}")
print(f"  Median: {df['accident_risk'].median():.4f}")
print(f"  Std Dev: {df['accident_risk'].std():.4f}")
print(f"  Min: {df['accident_risk'].min():.4f}")
print(f"  Max: {df['accident_risk'].max():.4f}")

# Accident count statistics
print(f"\nğŸš— Number of Accidents Statistics:")
print(f"  Total Accidents: {df['num_reported_accidents'].sum():,}")
print(f"  Mean per segment: {df['num_reported_accidents'].mean():.2f}")
print(f"  Median per segment: {df['num_reported_accidents'].median():.0f}")
print(f"  Max in single segment: {df['num_reported_accidents'].max()}")



# ============================================================================
#  INTERACTIVE  VISUALIZATIONS
# ============================================================================

# Figure 1: Accident Risk Distribution & Number of Accidents Distribution
fig, axes = plt.subplots(1, 2, figsize=(16,6))
sns.histplot(df['accident_risk'], bins=50, color='#E74C3C', ax=axes[0], kde=True)
axes[0].set_title('Accident Risk Distribution')
axes[0].set_xlabel('Accident Risk')
axes[0].set_ylabel('Frequency')

sns.histplot(df['num_reported_accidents'], bins=30, color='#3498DB', ax=axes[1], kde=True)
axes[1].set_title('Number of Accidents Distribution')
axes[1].set_xlabel('Number of Accidents')
axes[1].set_ylabel('Frequency')
plt.suptitle("Target Variable Distributions")
plt.tight_layout()
plt.show()


# Figure 2: Accident Risk by Road Type
risk_by_road = df.groupby('road_type').agg({
    'accident_risk': 'mean',
    'num_reported_accidents': 'sum'
}).reset_index().sort_values('accident_risk', ascending=False)

plt.figure(figsize=(12,6))
sns.barplot(data=risk_by_road, x='road_type', y='accident_risk', palette='Reds')
plt.title('Average Accident Risk by Road Type')
plt.xlabel('Road Type')
plt.ylabel('Average Accident Risk')
plt.show()

print("\nğŸ“Š INSIGHT: Road Type Analysis")
print(f"  Highest Risk: {risk_by_road.iloc[0]['road_type']} ({risk_by_road.iloc[0]['accident_risk']:.4f})")
print(f"  Lowest Risk: {risk_by_road.iloc[-1]['road_type']} ({risk_by_road.iloc[-1]['accident_risk']:.4f})")
print(f"  Risk Range: {(risk_by_road.iloc[0]['accident_risk'] - risk_by_road.iloc[-1]['accident_risk']):.4f}")



# Figure 3: Accident Risk by Time of Day
risk_by_time = df.groupby('time_of_day').agg({
    'accident_risk': 'mean',
    'num_reported_accidents': 'sum'
}).reset_index()

plt.figure(figsize=(10,6))
sns.barplot(data=risk_by_time, x='time_of_day', y='accident_risk', palette='Oranges')
plt.title('Accident Risk by Time of Day')
plt.xlabel('Time of Day')
plt.ylabel('Average Accident Risk')
plt.show()


# Figure 4: Accident Risk by Weather
risk_by_weather = df.groupby('weather').agg({
    'accident_risk': 'mean',
    'num_reported_accidents': 'sum'
}).reset_index().sort_values('accident_risk', ascending=False)

plt.figure(figsize=(12,6))
sns.barplot(data=risk_by_weather, x='weather', y='accident_risk', palette='Blues')
plt.title('Accident Risk by Weather Conditions')
plt.xlabel('Weather')
plt.ylabel('Average Accident Risk')
plt.show()

print("\nğŸŒ¦ï¸� INSIGHT: Weather Impact")
for idx, row in risk_by_weather.iterrows():
    print(f"  {row['weather']}: {row['accident_risk']:.4f} risk, {row['num_reported_accidents']} accidents")



# Figure 5: Accident Risk by Lighting Conditions
risk_by_lighting = df.groupby('lighting').agg({
    'accident_risk': 'mean',
    'num_reported_accidents': 'sum'
}).reset_index().sort_values('accident_risk', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(data=risk_by_lighting, x='lighting', y='accident_risk', palette='Purples')
plt.title('Accident Risk by Lighting Conditions')
plt.xlabel('Lighting')
plt.ylabel('Average Accident Risk')
plt.show()


# Figure 6: Accident Risk Distribution by Speed Limit
plt.figure(figsize=(14,7))
sns.boxplot(data=df, x='speed_limit', y='accident_risk', palette='Set2')
plt.title('Accident Risk Distribution by Speed Limit')
plt.xlabel('Speed Limit (mph)')
plt.ylabel('Accident Risk')
plt.show()

risk_by_speed = df.groupby('speed_limit')['accident_risk'].mean().sort_index()
print("\nğŸš¦ INSIGHT: Speed Limit Analysis")
for speed, risk in risk_by_speed.items():
    print(f"  {speed} mph: {risk:.4f} average risk")



# Figure 7: Number of Lanes Impact
risk_by_lanes = df.groupby('num_lanes').agg({
    'accident_risk': 'mean',
    'num_reported_accidents': 'sum',
    'id': 'count'
}).reset_index()
risk_by_lanes.columns = ['num_lanes', 'avg_risk', 'total_accidents', 'count']

fig, axes = plt.subplots(1,2, figsize=(16,6))
sns.lineplot(data=risk_by_lanes, x='num_lanes', y='avg_risk', marker='o', color='#E74C3C', ax=axes[0])
axes[0].set_title('Risk by Number of Lanes')
axes[0].set_xlabel('Number of Lanes')
axes[0].set_ylabel('Average Risk')

sns.barplot(data=risk_by_lanes, x='num_lanes', y='total_accidents', color='#16A085', ax=axes[1])
axes[1].set_title('Accident Count by Lanes')
axes[1].set_xlabel('Number of Lanes')
axes[1].set_ylabel('Total Accidents')
plt.tight_layout()
plt.show()



# Figure 8: Road Curvature Analysis
df['curvature_category'] = pd.cut(df['curvature'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
risk_by_curve = df.groupby('curvature_category')['accident_risk'].mean().reset_index()

plt.figure(figsize=(10,6))
sns.barplot(data=risk_by_curve, x='curvature_category', y='accident_risk', palette='Greens')
plt.title('Accident Risk by Road Curvature')
plt.xlabel('Curvature Level')
plt.ylabel('Average Accident Risk')
plt.show()


# Figure 9: Boolean Features Analysis
bool_features = ['road_signs_present', 'public_road', 'holiday', 'school_season']
bool_analysis = []

for feature in bool_features:
    stats = df.groupby(feature)['accident_risk'].agg(['mean', 'count']).reset_index()
    for idx, row in stats.iterrows():
        bool_analysis.append({
            'Feature': feature.replace('_', ' ').title(),
            'Value': 'Yes' if row[feature] else 'No',
            'Avg Risk': row['mean'],
            'Count': row['count']
        })

bool_df = pd.DataFrame(bool_analysis)

plt.figure(figsize=(12,6))
sns.barplot(data=bool_df, x='Feature', y='Avg Risk', hue='Value', palette={'Yes':'#27AE60', 'No':'#E74C3C'})
plt.title('Accident Risk by Boolean Features')
plt.xlabel('Feature')
plt.ylabel('Average Risk')
plt.legend(title='Value')
plt.show()

print("\nğŸš§ INSIGHT: Safety Features Impact")
for feature in bool_features:
    with_feature = df[df[feature] == True]['accident_risk'].mean()
    without_feature = df[df[feature] == False]['accident_risk'].mean()
    diff = with_feature - without_feature
    print(f"  {feature.replace('_', ' ').title()}:")
    print(f"    With: {with_feature:.4f} | Without: {without_feature:.4f} | Diff: {diff:+.4f}")



# Figure 10: Correlation Heatmap
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation = df[numeric_cols].corr()

plt.figure(figsize=(12,10))
sns.heatmap(correlation, annot=True, cmap='RdYlBu_r', center=0, fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.show()

print("\nğŸ”— INSIGHT: Key Correlations with Accident Risk")
risk_corr = correlation['accident_risk'].sort_values(ascending=False)
print(risk_corr)


# ============================================================================
# MULTI-DIMENSIONAL ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("SECTION 3: MULTI-DIMENSIONAL ANALYSIS")
print("="*80)

# Combined analysis: Road Type + Weather
pivot_road_weather = df.pivot_table(
    values='accident_risk',
    index='road_type',
    columns='weather',
    aggfunc='mean'
)

plt.figure(figsize=(14,8))
sns.heatmap(pivot_road_weather, annot=True, fmt=".4f", cmap="YlOrRd")
plt.title('Accident Risk: Road Type vs Weather')
plt.xlabel('Weather Condition')
plt.ylabel('Road Type')
plt.show()

# Time of day + Lighting
pivot_time_light = df.pivot_table(
    values='accident_risk',
    index='time_of_day',
    columns='lighting',
    aggfunc='mean'
)

plt.figure(figsize=(12,7))
sns.heatmap(pivot_time_light, annot=True, fmt=".4f", cmap="plasma")
plt.title('Accident Risk: Time of Day vs Lighting')
plt.xlabel('Lighting Condition')
plt.ylabel('Time of Day')
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style for better visuals
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# df = pd.read_csv('your_data.csv')

def analyze_accident_risk(df):
    """
    Comprehensive accident risk analysis with multiple visualizations
    """
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Distribution of Accident Risk
    ax1 = plt.subplot(3, 3, 1)
    sns.histplot(df['accident_risk'], bins=30, kde=True, color='darkred', ax=ax1)
    ax1.set_title('Distribution of Accident Risk', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Accident Risk')
    ax1.set_ylabel('Frequency')
    
    # 2. Accident Risk by Road Type
    ax2 = plt.subplot(3, 3, 2)
    road_type_risk = df.groupby('road_type')['accident_risk'].mean().sort_values(ascending=False)
    sns.barplot(x=road_type_risk.index, y=road_type_risk.values, palette='Reds_r', ax=ax2)
    ax2.set_title('Average Accident Risk by Road Type', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Road Type')
    ax2.set_ylabel('Average Accident Risk')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Accident Risk by Weather Conditions
    ax3 = plt.subplot(3, 3, 3)
    weather_risk = df.groupby('weather')['accident_risk'].mean().sort_values(ascending=False)
    sns.barplot(x=weather_risk.index, y=weather_risk.values, palette='YlOrRd', ax=ax3)
    ax3.set_title('Average Accident Risk by Weather', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Weather Condition')
    ax3.set_ylabel('Average Accident Risk')
    
    # 4. Accident Risk by Lighting Conditions
    ax4 = plt.subplot(3, 3, 4)
    lighting_risk = df.groupby('lighting')['accident_risk'].mean().sort_values(ascending=False)
    sns.barplot(x=lighting_risk.index, y=lighting_risk.values, palette='viridis', ax=ax4)
    ax4.set_title('Average Accident Risk by Lighting', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Lighting Condition')
    ax4.set_ylabel('Average Accident Risk')
    
    # 5. Accident Risk by Time of Day
    ax5 = plt.subplot(3, 3, 5)
    time_risk = df.groupby('time_of_day')['accident_risk'].mean().sort_values(ascending=False)
    colors = ['#ff6b6b', '#feca57', '#48dbfb']
    sns.barplot(x=time_risk.index, y=time_risk.values, palette=colors, ax=ax5)
    ax5.set_title('Average Accident Risk by Time of Day', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Time of Day')
    ax5.set_ylabel('Average Accident Risk')
    
    # 6. Correlation Heatmap
    ax6 = plt.subplot(3, 3, 6)
    numeric_cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents', 'accident_risk']
    correlation = df[numeric_cols].corr()
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax6)
    ax6.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
    
    # 7. Accident Risk vs Speed Limit
    ax7 = plt.subplot(3, 3, 7)
    sns.scatterplot(data=df, x='speed_limit', y='accident_risk', hue='road_type', 
                    style='weather', s=100, alpha=0.6, ax=ax7)
    ax7.set_title('Accident Risk vs Speed Limit', fontsize=14, fontweight='bold')
    ax7.set_xlabel('Speed Limit')
    ax7.set_ylabel('Accident Risk')
    
    # 8. Accident Risk vs Curvature
    ax8 = plt.subplot(3, 3, 8)
    sns.scatterplot(data=df, x='curvature', y='accident_risk', hue='lighting', 
                    s=100, alpha=0.6, ax=ax8)
    ax8.set_title('Accident Risk vs Curvature', fontsize=14, fontweight='bold')
    ax8.set_xlabel('Curvature')
    ax8.set_ylabel('Accident Risk')
    
    # 9. Box plot: Accident Risk by Road Signs Present
    ax9 = plt.subplot(3, 3, 9)
    sns.boxplot(data=df, x='road_signs_present', y='accident_risk', palette='Set2', ax=ax9)
    ax9.set_title('Accident Risk Distribution by Road Signs Presence', fontsize=14, fontweight='bold')
    ax9.set_xlabel('Road Signs Present')
    ax9.set_ylabel('Accident Risk')
    
    plt.tight_layout()
    plt.show()
    
    # Additional Analysis: Statistical Summary
    print("="*70)
    print("ACCIDENT RISK ANALYSIS SUMMARY")
    print("="*70)
    print(f"\nOverall Statistics:")
    print(f"  Mean Accident Risk: {df['accident_risk'].mean():.4f}")
    print(f"  Median Accident Risk: {df['accident_risk'].median():.4f}")
    print(f"  Std Dev: {df['accident_risk'].std():.4f}")
    print(f"  Min Risk: {df['accident_risk'].min():.4f}")
    print(f"  Max Risk: {df['accident_risk'].max():.4f}")
    
    print(f"\n{'='*70}")
    print("Top 5 Risk Factors:")
    print(f"{'='*70}")
    
    # Analyze categorical variables
    for col in ['road_type', 'weather', 'lighting', 'time_of_day']:
        risk_by_category = df.groupby(col)['accident_risk'].mean().sort_values(ascending=False)
        print(f"\n{col.upper().replace('_', ' ')}:")
        for cat, risk in risk_by_category.head().items():
            print(f"  {cat}: {risk:.4f}")
    
    print(f"\n{'='*70}")
    print("High Risk Observations (risk > 0.5):")
    print(f"{'='*70}")
    high_risk = df[df['accident_risk'] > 0.5]
    print(f"Count: {len(high_risk)} observations ({len(high_risk)/len(df)*100:.2f}%)")
    
    return fig

# Interactive Analysis Loop
def interactive_analysis(df):
    """
    Interactive loop for analyzing different aspects of accident risk
    """
    
    while True:
        print("\n" + "="*70)
        print("ACCIDENT RISK ANALYSIS MENU")
        print("="*70)
        print("1. Generate All Visualizations")
        print("2. Analyze by Road Type")
        print("3. Analyze by Weather Conditions")
        print("4. Analyze by Time of Day")
        print("5. Analyze High-Risk Scenarios (risk > threshold)")
        print("6. Compare Two Categories")
        print("7. Show Detailed Statistics")
        print("8. Exit")
        print("="*70)
        
        choice = input("\nEnter your choice (1-8): ").strip()
        
        if choice == '1':
            analyze_accident_risk(df)
            
        elif choice == '2':
            plt.figure(figsize=(12, 6))
            road_stats = df.groupby('road_type').agg({
                'accident_risk': ['mean', 'std', 'count']
            }).round(4)
            print("\nRoad Type Statistics:")
            print(road_stats)
            
            sns.violinplot(data=df, x='road_type', y='accident_risk', palette='muted')
            plt.title('Accident Risk Distribution by Road Type', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            
        elif choice == '3':
            plt.figure(figsize=(12, 6))
            weather_stats = df.groupby('weather').agg({
                'accident_risk': ['mean', 'std', 'count']
            }).round(4)
            print("\nWeather Condition Statistics:")
            print(weather_stats)
            
            sns.violinplot(data=df, x='weather', y='accident_risk', palette='Set1')
            plt.title('Accident Risk Distribution by Weather', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
        elif choice == '4':
            plt.figure(figsize=(12, 6))
            time_stats = df.groupby('time_of_day').agg({
                'accident_risk': ['mean', 'std', 'count']
            }).round(4)
            print("\nTime of Day Statistics:")
            print(time_stats)
            
            sns.violinplot(data=df, x='time_of_day', y='accident_risk', palette='pastel')
            plt.title('Accident Risk Distribution by Time of Day', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
        elif choice == '5':
            threshold = float(input("Enter risk threshold (e.g., 0.5): ").strip())
            high_risk = df[df['accident_risk'] > threshold]
            
            print(f"\nHigh-Risk Scenarios (risk > {threshold}):")
            print(f"Total observations: {len(high_risk)} ({len(high_risk)/len(df)*100:.2f}%)")
            
            if len(high_risk) == 0:
                print(f"\nNo observations found with accident_risk > {threshold}")
                print(f"Maximum risk in dataset: {df['accident_risk'].max():.4f}")
                print("Try a lower threshold.")
                continue
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Distribution by categories
            road_counts = high_risk['road_type'].value_counts()
            if len(road_counts) > 0:
                road_counts.plot(kind='bar', ax=axes[0,0], color='coral')
                axes[0,0].set_title('High-Risk: Road Type Distribution')
                axes[0,0].set_xlabel('Road Type')
            
            weather_counts = high_risk['weather'].value_counts()
            if len(weather_counts) > 0:
                weather_counts.plot(kind='bar', ax=axes[0,1], color='skyblue')
                axes[0,1].set_title('High-Risk: Weather Distribution')
                axes[0,1].set_xlabel('Weather')
            
            lighting_counts = high_risk['lighting'].value_counts()
            if len(lighting_counts) > 0:
                lighting_counts.plot(kind='bar', ax=axes[1,0], color='lightgreen')
                axes[1,0].set_title('High-Risk: Lighting Distribution')
                axes[1,0].set_xlabel('Lighting')
            
            time_counts = high_risk['time_of_day'].value_counts()
            if len(time_counts) > 0:
                time_counts.plot(kind='bar', ax=axes[1,1], color='plum')
                axes[1,1].set_title('High-Risk: Time of Day Distribution')
                axes[1,1].set_xlabel('Time of Day')
            
            plt.tight_layout()
            plt.show()
            
            print("\nTop High-Risk Observations:")
            print(high_risk.nlargest(5, 'accident_risk')[['road_type', 'weather', 'lighting', 
                                                            'speed_limit', 'accident_risk']])
            
        elif choice == '6':
            print("\nAvailable categories:")
            categorical_cols = ['road_type', 'weather', 'lighting', 'time_of_day', 'road_signs_present', 'public_road']
            for i, col in enumerate(categorical_cols, 1):
                print(f"{i}. {col}")
            
            cat1_idx = int(input("Enter first category number: ").strip()) - 1
            cat2_idx = int(input("Enter second category number: ").strip()) - 1
            
            cat1 = categorical_cols[cat1_idx]
            cat2 = categorical_cols[cat2_idx]
            
            pivot = df.groupby([cat1, cat2])['accident_risk'].mean().unstack()
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn_r', center=0.3)
            plt.title(f'Accident Risk: {cat1} vs {cat2}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
        elif choice == '7':
            print("\n" + "="*70)
            print("DETAILED STATISTICAL ANALYSIS")
            print("="*70)
            print("\nDescriptive Statistics:")
            print(df['accident_risk'].describe())
            
            print("\n" + "="*70)
            print("Risk by Multiple Factors:")
            print("="*70)
            multi_factor = df.groupby(['road_type', 'weather', 'lighting'])['accident_risk'].mean().sort_values(ascending=False)
            print(multi_factor.head(10))
            
        elif choice == '8':
            print("\nExiting analysis. Goodbye!")
            break
            
        else:
            print("\nInvalid choice. Please try again.")

# Run the interactive analysis
# Uncomment the line below to start:
interactive_analysis(df)


# ============================================================================
# BUSINESS INSIGHTS & RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print("SECTION 4: KEY FINDINGS & RECOMMENDATIONS")
print("="*80)

print("\nğŸ�¯ TOP 5 HIGHEST RISK SCENARIOS:")
high_risk = df.nlargest(5, 'accident_risk')[['road_type', 'weather', 'time_of_day', 'lighting', 'num_lanes', 'speed_limit', 'accident_risk']]
for idx, row in high_risk.iterrows():
    print(f"\n  Scenario {idx+1}:")
    print(f"    Risk Score: {row['accident_risk']:.4f}")
    print(f"    Road: {row['road_type']} | Lanes: {row['num_lanes']} | Speed: {row['speed_limit']} mph")
    print(f"    Conditions: {row['weather']} | {row['lighting']} | {row['time_of_day']}")

print("\n\nğŸ’¡ STRATEGIC RECOMMENDATIONS:")

# Recommendation 1: Highest risk road types
top_risk_roads = df.groupby('road_type')['accident_risk'].mean().nlargest(3)
print("\n1. PRIORITIZE SAFETY IMPROVEMENTS ON:")
for road, risk in top_risk_roads.items():
    count = len(df[df['road_type'] == road])
    accidents = df[df['road_type'] == road]['num_reported_accidents'].sum()
    print(f"   â€¢ {road}: {risk:.4f} risk | {count:,} segments | {accidents:,} accidents")

# Recommendation 2: Weather-based alerts
severe_weather = df.groupby('weather')['accident_risk'].mean().nlargest(2)
print("\n2. IMPLEMENT WEATHER-BASED ALERTS FOR:")
for weather, risk in severe_weather.items():
    print(f"   â€¢ {weather} conditions: {risk:.4f} risk")

# Recommendation 3: Road signs impact
sign_impact = df[df['road_signs_present'] == False]['accident_risk'].mean() - df[df['road_signs_present'] == True]['accident_risk'].mean()
print(f"\n3. ROAD SIGNAGE PROGRAM:")
print(f"   â€¢ Risk reduction with signs: {-sign_impact:.4f}")
print(f"   â€¢ Segments without signs: {len(df[df['road_signs_present'] == False]):,}")
print(f"   â€¢ Potential impact: HIGH PRIORITY")

# Recommendation 4: Time-based enforcement
time_risk = df.groupby('time_of_day')['accident_risk'].mean().nlargest(2)
print("\n4. ENHANCED ENFORCEMENT DURING:")
for time, risk in time_risk.items():
    print(f"   â€¢ {time}: {risk:.4f} risk")

# Recommendation 5: Speed limit analysis
high_speed_risk = df[df['speed_limit'] >= 60]['accident_risk'].mean()
low_speed_risk = df[df['speed_limit'] < 60]['accident_risk'].mean()
print(f"\n5. SPEED MANAGEMENT:")
print(f"   â€¢ High speed (â‰¥60 mph) risk: {high_speed_risk:.4f}")
print(f"   â€¢ Lower speed (<60 mph) risk: {low_speed_risk:.4f}")
print(f"   â€¢ Consider speed reduction on high-risk segments")

print("\n" + "="*80)
print("END OF ANALYSIS")
print("="*80)


# Kaggle notebook: Accident Risk - End-to-end (EDA -> Modeling -> Ensemble -> Submission)
# This single-cell notebook is designed to be pasted into a Kaggle notebook and run as-is.
# It includes: EDA, preprocessing, feature engineering, K-Fold CV with stratified bins, LightGBM baseline, CatBoost baseline,
# simple stacking, OOF saving, and final submission creation. Add/change things to suit your hardware/time.

import os
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import catboost as cb
import joblib
import matplotlib.pyplot as plt

# Settings
SEED = 42
N_FOLDS = 5
USE_CATBOOST = True
DATA_DIR = Path('/kaggle/input/playground-series-s5e10')
TRAIN_PATH = DATA_DIR / 'train.csv'
TEST_PATH = DATA_DIR / 'test.csv'
SAMPLE_SUB_PATH = DATA_DIR / 'sample_submission.csv'
OUT_DIR = Path('/kaggle/working')
OUT_DIR.mkdir(exist_ok=True)

# Helper functions
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

print('Loading data...')
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
sub = pd.read_csv(SAMPLE_SUB_PATH)

print('Train shape:', train.shape)
print('Test shape:', test.shape)

# Quick EDA (counts, types, missing)
print('\n--- Train head ---')
print(train.head())
print('\n--- dtypes ---')
print(train.dtypes.value_counts())
print('\n--- missing per column (train) ---')
print(train.isna().mean().sort_values(ascending=False).head(20))

# Target
TARGET = 'accident_risk'
if TARGET not in train.columns:
    raise ValueError('Target column not found in train')

# Basic distribution
print('\nTarget summary:')
print(train[TARGET].describe())

# Make a copy of id columns if present
ID_COL = 'id' if 'id' in train.columns else None

# Merge train+test for joint preprocessing
train['is_train'] = 1
test['is_train'] = 0

# Save shapes
n_train = train.shape[0]

df = pd.concat([train, test], ignore_index=True)
print('\nCombined shape:', df.shape)

# Identify numeric and categorical
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
if ID_COL in num_cols:
    num_cols.remove(ID_COL)
if TARGET in num_cols:
    num_cols.remove(TARGET)
if 'is_train' in num_cols:
    num_cols.remove('is_train')

print('\nNumeric cols count:', len(num_cols))
print('Categorical cols count:', len(cat_cols))

# Basic feature engineering ideas (customize per dataset):
# 1) If there are timestamp columns, extract hour/day/month
# 2) Create interaction features for risk (e.g., speed*weather)
# 3) Target-based aggregation from train set (mean target per category)

# Example: if dataset contains lat/lon or continuous sensors, make rolling stats, but we can't assume them here.

# === Target encoding for categorical features ===
# We'll compute mean target per category using only train rows to avoid leakage.
from collections import defaultdict

def target_encode(train_series, target, min_samples_leaf=100, smoothing=10):
    # Bayesian smoothing target encoding
    data = pd.DataFrame({ 'cat': train_series, 'target': target })
    mean = target.mean()
    agg = data.groupby('cat')['target'].agg(['count','mean'])
    counts = agg['count']
    means = agg['mean']
    smooth = (counts * means + smoothing * mean) / (counts + smoothing)
    return smooth

# Apply basic target encoding for object cols (only if some exist)
for c in cat_cols:
    # compute mapping from train rows
    train_mask = df['is_train']==1
    mapping = target_encode(df.loc[train_mask, c], df.loc[train_mask, TARGET])
    df[c+'_te'] = df[c].map(mapping)
    # fillna with global mean
    df[c+'_te'] = df[c+'_te'].fillna(df.loc[train_mask, TARGET].mean())

# Drop original object columns to avoid high-cardinality problems (unless you want to label encode)
for c in cat_cols:
    df.drop(columns=[c], inplace=True)

# === Missing value handling ===
# Impute numeric cols with median
imputer = SimpleImputer(strategy='median')
df[num_cols] = imputer.fit_transform(df[num_cols])

# === Scaling (optional) ===
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Split back
train = df[df['is_train']==1].drop(columns=['is_train']).reset_index(drop=True)
test = df[df['is_train']==0].drop(columns=['is_train', TARGET], errors='ignore').reset_index(drop=True)

FEATURES = [c for c in train.columns if c not in [ID_COL, TARGET]]
print('\nNum features used:', len(FEATURES))

# Create stratified folds by binning the target
bins = np.linspace(train[TARGET].min(), train[TARGET].max(), 20)
train['_bin'] = np.digitize(train[TARGET], bins)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds = list(skf.split(train, train['_bin']))

# LightGBM parameters (baseline)
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_data_in_leaf': 100,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'seed': SEED,
    'verbosity': -1
}

# Out-of-fold predictions
oof = np.zeros(len(train))
preds_test = np.zeros(len(test))

print('\nTraining LightGBM...')
for fold, (tr_idx, val_idx) in enumerate(folds):
    X_tr = train.loc[tr_idx, FEATURES]
    y_tr = train.loc[tr_idx, TARGET]
    X_val = train.loc[val_idx, FEATURES]
    y_val = train.loc[val_idx, TARGET]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dvalid = lgb.Dataset(X_val, label=y_val)

    
    clf = lgb.train(
    lgb_params,
    dtrain,
    num_boost_round=5000,
    valid_sets=[dtrain, dvalid],
    callbacks=[
        lgb.early_stopping(100),
        lgb.log_evaluation(100)
    ])


    oof[val_idx] = clf.predict(X_val, num_iteration=clf.best_iteration)
    preds_test += clf.predict(test[FEATURES], num_iteration=clf.best_iteration) / N_FOLDS

    print(f'Fold {fold} RMSE: {rmse(y_val, oof[val_idx]):.6f}')

print('\nOverall OOF RMSE:', rmse(train[TARGET], oof))

# Save OOF and test preds
train['oof_lgb'] = oof
joblib.dump(clf, OUT_DIR / 'lgb_final.pkl')

# CatBoost (optional) - simple baseline
if USE_CATBOOST:
    print('\nTraining CatBoost...')
    oof_cb = np.zeros(len(train))
    preds_test_cb = np.zeros(len(test))
    cb_params = {
        'iterations': 2000,
        'learning_rate': 0.03,
        'depth': 6,
        'loss_function': 'RMSE',
        'random_seed': SEED,
        'early_stopping_rounds': 100,
        'verbose': 100
    }
    for fold, (tr_idx, val_idx) in enumerate(folds):
        X_tr = train.loc[tr_idx, FEATURES]
        y_tr = train.loc[tr_idx, TARGET]
        X_val = train.loc[val_idx, FEATURES]
        y_val = train.loc[val_idx, TARGET]

        model = cb.CatBoostRegressor(**cb_params)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val))

        oof_cb[val_idx] = model.predict(X_val)
        preds_test_cb += model.predict(test[FEATURES]) / N_FOLDS

        print(f'CB Fold {fold} RMSE: {rmse(y_val, oof_cb[val_idx]):.6f}')

    print('\nOverall CatBoost OOF RMSE:', rmse(train[TARGET], oof_cb))
    train['oof_cb'] = oof_cb
    joblib.dump(model, OUT_DIR / 'catboost_final.pkl')

    # Simple ensemble (average)
    ensemble_oof = 0.6 * train['oof_lgb'] + 0.4 * train['oof_cb']
    print('\nSimple ensemble OOF RMSE:', rmse(train[TARGET], ensemble_oof))

    final_test_pred = 0.6 * preds_test + 0.4 * preds_test_cb
else:
    final_test_pred = preds_test

# Prepare submission
submission = sub.copy()
submission['accident_risk'] = final_test_pred
submission.to_csv(OUT_DIR / 'submission.csv', index=False)
print('\nSaved submission to', OUT_DIR / 'submission.csv')

# Feature importance for LightGBM (if booster has feature_importance)
try:
    fi = pd.DataFrame({'feature': FEATURES, 'importance': clf.feature_importance()})
    fi.sort_values('importance', ascending=False, inplace=True)
    print('\nTop features (LGB):')
    print(fi.head(30))
    fi.to_csv(OUT_DIR / 'lgb_feature_importance.csv', index=False)
except Exception as e:
    print('Could not extract feature importance:', e)

# Final notes and next steps printed for user to read in notebook
print('\n--- Done.\nNext recommended steps:')
print('1) Create new features tailored to domain knowledge (sensor combos, temporal aggregations).')
print('2) Use more advanced CV: time-aware or grouped folds if applicable.')
print('3) Run hyperparameter tuning (Optuna) for LGB/CB and try more model types (XGBoost, TabPFN, TabNet).')
print('4) Create stacking (meta-model) using OOF predictions.')
print('5) Use ensembling + model averaging with different seeds and feature subsets for robustness.')
print('6) Carefully check for data leakage and target leakage from future information.')

# Save OOF files for stacking
train[['id', TARGET, 'oof_lgb']].to_csv(OUT_DIR / 'train_oof_lgb.csv', index=False)
if USE_CATBOOST:
    train[['id', 'oof_cb']].to_csv(OUT_DIR / 'train_oof_cb.csv', index=False)

print('\nArtifacts saved to working directory: lgb_final.pkl, catboost_final.pkl (if used), submission.csv, feature importance, OOF csvs')





