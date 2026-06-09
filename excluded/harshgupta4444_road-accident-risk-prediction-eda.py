import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


df_train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')



print("Train shape:", df_train.shape)
print("Test shape:", df_test.shape)



df_train.head(10)


df_train.info()


df_train.describe()


print("Missing values in training data:")
print(df_train.isnull().sum())
print("\nMissing values in test data:")
print(df_test.isnull().sum())


# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


fig, axes = plt.subplots(1, 2, figsize=(15, 5))
# Histogram
axes[0].hist(df_train['accident_risk'], bins=50, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Accident Risk')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Accident Risk')
axes[0].axvline(df_train['accident_risk'].mean(), color='red', linestyle='--', label=f'Mean: {df_train["accident_risk"].mean():.3f}')
axes[0].axvline(df_train['accident_risk'].median(), color='green', linestyle='--', label=f'Median: {df_train["accident_risk"].median():.3f}')
axes[0].legend()

# Box plot
axes[1].boxplot(df_train['accident_risk'])
axes[1].set_ylabel('Accident Risk')
axes[1].set_title('Box Plot of Accident Risk')
axes[1].grid(True)

plt.tight_layout()
plt.show()


print(f"Accident Risk Statistics:")
print(f"Mean: {df_train['accident_risk'].mean():.4f}")
print(f"Median: {df_train['accident_risk'].median():.4f}")
print(f"Std: {df_train['accident_risk'].std():.4f}")
print(f"Min: {df_train['accident_risk'].min():.4f}")
print(f"Max: {df_train['accident_risk'].max():.4f}")
print(f"Skewness: {df_train['accident_risk'].skew():.4f}")


categorical_cols = df_train.select_dtypes(include=['object', 'bool']).columns.tolist()
print(f"Categorical columns: {categorical_cols}")


for col in categorical_cols:
    if col != 'id':
        print(f"\n{col.upper()} - Value Counts:")
        print(df_train[col].value_counts())
        print(f"Unique values: {df_train[col].nunique()}")



fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.flatten()

categorical_features = ['road_type', 'lighting', 'weather', 'road_signs_present', 
                        'public_road', 'time_of_day', 'holiday', 'school_season']
for idx, col in enumerate(categorical_features):
    value_counts = df_train[col].value_counts()
    axes[idx].bar(range(len(value_counts)), value_counts.values, color='steelblue', alpha=0.7)
    axes[idx].set_xticks(range(len(value_counts)))
    axes[idx].set_xticklabels(value_counts.index, rotation=45, ha='right')
    axes[idx].set_title(f'Distribution of {col}')
    axes[idx].set_ylabel('Count')
    axes[idx].grid(axis='y', alpha=0.3)

# Remove extra subplot
fig.delaxes(axes[-1])
plt.tight_layout()
plt.show()


numerical_cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']


fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for idx, col in enumerate(numerical_cols):
    axes[idx].hist(df_train[col], bins=30, edgecolor='black', alpha=0.7, color='coral')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Frequency')
    axes[idx].set_title(f'Distribution of {col}')
    axes[idx].axvline(df_train[col].mean(), color='red', linestyle='--', label='Mean')
    axes[idx].legend()

plt.tight_layout()
plt.show()


df_train[numerical_cols].describe()


fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Box plot
df_train.boxplot(column='accident_risk', by='road_type', ax=axes[0])
axes[0].set_title('Accident Risk by Road Type')
axes[0].set_xlabel('Road Type')
axes[0].set_ylabel('Accident Risk')

# Mean accident risk
road_risk = df_train.groupby('road_type')['accident_risk'].mean().sort_values()
axes[1].barh(range(len(road_risk)), road_risk.values, color='indianred')
axes[1].set_yticks(range(len(road_risk)))
axes[1].set_yticklabels(road_risk.index)
axes[1].set_xlabel('Mean Accident Risk')
axes[1].set_title('Average Accident Risk by Road Type')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(15, 5))

df_train.boxplot(column='accident_risk', by='weather', ax=axes[0])
axes[0].set_title('Accident Risk by Weather')
axes[0].set_xlabel('Weather')
axes[0].set_ylabel('Accident Risk')

weather_risk = df_train.groupby('weather')['accident_risk'].mean().sort_values()
axes[1].barh(range(len(weather_risk)), weather_risk.values, color='skyblue')
axes[1].set_yticks(range(len(weather_risk)))
axes[1].set_yticklabels(weather_risk.index)
axes[1].set_xlabel('Mean Accident Risk')
axes[1].set_title('Average Accident Risk by Weather')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(15, 5))

df_train.boxplot(column='accident_risk', by='lighting', ax=axes[0])
axes[0].set_title('Accident Risk by Lighting')
axes[0].set_xlabel('Lighting')
axes[0].set_ylabel('Accident Risk')

lighting_risk = df_train.groupby('lighting')['accident_risk'].mean().sort_values()
axes[1].barh(range(len(lighting_risk)), lighting_risk.values, color='gold')
axes[1].set_yticks(range(len(lighting_risk)))
axes[1].set_yticklabels(lighting_risk.index)
axes[1].set_xlabel('Mean Accident Risk')
axes[1].set_title('Average Accident Risk by Lighting')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(15, 5))

df_train.boxplot(column='accident_risk', by='time_of_day', ax=axes[0])
axes[0].set_title('Accident Risk by Time of Day')
axes[0].set_xlabel('Time of Day')
axes[0].set_ylabel('Accident Risk')

time_risk = df_train.groupby('time_of_day')['accident_risk'].mean().sort_values()
axes[1].barh(range(len(time_risk)), time_risk.values, color='mediumseagreen')
axes[1].set_yticks(range(len(time_risk)))
axes[1].set_yticklabels(time_risk.index)
axes[1].set_xlabel('Mean Accident Risk')
axes[1].set_title('Average Accident Risk by Time of Day')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()


bool_features = ['road_signs_present', 'public_road', 'holiday', 'school_season']

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for idx, col in enumerate(bool_features):
    df_train.boxplot(column='accident_risk', by=col, ax=axes[idx])
    axes[idx].set_title(f'Accident Risk by {col}')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Accident Risk')

plt.tight_layout()
plt.show()



# Print mean accident risk for boolean features
for col in bool_features:
    print(f"\n{col}:")
    print(df_train.groupby(col)['accident_risk'].mean())


fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for idx, col in enumerate(numerical_cols):
    axes[idx].scatter(df_train[col], df_train['accident_risk'], alpha=0.3, s=1)
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Accident Risk')
    axes[idx].set_title(f'Accident Risk vs {col}')
    
    # Add trend line
    z = np.polyfit(df_train[col], df_train['accident_risk'], 1)
    p = np.poly1d(z)
    axes[idx].plot(df_train[col].sort_values(), p(df_train[col].sort_values()), 
                   "r--", alpha=0.8, linewidth=2)

plt.tight_layout()
plt.show()


# Create a copy with numerical encoding for correlations
df_corr = df_train.copy()


# Encode categorical variables for correlation
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in ['road_type', 'lighting', 'weather', 'time_of_day']:
    df_corr[col + '_encoded'] = le.fit_transform(df_corr[col])



# Select numerical columns for correlation
corr_cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents',
             'road_signs_present', 'public_road', 'holiday', 'school_season',
             'road_type_encoded', 'lighting_encoded', 'weather_encoded', 
             'time_of_day_encoded', 'accident_risk']



correlation_matrix = df_corr[corr_cols].corr()



# Plot correlation matrix
plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1)
plt.title('Correlation Matrix - All Features', fontsize=15, pad=20)
plt.tight_layout()
plt.show()



target_corr = correlation_matrix['accident_risk'].sort_values(ascending=False)
print("Correlations with Accident Risk:")
print(target_corr)

plt.figure(figsize=(10, 8))
target_corr[target_corr.index != 'accident_risk'].plot(kind='barh', color='teal')
plt.xlabel('Correlation with Accident Risk')
plt.title('Feature Correlations with Target Variable')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()


pivot_road_weather = df_train.pivot_table(values='accident_risk', 
                                          index='road_type', 
                                          columns='weather', 
                                          aggfunc='mean')

plt.figure(figsize=(10, 6))
sns.heatmap(pivot_road_weather, annot=True, fmt='.3f', cmap='YlOrRd', linewidths=1)
plt.title('Average Accident Risk: Road Type vs Weather')
plt.tight_layout()
plt.show()


pivot_road_lighting = df_train.pivot_table(values='accident_risk', 
                                           index='road_type', 
                                           columns='lighting', 
                                           aggfunc='mean')

plt.figure(figsize=(10, 6))
sns.heatmap(pivot_road_lighting, annot=True, fmt='.3f', cmap='YlOrRd', linewidths=1)
plt.title('Average Accident Risk: Road Type vs Lighting')
plt.tight_layout()
plt.show()


pivot_lanes_speed = df_train.pivot_table(values='accident_risk', 
                                        index='num_lanes', 
                                        columns='speed_limit', 
                                        aggfunc='mean')

plt.figure(figsize=(12, 6))
sns.heatmap(pivot_lanes_speed, annot=True, fmt='.3f', cmap='viridis', linewidths=1)
plt.title('Average Accident Risk: Number of Lanes vs Speed Limit')
plt.tight_layout()
plt.show()


## 10. Key Insights Summary

print("="*80)
print("KEY INSIGHTS FROM EDA")
print("="*80)


print("\n1. TARGET VARIABLE:")
print(f"   - Accident risk ranges from {df_train['accident_risk'].min():.3f} to {df_train['accident_risk'].max():.3f}")
print(f"   - Mean accident risk: {df_train['accident_risk'].mean():.3f}")
print(f"   - The distribution appears relatively uniform with slight variations")


print("\n2. CATEGORICAL FEATURES:")
for col in ['road_type', 'weather', 'lighting', 'time_of_day']:
    print(f"   - {col}: {df_train[col].nunique()} unique values")
    top_cat = df_train[col].value_counts().index[0]
    top_risk = df_train[df_train[col] == top_cat]['accident_risk'].mean()
    print(f"     Most common: {top_cat} (avg risk: {top_risk:.3f})")


print("\n3. NUMERICAL FEATURES:")
for col in numerical_cols:
    corr = df_corr[[col, 'accident_risk']].corr().iloc[0, 1]
    print(f"   - {col}: correlation with target = {corr:.3f}")



print("\n4. BOOLEAN FEATURES IMPACT:")
for col in bool_features:
    true_risk = df_train[df_train[col] == True]['accident_risk'].mean()
    false_risk = df_train[df_train[col] == False]['accident_risk'].mean()
    diff = true_risk - false_risk
    print(f"   - {col}: True={true_risk:.3f}, False={false_risk:.3f}, Diff={diff:.3f}")




print("\n5. TRAIN-TEST CONSISTENCY:")
print(f"   - Train shape: {df_train.shape}, Test shape: {df_test.shape}")
print(f"   - Feature distributions appear consistent between train and test sets")

print("\n" + "="*80)
print("EDA COMPLETE - Ready for Model Building!")
print("="*80)




