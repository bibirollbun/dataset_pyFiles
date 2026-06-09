import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import scipy.stats as stats


train_df=pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_df=pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
sample=pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')


print('Train Data\n')
train_df.head(3)


print('Sample Data\n')
sample.head(2)


print('Unique value of "Temparature": \t',sorted(train_df['Temparature'].unique()))
print('Unique value of "Humidity": \t',sorted(train_df['Humidity'].unique()))
print('Unique value of "Moisture": \t',sorted(train_df['Moisture'].unique()))
print('Unique value of "Soil Type": \t',sorted(train_df['Soil Type'].unique()))
print('Unique value of "Crop Type": \t',sorted(train_df['Crop Type'].unique()))
print('Unique value of "Nitrogen": \t',sorted(train_df['Nitrogen'].unique()))
print('Unique value of "Potassium": \t',sorted(train_df['Potassium'].unique()))
print('Unique value of "Phosphorous": \t',sorted(train_df['Phosphorous'].unique()))
print('Unique value of "Fertilizer Name": \t',sorted(train_df['Fertilizer Name'].unique()))


columns = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
fig, axs = plt.subplots(len(columns), 2, figsize=(14, 4 * len(columns)))
for i, col in enumerate(columns):
    mean_val = train_df[col].mean()
    median_val = train_df[col].median()
    sns.histplot(train_df[col], bins=30, color='gray', edgecolor='black', ax=axs[i, 0])
    axs[i, 0].axvline(mean_val, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    axs[i, 0].axvline(median_val, color='orange', linestyle='-', linewidth=2, label=f'Median: {median_val:.2f}')
    axs[i, 0].set_title(f'{col} Distribution')
    axs[i, 0].set_xlabel(col)
    axs[i, 0].set_ylabel('Frequency')
    axs[i, 0].legend()
    sns.boxplot(x=train_df[col], color='lightgray', ax=axs[i, 1])
    axs[i, 1].set_title(f'Boxplot of {col}')
    axs[i, 1].set_xlabel(col)
plt.tight_layout()
plt.show()


fertilizer_usage_dist = train_df.groupby(['Soil Type', 'Fertilizer Name']).size().reset_index(name='Usage Count')
plt.figure(figsize=(14, 6))
sns.barplot(data=fertilizer_usage_dist, x='Soil Type', y='Usage Count', hue='Fertilizer Name', palette='Set2')
plt.title('Fertilizer Usage Distribution Across Soil Types')
plt.xlabel('Soil Type')
plt.ylabel('Usage Count')
plt.legend(title='Fertilizer', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


fert_soil_crosstab = pd.crosstab(train_df['Fertilizer Name'], train_df['Soil Type'])
fert_soil_crosstab.plot(kind='bar', stacked=True, figsize=(10, 6), color=['#ADD8E6', '#90EE90', '#FFB6C1', '#FFFACD', '#D3D3D3'])
plt.title("Fertilizer Usage Across Soil Types")
plt.xlabel("Fertilizer Name")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.legend(title="Soil Type" ,bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


import matplotlib.cm as cm
fert_crop_crosstab = pd.crosstab(train_df['Fertilizer Name'], train_df['Crop Type'])
light_colors = ['#B3CDE0', '#C1E1DC', '#F5B7B1', '#F9E79F', '#D4E157', '#FFEB3B', '#D1C4E9', '#A5D6A7', '#FFCDD2', '#FFEBEE']
fert_crop_crosstab.plot(kind='bar', stacked=True, figsize=(14, 7), color=light_colors[:len(fert_crop_crosstab.columns)])
plt.title("Fertilizer Usage Across Crop Types")
plt.xlabel("Fertilizer Name")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.legend(title="Crop Type", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


soil_crop_crosstab = pd.crosstab(train_df['Soil Type'], train_df['Crop Type'])
plt.figure(figsize=(12, 8))
sns.heatmap(soil_crop_crosstab, annot=True, cmap='Greys', fmt='d', linewidths=0.5)
plt.title('Relation Between Soil Type and Crop Type')
plt.xlabel('Crop Type')
plt.ylabel('Soil Type')
plt.tight_layout()
plt.show()


fert_crop_crosstab = pd.crosstab(train_df['Fertilizer Name'], train_df['Crop Type'])
plt.figure(figsize=(12, 8))
sns.heatmap(fert_crop_crosstab, annot=True, cmap='Greys', fmt='d', linewidths=0.5)
plt.title('Relation Between Fertilizer Name and Crop Type')
plt.xlabel('Crop Type')
plt.ylabel('Fertilizer Name')
plt.tight_layout()
plt.show()


custom_crop_group_map = {
    'Barley': 'Cereal',
    'Wheat': 'Cereal',
    'Maize': 'Grain',
    'Millets': 'Grain',
    'Paddy': 'Water Crop',
    'Sugarcane': 'Water Crop',
    'Pulses': 'Legume',
    'Ground Nuts': 'Legume',
    'Oil seeds': 'Oil Crop',
    'Cotton': 'Fiber Crop',
    'Tobacco': 'Specialty Crop'
}
train_df['crop_group'] = train_df['Crop Type'].map(custom_crop_group_map)


fert_crop_crosstab = pd.crosstab(train_df['Fertilizer Name'], train_df['crop_group'])
fertilizer_usage_by_group = fert_crop_crosstab.sum(axis=0).sort_values(ascending=False)
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
sns.heatmap(fert_crop_crosstab, annot=True, cmap='Greys', fmt='d', linewidths=0.5, ax=axes[0])
axes[0].set_title('Heat-Map of "Fertilizer Name" and "crop_group"')
axes[0].set_xlabel('crop_group')
axes[0].set_ylabel('Fertilizer Name')

axes[1].bar(fertilizer_usage_by_group.index, fertilizer_usage_by_group.values, color='gray', edgecolor='black')
axes[1].set_title('"Fertilizer Name" by "crop_group"')
axes[1].set_xlabel('crop_group')
axes[1].set_ylabel('Usage Count')
axes[1].tick_params(axis='x', rotation=0)
plt.tight_layout()
plt.show()


fertilizer_soil_crosstab = pd.crosstab(train_df['Fertilizer Name'], train_df['Soil Type'])
plt.figure(figsize=(10, 8))
sns.heatmap(fertilizer_soil_crosstab, annot=True, cmap='Greys', fmt='d', linewidths=0.5)
plt.title('Relation Between Fertilizer Name and Soil Type')
plt.xlabel('Soil Type')
plt.ylabel('Fertilizer Name')
plt.tight_layout()
plt.show()


print('Nitrogen: ',sorted(train_df['Nitrogen'].unique()))


def categorize_nitrogen_updated(nitrogen):
    if nitrogen <= 9:
        return 'low_nitrogen'
    elif nitrogen <= 15:
        return 'moderately_low_nitrogen'
    elif nitrogen <= 25:
        return 'medium_nitrogen'
    elif nitrogen <= 33:
        return 'moderately_high_nitrogen'
    else:
        return 'high_nitrogen'
train_df['Nitrogen_Category'] = train_df['Nitrogen'].apply(categorize_nitrogen_updated)


plt.figure(figsize=(10, 7))
sns.countplot(data=train_df, x='Nitrogen_Category', hue='Crop Type', palette='Set2')
plt.title('Distribution of Nitrogen Categories by "Crop Type"')
plt.xlabel('Nitrogen Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 7))
sns.countplot(data=train_df, x='Nitrogen_Category', hue='Fertilizer Name', palette='Set2')
plt.title('Distribution of Nitrogen Categories by "Fertilizer Name"')
plt.xlabel('Nitrogen Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


percentiles = {
    'Nitrogen': train_df['Nitrogen'].quantile([0.25, 0.5, 0.75]),
    'Potassium': train_df['Potassium'].quantile([0.25, 0.5, 0.75]),
    'Phosphorous': train_df['Phosphorous'].quantile([0.25, 0.5, 0.75])
}
def categy_by_percentile(value, nutrient):
    q25, q50, q75 = percentiles[nutrient]    
    if value <= q25:
        return 1
    elif value <= q75:
        return 2
    else:
        return 3
train_df['Nitrogen_Score'] = train_df['Nitrogen'].apply(lambda x: categy_by_percentile(x, 'Nitrogen'))
train_df['Potassium_Score'] = train_df['Potassium'].apply(lambda x: categy_by_percentile(x, 'Potassium'))
train_df['Phosphorous_Score'] = train_df['Phosphorous'].apply(lambda x: categy_by_percentile(x, 'Phosphorous'))
train_df['NPK_Score'] = train_df['Nitrogen_Score'] + train_df['Potassium_Score'] + train_df['Phosphorous_Score']


print('NPK_Score: ',sorted(train_df['NPK_Score'].unique()))


plt.figure(figsize=(9, 4))
sns.countplot(x='NPK_Score', data=train_df, palette='coolwarm')
plt.title('Countplot of Temperature')
plt.xlabel('NPK_Score')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


temperature_weight = 0.25
humidity_weight = 0.35
moisture_weight = 0.40
train_df['fertile_land'] = (
    train_df['Temparature'] * temperature_weight +
    train_df['Humidity'] * humidity_weight +
    train_df['Moisture'] * moisture_weight)


mean_temp = train_df['fertile_land'].mean()
median_temp = train_df['fertile_land'].median()
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.histplot(train_df['fertile_land'], bins=50, kde=True, color="gray", edgecolor="black", ax=axes[0])
axes[0].axvline(mean_temp, color='skyblue', linestyle='--', linewidth=1.5, label=f"Mean: {mean_temp:.2f}")
axes[0].axvline(median_temp, color='green', linestyle='--', linewidth=1.5, label=f"Median: {median_temp:.2f}")
axes[0].set_title("Distribution of fertile_land", fontsize=16)
axes[0].set_xlabel("fertile_land", fontsize=12)
axes[0].set_ylabel("Frequency", fontsize=12)
axes[0].legend(fontsize=12)

sns.boxplot(x=train_df['fertile_land'], ax=axes[1], color='gray', fliersize=5)
axes[1].set_title("Box Plot of fertile_land", fontsize=16)
axes[1].set_xlabel("fertile_land", fontsize=12)
plt.tight_layout()
plt.show()


rainfall = (
    0.03 * train_df['Humidity']**1.1 +
    0.04 * train_df['Moisture']**1.2 -
    0.02 * train_df['Temparature'] + 
    5                                
)
train_df['Rainfall_Alt'] = rainfall.apply(lambda x: max(x, 0))


plt.figure(figsize=(10, 6))
plt.hist(train_df['Rainfall_Alt'], bins=30, color='Grey', edgecolor='black')
plt.title('Distribution of "Rainfall_Alt"')
plt.xlabel('Rainfall_Alt (mm)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


train_df['Rainfall_Binary'] = (
    (0.4 * train_df['Humidity'] + 0.6 * train_df['Moisture'] - 0.3 * train_df['Temparature']) > 50
).astype(int)



train_df['Rainfall_Binary'].value_counts()


train_df.head()

