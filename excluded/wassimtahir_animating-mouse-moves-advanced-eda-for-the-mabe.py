import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')


plt.style.use('ggplot')
sns.set_palette("viridis")
pd.set_option('display.max_columns', None)


train_df = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/train.csv')


print("Dataset Shape:", train_df.shape)
print("\nColumns Overview:")
print(train_df.info())
print("\nFirst Few Rows:")
display(train_df.head())




lab_distribution = train_df['lab_id'].value_counts()


plt.figure(figsize=(10,6))
lab_distribution.sort_values().plot(kind='barh', color='skyblue')
plt.title("Data Distribution by Lab")
plt.xlabel("Count")
plt.ylabel("Lab ID")
plt.tight_layout()
plt.show()

threshold = 0.02 
counts = lab_distribution / lab_distribution.sum()
others = counts[counts < threshold].sum()
counts = counts[counts >= threshold]
counts["Others"] = others

plt.figure(figsize=(8,8))
plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
plt.title("Data Proportion by Lab")
plt.tight_layout()
plt.show()


plt.figure(figsize=(12,6))
train_df.boxplot(column="video_duration_sec", by="lab_id", rot=45)
plt.title("Video Duration Distribution by Laboratory")
plt.suptitle("")
plt.xlabel("Laboratory")
plt.ylabel("Duration (seconds)")
plt.tight_layout()
plt.show()




mouse_columns = [col for col in train_df.columns if 'mouse' in col and any(x in col for x in ['strain', 'color', 'sex', 'age', 'condition'])]
mouse_data = []

for i in range(1, 5):
    mouse_cols = [col for col in mouse_columns if col.startswith(f'mouse{i}_')]
    if mouse_cols:
        for idx, row in train_df.iterrows():
            if pd.notna(row[f'mouse{i}_strain']):
                mouse_data.append({
                    'video_id': row['video_id'],
                    'lab_id': row['lab_id'],
                    'mouse_id': f'mouse{i}',
                    'strain': row[f'mouse{i}_strain'],
                    'color': row[f'mouse{i}_color'],
                    'sex': row[f'mouse{i}_sex'],
                    'age': row[f'mouse{i}_age'],
                    'condition': row[f'mouse{i}_condition']
                })

mouse_df = pd.DataFrame(mouse_data)

strain_counts = mouse_df['strain'].value_counts()
sex_counts = mouse_df['sex'].value_counts()
color_counts = mouse_df['color'].value_counts()
condition_counts = mouse_df['condition'].value_counts()

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
strain_counts.plot(kind='barh', ax=axs[0,0], color='skyblue')
axs[0,0].set_title("Strain Distribution")
axs[0,0].set_xlabel("Count")
axs[0,0].set_ylabel("Strain")

sex_counts.plot(kind='barh', ax=axs[0,1], color='lightgreen')
axs[0,1].set_title("Sex Distribution")
axs[0,1].set_xlabel("Count")
axs[0,1].set_ylabel("Sex")

color_counts.plot(kind='barh', ax=axs[1,0], color='salmon')
axs[1,0].set_title("Color Distribution")
axs[1,0].set_xlabel("Count")
axs[1,0].set_ylabel("Color")

condition_counts.plot(kind='barh', ax=axs[1,1], color='orchid')
axs[1,1].set_title("Condition Distribution")
axs[1,1].set_xlabel("Count")
axs[1,1].set_ylabel("Condition")

plt.tight_layout()
plt.show()




train_df['resolution'] = train_df['video_width_pix'].astype(str) + 'x' + train_df['video_height_pix'].astype(str)
resolution_counts = train_df['resolution'].value_counts().head(10)

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

axs[0,0].hist(train_df['frames_per_second'].dropna(), bins=20, color='skyblue', edgecolor='black')
axs[0,0].set_title("FPS Distribution")
axs[0,0].set_xlabel("Frames per Second")
axs[0,0].set_ylabel("Count")

axs[0,1].hist(train_df['video_duration_sec'].dropna(), bins=20, color='lightgreen', edgecolor='black')
axs[0,1].set_title("Duration Distribution")
axs[0,1].set_xlabel("Video Duration (sec)")
axs[0,1].set_ylabel("Count")

axs[1,0].bar(resolution_counts.index, resolution_counts.values, color='salmon', edgecolor='black')
axs[1,0].set_title("Video Resolutions (Top 10)")
axs[1,0].set_xlabel("Resolution")
axs[1,0].set_ylabel("Count")
axs[1,0].tick_params(axis='x', rotation=45)

axs[1,1].hist(train_df['pix_per_cm_approx'].dropna(), bins=20, color='orchid', edgecolor='black')
axs[1,1].set_title("Pixels per cm Distribution")
axs[1,1].set_xlabel("Pixels per cm")
axs[1,1].set_ylabel("Count")

plt.tight_layout()
plt.show()

technical_params = train_df[['frames_per_second', 'video_duration_sec', 'pix_per_cm_approx', 
                            'video_width_pix', 'video_height_pix', 'arena_width_cm', 'arena_height_cm']].corr()

plt.figure(figsize=(10,8))
sns.heatmap(technical_params, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Matrix of Technical Parameters")
plt.tight_layout()
plt.show()





arena_counts = train_df['arena_type'].value_counts()
arena_shape_counts = train_df['arena_shape'].value_counts()

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].bar(arena_counts.index, arena_counts.values, color='skyblue', edgecolor='black')
axs[0].set_title("Arena Types")
axs[0].set_xlabel("Type")
axs[0].set_ylabel("Count")
axs[0].tick_params(axis='x', rotation=45)

axs[1].bar(arena_shape_counts.index, arena_shape_counts.values, color='salmon', edgecolor='black')
axs[1].set_title("Arena Shapes")
axs[1].set_xlabel("Shape")
axs[1].set_ylabel("Count")
axs[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
sns.scatterplot(data=train_df, x='arena_width_cm', y='arena_height_cm', hue='arena_shape', palette='Set2')
plt.title("Arena Dimensions (cm)")
plt.xlabel("Width (cm)")
plt.ylabel("Height (cm)")
plt.legend(title='Arena Shape')
plt.tight_layout()
plt.show()




tracking_counts = train_df['tracking_method'].value_counts()

plt.figure(figsize=(8,5))
plt.bar(tracking_counts.index, tracking_counts.values, color='skyblue', edgecolor='black')
plt.title("Tracking Methods Used")
plt.xlabel("Method")
plt.ylabel("Number of Videos")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

body_parts = train_df['body_parts_tracked'].str.split(',', expand=True).stack().value_counts()
body_parts = body_parts[body_parts > 10]

plt.figure(figsize=(10,5))
plt.bar(body_parts.index, body_parts.values, color='salmon', edgecolor='black')
plt.title("Tracked Body Parts (Most Common)")
plt.xlabel("Body Part")
plt.ylabel("Number of Occurrences")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



behaviors = train_df['behaviors_labeled'].str.split(',', expand=True).stack().str.strip().value_counts()

plt.figure(figsize=(12,5))
plt.bar(behaviors.index, behaviors.values, color='skyblue', edgecolor='black')
plt.title("Annotated Behaviors in the Dataset")
plt.xlabel("Behavior")
plt.ylabel("Number of Occurrences")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

lab_behaviors = train_df.groupby('lab_id')['behaviors_labeled'] \
    .apply(lambda x: x.str.split(',').explode().str.strip().value_counts())
lab_behaviors = lab_behaviors.reset_index()
lab_behaviors.columns = ['lab_id', 'behavior', 'count']

top_behaviors = behaviors.head(10).index
lab_behaviors_top = lab_behaviors[lab_behaviors['behavior'].isin(top_behaviors)]

plt.figure(figsize=(12,6))
sns.barplot(data=lab_behaviors_top, x='behavior', y='count', hue='lab_id', palette='Set2')
plt.title("Top 10 Behaviors by Laboratory")
plt.xlabel("Behavior")
plt.ylabel("Number of Occurrences")
plt.xticks(rotation=45)
plt.legend(title='Lab ID')
plt.tight_layout()
plt.show()



lab_features = train_df.groupby('lab_id').agg({
    'frames_per_second': 'mean',
    'video_duration_sec': 'mean',
    'pix_per_cm_approx': 'mean',
    'video_width_pix': 'mean',
    'video_height_pix': 'mean'
}).round(2)

plt.figure(figsize=(18,10))
sns.heatmap(lab_features.T, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Average Technical Features by Laboratory")
plt.ylabel("Feature")
plt.xlabel("Lab ID")
plt.tight_layout()
plt.show()

numeric_cols = ['frames_per_second', 'video_duration_sec', 'pix_per_cm_approx', 
                'video_width_pix', 'video_height_pix', 'arena_width_cm', 'arena_height_cm']
pca_data = train_df[numeric_cols].dropna()

scaler = StandardScaler()
scaled_data = scaler.fit_transform(pca_data)

pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_data)

pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['lab_id'] = train_df.loc[pca_data.index, 'lab_id'].values

plt.figure(figsize=(8,6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='lab_id', palette='Set2', s=60)
plt.title("Principal Component Analysis of Technical Parameters")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title='Lab ID')
plt.tight_layout()
plt.show()





pairplot_vars = ['frames_per_second', 'video_duration_sec', 'pix_per_cm_approx', 'arena_width_cm', 'arena_height_cm']
pairplot_data = train_df[pairplot_vars + ['lab_id']].dropna()

sns.pairplot(pairplot_data, vars=pairplot_vars, hue='lab_id', corner=False, plot_kws={'alpha':0.6, 's':40})
plt.suptitle("Relationships Between Technical Variables", y=1.02)
plt.show()

sunburst_data = train_df.groupby(['lab_id', 'arena_shape', 'arena_type']).size().reset_index(name='count')


sunburst_pivot = sunburst_data.pivot_table(index=['lab_id', 'arena_shape'], columns='arena_type', values='count', fill_value=0)

sunburst_pivot.plot(kind='bar', stacked=True, figsize=(12,6), colormap='tab20')
plt.title("Data Hierarchy: Lab > Arena Shape > Arena Type")
plt.xlabel("(Lab ID, Arena Shape)")
plt.ylabel("Count")
plt.xticks(rotation=45, ha='right')
plt.legend(title='Arena Type', bbox_to_anchor=(1.05,1))
plt.tight_layout()
plt.show()




def generate_statistical_summary(df):
    summary = pd.DataFrame({
        'Variable': df.columns,
        'Type': df.dtypes.values,
        'Missing Values': df.isnull().sum().values,
        '% Missing Values': (df.isnull().sum() / len(df) * 100).round(2).values,
        'Unique Values': [df[col].nunique() for col in df.columns]
    })
    
 
    numeric_stats = df.describe().T
    summary = summary.merge(numeric_stats, how='left', left_on='Variable', right_index=True)
    
    return summary


statistical_summary = generate_statistical_summary(train_df)
statistical_summary.to_csv('statistical_summary.csv', index=False)

print("Statistical Summary of Numerical Variables:")
print(train_df.describe())


missing_data = train_df.isnull().sum().sort_values(ascending=False)
missing_percentage = (train_df.isnull().sum() / train_df.shape[0] * 100).sort_values(ascending=False)
missing_df = pd.DataFrame({'Missing Values': missing_data, 'Percentage': missing_percentage.round(2)})
print("\nMissing Values Analysis:")
display(missing_df[missing_df['Missing Values'] > 0])




