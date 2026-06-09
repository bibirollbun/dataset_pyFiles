import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings("ignore")


# Comprehensive feature engineering with physiological context
def create_features(df):
    df = df.copy()
    
    # Basic physiological features
    df['BMI'] = df['Weight'] / (df['Height']/100)**2
    df['Sex_encoded'] = df['Sex'].map({'male': 1, 'female': 0})
    
    # Enhanced metabolic calculations
    df['BMR'] = df.apply(lambda x: 
        (10 * x['Weight'] + 6.25 * x['Height'] - 5 * x['Age'] + 5) if x['Sex'] == 'male' 
        else (10 * x['Weight'] + 6.25 * x['Height'] - 5 * x['Age'] - 161), axis=1)
    
    # Advanced heart rate features
    df['Max_HR'] = 208 - 0.7 * df['Age']  # Tanaka formula
    df['HR_Reserve'] = df['Heart_Rate'] / df['Max_HR']
    df['HR_Zone'] = pd.cut(df['HR_Reserve'], 
                          bins=[0, 0.6, 0.7, 0.8, 0.9, 1.0], 
                          labels=[1, 2, 3, 4, 5]).astype('float')
    
    # Activity intensity metrics
    df['MET'] = (df['Heart_Rate'] * df['Duration']) / (df['BMR'] / 24 / 60)
    df['Thermal_Load'] = (df['Body_Temp'] - 36.5) * df['Duration']
    df['Work_Volume'] = df['Duration'] * df['Heart_Rate'] / df['Weight']
    
    # Interaction terms with physiological context
    df['Age_Weight_Interaction'] = df['Age'] * df['Weight'] / 100
    df['HR_Temp_Interaction'] = df['Heart_Rate'] * (df['Body_Temp'] - 36)
    df['BMR_Duration'] = df['BMR'] * df['Duration'] / 1000
    
    # Additional features from original model
    df['Caloric_Rate'] = df['BMR'] * df['Duration'] / 1440
    df['Temp_HR_Ratio'] = df['Body_Temp'] / df['Heart_Rate']
    df['Age_HR_Interaction'] = df['Age'] * df['Heart_Rate'] / 100
    df['Weight_Temp_Interaction'] = df['Weight'] * (df['Body_Temp'] - 36)
    df['BMR_HR_Interaction'] = df['BMR'] * df['HR_Reserve']
    df['HR_Squared'] = df['Heart_Rate'] ** 2
    df['Duration_Sqrt'] = np.sqrt(df['Duration'])

    # This features from this notebook was added
    df['Resting_HR'] = df['Heart_Rate'].quantile(0.05)
    df['HR_Decline_Rate'] = df['Heart_Rate'] / df['Duration']
    df['VO2_Max_Estimate'] = (df['Max_HR'] / df['Resting_HR']) * 15
    df['Thermal_Stress'] = (df['Body_Temp'] - 36.5) * df['HR_Reserve']
    df['BMI_HR_Zone'] = df['BMI'] * df['HR_Zone']
    df['BMR_Duration_Sqrt'] = df['BMR'] * df['Duration_Sqrt']
    df['Log_Thermal_Load'] = np.log1p(df['Thermal_Load'])
    df['Log_Work_Volume'] = np.log1p(df['Work_Volume'])
    df['HR_Power3'] = (df['Heart_Rate'] / 100) ** 3
    
    return df


# Load processed data
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv').dropna()


train.head()


# Configure visual style
sns.set(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (12, 8)


# Target Variable Analysis with Gender Comparison
fig, ax = plt.subplots(1, 2, figsize=(18, 6))

# Original Calories Distribution by Gender
sns.histplot(data=train, x='Calories', kde=True, ax=ax[0], hue='Sex')
ax[0].set_title('Original Calories Distribution by Gender')

# Log-Transformed Calories Distribution by Gender
sns.histplot(data=train, x=np.log1p(train['Calories']), kde=True, ax=ax[1], hue='Sex')
ax[1].set_title('Log-Transformed Calories Distribution by Gender')
ax[1].set_xlabel('log(Calories + 1)')

plt.tight_layout()
plt.show()


# Set up the figure
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

# Features to plot against Calories
features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']

for i, feature in enumerate(features):
    sns.scatterplot(
        data=train,
        x=feature,
        y='Calories',
        hue='Sex',
        palette={'male': '#1f77b4', 'female': '#ff7f0e'},
        alpha=0.6,
        ax=axes[i]
    )
    axes[i].set_title(f'Calories vs {feature}', fontsize=12)
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Calories')
    axes[i].grid(True, linestyle='--', alpha=0.6)
    
    # Add regression lines for each gender
    sns.regplot(
        data=train[train['Sex'] == 'male'],
        x=feature,
        y='Calories',
        scatter=False,
        color='blue',
        ax=axes[i],
        label='Male Trend'
    )
    sns.regplot(
        data=train[train['Sex'] == 'female'],
        x=feature,
        y='Calories',
        scatter=False,
        color='red',
        ax=axes[i],
        label='Female Trend'
    )

# Remove empty subplot if odd number of features
if len(features) < len(axes):
    axes[-1].axis('off')

plt.tight_layout()
plt.suptitle('Relationship Between Basic Features and Calories (by Gender)', y=1.02, fontsize=14)
plt.legend()
plt.show()


from sklearn.cluster import KMeans

# Cluster users
cluster_features = train[['Duration', 'Heart_Rate', 'Body_Temp']].dropna()
kmeans = KMeans(n_clusters=3, random_state=42)
train['User_Cluster'] = kmeans.fit_predict(cluster_features)

# Set up the figure
plt.figure(figsize=(18, 12))

# Plot 1: Cluster distribution by original features
plt.subplot(2, 2, 1)
sns.scatterplot(
    data=train,
    x='Duration',
    y='Heart_Rate',
    hue='User_Cluster',
    palette='viridis',
    style='User_Cluster',
    s=100,
    alpha=0.7
)
plt.title('Cluster Separation in Duration vs Heart Rate Space')
plt.grid(True, linestyle='--', alpha=0.3)

# Plot 2: Cluster distribution by Body Temp
plt.subplot(2, 2, 2)
sns.scatterplot(
    data=train,
    x='Body_Temp',
    y='Heart_Rate',
    hue='User_Cluster',
    palette='viridis',
    style='User_Cluster',
    s=100,
    alpha=0.7
)
plt.title('Cluster Separation in Body Temp vs Heart Rate Space')
plt.grid(True, linestyle='--', alpha=0.3)

# Plot 3: Calories by Cluster
plt.subplot(2, 2, 3)
sns.boxplot(
    data=train,
    x='User_Cluster',
    y='Calories',
    palette='viridis',
    showmeans=True,
    meanprops={'marker':'o', 'markerfacecolor':'white', 'markeredgecolor':'black'}
)
plt.title('Calorie Distribution Across Clusters')
plt.xlabel('Cluster')
plt.grid(True, axis='y', linestyle='--', alpha=0.3)

# Plot 4: 3D Visualization (optional)
ax = plt.subplot(2, 2, 4, projection='3d')
for cluster in sorted(train['User_Cluster'].unique()):
    cluster_data = train[train['User_Cluster'] == cluster]
    ax.scatter(
        cluster_data['Duration'],
        cluster_data['Heart_Rate'],
        cluster_data['Body_Temp'],
        label=f'Cluster {cluster}',
        s=50,
        alpha=0.6
    )
ax.set_xlabel('Duration')
ax.set_ylabel('Heart Rate')
ax.set_zlabel('Body Temp')
ax.set_title('3D Cluster View')
plt.legend()

plt.tight_layout()
plt.suptitle('User Cluster Analysis: Duration, Heart Rate & Body Temp', y=1.02)
plt.show()

# Print cluster characteristics
print("\nCluster Characteristics:")
print(train.groupby('User_Cluster')[['Duration', 'Heart_Rate', 'Body_Temp', 'Calories']].mean())


train_processed = create_features(train)


# Comprehensive Feature Distributions Analysis
all_features = [
    # Basic features
    'BMI', 'BMR', 'Heart_Rate', 'Body_Temp', 'Duration',
    
    # Heart rate features
    'Max_HR', 'HR_Reserve', 'HR_Zone',
    
    # Activity metrics
    'MET', 'Thermal_Load', 'Work_Volume',
    
    # Interaction terms
    'Age_Weight_Interaction', 'HR_Temp_Interaction', 'BMR_Duration',
    'Age_HR_Interaction', 'Weight_Temp_Interaction', 'BMR_HR_Interaction',
    
    # Derived features
    'Caloric_Rate', 'Temp_HR_Ratio', 'HR_Squared', 'Duration_Sqrt'
]

# Set up the plotting style
plt.style.use('seaborn')
palette = {'male': '#1f77b4', 'female': '#ff7f0e'}

# Create a grid of plots
n_cols = 3
n_rows = (len(all_features) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
axes = axes.flatten()

for i, feature in enumerate(all_features):
    ax = axes[i]
    if feature in train_processed.columns:
        sns.histplot(
            data=train_processed,
            x=feature,
            kde=True,
            ax=ax,
            hue='Sex',
            palette=palette,
            element='step',
            alpha=0.6,
            common_norm=False
        )
        ax.set_title(f'{feature} Distribution', fontsize=12)
        ax.set_xlabel('')
    else:
        ax.axis('off')  # Turn off axes for empty subplots

# Remove any empty subplots
for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.suptitle('Feature Distributions by Gender', y=1.02, fontsize=16)
plt.show()


X = train_processed.drop(['Calories', 'Sex', 'id'], axis=1)
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = ['HR_Zone']


# 3. Correlation Analysis
corr_matrix = train_processed[numerical_features + ['Calories']].corr()
plt.figure(figsize=(20, 16))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm',
            cbar_kws={'shrink': 0.75}, annot_kws={'size': 8})
plt.title('Feature Correlation Matrix')
plt.show()


from sklearn.ensemble import RandomForestRegressor

# Get the features actually used in training
X = train_processed.drop(columns=['Calories', 'id', 'Sex'])
feature_names = X.columns  # This ensures correct length match

# Train model
model = RandomForestRegressor()
model.fit(X, train_processed['Calories'])

# Create feature importance series with correct feature names
feature_importances = pd.Series(model.feature_importances_, index=feature_names)
top_features = feature_importances.sort_values(ascending=False).head(10)

# Visualize
plt.figure(figsize=(10, 6))
sns.barplot(x=top_features.values, y=top_features.index, palette='viridis')
plt.title('Top 10 Important Features for Calories Prediction')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()

print("Top 10 Features:")
print(top_features)


# Calculate correlations with Calories
corr_with_calories = train_processed.drop(columns=['Sex']).corr()[['Calories']].sort_values('Calories', ascending=False)

# Plot heatmap
plt.figure(figsize=(8, 12))
sns.heatmap(
    corr_with_calories,
    annot=True,
    cmap='coolwarm',
    vmin=-1,
    vmax=1,
    center=0,
    fmt='.2f',
    linewidths=0.5
)
plt.title('Feature Correlations with Calories', pad=20)
plt.tight_layout()
plt.show()


top_features = [
    'Duration_Sqrt', 'Duration', 'HR_Reserve',          # Top 3 (>5% importance)
    'Thermal_Stress', 'BMR_HR_Interaction',             # Mid-range (1-5%)
    'User_Cluster', 'MET',                              # Contextual features
    'Sex_encoded', 'Temp_HR_Ratio', 'HR_Squared'        # Low-but-nonzero
]

# Option 1: Use only high-impact features (top 5)
X_high_impact = train_processed[top_features[:5]]

# Option 2: All top 10 + key basics
X_balanced = train_processed[top_features + ['Age', 'Weight', 'Heart_Rate']]

# Option 3: Top 10 + engineered interactions
X_enhanced = X_balanced.copy()
X_enhanced['Duration_HR_Synergy'] = X_enhanced['Duration_Sqrt'] * X_enhanced['HR_Reserve']


from sklearn.model_selection import cross_val_score

def evaluate_features(feature_set):
    model = RandomForestRegressor(random_state=42)
    scores = cross_val_score(model, train_processed[feature_set], 
                            train_processed['Calories'], 
                            cv=5, scoring='neg_mean_squared_error')
    return np.sqrt(-scores.mean())

print(f"Top 5 only RMSE: {evaluate_features(top_features[:5]):.2f}")
print(f"Balanced set RMSE: {evaluate_features(top_features + ['Age','Weight']):.2f}")

