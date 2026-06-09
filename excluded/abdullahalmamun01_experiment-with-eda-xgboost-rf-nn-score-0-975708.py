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

train_df=pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_df=pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


exception_df=train_df[train_df['id'].isin([1873,2234])].copy()
exception_df


train_df.head()


test_df.head()


train_df.describe()


# columns=train_df.columns
def missing(data):
    # numeric_columns=data.select_dtypes(include=['int','float']).columns
    columns=data.columns
    print("column".ljust(41), "missing")
    print("_"*49)
    for col in columns:
        spec=str(data[col].count())
        print(col.ljust(30-len(spec)),spec.ljust(20-len(spec)),data[col].isna().sum())

from tabulate import tabulate

def missing_pd(data):
    columns=data.columns
    missing_data = [[col, data[col].count(), data[col].isna().sum()] for col in columns]
    print(tabulate(missing_data, headers=['Column', 'Non-Missing', 'Missing'], tablefmt='pretty'))


missing_pd(train_df)
print('\nTest Data')
missing(test_df)


cnt=0

for columns in train_df.columns:
    for val in train_df[columns]:
        if(val==np.inf or val==-np.inf):
            cnt+=1

    print(columns,cnt)


train_imputer_df = train_df.copy()
test_imputer_df = test_df.copy()

category_columns=train_df.select_dtypes(include='object').columns
print(category_columns)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

for col in category_columns:
    le.fit(train_imputer_df[col].astype(str))
    train_imputer_df[col]=le.transform(train_imputer_df[col].astype(str))
    print(f"Column {col} mapping: {dict(zip(le.classes_, range(len(le.classes_))))}")

for col in category_columns:
    if col=="Personality":
        continue
    le.fit(test_imputer_df[col].astype(str))
    test_imputer_df[col]=le.transform(test_imputer_df[col].astype(str))
    print(f"Column {col} mapping: {dict(zip(le.classes_, range(len(le.classes_))))}")

# train_imputer_df=


train_imputer_df.replace([np.inf, -np.inf], np.nan, inplace=True)
test_imputer_df.replace([np.inf, -np.inf], np.nan, inplace=True)

category_columns=['Stage_fear', 'Drained_after_socializing']

for col in category_columns:
    train_imputer_df[col]=train_imputer_df[col].replace(2,np.nan)
    test_imputer_df[col]=train_imputer_df[col].replace(2,np.nan)

columns=list(train_df.columns)

columns.remove("Personality")

from sklearn.experimental import enable_iterative_imputer  # Required for IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# Initialize IterativeImputer with RandomForest
imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=100, random_state=42), max_iter=10, random_state=42)

# Impute all numeric columns at once
train_imputer_df[columns] = imputer.fit_transform(train_imputer_df[columns])

# Align test data columns with training data (handle missing or extra columns)
test_numeric_columns = test_imputer_df.reindex(columns)  # Subset to match train columns

# Impute test data using the fitted imputer
test_imputer_df[columns] = imputer.transform(test_imputer_df[columns])


train_imputer_df[category_columns] = np.clip(train_imputer_df[category_columns].round().astype(int), 0, 1)
test_imputer_df[category_columns] = np.clip(test_imputer_df[category_columns].round().astype(int), 0, 1)


missing(train_imputer_df)
print('\n')
missing(test_imputer_df)


test_imputer_df.head()
# test_imputer_df['Post_frequency'].astype(int)


import matplotlib.pyplot as plt
import seaborn as sns


plt.figure(figsize=(6,2))
sns.countplot(data=train_imputer_df,x="Personality")


train_imputer_df.columns


# Apply Seaborn's default style
sns.set_style("whitegrid")  # Use 'whitegrid' for a clean look with a grid, or 'darkgrid', 'white', etc.
plt.figure(figsize=(10, 6))

# Create scatter plot with hue
scatter_plot = sns.scatterplot(
    data=train_imputer_df,
    x='Friends_circle_size',
    y='Social_event_attendance',
    hue='Personality',  # Uses 0 and 1, mapped to colors by palette
    palette='deep',
    s=100,
    alpha=0.6,
    edgecolor='white',
    linewidth=0.5
)

# Customize legend to show names instead of numbers
plt.legend(
    title="Personality",
    handles=scatter_plot.legend_.legend_handles,
    labels=['Extrovert', 'Introvert'],  # Map 0 and 1 to names
    title_fontsize=11,
    fontsize=10,
    loc='upper left',
    bbox_to_anchor=(1, 1)
)

# Customize labels and title
plt.xlabel('Friends Circle Size', fontsize=12, fontweight='bold')
plt.ylabel('Social Event Attendance', fontsize=12, fontweight='bold')
plt.title('Relationship Between Friends Circle Size and Social Event Attendance by Personality',
          fontsize=14, fontweight='bold', pad=15)

# Improve layout
plt.tight_layout()

# Display the plot
plt.show()


sns.scatterplot(data=train_imputer_df,x='Friends_circle_size',y='Social_event_attendance',hue='Personality')


sns.scatterplot(data=train_imputer_df,x='Going_outside',y='Post_frequency',hue='Personality')


sns.scatterplot(data=train_imputer_df,x='Going_outside',y='Drained_after_socializing',hue='Personality')



import seaborn as sns
import matplotlib.pyplot as plt

#remove warning in output
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# train_imputer_df = train_imputer_df.replace([np.inf, -np.inf], np.nan)

# Separate plots for each personality
plt.figure(figsize=(10, 5))
for i, personality in enumerate([0, 1]):
    plt.subplot(1, 2, i+1)
    sns.histplot(data=train_imputer_df[train_imputer_df['Personality'] == personality], 
                 x='Going_outside', y='Post_frequency', bins=20, cmap='Blues', 
                 cbar=True, cbar_kws={'label': 'Count'});
    plt.title(f'{personality}')
    plt.xlabel('Going_outside')
    plt.ylabel('Post_frequency')
plt.tight_layout()
plt.show()


numeric_columns = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']

plt.figure(figsize=(10, 8))
i=0
for col in numeric_columns:
    plt.subplot(3, 2, i+1)
    sns.histplot(data=train_imputer_df, x=col, hue='Personality', kde=True)
    plt.title(f'Distribution of {col}')
    # plt.show()
    i+=1
plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 4))
correlation_matrix = train_imputer_df[numeric_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# sns.pairplot(train_df, hue='Personality', vars=numeric_columns)
# plt.show()


plt.figure(figsize=(10,6))
i=0
for col in numeric_columns:
    plt.subplot(3,2,i+1)
    sns.boxplot(data=train_df, y=col)
    plt.title(f'Boxplot of {col}')
    i+=1
plt.tight_layout()
plt.show()


from sklearn.ensemble import RandomForestClassifier
reg=RandomForestClassifier(n_estimators=100,random_state=42)

target='Personality'
feature_columns=[col for col in train_imputer_df.columns if col!='Personality']

X_imputer=train_imputer_df[feature_columns]
y_imputer=train_imputer_df[target]

reg.fit(X_imputer,y_imputer)
pred=reg.predict(test_imputer_df)

results_df=pd.DataFrame({
    'id':test_imputer_df['id'],
    'Personality':pred
})

results_df['Personality']=results_df['Personality'].map({0:"Extrovert",1:"Introvert"})
results_df

# print(pred)


importance_df=pd.DataFrame({
    'features':feature_columns,
    'importance':reg.feature_importances_ 
})
importance_df=importance_df.sort_values(by='importance',ascending=False)
sns.barplot(importance_df,x='importance',y='features')


def advanced_feature_engineering(df):
    """
    Comprehensive feature engineering for personality classification
    
    Parameters:
    df: DataFrame with original features
    
    Returns:
    DataFrame with engineered features
    """
    # Create a copy to avoid modifying original data
    df_engineered = df.copy()
    
    # ===== 1. INTERACTION FEATURES =====
    print("Creating interaction features...")
    
    # Social anxiety combination (stage fear + social draining)
    df_engineered['Social_Anxiety_Score'] = df_engineered['Stage_fear'] * df_engineered['Drained_after_socializing']
    
    # Social engagement level (going outside + social events - time alone)
    df_engineered['Social_Engagement'] = (df_engineered['Going_outside'] + df_engineered['Social_event_attendance']) - df_engineered['Time_spent_Alone']
    
    # Digital vs physical socializing
    df_engineered['Digital_vs_Physical_Social'] = df_engineered['Post_frequency'] / (df_engineered['Social_event_attendance'] + 1)
    
    # Social circle efficiency (posts per friend)
    df_engineered['Social_Circle_Efficiency'] = df_engineered['Post_frequency'] / (df_engineered['Friends_circle_size'] + 1)
    
    # Introvert tendency score (weighted combination of key indicators)
    df_engineered['Introvert_Tendency'] = (
        df_engineered['Stage_fear'] * 0.4 + 
        df_engineered['Drained_after_socializing'] * 0.3 + 
        df_engineered['Time_spent_Alone'] * 0.2 - 
        df_engineered['Going_outside'] * 0.1
    )
    
    # Social comfort level
    df_engineered['Social_Comfort'] = (df_engineered['Social_event_attendance'] + df_engineered['Going_outside']) / (df_engineered['Stage_fear'] + 1)
    
    # ===== 2. POLYNOMIAL FEATURES (for most important features) =====
    print("Creating polynomial features...")
    
    # Square of most important features
    df_engineered['Stage_fear_squared'] = df_engineered['Stage_fear'] ** 2
    df_engineered['Drained_after_socializing_squared'] = df_engineered['Drained_after_socializing'] ** 2
    df_engineered['Going_outside_squared'] = df_engineered['Going_outside'] ** 2
    
    # Cube root for some features (to capture different relationships)
    df_engineered['Stage_fear_cuberoot'] = np.cbrt(df_engineered['Stage_fear'])
    df_engineered['Post_frequency_cuberoot'] = np.cbrt(df_engineered['Post_frequency'])
    
    # ===== 3. RATIO FEATURES =====
    print("Creating ratio features...")
    
    # Social activity ratios
    df_engineered['Outside_to_Social_Events'] = df_engineered['Going_outside'] / (df_engineered['Social_event_attendance'] + 1)
    df_engineered['Posts_to_Friends'] = df_engineered['Post_frequency'] / (df_engineered['Friends_circle_size'] + 1)
    df_engineered['Social_to_Alone_Ratio'] = df_engineered['Social_event_attendance'] / (df_engineered['Time_spent_Alone'] + 1)
    
    # Fear to social activity ratio
    df_engineered['Fear_to_Social_Ratio'] = df_engineered['Stage_fear'] / (df_engineered['Social_event_attendance'] + 1)
    
    # ===== 4. BINNING/CATEGORIZATION =====
    print("Creating binned features...")
    
    # Bin continuous features into categories
    df_engineered['Stage_fear_category'] = pd.cut(df_engineered['Stage_fear'], 
                                                 bins=5, labels=['Very_Low', 'Low', 'Medium', 'High', 'Very_High'])
    df_engineered['Social_Level_category'] = pd.cut(df_engineered['Social_event_attendance'], 
                                                   bins=4, labels=['Low', 'Medium', 'High', 'Very_High'])
    
    # Convert categories to dummy variables
    stage_fear_dummies = pd.get_dummies(df_engineered['Stage_fear_category'], prefix='Stage_fear_cat')
    social_level_dummies = pd.get_dummies(df_engineered['Social_Level_category'], prefix='Social_level_cat')
    
    df_engineered = pd.concat([df_engineered, stage_fear_dummies, social_level_dummies], axis=1)
    
    # ===== 5. STATISTICAL FEATURES =====
    print("Creating statistical features...")
    
    # Create feature groups for statistical operations
    social_features = ['Social_event_attendance', 'Going_outside', 'Post_frequency', 'Friends_circle_size']
    introvert_features = ['Stage_fear', 'Drained_after_socializing', 'Time_spent_Alone']
    
    # Mean and std of feature groups
    df_engineered['Social_Features_Mean'] = df_engineered[social_features].mean(axis=1)
    df_engineered['Social_Features_Std'] = df_engineered[social_features].std(axis=1)
    df_engineered['Introvert_Features_Mean'] = df_engineered[introvert_features].mean(axis=1)
    df_engineered['Introvert_Features_Std'] = df_engineered[introvert_features].std(axis=1)
    
    # Range features
    df_engineered['Social_Features_Range'] = df_engineered[social_features].max(axis=1) - df_engineered[social_features].min(axis=1)
    df_engineered['Introvert_Features_Range'] = df_engineered[introvert_features].max(axis=1) - df_engineered[introvert_features].min(axis=1)
    
    # ===== 6. LOGARITHMIC TRANSFORMATIONS =====
    print("Creating logarithmic features...")
    
    # Log transform for features that might have exponential relationships
    df_engineered['Log_Post_frequency'] = np.log1p(df_engineered['Post_frequency'])
    df_engineered['Log_Friends_circle_size'] = np.log1p(df_engineered['Friends_circle_size'])
    df_engineered['Log_Social_event_attendance'] = np.log1p(df_engineered['Social_event_attendance'])
    
    # ===== 7. ADVANCED COMBINATIONS =====
    print("Creating advanced combination features...")
    
    # Weighted personality score
    df_engineered['Extrovert_Score'] = (
        df_engineered['Going_outside'] * 0.3 +
        df_engineered['Social_event_attendance'] * 0.3 +
        df_engineered['Post_frequency'] * 0.2 +
        df_engineered['Friends_circle_size'] * 0.1 -
        df_engineered['Stage_fear'] * 0.4 -
        df_engineered['Drained_after_socializing'] * 0.3 -
        df_engineered['Time_spent_Alone'] * 0.2
    )
    
    # Social consistency (how consistent are social behaviors)
    social_behaviors = ['Going_outside', 'Social_event_attendance', 'Post_frequency']
    df_engineered['Social_Consistency'] = df_engineered[social_behaviors].std(axis=1) / (df_engineered[social_behaviors].mean(axis=1) + 1)
    
    # Energy pattern (combination of draining and alone time)
    df_engineered['Energy_Pattern'] = df_engineered['Drained_after_socializing'] + df_engineered['Time_spent_Alone']
    
    # ===== 8. THRESHOLD FEATURES =====
    print("Creating threshold-based features...")
    
    # Binary features based on thresholds
    df_engineered['High_Stage_Fear'] = (df_engineered['Stage_fear'] > df_engineered['Stage_fear'].quantile(0.75)).astype(int)
    df_engineered['High_Social_Activity'] = (df_engineered['Social_event_attendance'] > df_engineered['Social_event_attendance'].quantile(0.75)).astype(int)
    df_engineered['Low_Social_Activity'] = (df_engineered['Social_event_attendance'] < df_engineered['Social_event_attendance'].quantile(0.25)).astype(int)
    df_engineered['High_Alone_Time'] = (df_engineered['Time_spent_Alone'] > df_engineered['Time_spent_Alone'].quantile(0.75)).astype(int)
    
    # ===== 9. CLUSTER-BASED FEATURES =====
    print("Creating cluster-based features...")
    
    # Distance from typical extrovert/introvert patterns
    typical_extrovert = [df_engineered['Stage_fear'].quantile(0.25), 
                        df_engineered['Drained_after_socializing'].quantile(0.25),
                        df_engineered['Going_outside'].quantile(0.75),
                        df_engineered['Social_event_attendance'].quantile(0.75)]
    
    typical_introvert = [df_engineered['Stage_fear'].quantile(0.75), 
                        df_engineered['Drained_after_socializing'].quantile(0.75),
                        df_engineered['Going_outside'].quantile(0.25),
                        df_engineered['Social_event_attendance'].quantile(0.25)]
    
    key_features = ['Stage_fear', 'Drained_after_socializing', 'Going_outside', 'Social_event_attendance']
    
    # Euclidean distance to typical patterns
    df_engineered['Distance_to_Typical_Extrovert'] = np.sqrt(
        sum((df_engineered[feat] - typical_extrovert[i])**2 for i, feat in enumerate(key_features))
    )
    
    df_engineered['Distance_to_Typical_Introvert'] = np.sqrt(
        sum((df_engineered[feat] - typical_introvert[i])**2 for i, feat in enumerate(key_features))
    )
    
    # ===== 10. CLEAN UP =====
    # Remove the categorical columns we created for dummy encoding
    df_engineered = df_engineered.drop(['Stage_fear_category', 'Social_Level_category'], axis=1)
    
    # Handle any infinite or NaN values
    df_engineered = df_engineered.replace([np.inf, -np.inf], np.nan)
    df_engineered = df_engineered.fillna(df_engineered.mean())
    
    print(f"Feature engineering complete!")
    print(f"Original features: {len(df.columns)}")
    print(f"Engineered features: {len(df_engineered.columns)}")
    print(f"Total new features created: {len(df_engineered.columns) - len(df.columns)}")
    
    return df_engineered

# ===== USAGE EXAMPLE =====
def demonstrate_feature_engineering():
    """
    Demonstrate how to use the feature engineering pipeline
    """
    # Create sample data (replace with your actual data loading)
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = pd.DataFrame({
        'Stage_fear': np.random.normal(5, 2, n_samples),
        'Drained_after_socializing': np.random.normal(4, 1.5, n_samples),
        'Going_outside': np.random.normal(6, 2, n_samples),
        'Post_frequency': np.random.normal(3, 1, n_samples),
        'Social_event_attendance': np.random.normal(4, 1.5, n_samples),
        'Time_spent_Alone': np.random.normal(5, 2, n_samples),
        'Friends_circle_size': np.random.normal(15, 5, n_samples)
    })
    
    # Ensure non-negative values
    sample_data = sample_data.clip(lower=0)
    
    print("Sample data shape:", sample_data.shape)
    print("\nOriginal features:")
    print(sample_data.head())
    
    # Apply feature engineering
    engineered_data = advanced_feature_engineering(sample_data)
    
    print("\nEngineered data shape:", engineered_data.shape)
    print("\nNew features created:")
    new_features = [col for col in engineered_data.columns if col not in sample_data.columns]
    for feature in new_features:
        print(f"- {feature}")
    
    return engineered_data


train_imputer_df=advanced_feature_engineering(train_imputer_df)
test_imputer_df=advanced_feature_engineering(test_imputer_df)


target='Personality'
feature_columns=[col for col in train_imputer_df.columns if col!='Personality']

X_imputer=train_imputer_df[feature_columns]
y_imputer=train_imputer_df[target]

reg.fit(X_imputer,y_imputer)
pred=reg.predict(test_imputer_df)

results_df=pd.DataFrame({
    'id':test_imputer_df['id'],
    'Personality':pred
})

results_df['Personality']=results_df['Personality'].map({0:"Extrovert",1:"Introvert"})
sumission_rf_engineered=results_df.to_csv('sumission_rf_engineered.csv',index=False)


results_df


plt.figure(figsize=(12,10))
importance_df=pd.DataFrame({
    'features':feature_columns,
    'importance':reg.feature_importances_ 
})

importance_df.head()

importance_df=importance_df.sort_values(by='importance',ascending=False)
sns.barplot(importance_df,x='importance',y='features')


import shap
# SHAP values
explainer = shap.TreeExplainer(reg)
shap_values = explainer.shap_values(X_imputer)

# Plot feature importance
shap.summary_plot(shap_values[1], X_imputer)  # For class 1
shap.plots.bar(shap_values[1])


# Extract top features from your existing SHAP values
def get_top_features_from_shap(shap_values, X_train, top_n):
    """Extract top N features based on SHAP importance"""
    
    # Calculate mean absolute SHAP values (this is what the plot uses)
    feature_importance = np.abs(shap_values).mean(axis=0)
    
    # Create DataFrame with features and importance
    features_df = pd.DataFrame({
        'Feature': X_imputer.columns,
        'SHAP_Importance': feature_importance,
        'Mean_SHAP': shap_values.mean(axis=0)  # Raw mean (can be negative)
    }).sort_values('SHAP_Importance', ascending=False)
    
    # Get top N features
    top_features = features_df.head(top_n)
    
    # Create new dataset with only top features
    top_feature_names = top_features['Feature'].tolist()
    X_new = X_imputer[top_feature_names].copy()
    
    return X_new, top_features

# Use with your existing SHAP values
X_new, top_features_df = get_top_features_from_shap(shap_values[1], X_imputer, top_n=38)

print("Top 20 features from SHAP plot:")
print(top_features_df)
print(f"\nNew dataset shape: {X_new.shape}")


from xgboost import XGBClassifier
y_new=train_imputer_df['Personality']
xgb=XGBClassifier(n_estimators=1000,max_depth=4,learning_rate=0.01,random_state=42,eval_metric='mlogloss')
xgb.fit(X_new,y_new)


new_test=test_imputer_df[top_features_df['Feature'].tolist()]

pred=xgb.predict(new_test)

results_df=pd.DataFrame({
    'id':test_imputer_df['id'],
    'Personality':pred
})

results_df['Personality']=results_df['Personality'].map({0:"Extrovert",1:"Introvert"})
sumission_xgb_engineered=results_df.to_csv('sumission_xgb_engineered.csv',index=False)


results_df


low_importance_features=importance_df[importance_df['importance']<0.002]['features'].tolist()
low_importance_features


import matplotlib.pyplot as plt
import seaborn as sns

# Select a few key features for visualization
viz_features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']

# Add the actual personality and predicted personality to a temporary DataFrame for visualization
viz_df = train_engineered_clustered.copy()
viz_df['Predicted_Personality'] = xgb_ros.predict(viz_df.drop(['Personality', 'Oversampling_Cluster'], axis=1))

# Map numerical personality labels back to original names for plotting
viz_df['Personality_Label'] = viz_df['Personality'].map({0: "Extrovert", 1: "Introvert"})
viz_df['Predicted_Personality_Label'] = viz_df['Predicted_Personality'].map({0: "Extrovert", 1: "Introvert"})

# Identify misclassified points
viz_df['Misclassified'] = viz_df['Personality'] != viz_df['Predicted_Personality']

print("Sample of data with predictions and misclassification status:")
display(viz_df[['Personality_Label', 'Predicted_Personality_Label', 'Misclassified'] + viz_features].head())

# Visualize relationships with actual and predicted personality, highlighting misclassified points
for i in range(len(viz_features)):
    for j in range(i + 1, len(viz_features)):
        feat1 = viz_features[i]
        feat2 = viz_features[j]

        plt.figure(figsize=(10, 6))

        # Plot correctly classified points
        sns.scatterplot(data=viz_df[~viz_df['Misclassified']],
                        x=feat1,
                        y=feat2,
                        hue='Personality_Label',
                        style='Personality_Label',
                        palette='viridis',
                        s=50,
                        alpha=0.6,
                        label='Correctly Classified')

        # Plot misclassified points
        sns.scatterplot(data=viz_df[viz_df['Misclassified']],
                        x=feat1,
                        y=feat2,
                        hue='Misclassified',
                        palette=['red'],
                        s=100,
                        marker='X',
                        label='Misclassified')


        plt.title(f'Relationship between {feat1} and {feat2} with Predictions')
        plt.xlabel(feat1)
        plt.ylabel(feat2)
        plt.legend(title='Status')
        plt.grid(True)
        plt.show()

# You can also create plots focusing on specific features or combinations that the feature importance indicated are important.
# For example, visualizing the distribution of key features for misclassified vs correctly classified points.




