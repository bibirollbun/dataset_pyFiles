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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df=pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
df1=pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')


test_ids=df1['id']
df1.drop('id',axis=1)


df.dropna(axis=1)


df.drop('BeatsPerMinute',axis=1)
y=df['BeatsPerMinute']



corr_matrix=df.corr()
plt.figure(figsize=(10,12))
sns.heatmap(corr_matrix,annot=True,cmap='rainbow',fmt='.2f')
plt.title('Corelation matrix')
plt.show()


X_scaled = StandardScaler().fit_transform(df)
X_test_scaled=StandardScaler().fit_transform(df1)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)
principal_test_components=pca.fit_transform(X_test_scaled)
pca_df1=pd.DataFrame(data=principal_test_components,columns=['PC1','PC2'])
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Target'] =y
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='PC1', 
    y='PC2', 
    hue='Target', 
    data=pca_df, 
    palette='viridis', # Use a continuous colormap
    alpha=0.6
)
plt.title('2D PCA of 11 Features (Colored by Target Value)')
plt.show()


pca_df.head()


BEST_XGB_PARAMS = {
    'n_estimators': 500,  # Example parameter
    'learning_rate': 0.05, # Example parameter
    'max_depth': 5,        # Example parameter
    'gamma': 0.1,          # Example parameter
    'subsample': 0.8,
    'colsample_bytree': 0.8
}



df2=pca_df.drop(['Target'],axis=1)


pca_df['Target']


models = {
    # 1. Optimal Boosting Model
    'XGBoost (Optimal)': XGBRegressor(
        objective='reg:squarederror', 
        random_state=42, 
        **BEST_XGB_PARAMS
    ),
    
    # 2. Bagging Model (Robust Baseline)
    'Random Forest': RandomForestRegressor(
        n_estimators=300,        # A reasonable number of trees
        max_depth=10,            # Limit depth to prevent massive overfitting
        random_state=42,
        n_jobs=-1
    ),
    
    # 3. Instance-Based Model (Non-parametric baseline)
    'K-Nearest Neighbors': KNeighborsRegressor(
        n_neighbors=10           # Common starting point for neighbors
    )
}

results = []

# --- TRAINING AND EVALUATION LOOP ---
for name, model in models.items():
    print(f"Training {name}...")
    
    # Train the model on the PCA training data
    model.fit(df2, pca_df['Target'])
    
    # Predict on the PCA test data
    y_pred = model.predict(pca_df1)
# --- COMPARISON TABLE ---
comparison_df = pd.DataFrame(results)



submission_df = pd.DataFrame({
    'id': test_ids,
    # IMPORTANT: Use the exact column name for the target expected by the competition
    'BeatsPerMinute': y_pred
})

# Save the DataFrame to a CSV file without the index
submission_df.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' generated successfully!")





