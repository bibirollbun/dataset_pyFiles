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


# Import necessary libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the training data
train_df = pd.read_csv('/kaggle/input/agriyield-2025/train.csv')
test_df = pd.read_csv('/kaggle/input/agriyield-2025/test.csv')

# Prepare features (X) and target (y)
X_train = train_df.drop(columns=['field_id', 'yield'])
y_train = train_df['yield']
X_test = test_df.drop(columns=['field_id'])

# Split the data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)




train = train_df
df = train_df


# ğŸ“¦ 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



# ğŸ“Š 3. Exploratory Data Analysis (EDA)
import matplotlib.pyplot as plt
import seaborn as sns

# Distribution of yield
plt.figure(figsize=(8,5))
sns.histplot(train_df['yield'], kde=True, bins=30, color='skyblue')
plt.title("Distribution of Agri Yield ")
plt.xlabel("Yield")
plt.ylabel("Frequency")
plt.show()



# ğŸ”— Correlation Heatmap
plt.figure(figsize=(16,12))
corr = train_df.drop(columns=["field_id"]).corr()
sns.heatmap(corr, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()



# ğŸ“ˆ Pairplot of Key Features
selected = ["soil_ph", "organic_matter", "temperature", "humidity", "rainfall", "ndvi", "yield"]
sns.pairplot(train_df[selected], diag_kind="hist")
plt.suptitle("Pairwise Relationships", y=1.02)
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
import pandas as pd

# Assuming X_train and y_train are defined

# Model definitions
models = {
    "LinearRegression": LinearRegression(),
    "XGBoost": XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
    "CatBoost": CatBoostRegressor(random_seed=42, verbose=0),
    "LightGBM": LGBMRegressor(random_state=42),
    # "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42)
}

# Train the models
for name, model in models.items():
    model.fit(X_train, y_train)

# Collect feature importances
feature_importances = {}

# Linear regression uses coefficients as feature importance
feature_importances["LinearRegression"] = np.abs(models["LinearRegression"].coef_)

# Other models use the feature_importances_ attribute
feature_importances["XGBoost"] = models["XGBoost"].feature_importances_
feature_importances["CatBoost"] = models["CatBoost"].get_feature_importance()
feature_importances["LightGBM"] = models["LightGBM"].feature_importances_
# feature_importances["HistGradientBoosting"] = models["HistGradientBoosting"].feature_importances_

# Create a dataframe to combine the feature importances for all models
importance_df = pd.DataFrame(feature_importances)

# Plot the feature importances for all models
plt.figure(figsize=(12, 8))

# Use seaborn's barplot for each model's feature importance
sns.barplot(data=importance_df, orient="h", palette="viridis")

# Add title and labels
plt.title("Feature Importances for Multiple Models")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()



# ğŸ•µï¸� 1. Check Missing Values
missing = train_df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if missing.empty:
    print("âœ… No missing values in the training set.")
else:
    print("âš ï¸� Missing values:\n", missing)



# ğŸ“‰ 2. Check Skewness of Numeric Features
from scipy.stats import skew

skewed_feats = train_df.drop(columns=["field_id"]).apply(skew).sort_values(ascending=False)
skewed_feats = skewed_feats[abs(skewed_feats) > 0.75]
print("Highly skewed features:\n", skewed_feats)



# ğŸš¨ 3. Detect Outliers with Boxplots
plt.figure(figsize=(16, 8))
for i, col in enumerate(["soil_ph", "organic_matter", "sand_pct", "temperature", "humidity", "rainfall", "ndvi"]):
    plt.subplot(2, 4, i+1)
    sns.boxplot(data=train_df, x=col, color="salmon")
    plt.title(f"{col} - Boxplot")
    plt.tight_layout()



# âš™ï¸� Install SHAP (Kaggle allows this)
!pip install shap -q



# ğŸ”� 1. Import and Prepare SHAP
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)



# ğŸŒ� 2. SHAP Summary Plot
shap.summary_plot(shap_values, X_train, plot_type="bar", show=True)



# ğŸŒˆ 3. SHAP Beeswarm Plot (Detailed per-feature impact)
shap.summary_plot(shap_values, X_train)



# ğŸ”¬ 4. SHAP Dependence Plot for Top Feature (e.g., NDVI)
shap.dependence_plot("ndvi", shap_values, X_train)



from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler



# Drop ID and target
features = train_df.drop(columns=["field_id", "yield"])
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)



# Reduce to 2 principal components
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_features)

# Plot PCA result
plt.figure(figsize=(8,6))
plt.scatter(pca_result[:,0], pca_result[:,1], c=train_df['yield'], cmap="viridis", edgecolor='k', alpha=0.7)
plt.colorbar(label="Yield (kg/ha)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("PCA Projection of Fields")
plt.grid(True)
plt.show()



# Reduce dimensionality with t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200, n_iter=1000)
tsne_result = tsne.fit_transform(scaled_features)

# Plot t-SNE result
plt.figure(figsize=(8,6))
plt.scatter(tsne_result[:,0], tsne_result[:,1], c=train_df['yield'], cmap="coolwarm", edgecolor='k', alpha=0.7)
plt.colorbar(label="Yield (kg/ha)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("t-SNE Clustering of Fields")
plt.grid(True)
plt.show()



# 1. Import Libraries
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Import models
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

print("Libraries imported successfully.")

class ModelPipeline:
    """
    Encapsulates the best-performing pipeline: full feature set with a robust stacking ensemble.
    """
    def __init__(self, train_path, test_path, submission_path):
        """
        Initializes the pipeline with file paths and model configurations.
        """
        self.train_path = train_path
        self.test_path = test_path
        self.submission_path = submission_path
        self.base_features = ['soil_ph', 'organic_matter', 'sand_pct', 'temperature', 'humidity', 'rainfall', 'ndvi']
        self.target_col = 'yield'
        self.id_col = 'field_id'
        self.models = self._get_models()
        self.meta_model = Ridge(alpha=1.0, random_state=42)

    def _get_models(self):
        """
        Defines and returns the base models with their parameters.
        """
        params = {
            'XGBoost': {'max_depth': 6, 'learning_rate': 0.03, 'n_estimators': 500, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'n_jobs': -1, 'verbosity': 0},
            'CatBoost': {'iterations': 600, 'depth': 6, 'learning_rate': 0.03, 'random_seed': 42, 'verbose': 0, 'allow_writing_files': False},
            'LightGBM': {'boosting_type': 'gbdt', 'num_leaves': 31, 'learning_rate': 0.03, 'n_estimators': 500, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'verbosity': -1},
            'HistGB': {'max_iter': 300, 'max_depth': 7, 'learning_rate': 0.03, 'min_samples_leaf': 20, 'random_state': 42}
        }
        
        xgb = XGBRegressor(**params['XGBoost'])
        cat = CatBoostRegressor(**params['CatBoost'])
        lgbm = LGBMRegressor(**params['LightGBM'])
        hgb = HistGradientBoostingRegressor(**params['HistGB'])

        return [('xgb', xgb), ('cat', cat), ('lgbm', lgbm), ('hgb', hgb)]

    def _create_features(self, df):
        """
        Creates polynomial, root, log, inverse, and interaction features.
        """
        df_new = df.copy()
        new_cols = {}

        for col in self.base_features:
            new_cols[f'{col}_sq'] = df_new[col] ** 2
            new_cols[f'{col}_sqrt'] = np.sqrt(df_new[col])
            new_cols[f'{col}_log'] = np.log1p(df_new[col])
            new_cols[f'{col}_inv'] = 1 / (df_new[col] + 1e-5)

        for f1, f2 in combinations(self.base_features, 2):
            new_cols[f'{f1}_plus_{f2}'] = df_new[f1] + df_new[f2]
            new_cols[f'{f1}_minus_{f2}'] = df_new[f1] - df_new[f2]
            new_cols[f'{f1}_times_{f2}'] = df_new[f1] * df_new[f2]
            new_cols[f'{f1}_div_{f2}'] = df_new[f1] / (df_new[f2] + 1e-5)

        new_features_df = pd.DataFrame(new_cols)
        df_new = pd.concat([df_new, new_features_df], axis=1)
        return df_new

    def run(self, n_folds=5):
        """
        Executes the full pipeline: load, feature engineer, train, predict, and submit.
        """
        # 1. Load Data
        print("Step 1: Loading data...")
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)
        test_ids = test_df[self.id_col]

        # 2. Feature Engineering
        print("Step 2: Creating features...")
        train_featured = self._create_features(train_df)
        test_featured = self._create_features(test_df)

        # 3. Prepare Data for Modeling
        print("Step 3: Preparing data for modeling...")
        drop_cols = [self.id_col, self.target_col] + [c for c in train_featured.columns if c.endswith('_missing')]
        feature_cols = [c for c in train_featured.columns if c not in drop_cols]
        
        X = train_featured[feature_cols].astype(float)
        y = train_featured[self.target_col]
        X_test = test_featured[feature_cols].astype(float)
        
        # CRITICAL FIX: Ensure test columns match train columns exactly in name and order.
        train_cols = X.columns.tolist()
        X_test = X_test.reindex(columns=train_cols, fill_value=0)

        # 4. K-Fold Stacking for Base Models
        print(f"Step 4: Training base models with {n_folds}-Fold CV...")
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        oof_preds = np.zeros((X.shape[0], len(self.models)))
        test_preds = np.zeros((X_test.shape[0], len(self.models)))

        for idx, (name, model) in enumerate(self.models):
            print(f"\n  -> Training model: {name}")
            test_fold_preds = []
            for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
                print(f"    - Fold {fold+1}/{n_folds}")
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_tr, y_tr)
                oof_preds[val_idx, idx] = model.predict(X_val)
                test_fold_preds.append(model.predict(X_test))
            
            test_preds[:, idx] = np.mean(test_fold_preds, axis=0)
            rmse = np.sqrt(mean_squared_error(y, oof_preds[:, idx]))
            print(f"  -> {name} CV RMSE: {rmse:.4f}")

        # 5. Train Meta-Model and Predict
        print("\nStep 5: Training meta-model (Ridge)...")
        self.meta_model.fit(oof_preds, y)
        final_predictions = self.meta_model.predict(test_preds)

        # 6. Create Submission File
        print("Step 6: Generating submission file...")
        submission = pd.DataFrame({self.id_col: test_ids, self.target_col: final_predictions})
        submission.to_csv(self.submission_path, index=False)
        print(f"âœ… Submission saved to {self.submission_path}")


# --- Main Execution ---
if __name__ == "__main__":
    # Define file paths
    TRAIN_FILE_PATH = '/kaggle/input/agriyield-2025/train.csv'
    TEST_FILE_PATH = '/kaggle/input/agriyield-2025/test.csv'
    SUBMISSION_FILE_PATH = '/kaggle/working/submission_final.csv'

    # Initialize and run the pipeline
    pipeline = ModelPipeline(
        train_path=TRAIN_FILE_PATH,
        test_path=TEST_FILE_PATH,
        submission_path=SUBMISSION_FILE_PATH
    )
    pipeline.run()

