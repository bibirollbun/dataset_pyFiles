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


path = "/kaggle/input/playground-series-s5e9"


train_path = path + "/train.csv"
test_path = path + "/test.csv"


import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import seaborn as sns

from sklearn.preprocessing import PowerTransformer, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

import warnings

# Ignore all warnings
warnings.filterwarnings('ignore')



train_df = pd.read_csv(train_path)
train_df.head()


test_df = pd.read_csv(test_path)


train_df.info()
train_df.describe(include="all")


print(train_df.isnull().sum())


# FINAL CHECKPOINT ANALYSIS 
target = "BeatsPerMinute"
plt.figure(figsize=(8,5))
sns.histplot(train_df[target],kde=True,bins=40)
plt.title("distribution of BPS")
plt.show()


num_features = train_df.select_dtypes(include=[np.number]).columns.tolist() 
num_features.remove("id")

n_features = len(num_features)
n_cols = 2
n_rows = int(np.ceil(n_features / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
axes = axes.flatten()

for i, col in enumerate(num_features):
    sns.boxplot(x=train_df[col], ax=axes[i], color="lightgreen")
    axes[i].set_title(f"Boxplot of {col}")

# removing empty subplots if features are odd
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
axes = axes.flatten()

for i, col in enumerate(num_features):
    sns.histplot(train_df[col], bins=30, kde=True, ax=axes[i], color="red")
    axes[i].set_title(f"Distribution of {col}")

# remove empty subplots if features are odd
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
axes = axes.flatten()

for i, col in enumerate(num_features):
    sns.violinplot(x=train_df[col], ax=axes[i], color="yellow")
    axes[i].set_title(f"Violin Plot of {col}")

# remove empty subplots if features are odd
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()



# melt dataframe into long form
melted = train_df[num_features].melt(var_name="Feature", value_name="Value")

plt.figure(figsize=(10, 6))
sns.violinplot(data=melted, x="Value", y="Feature", scale="width", inner="quartile", palette="coolwarm")
plt.title("Ridge-style Distribution Plot")
plt.show()


# qq plots
from scipy import stats
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
axes = axes.flatten()

for i, col in enumerate(num_features):
    stats.probplot(train_df[col].dropna(), dist="norm", plot=axes[i])
    axes[i].set_title(f"Q-Q Plot of {col}")

# remove empty subplots if odd features
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# corralation matrix
plt.figure(figsize=(10,8))
corr = train_df[num_features].corr()

sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", cbar=True, square=True)
plt.title("Correlation Heatmap of Numeric Features")
plt.show()



# Pairwise Correlation Scatter (with regression line)
sns.pairplot(train_df[num_features], diag_kind="kde", kind="reg", corner=True, plot_kws={'line_kws':{'color':'red'}})
plt.show()



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Features (9 predictors)
features = ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality', 
            'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore', 
            'TrackDurationMs', 'Energy']

# Target
X = train_df[features]
y = train_df['BeatsPerMinute']

# Skewed columns for transformation
skewed_cols = ['VocalContent', 'InstrumentalScore', 'LivePerformanceLikelihood', 'TrackDurationMs']

# Power transform (Box-Cox)
pt = PowerTransformer(method='box-cox', standardize=False)
X_transformed = X.copy()
X_transformed[skewed_cols] = pt.fit_transform(X[skewed_cols] + 1e-6)

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_transformed)

# Split for validation (80/20)
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("Preprocessing done. X_train shape:", X_train.shape)


X_train.shape , X_val.shape , y_train.shape , y_val.shape


import xgboost as xgb

# Baseline model
model_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=75, learning_rate=0.005, max_depth=8, random_state=42)
model_xgb.fit(X_train, y_train)

# Predict on validation
y_pred = model_xgb.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
mae = mean_absolute_error(y_val, y_pred)
print(f'Baseline XGBoost RMSE: {rmse} BPM, MAE: {mae} BPM')

# Feature importance
importances = pd.DataFrame({'Feature': features, 'Importance': model_xgb.feature_importances_}).sort_values('Importance', ascending=False)
print(importances)


X_test = test_df[features].copy()

# Transform skewed (use train-fitted PowerTransformer)
skewed_cols = ['VocalContent', 'InstrumentalScore', 'LivePerformanceLikelihood', 'TrackDurationMs']
X_test_transformed = X_test.copy()
X_test_transformed[skewed_cols] = pt.transform(X_test_transformed[skewed_cols] + 1e-6)  # Use transform, not fit_transform

# Scale (use train-fitted StandardScaler)
X_test_scaled = scaler.transform(X_test_transformed)

print("Test set preprocessed. Shape:", X_test_scaled.shape)


import xgboost as xgb
import numpy as np

# Predict on test set
y_test_pred = model_xgb.predict(X_test_scaled)

# Clip predictions to valid BPM range (46-206) for Kaggle compliance
y_test_pred = np.clip(y_test_pred, 46, 206)

# Create submission DataFrame with id and predictions
submission = pd.DataFrame({
    'id': test_df['id'],  # Assuming test_df has an 'id' column matching the sample (starting at 524164)
    'BeatsPerMinute': y_test_pred
})

# Save submission
submission.to_csv('submission.csv', index=False)
print("Submission saved to submission.csv. Submit to Kaggle now!")


lpath="/kaggle/working/submission.csv"
dd=pd.read_csv(lpath)
dd.head()


import pandas as pd
import numpy as np
from functools import reduce
import warnings
import shutil
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.preprocessing import PolynomialFeatures
import xgboost as xgb

warnings.filterwarnings('ignore')

# === Ensemble script for BeatsPerMinute prediction ===
class UltimateEnsemble:
    def __init__(self, random_seed=42):
        # Set random seed for reproducibility
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def load_base_submission(self, test_df, model_xgb, pt, scaler):
        """
        Generate base submission using the trained XGBoost model.
        """
        # Preprocess test set
        X_test = test_df[features].copy()
        skewed_cols = ['VocalContent', 'InstrumentalScore', 'LivePerformanceLikelihood', 'TrackDurationMs']
        X_test_transformed = X_test.copy()
        X_test_transformed[skewed_cols] = pt.transform(X_test_transformed[skewed_cols] + 1e-6)
        X_test_scaled = scaler.transform(X_test_transformed)

        # Predict
        base_preds = model_xgb.predict(X_test_scaled)

        # Clip to valid range
        base_preds = np.clip(base_preds, 46, 206)

        # Create base submission DataFrame
        base_submission = pd.DataFrame({
            'id': test_df['id'],  # Assuming test_df has 'id' column
            'BeatsPerMinute': base_preds
        })
        print(f"âœ… Base submission generated: {len(base_submission)} rows")
        return base_submission

    def create_advanced_noise_variants(self, base_preds):
        """
        Generate 5 advanced noise variants based on base predictions.
        """
        variants = []
        # 1. Original kernel-style random noise (intensity and sign)
        rand_int_1 = np.random.randint(0, 101, len(base_preds))
        rand_sign_1 = np.random.choice([-1, 1], len(base_preds))
        noise_1 = (rand_int_1 / 10000.0) * rand_sign_1
        variants.append(base_preds + noise_1)

        # 2. Gaussian noise
        gaussian_noise = np.random.normal(0, 0.001, len(base_preds))
        variants.append(base_preds + gaussian_noise)

        # 3. Quantile-based micro adjustments
        quantiles = np.percentile(base_preds, [25, 50, 75])
        micro_adjust = np.where(base_preds < quantiles[0], -0.0005,
                            np.where(base_preds > quantiles[2], 0.0005, 0))
        variants.append(base_preds + micro_adjust)

        # 4. Cyclical pattern noise
        cyclical_noise = 0.0003 * np.sin(2 * np.pi * np.arange(len(base_preds)) / 100)
        variants.append(base_preds + cyclical_noise)

        # 5. Statistical outlier adjustment using z-score
        z_scores = np.abs((base_preds - np.mean(base_preds)) / np.std(base_preds))
        outlier_adjust = np.where(z_scores > 2,
                                 np.random.uniform(-0.001, 0.001, len(base_preds)), 0)
        variants.append(base_preds + outlier_adjust)

        return variants

    def create_weighted_ensemble(self, variants, strategy='adaptive'):
        """
        Combine noise variants into a weighted ensemble using given strategy.
        Supported strategies: adaptive, exponential, uniform, custom
        """
        if strategy == 'adaptive':
            variances = [np.var(v) for v in variants]
            inv_var = [1/v if v > 0 else 1 for v in variances]
            weights = np.array(inv_var) / np.sum(inv_var)
        elif strategy == 'exponential':
            weights = np.array([0.4, 0.25, 0.15, 0.12, 0.08])
        elif strategy == 'uniform':
            weights = np.ones(len(variants)) / len(variants)
        else:  # custom
            weights = np.array([0.35, 0.20, 0.20, 0.15, 0.10])

        ensemble = np.average(variants, axis=0, weights=weights)
        print(f"ğŸ“Š Ensemble weights ({strategy}): {np.round(weights, 4)}")
        return ensemble

    def apply_post_processing(self, predictions):
        """
        Post-processing: smooth boundaries and optimize precision.
        """
        processed = predictions.copy()
        mean_pred = np.mean(processed)
        std_pred = np.std(processed)

        # Boundary smoothing (clip outliers)
        upper_bound = mean_pred + 2.5 * std_pred
        lower_bound = mean_pred - 2.5 * std_pred
        processed = np.where(processed > upper_bound,
                             processed * 0.999 + mean_pred * 0.001, processed)
        processed = np.where(processed < lower_bound,
                             processed * 0.999 + mean_pred * 0.001, processed)

        # Precision optimization
        processed = np.round(processed, 7)
        return processed

    def create_multiple_submissions(self, base_submission):
        """
        Generate multiple submission files using different ensemble strategies.
        """
        base_preds = base_submission['BeatsPerMinute'].values
        strategies = ['adaptive', 'exponential', 'uniform', 'custom']
        submissions = {}
        for strategy in strategies:
            print(f"\nğŸ”„ Creating {strategy} ensemble...")
            variants = self.create_advanced_noise_variants(base_preds)
            ensemble_preds = self.create_weighted_ensemble(variants, strategy)
            final_preds = self.apply_post_processing(ensemble_preds)

            result_df = base_submission.copy()
            result_df['BeatsPerMinute'] = final_preds
            filename = f'submission_{strategy}_enhanced.csv'
            result_df.to_csv(filename, index=False)
            submissions[strategy] = result_df
            print(f"âœ… {filename} created")
        return submissions

    def create_meta_ensemble(self, submissions):
        """
        Combine all ensemble results into a final meta-ensemble.
        """
        print(f"\nğŸ�¯ Creating meta-ensemble from {len(submissions)} strategies...")
        all_preds = [sub_df['BeatsPerMinute'].values for sub_df in submissions.values()]
        meta_weights = [0.3, 0.25, 0.25, 0.2]
        meta_ensemble = np.average(all_preds, axis=0, weights=meta_weights)
        final_meta = self.apply_post_processing(meta_ensemble)

        base_df = list(submissions.values())[0].copy()
        base_df['BeatsPerMinute'] = final_meta
        base_df.to_csv('submission_meta_ultimate.csv', index=False)
        print(f"ğŸ�† Meta-ensemble created: submission_meta_ultimate.csv")
        return base_df

def run_ensemble_iterations(iterations=2, test_df=None, model_xgb=None, pt=None, scaler=None):
    """
    Run the ensemble process for the specified number of iterations.
    """
    ensemble = UltimateEnsemble(random_seed=42)
    for i in range(iterations):
        print(f"\n{'='*20} Iteration {i+1} {'='*20}")
        # 1. Load base submission using trained model
        base_submission = ensemble.load_base_submission(test_df, model_xgb, pt, scaler)
        # 2. Create multiple ensembles
        submissions = ensemble.create_multiple_submissions(base_submission)
        # 3. Create meta-ensemble
        meta_submission = ensemble.create_meta_ensemble(submissions)
        # 4. Update 'submission.csv' for next iteration
        shutil.copy('submission_meta_ultimate.csv', 'submission.csv')
        print(f"\nâœ… 'submission.csv' has been updated for the next iteration.")

    print(f"\n{'='*20} All {iterations} iterations complete. {'='*20}")
    print(f"ğŸ�† Final result is in 'submission_meta_ultimate.csv' and 'submission.csv'.")

if __name__ == '__main__':
    

    # Features and preprocessing (from your code)
    features = ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality', 
                'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore', 
                'TrackDurationMs', 'Energy']
    X = train_df[features]
    y = train_df['BeatsPerMinute']
    skewed_cols = ['VocalContent', 'InstrumentalScore', 'LivePerformanceLikelihood', 'TrackDurationMs']
    pt = PowerTransformer(method='box-cox', standardize=False)
    X_transformed = X.copy()
    X_transformed[skewed_cols] = pt.fit_transform(X[skewed_cols] + 1e-6)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_transformed)
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train XGBoost (from your code)
    model_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=75, learning_rate=0.005, max_depth=8, random_state=42)
    model_xgb.fit(X_train, y_train)

    # Run ensemble with 2 iterations (faster for time constraints)
    run_ensemble_iterations(iterations=2, test_df=test_df, model_xgb=model_xgb, pt=pt, scaler=scaler)




