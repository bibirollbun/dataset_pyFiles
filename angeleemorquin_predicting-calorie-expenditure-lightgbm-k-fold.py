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


import pandas as pd               # Our trusty steed for data handling ğŸ��
import numpy as np                # For powerful numerical spells and calculations âœ¨
import matplotlib.pyplot as plt   # For drawing maps of our data ğŸ—ºï¸�
import seaborn as sns             # For making those maps even prettier ğŸ�¨

from sklearn.model_selection import KFold  # To wisely split our training grounds ğŸ��ï¸�
from sklearn.preprocessing import StandardScaler, OneHotEncoder # To polish our raw materials (features) âœ¨
from sklearn.impute import SimpleImputer         # To magically fill in missing footprints ğŸ‘£
from sklearn.compose import ColumnTransformer    # A master craftsman to combine preprocessing steps ğŸ› ï¸�
from sklearn.pipeline import Pipeline            # To create an assembly line for our tasks ğŸ�­
from sklearn.metrics import mean_squared_log_error # The Oracle's scale to measure our success âš–ï¸�

import lightgbm as lgb            # Our champion model, a swift and powerful Light Gradient Boosting Machine ğŸ�‰

print("âœ¨ Supplies gathered!")


# --- Configuration ---
DATA_PATH = "/kaggle/input/playground-series-s5e5/" # The village where our scrolls are kept
TRAIN_FILE = DATA_PATH + 'train.csv'
TEST_FILE = DATA_PATH + 'test.csv'
SUBMISSION_FILE = DATA_PATH + 'sample_submission.csv'
OUTPUT_SUBMISSION_FILE = 'submission.csv' # Our final treasure map

TARGET_COL = 'Calories' # The treasure we seek! ğŸ’°
ID_COL = 'id'           # The unique marking on each adventurer in the test scroll

RANDOM_STATE = 42       # A mystical seed for reproducible magic ğŸ�²

# --- Load Data ---
print("ğŸ“‚ Unrolling the ancient scrolls...")
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)
submission_df = pd.read_csv(SUBMISSION_FILE) # A template for our treasure map

print(f"Train scroll shape: {train_df.shape}, Test scroll shape: {test_df.shape}")
print("Scrolls unrolled successfully!")


print("\n--- Scouting the Training Scroll ---")
print(train_df.head())
train_df.info()
print(train_df.describe(include='all'))

print("\n--- Checking for Missing Footprints (NaNs) ---")
print("Missing in Training Scroll:")
print(train_df.isnull().sum()[train_df.isnull().sum() > 0])
print("\nMissing in Test Scroll:")
print(test_df.isnull().sum()[test_df.isnull().sum() > 0])
print("Terrain scouted!")


print("\n--- Applying the Alchemist's Secret to Calories ---")
# We transform the target for training
train_df[TARGET_COL + '_log'] = np.log1p(train_df[TARGET_COL])


# Let's look at the distribution before and after
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(train_df[TARGET_COL], kde=True)
plt.title('Original Calories Distribution')
plt.subplot(1, 2, 2)
sns.histplot(train_df[TARGET_COL + '_log'], kde=True)
plt.title('Log-Transformed Calories Distribution')
plt.tight_layout()
plt.show()
print("Target transformed! It looks much more cooperative now.")


y_train_log = train_df[TARGET_COL + '_log']
X_train_full = train_df.drop([ID_COL, TARGET_COL, TARGET_COL + '_log'], axis=1)
X_test_full = test_df.drop([ID_COL], axis=1)


print("\n--- Identifying Feature Types (Numerical & Categorical) ---")
numerical_features = X_train_full.select_dtypes(include=np.number).columns.tolist()
categorical_features = X_train_full.select_dtypes(include='object').columns.tolist()

print(f"â›°ï¸� Numerical paths: {numerical_features}")
print(f"ğŸŒ³ Categorical trails: {categorical_features}")


print("\n--- Crafting Preprocessing Pipelines --- âš™ï¸�")
# For numerical paths: fill missing footprints with the average and then standardize their scale.
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')), # Fill missing with average
    ('scaler', StandardScaler())                  # Standardize
])


# For categorical trails: fill missing footprints with the most common one and then turn categories into numbers (one-hot encoding).
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')), # Fill missing with most common
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # Convert to numbers
])



# The Master Craftsman: ColumnTransformer applies the right pipeline to the right features.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ],
    remainder='passthrough' # Keep any columns not specified (should be none if all handled)
)

print("Preprocessing tools forged!")


print("\n--- Summoning and Training Our Champion: LightGBM ---")

# The full pipeline: Preprocess data, then train the model
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('regressor', lgb.LGBMRegressor(random_state=RANDOM_STATE))])

# For this example, we'll train on the full training data.
# In a real scenario, use K-Fold for robust validation and hyperparameter tuning.
print("Champion is training on the full training scroll...")
model_pipeline.fit(X_train_full, y_train_log)
print("ğŸ�† Champion trained!")




# --- Optional: K-Fold Cross-Validation (A Glimpse into Advanced Training) ---
# This is more robust but takes longer. For this story, we'll keep it simpler.
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
oof_predictions = np.zeros(X_train_full.shape[0])
test_predictions_kfold = np.zeros(X_test_full.shape[0])

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full, y_train_log)):
    print(f"--- Fold {fold+1} ---")
    X_train_fold, X_val_fold = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
    y_train_fold, y_val_fold = y_train_log.iloc[train_idx], y_train_log.iloc[val_idx]

    model_pipeline.fit(X_train_fold, y_train_fold)
    val_preds_log = model_pipeline.predict(X_val_fold)
    oof_predictions[val_idx] = val_preds_log

    # Predict on test set for this fold
    test_preds_log_fold = model_pipeline.predict(X_test_full)
    test_predictions_kfold += test_preds_log_fold / kf.n_splits

    val_preds_original = np.expm1(val_preds_log) # Convert back from log
    val_preds_original = np.maximum(0, val_preds_original) # Ensure no negative calories
    y_val_original = np.expm1(y_val_fold)
    fold_rmsle = np.sqrt(mean_squared_log_error(y_val_original, val_preds_original))
    print(f"Fold {fold+1} RMSLE: {fold_rmsle}")

oof_rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_train_log), np.maximum(0, np.expm1(oof_predictions))))
print(f"\nOverall OOF RMSLE: {oof_rmsle}")


print("\n--- The Champion Makes a Prophecy (Predictions on Test Scroll) ---")
test_predictions_log = model_pipeline.predict(X_test_full)
print("Prophecy received (log-transformed predictions).")


print("\n--- Decoding the Prophecy (Inverse Transformation) ---")
final_predictions = np.expm1(test_predictions_log)

# Important: Ensure no negative calorie predictions!
final_predictions = np.maximum(0, final_predictions)
print("Prophecy decoded into actual calorie predictions!")
print(final_predictions[:10]) # Show a few decoded predictions


print("\n--- Crafting the Submission Scroll ---")
submission_df[TARGET_COL] = final_predictions
submission_df.to_csv(OUTPUT_SUBMISSION_FILE, index=False)

print(f"ğŸ�‰ Submission scroll '{OUTPUT_SUBMISSION_FILE}' crafted and ready for the Oracle!")
print(submission_df.head())

