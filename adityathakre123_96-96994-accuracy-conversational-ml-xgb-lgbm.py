import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


train_df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
train_df.head()


train_df.info()


x = train_df.drop('y', axis = 1)
y = train_df['y']


sns.countplot(x = y);


# Set a nice style for our plots
sns.set_style('whitegrid')

print("Libraries imported successfully, Master!")

# Load the datasets from the scrolls (the CSV files)
# Make sure you've uploaded your 'train.csv' and 'test.csv' files!
try:
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
    sample_submission_df = pd.read_csv('sample_submission.csv')
    print("Datasets loaded successfully!")
except FileNotFoundError:
    print("Oh dear, apprentice! I can't find the data files. Make sure they are in the same directory or provide the correct path.")

# --- Initial Inspection ---

# Let's see the first few rows to get a feel for the data
print("\nFirst 5 rows of the training data:")
display(train_df.head())

# Get a concise summary of the dataframe.
# This is your magical 'scrying spell' to see data types and non-null counts.
print("\nTraining Data Information:")
train_df.info()

# Let's look at the summary statistics for the numerical columns
print("\nNumerical Summary of Training Data:")
display(train_df.describe())

# And what about the categorical columns?
print("\nCategorical Summary of Training Data:")
display(train_df.describe(include=['object']))


# First, let's keep track of our original columns
# We'll separate the target 'y' and the 'id' which we need for submission
target = 'y'
submission_id = 'id'

# Keep a list of the original features, excluding id and target
original_features = [col for col in train_df.columns if col not in [target, submission_id]]

# --- Feature Engineering: The 'pdays' insight ---
# Let's create our new feature.
# It's 1 if pdays is not -1, and 0 if it is.
print("Forging new feature 'was_contacted'...")
train_df['was_contacted'] = (train_df['pdays'] != -1).astype(int)
test_df['was_contacted'] = (test_df['pdays'] != -1).astype(int)
print("Done.\n")

# --- Preprocessing: The One-Hot Encoding Alchemy ---
# Identify which columns are categorical (object type)
categorical_features = train_df.select_dtypes(include=['object']).columns
print(f"Identified {len(categorical_features)} categorical features to encode: {list(categorical_features)}")

# Use pandas' get_dummies to perform the one-hot encoding
# This powerful spell handles both train and test sets consistently
print("\nPerforming One-Hot Encoding...")
all_data = pd.concat([train_df.drop(target, axis=1), test_df], axis=0)
all_data_encoded = pd.get_dummies(all_data, columns=categorical_features, dummy_na=False)

# Separate back into training and testing sets
X = all_data_encoded[:len(train_df)].drop(submission_id, axis=1)
X_test = all_data_encoded[len(train_df):].drop(submission_id, axis=1)
y = train_df[target]

print("Alchemy complete!")
print(f"Our data now has {X.shape[1]} features after encoding.")
print("\nHere's a peek at the newly forged data:")
display(X.head())


!pip install -q ucimlrepo


test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
train_df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")


# We are getting the original data from uci repo
from ucimlrepo import fetch_ucirepo


import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo # The new, proper incantation

print("Preparing to enchant our dataset with new features...")

# This assumes 'train_df' and 'test_df' from the competition are already loaded.

# Let's use the combined dataset for maximum power
if 'combined_train_df' not in globals():
    print("Recreating the combined dataset using the official repository...")
    try:
        # --- THIS IS THE CORRECTED DATA LOADING ---
        bank_marketing = fetch_ucirepo(id=222)
        X_orig = bank_marketing.data.features
        y_orig = bank_marketing.data.targets
        original_df = pd.concat([X_orig, y_orig], axis=1)
        # -----------------------------------------

        original_df['y'] = original_df['y'].apply(lambda x: 1 if x == 'yes' else 0)
        # The library correctly names the column 'day_of_week', so we rename it
        original_df.rename(columns={'day_of_week': 'day'}, inplace=True)
        
        competition_cols = [col for col in train_df.columns if col != 'id']
        original_df = original_df[competition_cols]
        
        combined_train_df = pd.concat([train_df.drop('id', axis=1), original_df], ignore_index=True)
        print("Dataset recreated successfully.")
    except Exception as e:
        print(f"Failed to recreate dataset. Error: {e}")


# --- The Enchantment Spells (Feature Engineering Functions) ---
def create_interaction_features(df):
    """Creates new features based on interactions between existing ones."""
    df_copy = df.copy()
    epsilon = 1e-6
    
    df_copy['balance_age_ratio'] = df_copy['balance'] / (df_copy['age'] + epsilon)
    df_copy['duration_campaign_ratio'] = df_copy['duration'] / (df_copy['campaign'] + epsilon)
    df_copy['pdays_previous_ratio'] = df_copy['pdays'] / (df_copy['previous'] + epsilon)
    
    return df_copy

def create_binned_features(df):
    """Creates categorical bins for numerical features."""
    df_copy = df.copy()
    
    df_copy['age_bin'] = pd.cut(df_copy['age'], bins=[0, 25, 40, 60, 100], labels=['young', 'adult', 'middle_aged', 'senior'])
    df_copy['balance_bin'] = pd.qcut(df_copy['balance'], q=4, labels=['low', 'medium', 'high', 'very_high'], duplicates='drop')
    
    return df_copy

# --- Applying the Enchantments ---
full_df = pd.concat([combined_train_df.drop('y', axis=1), test_df.drop('id', axis=1)], ignore_index=True)

print("Applying interaction feature spells...")
full_df_enchanted = create_interaction_features(full_df)

print("Applying binning feature spells...")
full_df_enchanted = create_binned_features(full_df_enchanted)

print("Enchantment complete. Our data is now more powerful.")
display(full_df_enchanted.head())

# --- Re-running the Preprocessing on the Enchanted Data ---
print("\nRe-running preprocessing...")
y_enchanted = combined_train_df['y']

categorical_features = full_df_enchanted.select_dtypes(include=['object', 'category']).columns
all_data_encoded = pd.get_dummies(full_df_enchanted, columns=categorical_features, dummy_na=False)

X_enchanted = all_data_encoded[:len(combined_train_df)]
X_test_enchanted = all_data_encoded[len(combined_train_df):]

print(f"Our enchanted data now has {X_enchanted.shape[1]} features.")


# supress warnings
import warnings
warnings.filterwarnings('ignore')


# !pip install xgboost


import xgboost as xgb
print(f"Summoning the mighty XGBoost champion (Version: {xgb.__version__})...")

# --- Train a Single, Powerful XGBoost Model ---
# We will train one strong model first to see its power.
print("\n--- Training the XGBoost Master ---")

# A solid set of starting parameters for XGBoost
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'eta': 0.02,                   # This is like learning_rate
    'max_depth': 7,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_estimators': 2000,
    'tree_method': 'gpu_hist',     # This enables GPU training!
    'random_state': 42,
}

xgb_model = xgb.XGBClassifier(**xgb_params)

# We use early stopping just like with LightGBM
# Note: XGBoost wants the data in a specific format for early stopping
eval_set = [(X_enchanted, y_enchanted)]

xgb_model.fit(X_enchanted, y_enchanted, 
              eval_set=eval_set, 
              early_stopping_rounds=150, 
              verbose=False) # Set to True if you want to see the progress

# --- Make Predictions with the New Champion ---
print("XGBoost master has cast its vote.")
xgb_predictions = xgb_model.predict_proba(X_test_enchanted)[:, 1]

# Save the predictions to a file
xgb_submission_df = pd.DataFrame({'id': test_df['id'], 'y': xgb_predictions})
xgb_submission_df.to_csv('submission_xgboost.csv', index=False)

print("\nSubmission file 'submission_xgboost.csv' has been created.")
display(xgb_submission_df.head())


print("Blending the wisdom of the rival masters...")

# Load your best LightGBM ensemble submission
lgb_submission = pd.read_csv('/kaggle/input/prediction-from-lightgbm/submission_ensemble_enchanted.csv')

# Load the new XGBoost submission
xgb_submission = pd.read_csv('submission_xgboost.csv')

# --- Create the Blended Prediction ---
# We'll give each master equal weight in the final decision
blend_predictions = 0.5 * lgb_submission['y'] + 0.5 * xgb_submission['y']

# Create the final submission file
blend_submission_df = pd.DataFrame({'id': test_df['id'], 'y': blend_predictions})
blend_submission_df.to_csv('submission_final_blend.csv', index=False)

print("\nFinal blended submission file 'submission_final_blend.csv' is ready!")
print("This is the culmination of all our efforts. Submit this to the leaderboard.")
display(blend_submission_df.head())


print("Finding the perfect harmony between the masters...")

# Load the predictions from our two models
lgb_submission = pd.read_csv('/kaggle/input/prediction-from-lightgbm/submission_ensemble_enchanted.csv')
xgb_submission = pd.read_csv('submission_xgboost.csv')

# --- Test different blending weights ---
weights = [0.6, 0.7, 0.8] # Let's test giving more weight to our original LGBM champion

for w in weights:
    print(f"Testing blend: {w*100:.0f}% LightGBM, {(1-w)*100:.0f}% XGBoost")
    
    # Create the weighted blend
    blend_preds = w * lgb_submission['y'] + (1-w) * xgb_submission['y']
    
    # Create the submission file
    blend_df = pd.DataFrame({'id': test_df['id'], 'y': blend_preds})
    filename = f'submission_blend_lgb_{w*100:.0f}.csv'
    blend_df.to_csv(filename, index=False)
    print(f"Created {filename}")

print("\nThree new potential paths are ready. Submit them to see which harmony is sweetest.")


"""
Because CatBoost is a specialist, it requires the data in a specific stateâ€”before
we perform one-hot encoding. It wants to see the original categories like 
'management' or 'married' so it can work its unique magic.
""";


# This code assumes 'full_df_enchanted' is still in memory from the
# "Ritual of Enchantment" step. It is the fully feature-engineered
# dataframe BEFORE one-hot encoding.

print("Preparing the data for the CatBoost master...")

# Isolate the training data for CatBoost
X_cat = full_df_enchanted[:len(combined_train_df)]
y_cat = combined_train_df['y']
X_test_cat = full_df_enchanted[len(combined_train_df):]

# Find the names of the categorical columns for CatBoost to handle
categorical_features_for_catboost = X_cat.select_dtypes(include=['object', 'category']).columns
cat_features_indices = [X_cat.columns.get_loc(col) for col in categorical_features_for_catboost]

print("Preparation complete. CatBoost will now see the true categories.")


# You will likely need to install catboost
# !pip install catboost

import catboost
from catboost import CatBoostClassifier
import pandas as pd

# This assumes 'full_df_enchanted' from the "Ritual of Enchantment" is in memory.

# --- A Special Preparation for a Special Master ---
print("Preparing the data for the CatBoost master...")
X_cat = full_df_enchanted[:len(combined_train_df)].copy()
y_cat = combined_train_df['y'].copy()
X_test_cat = full_df_enchanted[len(combined_train_df):].copy()

categorical_features_for_catboost = X_cat.select_dtypes(include=['object', 'category']).columns

# --- THIS IS THE FIX ---
# We find all categorical columns and fill any potential NaN values with a string.
print("Cleaning categorical features to please the master...")
for col in categorical_features_for_catboost:
    X_cat[col] = X_cat[col].astype(str).fillna('missing')
    X_test_cat[col] = X_test_cat[col].astype(str).fillna('missing')
# ----------------------

print("Preparation complete.")


# --- The Ritual of Summoning ---
print(f"\nSummoning the wise CatBoost master (Version: {catboost.__version__})...")

cat_params = {
    'iterations': 2000,
    'learning_rate': 0.02,
    'depth': 8,
    'eval_metric': 'AUC',
    'task_type': 'GPU',
    'random_seed': 42,
    'verbose': 0,
}

cat_model = CatBoostClassifier(**cat_params)

print("Training the CatBoost master... this may take some time.")
# We pass the list of categorical feature NAMES to the model
cat_model.fit(X_cat, y_cat,
              cat_features=list(categorical_features_for_catboost),
              early_stopping_rounds=150)

print("CatBoost master has cast its vote.")
cat_predictions = cat_model.predict_proba(X_test_cat)[:, 1]

cat_submission_df = pd.DataFrame({'id': test_df['id'], 'y': cat_predictions})
cat_submission_df.to_csv('submission_catboost.csv', index=False)
print("\nSubmission file 'submission_catboost.csv' has been created.")


print("Convening the Grand Council for the final verdict...")

# Load our best 70/30 blend
best_lgbm_xgb_blend = pd.read_csv('/kaggle/working/submission_blend_lgb_70.csv')

# Load the new CatBoost predictions
cat_submission = pd.read_csv('submission_catboost.csv')

# --- Create the Grand Council Blend ---
# Let's give the new master a 30% voice in the council.
# The remaining 70% goes to our already-optimized blend.
final_blend_predictions = 0.70 * best_lgbm_xgb_blend['y'] + 0.30 * cat_submission['y']

# Create the final submission file
final_submission_df = pd.DataFrame({'id': test_df['id'], 'y': final_blend_predictions})
final_submission_df.to_csv('submission_grand_council.csv', index=False)

print("\nFinal submission file 'submission_grand_council.csv' is ready!")
print("This is the final technique in the scroll. Submit it and claim your place among the legends.")


final_submission_df




