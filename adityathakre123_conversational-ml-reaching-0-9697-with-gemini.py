# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import matplotlib.pyplot as plt
import seaborn as sns


train_df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
train_df.head()


train_df.info()


x = train_df.drop('y', axis = 1)
y = train_df['y']


x.shape


sns.countplot(x = y);


# Import the foundational library for data manipulation, pandas
import pandas as pd

# Let's also import numpy for numerical operations, a trusty tool
import numpy as np

# And for some basic plotting, let's bring in matplotlib and seaborn
import matplotlib.pyplot as plt
import seaborn as sns

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


# submission


test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


train_df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")


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


import pandas as pd
import lightgbm as lgb
import numpy as np

print("The time has come to assemble the council...")

# --- THIS IS THE FIX ---
# We must recalculate pos_weight to ensure it exists in this session.
# This uses the y_enchanted variable from the previous step.
print("Recalculating the class weight for our enchanted data...")
num_neg = y_enchanted.value_counts()[0]
num_pos = y_enchanted.value_counts()[1]
pos_weight = num_neg / num_pos
print(f"Class weight calculated: {pos_weight:.2f}")
# --------------------

# These are the best parameters from our most successful submission
best_synthetic_params = {
    'learning_rate': 0.030037969603951057,
    'num_leaves': 362,
    'device': 'gpu',
    'max_depth': 43,
    'min_child_samples': 67,
    'subsample': 0.5814432711895728,
    'colsample_bytree': 0.5681011875016719,
    'reg_alpha': 0.007997273127867015,
    'reg_lambda': 0.3470006444949454,
    'objective': 'binary',
    'metric': 'auc',
    'n_estimators': 2000,
    'n_jobs': -1,
    'scale_pos_weight': pos_weight, # Now this variable exists!
}

# This code assumes X_enchanted, y_enchanted, and X_test_enchanted are in memory
N_SEEDS = 5
test_predictions = []

for seed in range(N_SEEDS):
    print(f"\n--- Training Master {seed+1}/{N_SEEDS} ---")
    
    params = best_synthetic_params.copy()
    params['random_state'] = 42 + seed
    
    model = lgb.LGBMClassifier(**params)
    
    # Train on the full ENCHANTED training data
    model.fit(X_enchanted, y_enchanted)
    
    # Predict on the corresponding ENCHANTED test data
    preds = model.predict_proba(X_test_enchanted)[:, 1]
    test_predictions.append(preds)
    print(f"Master {seed+1} has cast its vote.")

# --- The Council's Final Verdict ---
print("\nAveraging the wisdom of the council...")
final_predictions = np.mean(test_predictions, axis=0)

# --- Create the Final Submission File ---
submission_df = pd.DataFrame({'id': test_df['id'], 'y': final_predictions})
submission_df.to_csv('submission_ensemble_enchanted.csv', index=False)

print("\nSubmission file 'submission_ensemble_enchanted.csv' has been created successfully!")
print("This is the result of our council's collective wisdom. Submit it with confidence.")
display(submission_df.head())




