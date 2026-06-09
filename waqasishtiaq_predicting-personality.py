# Suppress warnings for clean output
import warnings
warnings.filterwarnings('ignore')

# Core libraries
import os
import numpy as np
import pandas as pd

# Sklearn utilities
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.ensemble import VotingClassifier

# Gradient Boosting Libraries
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# List all files available in the Kaggle input directory
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


# Add match_p feature from external dataset
df_orig = pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data-backup/personality_datasert.csv')
df_orig = (
    df_orig
    .rename(columns={'Personality': 'match_p'})
    .drop_duplicates(['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
                      'Going_outside', 'Drained_after_socializing', 
                      'Friends_circle_size', 'Post_frequency'])
)


# Merge with original data
merge_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
              'Going_outside', 'Drained_after_socializing', 
              'Friends_circle_size', 'Post_frequency']
train = train.merge(df_orig, how='left', on=merge_cols)
test = test.merge(df_orig, how='left', on=merge_cols)


# Encode target
le = LabelEncoder()
train['target'] = le.fit_transform(train['Personality'])

# Prepare features
X = train.drop(['id', 'Personality', 'target'], axis=1)
y = train['target']
X_test = test.drop('id', axis=1)


# Combine for encoding
combined = pd.concat([X, X_test], axis=0)

# Encode all object columns
cat_cols = combined.select_dtypes(include='object').columns.tolist()
encoder = OrdinalEncoder()
combined[cat_cols] = encoder.fit_transform(combined[cat_cols])

# Split back
X = combined.iloc[:len(X)].reset_index(drop=True)
X_test = combined.iloc[len(X):].reset_index(drop=True)


# Calculate class weights
class_0 = y.sum()
class_1 = len(y) - class_0
scale_pos_weight = class_1 / class_0


# Define models with winning XGB parameters
xgb = XGBClassifier(
    max_depth=4,         
    learning_rate=0.01,   
    n_estimators=1000,    
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

cat = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.1,
    class_weights=[scale_pos_weight, 1],
    random_seed=42,
    verbose=0
)

lgbm = LGBMClassifier(
    num_leaves=31,
    learning_rate=0.1,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight={0: scale_pos_weight, 1: 1},
    random_state=42
)


# Create ensemble
ensemble = VotingClassifier(
    estimators=[
        ('xgb', xgb),
        ('cat', cat),
        ('lgbm', lgbm)
    ],
    voting='soft'
)


# Train with validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
ensemble.fit(X_train, y_train)


# Optimize threshold
val_probs = ensemble.predict_proba(X_val)[:, 1]
best_threshold = 0.5
best_acc = 0

for threshold in np.arange(0.4, 0.6, 0.01):
    preds = (val_probs >= threshold).astype(int)



# Generate predictions
test_probs = ensemble.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= best_threshold).astype(int)

# Create submission
submission = pd.DataFrame({
    'id': test['id'],
    'Personality': le.inverse_transform(test_preds)
})
submission.to_csv('submission.csv', index=False)

