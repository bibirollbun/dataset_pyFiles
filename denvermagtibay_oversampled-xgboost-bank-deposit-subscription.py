import pandas as pd
import numpy as np

# Paths to the data files
TRAIN_PATH = '/kaggle/input/playground-series-s5e8/train.csv'
TEST_PATH  = '/kaggle/input/playground-series-s5e8/test.csv'
BANK_FULL_PATH = '/kaggle/input/bank-marketing-dataset-full/bank-full.csv'

# Load datasets
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
bank_full = pd.read_csv(BANK_FULL_PATH)

# Quick sanity checks
print("=== TRAIN ===")
print("Shape:", train.shape)
print(train.head(), "\n")

print("=== TEST ===")
print("Shape:", test.shape)
print(test.head(), "\n")

print("=== ORIGINAL BANK FULL ===")
print("Shape:", bank_full.shape)
print(bank_full.head())


# 1. Data types and non-null counts
print("=== TRAIN INFO ===")
train.info()
print("\n=== TEST INFO ===")
test.info()

# 2. Missing value counts
print("\n=== TRAIN MISSING VALUES ===")
print(train.isnull().sum())
print("\n=== TEST MISSING VALUES ===")
print(test.isnull().sum())

# 3. Summary statistics for numeric columns
num_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
print("\n=== NUMERICAL FEATURE SUMMARY (train) ===")
print(train[num_cols].describe().T)


# 1. Reload the original with the correct separator
bank_full = pd.read_csv(BANK_FULL_PATH, sep=';')

# 2. Convert its target from “yes”/“no” to 1/0
bank_full['y'] = bank_full['y'].map({'no': 0, 'yes': 1})

# 3. Drop the `id` column from our playground train (we don’t need it for modeling)
train_nid = train.drop(columns=['id'])

# 4. Concatenate
combined = pd.concat([train_nid, bank_full], ignore_index=True)

# 5. Shuffle the rows
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

print("Combined shape:", combined.shape)
combined.head()


# Initial Data Exploration on `combined`

# 1. Overview: info() and first few rows
print("=== COMBINED INFO ===")
combined.info()
print("\n=== COMBINED HEAD ===")
print(combined.head())

# 2. Missing values check
print("\n=== MISSING VALUES PER COLUMN ===")
print(combined.isnull().sum())

# 3. Basic statistics for numerical features
num_cols = combined.select_dtypes(include=['int64', 'float64']).columns.tolist()
print("\n=== NUMERICAL FEATURES SUMMARY ===")
print(combined[num_cols].describe().T)


import matplotlib.pyplot as plt
import seaborn as sns

# 1. Class balance for the target
print("=== Class Distribution (y) ===")
print(combined['y'].value_counts(normalize=True).rename_axis('y').reset_index(name='proportion'))
sns.countplot(x='y', data=combined)
plt.title("Target Balance: y = 0 vs. y = 1")
plt.show()

# 2. Summary statistics for numerical features
num_cols = combined.select_dtypes(include=['int64','float64']).columns.tolist()
print("\n=== Numerical Feature Summary ===")
print(combined[num_cols].describe().T)

# 3. Categorical feature cardinality & top levels
cat_cols = combined.select_dtypes(include=['object']).columns.tolist()
print("\n=== Categorical Feature Cardinality & Top Levels ===")
for col in cat_cols:
    vc = combined[col].value_counts(dropna=False)
    print(f"\n• {col} ({vc.size} unique levels)")
    print(vc.head(5))



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# 1. Split into features & target
X = combined.drop(columns=['y'])
y = combined['y']

# 2. Hold-out split (stratified)
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 3. Combine X_train & y_train for resampling
train_df = pd.concat([X_train, y_train.rename('y')], axis=1)

# 4. Separate majority and minority classes
df_majority = train_df[train_df.y == 0]
df_minority = train_df[train_df.y == 1]

# 5. Upsample minority class to match majority size
df_minority_upsampled = resample(
    df_minority,
    replace=True,
    n_samples=len(df_majority),
    random_state=42
)

train_upsampled = pd.concat([df_majority, df_minority_upsampled])
train_upsampled = train_upsampled.sample(frac=1, random_state=42).reset_index(drop=True)

# 6. Split back into features and target
X_train_res = train_upsampled.drop(columns=['y'])
y_train_res = train_upsampled['y']

print("After upsampling, class proportions:")
print(y_train_res.value_counts(normalize=True))

# 7. Build preprocessing + model pipeline
num_cols = X_train_res.select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols = X_train_res.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_cols)
])

pipeline = Pipeline([
    ('preproc', preprocessor),
    ('clf', LogisticRegression(random_state=42, max_iter=1000))
])

# 8. Fit on the upsampled training data
pipeline.fit(X_train_res, y_train_res)

# 9. Evaluate on the original validation fold
y_val_proba = pipeline.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, y_val_proba)
print(f"Validation ROC AUC after Random Oversampling: {val_auc:.4f}")


from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

# 1. Identify categorical columns
cat_cols = X_train_res.select_dtypes(include=['object']).columns.tolist()

# 2. Ordinal-encode them (trees don’t need one-hot)
ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# 3. Apply to train and validation
X_train_enc = X_train_res.copy()
X_val_enc   = X_val.copy()

X_train_enc[cat_cols] = ord_enc.fit_transform(X_train_res[cat_cols])
X_val_enc[cat_cols]   = ord_enc.transform(X_val[cat_cols])

# 4. Instantiate XGBoost (no label encoder, use AUC)
xgb = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='auc'
)

# 5. Fit with early stopping
xgb.fit(
    X_train_enc, y_train_res,
    eval_set=[(X_val_enc, y_val)],
    early_stopping_rounds=50,
    verbose=50
)

# 6. Predict & evaluate
y_val_proba = xgb.predict_proba(X_val_enc)[:, 1]
val_auc = roc_auc_score(y_val, y_val_proba)
print(f"Validation ROC AUC (XGBoost): {val_auc:.4f}")



import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample

# 1. Prepare features and target
X = combined.drop(columns=['y'])
y = combined['y']

# 2. Identify categorical & numeric columns
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()

# 3. Preprocessor: ordinal‐encode categoricals, pass through numerics
preprocessor = ColumnTransformer([
    ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols)
], remainder='passthrough')

# 4. Stratified K-Fold setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []

# 5. Tuned XGBoost parameters from your RandomizedSearchCV
xgb_params = {
    'n_estimators':       576,
    'learning_rate':      0.3083651532392023,
    'max_depth':          8,
    'subsample':          0.8990875095589655,
    'colsample_bytree':   0.9074216057225236,
    'gamma':              0.21801885877216876,
    'reg_alpha':          0.2795603417967586,
    'reg_lambda':         0.883494022266259,
    'use_label_encoder':  False,
    'eval_metric':        'auc',
    'random_state':       42
}

# 6. Loop over folds
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
    # Split fold
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]
    
    # Combine for resampling
    tr_df = pd.concat([X_tr, y_tr.rename('y')], axis=1)
    maj = tr_df[tr_df.y == 0]
    minr = tr_df[tr_df.y == 1]
    minr_up = resample(
        minr,
        replace=True,
        n_samples=len(maj),
        random_state=42
    )
    tr_up = pd.concat([maj, minr_up]).sample(frac=1, random_state=42)
    
    # Separate back
    X_tr_up = tr_up.drop(columns=['y'])
    y_tr_up = tr_up['y']
    
    # Preprocess
    X_tr_enc = preprocessor.fit_transform(X_tr_up)
    X_val_enc = preprocessor.transform(X_val_fold)
    
    # Train tuned XGBoost
    clf = XGBClassifier(**xgb_params)
    clf.fit(X_tr_enc, y_tr_up)
    
    # Validate
    y_pred = clf.predict_proba(X_val_enc)[:, 1]
    auc = roc_auc_score(y_val_fold, y_pred)
    print(f"Fold {fold} ROC AUC: {auc:.4f}")
    auc_scores.append(auc)

# 7. Report overall CV performance
print(f"\nStratified 5-Fold CV ROC AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")


import numpy as np
import pandas as pd

def feature_engineering(df):
    df = df.copy()
    
    # 1. Age buckets
    df['age_bucket'] = pd.cut(
        df['age'],
        bins=[0, 25, 35, 50, 65, df['age'].max()],
        labels=['<=25', '26-35', '36-50', '51-65', '>65']
    ).astype(str)
    
    # 2. Balance buckets
    df['balance_bucket'] = pd.cut(
        df['balance'],
        bins=[df['balance'].min(), 0, 1000, 5000, df['balance'].max()],
        labels=['neg', '0-1k', '1k-5k', '>5k']
    ).astype(str)
    
    # 3. Duration quartile
    df['duration_q'] = pd.qcut(df['duration'], q=4, labels=False)
    
    # 4. Month → numeric → cyclical encoding
    month_map = {
        'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
        'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12
    }
    df['month_num'] = df['month'].map(month_map)
    df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)
    
    # 5. Day → cyclical encoding
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    
    # 6. Previous-contact features
    df['had_prev_contact'] = (df['pdays'] != -1).astype(int)
    # Map -1 → max+1 so it's a valid numeric
    max_pdays = df['pdays'].loc[df['pdays'] != -1].max()
    df['pdays_mod'] = df['pdays'].replace(-1, max_pdays + 1)
    
    # 7. Interaction: total number of contacts
    df['total_contacts'] = df['campaign'] + df['previous']
    
    # 8. Balance per campaign contact
    df['bal_per_camp'] = df['balance'] / (df['campaign'] + 1)
    
    return df

# Example: apply to your combined dataframe
combined_fe = feature_engineering(combined)

# Inspect the new features
print(combined_fe[['age_bucket','balance_bucket','duration_q',
                   'month_sin','month_cos','day_sin','day_cos',
                   'had_prev_contact','pdays_mod','total_contacts',
                   'bal_per_camp']].head())


import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils import resample
from xgboost import XGBClassifier

# 1. Define feature‐engineering function
def feature_engineering(df):
    df = df.copy()
    # Age buckets
    df['age_bucket'] = pd.cut(df['age'],
                              bins=[0,25,35,50,65,df['age'].max()],
                              labels=['<=25','26-35','36-50','51-65','>65']).astype(str)
    # Balance buckets
    df['balance_bucket'] = pd.cut(df['balance'],
                                  bins=[df['balance'].min(), 0, 1000, 5000, df['balance'].max()],
                                  labels=['neg','0-1k','1k-5k','>5k']).astype(str)
    # Duration quartile
    df['duration_q'] = pd.qcut(df['duration'], 4, labels=False)
    # Month → numeric → cyclical
    month_map = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
                 'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
    df['month_num'] = df['month'].map(month_map)
    df['month_sin'] = np.sin(2*np.pi*df['month_num']/12)
    df['month_cos'] = np.cos(2*np.pi*df['month_num']/12)
    # Day → cyclical
    df['day_sin'] = np.sin(2*np.pi*df['day']/31)
    df['day_cos'] = np.cos(2*np.pi*df['day']/31)
    # Previous contact flags
    df['had_prev_contact'] = (df['pdays'] != -1).astype(int)
    max_pdays = df.loc[df['pdays']!=-1,'pdays'].max()
    df['pdays_mod'] = df['pdays'].replace(-1, max_pdays+1)
    # Interaction features
    df['total_contacts'] = df['campaign'] + df['previous']
    df['bal_per_camp'] = df['balance'] / (df['campaign'] + 1)
    return df

# 2. File paths
TRAIN_PATH     = '/kaggle/input/playground-series-s5e8/train.csv'
TEST_PATH      = '/kaggle/input/playground-series-s5e8/test.csv'
BANK_FULL_PATH = '/kaggle/input/bank-marketing-dataset-full/bank-full.csv'

# 3. Load and prepare training data
train = pd.read_csv(TRAIN_PATH)
bank_full = pd.read_csv(BANK_FULL_PATH, sep=';')

train_nid = train.drop(columns=['id'])
train_nid['y']   = train_nid['y'].map({'no':0,'yes':1})
bank_full['y']   = bank_full['y'].map({'no':0,'yes':1})

combined = pd.concat([train_nid, bank_full], ignore_index=True)

# 4. Random oversample minority to match majority
df_maj = combined[combined.y==0]
df_min = combined[combined.y==1]
df_min_up = resample(
    df_min,
    replace=True,
    n_samples=len(df_maj),
    random_state=42
)
combined_up = pd.concat([df_maj, df_min_up]).sample(frac=1, random_state=42).reset_index(drop=True)

# 5. Feature-engineer the upsampled training set
combined_up_fe = feature_engineering(combined_up)
X_train_fe = combined_up_fe.drop(columns=['y'])
y_train_fe = combined_up_fe['y']

# 6. Load & feature-engineer test set
test = pd.read_csv(TEST_PATH)
test_fe = feature_engineering(test)
X_test_fe = test_fe.drop(columns=['id'])

# 7. Identify categorical columns for ordinal encoding
cat_cols = X_train_fe.select_dtypes(include=['object']).columns.tolist()

# 8. Ordinal‐encode categoricals
ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_train_enc = X_train_fe.copy()
X_train_enc[cat_cols] = ord_enc.fit_transform(X_train_fe[cat_cols])

X_test_enc = X_test_fe.copy()
X_test_enc[cat_cols] = ord_enc.transform(X_test_fe[cat_cols])

# 9. Train final XGBoost with tuned hyperparameters
xgb_params = {
    'n_estimators':     576,
    'learning_rate':    0.3083651532392023,
    'max_depth':        8,
    'subsample':        0.8990875095589655,
    'colsample_bytree': 0.9074216057225236,
    'gamma':            0.21801885877216876,
    'reg_alpha':        0.2795603417967586,
    'reg_lambda':       0.883494022266259,
    'use_label_encoder': False,
    'eval_metric':      'auc',
    'random_state':     42
}
final_model = XGBClassifier(**xgb_params)
final_model.fit(X_train_enc, y_train_fe)

# 10. Predict on test and save submission
y_test_proba = final_model.predict_proba(X_test_enc)[:, 1]
submission = pd.DataFrame({
    'id': test['id'],
    'y':  y_test_proba
})
submission.to_csv('submission.csv', index=False)
print("✅ submission.csv created!")


