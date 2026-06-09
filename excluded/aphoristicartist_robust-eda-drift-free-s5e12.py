
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import warnings

# Configuration
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", palette="viridis")
pd.set_option('display.max_columns', None)

# Constants
SEED = 42
N_SPLITS = 5
TARGET = 'diagnosed_diabetes'
    



# Load Data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')

print(f"Train Shape: {train_df.shape}")
print(f"Test Shape: {test_df.shape}")
display(train_df.head(3))
    



# Target Distribution
plt.figure(figsize=(6, 4))
ax = sns.countplot(data=train_df, x=TARGET)
plt.title(f'Target Distribution: {TARGET}')
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')
plt.show()
    



# Prepare Adversarial Data
train_adv = train_df.drop(columns=[TARGET, 'id']).copy()
test_adv = test_df.drop(columns=['id']).copy()

train_adv['is_test'] = 0
test_adv['is_test'] = 1

adv_df = pd.concat([train_adv, test_adv], ignore_index=True)
adv_y = adv_df.pop('is_test')

# Encode cats
for col in adv_df.select_dtypes('object').columns:
    le = LabelEncoder()
    adv_df[col] = le.fit_transform(adv_df[col].astype(str))

# Train LGBM
model_adv = lgb.LGBMClassifier(random_state=SEED, verbose=-1)
model_adv.fit(adv_df, adv_y)

# Check AUC
from sklearn.model_selection import cross_val_score
adv_auc = cross_val_score(model_adv, adv_df, adv_y, cv=3, scoring='roc_auc').mean()

print(f"Adversarial Validation AUC: {adv_auc:.4f}")
if adv_auc > 0.60:
    print("âš ï¸� Significant Drift Detected! We MUST use Global Aggregations.")
else:
    print("âœ… Drift is minimal.")
    
# Feature Importance
imp = pd.DataFrame({'feature': adv_df.columns, 'gain': model_adv.feature_importances_})
display(imp.sort_values('gain', ascending=False).head(10))
    


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from itertools import combinations

class UnifiedFeatureEngineer:
    """
    Generic Unified Feature Engineering Pipeline.
    Transferred from S5E11 Grandmaster Solution.
    Enhanced for S5E12 with Digit Features.
    """

    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state
        
        self.target_encodings = {}
        self.cat_means = {}
        self.train_columns = None
        
        # Specific columns from S5E12 EDA
        self.numeric_cols = [
            'age', 'alcohol_consumption_per_week', 'physical_activity_minutes_per_week', 
            'diet_score', 'sleep_hours_per_day', 'screen_time_hours_per_day', 
            'bmi', 'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 
            'heart_rate', 'cholesterol_total', 'hdl_cholesterol', 
            'ldl_cholesterol', 'triglycerides',
            'family_history_diabetes', 'hypertension_history', 'cardiovascular_history'
        ] 
        self.categorical_cols = [
            'gender', 'ethnicity', 'education_level', 
            'income_level', 'smoking_status', 'employment_status'
        ]

    def fit_transform(self, X, y, test_df=None):
        """
        Fit on training data and transform it.
        CRITICAL: If test_df is provided, calculates Global Aggregations to prevent drift.
        """
        df = X.copy()
        
        # Ensure y is Series
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        df['target'] = y
        
        # 1. Extract Digit Features (The "Magic")
        df = self._extract_digits(df)
        if test_df is not None:
            test_df = self._extract_digits(test_df)
            
        # Update categorical list with new digit features
        digit_cols = [c for c in df.columns if c.endswith('_last') or c.endswith('_len') or c.endswith('_first')]
        self.categorical_cols.extend(digit_cols)

        # 2. Global Aggregations (The Drift Fix)
        if test_df is not None:
            print("Using Global Aggregations (Train + Test)...")
            # Align columns before concat
            cols_to_keep = [c for c in df.columns if c != 'target']
            full_df = pd.concat([df[cols_to_keep], test_df[cols_to_keep]], ignore_index=True)
            self._fit_global_aggregations(full_df)
            df = self._apply_aggregations(df)
        else:
            print("Warning: No Test DF provided. Using OOF Aggregations (Drift Risk).")
            df = self._add_aggregations_oof(df, y)
        
        # 3. Target Encoding (OOF for Train)
        df = self._add_target_encoding_oof(df, y)
        
        # 4. Cleanup
        if 'target' in df.columns:
            df = df.drop(columns=['target'])
            
        self.train_columns = df.columns.tolist()
        return df

    def transform(self, X):
        df = X.copy()
        
        # 1. Digits
        df = self._extract_digits(df)
        
        # 2. Apply mappings
        df = self._apply_aggregations(df)
        df = self._apply_target_encoding(df)
        
        # 3. Align
        if self.train_columns:
            for col in self.train_columns:
                if col not in df.columns:
                    df[col] = 0
            df = df[self.train_columns]
            
        return df

    def _extract_digits(self, df):
        # Extract last digit, length, etc from integer-like numerics
        # Focus on the top drifting features: activity, triglycerides, bmi, cholesterol
        candidates = ['physical_activity_minutes_per_week', 'triglycerides', 'cholesterol_total', 'systolic_bp']
        
        for col in candidates:
            if col in df.columns:
                s = df[col].astype(str)
                df[f'{col}_last'] = s.str[-1]
                df[f'{col}_len'] = s.apply(len)
                # df[f'{col}_first'] = s.str[0] # Benford
        return df

    def _fit_global_aggregations(self, full_df):
        # Calculate Mean/Std of Numerics grouped by Categoricals
        for cat in self.categorical_cols:
            for num in self.numeric_cols:
                if cat not in full_df.columns or num not in full_df.columns: continue
                
                # Skip if num is same as cat source (e.g. digit features)
                if num in cat: continue 

                # Mean
                name_mean = f'agg_mean_{cat}_{num}'
                self.cat_means[name_mean] = full_df.groupby(cat)[num].mean().to_dict()
                self.cat_means[f'{name_mean}_global'] = full_df[num].mean()
                
                # Std
                name_std = f'agg_std_{cat}_{num}'
                self.cat_means[name_std] = full_df.groupby(cat)[num].std().to_dict()
                self.cat_means[f'{name_std}_global'] = full_df[num].std()

    def _add_aggregations_oof(self, df, y):
        # Fallback for when no test_df is provided
        # Use StratifiedKFold to match training pipeline and prevent leakage
        kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for cat in self.categorical_cols:
            for num in self.numeric_cols:
                if cat not in df.columns or num not in df.columns: continue
                if num in cat: continue
                
                name_mean = f'agg_mean_{cat}_{num}'
                df[name_mean] = np.nan
                self.cat_means[name_mean] = df.groupby(cat)[num].mean().to_dict() # Global for transform later
                self.cat_means[f'{name_mean}_global'] = df[num].mean()
                
                for train_idx, val_idx in kf.split(df, y):
                    X_tr, X_val = df.iloc[train_idx], df.iloc[val_idx]
                    means = X_tr.groupby(cat)[num].mean()
                    df.loc[X_val.index, name_mean] = X_val[cat].map(means)
                
                df[name_mean] = df[name_mean].fillna(self.cat_means[f'{name_mean}_global'])
        return df

    def _apply_aggregations(self, df):
        for cat in self.categorical_cols:
            for num in self.numeric_cols:
                name_mean = f'agg_mean_{cat}_{num}'
                if name_mean in self.cat_means:
                    df[name_mean] = df[cat].map(self.cat_means[name_mean]).fillna(self.cat_means[f'{name_mean}_global'])
                    
                name_std = f'agg_std_{cat}_{num}'
                if name_std in self.cat_means:
                    df[name_std] = df[cat].map(self.cat_means[name_std]).fillna(self.cat_means[f'{name_std}_global'])
        return df

    def _add_target_encoding_oof(self, df, y):
        # Use StratifiedKFold to match training pipeline
        kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        for cat in self.categorical_cols:
            new_col = f'te_{cat}'
            df[new_col] = np.nan
            
            # Global Fit
            self.target_encodings[new_col] = df.groupby(cat)['target'].mean().to_dict()
            self.target_encodings[f'{new_col}_global'] = df['target'].mean()
            
            # OOF Transform
            for train_idx, val_idx in kf.split(df, y):
                X_tr, X_val = df.iloc[train_idx], df.iloc[val_idx]
                means = X_tr.groupby(cat)['target'].mean()
                df.loc[X_val.index, new_col] = X_val[cat].map(means)
            
            df[new_col] = df[new_col].fillna(self.target_encodings[f'{new_col}_global'])
        return df

    def _apply_target_encoding(self, df):
        for cat in self.categorical_cols:
            new_col = f'te_{cat}'
            if new_col in self.target_encodings and cat in df.columns:
                df[new_col] = df[cat].map(self.target_encodings[new_col])
                df[new_col] = df[new_col].fillna(self.target_encodings[f'{new_col}_global'])
        return df



# Initialize and Run
print("Engineering Features...")
fe = UnifiedFeatureEngineer(n_splits=N_SPLITS, random_state=SEED)

# Pass test_df to fit_transform to enable Global Aggregations
X = fe.fit_transform(train_df.drop(columns=['id', TARGET]), train_df[TARGET], test_df=test_df.drop(columns=['id']))
X_test = fe.transform(test_df.drop(columns=['id']))
y = train_df[TARGET]

# Drop non-numeric columns (Original Categoricals) as we have Target Encoded them
X = X.select_dtypes(include=[np.number])
X_test = X_test.select_dtypes(include=[np.number])

print(f"Generated {X.shape[1]} features.")
display(X.head(3))
    



# Train CatBoost
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

params = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'learning_rate': 0.03,
    'iterations': 2000,
    'depth': 6,
    'subsample': 0.8,
    'colsample_bylevel': 0.8,
    'random_seed': SEED,
    'verbose': 500,
    'allow_writing_files': False
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = CatBoostClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=100)
    
    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS
    
    print(f"Fold {fold+1} AUC: {roc_auc_score(y_val, oof_preds[val_idx]):.5f}")

print(f"Overall CV AUC: {roc_auc_score(y, oof_preds):.5f}")
    



# Create Submission
submission = pd.DataFrame({'id': test_df['id'], 'diagnosed_diabetes': test_preds})
submission.to_csv('submission.csv', index=False)
print("Submission saved successfully!")
display(submission.head())
    

