import warnings
import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import log_loss, accuracy_score

warnings.filterwarnings("ignore")



train_df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
original_df = pd.read_csv("/kaggle/input/extrovert-vs-introvert-behavior-data/personality_datasert.csv").drop(columns={'Personality'}).drop_duplicates()



print(f"Train Dataset: {train_df.shape}")
display(train_df.head())

print("-"*130)

print(f"Test Dataset: {test_df.shape}")
display(test_df.head())

print("-"*130)

print(f"Original Dataset: {original_df.shape}")
display(original_df.head())


def dataset_summary(df, name='DataFrame'):
    print(f"\n===== Summary of {name} =====")
    
    print("\nâ�¡ï¸� General info:")
    df.info()
    
    print("\nâ�¡ï¸� Descriptive statistics:")
    display(df.describe())
    
    print(f"\nâ�¡ï¸� Number of duplicated rows: {df.duplicated().sum()}")
    
    print("\nâ�¡ï¸� Missing values per column:")
    print(df.isnull().sum())
    
    print("="*30 + "\n")



dataset_summary(train_df, 'train_df')
dataset_summary(test_df, 'test_df')
dataset_summary(original_df, 'original_df')


def show_categorical_value_counts(df,name='DataFrame'):
    categorical_cols = df.select_dtypes(include='object').columns
    print(f"\nğŸ“Š Value counts for categorical columns in {name}:\n")
    
    for col in categorical_cols:
        print(f"ğŸ”¹ Column: '{col}'")
        print(df[col].value_counts(dropna=False))
        print("-" * 50 + "\n")


show_categorical_value_counts(train_df,'train_df')
show_categorical_value_counts(test_df,'test_df')
show_categorical_value_counts(original_df,'original_df')


test_ids = test_df['id'].copy() 


merge_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
              'Going_outside', 'Drained_after_socializing',
              'Friends_circle_size', 'Post_frequency']

train_df = train_df.merge(original_df, on=merge_cols, how='left')
test_df = test_df.merge(original_df, on=merge_cols, how='left')


combined_df = pd.concat((train_df, test_df), axis=0)

print(f"combined Dataset: {combined_df.shape}")
display(combined_df.head())


combined_df = combined_df.drop('id', axis=1)
combined_df.isnull().sum()


mapping = {
    'Stage_fear': {'No': 0, 'Yes': 1},
    'Drained_after_socializing': {'No': 0, 'Yes': 1},
    'Personality': {'Extrovert': 0, 'Introvert': 1}  
}

for df in [combined_df]:
    for col, map_dict in mapping.items():
        if col in df.columns:
            df[col] = df[col].map(map_dict)


print(f"combined_df Dataset: {combined_df.shape}")
display(combined_df.head())



def fill_from_original(combined_df, original_df):
    target_columns = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
                      'Going_outside', 'Drained_after_socializing',
                      'Friends_circle_size', 'Post_frequency']
    
    for target_col in target_columns:
        key_columns = [col for col in target_columns if col != target_col]
        
        def find_value(row):
            if pd.isna(row[target_col]):
                matches = original_df[
                    (original_df[key_columns] == row[key_columns]).all(axis=1)
                ]
                if len(matches) == 1:
                    return matches[target_col].values[0]
            return row[target_col]
        
        combined_df[target_col] = combined_df.apply(find_value, axis=1)
    
    return combined_df



combined_df = fill_from_original(combined_df, original_df)


imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)

columns_to_impute = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
                     'Going_outside', 'Drained_after_socializing',
                     'Friends_circle_size', 'Post_frequency']

columns_with_missing = [col for col in columns_to_impute if combined_df[col].isnull().any()]

combined_df[columns_with_missing] = imputer.fit_transform(combined_df[columns_with_missing])

print("Missing values after IterativeImputer:")
print(combined_df[columns_with_missing].isnull().sum())



n_train = len(train_df)
train_imputed = combined_df.iloc[:n_train].copy()
test_imputed = combined_df.iloc[n_train:].copy().drop(columns=['Personality'])

print(f"Train imputed Dataset: {train_imputed.shape}")
display(train_imputed.head())

print("-" * 130)

print(f"Test imputed Dataset: {test_imputed.shape}")
display(test_imputed.head())



X = train_imputed.drop(columns=['Personality'])
y = train_imputed['Personality']
X_test = test_imputed.copy()

params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 4,
    "eta": 0.01,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}

skf = RepeatedStratifiedKFold(n_splits=7, n_repeats=5, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dval, "valid")],
        early_stopping_rounds=50,
        verbose_eval=False
    )

    oof_preds[val_idx] = model.predict(dval)
    test_preds += model.predict(dtest) / skf.get_n_splits()

ll = log_loss(y, oof_preds)
cv_acc = accuracy_score(y, oof_preds > 0.5)

print(f"\nCross-Validation log loss: {ll:.4f}, accuracy: {cv_acc:.4f}")



final_preds = (test_preds > 0.5).astype(int)

submission = pd.DataFrame({
    'id': test_ids,
    'Personality': final_preds
})

label_map = {0: 'Extrovert', 1: 'Introvert'}
submission['Personality'] = submission['Personality'].map(label_map)


submission.to_csv('submission.csv', index=False)

