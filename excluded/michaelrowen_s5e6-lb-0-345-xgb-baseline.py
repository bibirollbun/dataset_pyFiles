import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
df.head()


df = df.drop(columns=['id'])
df.info()


additional_df = pd.read_csv('/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv')
additional_df.head()


additional_df.info()


test_df = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
target_id = test_df['id']
test_df = test_df.drop(columns=['id'])
test_df.shape


full_df = pd.concat([df, additional_df], ignore_index=True)
full_df.info()


full_df.head()


fert_name = df['Fertilizer Name'].unique()
fert_mapping = {name: i for i, name in enumerate(fert_name)}
inverse_mapping = {i: name for i, name in enumerate(fert_name)}
full_df['Fertilizer Name'] = full_df['Fertilizer Name'].map(fert_mapping)
fert_name


numerical = full_df.select_dtypes(include=[np.number]).columns.tolist()
numerical.pop() # remove fertilizer name
categorical = full_df.select_dtypes(exclude=[np.number]).columns.tolist()
numerical, categorical


import warnings
warnings.filterwarnings('ignore')

plt.figure(figsize=(15, 12))
for i, col in enumerate(numerical, 1):
    plt.subplot(3, 3, i)
    sns.histplot(data=full_df, x=col, kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


plt.figure(figsize=(20, 15))
for i, feature in enumerate(numerical, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(data=full_df, x='Fertilizer Name', y=feature)
    plt.title(f'{feature} by Fertilizer Type')
    plt.xlabel('Fertilizer Name')
    plt.ylabel(feature)
    plt.xticks()

plt.tight_layout()
plt.show()


y = full_df['Fertilizer Name']
X = full_df.drop(columns=['Fertilizer Name'])
X.shape, y.shape


import optuna
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import cupy as cp
import numpy as np
scaler = StandardScaler()


X_train, y_train = X, y # Not so much for local test
for col in categorical:
    encoder = LabelEncoder()
    X_train[col] = encoder.fit_transform(X_train[col])
    test_df[col] = encoder.transform(test_df[col])

X_train= cp.asarray(X_train.values if hasattr(X_train, 'values') else X_train)


def objective(trial):
    """Optuna objective function for hyperparameter optimization"""

    params = {
        'objective': 'multi:softprob', 
        'num_class': len(np.unique(y_train)),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 2),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'mlogloss',
        'tree_method': 'hist',
        'device': "cuda"
    }
    
    model = XGBClassifier(**params)

    cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_log_loss')
    
    return np.median(cv_scores)


# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=30)

# print("Best parameters:", study.best_params)
# print("Best CV score:", study.best_value)


params = {'n_estimators': 319, 
          'max_depth': 9, 
          'learning_rate': 0.07726214802801594, 
          'subsample': 0.7870950567962739, 
          'colsample_bytree': 0.844347965836792, 
          'min_child_weight': 7, 
          'gamma': 0.5890318133775702, 
          'reg_alpha': 0.9406171225525337, 
          'reg_lambda': 1.6266675838412084,
          'random_state': 42,
          'tree_method': 'hist',
          'device': "cuda"  
         }
best_model = XGBClassifier(**params)
cv_res = cross_val_score(best_model, X_train, y_train, cv=3, scoring='neg_log_loss')

print(f"Final CV scores: {cv_res}")
print(f"Mean CV score: {cv_res.mean():.4f} ± {cv_res.std():.4f}")

best_model.fit(X_train, y_train)


probabilities = best_model.predict_proba(test_df)
predictions = []
for item in probabilities:
    top3_indices = np.argsort(item)[::-1][:3]

    names = [inverse_mapping[i] for i in top3_indices]
    predictions.append(' '.join(names))   

submission_df = pd.DataFrame({
    'id': target_id,
    'Fertilizer Name': predictions
})
submission_df.to_csv('submission.csv', index=False)
submission_df.head()

