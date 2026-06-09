import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import roc_auc_score


train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


train.describe().T


train.info()


print(f'Columns count is: {len(train.columns)}')
print(f'Train data count: {len(train.id)}')
print(f'Test data count: {len(test.id)}')


fig = plt.figure(figsize = (12, 8))
i = 1
for label in train.drop(columns=['id']).columns[:-2]:
    plt.subplot(3, 4, i)
    sns.scatterplot(train, x=label, y="rainfall", s=5)
    plt.title(label)
    i += 1
plt.tight_layout()
plt.show()


plt.figure(figsize=(8,8))
corr=train.drop(columns=['id']).corr()
sns.heatmap(corr,annot=True,cmap='mako',mask=np.triu(corr))
plt.show()


def create_corr_features(df, threshold=0.8, drop_original=False):
    df = df.copy()  # Avoid modifying the original dataframe
    corr = df.corr()

    # Identify highly correlated feature pairs (absolute correlation > threshold)
    high_corr_pairs = [(col1, col2) for col1 in corr.columns for col2 in corr.columns 
                       if col1 != col2 and abs(corr[col1][col2]) > threshold]

    # Generate new features
    for col1, col2 in high_corr_pairs:
        df[f'{col1}_{col2}_sum'] = df[col1] + df[col2]
        df[f'{col1}_{col2}_diff'] = df[col1] - df[col2]
        df[f'{col1}_{col2}_prod'] = df[col1] * df[col2]
        df[f'{col1}_{col2}_ratio'] = df[col1] / (df[col2] + 1e-6)  # Avoid division by zero

    # Optionally drop the original highly correlated features
    if drop_original:
        cols_to_drop = list(set([col for pair in high_corr_pairs for col in pair]))
        df.drop(columns=cols_to_drop, inplace=True)

    return df


random_state = 42

models = [
     RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, 
                            min_samples_leaf=2, random_state=random_state),
    RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=10, 
                           min_samples_leaf=4, max_features="sqrt", random_state=random_state),
    RandomForestClassifier(n_estimators=150, max_depth=15, min_samples_split=8, 
                           min_samples_leaf=3, max_features="log2", random_state=random_state),
    RandomForestClassifier(n_estimators=300, max_depth=25, min_samples_split=12, 
                           min_samples_leaf=5, max_features=0.8, random_state=random_state),
    RandomForestClassifier(n_estimators=250, max_depth=None, min_samples_split=2, 
                           min_samples_leaf=1, bootstrap=False, random_state=random_state),
    XGBClassifier(seed=random_state),
    LGBMClassifier(random_state=random_state),
    
    HistGradientBoostingClassifier(random_state=random_state)
]


# Step 1: Temporarily remove 'rainfall'
rainfall_column = train['rainfall']
train_temp = train.drop(columns=['id', 'rainfall'], axis=1)

# Step 2: Apply feature engineering to train
train_transformed = create_corr_features(train_temp, threshold=0.8, drop_original=False)

# Step 3: Handle missing values in test
test_filled = test.apply(lambda x: x.fillna(x.mean()), axis=0)

# Step 4: Apply feature engineering to test while ensuring the same columns as train
test_transformed = create_corr_features(test_filled.drop(columns=['id'], axis=1), threshold=0.8, drop_original=False)

# Step 5: Ensure test has the same columns as train
missing_cols = set(train_transformed.columns) - set(test_transformed.columns)
for col in missing_cols:
    test_transformed[col] = 0  # Add missing columns with default value (0)

extra_cols = set(test_transformed.columns) - set(train_transformed.columns)
test_transformed.drop(columns=extra_cols, inplace=True)  # Remove extra columns

# Step 6: Train-test split
X = train_transformed
y = rainfall_column

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=random_state)

# Step 7: Print data counts
print(f'X_train data count: {len(X_train)}')
print(f'y_train data count: {len(y_train)}')
print(f'X_val data count: {len(X_val)}')
print(f'y_val data count: {len(y_val)}')

# Step 8: Ensure test columns match train
test_transformed = test_transformed[X_train.columns]

print(f'X_train shape: {X_train.shape}')
print(f'Test data shape: {test_transformed.shape}')


best_model = 0
best_auc = 0

for model in models:
    model.fit(X_train, y_train)
    # pred_probs = model.predict_proba(X_val)[:, 1]
    pred_y = model.predict(X_val)

    auc_score = roc_auc_score(y_val, pred_y)
    if best_auc < auc_score:
        best_model = model
        best_auc = auc_score
        
    print(f'{model}')
    print(f'AUC sroce: {auc_score} \n\n')
    # print(pred_probs)

print(f'Best Model is: {best_model}')
print(f'Best Model auc score: {best_auc}')


y_test_pred = best_model.predict(test_transformed)
y_test_pred[:10]
y_test_pred.shape


submission = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
submission.head()


submission.rainfall = y_test_pred
submission.head()


submission.to_csv('submission.csv', index=False)




