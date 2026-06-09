import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import missingno as msno


train_df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
train_extra_df = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


merged_df = pd.concat([train_df, train_extra_df], ignore_index=True)
train = merged_df


train.sample(5)


total_rows = train.shape[0]

rows_with_missing = train.isnull().any(axis=1).sum()

rows_without_missing = total_rows - rows_with_missing

print(f"Total rows: {total_rows}")
print(f"Rows with missing data: {rows_with_missing}")
print(f"Rows without missing data: {rows_without_missing}")


missing_perc = (train.isnull().sum() / len(train)) * 100
missing_perc = missing_perc[missing_perc > 0].sort_values(ascending=False)

print("Missing value percentage per column:\n", missing_perc)
plt.figure(figsize=(12, 6))
sns.barplot(x=missing_perc.index, y=missing_perc.values)
plt.ylabel('Percentage of Missing Data')
plt.title('Missing Data Percentage by Feature')
plt.xticks(rotation=45)
plt.show()





msno.matrix(train.sample(10000), figsize=(12, 6))
plt.show()

msno.heatmap(train, figsize=(12, 6))
plt.show()


categorical_cols = ['Brand', 'Material', 'Size', 'Style', 'Color', 'Laptop Compartment', 'Waterproof']
plt.figure(figsize=(15, 25))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(len(categorical_cols), 1, i)
    sns.countplot(data=train, x=col, order=train[col].value_counts().index, palette="Set2")
    plt.xticks(rotation=45)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()



numerical_cols = ['Price', 'Weight Capacity (kg)', 'Compartments']
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, len(numerical_cols), i)
    sns.histplot(train[col], kde=False, color='skyblue')
    plt.title(f'Distribution of {col}')
    
    plt.subplot(2, len(numerical_cols), i+len(numerical_cols))
    sns.boxplot(x=train[col], color='lightgreen')
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()



plt.figure(figsize=(18, 18))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(x=col, y='Price', data=train, palette="pastel")
    plt.title(f'Price Distribution by {col}')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


train.drop(columns=['id'], inplace=True)
test_ids = test['id'].copy()
test.drop(columns=['id'], inplace=True)


test.isnull().sum()


train.isnull().sum()


train.shape,test.shape


def feature_engineering(df):
    size_mapping = {'Small': 1, 'Medium': 2, 'Large': 3}
    df['Size_Num'] = df['Size'].map(size_mapping)
    df['Compartments_per_Size'] = df['Compartments'] / df['Size_Num']    
    df['Weight_per_Compartment'] = df['Weight Capacity (kg)'] / df['Compartments'] 
    df['Waterproof'] = df['Waterproof'].map({'Yes': 1, 'No': 0})
    df['Laptop Compartment'] = df['Laptop Compartment'].map({'Yes': 1, 'No': 0})
    df['Waterproof_Laptop'] = df['Waterproof'] * df['Laptop Compartment']
    df['Is_Durable_Material'] = df['Material'].apply(lambda x: 1 if x in ['Leather', 'Nylon'] else 0)
    df['Is_Lightweight_Material'] = df['Material'].apply(lambda x: 1 if x in ['Canvas', 'Nylon'] else 0)
    df['Luxury_Material'] = df['Material'].apply(lambda x: 1 if x == 'Leather' else 0)
    df['Professional_Style'] = df['Style'].apply(lambda x: 1 if x in ['Messenger', 'Tote'] else 0)
    df['Casual_Style'] = df['Style'].apply(lambda x: 1 if x in ['Backpack', 'Duffle'] else 0)
    df['Is_Premium_Brand'] = df['Brand'].apply(lambda x: 1 if x in ['Nike', 'Under Armour', 'Adidas'] else 0)
    df['Is_Budget_Brand'] = df['Brand'].apply(lambda x: 1 if x == 'Jansport' else 0)
    df['Is_Small'] = df['Size'].apply(lambda x: 1 if x == 'Small' else 0)
    df['Is_Medium'] = df['Size'].apply(lambda x: 1 if x == 'Medium' else 0)
    df['Is_Large'] = df['Size'].apply(lambda x: 1 if x == 'Large' else 0)

    return df

train = feature_engineering(train)
test = feature_engineering(test)


cat = ['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment',
       'Waterproof', 'Style', 'Color']

train[cat] = train[cat].fillna('None').astype('string').astype('category')
median_weight = train['Weight Capacity (kg)'].median()
train['Weight Capacity (kg) categorical'] = train['Weight Capacity (kg)'].fillna(median_weight).astype('string')
train['Weight Capacity (kg)'] = train['Weight Capacity (kg)'].fillna(median_weight).astype('float64')

test[cat] = test[cat].fillna('None').astype('string').astype('category')
test['Weight Capacity (kg) categorical'] = test['Weight Capacity (kg)'].fillna(median_weight).astype('string')
test['Weight Capacity (kg)'] = test['Weight Capacity (kg)'].fillna(median_weight)


train.dtypes


y = train['Price'] 
train = train.drop(['Price'],axis=1)
X = train
X_test = test


print("Variance:", y.var())
print("Standard Deviation:",y.std())


from scipy.stats import skew
print("Skewness:", skew(y))


from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from sklearn.model_selection import KFold
import gc

cat_cols = ['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment','Waterproof', 'Style', 'Color', 'Weight Capacity (kg) categorical']


from catboost import  Pool
from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

catboost_params = {
    'learning_rate': 0.062,
    'l2_leaf_reg': 6,
    'depth': 6,
    'task_type': 'GPU',  
    'iterations': 3500,
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'random_seed': 69
}

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

scores = []
test_preds = []
X_test_pool = Pool(test, cat_features=cat_cols)

for fold, (train_idx, val_idx) in enumerate(kf.split(train, y)):
    print(f"Training fold {fold + 1}/{n_splits}...")
    
    X_train, X_val = train.iloc[train_idx], train.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    X_train_pool = Pool(X_train, y_train, cat_features=cat_cols)
    X_valid_pool = Pool(X_val, y_val, cat_features=cat_cols)
    
    model = CatBoostRegressor(**catboost_params)
    model.fit(X_train_pool, eval_set=X_valid_pool, early_stopping_rounds=200, verbose=100)
    
    val_pred = model.predict(X_valid_pool)
    score = rmse(y_val, val_pred)
    scores.append(score)
    
    test_pred = model.predict(X_test_pool)
    test_preds.append(test_pred)
    
    print(f"Fold {fold + 1} RMSE: {score:.4f}")

print(f'Optimized Cross-validated RMSE score: {np.mean(scores):.3f} +/- {np.std(scores):.3f}')
print(f'Max RMSE score: {np.max(scores):.3f}')
print(f'Min RMSE score: {np.min(scores):.3f}')


from sklearn.preprocessing import LabelEncoder

cat_cols = ['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment',
            'Waterproof', 'Style', 'Color', 'Weight Capacity (kg) categorical']

for col in cat_cols:
    encoder = LabelEncoder()
    combined_data = pd.concat([train[col], test[col]], axis=0).astype(str)
    encoder.fit(combined_data)
    train[col] = encoder.transform(train[col].astype(str))
    test[col] = encoder.transform(test[col].astype(str))

print(train.dtypes)



from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
    
xgb_params = {
    'learning_rate': 0.062,
    'max_depth': 4,
    'n_estimators': 5000,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'tree_method': 'hist',
    'device': 'cuda',
    'eval_metric': 'rmse',
    'enable_categorical': True,
    'early_stopping_rounds' : 200
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
xgb_scores = []
xgb_test_preds = []

for fold, (train_idx, val_idx) in enumerate(kf.split(train, y)):
    print(f"XGBoost - Training fold {fold + 1}/5...")
    
    X_train, X_val = train.iloc[train_idx], train.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    xgb_model = XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train, 
                  eval_set=[(X_val, y_val)], 
                  verbose=100)
    
    val_pred = xgb_model.predict(X_val)
    score = rmse(y_val, val_pred)
    xgb_scores.append(score)
    
    test_pred = xgb_model.predict(test)
    xgb_test_preds.append(test_pred)

print(f"XGBoost RMSE: {np.mean(xgb_scores):.3f} +/- {np.std(xgb_scores):.3f}")



print(f'CatBoost RMSE: {np.mean(scores):.3f} +/- {np.std(scores):.3f}')
print(f'XGBoost RMSE: {np.mean(xgb_scores):.3f} +/- {np.std(xgb_scores):.3f}')


best_model_name = min(
    [('CatBoost', np.mean(scores)), ('XGBoost', np.mean(xgb_scores))],
    key=lambda x: x[1]
)[0]

print(f"Best performing model: {best_model_name}")


if best_model_name == 'CatBoost':
    final_predictions = np.mean(test_preds, axis=0)
elif best_model_name == 'XGBoost':
    final_predictions = np.mean(xgb_test_preds, axis=0)

# SIMPLE ENSEMBLE IF NEEDED
#final_predictions = (np.mean(test_preds, axis=0) + np.mean(xgb_test_preds, axis=0)) / 2



submission = pd.DataFrame({
    'id': test_ids,   
    'Price': final_predictions
})

submission.to_csv('submission.csv', index=False)
print("ðŸ“‚ Submission file saved as 'submission.csv'")



submission.head()





