%%capture
!pip install catboost


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


df_train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv', index_col=0)
df_test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv', index_col=0)
df_sub = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')
df_train_extra = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv', index_col=0)


print(df_train.shape)
print(df_train_extra.shape)
print(df_test.shape)


print(df_train.columns, '\n')
print(df_train_extra.columns)


print(df_train.info())
print(df_train_extra.info())


print(df_train.isnull().sum(), '\n')
print(df_train_extra.isnull().sum(), '\n')
print(df_test.isnull().sum(), '\n')


print(df_train.duplicated().sum())
print(df_train_extra.duplicated().sum())
print(df_test.duplicated().sum())


print(df_train.describe())
print('')
print(df_train_extra.describe())
print('')
print(df_test.describe())
print('')


cols = ['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment',
       'Waterproof', 'Style', 'Color', 'Weight Capacity (kg)']
target = ['Price']
num_cols = ['Weight Capacity (kg)']
cat_cols = ['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']


for i in cols:
    print(i, df_train[i].nunique())
    print(i, df_train[i].unique())
    print('')


for i in cols:
    print(df_train[i].value_counts())
    print('')


dataframes = {'Train Data': df_train, 'Train Extra Data': df_train_extra, 'Test Data': df_test} 


for name, df in dataframes.items():
    missing_ratio = df.isnull().mean() * 100 
    missing_ratio = missing_ratio[missing_ratio > 0]  

    plt.figure(figsize=(14, 4))
    plt.bar(missing_ratio.index, missing_ratio.values, color='orange')
    plt.xlabel("Columns")
    plt.ylabel("Missing Value Percentage (%)")
    plt.title(f"Missing Values Percentage per Column ({name})")
    plt.xticks(rotation=45)
    plt.ylim(0, 5)
    plt.yticks(range(0, 6, 1))
    plt.grid(axis='y', linestyle='--', alpha=0.9)
    plt.show()


for df in [df_train, df_train_extra, df_test]:
    df[cat_cols] = df[cat_cols].fillna('missing')


print(df_train.isnull().sum(), '\n')
print(df_train_extra.isnull().sum(), '\n')
print(df_test.isnull().sum(), '\n')


num_cols = ['Weight Capacity (kg)', 'Price'] 

for name, df in dataframes.items():
    for col in num_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
            if not outliers.empty:
                print(f"Are there outliers? ({name}, {col}):")
                print(outliers)
            else:
                print(f"There are no outliers in {name} {col}")
        else:
            print(f"There are no {col} in {name}")


fig, axes = plt.subplots(4, 2, figsize=(12, 16))
axes = axes.flatten()

for idx, col in enumerate(cat_cols):
    
    cat_counts = df_train[col].value_counts()
    axes[idx].pie(cat_counts, labels=cat_counts.index, autopct='%1.1f%%', startangle=90)
    axes[idx].set_title(f"{col} Distribution")
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(4, 2, figsize=(12, 16))
axes = axes.flatten()

for idx, col in enumerate(cat_cols):
    
    cat_counts = df_train_extra[col].value_counts()
    axes[idx].pie(cat_counts, labels=cat_counts.index, autopct='%1.1f%%', startangle=90)
    axes[idx].set_title(f"{col} Distribution")
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(4, 2, figsize=(12, 16))
axes = axes.flatten()

for idx, col in enumerate(cat_cols):
    
    cat_counts = df_test[col].value_counts()
    axes[idx].pie(cat_counts, labels=cat_counts.index, autopct='%1.1f%%', startangle=90)
    axes[idx].set_title(f"{col} Distribution")
plt.tight_layout()
plt.show()


plt.figure(figsize=(4, 6))
sns.violinplot(y=df_train["Weight Capacity (kg)"], color='crimson')
plt.title("Weight Capacity (kg) Distribution")
plt.show()


plt.figure(figsize=(4, 6))
sns.violinplot(y=df_train_extra["Weight Capacity (kg)"], color='crimson')
plt.title("Weight Capacity (kg) Distribution")
plt.show()


plt.figure(figsize=(4, 6))
sns.violinplot(y=df_test["Weight Capacity (kg)"], color='crimson')
plt.title("Weight Capacity (kg) Distribution")
plt.show()


plt.figure(figsize=(4, 6))
sns.violinplot(df_train['Price'], color='crimson') 
plt.title("Price Distribution")
plt.show()


plt.figure(figsize=(4, 6))
sns.violinplot(y='Price', data=df_train_extra, color='crimson')
plt.title("Price Distribution")
plt.show()


fig, axes = plt.subplots(4, 2, figsize=(12, 24))
axes = axes.flatten()  

for idx, col in enumerate(cat_cols):
    sns.boxplot(x=df_train[col], y=df_train['Price'], ax=axes[idx])
    axes[idx].set_title(f"{col} vs Price")
    axes[idx].tick_params(axis='x', rotation=45) 

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(4, 2, figsize=(12, 24))
axes = axes.flatten()  

for idx, col in enumerate(cat_cols):
    sns.boxplot(x=df_train_extra[col], y=df_train_extra['Price'], ax=axes[idx])
    axes[idx].set_title(f"{col} vs Price")
    axes[idx].tick_params(axis='x', rotation=45) 

plt.tight_layout()
plt.show()


target = ['Price']
num_cols = ['Weight Capacity (kg)']

corr_matrix = df_train[target + num_cols].corr()

plt.figure(figsize=(6, 4))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation between Price and Weight Capacity (kg)')
plt.show()


target = ['Price']
num_cols = ['Weight Capacity (kg)']

corr_matrix = df_train_extra[target + num_cols].corr()

plt.figure(figsize=(6, 4))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation between Price and Weight Capacity (kg)')
plt.show()


merged_df = pd.concat([df_train, df_train_extra], axis=0).reset_index(drop=True)


# from MISHRA's notebook(https://www.kaggle.com/code/tarundirector/backpack-pred-baseline-ensemble-eda#%5B4%5D-%F0%9F%9B%A0%EF%B8%8F-Data-Preprocessing)

def fe(df):
    
    df['Brand_Material'] = df['Brand'] + '_' + df['Material']
    df['Brand_Size'] = df['Brand'] + '_' + df['Size']
    df['Style_Size'] = df['Style'] + '_' + df['Size']
    df['Compartments_Category'] = pd.cut(df['Compartments'], bins=[0, 2, 5, 10, np.inf], labels=['Few', 'Moderate', 'Many', 'Very Many'])
    df['Weight_Capacity_Ratio'] = df['Weight Capacity (kg)'] / df['Weight Capacity (kg)'].max()
    df['Weight_to_Compartments'] = df['Weight Capacity (kg)'] / (df['Compartments'] + 1)
    
    return df


merged_df = fe(merged_df)
df_test = fe(df_test)


# from DEOTTE's notebook (https://www.kaggle.com/code/cdeotte/feature-engineering-with-rapids-lb-38-847)

COMBO = []
for i,c in enumerate(cat_cols):
    combine = pd.concat([merged_df[c],df_test[c]],axis=0)
    combine,_ = pd.factorize(combine)
    merged_df[c] = combine[:len(merged_df)]
    df_test[c] = combine[len(merged_df):]
    n = f"{c}_wc"
    merged_df[n] = merged_df[c]*100 + merged_df["Weight Capacity (kg)"]
    df_test[n] = df_test[c]*100 + df_test["Weight Capacity (kg)"]
    COMBO.append(n)


cat_features = ['Brand', 'Material', 'Size', 'Compartments', 
                'Laptop Compartment','Waterproof', 'Style', 'Color',
                'Brand_Material', 'Brand_Size', 'Style_Size','Compartments_Category' ]
num_features = ['Weight Capacity (kg)', 'Weight_Capacity_Ratio', 'Weight_to_Compartments',
               'Brand_wc', 'Material_wc', 'Size_wc', 'Compartments_wc',
                'Laptop Compartment_wc', 'Waterproof_wc', 'Style_wc', 'Color_wc']


X = merged_df.drop(columns=['Price'])
y = merged_df['Price']


for col in cat_features:
    X[col] = X[col].astype(str)


kf = KFold(n_splits=10, shuffle=True, random_state=42)
oof_preds = np.zeros(X.shape[0])


fold_rmse = []
for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    print(f"Fold {fold+1}")
    
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    
    train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features)
    valid_pool = Pool(data=X_valid, label=y_valid, cat_features=cat_features)
    
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        eval_metric='RMSE',
        random_seed=42,
        early_stopping_rounds=50,
        verbose=100,
        task_type='GPU' 
    )
    
    model.fit(train_pool, eval_set=valid_pool, verbose=False)
    
    preds = model.predict(valid_pool)
    oof_preds[valid_idx] = preds
    
    rmse = mean_squared_error(y_valid, preds, squared=False)
    fold_rmse.append(rmse)
    print(f'Fold {fold+1} RMSE: {rmse:.4f}\n')

overall_rmse = mean_squared_error(y, oof_preds, squared=False)
print(f'Overall RMSE: {overall_rmse:.4f}')

print(f'Mean RMSE across folds: {np.mean(fold_rmse):.4f}')


feature_importances = model.get_feature_importance(train_pool)
feature_names = X_train.columns
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importances
}).sort_values(by='importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=importance_df, palette='viridis')
plt.title('CatBoost Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()


for col in cat_features:
    df_test[col] = df_test[col].astype(str)


y_pred = model.predict(df_test)


submission = pd.DataFrame({
    'id' : df_sub['id'],
    'Price' : y_pred
})
submission.to_csv('submission.csv', index=False)


df_confirm = pd.read_csv('submission.csv')
df_confirm.head()




