import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


df_train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
df_train_ex = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
df_test  = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
df_sub = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')


df_train.head()


df_train.describe()


df_train_ex.shape,df_train.shape


df_train = pd.concat([df_train_ex, df_train], axis=0).reset_index(drop=True)
df_train.shape


df_train = df_train[:4318]


df_test.head()


df_sub.head()





import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.histplot(df_train["Price"], bins=10, kde=True, color='blue')
plt.title("Price Distribution")
plt.xlabel("Price ($)")

plt.subplot(1, 3, 2)
sns.histplot(df_train["Compartments"], bins=10, kde=True, color='green')
plt.title("Compartments Distribution")
plt.xlabel("Number of Compartments")

plt.subplot(1, 3, 3)
sns.histplot(df_train["Weight Capacity (kg)"], bins=10, kde=True, color='red')
plt.title("Weight Capacity Distribution")
plt.xlabel("Weight Capacity (kg)")

plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.boxplot(x=df_train["Price"], color='blue')
plt.title("Boxplot of Price")

plt.subplot(1, 3, 2)
sns.boxplot(x=df_train["Compartments"], color='green')
plt.title("Boxplot of Compartments")

plt.subplot(1, 3, 3)
sns.boxplot(x=df_train["Weight Capacity (kg)"], color='red')
plt.title("Boxplot of Weight Capacity")

plt.tight_layout()
plt.show()


categorical_features = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]
plt.figure(figsize=(15, 18))

for i, col in enumerate(categorical_features, 1):
    plt.subplot(4, 2, i)
    sns.boxplot(x=df_train[col], y=df_train["Price"], palette="coolwarm")
    plt.xticks(rotation=45)
    plt.ylabel("Price ($)")
    plt.title(f"Price Distribution by {col}")

plt.tight_layout()


plt.figure(figsize=(12, 6))
sns.countplot(x='Brand', data=df_train, palette='viridis')
plt.title('Brand Distribution')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(10, 6))
sns.countplot(x='Material', data=df_train, palette='Set2')
plt.title('Material Distribution')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(8, 5))
corr = df_train.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()


missing_data = df_train.isnull().sum()
missing_data = missing_data[missing_data > 0] 

if not missing_data.empty:
    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing_data.index, y=missing_data.values)
    plt.title('Missing Value Distribution in df_train')
    plt.xlabel('Columns')
    plt.ylabel('Number of Missing Values')
    plt.xticks(rotation=90)
    plt.show()
else:
    print("No missing values in the dataset.")


from statsmodels.graphics.mosaicplot import mosaic

plt.figure(figsize=(12, 6))
mosaic(df_train, ['Brand', 'Style'], title="Brand vs. Style Distribution")
plt.show()



plt.figure(figsize=(8, 5))
sns.countplot(data=df_train, x="Material", hue="Waterproof", palette="pastel")
plt.xticks(rotation=45)
plt.title("Material vs. Waterproof Feature")
plt.xlabel("Material")
plt.ylabel("Count")
plt.legend(title="Waterproof")
plt.show()



df_train.drop(columns=['id'], inplace=True)
df_test.drop(columns=['id'], inplace=True)


df_test.isnull().sum()


df_train.isnull().sum()


df_train.shape,df_test.shape


#df_train = df_train[:1694318]


# ***********************************Feature ENgineering    ********************************************
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

df_train = feature_engineering(df_train)
df_test = feature_engineering(df_test)


df_train.dtypes


df_train.columns,df_test.columns


df_train.isnull().sum()


cat = ['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment',
       'Waterproof', 'Style', 'Color']

df_train[cat] = df_train[cat].fillna('None').astype('string').astype('category')
median_weight = df_train['Weight Capacity (kg)'].median()
df_train['Weight Capacity (kg) categorical'] = df_train['Weight Capacity (kg)'].fillna(median_weight).astype('string')
df_train['Weight Capacity (kg)'] = df_train['Weight Capacity (kg)'].fillna(median_weight).astype('float64')

df_test[cat] = df_test[cat].fillna('None').astype('string').astype('category')
df_test['Weight Capacity (kg) categorical'] = df_test['Weight Capacity (kg)'].fillna(median_weight).astype('string')
df_test['Weight Capacity (kg)'] = df_test['Weight Capacity (kg)'].fillna(median_weight)


df_train.dtypes


y = df_train['Price'] 
df_train = df_train.drop(['Price'],axis=1)
X = df_train
X_test = df_test


df_train.isnull().sum()


print("Variance:", y.var())
print("Standard Deviation:",y.std())



from scipy.stats import skew
print("Skewness:", skew(y))


scaled_train_data = X
scaled_test_data = X_test


X.columns


X.shape,X_test.shape


from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from sklearn.model_selection import KFold
import gc

cat_cols = ['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment','Waterproof', 'Style', 'Color', 'Weight Capacity (kg) categorical']



from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from sklearn.model_selection import KFold
import gc

catboost_params = {'learning_rate': 0.058385610787340024, 
                   'l2_leaf_reg': 7.322579713051955, 'depth': 4,
    #'task_type': 'GPU',  
    'iterations':2000, 'loss_function':'RMSE', 'eval_metric':'RMSE', 'random_seed':42
}

cat_cols = ['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment','Waterproof', 'Style', 'Color', 'Weight Capacity (kg) categorical']
#cat_cols = ['Brand', 'Material', 'Size', 'Style', 'Color']

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
catboost_predictions = np.zeros(len(df_train))
catboost_true_labels = np.zeros(len(df_train))
catboost_test_predictions = np.zeros(len(df_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(df_train, y)):
    print(f"Training fold {fold + 1}/{n_splits}...")

    X_train, X_val = df_train.iloc[train_idx], df_train.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    catboost_model = CatBoostRegressor(**catboost_params)
    catboost_model.fit(X_train, y_train,
                       eval_set=(X_val, y_val),cat_features=cat_cols,
                       verbose=False)
    catboost_fold_preds = catboost_model.predict(X_val)
    catboost_fold_test_preds = catboost_model.predict(df_test)
    catboost_predictions[val_idx] = catboost_fold_preds
    catboost_true_labels[val_idx] = y_val
    catboost_test_predictions += catboost_fold_test_preds / n_splits  
    fold_rmse = np.sqrt(mean_squared_error(y_val, catboost_fold_preds))
    print(f"Fold {fold + 1} RMSE: {fold_rmse:.4f}")
overall_rmse_catboost = np.sqrt(mean_squared_error(catboost_true_labels, catboost_predictions))
print(f"Overall RMSE (CatBoostRegressor): {overall_rmse_catboost:.4f}")


from catboost import  Pool

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

catboost_params = {
    'learning_rate': 0.062,
    'l2_leaf_reg': 7,
    'depth': 6,
   # 'task_type': 'GPU',  
    'iterations': 3000,
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'random_seed': 42
}

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

scores = []
test_preds = []
X_test_pool = Pool(df_test, cat_features=cat_cols)

for fold, (train_idx, val_idx) in enumerate(kf.split(df_train, y)):
    print(f"Training fold {fold + 1}/{n_splits}...")
    
    X_train, X_val = df_train.iloc[train_idx], df_train.iloc[val_idx]
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



catboost_residuals = np.array(catboost_predictions) - np.array(catboost_true_labels)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 10))
axes[0, 0].scatter(catboost_predictions, catboost_residuals, color='blue', alpha=0.5)
axes[0, 0].axhline(y=0, color='red', linestyle='--')
axes[0, 0].set_title('Residual Plot (CatBoost)')
axes[0, 0].set_xlabel('Predicted Values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].grid(True)

axes[0, 1].scatter(catboost_true_labels, catboost_predictions, color='blue', alpha=0.5)
axes[0, 1].plot([min(catboost_true_labels), max(catboost_true_labels)], [min(catboost_true_labels), max(catboost_true_labels)], color='red', linestyle='--')
axes[0, 1].set_title('Actual vs. Predicted Plot (CatBoost)')
axes[0, 1].set_xlabel('Actual Values')
axes[0, 1].set_ylabel('Predicted Values')
axes[0, 1].grid(True)

importances = model.get_feature_importance(prettified=True)
importances.plot(kind='bar', x='Feature Id', y='Importances', ax=axes[1, 0])
axes[1, 0].set_title('Feature Importance (CatBoost)')
axes[1, 0].set_xlabel('Feature')
axes[1, 0].set_ylabel('Importance')

axes[1, 1].hist(catboost_residuals, bins=30, color='blue', alpha=0.5)
axes[1, 1].set_title('Residual Distribution (CatBoost)')
axes[1, 1].set_xlabel('Residuals')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].grid(True)

plt.gcf().set_facecolor('cyan')
plt.tight_layout()
plt.show()


import shap
explainer = shap.Explainer(model)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)
shap.waterfall_plot(shap_values[0]) 


test_preds_flattened = np.mean(test_preds, axis=0)
test_preds_flattened


catboost_test_predictions.shape


df_test['Price'] = test_preds_flattened


df_sub.head()


df_sub['Price'] = df_test['Price']
df_sub.to_csv('submission.csv', index=False)


df_sub


df_sub['Price'].hist()

