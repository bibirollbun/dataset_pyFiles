import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import optuna
from optuna.samplers import TPESampler
from scipy.stats import randint,uniform
from xgboost import plot_importance
import joblib

import warnings
warnings.filterwarnings('ignore')


train_path = '/kaggle/input/playground-series-s5e10/train.csv'
test_path = '/kaggle/input/playground-series-s5e10/test.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)


def dataset_summary(datasets):
    summary = []

    for name, df, path in datasets:
        size_on_disk = os.path.getsize(path) / (1024 * 1024)  # MB
        size_in_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        rows, cols = df.shape

        summary.append({
            "Dataset": name,
            "Size on Disk (MB)": round(size_on_disk, 2),
            "Size in Memory (MB)": round(size_in_memory, 2),
            "# of Rows": rows,
            "# of Cols": cols
        })

    return pd.DataFrame(summary)



datasets = [
    ("train", train, train_path),
    ("test", test, test_path)
]

dataset_summary(datasets)


train.head()


test.head()


train['curvature'].value_counts()


train.isnull().sum()


train.duplicated().sum()


train.info()


pd.concat([train.drop('target', axis=1, errors='ignore').dtypes, 
           test.dtypes], axis=1, keys=['train', 'test'])


train.describe().T[['mean', 'std', 'min', 'max']]


test.describe().T[['mean', 'std', 'min', 'max']]


cols = ['curvature', 'speed_limit', 'num_lanes']  

for col in cols:
    plt.figure(figsize=(6,3))
    sns.kdeplot(train[col], label='Train', fill=True)
    sns.kdeplot(test[col], label='Test', fill=True)
    plt.title(f'Distribution of {col}')
    plt.legend()
    plt.show()


sns.histplot(x='accident_risk',data=train)
plt.title('Distribution of accident risk')
plt.show()


train['accident_risk'].skew()


sns.boxplot(x='accident_risk',data=train)
plt.show()


train.info()


bool_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']
for col in bool_cols:
    train[col] = train[col].map({True: 1, False: 0}) 
    test[col] = test[col].map({True: 1, False: 0}) 

train.info()


train.nunique()


ohe=OneHotEncoder(drop=None, handle_unknown='ignore')
cat_cols=train.select_dtypes(include='object').columns.tolist()
encoded_data=ohe.fit_transform(train[cat_cols]).toarray()
encoded=pd.DataFrame(encoded_data,columns=ohe.get_feature_names_out(cat_cols))
train_numeric=train.drop(columns=cat_cols)
train=pd.concat([train_numeric.reset_index(drop=True),
                       encoded.reset_index(drop=True)],axis=1)

joblib.dump(ohe, "encoder.pkl")


ohe = joblib.load("encoder.pkl")

cat_cols_test = test.select_dtypes(include='object').columns.tolist()

# Transform test (no fitting here)
encoded_data_test = ohe.transform(test[cat_cols_test]).toarray()
encoded_test = pd.DataFrame(encoded_data_test, columns=ohe.get_feature_names_out(cat_cols_test))

test_numeric = test.drop(columns=cat_cols_test)
test = pd.concat([test_numeric.reset_index(drop=True),
                        encoded_test.reset_index(drop=True)], axis=1)


pearson = train.corr(method='pearson')['accident_risk']
spearman = train.corr(method='spearman')['accident_risk']

comparison = pd.DataFrame({'Pearson': pearson, 'Spearman': spearman}).sort_values('Pearson', ascending=False)

comparison.plot(kind='barh', figsize=(10,8))
plt.title('Pearson vs Spearman Correlation with Accident Risk')
plt.xlabel('Correlation')
plt.show()


X=train.drop(columns=['id','accident_risk'],axis=1)
y=train['accident_risk']


scaler=StandardScaler()
X_cols=X.columns
X=scaler.fit_transform(X)
X = pd.DataFrame(X, columns=X_cols)


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)


XGB = XGBRegressor(random_state=42, n_jobs=-1)
XGB.fit(X_train, y_train)
y_pred = XGB.predict(X_test)

rmse=np.sqrt(mean_squared_error(y_test, y_pred))
rmse


best_params = { 
    'n_estimators': 461,
    'learning_rate': 0.18803922060548517 ,
    'max_depth': 9 ,
    'subsample': 0.7132825589513418 ,
    'colsample_bytree': 0.9820106027924349 ,
    'gamma': 0.018499753171678533 ,
    'min_child_weight': 8 ,
    'reg_alpha': 0.1556965924250059 ,
    'reg_lambda': 0.31695815801791216
}
final_model = XGBRegressor(**best_params)
final_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = final_model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"\nTest RMSE: {rmse:.4f}")


final_model.fit(X, y)


importances = pd.Series(final_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print(importances)


test.info()


test_ids = test['id'].copy()

test_model = test.drop(columns=['id'], errors='ignore') 
test_model = pd.DataFrame(scaler.transform(test_model), columns=X_cols)
test_model = test_model.reindex(columns=X.columns, fill_value=0)
        
y_pred = final_model.predict(test_model)

submission = pd.DataFrame({
    'id': test_ids,
    'accident_risk': y_pred
})

submission.to_csv('submission.csv', index=False)
print("Submission file created!")


joblib.dump(final_model, "model.pkl", compress=3)

