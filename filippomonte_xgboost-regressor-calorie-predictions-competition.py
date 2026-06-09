# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

trainFile = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
trainFile.head()


trainFile.describe()
trainFile.info()


trainFile['Sex'].value_counts(normalize = True)
sns.barplot(x='Sex', y='Calories', data=trainFile)
plt.show()
sns.histplot(trainFile['Calories'])
plt.show()


df_non_categorical = trainFile.select_dtypes(exclude='object')
corr = df_non_categorical.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')


#Engineered Features 
#trainFile['DxHR'] = trainFile['Duration'] * trainFile['Heart_Rate']
trainFile['HRxBTxD'] = trainFile['Duration'] * trainFile['Heart_Rate']* trainFile['Duration']
#trainFile['DxBT'] = trainFile['Duration'] * trainFile['Body_Temp']
#trainFile['BTxHR'] = trainFile['Body_Temp'] * trainFile['Heart_Rate']
print(trainFile.head())




encoded_df = trainFile.copy()
encoded_df["Sex"] = LabelEncoder().fit_transform(trainFile["Sex"]) 
target = 'Calories'
y = encoded_df[target]
features = encoded_df.columns.to_list()
features.remove(target)
features
features = encoded_df[features]
features = features.drop(columns=['id', 'Height'], axis=1)
X = features
print(X.head())


from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
import numpy as np
from xgboost import plot_importance

#saved to kernel

randomGrid = {
    'n_estimators': [100, 500, 1000],  
    'learning_rate': [0.01, 0.1, 0.2],   
    'max_depth': [3, 5, 7],   
    'min_child_weight': [1, 3, 5],  
    'gamma': [0, 0.1, 0.2],   
    'subsample': [0.6, 0.8, 1.0],  
    'colsample_bytree': [0.6, 0.8, 1.0],   
    'reg_alpha': [0, 0.1, 1],   
    'reg_lambda': [1, 1.1, 1.2]   
}

 
 

# RandomizedSearchCV setup
#random_search = RandomizedSearchCV(
 
 
#random_search.fit(X, y)

bestParams = {'subsample': 1.0, 'reg_lambda': 1, 'reg_alpha': 0, 'n_estimators': 1000, 'min_child_weight': 6, 'max_depth': 6, 'learning_rate': 0.1, 'gamma': 0.19, 'colsample_bytree': 0.8}
print(bestParams)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmsle_scores = []
model = ""
for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[trn_idx], y.iloc[trn_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    model = XGBRegressor(**bestParams,
        random_state=42,
        eval_metric='rmse'  
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_pred = np.maximum(0, y_pred)   

    rmsle = np.sqrt(mean_squared_log_error(y_val, y_pred))
    print(f"Fold {fold + 1} RMSLE: {rmsle:.5f}")
    rmsle_scores.append(rmsle)

print(f"\nAverage RMSLE across folds: {np.mean(rmsle_scores):.5f}")
plot_importance(model, importance_type='gain', max_num_features=10)
plt.show()


testDF = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
testDF["Sex"] = LabelEncoder().fit_transform(testDF["Sex"]) 
#testDF['DxHR'] = testDF['Duration'] * testDF['Heart_Rate']
testDF['HRxBTxD'] = testDF['Duration'] * testDF['Heart_Rate']* testDF['Duration']
ids = testDF['id']
testDF = testDF.drop(columns=['Height', 'id'], axis=1)
print(testDF.head())
print(X.head(), y.head())
model = XGBRegressor(**bestParams, eval_metric='rmse', random_state=42)
model.fit(X,y)
 
 
 
preds = model.predict(testDF)
preds = np.maximum(0, preds)

submission = pd.DataFrame({
    'id': ids,
    'Calories': preds
})
submission.to_csv('submission.csv', index=False)


