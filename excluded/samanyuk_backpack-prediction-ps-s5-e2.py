# importing
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv") # original training data
train1= pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv") # extra training data
df= pd.concat([train,train1],axis=0,ignore_index=True) # merging both

te=pd.read_csv(r'/kaggle/input/playground-series-s5e2/test.csv') # testing data

print("Final Train shape: ",df.shape,"\nTesting shape: ",te.shape)


# importing
import seaborn as sns
import matplotlib.pyplot as plt
#from cuml.preprocessing import TargetEncoder


df.info()


# Let us first look at the null values and visualize them using a heatmap
plt.figure(figsize=(20,10))
sns.heatmap(df.isnull())
plt.show()


cols=[col for col in df.columns if df[col].isnull().any()]
for col in cols:
    print(col,'->',df[col].dtypes)


# Let us fill all the categorical features null values with None
# Let us fill the null value of the reamaining with its median
cols=[col for col in df.columns if df[col].isnull().any()]
for col in cols:
    if df[col].dtype=='object':
        df[col]=df[col].fillna('None')
df['Weight Capacity (kg)']=df['Weight Capacity (kg)'].fillna(df['Weight Capacity (kg)'].median())


# Let us drop duplicates and then drop the id column
df.drop_duplicates(inplace=True,keep='first')
df.drop(columns='id',inplace=True)


df.info()


# We will use Target Encoder to encode the categorical columns, let us do that after we clean our test data and visualize the data

cols=[col for col in te.columns if te[col].isnull().any()]
for col in cols:
    if te[col].dtype=='object':
        te[col]=te[col].fillna('None')
te['Weight Capacity (kg)'] = te['Weight Capacity (kg)'].fillna(te['Weight Capacity (kg)'].median())
tte=te.copy()
te.drop(columns='id',inplace=True)


# This section will only show a small part of the visualization
df.info()


# Let us view the distributions of the categorical features with the help of a countplot
fig,ax=plt.subplots(4,2,figsize=(20,20))
ax=ax.flatten()
i=0
for col in df.columns[df.dtypes=='object']:
    sns.countplot(data=df,x=col,ax=ax[i])
    i+=1
sns.countplot(data=df,x='Compartments',ax=ax[i])
plt.tight_layout()
plt.show()


# Let us plot Weight Capacity and our Target Variable Price using a kdeplot
fig,ax=plt.subplots(1,2,figsize=(20,5))
sns.kdeplot(data=df,x='Weight Capacity (kg)',ax=ax[0])
sns.kdeplot(data=df,x='Price',ax=ax[1])
plt.tight_layout()
plt.show()


# Let us look at each of our brands and what it is made up of 
# We will plot the same bar plot as above but differentiate it based on brands (Hue-> Brand)
# We will then plot two more box plots to 

fig,ax=plt.subplots(4,2,figsize=(20,20))
ax=ax.flatten()
i=0
for col in df.columns[df.dtypes=='object']:
    if col !='Brand':
        sns.countplot(data=df,x=col,ax=ax[i],hue='Brand')
        i+=1
sns.countplot(data=df,x='Compartments',ax=ax[i],hue='Brand')
ax[7].axis('off')
plt.tight_layout()
plt.show()



fig,ax=plt.subplots(4,2,figsize=(20,20))
ax=ax.flatten()
i=0
for col in df.columns[df.dtypes=='object']:
    sns.boxplot(data=df,y='Weight Capacity (kg)',ax=ax[i],x=col)
    i+=1
sns.boxplot(data=df,y='Weight Capacity (kg)',ax=ax[i],x='Compartments')
plt.tight_layout()
plt.show()


fig,ax=plt.subplots(4,2,figsize=(20,20))
ax=ax.flatten()
i=0
for col in df.columns[df.dtypes=='object']:
    sns.boxplot(data=df,y='Price',ax=ax[i],x=col)
    i+=1
sns.boxplot(data=df,y='Price',ax=ax[i],x='Compartments')
plt.tight_layout()
plt.show()


# Let us Target Encode our features
# importing
from cuml.preprocessing import TargetEncoder


en=TargetEncoder(n_folds=25, smooth=20, split_method='random', stat='mean')

for col in df.columns[df.dtypes == 'object']:  
    en.fit(train[col], train['Price'])
    df[col] = en.transform(df[col]) 
    te[col] = en.transform(te[col])


# importing
from sklearn.feature_selection import mutual_info_regression


# We will visualize the correlation and calculate the Mutual information
# Calculating Mutual Information
x=df.drop(columns='Price')
y=df['Price']
mi=mutual_info_regression(x,y)
mi_df=pd.DataFrame({'cols':x.columns,'mi':mi})
mi_df.sort_values(by='mi',inplace=True,ascending=False)


fig,ax=plt.subplots(2,1,figsize=(20,8))
sns.heatmap(df.corr(),annot=True,ax=ax[0])
sns.barplot(data=mi_df,x='mi',y='cols',ax=ax[1])
plt.tight_layout()
plt.show()


# importing
from sklearn.model_selection import train_test_split


X=df.drop(columns='Price') 
y=df['Price']
# train and validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)


# importing
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV,KFold
from sklearn.metrics import mean_squared_error


if isinstance(X, pd.DataFrame):
    X = X.apply(pd.to_numeric, errors="coerce")  # Converts non-numeric to NaN
    X.fillna(0, inplace=True)
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

xgb_model = XGBRegressor(
    tree_method="hist",
    device="cuda",
    predictor="gpu_predictor",
    max_depth=5,
    n_estimators=1000,
    learning_rate=0.015,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=20,
    eval_metric="rmse"
)

kf = KFold(n_splits=3, shuffle=True, random_state=42)
cv_scores = []
for train_index, val_index in kf.split(X):
    X_train_fold, X_val_fold = X[train_index], X[val_index]
    y_train_fold, y_val_fold = y[train_index], y[val_index]

    xgb_model.fit(X_train_fold, y_train_fold)
    y_pred_fold = xgb_model.predict(X_val_fold)
    fold_rmse = mean_squared_error(y_val_fold, y_pred_fold, squared=False)
    cv_scores.append(fold_rmse)

mean_cv_rmse = np.mean(cv_scores)
print(f"Mean RMSE from 3-Fold CV: {mean_cv_rmse:.4f}")

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=200
)

y_pred = xgb_model.predict(X_train)
rmse = mean_squared_error(y_train, y_pred, squared=False)
print(f"Final RMSE on Training Data: {rmse:.4f}")

param_grid = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.015, 0.02],
    "n_estimators": [500, 1000],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9]
}

random_search = RandomizedSearchCV(
    xgb_model, param_distributions=param_grid, n_iter=10,
    cv=3, scoring="neg_root_mean_squared_error", verbose=1, n_jobs=-1
)

random_search.fit(X_train, y_train)
print(f"Best Parameters: {random_search.best_params_}")



pred= random_search.predict(te)
print("Predictions on X_te:", pred)


final=pd.DataFrame({'id':tte['id'],'Price':pred})
final.to_csv('submission.csv',index=False)

