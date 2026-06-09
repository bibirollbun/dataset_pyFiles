#!pip download MissForest
#!pip download lifelines


!pip install /kaggle/input/download-missforest/MissForest-4.2.3-py3-none-any.whl



pip install lightgbm



!pip install /kaggle/input/cibmtr-whl-files-for-installation/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/cibmtr-whl-files-for-installation/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/cibmtr-whl-files-for-installation/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/cibmtr-whl-files-for-installation/formulaic-1.1.1-py3-none-any.whl
!pip install /kaggle/input/cibmtr-whl-files-for-installation/lifelines-0.30.0-py3-none-any.whl


import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
from missforest import MissForest
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.width', None)
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from catboost import CatBoostClassifier
from lifelines.utils import concordance_index
from sklearn.metrics import roc_curve, auc,accuracy_score,roc_auc_score



test=pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")



data=pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
data.head()


data.shape


data["efs"].value_counts()


target=data["efs"].value_counts()
target.plot(kind="bar")
plt.show()


number=data.select_dtypes(["int64","float64"])
cat=data.select_dtypes(["object"])


total_rows=number.shape[0]
null_data_number=number.isna().sum().reset_index().rename(columns={0:"Null_counts","index":"Column_name"}).sort_values(by="Null_counts",ascending=False)
null_data_number["Null_percentage"]=(null_data_number["Null_counts"]/total_rows)*100
null_data_number


total_rows=cat.shape[0]
null_data_cat=cat.isna().sum().reset_index().rename(columns={0:"Null_counts","index":"Column_name"}).sort_values(by="Null_counts",ascending=False)
null_data_cat["Null_percentage"]=(null_data_cat["Null_counts"]/total_rows)*100
null_data_cat


plt.figure(figsize=(10,5))
correlation_matrix=number.corr()
sns.heatmap(correlation_matrix,annot=False,cmap='coolwarm',vmin=-1,vmax=1,linewidths=0.5)
plt.show()


data.duplicated().sum()


def find_repeated_Values(df,threshold=0.8):
    repeated_values={}
    for column in df.columns:
        value_counts=df[column].value_counts(normalize=True)
        high_freq_value=value_counts[value_counts>=threshold].index.tolist()
        if high_freq_value:
            repeated_values[column]=high_freq_value
    return repeated_values
results=find_repeated_Values(data,0.8)
print(results)


cat.info()


number.info()


ms = MissForest()

for col, percentage in zip(null_data_number['Column_name'], null_data_number['Null_percentage']):
    if percentage < 14:
        number[col].fillna(number[col].median(), inplace=True)
    else:
        ms.fit(number)
        train_imp = ms.transform(number)



train_imp.isnull().sum()



for col, percentage in zip(null_data_cat['Column_name'], null_data_cat['Null_percentage']):
    if percentage>55:
        cat.drop(col,axis=1,inplace=True)
    else:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        cat_data_imputed = pd.DataFrame(
        cat_imputer.fit_transform(cat),
        columns=cat.columns
        )

        



cat_data_imputed.isnull().sum()


def detedct_outliers(df):
    outliers=[]
    for column in df.columns:
        q1=df[column].quantile(0.25)
        q3=df[column].quantile(0.75)
        iqr=q3-q1
        lower_bound=q1-1.5*iqr
        upper_bound=q3+1.5*iqr
        if df[(df[column] < lower_bound) | (df[column] > upper_bound)].shape[0] > 0:
               outliers.append(column)   
    return outliers
outliers=detedct_outliers(train_imp)
outliers


train_imp["efs_time"].max()



emfs=train_imp["efs_time"]
train_imp.drop("efs_time",axis=1,inplace=True)
def clamp_outliers(data):
    for col in data.columns:
        Q1 = np.percentile(data[col], 25, method='midpoint')
        Q3 = np.percentile(data[col], 75, method='midpoint')
        IQR = Q3 - Q1
        Lower_Bound = Q1 - 1.5 * IQR
        Upper_Bound = Q3 + 1.5 * IQR
        data[col] = np.clip(data[col], Lower_Bound, Upper_Bound)
    return data
train_imp = clamp_outliers(train_imp)



train_imp=pd.concat([train_imp,emfs],axis=1)


train_imp.head()


from scipy.stats import skew

skew_value = skew(train_imp)
print(f"Original Skewness: {skew_value}")

plt.hist(train_imp, bins=30, alpha=0.5, label='Original Data')
plt.title('Original Distribution')
plt.show()

def log_transform(col):
    original_skew=skew(col)
    if original_skew > 0.5:
        return np.log1p(col)
    elif original_skew < -0.5:
        return np.square(col)
    else:
        return col        
     
data_log = train_imp.apply(log_transform,axis=0)
transformed_skew_values = data_log.apply(skew)
print(f"Original Skewness:\n{train_imp.apply(skew)}")
print(f"Transformed Skewness:\n{transformed_skew_values}")
plt.hist(data_log, bins=30, alpha=0.5, label='Transformed Data')
plt.title('Transformed Distribution')
plt.show()


plt.hist(transformed_skew_values, bins=30, alpha=0.5, label='Transformed Data')
plt.title('Transformed Distribution')
plt.show()


train_imp.head()


train_imp = train_imp.abs()



import statsmodels.api as sm
import matplotlib.pyplot as plt
ncols = 3
nrows=10
fig, axes = plt.subplots(nrows = nrows, ncols = ncols, figsize=(20, 40))
i=0
j=0
for col in train_imp:
    sm.qqplot(train_imp[col],fit = False, line='q', ax = axes[i, j])
    axes[i, j].set_title(col)
    if(j<ncols-1):
        j+=1
    else:
        i+=1
        j=0
plt.show()


target=train_imp["efs"]
train_imp=train_imp.drop("efs",axis=1)
standard_scaler=StandardScaler()
standard_scaler_transformer=standard_scaler.fit_transform(train_imp)
normalization_data=pd.DataFrame(standard_scaler_transformer,columns=train_imp.columns)

last_number_data=pd.concat([normalization_data,target],axis=1)
last_number_data.head()


last_number_data=last_number_data.abs()


def detedct_outliers(df):
    outliers=[]
    for column in df.columns:
        q1=df[column].quantile(0.25)
        q3=df[column].quantile(0.75)
        iqr=q3-q1
        lower_bound=q1-1.5*iqr
        upper_bound=q3+1.5*iqr
        if df[(df[column] < lower_bound) | (df[column] > upper_bound)].shape[0] > 0:
               outliers.append(column)   
    return outliers
outliers=detedct_outliers(last_number_data)
outliers


emfs=last_number_data["efs_time"]
last_number_data.drop("efs_time",axis=1,inplace=True)
def clamp_outliers(data):
    for col in data.columns:
        Q1 = np.percentile(data[col], 25, method='midpoint')
        Q3 = np.percentile(data[col], 75, method='midpoint')
        IQR = Q3 - Q1
        Lower_Bound = Q1 - 1.5 * IQR
        Upper_Bound = Q3 + 1.5 * IQR
        data[col] = np.clip(data[col], Lower_Bound, Upper_Bound)
    return data
last_number_data = clamp_outliers(last_number_data)
last_number_data=pd.concat([last_number_data,emfs],axis=1)
def detedct_outliers(df):
    outliers=[]
    for column in df.columns:
        q1=df[column].quantile(0.25)
        q3=df[column].quantile(0.75)
        iqr=q3-q1
        lower_bound=q1-1.5*iqr
        upper_bound=q3+1.5*iqr
        if df[(df[column] < lower_bound) | (df[column] > upper_bound)].shape[0] > 0:
               outliers.append(column)   
    return outliers
outliers=detedct_outliers(last_number_data)
outliers


features=last_number_data.drop(["efs","ID","efs_time"],axis=1)
y=last_number_data["efs"]
x_train,x_test,y_train,y_test=train_test_split(features,y,test_size=0.2,random_state=42)


x_train.head()


cat_data_imputed.head()


x=cat_data_imputed.copy()
x_train_cat,x_test_cat,y_train_cat,y_test_cat=train_test_split(x,y,test_size=0.2,random_state=42)


cat_data=x.columns.tolist()


model = CatBoostClassifier()
model.fit(x_train_cat, y_train_cat,cat_features=cat_data)
y_cat = model.predict(x_test_cat)
print(y_cat)


cat_pred_proba = model.predict_proba(x_test_cat)[:, 1]
fpr, tpr, _ = roc_curve(y_test_cat, cat_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


import lightgbm as lgb

model=lgb.LGBMClassifier()
model.fit(x_train,y_train)
y_xgb=model.predict(x_test)
xg_pred_proba = model.predict_proba(x_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, xg_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# Assuming xg is your trained XGBClassifier and x_train is a DataFrame
feature_names = x_train.columns  # Feature names from the dataset

# Get feature importances from XGBClassifier
feature_importances = model.feature_importances_

# Create a DataFrame for feature importances
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})

# Sort by importance
feature_importance = feature_importance.sort_values(by='Importance', ascending=False).head(20)

# Plot the top 10 most significant features
plt.figure(figsize=(20, 15))
sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importance
)
plt.title('Top 10 Most Significant Features')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()



combined_preds_prob = (cat_pred_proba + xg_pred_proba) / 2
combined_preds = (combined_preds_prob > 0.5).astype(int)

# Evaluate performance
accuracy = accuracy_score(y_test_cat, combined_preds)
print(f"Accuracy of combined model: {accuracy:.4f}")


accuracy = accuracy_score(y_test_cat, combined_preds)
roc_auc = roc_auc_score(y_test_cat, combined_preds_prob)
combined_cindex = concordance_index(y_test_cat, combined_preds_prob)


print(combined_cindex)


import pandas as pd
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
test_data = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")

# Fill missing values
data.fillna(-999, inplace=True)
test_data.fillna(-999, inplace=True)

# Target column and drop unnecessary column
target = 'efs'
data = data.drop("efs_time", axis=1)

# Separate categorical and numerical features
categorical_features = data.select_dtypes(include=['object']).columns.tolist()
numerical_features = data.select_dtypes(include=['int64', 'float64']).columns.drop([target]).tolist()

# Convert categorical features to category dtype
for col in categorical_features:
    data[col] = data[col].astype('category')
    test_data[col] = test_data[col].astype('category')

# Prepare data for CatBoost
X_cat = data.drop(columns=[target])
y_cat = data[target]

X_train_cat, X_val_cat, y_train_cat, y_val_cat = train_test_split(X_cat, y_cat, test_size=0.2, random_state=42)

cat_model = CatBoostClassifier(
    cat_features=categorical_features,
    iterations=500,
    learning_rate=0.05,
    depth=6,
    verbose=100
)
cat_model.fit(X_train_cat, y_train_cat, eval_set=(X_val_cat, y_val_cat), early_stopping_rounds=50)

cat_test_pred = cat_model.predict_proba(test_data)[:, 1]

# Prepare data for LightGBM
X_lgb = data[numerical_features]
y_lgb = data[target]

test_lgb = test_data[numerical_features]

X_train_lgb, X_val_lgb, y_train_lgb, y_val_lgb = train_test_split(X_lgb, y_lgb, test_size=0.2, random_state=42)

lgb_model = lgb.LGBMClassifier(
    boosting_type='gbdt',
    objective='binary',
    metric='auc',
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    n_estimators=500,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42
)

lgb_model.fit(X_train_lgb, y_train_lgb, eval_set=[(X_val_lgb, y_val_lgb)])

lgb_test_pred = lgb_model.predict_proba(test_lgb)[:, 1]

# Combine predictions from CatBoost and LightGBM
test_predictions = 0.5 * cat_test_pred + 0.5 * lgb_test_pred

# Save test predictions
test_results = pd.DataFrame({
    'ID': test_data['ID'],  # Ensure the test data has an 'ID' column
    'prediction': test_predictions
})

test_results.to_csv('submission.csv', index=False)



test_results

