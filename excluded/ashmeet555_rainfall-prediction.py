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


#
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')


train_data=pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_data=pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
sample_data=pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')


print(train_data.shape)
print(test_data.shape)


train_data.head()


test_data.head()


train_data.info()


test_data.info()


test_data.describe().T


train_data.describe().T


# Creating tables regarding missing values
missing_train = pd.DataFrame({
    'Feature': train_data.columns,
    '[Train] No of missing values': train_data.isnull().sum().values,
    '[Train] % of missing values': (train_data.isnull().sum().values / len(train_data) * 100)
})

missing_test = pd.DataFrame({
    'Feature': test_data.columns,
    '[Test] No of missing values': test_data.isnull().sum().values,
    '[Test] % of missing values': (test_data.isnull().sum().values / len(test_data) * 100)
})

# missing_original = pd.DataFrame({
#     'Feature': original_data.columns,
#     '[Original] No of missing values': original_data.isnull().sum().values,
#     '[Original] % of missing values': (original_data.isnull().sum().values / len(original_data) * 100)
# })

unique = pd.DataFrame({
    'Feature': train_data.columns,
    'No of unique values from [train]': train_data.nunique().values
})

feature_types = pd.DataFrame({
    'Feature': train_data.columns,
    'DataType': train_data.dtypes.values
})

# Merging all together
merged_df = pd.merge(missing_train, missing_test, on='Feature', how='left')
# merged_df = pd.merge(merged_df, missing_original, on='Feature', how='left')
merged_df = pd.merge(merged_df, unique, on='Feature', how='left')
merged_df = pd.merge(merged_df, feature_types, on='Feature', how='left')

# Display final merged table
merged_df


print(train_data.columns)
print(test_data.columns)


numerical_variables=['id', 'day', 'pressure', 'maxtemp', 'temparature', 'mintemp',
       'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection',
       'windspeed']
target_variable='rainfall'
categorical_variables=[]
test_data['winddirection'].fillna(test_data["winddirection"].mean(), inplace=True)


# 1. numerical
custom = ['#3498db', '#e74c3c','#2ecc71']
train_data['Dataset']='Train'
test_data['Dataset']='Test'
# original['Dataset']='original'
vari=[col for col in train_data.columns if col in numerical_variables]
def varplots(vari):
    sns.set_style('whitegrid')
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    plt.subplot(1,2,1)
    sns.boxplot(data=pd.concat([train_data,test_data]),x=vari,y='Dataset',palette=custom)
    plt.xlabel(vari)
    plt.title(f"Box Plot for {vari}")
    plt.subplot(1,2,2)
    sns.histplot(data=train_data,x=vari,color=custom[0],kde=True,bins=30,label='Train')
    sns.histplot(data=test_data,x=vari,color=custom[1],kde=True,bins=30,label='Test')
    # sns.histplot(data=original.dropna(),x=vari,color=custom[2],kde=True,bins=30,label='Original')
    plt.xlabel(vari)
    plt.ylabel("Freq")
    plt.title(f"Histogram for {vari} [Train , test]")
    plt.legend()
    plt.tight_layout()
    plt.show()


for v in vari:
    varplots(v)

train_data.drop('Dataset',axis=1,inplace=True)
test_data.drop('Dataset',axis=1,inplace=True)
# original.drop('Dataset',axis=1,inplace=True)


# target variable
plt.figure(figsize=(12,8))
sns.countplot(x=train_data['rainfall'],palette='coolwarm')
plt.title('Rainfall Distribution')
plt.xlabel('rainfall')
plt.ylabel('count')
plt.show()


# kde plo for feature target realtion shoi
plt.figure(figsize=(14,10))
for i,col in enumerate(numerical_variables,1):
    plt.subplot(3,4,i)
    sns.kdeplot(train_data[col][train_data['rainfall'] == 1], color='red', label='Rainfall: 1')
    sns.kdeplot(train_data[col][train_data['rainfall'] == 0], color='blue', label='Rainfall: 0')
    plt.title(f"Distribution for {col} by rainfall")
    plt.legend()
plt.tight_layout()
plt.show()




train_data.head()


# just checking corelation
import pandas as pd

for col1 in ['dewpoint', 'humidity', 'cloud', 'windspeed', 'pressure', 'sunshine', 'temparature']:
    for col2 in ['dewpoint', 'humidity', 'cloud', 'windspeed', 'pressure', 'sunshine', 'temparature']:
        if col1 != col2:
            new_feature = train_data[col1] * train_data[col2]
            corr = new_feature.corr(train_data['rainfall'])
            if abs(corr) > 0.3:  # adjust threshold
                print(f"{col1} * {col2} â†’ corr = {corr:.3f}")



def preprocessing(data):
    data['dew_humidity']=data['dewpoint']*data['humidity'] # how much moisture
    data['dew_cloud']=data['dewpoint']*data['cloud']
    data['hum_cloud']=data['dewpoint']*data['cloud']
    data['cloud_to_humidity']=data['cloud']/data['humidity']
    data['cloud_temparature']=data['cloud']*data['temparature']
    data['temp_to_sunshine']=data['temparature']/data['sunshine']
    data['temp_range']=data['maxtemp']-data['mintemp']
    data['cloud_to_sun']=data['cloud']/data['sunshine']
    data['dew_humidity_ratio'] = data['dewpoint'] / (data['humidity'] + 1)
    data['wind_temp_interaction'] = data['windspeed'] * data['temparature']
    data['cloud_humidity_ratio'] = data['cloud'] + (data['humidity'])
    data['pressure_temp_ratio'] = data['pressure'] / (data['temparature'] + 1)
    data['cloud_wind_ratio'] = data['cloud'] / (data['windspeed'] + 1)
    data['cloud_coverage_rate'] = data['cloud'] / 100  # Normalize to 0-1 range
    data['cloud_sun_ratio']=data['cloud']/data['sunshine']
    data['dew_humidity/sun']=data['dewpoint']*data['humidity']/(data['sunshine'])
    data["dew_humidity_+"] = data["dewpoint"] * data["humidity"]


    data['humidity_sunshine_*'] = data["humidity"] * data['sunshine']

    data["cloud_humidity/pressure"] = (data["cloud"] * data["humidity"]) / data["pressure"]
    data['month'] = ((data['day'] - 1) // 30 + 1).clip(upper=12)
    data['season'] = data['month'].apply(lambda x: 1 if 3 <= x <= 5  # Spring
                                         else 2 if 6 <= x <= 8  # Summer
                                         else 3 if 9 <= x <= 11  # Autumn
                                         else 0)  # Winter
    data['season_cloud_trend'] = data['cloud'] * data['season']
    data['season_cloud_deviation'] = data['cloud'] - data.groupby('season')['cloud'].transform('mean')
    data['season_temperature'] = data['temparature'] * data['season']  # Interaction of temper
    data = data.drop(columns=["month"])
    data = data.drop(columns=["maxtemp", "winddirection","humidity","temparature","pressure","day","season"])

    return data
train_data = preprocessing(train_data)
test_data = preprocessing(test_data)



train_data.head()


X = train_data.drop(['rainfall', 'id'], axis=1)
y = train_data['rainfall']
X_test = test_data.drop(['id'], axis=1)

# Clean up invalid values
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)



# âœ… Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# ===============================
# âœ… Data Preparation
# ===============================
# Assume train_data and test_data are already loaded

# ===============================
# âœ… Model Setup
# ===============================
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Support Vector Machine": SVC(probability=True, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Neural Network": MLPClassifier(random_state=42, max_iter=200, hidden_layer_sizes=(10,)),
    "XGBoost": XGBClassifier(random_state=42, n_estimators=100, learning_rate=0.05, max_depth=6, use_label_encoder=False, eval_metric='logloss'),
    "CatBoost": CatBoostClassifier(random_state=42, iterations=100, learning_rate=0.14, depth=6, verbose=0)
}

# ===============================
# âœ… K-Fold Training and Evaluation
# ===============================
Folds = 13
skf = StratifiedKFold(n_splits=Folds, shuffle=True, random_state=42)
auc_scores = {}
roc_curves = {}

for name, model in models.items():
    print(f"\nğŸ”¹ Training {name}...")
    oof_preds = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Fit model
        if hasattr(model, 'fit'):
            if "eval_set" in model.fit.__code__.co_varnames:
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
            else:
                model.fit(X_train, y_train)

        # Predict probabilities
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

    # Compute AUC for full OOF predictions
    auc_score = roc_auc_score(y, oof_preds)
    auc_scores[name] = auc_score

    # ROC curve
    fpr, tpr, _ = roc_curve(y, oof_preds)
    roc_curves[name] = (fpr, tpr, auc_score)

    print(f"âœ… {name}: AUC = {auc_score:.4f}")

# ===============================
# âœ… Display Summary
# ===============================
print("\nğŸ“Š Final AUC Scores:")
for name, score in auc_scores.items():
    print(f"{name}: {score:.4f}")



# plot roc curves
plt.figure(figsize=(8,6))
for model_name,(fpr,tpr,auc_score) in roc_curves.items():
    plt.plot(fpr,tpr,label=f"{model_name} (AUC = {auc_score:.4f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()


# plot auc scores
plt.figure(figsize=(8,6))
ax=sns.barplot(x=list(auc_scores.keys()),y=list(auc_scores.values()))

for i, score in enumerate(auc_scores.values()):
    ax.text(i, score + 0.01, f'{score:.4f}', ha='center', va='bottom', fontsize=12)

plt.xticks(rotation=45)
plt.ylabel("AUC Score")
plt.xlabel("Models")
plt.title("Model AUC Score Comparison")
plt.ylim(0.5, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()



# finding bes model
best_model_name=max(auc_scores,key=auc_scores.get)
bests_model=models[best_model_name]
print(f"BEs toverall model : {best_model_name} bwith auc ={auc_scores[best_model_name]:.4f}")


# visualizing feature importance in
if hasattr(bests_model, 'feature_importances_'):
    feature_importance=bests_model.feature_importances_
    importance_type='Feature Importance'
else:
    feature_importance=np.abs(best_model_coef_[0])
    importance_type='Cofficient Magnitutues'
# data frame
feature_df=pd.DataFrame({
    'Feature': train_data.drop(['rainfall','id'],axis=1).columns,
    'Importance': feature_importance
})
# sort
feature_df=feature_df.sort_values(by='Importance',ascending=False)
plt.figure(figsize=(8,6))
sns.barplot(x='Importance',y='Feature',data=feature_df)
plt.title(f"{importance_type}({best_model_name}) with best auc")
plt.show()



# Select the best model based on AUC
best_model_name = max(auc_scores, key=auc_scores.get)
best_model = models[best_model_name]

# Check if the model has feature_importances_ attribute
if hasattr(best_model, 'feature_importances_'):
    feature_importance = best_model.feature_importances_
    importance_type = 'Feature Importance'
else:
    # For logistic regression, use coefficients as importance
    feature_importance = np.abs(best_model.coef_[0])
    importance_type = 'Coefficient Magnitudes'

# Create a DataFrame to combine feature names and their importance values
feature_df = pd.DataFrame({
    'Feature': train_data.drop(['rainfall', 'id'], axis=1).columns,
    'Importance': feature_importance
})

# Sort the features by importance in descending order
feature_df = feature_df.sort_values(by='Importance', ascending=False)


# âœ… FIXED version of top-N feature selection with scaled X and test

# List of top N features to try
top_feature_counts = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

# Variables to track the best AUC and corresponding top features
best_auc_top = 0
best_top_n = 0
best_oof_preds_top = None

# Get the original feature names (before scaling)
all_feature_names = train_data.drop(['rainfall', 'id'], axis=1).columns.tolist()

for top_n in top_feature_counts:
    # Select top N feature names from feature importance DataFrame
    top_features = feature_df.head(top_n)['Feature'].tolist()

    # Get integer indices of these features in the scaled arrays
    top_indices = [all_feature_names.index(col) for col in top_features if col in all_feature_names]

    # Select corresponding columns from scaled arrays
    X_top = X_scaled[:, top_indices]
    X_test_top = X_test_scaled[:, top_indices]

    # Prepare OOF predictions
    oof_preds_top = np.zeros(len(y))

    # Cross-validation loop
    for train_idx, val_idx in skf.split(X_top, y):
        X_train, X_val = X_top[train_idx], X_top[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Fit model with early stopping if supported
        if "eval_set" in best_model.fit.__code__.co_varnames:
            best_model.fit(X_train, y_train,
                           eval_set=[(X_val, y_val)],
                           early_stopping_rounds=50,
                           verbose=0)
        else:
            best_model.fit(X_train, y_train)

        # Predict probabilities
        oof_preds_top[val_idx] = best_model.predict_proba(X_val)[:, 1]

    # Calculate AUC
    auc_score_top = roc_auc_score(y, oof_preds_top)
    print(f"AUC for top {top_n} features model: {auc_score_top:.4f}")

    # Track the best result
    if auc_score_top > best_auc_top:
        best_auc_top = auc_score_top
        best_top_n = top_n
        best_oof_preds_top = oof_preds_top

# Display the best feature set
print(f"\nğŸ�† Best AUC = {best_auc_top:.4f} with top {best_top_n} features")

best_features = feature_df.head(best_top_n)
best_features



# plotting the feature importace for the best model with the highest auc
plt.figure(figsize=(10,6))
sns.barplot(x='Importance',y='Feature',data=best_features,palette='mako')
plt.title(f"{importance_type} for Top {best_top_n} Features ({best_model_name})")
plt.show()

print("=" * 50)
print(f"ğŸ�† Best Model: {best_model_name}")
print(f"ğŸ�¯ Best AUC: {best_auc_top:.4f} using Top {best_top_n} Features")
print("=" * 50)


test_preds=best_model.predict_proba(X_test_top)[:,1]
submission=pd.DataFrame({
    'id':test_data['id'],'rainfall':test_preds
})
submission.to_csv("submission.csv",index=False)
print('\nSubmission file saved')


submission

