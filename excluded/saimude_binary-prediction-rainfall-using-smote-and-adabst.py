import numpy as np
import pandas as pd 
from collections import Counter 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import StratifiedKFold, cross_val_score,train_test_split
from sklearn.metrics import auc,accuracy_score,f1_score,roc_auc_score,roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
random_state=24
sns.set_style("whitegrid")
import warnings
warnings.filterwarnings("ignore")


train_data=pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv",index_col="id")


print(train_data.isnull().sum())


train_data.describe().T


plt.figure(figsize=(6, 4))
ax = sns.countplot(x="rainfall", data=train_data, palette="pastel")

# Add count values on top of bars
for p in ax.patches:
    ax.annotate(f"{p.get_height()}", (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')




columns=list(train_data.columns)
columns.remove('rainfall')
feature_cols=columns
target_columns="rainfall"




plt.figure(figsize=(10, 8))  
for i, name in enumerate(feature_cols):
    plt.subplot(3,4, i + 1) 
    sns.boxplot(data=train_data, y=name, hue="rainfall",palette="pastel", linewidth=2.5, width=0.5, fliersize=3)
    plt.title(name, fontsize=12, fontweight='bold', color='darkblue')  
    plt.yticks(fontsize=10)  
plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 10))  
for i, name in enumerate(feature_cols):
    plt.subplot(3,4, i + 1) 
    sns.histplot(data=train_data, x=name, hue="rainfall",kde=True)
    plt.title(name, fontsize=12, fontweight='bold', color='darkblue')  
    plt.yticks(fontsize=10)  
plt.tight_layout()
plt.show()


corr=train_data[feature_cols].corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr,annot=True)


X,y=(train_data[feature_cols].values,train_data["rainfall"].values)


X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,shuffle=True,test_size=0.2)


X_train,y_train


sm=SMOTE(sampling_strategy="auto",random_state=random_state)
X_up,y_up=sm.fit_resample(X_train,y_train)
print(Counter(y_up))


plt.figure(figsize=(15, 10))  
for i, name in enumerate(feature_cols):
    plt.subplot(3,4, i + 1) 
    sns.histplot(x=X_up[:,i], hue=y_up,kde=True)
    plt.title(name, fontsize=12, fontweight='bold', color='darkblue')  
    plt.yticks(fontsize=10)  
plt.tight_layout()
plt.show()


columns


def get_season(day):
    if 1 <= day <= 79 or 356 <= day <= 365:
        return 1
    elif 80 <= day <= 171:
        return 2
    elif 172 <= day <= 265:
        return 3
    elif 266 <= day <= 355:
        return 4
    else:
        return -1

def magnus_formula(temp, humid):
    a = 17.27
    b = 237.7
    alpha = (a * temp) / (b + temp) + np.log(humid / 100.0)
    moisture = (b * alpha) / (a - alpha)
    return moisture
def calculate_wind_chill(temp, wind_speed):
    wind_chill = 13.12 + 0.6215 * temp - 11.37 * (wind_speed ** 0.16) + 0.3965 * temp * (wind_speed ** 0.16)
    return wind_chill
def sunshine_duration(sunshine,cloud):
    return sunshine*(1-(cloud/100))

def feature_engineering(data):

    ##temp related
    data["temp_range"]=data["maxtemp"]-data["mintemp"]
    data["mean_temp"]=data[["mintemp","maxtemp","temparature"]].mean(axis=1)
    data["temp_max_diff"]=data["maxtemp"]-data["temparature"]
    data["temp_min_diff"]=data["temparature"]-data["mintemp"]
    
    ##day related 
    data["month"]=data["day"]//30+1
    data["season"]=data["day"].apply(get_season)
    data['sin_day'] = np.sin(2 * np.pi * data['day'] / 365)
    data['cos_day'] = np.cos(2 * np.pi * data['day'] / 365)

    ## misc
    data["moisture"]=data.apply(lambda x:magnus_formula(x["temparature"],x["humidity"]),axis=1)
    data["wind_chill"]=data.apply(lambda x: calculate_wind_chill(x["temparature"],x["windspeed"]),axis=1)
    data["sunshine_dur"]=data.apply(lambda x:sunshine_duration(x["sunshine"],x["cloud"]),axis=1)
    data["air_density"]=data.apply(lambda x: (x["pressure"]/x["temparature"]),axis=1)
    
    return data



train_sampled_data=pd.DataFrame(X_up,columns=feature_cols)
train_sampled_data['rainfall']=y_up

test_data=pd.DataFrame(X_test,columns=feature_cols)
test_data["rainfall"]=y_test

train_sampled_data=feature_engineering(train_sampled_data)
test_data=feature_engineering(test_data)


X_final_train=train_sampled_data.drop(["maxtemp","mintemp","day","rainfall"],axis=1)
y_final_train=train_sampled_data["rainfall"]

X_final_test=test_data.drop(["maxtemp","mintemp","day","rainfall"],axis=1)
y_final_test=test_data["rainfall"]


catboost_params={'iterations':1500,
    'learning_rate': 0.16582804455334454,
    'depth': 7,
    'l2_leaf_reg': 8.487140799606516,
    'border_count': 76,
    'bagging_temperature': 0.5193248419470913,
    'random_strength': 7.667803596935708,
    'verbose': 0,
    'random_state': 42
}
models={"logistic":LogisticRegression(max_iter=1000,random_state=random_state),
       "decisontree":DecisionTreeClassifier(max_depth=10,min_samples_split=5,random_state=random_state),
        "randomforest":RandomForestClassifier(max_depth=10,min_samples_split=5,random_state=random_state),
        "adaboost":AdaBoostClassifier(n_estimators=100,learning_rate=0.1,random_state=random_state),
        "xgb":XGBClassifier(n_estimators=50, max_depth=5, learning_rate=0.1, objective='binary:logistic'),
        "catboost":CatBoostClassifier(**catboost_params)
}

y_preds={}
models_trained={}
for model_name in models.keys() :
    model=models[model_name]
    model.fit(X_final_train,y_final_train)
    models_trained[model_name]=model
    y_train_pred=model.predict_proba(X_final_train)[:,1]
    y_test_pred=model.predict_proba(X_final_test)[:,1]
    auc_train=roc_auc_score(y_final_train,y_train_pred)
    auc_test=roc_auc_score(y_final_test,y_test_pred)
    
    y_preds[model_name]=y_test_pred
    print(f"{model_name} ;\n the train auc score : {auc_train:.4f} ; the test auc score :{auc_test}")


best_model=AdaBoostClassifier(n_estimators=100,learning_rate=0.1,random_state=random_state)
best_model.fit(X_final_train,y_final_train)



feature_importance = best_model.feature_importances_
feature_names = best_model.feature_names_in_
# Create a DataFrame for better visualization
import pandas as pd
df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})

# Sort the features by importance
df = df.sort_values(by="Importance", ascending=False)

# Plot the feature importance bar plot
plt.figure(figsize=(8, 5))
sns.barplot(x="Importance", y="Feature", data=df, palette="viridis")
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance Bar Plot")
plt.show()


test_submission=pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv",index_col="id")
test_submission.head()


test_submission.isnull().sum()


test_submission["winddirection"]=test_submission["winddirection"].fillna(test_submission["winddirection"].median())


test_submission=feature_engineering(test_submission)
X_submission=test_submission.drop(["maxtemp","mintemp","day",],axis=1)


y_submission_preds=best_model.predict_proba(X_submission)[:,1]


submission_file=pd.DataFrame({"id":test_submission.index,"rainfall":y_submission_preds})


submission_file.to_csv("/kaggle/working/submission.csv",index=False)

