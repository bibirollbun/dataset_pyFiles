import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from xgboost import XGBClassifier 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


df_train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


df_train.head()


df_test.head()


df_test.info()


df_train.info()


df_train["Separator"] = "Training"
df_test["Separator"] = "Testing"
df_train.head(2)


df_test.head(2)


combined_data = pd.concat([df_train, df_test], axis=0)
combined_data.info()


combined_data.duplicated().sum()


combined_data.isnull().sum()


combined_data["winddirection"].fillna(combined_data["winddirection"].mode()[0],inplace = True)


combined_data_x = combined_data.drop("rainfall", axis=1)
combined_data_y = combined_data["rainfall"]


combined_data_x.head()


combined_data_x.describe()


def dist(data, bins=10, color='royalblue'):
    num_cols = data.select_dtypes(include=['number']).columns 
    n_cols = 3
    n_rows = (len(num_cols) + n_cols - 1) // n_cols
    
    plt.figure(figsize=(n_cols * 5, n_rows * 4))

    for i, col in enumerate(num_cols, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.histplot(data[col], bins=bins, kde=True, color=color, edgecolor='black', alpha=0.7)
        sns.kdeplot(data[col], color='darkred', linewidth=2)
        plt.title(f'Distribution of {col}', fontsize=14, fontweight='bold', color='darkblue')
        plt.xlabel(col, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout() 
    plt.show()
dist(combined_data_x, bins=10, color='teal')


combined_data_x_num = combined_data_x.select_dtypes(include=['number'])
combined_data_x_cat = combined_data_x.select_dtypes(include=['object'])


def corr_matrix(data, cmap='viridis'):

    plt.figure(figsize=(14, 14)) 
    sns.set_style("whitegrid")
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap=cmap, 
                linewidths=0.5, linecolor='black',
                cbar=True)
    plt.title("Correlation Matrix", fontsize=16, fontweight='bold', color='darkblue')
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12, rotation=0)
    plt.show()
corr_matrix(combined_data_x_num, cmap='viridis')


scaled_data = MinMaxScaler()
scaled_data = scaled_data.fit_transform(combined_data_x_num)
scaled_data = pd.DataFrame(scaled_data, columns =  combined_data_x_num.columns)


combined_data_x_num["day"].value_counts().sum().sum()


#After scaling we are not worry about outliers
scaled_data.describe()


scaled_data = scaled_data.drop(['id','day'],axis = 1)


scaled_data.head(2)


scaled_data = scaled_data.reset_index(drop=True)
combined_data_x_cat = combined_data_x_cat.reset_index(drop=True)
combined_data = pd.concat([scaled_data,combined_data_x_cat],axis = 1)
combined_data.head(2)


train_data = combined_data[combined_data["Separator"] == "Training"].drop(columns=["Separator"])
test_data = combined_data[combined_data["Separator"] == "Testing"].drop(columns=["Separator"])
train_data.head(2)


test_data.head(2)


train_data.head()


train_data.head(2)


combined_data_y.head(2)


# above we merginig test and train data that's why here in train data in respons varaible have missing values
# so we dropping missing values
print(combined_data_y.isnull().sum()) 
combined_data_y = combined_data_y.dropna()


combined_data_y.value_counts()
# we have immbalenced data set


smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(train_data, combined_data_y)
y_train_smote.value_counts()


x_train, x_test, y_train, y_test =train_test_split(X_train_smote,y_train_smote, test_size = 0.2,random_state = 42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


logit = LogisticRegression()
logit.fit(x_train,y_train)
y_train_pred =logit.predict(x_train)
y_test_pred = logit.predict(x_test)


print(classification_report(y_train,y_train_pred))
print("-"*60)
print(classification_report(y_test,y_test_pred))


# checking accuracy for training and testing data
print(accuracy_score(y_train,y_train_pred))
print("-"*10)
print(accuracy_score(y_test,y_test_pred))


rf = RandomForestClassifier(n_estimators = 100)
rf.fit(x_train,y_train)


y_pred_train_rf = rf.predict(x_train)
y_pred_test_rf = rf.predict(x_test)


# checking accuracy for training and testing data
print(accuracy_score(y_train,y_pred_train_rf))
print("-"*10)
print(accuracy_score(y_test,y_pred_test_rf))


from sklearn.model_selection import RandomizedSearchCV
import pandas as pd

parameter = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(rf, param_distributions=parameter, 
                                   n_iter=20, cv=5, scoring='accuracy', 
                                   n_jobs=-1, return_train_score=True, random_state=42)

random_search.fit(x_train, y_train)

results_df = pd.DataFrame(random_search.cv_results_)
results_df = results_df[['param_n_estimators', 'param_max_depth', 
                         'param_min_samples_split', 'param_min_samples_leaf', 
                         'mean_train_score', 'mean_test_score']]

best_results = results_df.sort_values(by="mean_train_score", ascending=False)
best_params = random_search.best_params_
best_acc = random_search.best_score_

print(f"Best Hyperparameters: {best_params}")
print(f"Best Accuracy: {best_acc:.2f}")
print(best_results)


rf = RandomForestClassifier(n_estimators= 100, min_samples_split= 2, min_samples_leaf= 5, max_depth=10)
rf.fit(x_train,y_train)
y_pred_train_rf = rf.predict(x_train)
y_pred_test_rf = rf.predict(x_test)


# checking accuracy for training and testing data
print(accuracy_score(y_train,y_pred_train_rf))
print("-"*10)
print(accuracy_score(y_test,y_pred_test_rf))


confusion_matrix(y_train,y_pred_train_rf)


# confusion Matrix
cm_train = confusion_matrix(y_train, y_pred_train_rf)
cm_test = confusion_matrix(y_test,y_pred_test_rf)
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sns.heatmap(cm_train, annot=True, fmt="d", cmap="viridis")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Training Data")

plt.subplot(1, 2, 2)
sns.heatmap(cm_test, annot=True, fmt="d", cmap="viridis")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Testing Data")
plt.show()



print(classification_report(y_train, y_pred_train_rf))
print("-"*60)
print(classification_report(y_test,y_pred_test_rf))


rf = RandomForestClassifier(n_estimators= 100, min_samples_split= 2, min_samples_leaf= 5, max_depth=10)
rf.fit(X_train_smote, y_train_smote) 
y_pred_rf = rf.predict(test_data)


test_data_pred = pd.DataFrame(y_pred_rf, columns=["rainfall"])
submission = pd.concat([df_test[["id"]], test_data_pred], axis=1)
submission.to_csv("sample_submission.csv", index=False)

