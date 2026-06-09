import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split
import lightgbm as lgb

from sklearn.metrics import roc_curve, auc,accuracy_score,roc_auc_score



data=pd.read_csv("/kaggle/input/playground-series-s4e10/train.csv")
data.head()


test=pd.read_csv("/kaggle/input/playground-series-s4e10/test.csv")
test


data.isnull().sum()


sns.countplot(x='loan_status', data=data)
plt.show()


data.info()


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


number_data=data.select_dtypes(["int64","float64"])


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
outliers=detedct_outliers(number_data)
outliers


import matplotlib.pyplot as plt
import seaborn as sns

# Boxplot to visualize outliers
sns.boxplot(x=data["person_age"])
plt.title("Age Distribution with Outliers")
plt.show()



def clamp_outliers(data):
    for col in data.columns:
        q1 = np.percentile(data[col], 25, method='midpoint')
        q3 = np.percentile(data[col], 75, method='midpoint')
        IQR = q3 - q1
        lower_bound = q1 - 1.5 * IQR
        higher_bound = q3 + 1.5 * IQR
        data[col] = np.clip(data[col], lower_bound, higher_bound)
    return data

columns = number_data[["person_age", "loan_status"]] 
number = number_data.drop(["person_age", "loan_status"], axis=1) 

data_outliers = clamp_outliers(number) 


data_outliers


number_data=pd.concat([data_outliers,columns],axis=1)


number_data


number_test=test.select_dtypes(['int64','float64'])
def clamp_outliers(data):
    for col in data.columns:
        q1 = np.percentile(data[col], 25, method='midpoint')
        q3 = np.percentile(data[col], 75, method='midpoint')
        IQR = q3 - q1
        lower_bound = q1 - 1.5 * IQR
        higher_bound = q3 + 1.5 * IQR
        data[col] = np.clip(data[col], lower_bound, higher_bound)
    return data

columns_test = number_test["person_age"] 
number_test = number_test.drop(["person_age"], axis=1) 

test_outliers = clamp_outliers(number_test) 


number_test=pd.concat([test_outliers,columns_test],axis=1)


cat_data=data.select_dtypes(["object"])
cat_test=test.select_dtypes(["object"])


number_data=number_data.drop(["loan_status"],axis=1)
scaler=StandardScaler()

scaling_data = pd.DataFrame(scaler.fit_transform(number_data), columns=number_data.columns)
scaling_test = pd.DataFrame(scaler.fit_transform(number_test), columns=number_test.columns)




scaling_data=pd.concat([scaling_data,data["loan_status"]],axis=1)
scaling_data


x=cat_data.copy()
y=scaling_data["loan_status"]
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=45)



X_train


cat_data


from sklearn.utils.class_weight import compute_class_weight
classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = dict(zip(classes, weights))
cat_data=x.columns.tolist()
cat_model=CatBoostClassifier(class_weights=class_weights)
cat_model.fit(X_train,y_train,cat_features=cat_data)


y_cat = cat_model.predict(X_test)
y_cat


y_cat_test = cat_model.predict(cat_test)
y_cat_test


cat_pred_proba = cat_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, cat_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


x=scaling_data.drop(["loan_status","id"],axis=1)
y=scaling_data["loan_status"]
X_train_number,X_test_number,y_train_number_number,y_test_number=train_test_split(x,y,test_size=0.2,random_state=45)




model=lgb.LGBMClassifier(class_weight="balanced")
model.fit(X_train_number,y_train_number_number)
y_xgb=model.predict(X_test_number)


scaling_test = scaling_test.drop(columns=["id"])  

scaling_test = scaling_test[X_train_number.columns]  

y_xgb_test = model.predict(scaling_test)

print(y_xgb_test)


xg_pred_proba = model.predict_proba(X_test_number)[:, 1]
fpr, tpr, _ = roc_curve(y_test_number, xg_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


feature_names = X_train_number.columns  # Feature names from the dataset

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
accuracy = accuracy_score(y_test, combined_preds)
print(f"Accuracy of combined model: {accuracy:.4f}")


roc_auc = roc_auc_score(y_test, combined_preds_prob)
roc_auc



combined_preds_prob_test = (y_cat_test + y_xgb_test) / 2
combined_preds = (combined_preds_prob > 0.5).astype(int)

accuracy = accuracy_score(y_test, combined_preds)
print(f"Accuracy of combined model: {accuracy:.4f}")


combined_preds


submission_df = pd.DataFrame({'id': test['id'][:len(combined_preds)], 
                             'loan_status': combined_preds})
submission_df.to_csv('submission.csv', index=False)
sub = pd.read_csv('submission.csv')
sub.head()

