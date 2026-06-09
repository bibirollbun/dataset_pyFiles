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


from sklearn.metrics import accuracy_score , r2_score, mean_absolute_error ,f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split , cross_val_score , GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder ,OrdinalEncoder ,OneHotEncoder , StandardScaler
from sklearn.impute import SimpleImputer
from scipy.stats import skew
import seaborn as sns
import matplotlib.pyplot as plt


#load data
df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
print(df.shape)
df.info()


#check for null or unique values
print(df.columns)
print(df.nunique())



print("NULL values")
print(df.isnull().sum())
sns.heatmap(df.isnull(),cbar=False,cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()


cat_features = df.select_dtypes(include=['object']).drop(['Personality'],axis=1).columns.tolist()
num_features = df.select_dtypes(exclude=['object']).drop(['id'],axis=1).columns.tolist()
X = df[num_features + cat_features]
y = df.Personality

X_train,X_valid , y_train, y_valid = train_test_split(X,y , test_size=0.2 , random_state=1)



num_imputer = SimpleImputer(strategy='most_frequent')

#Imputing numerical values
X_train[num_features] = num_imputer.fit_transform(X_train[num_features])
X_valid[num_features] = num_imputer.transform(X_valid[num_features])

sns.heatmap(X_train.isnull(),cbar=False,cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()


def Feature(df):
    df['Tired_level'] = df['Drained_after_socializing'].fillna('Unknown')
    df["Social_Fear_Score"] = df["Tired_level"].map({"Yes": 1, "No": 0, "Unknown": 0.5})
    df['Social_Activity_Score'] = df['Going_outside'] + df['Post_frequency'] - df['Time_spent_Alone']
    df['Social_Engagement_Score'] = df['Social_event_attendance'] + df['Friends_circle_size'] + df['Post_frequency']
    df['Score_Ratio'] = df['Social_Engagement_Score'] / (df['Time_spent_Alone']+1)
    df.drop(columns=["Stage_fear", "Drained_after_socializing",'Tired_level'], inplace=True)
    return df


X_train ,X_valid = X_train.align(X_valid, join='left', axis=1, fill_value=0)

X_train = Feature(X_train)
X_valid = Feature(X_valid)

#LabelEncoding()
le = LabelEncoder()

y_train = le.fit_transform(y_train)
y_valid = le.transform(y_valid)


#SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

svm = SVC(kernel='linear', C=1.0, gamma='scale')  # Try linear, poly, or rbf
svm.fit(X_train_scaled, y_train)
svm_pred = svm.predict(X_valid_scaled)
svm_acc = accuracy_score(y_valid, svm_pred)
svm_f1 = f1_score(y_valid, svm_pred, average='weighted')
print(f"SVM: ACC: {svm_acc} F1: {svm_f1}")


X_full_data = df[num_features + cat_features].copy()
y_full_data = df.Personality.copy()
X_full_data[num_features] = num_imputer.transform(X_full_data[num_features])

X_full_data = Feature(X_full_data)
y_full_data= le.transform(y_full_data)


X_full_scaled = scaler.fit_transform(X_full_data)


finale_model = svm
finale_model.fit(X_full_scaled,y_full_data)


# Prepare test data
X_test = test_data[num_features + cat_features].copy()
X_test[num_features] = num_imputer.transform(X_test[num_features])
X_test = Feature(X_test)
X_test_scaled = scaler.transform(X_test)

test_preds = finale_model.predict(X_test_scaled)
test_preds = le.inverse_transform(test_preds)

# Submission
submission = pd.DataFrame({
    'id': test_data['id'],
    'Personality': test_preds
})
submission.to_csv('submission.csv', index=False)




