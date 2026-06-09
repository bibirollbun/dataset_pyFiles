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


import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')

import warnings
warnings.filterwarnings('ignore')


train_data = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)


train_df.head()


test_df.head()


train_df.info()


test_df.info()


train_df.isnull().sum()


test_df.isnull().sum()


train_df.describe()


for col in train_df.columns:
    print(col,'---->',train_df[col].nunique())


numeric_vals = ['day','pressure','maxtemp','temparature','mintemp',"dewpoint","humidity","cloud","sunshine","winddirection","windspeed"]

for col in numeric_vals:
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(train_df[col], kde=True)
    plt.subplot(1, 2, 2)
    sns.boxplot(x=train_df[col])
    plt.tight_layout()
    plt.show()


sns.countplot(y = "rainfall", data = train_df, order = train_df["rainfall"].value_counts().index)
plt.title(f'Distribution of rainfall')
plt.xlabel('Count')
plt.ylabel("rainfall")
plt.show()


import plotnine as p9 
from plotnine import *


ggplot(train_df, aes(x='id', y='pressure')) + geom_line()


ggplot(train_df, aes(x='id', y='temparature')) + geom_line()


ggplot(train_df, aes(x='id', y='dewpoint')) + geom_line()


ggplot(train_df, aes(x='id', y='humidity')) + geom_line()


ggplot(train_df, aes(x='id', y='winddirection')) + geom_line()


ggplot(train_df, aes(x='id', y='windspeed')) + geom_line()


from sklearn.utils import resample


majority = train_df[train_df.rainfall == 1]
minority = train_df[train_df.rainfall == 0]

minority_oversampled = resample(minority,
                               replace= True,
                               n_samples =len(majority),
                               random_state=42)

oversampled_data = pd.concat([majority,minority_oversampled])

oversampled_distribution = oversampled_data.rainfall.value_counts()
oversampled_distribution


numeric_df = oversampled_data.select_dtypes(include = ['number'])
corr_matrix = numeric_df.corr()


print(corr_matrix['rainfall'].sort_values(ascending = False).to_string())


plt.figure(figsize=(20,20))
sns.heatmap(corr_matrix,annot=True,cmap = 'coolwarm', fmt = ".2f")
plt.show()


High_correlate = corr_matrix[corr_matrix >= 0.75]


print(High_correlate[High_correlate<1.0].stack().to_string())


High_correlate_inverse = corr_matrix[corr_matrix <= -0.75]


print(High_correlate_inverse[High_correlate_inverse<1.0].stack().to_string())


from statsmodels.stats.outliers_influence import variance_inflation_factor


oversampled_data.insert(0, 'Intercept', 1)


vif_data = pd.DataFrame()
vif_data["Feature"] = oversampled_data.columns
vif_data["VIF"] = [variance_inflation_factor(oversampled_data.values, i) for i in range(oversampled_data.shape[1])]


print(vif_data)


oversampled_data = oversampled_data.drop(columns= ['id',"Intercept",'maxtemp','mintemp','dewpoint'],axis=1)


oversampled_data.columns


x_oversampled_data = oversampled_data.drop('rainfall',axis = 1)
y_oversampled_data = oversampled_data["rainfall"]


from sklearn.model_selection import train_test_split


x_train,x_val,y_train,y_val = train_test_split(x_oversampled_data,y_oversampled_data,test_size = 0.3,random_state = 42)


from sklearn.preprocessing import MinMaxScaler


mm = MinMaxScaler()
x_train_scaled = mm.fit_transform(x_train)
x_val_scaled = mm.transform(x_val)


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def model_acc(model):
    model.fit(x_train_scaled,y_train)
    y_pred = model.predict(x_val_scaled)
    y_pred_train = model.predict(x_train_scaled)
    train_accuracy = accuracy_score(y_train,y_pred_train)
    test_accuracy = accuracy_score(y_val, y_pred)
    print(str(model)+'-->'+str(train_accuracy))
    print(str(model)+'-->'+str(test_accuracy))
    print("Confusion Matrix:\n", confusion_matrix(y_val,y_pred))
    print("Classification Report:\n", classification_report(y_val, y_pred))


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
model_acc(lr)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
model_acc(dt)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
model_acc(rf)

import xgboost as xgb
xgb = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42)
model_acc(xgb)


from sklearn.metrics import roc_curve, auc, roc_auc_score


y_pred_prob = rf.predict_proba(x_val_scaled)[:, 1]  # Probability for class 1

# Compute ROC metrics
fpr, tpr, thresholds = roc_curve(y_val, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
plt.title('ROC Curve for Random Forest')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend(loc='lower right')
plt.show()

print(f"ROC AUC Score: {roc_auc:.2f}")


y_pred_prob = xgb.predict_proba(x_val_scaled)[:, 1]  # Probability for class 1

# Compute ROC metrics
fpr, tpr, thresholds = roc_curve(y_val, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
plt.title('ROC Curve for XGBOOST')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend(loc='lower right')
plt.show()

print(f"ROC AUC Score: {roc_auc:.2f}")


test_df['winddirection'] = test_df['winddirection'].fillna(test_df['winddirection'].median())


final_df = test_df.drop(columns= ['id','maxtemp','mintemp','dewpoint'],axis=1)


x_test_scaled = mm.transform(final_df)


y_test_pred = rf.predict(x_test_scaled)


submission_df = pd.DataFrame({
    'id': test_df['id'],
    'rainfall': y_test_pred 
})


submission_df.head(10)

