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
%matplotlib inline
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier

# Libtune to tune model, get different metric scores
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


train = pd.read_csv('/kaggle/input/exploring-predictive-health-factors/train.csv')
train.head()


test = pd.read_csv('/kaggle/input/exploring-predictive-health-factors/test.csv')
test.head()


train.info()


train.shape


train.describe(include = 'all').T


train.duplicated().sum()


df = train.copy()


print(df['Age'].value_counts())
print(df['PCOS'].value_counts())
print(df['Hormonal_Imbalance'].value_counts())
print(df['Hyperandrogenism'].value_counts())
print(df['Hirsutism'].value_counts())
print(df['Conception_Difficulty'].value_counts())
print(df['Insulin_Resistance'].value_counts())
print(df['Exercise_Frequency'].value_counts())
print(df['Exercise_Type'].value_counts())
print(df['Exercise_Duration'].value_counts())
print(df['Sleep_Hours'].value_counts())
print(df['Exercise_Benefit'].value_counts())


df.isnull().sum()


def countplots(data, feature,perc):
  ax = sns.countplot(data = df, x = feature)
  plt.xticks(rotation = 90)
  for p in ax.patches:
        if perc == True:
            label = "{:.1f}%".format(
                100 * p.get_height() / len(data[feature])
            )  # percentage of each class of the category
        else:
            label = p.get_height()  # count of each level of the category

        x = p.get_x() + p.get_width() / 2  # width of the plot
        y = p.get_height()  # height of the plot

        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="center",
            size=12,
            xytext=(0, 5),
            textcoords="offset points",
        )  # annotate the percentage
  plt.show()


df.info()


countplots(df,'Age', perc = True)


countplots(df,'PCOS', perc = True)


countplots(df,'Hormonal_Imbalance', perc = True)


countplots(df,'Hyperandrogenism', perc = True)


countplots(df,'Hirsutism', perc = True)


countplots(df,'Conception_Difficulty', perc = True)


countplots(df,'Insulin_Resistance', perc = True)


countplots(df,'Exercise_Frequency', perc = True)


countplots(df,'Exercise_Type', perc = True)


countplots(df,'Exercise_Duration', perc = True)


countplots(df,'Sleep_Hours', perc = True)


countplots(df,'Exercise_Benefit', perc = True)


sns.boxplot(df, x = 'Weight_kg');


def stacked_barplot(data, predictor, target):
    """
    Print the category counts and plot a stacked bar chart

    data: dataframe
    predictor: independent variable
    target: target variable
    """
    count = data[predictor].nunique()
    sorter = data[target].value_counts().index[-1]
    tab1 = pd.crosstab(data[predictor], data[target], margins=True).sort_values(
        by=sorter, ascending=False
    )
    print(tab1)
    print("-" * 120)
    tab = pd.crosstab(data[predictor], data[target], normalize="index").sort_values(
        by=sorter, ascending=False
    )
    tab.plot(kind="bar", stacked=True, figsize=(count + 5, 6))
    plt.legend(
        loc="lower left", frameon=False,
    )
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()


stacked_barplot(df,'Age' ,'PCOS')


stacked_barplot(df,'Hormonal_Imbalance' ,'PCOS')


stacked_barplot(df,'Hyperandrogenism' ,'PCOS')


stacked_barplot(df,'Hirsutism' ,'PCOS')


stacked_barplot(df,'Conception_Difficulty' ,'PCOS')


stacked_barplot(df,'Insulin_Resistance' ,'PCOS')


stacked_barplot(df,'Exercise_Frequency' ,'PCOS')


stacked_barplot(df,'Exercise_Type' ,'PCOS')


stacked_barplot(df,'Exercise_Duration' ,'PCOS')


stacked_barplot(df,'Sleep_Hours' ,'PCOS')


stacked_barplot(df,'Exercise_Benefit' ,'PCOS')


sns.boxplot(df, x = 'Weight_kg', y = "PCOS");


train_id = df['ID']
test_id = test['ID']


X = df.drop(['PCOS','ID'], axis=1)
y = df['PCOS']
test_data = test.drop(columns=["ID"],axis=1)


test_data


X.info()


y = y.map({'Yes': 1, 'No': 0})


import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

#X.columns = X.columns.str.strip().str.lower()

num_features = X.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

print("Numerical Features:", num_features)
print("Categorical Features:", cat_features)

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  
    ('scaler', StandardScaler())                
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  
    ('onehot', OneHotEncoder(handle_unknown='ignore',sparse=False))     
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, num_features),
        ('cat', categorical_transformer, cat_features)
    ]
)

X_train = preprocessor.fit_transform(X)
X_test = preprocessor.transform(test_data)


from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y)


from xgboost import XGBClassifier
xgb = XGBClassifier(random_state=42)
xgb.fit(X_train_balanced,y_train_balanced)


predictions =  xgb.predict_proba(X_test)[:, 1]
output = pd.DataFrame({'ID': test.ID, 'PCOS': predictions})
output.to_csv('submission1.csv', index=False)


rf_estimator = RandomForestClassifier(random_state=1)
rf_estimator.fit(X_train_balanced,y_train_balanced)


predictions =  rf_estimator.predict_proba(X_test)[:, 1]
output = pd.DataFrame({'ID': test.ID, 'PCOS': predictions})
output.to_csv('submission2.csv', index=False)


gb_classifier = GradientBoostingClassifier(random_state=1)
gb_classifier.fit(X_train_balanced,y_train_balanced)


predictions =  gb_classifier.predict_proba(X_test)[:, 1]
output = pd.DataFrame({'ID': test.ID, 'PCOS': predictions})
output.to_csv('submission3.csv', index=False)


from catboost import CatBoostClassifier
cat_boost = CatBoostClassifier(random_state=1, verbose=200)
cat_boost.fit(X_train_balanced,y_train_balanced)



predictions =  cat_boost.predict_proba(X_test)[:, 1]
output = pd.DataFrame({'ID': test.ID, 'PCOS': predictions})
output.to_csv('submission4.csv', index=False)


from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

model1 = GradientBoostingClassifier(random_state=1)
model2 = XGBClassifier(random_state=1)
model3 = CatBoostClassifier(random_state=1, verbose=0)

ensemble_model = VotingClassifier(estimators=[('gbc', model1), ('xgb', model2), ('catboost', model3)], voting='soft')
ensemble_model.fit(X_train_balanced, y_train_balanced)


predictions =  ensemble_model.predict_proba(X_test)[:, 1]
output = pd.DataFrame({'ID': test.ID, 'PCOS': predictions})
output.to_csv('submission5.csv', index=False)


from IPython.display import FileLink

# Path to your file
file_path = 'submission2.csv'

# Generate and display the download link
FileLink(file_path)


import joblib
joblib.dump(rf_estimator, 'pcos_model_rf.pkl')


from IPython.display import FileLink

# Path to your file
file_path = 'pcos_model_rf.pkl'

# Generate and display the download link
FileLink(file_path)




