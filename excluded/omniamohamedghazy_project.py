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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import factorize
from scipy import stats

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn import preprocessing


from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
df.head(20)


df.shape


df.describe(exclude="object")


df.describe(include="object")


df.info()


df.isna().sum().sum()


df.isnull().sum().to_frame('NaN').T



df.duplicated().sum()


   print("name >>nunique")
for col in df:
    print(f"{col}: {df[col].nunique()}")


cate_cols = list(df.select_dtypes("object"))
cate_cols


number_cat= (df.describe(include= "object").T)["unique"]
number_cat


num_cols = list(df.select_dtypes(np.number))
num_cols


   print("name >>nunique")
for col in num_cols:
    print(f"{col}: {df[col].nunique()}")


print(cate_cols)



for col in cate_cols:
    fig=plt.figure(figsize=(12,10))
    sns.countplot(data = df , x=col)
    plt.title(f"{col}")
    plt.show()



for col in cate_cols:
    print(9*" ==")
    print(df[col].value_counts(normalize = True))



df= df.drop(columns=["poutcome"])
df


# for col in cate_cols:
#  if col != "poutcome":
#      df= df[df[col] != 'unknown']

# df.head()
    



df.shape


num_cols


df = df.drop("id" , axis =1)



num_cols =num_cols[1:]
num_cols


for col in num_cols:
    sns.histplot(df[col])
    plt.show()


df["previous"].value_counts(normalize=True)


df = df.drop("previous" ,axis= 1)


num_cols.remove("previous")


num_cols


for col in num_cols:
    
    sns.boxplot(df[col])
    plt.title(col)
    plt.show()


num_cols


log = ['age',  'duration', 'campaign', 'pdays']
for col in log:
    df[col] = np.log(df[col])
    sns.histplot(df[col])
    plt.title(col)
    plt.show()


num_cols





def calculate_outlier_percentage(column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    outlier_percentage = (len(outliers) / len(df)) * 100

    return outlier_percentage

for i in num_cols[:6]:
  if col != "y" :   
    outlier_percentage = calculate_outlier_percentage(i)
    print(f"Column: {i},    Outlier Percentage: {outlier_percentage:.2f}%")


df


df.drop(['pdays', 'day' ], axis = 1, inplace = True)



df =df.drop("month",axis =1 )


df.info()


df.columns


num_cols = ["age" ,"balance","duration","campaign","y"]
cate_cols =[ 'job', 'marital', 'education', 'default', 'housing',
       'loan', 'contact',]


df


cols = ["balance" ,"age" ,"campaign","duration"]
for col in cols:
    sns.boxplot(df[col])
    plt.title(col)
    plt.show()


for col in cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower=lower, upper=upper)


for col in cols:
    sns.boxplot(df[col])
    plt.title(col)
    plt.show()


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder, StandardScaler



# The mapping dictionary
mask = {
    'default': {'yes': 1, 'no': 0},
    'housing': {'yes': 1, 'no': 0},
    'loan': {'yes': 1, 'no': 0},
    'contact': {'unknown': 0, 'telephone': 1, 'cellular': 2},
    'job': {'admin.': 0, 'unknown': 1, 'unemployed': 2, 'management': 3, 'housemaid':4 , 'entrepreneur': 5, 'student': 6, 'blue-collar': 7, 'self-employed': 8, 'retired': 9, 'technician': 10, 'services': 11},
    'education': {'unknown': 0, 'secondary': 1, 'primary': 2, 'tertiary': 3},
    "marital" :{"married":0 ,"single":1 ,"divorced":2}
}

# The list of categorical columns (assuming 'marital' is handled separately)
cat_cols = [ 'default', 'housing', 'loan', 'contact',  'job', 'education', "marital"]

# Assuming 'df' is your DataFrame, this is the corrected loop
for col in cat_cols:
    df[col] = df[col].map(mask[col])


X = df.iloc[:, :-1]
y = df.iloc[:, -1]
y


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.21, random_state = 50)




df


# trans = ColumnTransformer(transformers=[
#     ('ordinal', OrdinalEncoder(categories=[['unknown', 'primary', 'secondary', 'tertiary']]), [3]),
#     ('onehot', OneHotEncoder(sparse_output=False, drop='first'), [1, 2, 4, 6, 7, 8]),
#     ('scaling', StandardScaler(), num_cols)
# ], remainder='passthrough')

# pipeline = Pipeline([
#     ('transformation', trans)
# ])



# X_train_transformed = pipeline.fit_transform(X_train)

# X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=pipeline.get_feature_names_out())

# X_test_transformed = pipeline.transform(X_test)

# X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=pipeline.get_feature_names_out())

# df_train_transformed = pd.concat([X_train_transformed_df, pd.Series(y_train, name='target')], axis=1)
# df_test_transformed = pd.concat([X_test_transformed_df, pd.Series(y_test, name='target')], axis=1)


from sklearn.preprocessing import MaxAbsScaler,RobustScaler
max_abs_scaler = MaxAbsScaler()
robust_scaler = RobustScaler()
standard = StandardScaler()
for col in cols:

    X_train[col] = standard.fit_transform(X_train[[col]])
    X_test[col] = standard.transform(X_test[[col]])



X_train


X_train.isna().sum()


log_model = LogisticRegression(max_iter = 750)
log_model.fit(X_train, y_train)
y_pred_lr = log_model.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

print(accuracy_score(y_test, y_pred_lr))

cm = confusion_matrix(y_test, y_pred_lr)
ConfusionMatrixDisplay(cm).plot()
plt.show()



from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(min_samples_split=200, min_samples_leaf=75)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print(accuracy_score(y_test, y_pred_dt))

cm = confusion_matrix(y_test, y_pred_dt)
ConfusionMatrixDisplay(cm).plot()
plt.show()





%pip install imbalanced-learn


from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


log_model = LogisticRegression(max_iter = 750)
log_model.fit(X_train_resampled, y_train_resampled)
y_pred_lr = log_model.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

print(accuracy_score(y_test, y_pred_lr))

cm = confusion_matrix(y_test, y_pred_lr)
ConfusionMatrixDisplay(cm).plot()
plt.show()



from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(min_samples_split=400, min_samples_leaf=95)
dt.fit(X_train_resampled, y_train_resampled)
y_pred_dt = dt.predict(X_test)
print(accuracy_score(y_test, y_pred_dt))

cm = confusion_matrix(y_test, y_pred_dt)
ConfusionMatrixDisplay(cm).plot()
plt.show()




