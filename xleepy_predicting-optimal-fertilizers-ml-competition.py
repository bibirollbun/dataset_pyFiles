import numpy as np # Linear algebra
import pandas as pd # Data processing
import matplotlib.pyplot as plt # Visualization
from sklearn import linear_model # Linear model for logistic regression
from sklearn import svm # SVM model
from sklearn.tree import DecisionTreeClassifier # Decision tree for classification
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier # RF and Bagging
from xgboost import XGBClassifier # XGB classification
import lightgbm as lgb # LightGBM
from sklearn.naive_bayes import GaussianNB, MultinomialNB # Naive Bayes
from sklearn.preprocessing import PolynomialFeatures # Polynomial Features
from sklearn.preprocessing import OneHotEncoder # One-hot encoding
from sklearn import preprocessing # Label encoding
import seaborn as sns # Boxplotting and heatmaps
from sklearn.preprocessing import MinMaxScaler # Normalizer
from sklearn.preprocessing import StandardScaler # Standardizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV # Train/test split and hyperparameter tuning
from sklearn import metrics # For confusion matrix and classification report
from scipy.stats import loguniform # Log uniform


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
train


train.info()


x = train.drop(['id', 'Fertilizer Name'], axis=1)
test = test.drop(['id'], axis=1)
x


y = pd.DataFrame(train['Fertilizer Name'])
y


# balanced but may require augmentation for DAP/UREA
y.value_counts()


for i in range(8):
    print(x.columns[i])
    print(x.iloc[:, i].unique())


x = pd.get_dummies(x, columns=['Soil Type', 'Crop Type'])
test = pd.get_dummies(test, columns=['Soil Type', 'Crop Type'])
x


figs, axes = plt.subplots(6, 1, figsize=(5, 15))

for i in range(6):
    sns.boxplot(x[x.columns[i]], ax=axes[i])
    axes[i].set_title(x.columns[i])

plt.show()


figs, axes = plt.subplots(6, 1, figsize=(5, 15))

for i in range(6):
    axes[i].hist(x[x.columns[i]])
    axes[i].set_title(x.columns[i])

plt.show()


cols = x.columns
x.iloc[:, 0:5] = MinMaxScaler().fit_transform(x.iloc[:, 0:5])
x = pd.DataFrame(x, columns=cols)

test.iloc[:, 0:5] = MinMaxScaler().fit_transform(test.iloc[:, 0:5])
test = pd.DataFrame(test, columns=cols)

x


from sklearn.preprocessing import LabelEncoder
y = LabelEncoder().fit_transform(y)
y = pd.DataFrame(y)
y


test


# Correlation Matrix
df = pd.concat([x,y], axis=1)
plt.figure(figsize=(20,20))
sns.heatmap(df.corr(),annot=True,cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


columns_to_drop = [
    "Temparature", "Moisture", "Potassium", "Soil Type_Black", "Soil Type_Red",
    "Crop Type_Barley", "Crop Type_Cotton", "Crop Type_Ground Nuts",
    "Crop Type_Maize", "Crop Type_Millets", "Crop Type_Oil seeds",
    "Crop Type_Paddy", "Crop Type_Wheat"
]

x = x.drop(columns=columns_to_drop)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# XGB classifier

model = XGBClassifier(random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
print(confusion_matrix)
report = metrics.classification_report(y_test, y_pred)
print(report)

