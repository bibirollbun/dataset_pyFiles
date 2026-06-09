import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split


print(sys.version)
modules = [
    ("numpy", np),
    ("pandas", pd),
    ("seaborn", sns),
    ("xgboost", xgb),
    ("matplotlib", matplotlib),
    ("scikit-learn", sklearn),
]

for name, module in modules:
    print(f"{name}: {module.__version__}")


df_train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

# Display basic info about the training data
df_train.describe()
df_train.info()
df_train.head()


df_train.isnull().sum().sort_values(ascending=False)


print(df_train['Soil Type'].unique())


print(df_train['Crop Type'].unique())


# delete 'id' column
df_train = df_train.drop(['id'], axis=1)
df_test = df_test.drop(['id'], axis=1)

df_train.head()


df_train['Soil Type'].unique()


ordinal_encoder = OrdinalEncoder(categories = [['Clayey', 'Sandy', 'Red', 'Loamy', 'Black'],['Sugarcane', 'Millets', 'Barley', 'Paddy', 'Pulses', 'Tobacco',
       'Ground Nuts', 'Maize', 'Cotton', 'Wheat', 'Oil seeds']])


df_train[['Soil Type','Crop Type']] = ordinal_encoder.fit_transform(df_train[['Soil Type','Crop Type']])
df_test[['Soil Type','Crop Type']] = ordinal_encoder.fit_transform(df_test[['Soil Type','Crop Type']])


df_train.head()


df_train


le = LabelEncoder()


df_train['Fertilizer Name'] = le.fit_transform(df_train['Fertilizer Name'])
df_train.head()


def describe_numeric_columns(col):
    print(f"Min: {col.min()}")
    print(f"Max: {col.max()}")
    print(f"Median: {col.median()}")

    sns.histplot(col)
    plt.show()

numeric_columns = [
    "Temparature", "Humidity", "Moisture", "Soil Type", "Crop Type", "Nitrogen",
    "Phosphorous"]

for columns in numeric_columns:
    print(f"Column Name : {columns}")
    describe_numeric_columns(df_train[columns])


sns.pairplot(df_train[numeric_columns].sample(1000))  


X_train,X_val,y_train,y_val = train_test_split(df_train.drop('Fertilizer Name',axis=1), df_train['Fertilizer Name'],random_state=42, test_size=0.2)


vanilla_model = xgb.XGBClassifier() 


vanilla_model.fit(X_train, y_train)


xgb.plot_importance(vanilla_model, 
                importance_type='gain', 
                height=0.5, 
                title="Feature Importances")

plt.show()


def mapk(y_true, y_pred, k=3):
    score = 0.0
    for true, pred in zip(y_true, y_pred):
        pred_topk = list(pred[:k])  
        if true in pred_topk:
            score += 1.0 / (pred_topk.index(true) + 1)
    return score / len(y_true)


probs = vanilla_model.predict_proba(X_val)
top_3 = np.argsort(probs, axis=1)[:, -3:][:, ::-1]
mapk_score = mapk(y_val, top_3, k=3)
print(f"MAP@3: {mapk_score}")

