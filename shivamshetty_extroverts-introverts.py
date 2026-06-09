# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")


df.head()


df.shape


df.info()


df.describe()


df.isnull().mean()


df.duplicated().sum()


def plot_hist(df):
    numerical_cols = df.select_dtypes(include='number').columns

    for col in numerical_cols:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], kde=True, bins=30, stat='density', edgecolor='black')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Density')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
plot_hist(df)


for i in df.select_dtypes('object').columns:
    counts = df[i].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140, shadow=True)
    plt.title(f'Distribution of {i}')
    plt.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle.
    plt.tight_layout()
    plt.show()


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5, weights='uniform')


from sklearn.preprocessing import OneHotEncoder, LabelEncoder


label_encoder = LabelEncoder()


X = df.drop(columns = ["id", "Personality"])


y = df['Personality']


y[1]


y = label_encoder.fit_transform(y)


y[1]


categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(exclude=['object']).columns.tolist()


preprocessor = ColumnTransformer(
    transformers = [
        ('num', imputer, numerical_features),
        ('cat', OneHotEncoder(sparse_output = False, handle_unknown='ignore'), categorical_features)
    ],
    remainder = 'passthrough'
)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)


from sklearn.ensemble import RandomForestClassifier


rfc = RandomForestClassifier(
    n_estimators=200,
    class_weight = 'balanced',
    max_depth = 10,
    max_features = 'sqrt',
    min_samples_leaf = 2,
    min_samples_split = 5
)
# {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 200}


model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', rfc)
])


from xgboost import XGBClassifier


scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])


xgb_model = Pipeline([
    ('preprocessor ', preprocessor),
    ('classifier',  XGBClassifier(
    colsample_bytree=0.8,
    learning_rate=0.1,
    max_depth=3,
    n_estimators=100,
    subsample=1.0,
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
))
])


xgb_model.fit(X_train, y_train)


# from sklearn.model_selection import GridSearchCV

# param_grid = {
#     'n_estimators': [100, 200],
#     'max_depth': [3, 6, 10],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'subsample': [0.8, 1.0],
#     'colsample_bytree': [0.8, 1.0]
# }
# X_train_1 = preprocessor.transform(X_train)

# grid = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
#                     param_grid, scoring='f1_weighted', cv=5)
# grid.fit(X_train_1, y_train)
# print("Best Params:", grid.best_params_)



from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))
print("Classification Report")
print(classification_report(y_test, y_pred))



model_pipeline.fit(X_train, y_train)


from joblib import dump
dump(model_pipeline, "/kaggle/working/introvert-extrovert-rfc.joblib")


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = model_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))
print("Classification Report")
print(classification_report(y_test, y_pred))



# from sklearn.model_selection import GridSearchCV

# param_grid = {
#     'n_estimators': [100, 200, 500],
#     'max_depth': [10, 20, None],
#     'min_samples_split': [2, 5],
#     'min_samples_leaf': [1, 2],
#     'max_features': ['sqrt', 'log2']
# }
# X_train_1 = preprocessor.transform(X_train)

# grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')

# grid_search.fit(X_train_1, y_train)
# print(grid_search.best_params_)



df_sub = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")


df_sub.head()


test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


test_df.head()


X_test_sub = test_df.drop(columns=['id'])
y_pred_sub = model_pipeline.predict(X_test_sub)



y_pred_str = np.where(y_pred_sub == 0, 'Extrovert', 'Introvert')



submission_df = pd.DataFrame({
    'id': test_df['id'],
    'Personality': y_pred_str
})


submission_df.head()


submission_df.to_csv('/kaggle/working/submission.csv', index = False)


print(os.listdir('/kaggle/working'))




