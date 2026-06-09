import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline


train_df = pd.read_csv("/kaggle/input/forest-cover-type-prediction/train.csv")
test_df  = pd.read_csv("/kaggle/input/forest-cover-type-prediction/test.csv")
train_df.head()


print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)


target = train_df["Cover_Type"]
train_df = train_df.drop(['Cover_Type'], axis=1)


target.value_counts()


train_df.isna().count()


def data_processing(df):
    df["Hillshade_Total"] = df["Hillshade_9am"] + df["Hillshade_Noon"] + df["Hillshade_3pm"]
    df["Hillshade_Mean"] = df["Hillshade_Total"] / 3.0
    df = df.drop(['Hillshade_9am'], axis=1)
    df = df.drop(['Hillshade_Noon'], axis=1)
    df = df.drop(['Hillshade_3pm'], axis=1)
    df = df.drop(['Hillshade_Total'], axis=1)
    df = df.drop(['Id'], axis=1)
    return df
    
test_df_ = test_df.copy()
train_df = data_processing(train_df)
test_df = data_processing(test_df)


import seaborn as sns

corr = train_df.corr()
cmap = sns.diverging_palette(5, 250, as_cmap=True)

def magnify():
    return [dict(selector="th",
                 props=[("font-size", "7pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '12pt')])
]

corr.style.background_gradient(cmap, axis=1)\
    .format(precision=3)\
    .set_properties(**{'max-width': '80px', 'font-size': '12pt'})\
    .set_caption("Корреляция непрерывных признаков")\
    .set_table_styles(magnify())


train_df['Soil_Type15'].value_counts()


from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
target = le.fit_transform(target)

X_train, X_valid, y_train, y_valid = train_test_split(
    train_df, target, test_size=0.2, random_state=42, stratify=target
)


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier


classifiers = {
    'XGB': {
        'model': XGBClassifier(random_state=42, eval_metric="mlogloss"),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 1.0]
        }
    }
}


from sklearn.model_selection import GridSearchCV

results = {}

for name, clf_info in classifiers.items():
    grid_search = GridSearchCV(
        estimator=clf_info['model'],
        param_grid=clf_info['params'],
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(train_df, target)

    best_model = grid_search.best_estimator_
    best_score = grid_search.best_score_
    results[name] = {
        'model': best_model,
        'best_params': grid_search.best_params_,
        'best_accuracy': best_score
    }



best_classifier_name = max(results.keys(), key=lambda k: results[k]['best_accuracy'])
best_model = results[best_classifier_name]['model']
results


best_model


y_pred_test = best_model.predict(test_df)
y_pred_test = le.inverse_transform(y_pred_test)

submission = pd.DataFrame({
    'Id': test_df_['Id'],
    'Cover_Type': y_pred_test
})

submission.to_csv('submission.csv', index=False)


submission

