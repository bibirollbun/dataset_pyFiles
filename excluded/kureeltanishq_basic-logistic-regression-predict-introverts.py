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


# pip install pandas-profiling


from ydata_profiling import ProfileReport


import kagglehub

path = kagglehub.dataset_download("rakeshkapilavai/extrovert-vs-introvert-behavior-data")

print("Path to dataset files:", path)


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


og = pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data/personality_dataset.csv')


profile_train = ProfileReport(train)
profile_train


profile_test = ProfileReport(test)
profile_test


profile_og = ProfileReport(og)
profile_og


from sklearn.model_selection import train_test_split
X = train.drop(columns = ['id','Personality'])
y = train['Personality']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state=42, stratify = y)


y.value_counts(normalize = True)


categoric = X.select_dtypes(include = 'object').columns
categoric


numeric = X.select_dtypes(exclude = 'object').columns
numeric


import seaborn as sns
import matplotlib.pyplot as plt

for col in numeric:
    sns.kdeplot(data=train, x=col, hue='Personality', common_norm=False)
    plt.title(col)
    plt.show()


for col in categoric:
    sns.countplot(data=train, x=col, hue='Personality')
    plt.title(col)
    plt.show()


numeric


categoric


# import pandas as pd
# import numpy as np
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import OneHotEncoder

# numeric_to_bin = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
#                   'Friends_circle_size', 'Post_frequency']

# categorical_cols = ['Stage_fear', 'Drained_after_socializing']

# num_imputer = SimpleImputer(strategy='median')
# X_train[numeric_to_bin] = num_imputer.fit_transform(X_train[numeric_to_bin])
# X_val[numeric_to_bin] = num_imputer.transform(X_val[numeric_to_bin])

# def apply_binning(df):
#     df['Time_spent_Alone_bin'] = pd.cut(df['Time_spent_Alone'],
#                                         bins=[-np.inf, 3.5, 4.4, np.inf],
#                                         labels=['low', 'medium', 'high'])

#     df['Social_event_attendance_bin'] = pd.cut(df['Social_event_attendance'],
#                                                bins=[-np.inf, 3, 4, 9, np.inf],
#                                                labels=['low', 'mid', 'high', 'too_high'])

#     df['Going_outside_bin'] = pd.cut(df['Going_outside'],
#                                      bins=[-np.inf, 2.5, 3.5, np.inf],
#                                      labels=['low', 'mid', 'high'])

#     df['Friends_circle_size_bin'] = pd.cut(df['Friends_circle_size'],
#                                            bins=[-np.inf, 6, np.inf],
#                                            labels=['small', 'big'])

#     df['Post_frequency_bin'] = pd.cut(df['Post_frequency'],
#                                       bins=[-np.inf, 3, np.inf],
#                                       labels=['small', 'big'])

#     return df

# X_train = apply_binning(X_train)
# X_val = apply_binning(X_val)

# cat_imputer = SimpleImputer(strategy='most_frequent')

# X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
# X_val[categorical_cols] = cat_imputer.transform(X_val[categorical_cols])

# bin_columns = ['Time_spent_Alone_bin', 'Social_event_attendance_bin', 
#                'Going_outside_bin', 'Friends_circle_size_bin', 'Post_frequency_bin']

# all_categoricals = bin_columns + categorical_cols

# ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

# X_train_encoded = pd.DataFrame(ohe.fit_transform(X_train[all_categoricals]),
#                                columns=ohe.get_feature_names_out(all_categoricals),
#                                index=X_train.index)

# X_val_encoded = pd.DataFrame(ohe.transform(X_val[all_categoricals]),
#                              columns=ohe.get_feature_names_out(all_categoricals),
#                              index=X_val.index)

# X_train = pd.concat([X_train.drop(columns=numeric_to_bin + all_categoricals), X_train_encoded], axis=1)
# X_val = pd.concat([X_val.drop(columns=numeric_to_bin + all_categoricals), X_val_encoded], axis=1)



from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import pandas as pd

numeric = ['Time_spent_Alone', 'Social_event_attendance']
categoric = ['Stage_fear', 'Drained_after_socializing']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric),
    ('cat', categorical_transformer, categoric)
])

preprocessor.fit(X_train)

X_train_imputed = preprocessor.transform(X_train)
X_val_imputed = preprocessor.transform(X_val)

imputed_columns = numeric + categoric

X_train_df = pd.DataFrame(X_train_imputed, columns=imputed_columns, index=X_train.index)
X_val_df = pd.DataFrame(X_val_imputed, columns=imputed_columns, index=X_val.index)



from catboost import CatBoostClassifier

model = CatBoostClassifier(
    class_weights=[2.8, 1],
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function='Logloss',
    eval_metric='Accuracy',
    cat_features=['Stage_fear', 'Drained_after_socializing'],
    verbose=100,
    random_state=42
)

model.fit(X_train_df, y_train,
          eval_set=(X_val_df, y_val),
          early_stopping_rounds=50)



y_val_pred = model.predict(X_val_df)


# import matplotlib.pyplot as plt

# feature_importances = model.get_feature_importance()
# features = X_train_df.columns

# plt.figure(figsize=(10, 6))
# plt.barh(features, feature_importances)
# plt.xlabel("Importance")
# plt.title("CatBoost Feature Importance")
# plt.show()



from sklearn.metrics import accuracy_score, roc_auc_score
print("Accuracy: ", accuracy_score(y_val, y_val_pred))


from sklearn.metrics import precision_recall_curve
y_proba = model.predict_proba(X_val_df)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba, pos_label='Extrovert')


import matplotlib.pyplot as plt

plt.plot(thresholds, precisions[:-1], label='Precision')
plt.plot(thresholds, recalls[:-1], label='Recall')
plt.xlabel('Threshold')
plt.legend()
plt.grid()
plt.title('Precision-Recall vs Threshold')
plt.show()


from sklearn.metrics import classification_report, f1_score

y_val_binary = (y_val == 'Extrovert').astype(int)

y_proba = model.predict_proba(X_val_df)[:, 1]

thresholds_to_test = [0.455, 0.5]

from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_val_binary, y_proba)
f1_scores = [f1_score(y_val_binary, (y_proba >= t).astype(int)) for t in thresholds]
best_f1_threshold = thresholds[f1_scores.index(max(f1_scores))]

thresholds_to_test.append(best_f1_threshold)

for threshold in thresholds_to_test:
    y_pred_thresh = (y_proba >= threshold).astype(int)
    print(f"\n Threshold = {threshold:.4f}")
    print(classification_report(y_val_binary, y_pred_thresh, target_names=['Introvert', 'Extrovert']))



# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, roc_auc_score


# clf = LogisticRegression(class_weight = 'balanced', max_iter = 1000, random_state = 42)
# clf.fit(X_train_transformed, y_train)


# y_val_pred = clf.predict(X_val_transformed)
# y_val_proba = clf.predict_proba(X_val_transformed)[:,1]


# print("Accuracy: ", accuracy_score(y_val, y_val_pred))
# print("ROC - AUC : ",roc_auc_score(y_val, y_val_proba))


# X_test = test.drop(columns=['id']) 
# X_test_transformed = preprocessor.transform(X_test)


# y_test_pred = clf.predict(X_test_transformed)


# submission = pd.DataFrame({
#     'id': test['id'],
#     'target': y_test_pred
# })



# submission.to_csv('/kaggle/working/submission.csv', index=False)


# pip install catboost


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_transformed, y_train)


y_val_pred = rf.predict(X_val_transformed)
y_val_proba = rf.predict_proba(X_val_transformed)[:, 1]

print("Random Forest Validation Results")
print("Accuracy:", accuracy_score(y_val, y_val_pred))
print("ROC-AUC: ", roc_auc_score(y_val, y_val_proba))


X_full = pd.concat([X_train, X_val])
y_full = pd.concat([y_train, y_val])

preprocessor.fit(X_full)
X_full_transformed = preprocessor.transform(X_full)
X_test_transformed = preprocessor.transform(X_test)
rf.fit(X_full_transformed, y_full)



y_test_proba = rf.predict(X_test_transformed)

submission = pd.DataFrame({
    'id': test['id'],
    'Personality': y_test_proba
})
submission.to_csv('/kaggle/working/submission_rf.csv', index=False)

print("Submission file saved as submission_rf.csv")


submission




