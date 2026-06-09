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


import kagglehub

# Download latest version
path = kagglehub.dataset_download("rakeshkapilavai/extrovert-vs-introvert-behavior-data")

print("Path to dataset files:", path)


from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import OneHotEncoder,StandardScaler,MinMaxScaler


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
original_data = pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data/personality_datasert.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


original_data.isnull().sum()


train.isnull().sum()


test.isnull().sum()


train.drop(columns = ['id'], axis =1, inplace = True)


assert (train.columns == original_data.columns).all()


num_columns = train.select_dtypes('number').columns
cat_columns = train.select_dtypes('object').columns


num_columns, cat_columns


one = OneHotEncoder()
one_test = OneHotEncoder()


train_enc = pd.DataFrame(one.fit_transform(train[cat_columns]).toarray(),columns = one.get_feature_names_out())
original_data_enc = pd.DataFrame(one.transform(original_data[cat_columns]).toarray(),columns = one.get_feature_names_out())
test_enc = pd.DataFrame(one_test.fit_transform(test[[col for col in cat_columns if col not in 'Personality']]).toarray(),columns = one_test.get_feature_names_out())


assert (train_enc.columns == original_data_enc.columns).all()


scaler = MinMaxScaler()


train_scaled = pd.DataFrame(scaler.fit_transform(train[num_columns]), columns = scaler.get_feature_names_out())
original_data_scaled = pd.DataFrame(scaler.fit_transform(original_data[num_columns]), columns = scaler.get_feature_names_out())
test_scaled = pd.DataFrame(scaler.fit_transform(test[num_columns]), columns = scaler.get_feature_names_out())


assert (train_scaled.columns == original_data_scaled.columns).all()
assert (train_scaled.columns == test_scaled.columns).all()


train_final = pd.concat([train_enc,train_scaled], axis = 1)
original_data_final = pd.concat([original_data_enc,original_data_scaled], axis = 1)
test_final = pd.concat([test_enc,test_scaled], axis = 1)


assert train_final.shape[1] == original_data_final.shape[1]


dist_matrix_train_original = cdist(train_final.values, original_data_final.values, metric = 'cosine')
dist_matrix_test_original = cdist(test_final.values, original_data_final[[col for col in original_data_final.columns if col not in ['Personality_Extrovert','Personality_Introvert']]].values, metric = 'cosine')


most_similar_indices_train = dist_matrix_train_original.argmax(axis = 1)
most_similar_indices_test = dist_matrix_test_original.argmax(axis = 1)


assert len(most_similar_indices_train) == train.shape[0]
assert len(most_similar_indices_test) == test.shape[0]


most_similar_rows_train = original_data_final.iloc[most_similar_indices_train].reset_index(drop =True)
most_similar_rows_test = original_data_final[[col for col in original_data_final.columns if col not in ['Personality_Extrovert','Personality_Introvert']]].iloc[most_similar_indices_test].reset_index(drop =True)


similar_df_cat_train = pd.DataFrame(one.inverse_transform(most_similar_rows_train[[col for col in most_similar_rows_train.columns if col not in num_columns]]), columns = ['Stage_fear','Drained_after_socializing','Personality'])
similar_df_num_train = pd.DataFrame(scaler.inverse_transform(most_similar_rows_train[num_columns]), columns =  scaler.get_feature_names_out())


similar_df_cat_test = pd.DataFrame(one_test.inverse_transform(most_similar_rows_test[[col for col in most_similar_rows_test.columns if col not in num_columns]]), columns = ['Stage_fear','Drained_after_socializing'])
similar_df_num_test = pd.DataFrame(scaler.inverse_transform(most_similar_rows_test[num_columns]), columns =  scaler.get_feature_names_out())


similar_df_final_train = pd.concat([similar_df_cat_train, similar_df_num_train], axis = 1)
similar_df_final_test = pd.concat([similar_df_cat_test, similar_df_num_test], axis = 1)


assert train.shape[1] == similar_df_final_train.shape[1]
assert (train.columns.sort_values() == similar_df_final_train.columns.sort_values()).all()

assert test.shape[1]-1 == similar_df_final_test.shape[1]
assert (test[[col for col in test.columns if col not in 'id']].columns.sort_values() == similar_df_final_test.columns.sort_values()).all()


similar_df_final_train = similar_df_final_train[[col for col in similar_df_final_train.columns.sort_values()]]
train = train[[col for col in train.columns.sort_values()]]


similar_df_final_test = similar_df_final_test[[col for col in similar_df_final_test.columns.sort_values()]]
test = test[[col for col in test.columns.sort_values()]]


for r in range(train.shape[0]):
    for col in train.columns:
        if pd.isnull(train.loc[r,col]):
            train.loc[r,col] = similar_df_final_train.loc[r,col]


for r in range(test.shape[0]):
    for col in test.columns:
        if pd.isnull(test.loc[r,col]):
            test.loc[r,col] = similar_df_final_test.loc[r,col]


train.isnull().sum()


test.isnull().sum()


from sklearn.model_selection import KFold
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, learning_curve,validation_curve
import matplotlib.pyplot as plt


num_columns


cat_columns = [c for c in cat_columns if c not in 'Personality' ]


col_transformer = ColumnTransformer([
    ('num',scaler, num_columns),
    ('cat', one, cat_columns)
])


train_trasf  = pd.DataFrame(col_transformer.fit_transform(train), columns = col_transformer.get_feature_names_out())
test_trasf = pd.DataFrame(col_transformer.transform(test),columns = col_transformer.get_feature_names_out())


train_trasf.head(1)


test_trasf.head(1)


assert train_trasf.shape[1] == test_trasf.shape[1]


assert (train_trasf.columns == test_trasf.columns).all()


X = train_trasf
y = train['Personality'].map({'Introvert':1, 'Extrovert':0})


X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state = 0 , test_size = 0.3)


kf = KFold(n_splits=5, shuffle = True)


xgb = XGBClassifier(eval_metric='logloss')


param_grid = { 'n_estimators': [100, 200,500,1000],
    'max_depth': [3, 4, 5,7],
    'learning_rate': [0.01, 0.1,0.2]}


grid_search = GridSearchCV(
    estimator = xgb,
    param_grid = param_grid,
    scoring = 'accuracy',
    cv = kf,
    verbose = 1,
    n_jobs = -1
)


grid_search.fit(X_train, y_train)


print("Best Parameters:", grid_search.best_params_)
print("Best CV Score:", grid_search.best_score_)


best_model = grid_search.best_estimator_


train_sizes, train_scores, val_scores = learning_curve(
    estimator=best_model,
    X=X,
    y=y,
    cv=kf,
    scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)


plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
plt.plot(train_sizes, val_mean, 'o-', color='g', label='Cross-validation score')
plt.title('Learning Curve for XGBoost Classifier')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid()
plt.tight_layout()
plt.show()


train_scores, val_scores = validation_curve(
    estimator =best_model,
    X=X, y=y,
    param_name='n_estimators',
    param_range=[100,200,300,500],
    cv=kf,
    scoring='accuracy',
    n_jobs=-1
)


train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
param_range=[100,200,300,500]
param_name = 'Nr. estimators'
plt.figure(figsize=(8, 6))
plt.plot(param_range, train_mean, label='Training Score', color='blue', marker='o')
plt.plot(param_range, val_mean, label='Cross-Validation Score', color='green', marker='s')

plt.title('Validation Curve for XGBoost (max_depth)')
plt.xlabel(param_name)
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid()
plt.tight_layout()
plt.show()


y_pred = best_model.predict(test_trasf)


replace_map = {1: 'Introvert',0: 'Extrovert'}


submission = pd.DataFrame({
    'id': test['id'],         
    'Personality': [replace_map.get(x, x) for x in y_pred]
})




assert test.shape[0] == submission.shape[0]


# Save the DataFrame to a CSV file
submission.to_csv('submission.csv', index=False)
print("Submission created")

