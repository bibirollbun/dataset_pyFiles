


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
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


print('Hello world')


df_train=pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
df_train.head()


df_train.shape


df_test=pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
df_test.head()


df_test.shape





df_train.shape





df=pd.concat([df_train,df_test],axis=0,ignore_index=True)


df.shape


df.isna().sum()


plt.figure(figsize=(20,8))
sns.heatmap(df.isna(),cbar=False,cmap='viridis')
plt.title("Missing value")
plt.show()


# sns.pairplot(df)





df['pdays'].tail()


df['pdays'].value_counts()


# df['balance_log']=np.log1p(df['balance'])
df['previous_log']=np.log1p(df['previous'])
df['campaign_log']=np.log1p(df['campaign'])
df['duration_log']=np.log1p(df['duration'])
# df['pdays_log']=np.log1p(df['pdays'])




df.columns





df.columns


df.info()








month_to_num = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6,
                'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
df['month_num'] = df['month'].map(month_to_num)
df['day_of_year'] = (df['month_num'] - 1)*30 + df['day']  # approx



df.columns





df['campaign'].value_counts()


df=df.drop(['previous','month','campaign','duration'],axis=1,errors='ignore')


df.head()


# df['pdays'].value_counts()


df.head()


df.shape


df_dataset_train=df[:750000]
df_dataset_test=df[750000:]


df_dataset_train_x=df_dataset_train.drop(['y'],axis=1)









df['default']=df['default'].map({'yes':1,'no':0})
df['housing']=df['housing'].map({'yes':1,'no':0})
df['loan']=df['loan'].map({'yes':1,'no':0})




skewness = df.skew(numeric_only=True)
skewness[abs(skewness) > 1]  # hig


df.head()





# from sklearn.preprocessing import OneHotEncoder,StandardScaler
# from sklearn.compose import ColumnTransformer

# numerical_feature=['age','day','day_of_year']
# categorical_feature=['job','marital','education','contact','poutcome']


# nontreebased_transformer=ColumnTransformer([
#     ('num',StandardScaler(),numerical_feature),
#     ('cat',OneHotEncoder(),categorical_feature)
# ])


# nontreetransformed_train_x=nontreebased_transformer.fit_transform(df_dataset_train_x)
# nontreetransformed_test_x=nontreebased_transformer.transform(df_dataset_test)




# from sklearn.preprocessing import OneHotEncoder,StandardScaler
# from sklearn.compose import ColumnTransformer

# categorical_feature=['job','marital','education','contact','poutcome']


# treebased_transformer=ColumnTransformer([
#     ('cat',OneHotEncoder(),categorical_feature)
# ])


# treetransformed_train_x=nontreebased_transformer.fit_transform(df_dataset_train_x)
# treetransformed_test_x=nontreebased_transformer.transform(df_dataset_test)




# good but the better way of doing is below




df.shape


df_final=df.drop(['id'],axis=1)





df_train_CD=df_final[:750000]
df_train_CD.head()



df_train_CD_x=df_train_CD.drop(['y'],axis=1)
df_train_CD_y=df_train_CD['y']
df_train_CD_y


df_test_CD=df_final[750000:]
df_test_CD.head()


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

numeric_features=['age','day','day_of_year']
categorical_features=['job','marital','education','contact','poutcome']

# Tree-based pipeline
tree_preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
], remainder='passthrough')



tree_models={
    'RFModel' : Pipeline([
    ('preprocessor', tree_preprocessor),
    ('model', RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))
]),
    'GBModel' : Pipeline([
    ('preprocessor', tree_preprocessor),
    ('model', GradientBoostingClassifier())
]),
        "XGBoostModel": Pipeline([
        ('preprocessor', tree_preprocessor),
        ('model', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    ])
    
}
    


# Non-tree pipeline
non_tree_preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
],remainder='passthrough')


non_tree_models={
    
    'LRModel':Pipeline([
        ('preprocessor',non_tree_preprocessor),
        ('model',LogisticRegression())
    ]),
    
    'KNNModel':Pipeline([
    ('preprocessor',non_tree_preprocessor),
    ('model',KNeighborsClassifier())
    ]),
    'SVMModel':Pipeline([
        ('preprocessor',non_tree_preprocessor),
        ('model',SVC())
    ])
    
    
    
}







from sklearn.model_selection import cross_val_score, train_test_split, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score



# Split into train/test for confusion matrix visualization (optional)
X_train, X_test, y_train, y_test = train_test_split(df_train_CD_x, df_train_CD_y, test_size=0.2, random_state=42)



# Combine both tree and non-tree models
# all_models = 
# {**tree_models, **non_tree_models}




# for name, model_pipeline in tree_models.items():
#     print(f"\n=== {name} ===")
    
#     # Fit the pipeline on training data
#     model_pipeline.fit(X_train, y_train)
    
#     # Predictions
#     y_pred = model_pipeline.predict(X_test)
    
#     # Confusion matrix
#     cm = confusion_matrix(y_test, y_pred)
#     print("Confusion Matrix:\n", cm)
    
#     # Classification report (accuracy, precision, recall, f1-score)
#     print("Classification Report:\n", classification_report(y_test, y_pred))
    
#     # Optional: cross-validation score for more robust metric
#     scores = cross_val_score(model_pipeline, df_train_CD_x, df_train_CD_y, cv=5, scoring='accuracy')
#     print("5-fold CV Accuracy:", scores.mean())





# for name, model_pipeline in non_tree_models.items():
#     print(f"\n=== {name} ===")
    
#     # Fit the pipeline on training data
#     model_pipeline.fit(X_train, y_train)
    
#     # Predictions
#     y_pred = model_pipeline.predict(X_test)
    
#     # Confusion matrix
#     cm = confusion_matrix(y_test, y_pred)
#     print("Confusion Matrix:\n", cm)
    
#     # Classification report (accuracy, precision, recall, f1-score)
#     print("Classification Report:\n", classification_report(y_test, y_pred))
    
#     # Optional: cross-validation score for more robust metric
#     scores = cross_val_score(model_pipeline, df_train_CD_x, df_train_CD_y, cv=5, scoring='accuracy')
#     print("5-fold CV Accuracy:", scores.mean())



# from sklearn.model_selection import GridSearchCV

# param_grid = {
#     'model__n_estimators': [100, 200, 300],
#     'model__max_depth': [3, 5, 7],
#     'model__learning_rate': [0.01, 0.1, 0.3],
#     'model__subsample': [0.8, 1.0],
#     'model__colsample_bytree': [0.8, 1.0]
# }

# grid_search = GridSearchCV(tree_models['XGBoostModel'], param_grid, 
#                            cv=3, scoring='accuracy', n_jobs=-1)

# grid_search.fit(X_train, y_train)

# print("Best Parameters:", grid_search.best_params_)
# print("Best CV Accuracy:", grid_search.best_score_)



df.shape



from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

best_params = {
    'colsample_bytree': 1.0,
    'learning_rate': 0.1,
    'max_depth': 7,
    'n_estimators': 300,
    'subsample': 0.8
}

final_model = Pipeline([
    ('preprocessor', tree_preprocessor),
    ('model', XGBClassifier(**best_params, random_state=42, use_label_encoder=False, eval_metric='logloss'))
])

final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)



from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Error Rate:", 1 - accuracy_score(y_test, y_pred))




df_test_CD_x=df_test_CD.drop(['y'],axis=1)


df_test_CD_x.head()


test_preds=final_model.predict(df_test_CD_x)
test_preds


submission = pd.DataFrame({
    "id": df_test["id"],  # must come from original test.csv
    "y": test_preds   # 1D array is fine
})

# Save for Kaggle
submission.to_csv("submission.csv", index=False)

print("✅ submission.csv ready with shape:", submission.shape)
submission.head()




