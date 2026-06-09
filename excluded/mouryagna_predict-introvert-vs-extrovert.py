import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
full_x_test=pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
full_y_test=pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


df.head()


df.info()


df.describe()


df.isnull().mean()*100


corr = df.select_dtypes(include='number').corr()
sns.heatmap(corr)


numeric_col=[col for col in df.columns if df[col].dtype!='object']
numeric_col.remove('id')
numeric_col


for col in numeric_col:
    sns.distplot(x=df[col])
    plt.title(col)
    plt.tight_layout()
    plt.show()


for col in numeric_col:
    sns.boxplot(x=df[col])
    plt.title(col)
    plt.tight_layout()
    plt.show()


df=df.drop(columns=['id'])


df.Personality.value_counts()


def remove_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] >= lower) & (df[col] <= upper)]    
df = remove_outliers_iqr(df,'Time_spent_Alone')


from sklearn.model_selection import train_test_split

x=df.drop(columns=['Personality'])
y=df.Personality

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=12)


from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
from sklearn.impute import KNNImputer,SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


le=LabelEncoder()
y_train=le.fit_transform(y_train)
y_test=le.transform(y_test)


cat_cols=['Drained_after_socializing','Stage_fear']
numeric_col=[col for col in x_train.columns if df[col].dtype!='object']


numeric_trf=(
    'num_trf',Pipeline([
        ('impute',SimpleImputer())
    ]),numeric_col
)
cat_trf=(
    'cat',Pipeline([
        ('impute',SimpleImputer(strategy='most_frequent')),
        ('encode',OrdinalEncoder())
    ]),cat_cols
)


transformer=ColumnTransformer(transformers=[numeric_trf,cat_trf],remainder='passthrough')


from xgboost import XGBClassifier

xgb=XGBClassifier(colsample_bytree= 0.6,
 gamma= 5,
 learning_rate= 0.1,
 max_depth= 3,
 n_estimators= 200,
 subsample= 0.6,n_jobs=-1,random_state=12)


pipe=Pipeline([
    ('transformer',transformer),
    ('clf',xgb)
])


pipe.fit(x_train,y_train)
y_pred=pipe.predict(x_test)


from sklearn.metrics import accuracy_score,ConfusionMatrixDisplay


accuracy_score(y_test,y_pred)


ConfusionMatrixDisplay.from_predictions(y_test,y_pred)


from sklearn.model_selection import GridSearchCV


# param_grid = {
#     'clf__n_estimators': [100, 200],
#     'clf__max_depth': [3, 5, 7],
#     'clf__learning_rate': [0.01, 0.05, 0.1,0.2],
#     'clf__subsample': [0.6, 0.8, 1.0],
#     'clf__colsample_bytree': [0.6, 0.8, 1.0],
#     'clf__gamma': [0, 1, 5]
# }
# xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
# pipe_xgb_grid=Pipeline([
#     ('transformer',transformer),
#     ('clf',xgb)
# ])
# grid_xgb=GridSearchCV(estimator=pipe_xgb_grid, param_grid=param_grid,cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
# grid_xgb.fit(x_train,y_train)
# y_pred=grid_xgb.predict(x_test)
# accuracy_score(y_test,y_pred)
# grid_xgb.best_params_


full_x_test.drop(columns=['id'],inplace=True)


y_encode=le.fit_transform(y)


pipe.fit(x, y_encode)


test_preds = pipe.predict(full_x_test)
test_preds_original = le.inverse_transform(test_preds)


submission = full_y_test.copy()
submission['Personality'] = test_preds_original
submission.to_csv('final_submission.csv', index=False)




