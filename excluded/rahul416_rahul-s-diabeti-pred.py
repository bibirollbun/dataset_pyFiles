import numpy as np # linear algebra
import pandas as pd

train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
train.head()


train.info()


train.columns


cols_to_drop = ['id','ethnicity', 'education_level','income_level', 'employment_status']
idx = test['id']
train_df = train.drop(cols_to_drop,axis=1)
test_df = test.drop(cols_to_drop,axis=1)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_df['gender'] = le.fit_transform(train_df['gender'])
test_df['gender'] = le.transform(test_df['gender'])

train_df['smoking_status'] = le.fit_transform(train_df['smoking_status'])
test_df['smoking_status'] = le.transform(test_df['smoking_status'])


train_df


df_1 = train_df[train_df['diagnosed_diabetes']==1]
df_0 = train_df[train_df['diagnosed_diabetes']==0]
df_1 = df_1.sample(len(df_0))
train_df = pd.concat([df_1,df_0])
train_df


X = train_df.drop('diagnosed_diabetes',axis=1)
y = train_df['diagnosed_diabetes']
# test_df


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.4)


def clip_outliers_iqr(df, factor=1.5):
    df_clipped = df.copy()
    numeric_cols = df_clipped.select_dtypes(include="number").columns

    for col in numeric_cols:
        Q1 = df_clipped[col].quantile(0.25)
        Q3 = df_clipped[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR

        df_clipped[col] = df_clipped[col].clip(lower, upper)

    return df_clipped



x_train = clip_outliers_iqr(x_train)


from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier, BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, VotingClassifier
import xgboost as xgb
import lightgbm as lgb

xgb = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42)

lgb = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        learning_rate=0.02,      # Lower learning rate for smoother convergence
        num_leaves=31,           # Moderate leaf size to reduce overfitting
        n_estimators=500,        # Large upper bound; use early stopping to control
        colsample_bytree=0.7,    # Feature subsampling per tree
        subsample=0.7,           # Row subsampling for robustness
        random_state=42,         # Reproducibility
        n_jobs=-1,               # Use all available CPU cores
        verbosity=-1             # Suppress training logs
        )

gb = GradientBoostingClassifier()
bagging = BaggingClassifier()
ada = AdaBoostClassifier()
rf = RandomForestClassifier()
hist = HistGradientBoostingClassifier()

ml = VotingClassifier(
    estimators=[('xgb',xgb),
                ('xgb1',xgb),
                ('xgb2',xgb),
                ('lgb',lgb),
                ('lgb1',lgb),
                ('lgb2',lgb),
                ('rf',rf),
                ('ada',ada),
                ('gb',gb),
                ('hist',hist),
                ('bag',bagging)
               ],
    voting='soft'
)

ml.fit(x_train,y_train)


from sklearn.metrics import classification_report

y_pred = ml.predict(x_test)
print(classification_report(y_test,y_pred))


y_pred = ml.predict_proba(test_df)[:,1]

data = {
    'id':idx,
    'diagnosed_diabetes': y_pred
}

df = pd.DataFrame(data)
df.to_csv('submission.csv',index=False)



































# train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
# test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
# train.head()


# train.columns


# cols_to_remove = ['id','screen_time_hours_per_day','gender','ethnicity','education_level','income_level','employment_status']
# idx = test['id']
# train.drop(cols_to_remove,axis=1,inplace=True)
# test.drop(cols_to_remove,axis=1,inplace=True)


# train.smoking_status.unique()
# status = ['Current', 'Never', 'Former']
# train['smoking_status'] = train['smoking_status'].replace(status,[0,1,2])
# test['smoking_status'] = test['smoking_status'].replace(status,[0,1,2])


# train.corr()['diagnosed_diabetes']


# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.scatterplot(data= train.sample(100) ,x = 'sleep_hours_per_day',y = 'age',hue = 'diagnosed_diabetes')
# plt.show()


# train.drop(["hdl_cholesterol"],axis=1,inplace=True)
# test.drop(["hdl_cholesterol"],axis=1,inplace=True)



# df_1 = train[train['diagnosed_diabetes']==1]
# df_0 = train[train['diagnosed_diabetes']==0]
# df_1 = df_1.sample(len(df_0))
# train = pd.concat([df_1,df_0])
# train


# X = train.drop('diagnosed_diabetes',axis=1)
# y = train['diagnosed_diabetes']


# from sklearn.preprocessing import StandardScaler
# norm = StandardScaler()
# X_ = norm.fit_transform(X)

# test_ = norm.transform(test)



# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier

# X_train,X_test,y_train,y_test = train_test_split(X_,y)

# reg_tree = DecisionTreeClassifier(criterion='gini',max_depth = 8)

# reg_tree.fit(X_train,y_train)


# from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.ensemble import StackingClassifier

# from sklearn.ensemble import StackingClassifier

# # estimators = [
# #     ('lr1', LogisticRegression(max_iter=1000)),
# #     ('Dt', DecisionTreeClassifier(max_depth=8)),
# #     ('GB', GradientBoostingClassifier(n_estimators=10)),
# #     ('lr2', LogisticRegression(max_iter=1000))
# # ]

# # stack = StackingClassifier(
# #     estimators=estimators,
# #     final_estimator=RandomForestClassifier(n_estimators=200),
# #     cv=4,
# #     passthrough=False
# # )

# # stack.fit(X_train,y_train)



# import lightgbm as lgb
# clf = lgb.LGBMClassifier(
#     objective="binary",
#     metric="auc",
#     learning_rate=0.02,      # Lower learning rate for smoother convergence
#     num_leaves=31,           # Moderate leaf size to reduce overfitting
#     n_estimators=5000,       # Large upper bound; use early stopping to control
#     colsample_bytree=0.7,    # Feature subsampling per tree
#     subsample=0.7,           # Row subsampling for robustness
#     random_state=42,         # Reproducibility
#     n_jobs=-1,               # Use all available CPU cores
#     verbosity=-1             # Suppress training logs
# )
# clf.fit(X_train, y_train)


# from sklearn.metrics import classification_report
# y_dt = clf.predict(X_test)
# print(classification_report(y_test,y_dt))


# clf.predict_proba(X_test)


# import xgboost as xgb
# from sklearn.metrics import roc_auc_score, roc_curve

# model = xgb.XGBClassifier(
#     objective='binary:logistic',
#     eval_metric='auc',
#     n_estimators=300,
#     learning_rate=0.05,
#     max_depth=5,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     random_state=42)

# model.fit(X_train, y_train)

# y_pred_proba = model.predict_proba(X_test)[:, 1]


# from sklearn.ensemble import VotingClassifier
# from sklearn.ensemble import RandomForestClassifier
# ml = VotingClassifier(
#     estimators=[('xgb',model),
#                 ('lgb',clf),
#                 ('rf',RandomForestClassifier(n_estimators=200))],
#     voting='soft'
# )
# ml.fit(X_train,y_train)


# from sklearn.metrics import classification_report
# y_dt = ml.predict(X_test)
# print(classification_report(y_test,y_dt))


# y_pred = ml.predict_proba(test_)[:,1]



# # y_pred = ml.predict_proba(test_)[:,1]

# data = {
#     'id':idx,
#     'diagnosed_diabetes': y_pred
# }

# df = pd.DataFrame(data)
# df.to_csv('submission.csv',index=False)





df




