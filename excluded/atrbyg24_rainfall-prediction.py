import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, GridSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier


rainfall_train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv',index_col = 'id')
rainfall_test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv',index_col='id')


rainfall_train.head()


rainfall_train.info()


rainfall_train.describe()


corr = rainfall_train.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(data=corr,annot=True)


corr['rainfall'].sort_values(ascending=False)


for col in rainfall_train.columns:
    plt.figure()
    sns.histplot(data=rainfall_train,x=col,bins=50)
    plt.title(f"Histogram of {col}")
    plt.show()
    plt.clf()


for col in rainfall_train.columns:
    plt.figure()
    sns.histplot(data=rainfall_train,x=col,bins=50,hue='rainfall')
    plt.title(f"Histogram of {col}")
    plt.show()
    plt.clf()


for col in rainfall_train.columns:
    plt.figure(figsize=(20, 5))  
    sns.scatterplot(x=rainfall_train.index, y=col, data=rainfall_train, hue='rainfall')
    plt.xticks(range(0, len(rainfall_train.index), 60))
    plt.title(f"Scatterplot of {col}")
    plt.show()
    plt.clf()


y_train = rainfall_train['rainfall']
X_train = rainfall_train.drop('rainfall',axis=1)
X_test = rainfall_test


def shift_features(df):
    for c in ['pressure', 'maxtemp', 'temparature', 'mintemp','dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection','windspeed']:
        if c != 'day':
            for gap in [1]:
                df[c+f"_shift{gap}"]=df[c].shift(gap)
                df[c+f"_diff{gap}"]=df[c].diff(gap)
    return df

def drop_features(df):
    return df.drop(columns=['temparature','maxtemp','mintemp'],axis=1)

def day_features(df):
    df['sin_day']=np.sin(2*np.pi*df['day']/365)
    df['cos_day']=np.cos(2*np.pi*df['day']/365)
    return df

shift_transformer = make_pipeline(
    FunctionTransformer(shift_features),
    SimpleImputer(strategy='mean'),
    StandardScaler()
)

drop_transformer = make_pipeline(
    FunctionTransformer(drop_features),    
)

day_transformer = FunctionTransformer(day_features)

numerical_transformer = make_pipeline(
    SimpleImputer(strategy='mean'),
    StandardScaler()
)

preprocessor_lr = ColumnTransformer(
    transformers=[
        ('drop',drop_transformer,['temparature','maxtemp','mintemp']),
        ('day',day_transformer,['day']),
    ],remainder = numerical_transformer)

preprocessor_xgb = ColumnTransformer(
    transformers=[
        ('shift',shift_transformer,['pressure', 'maxtemp', 'temparature', 'mintemp','dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection','windspeed']),
        ('day',day_transformer,['day']),
    ],remainder = numerical_transformer)

lr_pipeline = make_pipeline(
    preprocessor_lr,
    LogisticRegression(solver='liblinear', random_state=42)
)

xgb_pipeline = make_pipeline(
    preprocessor_xgb,
    XGBClassifier(random_state=42)
)


param_grid_lr = {
    'logisticregression__C': [0.001, 0.01, 0.1, 1, 10, 100] 
}

tscv = TimeSeriesSplit(n_splits=6)

grid_search_lr = GridSearchCV(lr_pipeline, param_grid_lr, scoring = 'roc_auc', cv= tscv)
grid_search_lr.fit(X_train, y_train)

print(f"Best parameters: {grid_search_lr.best_params_}")
best_pipeline_lr = grid_search_lr.best_estimator_

for i, (train_index, test_index) in enumerate(tscv.split(X_train)):
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

    best_pipeline_lr.fit(X_train_fold, y_train_fold)
    y_probs_fold = best_pipeline_lr.predict_proba(X_test_fold)[:, 1]
    auc_score_fold = roc_auc_score(y_test_fold, y_probs_fold)
    print(f"Fold {i+1} AUC-ROC Score: {auc_score_fold}")

y_probs_lr = best_pipeline_lr.predict_proba(X_train)[:, 1]
auc_score_lr = roc_auc_score(y_train, y_probs_lr)
print(f"Overall AUC-ROC Score: {auc_score_lr}")

fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_train, y_probs_lr)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, label=f'AUC = {auc_score_lr:.2f}')
plt.plot([0, 1], [0, 1], 'k--')  
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

y_probs_best_lr = best_pipeline_lr.predict_proba(X_test)[:, 1]  


param_grid_xgb = {
    'xgbclassifier__n_estimators': [100, 150, 200],
    'xgbclassifier__learning_rate': [0.001, 0.01, 0.1],
    'xgbclassifier__max_depth': [3, 4, 5]
}

tscv = TimeSeriesSplit(n_splits=6)


grid_search_xgb = GridSearchCV(xgb_pipeline, param_grid_xgb, scoring ='roc_auc', cv=tscv
)
grid_search_xgb.fit(X_train, y_train)

print(f"Best parameters: {grid_search_xgb.best_params_}")
best_pipeline_xgb = grid_search_xgb.best_estimator_

for i, (train_index, test_index) in enumerate(tscv.split(X_train)):
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

    best_pipeline_xgb.fit(X_train_fold, y_train_fold)
    y_probs_fold = best_pipeline_xgb.predict_proba(X_test_fold)[:, 1]
    auc_score_fold = roc_auc_score(y_test_fold, y_probs_fold)
    print(f"Fold {i+1} AUC-ROC Score: {auc_score_fold}")

y_probs_xgb = best_pipeline_xgb.predict_proba(X_train)[:, 1]
auc_score_xgb = roc_auc_score(y_train, y_probs_xgb)
print(f"Overall AUC-ROC Score: {auc_score_xgb}")

fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(y_train, y_probs_xgb)

plt.figure(figsize=(8, 6))
plt.plot(fpr_xgb, tpr_xgb, label=f'AUC = {auc_score_xgb:.2f}')
plt.plot([0, 1], [0, 1], 'k--')  
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

y_probs_best_xgb = best_pipeline_xgb.predict_proba(X_test)[:, 1]  


df = pd.DataFrame({'rainfall':y_probs_best_lr}) 
df.index = range(2190, 2920)
df.index.name = 'id'


df.to_csv('submission.csv') 

