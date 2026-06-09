
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
from sklearn.pipeline import Pipeline
#基准模型
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score


sample_submission=pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')
train_df=pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test_df=pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


train_df.head()


test_df.head()


train_df.info()


train_df.describe()


train_df.isnull().sum()


test_df.isnull().sum()


def select_type(df):
    columns=df.columns
    columns = columns.drop(['y','id'])
    column_num=[]
    column_category=[]
    for col in columns:
        if df[col].dtype=='int64':
            column_num.append(col)
        else:
            column_category.append(col)
    return column_num,column_category
train_num_col,train_category_col=select_type(train_df)
print(train_num_col)
ohe_cols = ["job", "marital", "contact", "poutcome", "month"]
ord_cols = ["education"]
bin_cols = ["default", "housing", "loan"] 


categorical_transformer_ohe = OneHotEncoder(handle_unknown='ignore') #分类变量独热编码

categorical_transformer_ord = Pipeline([ #序数编码
    ('ord', OrdinalEncoder(categories=[['primary','secondary','tertiary']], 
                          handle_unknown='use_encoded_value', 
                          unknown_value=-1))
])

binary_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1) #二元值编码

numerical_transformer = Pipeline([
    ('scaler', StandardScaler()) #对数据进行标准化为模型训练进行铺垫
])

preprocessor = ColumnTransformer([
    ('ohe', categorical_transformer_ohe, ohe_cols),  
    ('ord', categorical_transformer_ord, ord_cols),  
    ('bin', binary_transformer, bin_cols),          
    ('num', numerical_transformer, train_num_col)       
])


for col in train_category_col:
    proportions = (
        train_df.groupby(col)["y"]
        .mean()
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(14,4))

  
    sns.barplot(data=proportions, x=col, y='y', ax=axes[0],
                order=proportions.sort_values('y', ascending=False)[col])
    axes[0].set_title(f'Proportion of y by {col}')
    axes[0].tick_params(axis='x', rotation=45)

    
    sns.countplot(data=train_df, x=col, ax=axes[1],  #计算每个列中的属性所属的总样本数
                  order=train_df[col].value_counts().index)
    axes[1].set_title(f'Counts per {col}')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()



import warnings
warnings.filterwarnings('ignore')
for col in train_num_col:
    
    fig, axes = plt.subplots(1, 2, figsize=(12,4))

    sns.histplot(train_df[col], kde=True, ax=axes[0],bins=30, color='skyblue') 
    axes[0].set_title(f'Distribution of {col}')

    sns.boxplot(x=train_df[col], ax=axes[1], color='lightgreen')
    axes[1].set_title(f'Boxplot of {col}')

    plt.tight_layout()
    plt.show()
    


train_num_col=[x for x in train_df.columns if train_df[x].dtype in ['float64','int64']]
def corr_head(df,col):
    corr_m=df[col].corr()
    plt.figure(figsize=(10, 8))

    sns.heatmap(
        corr_m,
        annot=True,          
        cmap='coolwarm',     
        center=0,           
        fmt='.2f',           
        linewidths=0.5,      
        square=True          
    )

    plt.title(f'num of Heatmap', fontsize=15)
    plt.tight_layout()
    plt.show()
corr_head(train_df,train_num_col)


X=train_df.drop(['y','id'],axis=1)
y=train_df['y']


# baseline_models = {
#     'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
#     'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
#     # 'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
#     'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
#     # 'SVM': SVC(random_state=42, probability=True),
#     # 'Naive Bayes': GaussianNB(),
#     # 'K-NN': KNeighborsClassifier(),
#     'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
#     'LightGBM': LGBMClassifier(random_state=42)
# }

# def run_baseline_benchmark(X, y, test_size=0.2, cv=5, random_state=42, preprocessor=preprocessor):
#     # 划分训练测试集（使用分层抽样，适合分类问题）
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=test_size, random_state=random_state, stratify=y
#     )
    
#     results = []
    
#     for model_name, model in baseline_models.items():
#         try:
#             print(f"正在训练 {model_name}...")
            
        
#             pipeline = Pipeline([
#                 ('preprocessor', preprocessor), 
#                 ('classifier', model)            
#             ])
            
            
#             pipeline.fit(X_train, y_train)
            
#             # 预测
#             y_pred = pipeline.predict(X_test)
            
#             # 计算评估指标（使用加权平均，适合多分类）
#             accuracy = accuracy_score(y_test, y_pred)
#             precision = precision_score(y_test, y_pred, average='weighted')
#             recall = recall_score(y_test, y_pred, average='weighted')
#             f1 = f1_score(y_test, y_pred, average='weighted')
            
#             # 交叉验证
#             cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
#             cv_mean = cv_scores.mean()
#             cv_std = cv_scores.std()
            
#             # 存储结果
#             results.append({
#                 'Model': model_name,
#                 'Accuracy': round(accuracy, 4),
#                 'Precision': round(precision, 4),
#                 'Recall': round(recall, 4),
#                 'F1-Score': round(f1, 4),
#                 'CV Mean': round(cv_mean, 4),
#                 'CV Std': round(cv_std, 4),
#             })
            
#         except Exception as e:
#             print(f"{model_name} 训练失败: {str(e)}")
#             results.append({
#                 'Model': model_name,
#                 'Accuracy': 'N/A',
#                 'Precision': 'N/A',
#                 'Recall': 'N/A',
#                 'F1-Score': 'N/A',
#                 'CV Mean': 'N/A',
#                 'CV Std': 'N/A',
#             })
    
#     # 返回排序后的结果
#     results_df = pd.DataFrame(results)
#     return results_df.sort_values('Accuracy', ascending=False)

# results_df = run_baseline_benchmark(X, y)
# print(results_df)
    


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
rc=RandomForestClassifier(random_state=42, n_estimators=100)
pipeline = Pipeline([
                ('preprocessor', preprocessor), 
                ('classifier', rc)            
            ])
pipeline.fit(X_train,y_train)
y_pred = pipeline.predict(X_test)   
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)


# def randomized_search_rf(X, y, preprocessor,n_iter=100):

#     pipeline=Pipeline([
#         ('preprocessor', preprocessor),
#         ('classifier', RandomForestClassifier(random_state=42))
#     ])

    
#     param_dist = {
#         'classifier__n_estimators': [50, 100, 200],
#         'classifier__max_depth': [2,5,7],
#         # 'min_samples_split': [2, 5, 10, 15, 20],
#         # 'min_samples_leaf': [1, 2, 4, 6, 8],
#         # 'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7],
#         # 'bootstrap': [True, False],
#         # 'criterion': ['gini', 'entropy'] if problem_type == 'classification' else ['squared_error', 'absolute_error']
#     }
    
#     # 创建RandomizedSearchCV对象
#     random_search = RandomizedSearchCV(
#         estimator=pipeline,
#         param_distributions=param_dist,
#         n_iter=n_iter,
#         scoring='accuracy',
#         cv=5,
#         n_jobs=-1,
#         random_state=42,
#         verbose=1
#     )
    
#     random_search.fit(X, y)
    
#     print(f"最佳参数: {random_search.best_params_}")
#     print(f"最佳分数: {random_search.best_score_:.4f}")
    
#     return random_search.best_estimator_
# best_model=randomized_search_rf(X_train,y_train,preprocessor)



pipeline = Pipeline([
    ('preprocessor', preprocessor),  
    ('model', RandomForestClassifier(
        n_estimators=100,  
        max_depth=10,      
        random_state=42
    ))
])
model=pipeline.fit(X,y)
X_test=test_df.drop(['id'],axis=1)
y_test_pred=model.predict(X_test)


submission=pd.DataFrame({
    'id':test_df['id'],
    'y':y_test_pred
    })
submission.to_csv('/kaggle/working/submission.csv')




