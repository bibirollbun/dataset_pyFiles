import warnings
warnings.simplefilter("ignore")
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder


train_df=pd.read_csv('/kaggle/input/playground-series-s4e6/train.csv')
test_df=pd.read_csv('/kaggle/input/playground-series-s4e6/test.csv')
train_ids=train_df['id']
test_ids=test_df['id']
train_df.drop('id', axis=1, inplace=True)
test_df.drop('id', axis=1, inplace=True)


print(f"Training set shape: {train_df.shape}")
print(f"Testing set shape: {test_df.shape}")


train_df.info()


train_df.describe().T


train_df.isna().sum()


plt.figure(figsize=(10, 6))
sns.heatmap(train_df.isnull(), cmap="viridis", cbar=False, yticklabels=False)
plt.title("Missing values:")
plt.show()


for col in train_df.columns:
    plt.title(f"Distribution of {col}")
    sns.histplot(train_df[col], kde=True)
    plt.show()


for col in train_df.iloc[:,:-1].columns:
    plt.title(col)
    sns.boxplot(train_df[col])
    plt.show()



corr=train_df.corr(numeric_only=True)
plt.figure(figsize=(20,20))
sns.heatmap(corr)
plt.show()


for col in train_df.iloc[:,:-1].columns:
    print("*"*30)
    print(f"Relvant Correlations with \033[1m{col}\033[0m:")
    corr_index=corr[col].drop(index=col)
    for col,v in corr_index.items():
        if abs(v)>0.3:
            print(f"{col}: {v:.2f}")
        


target=train_df['Target']
train_df.drop('Target',axis=1,inplace=True)


target_dict={
    'Graduate':0,
    'Dropout':1,
    'Enrolled':2
}
target_num=target.map(target_dict)
X_train,X_test,y_train,y_test=train_test_split(train_df,target_num,test_size=0.2, random_state=42)



log_transformer=FunctionTransformer(np.log1p)
num_pipeline=make_pipeline(StandardScaler())
preprocessing=ColumnTransformer([
    ('num',num_pipeline,train_df.columns)
])


catboost=CatBoostClassifier(verbose=0)
xgb=XGBClassifier()
rf=RandomForestClassifier()
tree=DecisionTreeClassifier()
sgd=SGDClassifier()

classifiers_dict={
    'CatBoost':catboost,
    'XGBoost':xgb,
    'RandomForest':rf,
    'DecisionTree':tree,
    'SGD':sgd
}
for name, classifier in classifiers_dict.items():
    print('*'*30)
    print(name)
    print(np.sum(cross_val_score(make_pipeline(preprocessing,classifier),X_train,y_train,cv=3, scoring='accuracy'))/3)



def catboost_cross_val_accuracy(X,y):
    return np.sum(cross_val_score(make_pipeline(preprocessing,catboost),X_train,y_train,cv=3, scoring='accuracy'))/3


train_df.columns


#ids_to_drop = train_df.query('`Marital status` > 5').index
#train_df['target_num']=target_num
#train_df.drop(index=ids_to_drop, inplace=True)
#target_num=train_df['target_num']
#train_df.drop('target_num', axis=1,inplace=True)
#X_train,X_test,y_train,y_test=train_test_split(train_df,target_num,test_size=0.2, random_state=42)



catboost_cross_val_accuracy(X_train,y_train)


cat_cols = [
    "Father's occupation",
    "Mother's occupation",
    "Father's qualification",
    "Mother's qualification",
    "Nacionality",
    "Previous qualification",
    "Course",
    "Application order",
    "Application mode",
    "Marital status"]
cat_pipeline=make_pipeline(OneHotEncoder(handle_unknown='ignore'))
num_pipeline=make_pipeline(StandardScaler())
num_cols=[]
for col in train_df.columns:
    if col not in cat_cols:
        num_cols.append(col)
        
preprocessing=ColumnTransformer([
    ('num',num_pipeline,num_cols),
    ('cat',cat_pipeline,cat_cols)
])


for name, classifier in classifiers_dict.items():
    print('*'*30)
    print(name)
    print(np.sum(cross_val_score(make_pipeline(preprocessing,classifier),X_train,y_train,cv=3, scoring='accuracy'))/3)


for col in cat_cols:
    print(col)
    print(train_df[col].value_counts())
    print('*'*30)


def speciffic_value_below_treshold(df,numeric_treshold,column_name,value):
    value_counts=df[column_name].value_counts()
    def check_count(x):
        if value_counts[x] >= numeric_treshold:
            return x
        else:
            return value
    df[column_name] = df[column_name].apply(check_count)
    return df


train_df=speciffic_value_below_treshold(train_df, 100, "Father's occupation", -1)
train_df=speciffic_value_below_treshold(train_df, 100, "Mother's occupation", -1)
train_df=speciffic_value_below_treshold(train_df, 100, "Father's qualification", -1)
train_df=speciffic_value_below_treshold(train_df, 100, "Mother's qualification", -1)
train_df=speciffic_value_below_treshold(train_df, 50, "Nacionality", -1)
train_df=speciffic_value_below_treshold(train_df, 100, "Previous qualification", -1)
train_df=speciffic_value_below_treshold(train_df, 100, "Course", -1)
train_df=speciffic_value_below_treshold(train_df, 30, "Application order", -1)
train_df=speciffic_value_below_treshold(train_df, 50, "Application mode", -1)
train_df=speciffic_value_below_treshold(train_df, 100, "Marital status", -1)


X_train,X_test,y_train,y_test=train_test_split(train_df,target_num,test_size=0.2, random_state=42)


print(np.sum(cross_val_score(make_pipeline(preprocessing,catboost),X_train,y_train,cv=3, scoring='accuracy'))/3)


final_catboost=make_pipeline(preprocessing,catboost)


test_df=speciffic_value_below_treshold(test_df, 70, "Father's occupation", -1)
test_df=speciffic_value_below_treshold(test_df, 70, "Mother's occupation", -1)
test_df=speciffic_value_below_treshold(test_df, 70, "Father's qualification", -1)
test_df=speciffic_value_below_treshold(test_df, 70, "Mother's qualification", -1)
test_df=speciffic_value_below_treshold(test_df, 35, "Nacionality", -1)
test_df=speciffic_value_below_treshold(test_df, 70, "Previous qualification", -1)
test_df=speciffic_value_below_treshold(test_df, 70, "Course", -1)
test_df=speciffic_value_below_treshold(test_df, 20, "Application order", -1)
test_df=speciffic_value_below_treshold(test_df, 35, "Application mode", -1)
test_df=speciffic_value_below_treshold(test_df, 70, "Marital status", -1)


final_catboost.fit(train_df, target)


preds=final_catboost.predict(test_df)
result_df = pd.DataFrame({'id': test_ids, 'Target': preds.flatten()})
result_df.to_csv('Adacemic Succes submission.csv',index=False)




