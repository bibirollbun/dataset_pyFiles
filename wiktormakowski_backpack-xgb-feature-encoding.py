import numpy as np # linear algebra
import pandas as pd 
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



df_train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
df_train_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
targetCol = 'Price'
print(f'Shape df_train: {df_train.shape},\n Shape df_test: {df_test.shape} Shape df_train_extra: {df_train_extra.shape}\n ')


df_train.drop(columns='id',inplace=True)
df_train_extra.drop(columns='id',inplace=True)
df_train.head()


df_train.info()


cat_cols = df_train.select_dtypes(include=['object']).columns
num_cols = df_train.select_dtypes(exclude=['object']).columns.difference([targetCol])
features = df_train.columns.difference([targetCol])
cat_cols,num_cols,features


info={}
for col in df_train.columns:
    num_of_na = df_train[col].isna().sum()
    pctNaN = f'{100*num_of_na/df_train[col].count():.2f}%'
    num_of_unique = df_train[col].unique().shape[0]
    info[col] = [num_of_na,pctNaN,num_of_unique]
    # if num_of_na:
    #     print(f'Number of NaN records {num_of_na}')
    #     print(f'Percentage of NaN records: {"{0:.4f}".format(pctNaN)}%, column: "{col}"')
df_info = pd.DataFrame(info).reset_index()
df_info.iloc[0,0] = "Number of NaN"
df_info.iloc[1,0] = "Percentage of NaN"
df_info.iloc[2,0] = "Number of unique"


print(f'Contains duplicates? {df_train.duplicated().any()}') # there is no duplicated in the dataset
df_info = df_info.T
df_info.columns = df_info.iloc[0]
df_info = df_info.iloc[1:]
df_info


import seaborn as sns
fig,ax = plt.subplots(1,1)
print(df_train[targetCol].describe())
print(df_train[df_train[targetCol]<150][targetCol].describe()) # without capped values
sns.histplot(data=df_train,x=targetCol,kde=True,ax=ax)


df_train[targetCol].value_counts().sort_values(ascending=False)
df_train[df_train[targetCol]<150][targetCol].describe()


def barplot(xaxis,yaxis,ax,colName,font_size=7,orient='v',hue=None):
    
    pal = np.array(sns.color_palette('pastel',len(yaxis)))
    
    barplt = sns.barplot(x=xaxis,y=yaxis,orient=orient,hue=hue,ax=ax,palette=pal)
    sns.despine(left=True)
    
    for bar in barplt.patches:
        height = bar.get_height()
        width = bar.get_width()
        ax.set_yticks([])
        ax.text( bar.get_x()+width/2,
                max(yaxis)*0.03,
                f'{height:.0f}',
                ha='center',
                va='center',
                fontweight=700,
                fontsize=font_size,
             
                color='#222222')
    ax.set_title(
        f'Counts column: {colName}'
    )
    ax.set_xticklabels(xaxis,rotation=0)


fig,axes = plt.subplots(nrows=7,ncols=1,figsize=(8,40))
for i,col in enumerate(cat_cols):
    counts = df_train[col].value_counts()
    mean_by_target = df_train.groupby(col)[targetCol].mean().round(2)
    label = [f'{idx} \n Mean target: \n {val}' for idx,val in zip(mean_by_target.index, mean_by_target.values)]
    barplot(label,counts.values,axes[i],col)
    print(counts.values,counts.index)
plt.subplots_adjust(hspace=0.4)


from itertools import combinations
new_cols_pairwise = pd.DataFrame()
for col1,col2 in combinations(cat_cols,2):
    print(col1,col2)
    new_cols_pairwise[f'{col1}_{col2}'] = df_train[col1].astype(str)+'_'+df_train[col2].astype(str)
new_cols_pairwise


import phik
corr_num = list(num_cols) + [targetCol]
df_train[corr_num].corr()['Price']

# Compute correlation matrix with Phi-K
corr_matrix = df_train.phik_matrix()
corr_matrix2 = pd.concat([df_train[targetCol],new_cols_pairwise],axis=1).phik_matrix()

# Plot heatmap
plt.figure(figsize=(9, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".3f")
plt.title("Phi-K Correlation Heatmap")
plt.show()

plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix2, annot=True, cmap="coolwarm", fmt=".3f",annot_kws={"size":6})
plt.title("Phi-K Correlation Heatmap")
plt.show()


from sklearn.base import TransformerMixin, BaseEstimator

class featureEncoding(TransformerMixin,BaseEstimator):
    def __init__(self):
        self.size_dict = {"Small":1,"Medium":2,"Large":3}
        self.cat_cols = None
        self.feature_names = df_train.drop(columns=['Price']).columns
    def __printinfo__(self,stage,X):
        
        print(f"\nğŸ”� {stage} ğŸ”�")
        print("Shape:", X.shape)
        print("Column Names:", list(X.columns))
        print(X.info())
        print("="*10)
        
    def fit(self,X,y=None):
        if isinstance(X,pd.DataFrame):
            self.feature_names = X.columns
            self.cat_cols = X.select_dtypes(include=['object']).columns
        return self
        
    def transform(self,X):

        if isinstance(X, np.ndarray):  # If X is a NumPy array, convert it to DataFrame
            X = pd.DataFrame(X, columns=self.feature_names)

        # Make sure numerical columns are numerical 
        for col in num_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')

        # Debugging
        self.__printinfo__("Before transformation",X)
        
        # Categorical pairwise combinations
        new_cols_pairwise = pd.DataFrame()
        for col1,col2 in combinations(cat_cols,2):
            new_cols_pairwise[f'{col1}_{col2}'] = X[col1].astype(str)+'_'+X[col2].astype(str)


            
        X = pd.concat([X,new_cols_pairwise],axis=1)
        print(X.columns)
        
        # Size mapping:
        X['Size'] = X['Size'].map(self.size_dict)

        # Size, compartments - capacity relation
        # We'll see if these combinations explain price better than each of them individually
        X['Size_To_Capacity_Ratio'] = X['Size']/X['Weight Capacity (kg)']
        X['Size_x_Capacity'] = X['Size'] * X['Weight Capacity (kg)']
        X['Size2_x_Capacity'] = (X['Size']**2) * X['Weight Capacity (kg)']
        X['Compartments_To_Capacity_Ratio'] =  X['Compartments']/X['Weight Capacity (kg)']


        self.__printinfo__("After transformation",X)
        
        return X


type(new_cols_pairwise)



from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer,make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score

ordinal_enc = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', make_column_selector(dtype_exclude="category")),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), make_column_selector(dtype_include="category"))  
    ]
)

pipelinePreprocessing = Pipeline([
    ('feature_encoding', featureEncoding()),
])
#XGB and neural network ensemble???


df_full = pd.concat([df_train,df_train_extra])
X_train = df_full.drop(columns=["Price"])
y_train = df_full["Price"]
df_test.drop(columns=["id"],inplace=True)
X_train.shape,y_train.shape


X_train_transformed = pipelinePreprocessing.fit_transform(X_train)
X_test_transformed = pipelinePreprocessing.transform(df_test)


X_train_transformed.dtypes


ordinal_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
cat_cols = X_train_transformed.select_dtypes(include=['object']).columns
num_cols =  X_train_transformed.select_dtypes(exclude=['object']).columns

X_train_transformed[cat_cols] = ordinal_enc.fit_transform(X_train_transformed[cat_cols]) 
X_test_transformed[cat_cols] = ordinal_enc.fit_transform(X_test_transformed[cat_cols])


X_train_transformed.head()


xgb = XGBRegressor(objective='reg:squarederror',
                         n_estimators=100, 
                         learning_rate=0.02, 
                         max_depth=6, 
                         subsample=0.8, 
                         colsample_bytree=0.8, 
                         random_state=42,  
                         verbosity=2)

cv_score = cross_val_score(xgb,X_train_transformed,y_train,cv=3,scoring="neg_root_mean_squared_error",error_score='raise',n_jobs=-1)
rmse_positive = -cv_score

print(f"Cross-Validation RMSE Scores: {rmse_positive}")
print(f"Mean RMSE: {rmse_positive.mean():.4f}")
print(f"Standard Deviation: {rmse_positive.std():.4f}")


# import optuna

# def objective(trial):
#     params = {
#         "n_estimators": trial.suggest_int("n_estimators", 100, 100),
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
#         "max_depth": trial.suggest_int("max_depth", 3, 12),
#         "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
#         "subsample": trial.suggest_float("subsample", 0.5, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
#         "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 4.0),
#         "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 4.0)
#     }

#     xgb = XGBRegressor(objective="reg:squarederror",**params)
#     cv_score = -cross_val_score(xgb,X_train_transformed,y_train,cv=3,scoring="neg_root_mean_squared_error",error_score='raise',n_jobs=-1).mean()

#     return cv_score
    
# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=20)




# best_params = study.best_params
# print("Best Parameters:", best_params)
# xgb = XGBRegressor(**best)


optuna_params = { 'learning_rate': 0.10452790683034166, 
                 'max_depth': 7, 'min_child_weight': 20, 
                 'subsample': 0.6174565400510688,
                 'colsample_bytree': 0.6708587003174691, 
                 'reg_alpha': 2.4346567544929094, 
                 'reg_lambda': 2.489846033658522}


xgb = XGBRegressor(**optuna_params,n_estimators=200,objective="reg:squarederror")
cv_score = -cross_val_score(xgb,X_train_transformed,y_train,cv=5,scoring="neg_root_mean_squared_error",error_score='raise',n_jobs=-1)
print(f'CV scores: {cv_score}')
print(f'CV scores: {cv_score.mean()}')
print(f'CV scores: {cv_score.std()}')


X_train


X_train_final,X_cv,y_train_final,y_cv = train_test_split(X_train_transformed,y_train,test_size=0.2,random_state=42)


X_train_final


df_test


X_train_enc = X_train.copy()
new_cat_cols = X_train.select_dtypes(include=['object']).columns
print(new_cat_cols)
X_test_enc = df_test.copy()
X_train_enc[new_cat_cols] = ordinal_enc.fit_transform(X_train_enc[new_cat_cols])
X_test_enc[new_cat_cols] = ordinal_enc.fit_transform(X_test_enc[new_cat_cols])
xgb.fit(X_train_enc,y_train)
y_pred = xgb.predict(X_test_enc)


X_train_enc


xgb.feature_importances_


feature_importance_df = pd.DataFrame({
    'Feature': X_train_enc.columns,
    'Importance': xgb.feature_importances_
})

# Sort by importance (descending)
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
feature_importance_df


xgb_final = XGBRegressor(**optuna_params,n_estimators=500,objective="reg:squarederror")
xgb_final.fit(X_train_final,y_train_final,eval_set=[(X_cv, y_cv)],early_stopping_rounds=20)



xgb_final
feature_importance_df = pd.DataFrame({
    'Feature': X_train_transformed.columns,
    'Importance': xgb_final.feature_importances_
}).sort_values(by='Importance', ascending=True)  # Least important first

# new_features = feature_importance_df['Feature'][20:].values
feature_importance_df


new_features = feature_importance_df['Feature'][10:].values
feature_importance_df


xgb_final = XGBRegressor(**optuna_params,n_estimators=500,objective="reg:squarederror")
xgb_final.fit(X_train_final[new_features],y_train_final,eval_set=[(X_cv[new_features], y_cv)],early_stopping_rounds=20)



y_pred = xgb_final.predict(X_test_transformed[new_features])


submission = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")
submission['Price'] = y_pred
submission.to_csv("28-02-Submission04.csv",index=False)
submission.head()

