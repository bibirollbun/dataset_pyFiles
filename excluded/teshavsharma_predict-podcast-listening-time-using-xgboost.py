import numpy as np 
import pandas as pd 
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,GridSearchCV, KFold
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error


df=pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
df_test=pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


df.head()


df.shape,df_test.shape


df.info()


df.describe()


df_test.describe()


df.isnull().sum()


df_test.isnull().sum()


df['Episode_Length_minutes'].fillna(df['Episode_Length_minutes'].mean(),inplace=True)
df['Guest_Popularity_percentage'].fillna(df['Guest_Popularity_percentage'].median(),inplace=True)
df['Number_of_Ads'].fillna(df['Number_of_Ads'].median(),inplace=True)
df_test['Episode_Length_minutes'].fillna(df_test['Episode_Length_minutes'].mean(),inplace=True)
df_test['Guest_Popularity_percentage'].fillna(df_test['Guest_Popularity_percentage'].median(),inplace=True)


df.isnull().sum(), df_test.isnull().sum()


df.duplicated().sum(), df_test.duplicated().sum()


df['Episode_Title']=df['Episode_Title'].str.replace('Episode ','').astype(int)
df_test['Episode_Title']=df_test['Episode_Title'].str.replace('Episode ','').astype(int)


df.head()


df['Genre'].value_counts()


df['Episode_Sentiment'].value_counts()


df['Publication_Day'].value_counts()


df['Publication_Time'].value_counts()


df['Episode_Sentiment'].value_counts()


df['Podcast_Name'].value_counts()


df.head()


num_features=[features for features in df.columns if df[features].dtype!='O']
num_features


cat_features=[features for features in df.columns if df[features].dtype=='O']
cat_features


plt.figure(figsize=(8,30))
for i in range(0,len(num_features)):
    plt.subplot(7,1,i+1)
    sns.kdeplot(x=df[num_features[i]],color='g',shade=True)
    plt.xlabel(num_features[i],fontweight='bold')
    plt.tight_layout()


plt.figure(figsize=(16, len(cat_features) * 8))
for i, col in enumerate(cat_features, 1):
    plt.subplot(len(cat_features), 1, i)
    sns.countplot(x=df[col], order=df[col].value_counts().index, palette="hls")
    plt.xlabel(col,fontweight='bold')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


X = df.drop(columns=['id', 'Listening_Time_minutes'])
y = df['Listening_Time_minutes']
numeric_features=['Episode_Title','Episode_Length_minutes','Host_Popularity_percentage','Guest_Popularity_percentage','Number_of_Ads']
categorical_features=['Podcast_Name','Genre','Publication_Day','Publication_Time','Episode_Sentiment']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('num', StandardScaler(), numeric_features)
])
pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(objective='reg:squarederror', random_state=42))
])

param_grid = [{
    'model__n_estimators': [100],
    'model__max_depth': [12],
    'model__learning_rate': [0.055356, 0.085465, 0.064563],
    'model__subsample': [0.86456],
   'model__colsample_bytree': [0.87422],
    'model__gamma': [1.40345],
   'model__reg_alpha': [2.78558],
    'model__reg_lambda': [3.57657],
    'model__min_child_weight': [6],
    'model__tree_method': ['exact']
}]


k_cv = KFold(n_splits=4, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=k_cv,
    scoring={'r2': 'r2', 'mse': 'neg_mean_squared_error'},
    refit='mse',
    n_jobs=-1,
    verbose=2
)
grid_search.fit(X_train, y_train)
print(f"Best score: {grid_search.best_score_}")
print(f"Best Estimator: \n {grid_search.best_estimator_}")


def evaluate_model(true,predicted):
    mae=mean_absolute_error(true,predicted)
    mse=mean_squared_error(true,predicted)
    rmse=np.sqrt(mean_squared_error(true,predicted))
    r2_s=r2_score(true,predicted)
    return mae,rmse,r2_s
y_train_pred = grid_search.best_estimator_.predict(X_train)
y_test_pred = grid_search.best_estimator_.predict(X_test)


model_train_mae, model_train_rmse, model_train_r2 = evaluate_model(y_train, y_train_pred)
model_test_mae, model_test_rmse, model_test_r2 = evaluate_model(y_test, y_test_pred)

print('Model Performance for Training Dataset')
print('- Mean Absolute Error: {:.4f}'.format(model_train_mae))
print('- Root Mean Squared Error: {:.4f}'.format(model_train_rmse))
print('- R2 Score: {:.4f}'.format(model_train_r2))
print('='*50)
print('Model Performance for Test Dataset')
print('- Mean Absolute Error: {:.4f}'.format(model_test_mae))
print('- Root Mean Squared Error: {:.4f}'.format(model_test_rmse))
print('- R2 Score: {:.4f}'.format(model_test_r2))

print('\n')


best_pipeline = grid_search.best_estimator_
best_pipeline.fit(X_train, y_train)
onehot = best_pipeline.named_steps['preprocessor'].named_transformers_['cat']
ohe_feature_names = onehot.get_feature_names_out(categorical_features)
all_feature_names = np.concatenate([numeric_features, ohe_feature_names])
xgb_model = best_pipeline.named_steps['model']
importances = xgb_model.feature_importances_

feature_imp_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_imp_df.head(20), palette='coolwarm')
plt.title('Top 20 Feature Importances (XGBoost)')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()


df_test['predict'] = grid_search.best_estimator_.predict(df_test.drop(columns = ['id']))
df_submission = pd.DataFrame({
    'id': df_test['id'], 
    'Listening_Time_minutes' : df_test['predict']
})
df_submission.to_csv('submission.csv', index = False)
df_submission.info()




