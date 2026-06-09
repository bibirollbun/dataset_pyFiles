import numpy as np 
import pandas as pd 
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,GridSearchCV, KFold
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error


df=pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
df_test=pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


df.head()


df.shape,df_test.shape


df.info()


df.describe()


df_test.describe()


df.isnull().sum()


df_test.isnull().sum()


df.duplicated().sum()


df_test.duplicated().sum()


df['Sex'].value_counts()


df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2
df_test['BMI'] = df_test['Weight'] / (df_test['Height'] / 100) ** 2


num_features=[features for features in df.columns if df[features].dtype!='O']
print('Numeric Features {}'.format(num_features))


cat_features=[features for features in df.columns if df[features].dtype=='O']
print('Categorical Features {}'.format(cat_features))


plt.figure(figsize=(8,25))
for i in range(0,len(num_features)):
    plt.subplot(9,1,i+1)
    sns.histplot(x=df[num_features[i]],kde=True,color='g',line_kws={'color': '#8e44ad', 'linewidth': 2})
    plt.xlabel(num_features[i],fontweight='bold')
    plt.tight_layout()


Sex_counts = df.groupby('Sex')['Calories'].mean().sort_values(ascending=False).reset_index()
colors = ['lightcoral', 'indianred']
explode = [0.05] * len(Sex_counts)
plt.figure(figsize=(5, 6))
plt.pie(Sex_counts['Calories'],labels=Sex_counts['Sex'],autopct='%1.0f%%',startangle=100,colors=colors,explode=explode,wedgeprops=dict(width=0.3))
plt.title('Sex-wise Average Calorie Burn Distribution', fontsize=14)
plt.legend(title="Sex")
plt.tight_layout()
plt.show()


numeric_df = df.select_dtypes(include=['int64', 'float64'])
cor = numeric_df.corr()
sns.heatmap(cor,annot=True,cmap='coolwarm',fmt='.2f')
plt.title('Correlation Matrix',fontweight='bold')
plt.show()


df.head()


X = df.drop(columns=['id', 'Calories'])
y = df['Calories']
numeric_features=['Age','Height','Weight','Duration','Heart_Rate','Body_Temp','BMI']
categorical_features=['Sex']
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('num', StandardScaler(), numeric_features)
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=12,
    learning_rate=0.085465,
    subsample=0.86456,
    colsample_bytree=0.87422,
    tree_method='exact',
    eval_metric='rmse',
    n_jobs=-1,
    random_state=42
)
pipe = Pipeline([
    ('preprocess', preprocessor),
    ('model', xgb_model)
])

pipe.fit(X_train, y_train)


def rmsle_fun(y_true, y_pred):
    y_true = np.clip(y_true, 0, None)
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))

def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mse)
    r2_s = r2_score(true, predicted)
    rmsle = rmsle_fun(true, predicted)
    return mae, rmse, r2_s, rmsle
y_train_pred = pipe.predict(X_train)
y_test_pred = pipe.predict(X_test)

model_train_mae, model_train_rmse, model_train_r2, model_train_rmsle = evaluate_model(y_train, y_train_pred)
model_test_mae, model_test_rmse, model_test_r2, model_test_rmsle = evaluate_model(y_test, y_test_pred)

print('Model Performance for Training Dataset')
print('- Mean Absolute Error     : {:.4f}'.format(model_train_mae))
print('- Root Mean Squared Error : {:.4f}'.format(model_train_rmse))
print('- RMSLE                   : {:.4f}'.format(model_train_rmsle))
print('- R2 Score                : {:.4f}'.format(model_train_r2))
print('='*50)
print('Model Performance for Test Dataset')
print('- Mean Absolute Error     : {:.4f}'.format(model_test_mae))
print('- Root Mean Squared Error : {:.4f}'.format(model_test_rmse))
print('- RMSLE                   : {:.4f}'.format(model_test_rmsle))
print('- R2 Score                : {:.4f}'.format(model_test_r2))


onehot = pipe.named_steps['preprocess'].named_transformers_['cat']
ohe_feature_names = onehot.get_feature_names_out(categorical_features)
all_feature_names = np.concatenate([numeric_features, ohe_feature_names])
xgb_model = pipe.named_steps['model']
importances = xgb_model.feature_importances_
feature_imp_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_imp_df.head(20), palette='coolwarm')
plt.title('Feature Importances (XGBoost)')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()


X_test_df = df_test.drop(columns=['id'])
y_test_pred = pipe.predict(X_test_df)
submission = pd.DataFrame({
    'id': df_test['id'], 
    'Calories': y_test_pred.clip(0)
})
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("Submission file 'submission.csv' has been created.")




