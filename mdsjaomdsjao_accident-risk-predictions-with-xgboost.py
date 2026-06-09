import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings("ignore")


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv',)
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')


train.head(3)


train.sample(2)


test.head(3)


train.isnull().sum()


train.info()


train['accident_risk'].describe()


train.columns


num = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']
boolean = ['road_signs_present','public_road','holiday','school_season']
categorical = ['road_type','lighting','weather','time_of_day']


train.columns


names = [ 'num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents','accident_risk']

fig, axs = plt.subplots(1, 5, figsize=(15, 4)) 
for i in range(0,5):
    axs[i].hist(train[names[i]], bins=20, color='lightblue', edgecolor='black')
    axs[i].set_title(names[i])

plt.suptitle('Feature values Distribution', fontsize= 15)
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(10,6))


cols = boolean+num+['accident_risk']
corr = train[cols].corr()

sns.heatmap(corr, cmap = 'crest', annot = True)
plt.title('Non categorical Feature correlation Heatmap',fontsize = 15, pad=10)
plt.tight_layout()
plt.show()


X = train.drop(['id','accident_risk'], axis=1)
y = train['accident_risk']

X_test = test.copy().drop(columns=['id'], axis=1)


from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


#preprocessor

prep = ColumnTransformer(transformers = [('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
                                    ('num', StandardScaler(), num)],
                                    remainder='passthrough')


params = {'objective': "reg:squarederror",
    'n_estimators': 3000,
    'learning_rate': 0.03,
    'max_depth': 10,
   'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_lambda': 1.0,
    'reg_alpha': 1.0,
    'random_state':42
}

from xgboost import XGBRegressor

model = Pipeline(steps = [('preprocessor', prep),
                            ('model', XGBRegressor(**params))
                            ])


from sklearn.model_selection import train_test_split


# split the data into training and validation sets and stratifying by the target variable
X_train, X_val, y_train, y_val = train_test_split(X,y, random_state=42, test_size = 0.2, stratify= y)

model.fit(X_train, y_train)


y_pred = model.predict(X_val)


# some performance metrics (validation)

from sklearn.metrics import mean_squared_error, r2_score

print(f'mean_squared_error [XGB Regressor]: {mean_squared_error(y_val, y_pred)}\nr2_score [XGB Regressor]: {r2_score(y_val, y_pred)}')


y_pred = model.predict(X_test)


submission['accident_risk'] = y_pred

submission.to_csv('submission.csv', index=False) # saving to submission file



#getting all feature names
feature_names = model['preprocessor'].get_feature_names_out()

importances = model['model'].feature_importances_

df_importances = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
})

df_importances = df_importances.sort_values(by='importance', ascending=False).head(10)


plt.figure(figsize=(8, 5))

plt.barh(df_importances['feature'], df_importances['importance'], color='tomato')

plt.title('Feature importances in XGBoost')
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()


submission.head()

