import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error,mean_squared_log_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor



df_train=pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
df_train.drop(columns=['id'],inplace=True)
df_train.head(10)




df_test=pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
df_test.drop(columns=['id'],inplace=True)
df_test.head(10)


df_train.info(),df_test.info()


df_train.isnull().sum(),df_test.isnull().sum()


df_train['Sex'].value_counts(),df_test['Sex'].value_counts()


# df_train['Sex']=pd.get_dummies(df_train['Sex'],drop_first=True,dtype=int)
# df_test['Sex']=pd.get_dummies(df_test['Sex'],drop_first=True,dtype=int)



from sklearn.preprocessing import LabelEncoder,StandardScaler
le=LabelEncoder()

df_train['Sex']=le.fit_transform(df_train['Sex'])
df_test['Sex']=le.transform(df_test['Sex'])


df_train.sample(10)


df_train.info(),df_test.info()


df_train.columns


numerical_col=['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate',
       'Body_Temp']
all_numerical_col=['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp',
       'Calories']
plt.figure(figsize=(15, 20))

i = 1
for col in all_numerical_col:
    plt.subplot(4, 2, i)
    sns.boxplot(x=df_train[col])
    plt.title(col)
    i += 1

plt.tight_layout()
plt.show()




# for col in numerical_col:
#   q1=df_train[col].quantile(0.25)
#   q3=df_train[col].quantile(0.75)
#   iqr=q3-q1

#   lower_bound=q1-1.5*iqr
#   upper_bound=q3+1.5*iqr
#   df_train=df_train[(df_train[col]> lower_bound) & (df_train[col]<upper_bound)]

# # Remove outliers from df_test
# for col in numerical_col:
#     q1 = df_test[col].quantile(0.25)
#     q3 = df_test[col].quantile(0.75)
#     iqr = q3 - q1

#     lower_bound = q1 - 1.5 * iqr
#     upper_bound = q3 + 1.5 * iqr

#     df_test = df_test[(df_test[col] > lower_bound) & (df_test[col] < upper_bound)]






plt.figure(figsize=(15, 20))

i = 1
for col in all_numerical_col:
    print(f'{col}: ',df_train[col].skew())
    plt.subplot(4, 2, i)
    sns.kdeplot(data=df_train,x=df_train[col])
    plt.title(col)
    i += 1

plt.tight_layout()
plt.show()






df_train.info(),df_test.info()


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()






df_train['Calories'] = np.log1p(df_train['Calories'])
X = df_train.drop('Calories', axis=1)
y = df_train['Calories']

X_test = df_test.copy()

X_scaled=sc.fit_transform(X)
X_test_scaled=sc.transform(X_test)





###practics
xgb_model=XGBRegressor(max_depth=6,n_estimators=1000,learning_rate=0.04,random_state=42)

xgb_model.fit(X_scaled,y)
y_pred_log = xgb_model.predict(X_test_scaled)
y_pred = np.expm1(y_pred_log)

y_pred_train_log = xgb_model.predict(X_scaled)
y_pred_train = np.expm1(y_pred_train_log)

rmsle = np.sqrt(mean_squared_log_error(np.expm1(y), y_pred_train))
print(f'RMSLE: {rmsle:.5f}')









model = LGBMRegressor(
    max_depth=6,
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=50,
    feature_fraction=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)


model.fit(X_scaled, y)


y_pred = np.expm1(model.predict(X_test_scaled))
y_train_pred = np.expm1(model.predict(X_scaled))


rmsle = np.sqrt(mean_squared_log_error(np.expm1(y), y_train_pred))
print(f'RMSLE: {rmsle:.5f}')


submission=pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
submission.head(10)


submission.shape


submission = pd.DataFrame({
    'id': submission['id'],
    'Calories': y_pred
})
submission.head(10)


submission.shape


submission.tail(10)


submission.info()


submission.to_csv('submission.csv', index=False)







