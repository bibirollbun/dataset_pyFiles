!unzip playground-series-s5e5.zip


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
print(data.head(5))
print(data.shape)


from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

def feature_engeneering(df):
  if 'id' in df.columns:
    df.drop(columns = ['id'], inplace = True)
  if 'Sex' in df.columns:
    df['Sex'] = df['Sex'].map({'female': 1, 'male': 2})
  df['AgeSex'] = df['Age'].astype(str) + df['Sex'].astype(str)
  df['AgeSex'] = LabelEncoder().fit_transform(df['AgeSex']) + 1
  df['BMI'] = df['Weight'] / (df['Height'] ** 2)
  df['Heart_Rate_per_Age'] = df['Heart_Rate'] / df['Age']
  df['Temp_Heart_Interaction'] = df['Body_Temp'] * df['Heart_Rate']
  group_stats = df.groupby('Sex')['Heart_Rate'].agg(['mean', 'std']).rename(columns={'mean': 'Mean_HR_by_Sex', 'std': 'Std_HR_by_Sex'})
  df = df.merge(group_stats, on='Sex', how='left')
  df['HR_above_group_mean'] = df['Heart_Rate'] - df['Mean_HR_by_Sex']
  df['Log_Weight'] = np.log1p(df['Weight'])
  df['Sqrt_Height'] = np.sqrt(df['Height'])
  df['Age_squared'] = df['Age'] ** 2
  df['Weight_cubed'] = df['Weight'] ** 3

  features = ['Age', 'Weight', 'Height', 'Body_Temp', 'Heart_Rate', 'Duration', 'Sex', 'AgeSex']
  for comb in itertools.combinations(features, 2):
      df[" * ".join(comb)] = df[list(comb)].prod(axis=1)

  return df


data_test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
print(data_test.shape)


import itertools
df_train = feature_engeneering(data)
df_test = feature_engeneering(data_test)
print(f"Train: {df_train.shape}")
print(df_test.shape)


from catboost import CatBoostRegressor, Pool
import numpy as np
def split_dataset(x, test_size = 0.25):
  test_indeces = np.random.rand(len(x)) < test_size
  return x[~test_indeces], x[test_indeces]
train_ds_pd, valid_ds_pd = split_dataset(df_train, test_size = 0.25)
train_ds_x = train_ds_pd.drop(['Calories'], axis = 1)
train_ds_y = train_ds_pd['Calories']
valid_ds_x = valid_ds_pd.drop(['Calories'], axis = 1)
valid_ds_y = valid_ds_pd['Calories']
y_train_log = np.log1p(train_ds_y)
y_test_log = np.log1p(valid_ds_y)
train_pool = Pool(train_ds_x, y_train_log)
test_pool = Pool(valid_ds_x, y_test_log)
model = CatBoostRegressor(
    iterations = 9000,
    learning_rate = 0.002,
    depth = 16,
    loss_function = 'RMSE',
    eval_metric = 'RMSE',
    verbose = 100,
    early_stopping_rounds = 50,
    task_type='GPU',
    devices='0:1'
)

model.fit(train_pool, eval_set = test_pool)


y_pred_log = model.predict(df_test)
predictions = np.expm1(y_pred_log)
results = pd.DataFrame({
    'id': range(750000, 750000 + len(predictions)),
    'Calories': predictions.flatten()
})

results.to_csv('submission.csv', index = False)

