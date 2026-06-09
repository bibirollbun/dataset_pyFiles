import pandas as pd
import h2o
from h2o.automl import H2OAutoML
from itertools import combinations
from scipy.stats import gmean, hmean
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
h2o.init()


data_folder = "/kaggle/input/modified-data/"



df_train = pd.read_csv('/kaggle/input/modified-data/train (1).csv')
df_test  = pd.read_csv('/kaggle/input/modified-data/test.csv')
df_sub = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')


df_train.shape


df_train.shape


df_train.drop(columns=['id'], inplace=True)
df_test.drop(columns=['id'], inplace=True)


df_train.columns


df_train.head()


train_data = h2o.H2OFrame(df_train)


from h2o.frame import H2OFrame
with h2o.utils.threading.local_context(polars_enabled=True, datatable_enabled=True):
    pandas_df = train_data.as_data_frame()


import seaborn as sns
import matplotlib.pyplot as plt
sns.barplot(data=df_train, x='country', y='num_sold', estimator='sum')
sns.barplot(data=df_train, x='store', y='num_sold', estimator='sum')
sns.barplot(data=df_train, x='product', y='num_sold', estimator='sum')
plt.show()



top_products = df_train.groupby('product')['num_sold'].sum().sort_values(ascending=False)
top_products.plot(kind='bar', title="Top-Selling Products")
plt.show()


sns.boxplot(data=df_train, x='product', y='num_sold')
plt.show()



plt.figure(figsize=(10, 8))  
df_train.groupby('country')['num_sold'].sum().plot(
    kind='pie', 
    autopct='%1.1f%%', 
    title="Sales by Country"
)
plt.ylabel('')  
plt.show()



product_store_sales = df_train.pivot_table(index='product', columns='store', values='num_sold', aggfunc='sum')
sns.heatmap(product_store_sales, annot=True, cmap='Blues')
plt.show()



#pandas_df[feature]


train_data


test_data = h2o.H2OFrame(df_test)


aml = H2OAutoML(max_runtime_secs=590,seed=42)
aml.train(y='num_sold', training_frame=train_data)


leaderboard = aml.leaderboard
print(leaderboard)
best_model = aml.leader
print(best_model)


best_model = aml.leader



df_test = h2o.H2OFrame(df_test)


predictions = best_model.predict(df_test)
predictions_df = predictions.as_data_frame()


y_pred_original = np.expm1((predictions_df['predict'].values))  


y_pred_original


df_sub['num_sold'] =y_pred_original


df_sub.head()


df_sub.to_csv('submission.csv', index=False)




