import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score 
from sklearn.model_selection import cross_val_score


rainfall_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')


rainfall_df = rainfall_df.drop(columns= ["id", "day"])


rainfall_df


rainfall_df.info()


rainfall_df.describe()


rainfall_df.hist(grid=True, layout=(4, 3), figsize=(12, 8), bins=15)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


rainfall_df.corr()[["rainfall"]]


plt.figure(figsize=(8,6))
sns.heatmap(rainfall_df.corr()*100, annot=True, cmap='coolwarm_r', vmin=-100, vmax=100)
plt.show()


target = rainfall_df["rainfall"]
rainfall_df.drop(columns="rainfall", inplace=True)


rainfall_df.head(2)


rainfall_df["cosine"] = rainfall_df["winddirection"].apply(np.cos) 
rainfall_df["sine"] = rainfall_df["winddirection"].apply(np.sin) 


rainfall_df.drop(columns="winddirection", inplace=True)


rainfall_df.corr()*100





scaler = StandardScaler()


scaled_df = pd.DataFrame(scaler.fit_transform(rainfall_df), columns = scaler.get_feature_names_out())

scaled_df.head(3)


scaled_df.hist(layout=(4,3), figsize=(12,8))
plt.tight_layout(rect=[0,0,1,0.96])
plt.show()


log_reg= LogisticRegression(max_iter=500, penalty=None)
log_reg.fit(scaled_df, target)
log_reg_preds = log_reg.predict(scaled_df)
roc_auc_score(target, log_reg_preds)


cross_val_score(log_reg, scaled_df, target, scoring="accuracy", cv=10)


test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


test_df["sine"] = test_df["winddirection"].apply(np.sin)
test_df["cosine"] = test_df["winddirection"].apply(np.cos)
test_df.drop(columns=['id', 'day', 'winddirection'], inplace=True)


scaler.feature_names_in_


test_df = test_df[['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity',
       'cloud', 'sunshine', 'windspeed', 'cosine', 'sine']]


test_p_df = pd.DataFrame(scaler.transform(test_df), columns=scaler.get_feature_names_out())


test_p_df.fillna(0,inplace=True)


x = log_reg.predict_proba(test_p_df)[:,1]


x = pd.DataFrame(x)
x["id"] = range(2190,2920)


x.to_csv('submission.csv',index=False)




