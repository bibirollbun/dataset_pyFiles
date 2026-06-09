!pip install tslearn


!cp /kaggle/input/web-traffic-time-series-forecasting/train_1.csv.zip /kaggle/working/


!unzip /kaggle/working/train_1.csv.zip


from tslearn.clustering.kshape import KShape
import pandas as pd


data = pd.read_csv("/kaggle/working/train_1.csv")


data.dropna(how='any', inplace=True)


data.head()


page = data['Page']
data.drop(columns=['Page'], inplace=True)


page.to_csv("page_index.csv")


data.head()


data.shape


from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()


data = scaler.fit_transform(data)


data.shape


!pip install joblib


import joblib


joblib.dump(scaler, "my_standard_scaler.save")


data = data[:20000, :]


data.shape


model = KShape(n_clusters=24, max_iter=30, verbose=True, random_state=42)


data = data.reshape(data.shape[0], data.shape[1], 1)


model.fit(data)


model.to_pickle("kshape_model.pkl")




