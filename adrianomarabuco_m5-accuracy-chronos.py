# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm.notebook import tqdm, trange

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')
train_val = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')
sell_prices = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')
train_eval = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv')


calendar.head(10)


len(calendar)


print(calendar.head(1).date)
print(calendar.tail(1).date)


train_val.head()


train_val.columns[-1].split('_')[1]


1969 - 1913


print(len(train_val), 'séries  temporais')


train_val[train_val.columns[:6]].describe()


sell_prices.head()


train_eval.head()


!pip install git+https://github.com/amazon-science/chronos-forecasting.git


import torch
from chronos import BaseChronosPipeline

pipeline1 = BaseChronosPipeline.from_pretrained(
    # "amazon/chronos-bolt-base",  # use "amazon/chronos-bolt-small" for the corresponding Chronos-Bolt model
    # "amazon/chronos-bolt-base",
    "amazon/chronos-t5-base", 
    device_map="cuda",  # use "cpu" for CPU inference
    torch_dtype=torch.bfloat16,
)


help(pipeline1)


def forecast(X):
    # context must be either a 1D tensor, a list of 1D tensors,
    # or a left-padded 2D tensor with batch as the first dimension
    # quantiles is an fp32 tensor with shape [batch_size, prediction_length, num_quantile_levels]
    # mean is an fp32 tensor with shape [batch_size, prediction_length]
    return pipeline1.predict(
        inputs=torch.tensor(X),
        prediction_length=56,
        num_samples=1
    )


row[1].iloc[-365:]


pipeline1.predict


predictions = []
for row in train_val.iterrows():
    item = row[1].iloc[0]
    # print(row[1].iloc[-365:])
    prediction = forecast(row[1].iloc[-365:])
    print(prediction)
    # predictions.append({item:prediction})
    break


pd.DataFrame(prediction[0][0]).plot()


tmpdf = pd.concat([pd.DataFrame(prediction[0][0]), train_eval.iloc[0,-56:].reset_index(drop=True)],axis=1)
tmpdf.plot()


pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-base",  # use "amazon/chronos-bolt-small" for the corresponding Chronos-Bolt model
    device_map="cuda",  # use "cpu" for CPU inference
    torch_dtype=torch.bfloat16,
)


help(pipeline)


def forecastq(X):
    # context must be either a 1D tensor, a list of 1D tensors,
    # or a left-padded 2D tensor with batch as the first dimension
    # quantiles is an fp32 tensor with shape [batch_size, prediction_length, num_quantile_levels]
    # mean is an fp32 tensor with shape [batch_size, prediction_length]
    return pipeline.predict_quantiles(
        inputs=torch.tensor(X),
        prediction_length=56,
        quantile_levels=[0.1, 0.5, 0.9],
    )
    
    # quantiles, mean = pipeline.predict(
    #     context=torch.tensor(X),
    #     prediction_length=56,
    #     # quantile_levels=[0.1, 0.5, 0.9],
    # )
    # return quantiles, mean


predictions = []
for row in train_val.iterrows():
    item = row[1].iloc[0]
    # print(row[1].iloc[-365:])
    prediction = forecastq(row[1].iloc[-365:])
    print(prediction)
    # predictions.append({item:prediction})
    break


prediction[0][:,0] # quantil 0.05
prediction[0][:,1] # quantil 0.5
prediction[0][:,2] # quantil 0.95


prediction[0].shape


pd.DataFrame(prediction[0].reshape(3,-1)).T.plot(figsize=(20,5))


pd.DataFrame(prediction[0][0]).mean(axis=1).plot()


prediction[1][0]
pd.Series(prediction[1][0]).plot()


tmpdf = pd.concat([pd.DataFrame(prediction[0][0]).mean(axis=1), pd.DataFrame(prediction[0][0][:,1]), pd.Series(prediction[1][0])],axis=1)
tmpdf.columns = ['mean_quantiles', 'median', 'mean']
tmpdf.plot()
pd.concat([pd.DataFrame(prediction[0][0]).mean(axis=1), pd.DataFrame(prediction[0][0][:,1]), pd.Series(prediction[1][0])],axis=1, names=[0,1,2]).plot()


# del X
# torch.cuda.empty_cache()  # If using GPU


X = torch.tensor(train_val.iloc[:500,-365:].values, dtype=torch.bfloat16)


from torch.utils.data import DataLoader, TensorDataset


loader = DataLoader(X, batch_size=100, pin_memory=True)
predictions = []

with torch.no_grad():
    for batch in tqdm(loader):
        batch_preds = forecast(batch)
        predictions.append(batch_preds)

predictions = torch.cat(predictions, dim=0)


predictions2 = forecastq(X)


pd.DataFrame(predictions2[0][:][99]).T.plot()


result = pd.DataFrame(predictions[:][:].reshape(-1,56))



result.head()


import numpy as np

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


rmse(result.values, train_eval.iloc[:500,-56:].values)


comp = pd.concat([result.iloc[499], train_eval.iloc[499,-56:].reset_index(drop=True)], axis=1)
comp.columns = ['pred','true']
comp.plot()


# context must be either a 1D tensor, a list of 1D tensors,
# or a left-padded 2D tensor with batch as the first dimension
context = torch.tensor(train_val.iloc[499][-365:])
embeddings, tokenizer_state = pipeline.embed(context)


context


pd.Series(context).plot()


embeddings.shape


embeddings.float().numpy()


dfe = pd.DataFrame(embeddings.float().numpy().reshape(-1,768))


from sklearn.decomposition import PCA
import plotly.express as px

# Criar PCA com 3 componentes
pca = PCA(n_components=3)
xe = pca.fit_transform(dfe)

# Criar DataFrame com as 3 componentes
df_pca = pd.DataFrame(xe, columns=['PC1', 'PC2', 'PC3'])

# Criar gráfico 3D
fig = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3')
fig.show()


!pip install -U autogluon transformers accelerate


from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame


df_long = pd.melt(train_val, 
                  id_vars=['id'],     # colunas que identificam cada registro
                  value_vars=train_val.columns[-365:], # colunas que serão "empilhadas"
                  var_name='timestamp',    # nome da nova coluna com os nomes das variáveis
                  value_name='target') 


days = {r[0]:r[1] for r in calendar[['d','date']].values}


df_long.timestamp = df_long.timestamp.apply(lambda x: days[x])
df_long.timestamp = pd.to_datetime(df_long.timestamp)


df_long.rename({"id":"item_id"}, axis=1, inplace=True)


df_long.info()


df_long


data = TimeSeriesDataFrame.from_data_frame(df_long)
data.head()


prediction_length = 56
# train_data, test_data = data.train_test_split(prediction_length)

predictor = TimeSeriesPredictor(prediction_length=prediction_length).fit(
    df_long, presets="bolt_small",
)


predictions = predictor.predict(data)
predictor.plot(
    data=data,
    predictions=predictions,
    item_ids=data.item_ids[:2],
    max_history_length=500,
);

