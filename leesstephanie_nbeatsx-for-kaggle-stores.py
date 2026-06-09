import warnings
warnings.filterwarnings(action='ignore')


!pip install neuralforecast


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import neuralforecast
from neuralforecast import NeuralForecast
from neuralforecast.tsdataset import TimeSeriesDataset
from neuralforecast.models import NBEATSx, NBEATS
from neuralforecast.losses.pytorch import SMAPE
import torch

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df = pd.read_csv('/kaggle/input/tabular-playground-series-jan-2022/train.csv', parse_dates=['date'])
df_test = pd.read_csv('/kaggle/input/tabular-playground-series-jan-2022/test.csv', parse_dates=['date'])
holidays_exog = pd.read_csv('/kaggle/input/holidays-finland-norway-sweden-20152019/Holidays_Finland_Norway_Sweden_2015-2019.csv',
                            parse_dates=['Date'])


df.head()


df.describe(include='all')


df = df.sort_values(['product', 'store', 'country'])
df_test = df_test.sort_values(['product', 'store', 'country'])


df.head(20)


g = sns.FacetGrid(df, col='store', row='country', hue='product', hue_kws={'color': ['#008e65', '#7a48c6', '#033d94']}, 
                  sharex=True, height=3, aspect=2)
g.map_dataframe(sns.lineplot, x='date', y='num_sold')
g.add_legend();


plt.figure(figsize=(16,3), dpi=300)
plt.plot(df.loc[(df['product']=='Kaggle Hat') & (df.store=='KaggleRama') & (df.country=='Norway'), 'num_sold'], color='#008e65');
# plt.savefig('HatRamaNorway.jpg')


plt.figure(figsize=(16,3), dpi=300)
plt.plot(df.loc[(df['product']=='Kaggle Mug') & (df.store=='KaggleMart') & (df.country=='Sweden'), 'num_sold'], color='#7a48c6');
# plt.savefig('MugMartSweden.jpg')


plt.figure(figsize=(16,3), dpi=300)
plt.plot(df.loc[(df['product']=='Kaggle Sticker') & (df.store=='KaggleMart') & (df.country=='Finland'), 'num_sold'], color='#033d94');
# plt.savefig('StickerMartFinland.jpg')


df['product'] = df['product'].str.removeprefix('Kaggle ')
df['store'] = df['store'].str.removeprefix('Kaggle')

df_test['product'] = df_test['product'].str.removeprefix('Kaggle ')
df_test['store'] = df_test['store'].str.removeprefix('Kaggle')


holidays_exog


holidays_exog['holiday_code'] = 1
holidays_exog


df = df.merge(holidays_exog, how='left', left_on=['date', 'country'], right_on=['Date', 'Country']).fillna(0)
df_test = df_test.merge(holidays_exog, how='left', left_on=['date', 'country'], right_on=['Date', 'Country']).fillna(0)


df


thedata = pd.DataFrame({
    'ds': df['date'],
    'unique_id': df[['product', 'store', 'country']].apply(lambda row: '-'.join(row), axis=1),
    'holiday': df['holiday_code'],
    'y': df['num_sold']})

thedata_t = thedata.loc[thedata.ds.dt.year < 2018]
thedata_v = thedata.loc[thedata.ds.dt.year == 2018]


exog_df_v = thedata_v[['ds', 'unique_id', 'holiday']]
exog_df_test = pd.DataFrame({'ds': df_test['date'],
                             'unique_id': df_test[['product', 'store', 'country']].apply(lambda row: '-'.join(row), axis=1),
                             'holiday': df_test['holiday_code']})


exog_df_test


nbeatsxI = NBEATSx(h=365, input_size=365, futr_exog_list=['holiday'], n_polynomials=2, n_harmonics=12,
                   stack_types=['trend', 'seasonality', 'exogenous'], max_steps=1000, loss=SMAPE(), alias='NBEATSx-I')


nbeatsxG = NBEATSx(h=365, input_size=365, futr_exog_list=['holiday'], 
                   stack_types=['identity', 'exogenous'], max_steps=1000, loss=SMAPE(), alias='NBEATSx-G')


nbeatsxA = NBEATSx(h=365, input_size=365, futr_exog_list=['holiday'], n_polynomials=2, n_harmonics=12,
                   stack_types=['trend', 'seasonality', 'exogenous', 'identity'], n_blocks=[1,1,1,1],
                   mlp_units=[[512,512], [512, 512], [512, 512], [512, 512]],
                   max_steps=1000, loss=SMAPE(), alias='NBEATSx-A')


model = NeuralForecast([nbeatsxI, nbeatsxG, nbeatsxA], freq='D')


# help(nbeatsxI)


# help(model)


# these two are equivalent
#model.fit(thedata_t, static_df=pddata_t[['unique_id', 'holiday']])
model.fit(thedata_t)


predictions = model.predict(futr_df=exog_df_v, verbose=True)
predictions


predictions.unique_id.unique()


# go through the predictions from each model
scores = {}
smape_scorer = SMAPE()
for col in range(2, predictions.shape[1]):
    
    # take the prediction and real values for all time series from a column (model)
    all_preds = predictions.iloc[:, col].copy()
    real_vals = thedata_v['y'].copy().reset_index(drop=True)
    
    # prepare a dictionary to store the scores
    scores[all_preds.name] = []
    
    # take each prediction and compute the SMAPE
    for i in range(18):
        pred_val = all_preds.iloc[:365].to_numpy()
        real_val = real_vals.iloc[:365].to_numpy()
        all_preds = all_preds.drop(np.arange(365)).reset_index(drop=True)
        real_vals = real_vals.drop(np.arange(365)).reset_index(drop=True)
        smape = smape_scorer(torch.from_numpy(pred_val), torch.from_numpy(real_val))
        scores[all_preds.name].append(smape)


mean_scores = [np.round(np.mean(arr), 3) for arr in scores.values()]


plt.plot(scores['NBEATSx-I'], color='#5D4F46', linestyle='--', label='NBEATSx-I')
plt.plot(scores['NBEATSx-G'], color='#5D4F46', linestyle=':', label='NBEATSx-G')
plt.plot(scores['NBEATSx-A'], color='#5D4F46', label='NBEATSx-A')
plt.legend()
plt.title(f"""average SMAPE-I: {str(mean_scores[0])}
average SMAPE-G: {str(mean_scores[1])}
average SMAPE-A: {str(mean_scores[2])}""");


modelname = 'NBEATSx-A'
smapes = scores[modelname]
_, axs = plt.subplots(nrows=6, ncols=3, sharex=True, figsize=(12,9), tight_layout=True)
unique_ids = predictions['unique_id'].unique()
for i, ax in enumerate(axs.flat):
    # taking the i-th ground truth time series
    real_val = thedata_v['y'].iloc[(i*365):((i+1)*365)]

    # taking the i-th prediction series
    pred_val = predictions[modelname].iloc[(i*365):((i+1)*365)]

    # plotting
    ax.plot(np.arange(365), real_val, color='#553a75', label='ground truth')
    ax.plot(np.arange(365), pred_val, color='#f173d8', label='forecast')
    ax.set_title(f'{unique_ids[i]} SMAPE: {str(np.round(smapes[i], 3))}');
ax.legend();


model = NeuralForecast([nbeatsxI, nbeatsxG, nbeatsxA], freq='D')
model.fit(thedata)
theprediction = model.predict(futr_df=exog_df_test)


theprediction


modelname = 'NBEATSx-I'
thedata = thedata[['ds', 'unique_id', 'y']]
thedata = pd.concat([thedata, theprediction[['ds', 'unique_id', modelname]].rename(columns={modelname: 'y'})])
thedata['forecast'] = thedata['ds'].dt.year == 2019


g = sns.FacetGrid(thedata, col='unique_id', hue='forecast', hue_kws={'color': ['grey', '#9064aa']}, 
                  col_wrap=3, aspect=1.5, sharey=False)
g.map_dataframe(sns.lineplot, x='ds', y='y')
plt.title(modelname);


submission = pd.read_csv('/kaggle/input/tabular-playground-series-jan-2022/sample_submission.csv')
submission.head()


def create_submission_file(model_name):
    df_test['num_sold'] = theprediction[model_name]
    df_test.sort_values('row_id', ignore_index=True, inplace=True)
    submission['num_sold'] = df_test['num_sold']
    submission.to_csv(f'submission-{model_name}.csv', index=False)
    return print('The file has been created.')

for name in ['NBEATSx-I', 'NBEATSx-G', 'NBEATSx-A']:
    create_submission_file(name)
    df_test.sort_values(['product', 'store', 'country'], inplace=True)

