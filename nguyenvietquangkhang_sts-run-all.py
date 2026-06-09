import collections

import numpy as np
import tensorflow.compat.v2 as tf
import tf_keras
import tensorflow_probability as tfp
import pandas as pd
from tensorflow_probability import sts
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')


tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)


inventory = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv').drop(['warehouse','product_unique_id'],axis=1)
calendar = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv', parse_dates=['date'])
train = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv', parse_dates=['date'])
test = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv', parse_dates=['date'])
submit = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/solution.csv')


test_id = test.unique_id.unique()

train = train[train['unique_id'].isin(test_id)]
train = train.sort_values(by='date').reset_index(drop=True)


train_merge = train.merge(calendar, left_on=['date','warehouse'],right_on=['date','warehouse'])
train_merge['discount'] = train_merge[['type_0_discount', 'type_1_discount', 'type_2_discount','type_3_discount','type_4_discount','type_5_discount','type_6_discount']].max(axis=1)
train_merge = train_merge.drop(['availability','type_0_discount', 'type_1_discount', 'type_2_discount','type_3_discount','type_4_discount','type_5_discount','type_6_discount'],axis=1)


train_data = train_merge[['unique_id','date','sales']].copy()

future_dates = pd.date_range(start='2024-06-03', periods=14, freq='D')


def process_df(train_data,i):
    df_tmp=train_data[train_data['unique_id']==i]
    date_range = pd.date_range(start=max(train_data['date'].min(),df_tmp['date'].min()), end=train_data['date'].max())
    df_tmp=df_tmp.sort_values(by='date')
    df_tmp = df_tmp.set_index('date').reindex(date_range)
    df_tmp['unique_id'] = df_tmp['unique_id'].fillna(i)
    df_tmp['sales'] = df_tmp['sales'].interpolate().fillna(0)
    #df_tmp['holiday'] = df_tmp['holiday'].fillna(0)
    #df_tmp['discount'] = df_tmp['discount'].fillna(0)
    #past_dates = future_dates - pd.DateOffset(years=1) 
    #future_holiday = df_tmp.loc[past_dates, 'holiday'].reset_index(drop=True)
    #future_discount = df_tmp.loc[past_dates, 'discount'].reset_index(drop=True)
    #extended_holiday = pd.concat([
    #    df_tmp['holiday'].reset_index(drop=True),
    #    future_holiday
    #], ignore_index=True)
    
    #extended_discount = pd.concat([
    #    df_tmp['discount'].reset_index(drop=True),
    #    future_discount
    #], ignore_index=True)
    #df_1id.reset_index()
    #df_1id['date'] = df_1id.index
    #df_1id['indexi'] = range(1, len(df_1id) + 1)
    #df_1id = df_1id.reset_index(drop=True)
    #df_1id = df_1id.drop(['indexi'],axis=1)
    return df_tmp
    


def build_model(observed_time_series):
    day_of_week_effect = sts.Seasonal(
        num_seasons=7, num_steps_per_season=1,
        observed_time_series=observed_time_series,
        name='day_of_week_effect'
    )

    autoregressive = sts.Autoregressive(
        order = 3,
        observed_time_series=observed_time_series,
        name='autoregressive'
    )

    white_noise = tfp.distributions.LogNormal(
        loc=tf.constant(0., dtype=tf.float64),
        scale=tf.constant(1., dtype=tf.float64)
    )

    local_level = sts.LocalLevel(observed_time_series = observed_time_series )
    
    model = sts.Sum(
        components=[autoregressive, day_of_week_effect, local_level],
        observed_time_series=observed_time_series,
        observation_noise_scale_prior=white_noise
    )

    return model


def predict(model):

    variational_posteriors = tfp.sts.build_factored_surrogate_posterior(
    model=model)
    num_variational_steps = 130 
    num_variational_steps = int(num_variational_steps)
# Build and optimize the variational loss function.
    elbo_loss_curve = tfp.vi.fit_surrogate_posterior(
        target_log_prob_fn=model.joint_distribution(
        observed_time_series=df_tmp['sales']).log_prob,
        surrogate_posterior=variational_posteriors,
        optimizer=tf_keras.optimizers.Adam(learning_rate=0.1),
        num_steps=num_variational_steps,
        jit_compile=False)
    q_samples_demand_ = variational_posteriors.sample(50)
    forecast_dist = tfp.sts.forecast(
    model=model,
    observed_time_series=df_tmp['sales'],
    parameter_samples=q_samples_demand_,
    num_steps_forecast=14)
    num_samples=30
    forecast_mean = forecast_dist.mean().numpy()[..., 0]
    return forecast_mean


def submit_process(p):
    p['unique_id'] = p['unique_id']
    p['solution_id'] = p['unique_id'].astype(str)+'_'+p['ds'].astype(str)
    solution_id_to_sales_hat = p.set_index('solution_id')['predict_values']
    submit['sales_hat'] = submit['id'].map(solution_id_to_sales_hat).fillna(submit['sales_hat'])


for _,i in enumerate(test_id[1500:1600]):
    print(f'Predict for UniqueID {i}...')
    df_tmp = process_df(train_data,i)    
    model = build_model(df_tmp['sales'])
    forecast_mean = predict(model)
    ds = future_dates
    df_predict = pd.DataFrame({
    'unique_id': i,
    'ds': ds,
    'predict_values': forecast_mean
    })
    submit_process(df_predict)


submit.to_csv('submit.csv',index=False)




