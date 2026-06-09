import collections

import numpy as np
from scipy import stats
from scipy import special
import tensorflow.compat.v2 as tf
import tf_keras
import tensorflow_probability as tfp
import pandas as pd
from tensorflow_probability import sts
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')
import gc


#inventory = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv').drop(['warehouse','product_unique_id'],axis=1)
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
test['discount'] = test[['type_0_discount', 'type_1_discount', 'type_2_discount','type_3_discount','type_4_discount','type_5_discount','type_6_discount']].max(axis=1)
test = test.drop(['type_0_discount', 'type_1_discount', 'type_2_discount','type_3_discount','type_4_discount','type_5_discount','type_6_discount'],axis=1)


train_data = train_merge[['unique_id','date','sales','discount']].copy()
future_dates = pd.date_range(start='2024-06-03', periods=14, freq='D')
del train_merge, train
print('-----------')
gc.collect()
print('========================')


# Fourier transform
train_data['day_of_year'] = train_data['date'].dt.dayofyear
train_data['cos_day'] = np.cos(train_data['day_of_year']*2*np.pi/365)
train_data['sin_day'] = np.sin(train_data['day_of_year']*2*np.pi/365)
test['day_of_year'] = test['date'].dt.dayofyear
test['cos_day'] = np.cos(test['day_of_year']*2*np.pi/365)
test['sin_day'] = np.sin(test['day_of_year']*2*np.pi/365)
# sqrt tranform
train_data['sales'] = np.sqrt(train_data['sales'])


def process_df(unique_id):
    df_tmp=train_data[train_data['unique_id']==unique_id]
    date_range = pd.date_range(start=max(train_data['date'].min(),df_tmp['date'].min()), end='2024-06-02')
    df_tmp=df_tmp.sort_values(by='date')
    df_tmp = df_tmp.set_index('date').reindex(date_range)
    df_tmp['date'] = df_tmp.index
    df_tmp['day_of_year']=df_tmp['day_of_year'].fillna(df_tmp['date'].dt.dayofyear)
    df_tmp['unique_id'] = df_tmp['unique_id'].fillna(i)
    df_tmp['sales'] = df_tmp['sales'].fillna(0)
    df_tmp['discount'] = df_tmp['discount'].fillna(0) 
    df_tmp['sin_day'] = df_tmp['sin_day'].fillna(np.cos(df_tmp['day_of_year']*2*np.pi/365))
    df_tmp['cos_day'] = df_tmp['cos_day'].fillna(np.cos(df_tmp['day_of_year']*2*np.pi/365))     
    future_df = test[test['unique_id'] == i].copy()
    future_df=future_df.sort_values(by='date')
    future_df = future_df.set_index('date').reindex(future_dates)
    future_discount = future_df['discount'].reset_index(drop=True).fillna(0)    
    sin_future = future_df['sin_day'].reset_index(drop=True).fillna(0)
    cos_future = future_df['cos_day'].reset_index(drop=True).fillna(0)
    extended_sinday = pd.concat([
        df_tmp['sin_day'].reset_index(drop=True),
        sin_future
    ], ignore_index=True)    
    extended_cosday = pd.concat([
        df_tmp['cos_day'].reset_index(drop=True),
        cos_future
    ], ignore_index=True)    
    extended_discount = pd.concat([
        df_tmp['discount'].reset_index(drop=True),
        future_discount
    ], ignore_index=True)
    return df_tmp,extended_discount, extended_sinday, extended_cosday
    


def build_model(observed_time_series, discount, sin_day, cos_day):
    
    day_of_week_effect = sts.Seasonal(
        num_seasons=7, num_steps_per_season=1,
        observed_time_series=observed_time_series,
        name='day_of_week_effect')
    white_noise = tfp.distributions.LogNormal(
        loc=tf.constant(0., dtype=tf.float64),
        scale=tf.constant(1., dtype=tf.float64))    
    local_level = sts.LocalLevel(observed_time_series = observed_time_series)
    external_features = tf.stack([ tf.reshape(discount, [-1]), tf.reshape(sin_day, [-1]), 
                                   tf.reshape(cos_day, [-1])], axis=-1)
    external_regressor = sts.LinearRegression(
        design_matrix=tf.cast(external_features, dtype=tf.float64),
        weights_prior = tfp.distributions.Normal(tf.constant(0., dtype=tf.float64), 
                                                 tf.constant(1., dtype=tf.float64)),
        name='external_features')    
    autoregressive = sts.Autoregressive(
        order=3,
        observed_time_series=observed_time_series,
        name='autoregressive')    
    model = sts.Sum(
        components=[day_of_week_effect, autoregressive, local_level, external_regressor],
        observed_time_series=observed_time_series,
        observation_noise_scale_prior=white_noise)

    return model


def predict(model, observed_time_series):
# Xây dựng hàm xấp xỉ q(θ)
    variational_posteriors = tfp.sts.build_factored_surrogate_posterior(model=model)
    num_variational_steps = 200 # Số vòng lặp chạy tối ưu
    num_variational_steps = int(num_variational_steps)
# Xây dựng hàm ELBO loss và tối ưu.
    elbo_loss_curve = tfp.vi.fit_surrogate_posterior(
        target_log_prob_fn=model.joint_distribution(observed_time_series=observed_time_series).log_prob,
        surrogate_posterior=variational_posteriors,
        optimizer=tf_keras.optimizers.Adam(learning_rate=0.1),
        num_steps=num_variational_steps,
        jit_compile=False)
# Lấy mẫu các tham số từ phân phối hậu nghiệm đã xấp xỉ
    q_samples_demand_ = variational_posteriors.sample(50)
# Dự đoán 14 ngày tiếp theo
    forecast_dist = tfp.sts.forecast(
        model=model,
        observed_time_series=observed_time_series,
        parameter_samples=q_samples_demand_,
        num_steps_forecast=14)
        num_samples=300
    (
        forecast_mean,
        forecast_scale,
        forecast_samples
    ) = (
        forecast_dist.mean().numpy()[..., 0],
        forecast_dist.stddev().numpy()[..., 0],
        forecast_dist.sample(num_samples).numpy()[..., 0] # Lấy mẫu từ phân phối dự đoán
        )    
    return forecast_mean


def submit_process(p):
    p['unique_id'] = p['unique_id']
    p['solution_id'] = p['unique_id'].astype(str)+'_'+p['ds'].astype(str)
    solution_id_to_sales_hat = p.set_index('solution_id')['predict_values']
    # Update df1['sales_hat'] based on the mapping, leaving unmatched rows intact
    submit['sales_hat'] = submit['id'].map(solution_id_to_sales_hat).fillna(submit['sales_hat'])


for _,i in enumerate(test_id[3620:]):
    print(f'Predict for UniqueID {i}...')
    df_tmp,extended_discount,sin_day, cos_day = process_df(i)
    observed_time_series = df_tmp['sales']
    model = build_model(observed_time_series, extended_discount,sin_day, cos_day)
    forecast_mean = predict(model, observed_time_series)
    ds = future_dates
    df_predict = pd.DataFrame({
        'unique_id': i,
        'ds': ds,
        'predict_values': forecast_mean**2
    })
    submit_process(df_predict)


submit.to_csv('submit.csv',index=False)








