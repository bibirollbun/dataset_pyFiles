import pandas as pd 
import numpy as np
import polars as pl 
from datetime import datetime
import os 
pl.Config(tbl_rows=20)


DATA_PATH = '/kaggle/input/user-retention-prediction/'


act_logs = pl.read_csv(os.path.join(DATA_PATH, 'train.csv'))
sub = pl.read_csv(os.path.join(DATA_PATH, 'submit_sample.csv'))


act_logs = act_logs.with_columns(
    (pl.col('Timestamp')*1000)
    .cast(pl.Datetime(time_unit='ms', time_zone='UTC'))
    .dt.convert_time_zone('Asia/Shanghai')
)


act_logs = act_logs.with_columns(
    pl.col('Timestamp').dt.strftime('%Y-%m-%d').alias('Date').cast(pl.Date)
)
act_logs.head()


daily_act = act_logs.select(
    pl.col('ID'),
    pl.col('Date')
).unique().sort(by = ['ID','Date'])
print(daily_act.shape)
daily_act.head()


user_act_boundary = daily_act.group_by('ID').agg(
    pl.col('Date').min().alias('User_earliest_act_date'),
    pl.col('Date').max().alias('User_latest_act_date')
)
user_act_boundary.head()


# 标签是未来7天的活跃天数，从最后日期-7往后。未来日期不满7天的，视为无效标签
user_valid_date = user_act_boundary.with_columns(
    #v6改了这里
    User_valid_end_date = pl.col('User_latest_act_date').max() - pl.duration(days=7)
).with_columns(
    #有效日期范围
    User_valid_date = pl.date_ranges(
        start = pl.col('User_earliest_act_date'),
        end = pl.col('User_valid_end_date'),
        interval = '1d'
    )
#展开成长表并选择需要的列
).explode('User_valid_date').select('ID','User_valid_date')
user_valid_date.head()


# 为每个可能的有效日期生成 [t+1, t+7] 的未来日期范围
future_dates = user_valid_date.with_columns(
    Future_date = pl.date_ranges(
        start=pl.col("User_valid_date") + pl.duration(days=1),
        end=pl.col("User_valid_date") + pl.duration(days=7),
        interval="1d"
    )
  #展开日期列表
).explode("Future_date")
future_dates.head()


#v6新增一列辅助列，避免日期匹配的时候日期被吞了
daily_act = daily_act.with_columns(
    pl.lit(1).alias("IsActive")
)


future_dates_act = future_dates.join(
    daily_act,
    how = 'left',
    left_on = ['ID','Future_date'],
    right_on= ['ID','Date']
)
future_dates_act.head()


User_daily_labels = future_dates_act.group_by(['ID', 'User_valid_date']).agg([
    pl.col("IsActive").sum().fill_null(0).alias("Label")
])

daily_act = daily_act.join(
    User_daily_labels,
    how = 'left',
    left_on=['ID','Date'],
    right_on=['ID','User_valid_date']
).sort(by = ['ID','Date']).drop(['IsActive'])

daily_act.head()


#抽个样本看看是否准确
daily_act.filter(
    pl.col('ID') == 328258
).sort('Date')


sample_id = daily_act['ID'].sample(1)
daily_act.filter(
    pl.col('ID') == sample_id
).sort('Date')


# 检查数据量
print(daily_act.shape)




