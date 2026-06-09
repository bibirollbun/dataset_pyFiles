# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install autogluon.tabular


!pip install duckdb


import pandas as pd 
import duckdb as db
from autogluon.tabular import TabularPredictor
from sklearn.metrics import root_mean_squared_error
import numpy as np
import matplotlib.pyplot as plt


sales = pd.read_csv('/kaggle/input/ml-zoomcamp-2024-competition/sales.csv', parse_dates=['date'])
online = pd.read_csv('/kaggle/input/ml-zoomcamp-2024-competition/online.csv', parse_dates=['date'])
test = pd.read_csv('/kaggle/input/ml-zoomcamp-2024-competition/test.csv', parse_dates=['date'], sep=';')


combined_sales = db.sql(f"""
    with date_dim as (
        select distinct date 
        from (
            select date from sales 
            union 
            select date from online
        )
    ),
    store_item as (
        select distinct 
            store_id,
            item_id
        from (
            select store_id, item_id from sales
            union
            select store_id, item_id from online
        )
    ),
    date_store_item_cross as (
        select 
            d.date,
            si.store_id,
            si.item_id
        from date_dim d
        cross join store_item si
    ),
    daily_sales as (
        select 
            dsi.date,
            dsi.item_id,
            extract(month from dsi.date) as month,
            extract(year from dsi.date) as year,
            extract(day from dsi.date) as day,
            extract(week from dsi.date) as week,
            case 
                when extract(dow from dsi.date) in (0, 6) then 'weekend'
                else 'weekday'
            end as day_type,
            1 as sample_weight,
            extract(dow from dsi.date) as day_of_week,
            dsi.store_id,
            coalesce(a.quantity, 0) + coalesce(b.quantity, 0) as quantity
        from date_store_item_cross dsi
        left join sales as a 
            on dsi.date = a.date
            and dsi.item_id = a.item_id
            and dsi.store_id = a.store_id
        left join online as b 
            on dsi.date = b.date
            and dsi.item_id = b.item_id
            and dsi.store_id = b.store_id
    )
    
    select 
        *,
        coalesce(avg(quantity) over (
            partition by store_id, item_id
            order by date 
            rows between 7 preceding and 1 preceding
        ), 0) as rolling_7day_avg,
        coalesce(avg(quantity) over (
            partition by store_id, item_id
            order by date 
            rows between 14 preceding and 1 preceding
        ), 0) as rolling_14day_avg,
        coalesce(avg(quantity) over (
            partition by store_id, item_id
            order by date 
            rows between 30 preceding and 1 preceding
        ), 0) as rolling_30day_avg
    from daily_sales
    order by date
""").df()


# Example
combined_sales[(combined_sales['store_id'] == 1) & (combined_sales['item_id'] == 'b0d24502fb66')].head(10).T


test = db.sql("""
    select 
        a.row_id,
        a.item_id,
        a.store_id,
        extract(month from a.date) as month,
        extract(year from a.date) as year,
        extract(day from a.date) as day,
        extract(week from a.date) as week,
        extract(dayofweek from a.date) as day_of_week,
        coalesce(avg(b.rolling_7day_avg), 0) as rolling_7day_avg,
        coalesce(avg(b.rolling_14day_avg), 0) as rolling_14day_avg,
        coalesce(avg(b.rolling_30day_avg), 0) as rolling_30day_avg
    from test as a 
    left join combined_sales as b  
        on a.item_id = b.item_id
        and a.store_id = b.store_id
        and b.week in (38,39)
        and b.year = 2024
    group by 1,2,3,4,5,6,7,8 
""").df()
test.head()


features = ['item_id', 
            'store_id', 
            'day', 
            'week', 
            'day_of_week',
            'rolling_7day_avg',	
            'rolling_14day_avg',	
            'rolling_30day_avg', 
            'quantity',]

train = combined_sales[features]
train.shape



train = db.sql("""
    WITH quantiles AS (
        SELECT 
            item_id,
            store_id,
            day_of_week,
            percentile_cont(0.01) WITHIN GROUP (ORDER BY quantity) as q01,
            percentile_cont(0.99) WITHIN GROUP (ORDER BY quantity) as q99
        FROM train
        GROUP BY item_id, store_id, day_of_week
    )
    SELECT t.*
    FROM train t
    JOIN quantiles q 
        ON t.item_id = q.item_id 
        AND t.store_id = q.store_id
        AND t.day_of_week = q.day_of_week
    WHERE t.quantity > q.q01 
        AND t.quantity < q.q99
""").df()

train.shape



predictor = TabularPredictor(label='quantity', eval_metric='root_mean_squared_error')


tree_hyperparameters = {
    'GBM': {},   
    'XGB': {},   
    'RF': {},    
    'XT': {},    
    'CAT': {}    
}


predictor.fit(train, 
              presets='medium_quality', 
              time_limit=60*15, 
              verbosity=2, 
              hyperparameters = tree_hyperparameters)


print(predictor.feature_importance(train))


features_test = ['item_id', 
            'store_id', 
            'day', 
            'week', 
            'day_of_week',
            'rolling_7day_avg',	
            'rolling_14day_avg',	
            'rolling_30day_avg', 
            ]

test[features_test].head()


# Make predictions
test['quantity'] = predictor.predict(test[features_test])
test.head()


submission = pd.read_csv('/kaggle/input/ml-zoomcamp-2024-competition/sample_submission.csv')
submission.head()


# join on item_id, store_id, date
submission = db.sql("""
    select 
        a.row_id,
        b.quantity
    from submission as a
    left join test as b
        on a.row_id = b.row_id
""").df()
submission.head(), submission.shape


# Create submission file
submission.to_csv('submission_autogluon_best_recent_min_with_rolling_averages.csv', index=False)


submission.info()

