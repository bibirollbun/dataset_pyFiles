import numpy as np
import pandas as pd
import plotly.express as px

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


calendar = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv')


calendar.head(5)


sorted(calendar['date'].str.split('-').apply(lambda x: x[0]).unique())


print('No. of Years: ',len(sorted(calendar['date'].str.split('-').apply(lambda x: x[0]).unique())))


fig = px.histogram(calendar, x="warehouse", y=["holiday", 'shops_closed'], barmode='group', height=400)
fig.show()


calendar[(calendar['warehouse']=='Frankfurt_1')  & (calendar['shops_closed']==1)].sort_values('date')


calendar[(calendar['shops_closed']==1) & (pd.isna(calendar['holiday_name']))].sort_values('date')


calendar[(calendar['warehouse']=='Frankfurt_1') & (calendar['school_holidays']==1)].sort_values('date')


calendar[(calendar['warehouse']=='Frankfurt_1') & (calendar['winter_school_holidays']==1)].sort_values('date')




