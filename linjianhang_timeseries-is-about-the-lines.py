import pandas as pd 
import numpy as np
import polars as pl 

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

import warnings 
warnings.filterwarnings("ignore")

import os 

from datetime import datetime


from IPython.display import display, HTML

def display_opinion_box(opinion_text):
    html_content = f"""
    <div style="background-color: #e6f7ff; border-radius: 10px; padding: 15px; border: 1px solid ; color:#d35400;">
      <strong>ğŸ’¡ Insights <br></strong> {opinion_text}
    </div>
    """
    display(HTML(html_content))


DATA_PATH = '/kaggle/input/user-retention-prediction'


act_logs = pl.read_csv(os.path.join(DATA_PATH, 'train.csv'))
sub = pl.read_csv(os.path.join(DATA_PATH, 'submit_sample.csv'))


act_logs.head()


act_logs.schema


act_logs.shape


act_logs.null_count()


act_logs.select(pl.all().n_unique())


display_opinion_box("""
1.æ“�ä½œæ—¥å¿—ä¸€å…±å°†è¿‘å…«å�ƒä¸‡è¡Œä¹‹å¤š(78793386), 4åˆ—ï¼šç”¨æˆ·idï¼Œæ“�ä½œçš„æ—¶é—´æˆ³ï¼Œæ“�ä½œç±»å�‹ï¼Œæ“�ä½œid<br>
2.æ²¡æœ‰ä»»ä½•ç©ºå€¼ã€‚<br>
3.ä¸€å…±2144882ä¸ªç”¨æˆ·ï¼Œ5ç§�æ“�ä½œç±»å�‹ï¼Œ130ç§�æ“�ä½œidã€‚<br>
4.æ—¶é—´æˆ³æ˜¯æŒ‰UTCè®°å½•çš„ï¼Œè¦�è½¬æ—¥æœŸæ³¨æ„�+8åˆ°åŒ—äº¬æ—¶é—´ã€‚<br>
5.æ“�ä½œç±»å�‹ï¼Œæ“�ä½œidåº”è¯¥éƒ½æ˜¯ç±»åˆ«å�˜é‡�ï¼Œç›¸å¯¹å¤§å°�æ— æ„�ä¹‰ï¼Œå€¼çš„æ„�ä¹‰ä¸�å…·ä½“ã€‚<br>
*6.æ²¡æœ‰ç”¨æˆ·æ ‡ç­¾ï¼Œéœ€è¦�è‡ªå·±é€ ã€‚<br>
""")


act_logs = act_logs.with_columns(
    (pl.col('Timestamp')*1000)
    .cast(pl.Datetime(time_unit='ms', time_zone='UTC'))
    .dt.convert_time_zone('Asia/Shanghai')
)


act_logs.describe()


act_logs = act_logs.with_columns(
    pl.col('ID').cast(pl.Utf8).str.len_chars().alias('ID_len')
)


act_logs.head()


act_logs['ID_len'].unique()


act_logs['ID'].unique().sort()


start_id = 0
end_id = 2237391
consecutive_id_list = pl.DataFrame().select(
    pl.int_range(start_id, end_id + 1, dtype=pl.Int32).alias("ID")
)


lacking_id = pl.DataFrame(
    {'ID': list(set(consecutive_id_list['ID']) - set(act_logs['ID']))}
    ).with_columns(
    pl.col('ID').cast(pl.Utf8).str.len_chars().alias('ID_len')
)
lacking_id.shape


2237391-2144882 + 1


missing_count = lacking_id.group_by("ID_len").agg(pl.count().alias("count"))

max_id = 2237391
possible_counts = pl.DataFrame({
    "ID_len": [2, 3, 4, 5, 6, 7],  # ID çš„é•¿åº¦èŒƒå›´
    "total_possible": [
        99 - 10 + 1,          # 2 ä½�æ•°: 10-99
        999 - 100 + 1,        # 3 ä½�æ•°: 100-999
        9999 - 1000 + 1,      # 4 ä½�æ•°: 1000-9999
        99999 - 10000 + 1,    # 5 ä½�æ•°: 10000-99999
        999999 - 100000 + 1,  # 6 ä½�æ•°: 100000-999999
        max_id - 1000000 + 1  # 7 ä½�æ•°: 1000000-2237391
    ]
}).with_columns(pl.col("ID_len").cast(pl.UInt32))  

# è®¡ç®—ç¼ºå¤± ID çš„å� æ¯”
result = missing_count.join(possible_counts, on="ID_len").with_columns(
    (pl.col("count") / pl.col("total_possible")).alias("missing_ratio")
)

result.head(10)


display_opinion_box("""
1.æ ·æœ¬æ˜¯ä»�0åˆ°2237391æŠ½æ ·æ�¥çš„ä¸�è¿�ç»­æ•´æ•°åˆ—<br>
2.ç¼ºå¤±çš„92510ä¸ªç”¨æˆ·ï¼Œå¾ˆå¤§çš„é‡�åœ¨6ã€�7ä½�æ•°idé‡Œé�¢(å› ä¸ºä½�æ•°è¶Šå¤šï¼Œå�¯èƒ½çš„æ•°æ›´å¤š)<br>
3.idæ˜¯æŒ‰æ—¶é—´é¡ºåº�åˆ†é…�çš„å�—ï¼Œidè¶Šå°�ï¼Œç”¨æˆ·è¶Šæ–°ï¼Ÿ
""")


act_logs = act_logs.with_columns(
    pl.col('Timestamp').dt.date().alias('date'),
    pl.col('Timestamp').dt.year().alias('year'),     
    pl.col('Timestamp').dt.month().alias('month'),  
    pl.col('Timestamp').dt.week().alias('week'),
    pl.col('Timestamp').dt.weekday().alias('weekday'),
    pl.col('Timestamp').dt.day().alias('day'),  
    pl.col('Timestamp').dt.hour().alias('hour'),
    pl.col('Timestamp').dt.minute().alias('minute'),
    pl.col('Timestamp').dt.second().alias('second')
)
act_logs.head()


start_time = act_logs['Timestamp'].min()
end_time = act_logs['Timestamp'].max()
range_days = (end_time - start_time).days
print(f"ä»Šå¹´æ˜¯2025å¹´ï¼Œä½†æ˜¯æ•°æ�®é›†æ˜¯{act_logs['year'].unique().to_list()}å¹´{act_logs['month'].unique().to_list()}æœˆçš„")
print(f"æ“�ä½œåº�åˆ—ä»�{start_time}å¼€å§‹ï¼Œåˆ°{end_time}ç»“æ�Ÿã€‚ä¸€å…±{range_days}å¤©ã€‚")


def create_date_range_df(min_date: str, max_date: str) -> pl.DataFrame:
    # ç”Ÿæˆ�æ—¥æœŸåº�åˆ—ï¼ˆåŒ…å�«ä¸¤ç«¯æ—¥æœŸï¼‰
    date_series = pl.date_range(
        start=min_date,
        end=max_date,
        interval="1d",
        eager=True
    )
    
    return pl.DataFrame({"date": date_series})

start_date = start_time.date()
end_date = end_time.date()

date_range = create_date_range_df(start_date, end_date)
date_range = date_range.with_columns(
    pl.col('date').dt.year().alias('year'),     
    pl.col('date').dt.month().alias('month'),  
    pl.col('date').dt.week().alias('week'),
    pl.col('date').dt.weekday().alias('weekday'),
    pl.col('date').dt.day().alias('day')
).sort('date')
date_range.head()


weekday_count = date_range.group_by('weekday').agg(
    pl.count().alias('weekday_count')
).sort('weekday')
weekday_count


def plot_act_times_count_by_col(df , col):
    
    temp = df.group_by(col).agg(
        pl.count().alias('count')
    )

    fig , ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        x = temp[col],
        height = temp['count']
    )

    ax.set_xlabel(f"{col}")
    ax.set_ylabel('Act Times Count')
    ax.set_title(f'Act Times Count by {col}')
    plt.show()


plot_act_times_count_by_col(act_logs,'day')


plot_act_times_count_by_col(act_logs,'weekday')


temp = act_logs.group_by('weekday').agg(
    pl.count().alias('count')
)
temp = temp.join(
    weekday_count,
    how = 'left',
    on = 'weekday'
)
temp = temp.with_columns(
    (pl.col('count') / pl.col('weekday_count')).alias('avg_act_times_count')
)

fig , ax = plt.subplots(figsize=(12, 6))
ax.bar(
    x = temp['weekday'],
    height = temp['avg_act_times_count']
)
ax.set_xlabel(f"avg_act_times_count")
ax.set_ylabel('Act Times Count')
ax.set_title(f'Act Times Count by Weekday')
plt.show()


plot_act_times_count_by_col(act_logs,'hour')


display_opinion_box("""
1.æ ·æœ¬æ�¥è‡ªå�†å�²æ•°æ�®ï¼Œä¸€å…±29å¤©çš„é‡�<br>
2.27å�·æ´»è·ƒæ¬¡æ•°éª¤å�‡ï¼Œå½“å¤©å�‘ç”Ÿäº†ä»€ä¹ˆäº‹ï¼Ÿ<br>
3.æ´»è·ƒæ¬¡æ•°ä¼¼ä¹�æœ‰ç‚¹å‘¨æœŸæ•ˆåº”ä½†ä¸�å¤šã€‚å‘¨å››ä¼šæœ‰æ›´å¤šçš„æ´»è·ƒæ¬¡æ•°ï¼Œ0ç‚¹ä¼šæœ‰æ›´å¤šçš„æ´»è·ƒã€‚ä½†æ˜¯ä¸�ä¸€èˆ¬çš„ç»�éªŒæœ‰åˆ«ã€‚é€šå¸¸æ¸¸æˆ�çš„æ´»è·ƒå¢�å¤šåº”è¯¥å�‘ç”Ÿåœ¨å‘¨æœ«ï¼Œæˆ–è€…æ”¾å­¦ã€�ä¸‹ç�­æœŸé—´çš„æ´»è·ƒä¼šæ¯”è¾ƒå¤šã€‚è¿™æ˜¯ä¸ºä»€ä¹ˆï¼Ÿ<br>
""")


total_act_times_count = act_logs.shape[0]

act_logs.group_by('ActionType').agg(
    pl.count().alias('count'),
    (pl.count().alias('count') / total_act_times_count ).alias('percentage')
).sort('ActionType')


display_opinion_box("""
1. å�„ç§�ç±»å�‹ä¸�æ¸…æ¥šä»£è¡¨äº†ä»€ä¹ˆï¼Œä½†å�¯ä»¥çœ‹åˆ°ç±»å�‹1çš„æ“�ä½œæ¬¡æ•°æœ€å¤šï¼Œç±»å�‹2æ¬¡ä¹‹ï¼Œ3æ˜¯æœ€å°‘çš„ã€‚<br>
2. ä»�æ“�ä½œæ¬¡æ•°ç§�å�¯å�¦çŒœæµ‹å‡ºå…·ä½“çš„ç±»å�‹æ˜¯ä»€ä¹ˆã€‚
<br>
ç±»å�‹1æœ€é«˜é¢‘ï¼Œåº”è¯¥æ˜¯ä¸€äº›æœ€åŸºæœ¬çš„æ“�ä½œï¼Œæ¯”å¦‚ç§»åŠ¨ï¼Œè·³è·ƒï¼Œæ”»å‡»ï¼Œè§†è§’è°ƒæ•´ç­‰ã€‚<br>
æ¬¡é«˜é¢‘çš„ç±»å�‹0ï¼Œåº”è¯¥æ˜¯ä¸€äº›æ¯”è¾ƒå¤�æ�‚çš„æ“�ä½œï¼Œæ¯”å¦‚ä½¿ç”¨é�“å…·ï¼Œä½¿ç”¨æŠ€èƒ½ï¼Œè·Ÿç‰©å“�ã€�æ€ªç‰©çš„ä¸€äº›äº¤äº’<br>
2ã€�3ã€�4 éƒ½å·®ä¸�å¤šï¼Œæœ€å°‘çš„3ï¼Œå�¯èƒ½æ˜¯ä¸€äº›ç•Œé�¢æ“�ä½œï¼Œç™»å½•/é€€å‡ºï¼Œé¢†å¥–ï¼Œä»˜è´¹ä¹‹ç±»çš„ã€‚<br>
2ã€�4å�¯èƒ½æ˜¯ä¸€äº›å…³é”®æ“�ä½œæˆ–è€…æœºåˆ¶ï¼Œå®Œæˆ�ä»»åŠ¡/æ��äº¤ä»»åŠ¡ï¼Œè¿›å…¥å‰¯æœ¬ã€�åŒ¹é…� PVPï¼Ÿäº¤æ˜“ã€�å¼ºåŒ–è£…å¤‡ï¼ŒæŠ€èƒ½åŠ ç‚¹ï¼Ÿã€‚<br>

ä¸€æ¬¾æ¸¸æˆ�çš„æ“�ä½œæ¬¡æ•°æœ€å¤šçš„æ“�ä½œç±»å�‹ï¼Œå…·ä½“è¿˜å¾—çœ‹æ¸¸æˆ�çš„ç±»å�‹ã€‚<br> 
""")


total_ActionId_times_count = act_logs.group_by('ActionId').agg(
    
    pl.count().alias('count'),
    (pl.count().alias('count') / total_act_times_count ).alias('percentage')
).sort('count')

total_ActionId_times_count = total_ActionId_times_count.with_columns(
    pl.col('ActionId').cast(pl.Utf8).str.len_chars().alias('ActionId_len')
)
total_ActionId_times_count.head()


total_ActionId_times_count.describe()


total_ActionId_times_count.group_by('ActionId_len').count()


start_id = 1
end_id = 144
consecutive_ActionId_list = pl.DataFrame().select(
    pl.int_range(start_id, end_id + 1, dtype=pl.Int32).alias("ActionId")
)

lacking_Action_id = pl.DataFrame(
    {'ActionId': list(set(consecutive_ActionId_list['ActionId']) - set(total_ActionId_times_count['ActionId']))}
    ).with_columns(
    pl.col('ActionId').cast(pl.Utf8).str.len_chars().alias('ActionId_len')
)
lacking_Action_id.head()


lacking_Action_id.group_by('ActionId_len').agg(
    pl.count().alias('count')).sort('ActionId_len')


fig , axes = plt.subplots(1,2,figsize=(16,6))
axes = axes.flatten()
sns.scatterplot(total_ActionId_times_count['count'] , ax = axes[0])
sns.kdeplot(total_ActionId_times_count['count'] , ax = axes[1])

axes[0].set_title('ActionId count scatter')
axes[0].set_xlabel('index')
axes[0].set_ylabel('ActionId count')

axes[1].set_title('ActionId count kde')
axes[1].set_xlabel('ActionId count')

plt.tight_layout()
plt.show()


total_ActionId_times_count.head(10)


total_ActionId_times_count.tail(10)


display_opinion_box("""
1. å�•çœ‹ä¸€ä¸ªå�˜é‡�,å�ªèƒ½çŸ¥æ™“æœ‰äº›æ“�ä½œæ¯”è¾ƒé«˜é¢‘çš„ï¼Œåˆ«çš„æ²¡æœ‰äº†<br>
2. æ­¤å¤–ï¼Œæˆ‘æƒ³çŸ¥é�“ï¼Œè¿™äº›æ“�ä½œï¼Œæ˜¯å�¦æœ‰æŸ�äº›è§„å¾‹ã€‚æ¯”å¦‚ï¼Œæ˜¯å�¦æœ‰å‘¨æœŸæ€§ï¼Ÿæˆ–è€…æ—¶é—´ä¸Šçš„å…ˆå��é¡ºåº�ã€‚å®ƒè·Ÿæ“�ä½œç±»å�‹çš„å®šä¹‰æœ‰æ²¡æœ‰è�”ç³»<br>
""")


#æ¯�æ—¥æ´»è·ƒæ¬¡æ•°ï¼Œç”¨æˆ·æ•°
daily_user_act_stat = act_logs.group_by('date').agg(
    pl.count().alias('act_times'),
    pl.n_unique('ID').alias('login_users')
).sort('date')
daily_user_act_stat.head()


# æ—¥å¿—æ—¥æœŸèŒƒå›´åŒ¹ä¸Šä¸€æ®µæ—¥ç»Ÿè®¡é‡�
user_act_date = date_range.join(
    daily_user_act_stat,
    left_on = 'date',
    right_on = 'date',
    how = 'left'
)
user_act_date.head()


user_act_date.select(pl.all().is_null().sum())


# æ—¥å¿—æ—¥æœŸèŒƒå›´å†…çš„å‘¨æœ«æ—¥æœŸ
weekend_dates = date_range.filter(pl.col('weekday').is_in([6, 7]))['date'].to_list()


def plot_axvline(ax, dates,color,label):
    for date in dates:
        ax.axvline(
            x = date,
            color = color,
            linestyle='--',
            alpha=0.5,  # é€�æ˜�åº¦
            label=label if date == dates[0] else ""  # å�ªæ·»åŠ ä¸€æ¬¡å›¾ä¾‹
        )

def plot_axvspan(ax, date_start = datetime(2018, 10, 1),date_end = datetime(2018, 10, 7),color = 'orange',label = 'National Day'):
    ax.axvspan(
        xmin=date_start,
        xmax=date_end,
        color=color,
        alpha=0.2,
        label=label
    )

def plot_line(data , ax, x ,y,color,if_xlabel = True,if_ylabel = True):
    if if_xlabel:
        ax.set_xlabel(x)
    if if_ylabel:
        ax.set_ylabel(y)
    ax.plot(data[x],data[y],color = color,label = y)
    ax.tick_params(axis='y', labelcolor = color)
    return ax 


fig , ax1 = plt.subplots(figsize=(12, 6))
ax1.grid(True, linestyle='--', alpha=0.7)

color1 = 'tab:blue'
x1 = 'date'
y1 = 'act_times'

color2 = 'tab:red'
y2 = 'login_users'

ax2 = ax1.twinx()
ax1 = plot_line(daily_user_act_stat,ax1,x1,y1,color1)
ax2 = plot_line(daily_user_act_stat,ax2,x1,y2,color2)


max_login_users_date = daily_user_act_stat.filter(
    pl.col('login_users') == pl.col('login_users').max()
)['date'].to_list()[0]  

min_login_users_date = daily_user_act_stat.filter(
    pl.col('login_users') == pl.col('login_users').min()
)['date'].to_list()[0]  

print('peak date:', max_login_users_date)
print('valley date:', min_login_users_date)


ax1.axvline(
    x = max_login_users_date,
    linestyle = '--',
    color = 'green',
    label = 'max login users'
)

ax1.axvline(
    x = min_login_users_date,
    linestyle = '--',
    color = 'purple',
    label = 'min login users'
)

plot_axvspan(ax1)
plot_axvline(ax1, weekend_dates, color='green', label='Weekend')

fig.autofmt_xdate()
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.title('Daily Act Times and Login Users')
plt.show()


# ç”¨æˆ·çˆ†å�‘ä¹‹å‰�å�„ä½�æ•°idåˆ†å¸ƒå� æ€»ä½“åˆ†å¸ƒçš„æ¯”ä¾‹
act_logs.group_by('ID_len').agg(pl.n_unique('ID').alias('count')).join(act_logs.filter(pl.col('date') <= min_login_users_date).group_by('ID_len').agg(pl.n_unique('ID').alias('count')).sort('ID_len'),
how = 'left',
on = 'ID_len').with_columns(
    (pl.col('count_right') / pl.col('count')).alias('ratio')
).sort('ID_len')



original_act_mean = daily_user_act_stat["act_times"].mean()
original_user_mean = daily_user_act_stat["login_users"].mean()

trimmed_act_mean = daily_user_act_stat.filter(
    (pl.col("act_times") > pl.col("act_times").min()) & 
    (pl.col("act_times") < pl.col("act_times").max())
)["act_times"].mean()

trimmed_user_mean = daily_user_act_stat.filter(
    (pl.col("login_users") > pl.col("login_users").min()) & 
    (pl.col("login_users") < pl.col("login_users").max())
)["login_users"].mean()

# è¾“å‡ºç»“æ�œ
print(f"å¹³å�‡æ¯�å¤©æ´»è·ƒæ¬¡æ•°ï¼š{original_act_mean}, æ´»è·ƒç”¨æˆ·æ•°ï¼š{original_user_mean}")
print(f"å‰”é™¤æ��ç«¯å€¼å��çš„å¹³å�‡æ´»è·ƒæ¬¡æ•°ï¼š{trimmed_act_mean}, æ´»è·ƒç”¨æˆ·æ•°ï¼š{trimmed_user_mean}")


d3_moving_avg_usr_act_stat = user_act_date.sort('date').select(
    pl.col('date'),
    pl.col('act_times').rolling_mean(window_size=3).alias('d3_moving_avg_act_times'),
    pl.col('login_users').rolling_mean(window_size=3).alias('d3_moving_avg_login_users')
)


fig , ax1 = plt.subplots(figsize=(12, 6))
ax1.grid(True, linestyle='--', alpha=0.7)

color1 = 'tab:blue'
x1 = 'date'
y1 = 'd3_moving_avg_act_times'

color2 = 'tab:red'
y2 = 'd3_moving_avg_login_users'
ax2 = ax1.twinx()

ax1 = plot_line(d3_moving_avg_usr_act_stat,ax1,x1,y1,color1)
ax2 = plot_line(d3_moving_avg_usr_act_stat,ax2,x1,y2,color2)

plot_axvspan(ax1)
plot_axvline(ax1, weekend_dates, color='green', label='Weekend')

fig.autofmt_xdate()
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.title('Trend of Act Times and Login Users')
plt.show()


daily_growth_stat = daily_user_act_stat.sort('date').select(
    pl.col('date'),

    pl.col('act_times'),
    pl.col('act_times').diff(1).alias('growth_act_times'),
    (pl.col('act_times').diff(1)/pl.col('act_times').shift(1)).alias('growth_ratio_act_times'),

    pl.col('login_users'),
    pl.col('login_users').diff(1).alias('growth_login_users'),
    (pl.col('login_users').diff(1)/pl.col('login_users').shift(1)).alias('growth_ratio_login_users'),
).fill_null(0)

daily_growth_stat.head(5)


fig , ax1 = plt.subplots(figsize=(12, 6))
ax1.grid(True, linestyle='--', alpha=0.7)

color1 = 'tab:blue'
x1 = 'date'
y1 = 'growth_act_times'

color2 = 'tab:red'
y2 = 'growth_ratio_act_times'

ax2 = ax1.twinx()
ax1 = plot_line(daily_growth_stat,ax1,x1,y1,color1)
ax2 = plot_line(daily_growth_stat,ax2,x1,y2,color2)

#å›½åº†åŒºé—´
plot_axvspan(ax1)
# ç»˜åˆ¶å‘¨æœ«ç«–çº¿
plot_axvline(ax1, weekend_dates, color='green', label='Weekend')

fig.autofmt_xdate()
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.title('Daily Growth of Act Times')
plt.show()


print(f"æœ€å¤§å¢�é•¿ç�‡ï¼š{daily_growth_stat.filter(pl.col('growth_ratio_login_users') == pl.col('growth_ratio_login_users').max())['growth_ratio_login_users'].to_list()[0]}")


fig , ax1 = plt.subplots(figsize=(12, 6))
ax1.grid(True, linestyle='--', alpha=0.7)

color1 = 'tab:blue'
x1 = 'date'
y1 = 'growth_login_users'

color2 = 'tab:red'
y2 = 'growth_ratio_login_users'

ax2 = ax1.twinx()
ax1 = plot_line(daily_growth_stat,ax1,x1,y1,color1)
ax2 = plot_line(daily_growth_stat,ax2,x1,y2,color2)

#å›½åº†åŒºé—´
plot_axvspan(ax1)
# ç»˜åˆ¶å‘¨æœ«ç«–çº¿
plot_axvline(ax1, weekend_dates, color='green', label='Weekend')

fig.autofmt_xdate()
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.title('Daily Growth of Login Users')
plt.show()


user_act = act_logs.select(["ID", "date"]).unique().sort('date')
user_act.head()


daily_retained_usrs = user_act.join(
    user_act,
    on = 'ID',
    suffix='_prev'
).filter(
    (pl.col('date') == pl.col('date_prev') + pl.duration(days=1))
)
daily_retained_usrs.head()


daily_retained_usrs_stats = daily_retained_usrs.group_by('date').agg(
    pl.col('ID').n_unique().alias('retained_users'),
).sort('date').with_columns(
    pl.col('retained_users').diff().alias('growth_retained_users'),
    (pl.col('retained_users').diff() / pl.col('retained_users').shift(1)).alias('growth_ratio_retained_users')
).fill_null(0)
daily_retained_usrs_stats.head()


fig , axes = plt.subplots(2,1,figsize=(12, 10))
axes.flatten()

axes[0].grid(True, linestyle='--', alpha=0.7)
axes[1].grid(True, linestyle='--', alpha=0.7)

color01 = 'tab:orange'
x01 = 'date'
y01 = 'retained_users'
axes[0] = plot_line(daily_retained_usrs_stats,axes[0],x01,y01,color01,if_xlabel=False)
axes[0].tick_params(axis='x', rotation=15)
#å›½åº†åŒºé—´
plot_axvspan(axes[0])
# ç»˜åˆ¶å‘¨æœ«ç«–çº¿
plot_axvline(axes[0], weekend_dates, color='green', label='Weekend')

axes[0].legend(loc='upper right')


color11 = 'tab:blue'
x11 = 'date'
y11 = 'growth_retained_users'

color12 = 'tab:red'
y12 = 'growth_ratio_retained_users'

ax12 = axes[1].twinx()
ax1 = plot_line(daily_retained_usrs_stats,axes[1],x11,y11,color11)
ax2 = plot_line(daily_retained_usrs_stats,ax12,x11,y12,color12)
axes[1].tick_params(axis='x', rotation=15)

#å›½åº†åŒºé—´
plot_axvspan(axes[1])
# ç»˜åˆ¶å‘¨æœ«ç«–çº¿
plot_axvline(axes[1], weekend_dates, color='green', label='Weekend')

lines1, labels1 = axes[1].get_legend_handles_labels()
lines2, labels2 = ax12.get_legend_handles_labels()
axes[1].legend(lines1 + lines2, labels1 + labels2, loc='upper right')

fig.suptitle('Daily Retained Users')
fig.tight_layout()
fig.show()


original_user_mean = daily_retained_usrs_stats["retained_users"].mean()

trimmed_user_mean = daily_retained_usrs_stats.filter(
    (pl.col("retained_users") > pl.col("retained_users").min()) & 
    (pl.col("retained_users") < pl.col("retained_users").max())
)["retained_users"].mean()

# è¾“å‡ºç»“æ�œ
print(f"å¹³å�‡æ¯�å¤©ç•™å­˜ç”¨æˆ·æ•°ï¼š{original_user_mean}")
print(f"å‰”é™¤æ��ç«¯å€¼å��çš„å¹³å�‡ç•™å­˜ç”¨æˆ·æ•°ï¼š{trimmed_user_mean}")


daily_new_users = user_act.group_by('ID').agg(
    pl.col('date').min().alias('first_act_date')
)
daily_new_users.head()


daily_new_users.group_by('first_act_date').agg(
    pl.col('ID').min().alias('daily_min_id'),
    pl.col('ID').max().alias('daily_max_id'),
    pl.col('ID').mean().alias('daily_avg_id'),
    pl.col('ID').median().alias('daily_median_id'),
    pl.col('ID').std().alias('daily_std_id')
).sort('first_act_date')


daily_new_users_stat = daily_new_users.group_by('first_act_date').agg(
    pl.col('ID').n_unique().alias('new_users'),
).sort('first_act_date').with_columns(
    pl.col('new_users').diff().alias('growth_new_users'),
    (pl.col('new_users').diff() / pl.col('new_users').shift(1)).alias('growth_ratio_new_users')
).fill_null(0)
daily_new_users_stat.head()


fig , axes = plt.subplots(2,1,figsize=(12, 10))
axes.flatten()

axes[0].grid(True, linestyle='--', alpha=0.7)
axes[1].grid(True, linestyle='--', alpha=0.7)

color01 = 'tab:orange'
x01 = 'first_act_date'
y01 = 'new_users'
axes[0] = plot_line(daily_new_users_stat,axes[0],x01,y01,color01,if_xlabel=False)
axes[0].tick_params(axis='x', rotation=15)
#å›½åº†åŒºé—´
plot_axvspan(axes[0])
# ç»˜åˆ¶å‘¨æœ«ç«–çº¿
plot_axvline(axes[0], weekend_dates, color='green', label='Weekend')

axes[0].legend(loc='upper right')


color11 = 'tab:blue'
x11 = 'first_act_date'
y11 = 'growth_new_users'

color12 = 'tab:red'
y12 = 'growth_ratio_new_users'

ax12 = axes[1].twinx()
ax1 = plot_line(daily_new_users_stat,axes[1],x11,y11,color11)
ax2 = plot_line(daily_new_users_stat,ax12,x11,y12,color12)
axes[1].tick_params(axis='x', rotation=15)

#å›½åº†åŒºé—´
plot_axvspan(axes[1])
# ç»˜åˆ¶å‘¨æœ«ç«–çº¿
plot_axvline(axes[1], weekend_dates, color='green', label='Weekend')

lines1, labels1 = axes[1].get_legend_handles_labels()
lines2, labels2 = ax12.get_legend_handles_labels()
axes[1].legend(lines1 + lines2, labels1 + labels2, loc='upper right')

fig.suptitle('Daily New Users')
fig.tight_layout()
fig.show()


original_user_mean = daily_new_users_stat["new_users"].mean()

trimmed_user_mean = daily_new_users_stat.filter(
    (pl.col("new_users") > pl.col("new_users").min()) & 
    (pl.col("new_users") < pl.col("new_users").max())
)["new_users"].mean()

# è¾“å‡ºç»“æ�œ
print(f"å¹³å�‡æ¯�å¤©æ–°å¢�ç”¨æˆ·æ•°ï¼š{original_user_mean}")
print(f"å‰”é™¤æ��ç«¯å€¼å��çš„å¹³å�‡æ–°å¢�ç”¨æˆ·æ•°ï¼š{trimmed_user_mean}")


# æ¯�ä¸ªç”¨æˆ·çš„æœ€å��æ´»è·ƒæ—¥æœŸï¼ˆè§†ä¸ºæµ�å¤±æ—¥æœŸï¼‰
daily_churned_users = user_act.group_by('ID').agg(
    pl.col('date').max().alias('last_act_date')  
)

# æŒ‰æµ�å¤±æ—¥æœŸç»Ÿè®¡æ¯�æ—¥æµ�å¤±ç”¨æˆ·æ•°
daily_churned_users_stat = daily_churned_users.group_by('last_act_date').agg(
    pl.col('ID').n_unique().alias('churned_users'),
).sort('last_act_date').with_columns(
    pl.col('churned_users').diff().alias('growth_churned_users'), 
    (pl.col('churned_users').diff() / pl.col('churned_users').shift(1)).alias('growth_ratio_churned_users') 
).fill_null(0)

daily_churned_users_stat.head()


fig , axes = plt.subplots(2,1,figsize=(12, 10))
axes.flatten()

axes[0].grid(True, linestyle='--', alpha=0.7)
axes[1].grid(True, linestyle='--', alpha=0.7)

color01 = 'tab:orange'
x01 = 'last_act_date'
y01 = 'churned_users'
axes[0] = plot_line(daily_churned_users_stat,axes[0],x01,y01,color01,if_xlabel=False)
axes[0].tick_params(axis='x', rotation=15)
#å›½åº†åŒºé—´
plot_axvspan(axes[0])
# ç»˜åˆ¶å‘¨æœ«ç«–çº¿
plot_axvline(axes[0], weekend_dates, color='green', label='Weekend')

axes[0].legend(loc='upper right')


color11 = 'tab:blue'
x11 = 'last_act_date'
y11 = 'growth_churned_users'

color12 = 'tab:red'
y12 = 'growth_ratio_churned_users'

ax12 = axes[1].twinx()
ax1 = plot_line(daily_churned_users_stat,axes[1],x11,y11,color11)
ax2 = plot_line(daily_churned_users_stat,ax12,x11,y12,color12)
axes[1].tick_params(axis='x', rotation=15)

#å›½åº†åŒºé—´
plot_axvspan(axes[1])
# ç»˜åˆ¶å‘¨æœ«ç«–çº¿
plot_axvline(axes[1], weekend_dates, color='green', label='Weekend')

lines1, labels1 = axes[1].get_legend_handles_labels()
lines2, labels2 = ax12.get_legend_handles_labels()
axes[1].legend(lines1 + lines2, labels1 + labels2, loc='upper right')

fig.suptitle('Daily Churned Users')
fig.tight_layout()
fig.show()


original_user_mean = daily_churned_users_stat["churned_users"].mean()

trimmed_user_mean = daily_churned_users_stat.filter(
    (pl.col("churned_users") > pl.col("churned_users").min()) & 
    (pl.col("churned_users") < pl.col("churned_users").max())
)["churned_users"].mean()

trimmed_user_std = daily_churned_users_stat.filter(
    (pl.col("churned_users") > pl.col("churned_users").min()) & 
    (pl.col("churned_users") < pl.col("churned_users").max())
)["churned_users"].std()

# è¾“å‡ºç»“æ�œ
print(f"å¹³å�‡æ¯�å¤©æµ�å¤±ç”¨æˆ·æ•°ï¼š{original_user_mean}")
print(f"å‰”é™¤æ��ç«¯å€¼å��çš„å¹³å�‡æµ�å¤±ç”¨æˆ·æ•°ï¼š{trimmed_user_mean}")
print(f"å‰”é™¤æ��ç«¯å€¼å��çš„å¹³å�‡æµ�å¤±ç”¨æˆ·æ•°æ ‡å‡†å·®ï¼š{trimmed_user_std}")


# å…¨å±€è¿�ç»­æ´»è·ƒå¤©æ•°
# ctn mears continuous
ctn_act_usrs = user_act.with_columns(
    (pl.col('date').sort().rank().over("ID")).alias("rank")
).with_columns(
   assistant_date = pl.col('date')  - pl.duration(days=pl.col("rank"))
)
ctn_act_usrs.head()


overall_ctn_act_usrs = ctn_act_usrs.group_by(['ID','assistant_date']).agg(
    pl.count().alias('ctn_days')
)
overall_ctn_act_usrs.head()


overall_ctn_act_usrs_stat = overall_ctn_act_usrs.group_by(['ctn_days']).agg(
    pl.col('ID').n_unique().alias('count')
).sort(['ctn_days'])
overall_ctn_act_usrs_stat.head()


fig , ax = plt.subplots(figsize=(12, 6))
#å�¯èƒ½ç‰ˆæœ¬é—®é¢˜è½¬ä¸€ä¸‹
overall_ctn_act_usrs_stat = overall_ctn_act_usrs_stat.to_pandas()
bar = sns.barplot(overall_ctn_act_usrs_stat, x = 'ctn_days',y = 'count' , ax = ax)
ax.bar_label(bar.containers[0],rotation=30)
ax.set_title('Overall Continuous Act Users Count by Days')
fig.tight_layout()
fig.show()


daily_user_gap_days = (
    user_act.sort(["ID", "date"]).with_columns(
        pl.col("date").shift(-1).over(["ID"]).alias("next_date")  
    ).with_columns(
        (pl.col("next_date") - pl.col("date")).dt.total_days().alias("days_to_next_active")
    ).drop("next_date")  
)
daily_user_gap_days.head()


daily_user_gap_days_stat = daily_user_gap_days.group_by('date').agg(
    pl.col('days_to_next_active').mean().alias('avg_days_to_next_active'),
).sort('date')
daily_user_gap_days_stat.head()


fig , ax1 = plt.subplots(figsize=(12, 6))
ax1.grid(True, linestyle='--', alpha=0.7)

color1 = 'tab:blue'
x1 = 'date'
y1 = 'avg_days_to_next_active'

ax1 = plot_line(daily_user_gap_days_stat,ax1,x1,y1,color1)

#å›½åº†åŒºé—´
plot_axvspan(ax1)
# ç»˜åˆ¶å‘¨æœ«ç«–çº¿
plot_axvline(ax1, weekend_dates, color='green', label='Weekend')

fig.autofmt_xdate()
ax1.legend(loc='upper right')

plt.title('Daily Avg Days To Next Active')
plt.show()


display_opinion_box("""
1. æ“�ä½œæ•°è·Ÿç”¨æˆ·ç»�å�†äº†ä¸€ä¸ªä»�ä½�è°·åˆ°æš´å¢�å†�åˆ°ç¨³å®šçš„è¿‡ç¨‹ã€‚<br>
2. æš´å¢�é˜¶æ®µæ˜¯å› ä¸ºæ¶Œå…¥äº†å¤§é‡�äº†æ–°ç”¨æˆ·ï¼Œè¿™äº›æ–°ç”¨æˆ·å�ˆå¾ˆå¿«åœ°ç¦»å¼€äº†ï¼Œå��ç»­çš„æ´»è·ƒä¸»è¦�ä¾�é� è€�ç”¨æˆ·<br>
3. è¿‘ä¸€ä¸ªæœˆçš„ç”¨æˆ·éƒ½æ˜¯æ–­æ–­ç»­ç»­ç™»å½•ä¸€å¤©çš„ä¸ºä¸»ã€‚<br>
4. å¾€å��ç”¨æˆ·çš„ç²˜æ€§å¼€å§‹é€�æ¸�å¢�åŠ ã€‚<br>
5. å�¦å¤–æ³¨æ„�åˆ°ï¼Œå›½åº†ï¼Œå‘¨æœ«éƒ½æ²¡æœ‰ç‰¹åˆ«åœ°è·Ÿå¢�é‡�çš„æ—¥æœŸå� åŠ ä¸Šã€‚æ²¡æœ‰æ˜�æ˜¾çš„å›½åº†ï¼Œå‘¨æœ«æ•ˆåº”ã€‚<br>
""")


opinion_text = """
æ ¹æ�®ç›®å‰�çš„æƒ…å†µï¼Œå¤§èƒ†çŒœæµ‹ä¸‹ã€‚<br>  
é¦–å…ˆè¿™æ¬¾æ¸¸æˆ�å�¯èƒ½ä¸»è¦�ä¸�æ˜¯é�¢å�‘å›½å†…æˆ–è€…æ³›ä¸­å��åŒºï¼Œå› ä¸ºå›½åº†å‰�ï¼Œæˆ–è€…åˆ�æœŸå�„ä¸ªæŒ‡æ ‡æ²¡æœ‰å¤§å¢�é‡�ï¼Œæ›´å�¯èƒ½æ˜¯é�¢å�‘æµ·å¤–çš„ï¼Œè€Œä¸”å¾ˆæœ‰å�¯èƒ½æ˜¯æ¬§æ´²é‚£è¾¹ã€‚æ¬§æ´²é‚£è¾¹ä¼ è¨€æ˜¯å››å¤©å·¥ä½œæ—¥åˆ¶ï¼Œä¸”æ—¶åŒºåœ¨UTC+0 åˆ° UTC + 2ä¹‹é—´ï¼Œå½“ä¸­å›½æ˜¯å‘¨äº”å‡Œæ™¨0ç‚¹çš„æ—¶å€™ï¼Œæœ€è¿œçš„UTC+0çš„è‹±å›½æ­£å¥½æ˜¯å‘¨å››16ç‚¹ã€‚ä¸‹æ²¡ä¸‹ç�­ä¸�å¥½è¯´ï¼Œä½†åº”è¯¥æ˜¯æ”¾å­¦äº†ã€‚æ‰€ä»¥ä¸»è¦�ç”¨æˆ·åº”è¯¥æ˜¯æ¬§æ´²åœ°åŒºçš„å­¦ç”Ÿã€‚  <br>  
å†�æœ‰ï¼Œæ–°ç”¨æˆ·æœ‰ä¸ªçŸ­æœŸæš´å¢�çš„ç‚¹ï¼Œè¦�ä¹ˆæ˜¯å› ä¸ºå�šæ´»åŠ¨æˆ–è€…è�¥é”€å¸¦æ�¥çš„ï¼Œä½†æ˜¯è¿™ç±»å¢�ç�‡ä¸�å¤ªå�¯èƒ½è¾¾åˆ°3.69å€�ä¹‹å¤šï¼Œä¹‹å��å�ˆçˆ†å‡�ï¼Œè¦�ä¹ˆå�ªèƒ½æ˜¯æ–°å¼€æœ�äº†ã€‚æŠ›å¼€ç”¨æˆ·æ˜¯éƒ¨åˆ†é‡‡æ ·çš„å�¯èƒ½ï¼Œæ€»ä½“çš„ç”¨æˆ·idåº”è¯¥æ˜¯æ‰€æœ‰å�¯èƒ½çš„è¿�ç»­æ•´æ•°æ‰�å¯¹ï¼Œè€Œä¸”æ—¥æœŸè¶Šæ—©idè¶Šå°�ã€‚ä½†æ˜¯ç›®å‰�ç¼ºäº†ä¸€éƒ¨åˆ†idï¼Œå¹¶ä¸”ä¸�ç®¡æ˜¯çˆ†å�‘æœŸå‰�çš„idï¼Œè¿˜æ˜¯ç¼ºçš„idçš„åˆ†å¸ƒéƒ½å¾ˆå�‡åŒ€ã€‚çŒœæµ‹å�¯èƒ½æ˜¯æ¸¸æˆ�æ��å‰�æ”¾å¼€äº†æ³¨å†Œé¢„çº¦ï¼Œæœ‰å¤šä¸ªæœ�åŠ¡å™¨å¹¶ä¸”ä¸€å¼€å§‹å�ªæ˜¯åœ¨æŸ�äº›æœ�åŠ¡å™¨ä¸Šå°�èŒƒå›´å¼€æœ�ï¼Œä¹‹å��å†�å…¨é�¢å¼€æœ�ã€‚ <br>   
æœ€å��ï¼Œå…¨é�¢å¼€æœ�å��å�¯èƒ½æœ‰ä»¥å‘¨ä¸ºå‘¨æœŸçš„æ´»åŠ¨,è¿™é€ æˆ�äº†è€�ç”¨æˆ·çš„æ´»è·ƒå°�çˆ†å�‘æœŸã€‚
"""


html_content = f"""
<div style="background-color: #DAA520; border-radius: 10px; padding: 15px; border: 1px solid ; color:white;">
  <strong>ğŸ’¡ Insights <br></strong> {opinion_text}
</div>
"""
display(HTML(html_content))


daily_AT_distribution = act_logs.group_by(['date','ActionType']).agg(
    pl.count().alias('daily_at_times'),
).with_columns(
    (pl.col("daily_at_times") / pl.sum("daily_at_times").over("date")).alias("proportion")
).sort(["date", "ActionType"])
daily_AT_distribution.head()


fig , axes = plt.subplots(2, 1, figsize = (12, 10))
axes = axes.flatten()

axes[0].grid(True, linestyle='--', alpha=0.7)

x1 = 'date'
y1 = 'daily_at_times'
ax1 = sns.lineplot(data=daily_AT_distribution, x=x1, y=y1, ax=axes[0],hue = 'ActionType',palette = 'tab10')

#å›½åº†åŒºé—´
plot_axvspan(ax1)
# ç»˜åˆ¶å‘¨æœ«ç«–çº¿
plot_axvline(ax1, weekend_dates, color='green', label='Weekend')


ax1.tick_params(axis='x', rotation=15)
ax1.set_title('Daily AT Times')

y2 = 'proportion'

# ç»˜åˆ¶å †å� æŸ±çŠ¶å›¾
ax2 = axes[1]
# è½¬ä¸‹pandas polarså¥½åƒ�ä¸�å¤ªæ”¯æŒ�
stack_data = (
daily_AT_distribution.pivot(index='date', columns='ActionType', values='proportion').fill_null(0).to_pandas().set_index('date')
)

action_types = daily_AT_distribution['ActionType'].unique().sort().to_list()
colors = sns.color_palette("tab10", n_colors=len(action_types))

stack_data.plot(
    kind='bar', 
    stacked=True, 
    ax=ax2,
    color=colors,
    width=0.8,
    edgecolor='white'
)

ax2.set_title('Daily AT Times proportion', fontsize=14, pad=12)
ax2.set_xlabel('date')
ax2.set_ylabel('proportion')
ax2.tick_params(axis='x', rotation=15)
ax2.grid(True, axis='y', linestyle='--', alpha=0.7)

date_labels = stack_data.index.strftime('%m-%d').tolist()
ax2.set_xticklabels(date_labels, rotation=90)

ax2.legend(
    bbox_to_anchor=(1, 1), 
    title='ActionType',
    )

plt.tight_layout()
plt.show()


# E means the earliest, L means latest
daily_EL_AT = act_logs.sort('Timestamp').group_by(['date']).agg(
    pl.col('ActionType').first().alias('daily_E_AT'),
    pl.col('ActionType').last().alias('daily_L_AT')
).sort('date')
daily_EL_AT.head()


daily_EL_AT.group_by('daily_E_AT').agg(
    pl.count()
).sort('daily_E_AT')


daily_EL_AT.group_by('daily_L_AT').agg(
    pl.count()
).sort('daily_L_AT')


daily_AT_frequency =act_logs.sort('Timestamp').group_by(['date','ActionType']).agg(
    pl.col('Timestamp').diff(1).dt.total_seconds().mean().alias('time_diff(sec)')
)


daily_AT_frequency.group_by('ActionType').agg(
    pl.col('time_diff(sec)').mean()
).sort('ActionType')


fig , axes = plt.subplots(5,1,figsize = (12, 15))
axes = axes.flatten()

data = daily_AT_frequency
x = 'date'
y = 'time_diff(sec)'

for i in range(5):
    temp = data.filter(
        pl.col('ActionType') == i
    ).sort('date')

    axes[i].grid(True, linestyle='--', alpha=0.7)
    plot_line(temp,ax = axes[i],x = x ,y = y , color = colors[i])

    #å›½åº†åŒºé—´
    plot_axvspan(axes[i])
    # ç»˜åˆ¶å‘¨æœ«ç«–çº¿
    plot_axvline(axes[i], weekend_dates, color='green', label='Weekend')


    axes[i].tick_params(axis='x', rotation=15)
    axes[i].set_title(f'Daily AT {i} frequency')

plt.tight_layout()
plt.show()


# æ¨ªç�€çœ‹æ˜�ç»†
daily_AT_frequency.filter(
    pl.col('ActionType') == 3
).sort('date').transpose()


daily_AT_frequency.filter(
    pl.col('ActionType') == 4
).sort('date').transpose()


daily_AID_distribution = act_logs.group_by(['date','ActionId']).agg(
    pl.count().alias('daily_at_times'),
).with_columns(
    (pl.col("daily_at_times") / pl.sum("daily_at_times").over("date")).alias("proportion")
).sort(["date", "ActionId"])
daily_AID_distribution.head()


fig , axes = plt.subplots(2, 1, figsize = (20, 15))
axes = axes.flatten()

axes[0].grid(True, linestyle='--', alpha=0.7)

x1 = 'date'
y1 = 'daily_at_times'
ax1 = sns.lineplot(data=daily_AID_distribution, x=x1, y=y1, ax=axes[0],hue = 'ActionId',palette = 'tab10',legend=False)

#å›½åº†åŒºé—´
plot_axvspan(ax1)
# ç»˜åˆ¶å‘¨æœ«ç«–çº¿
plot_axvline(ax1, weekend_dates, color='green', label='Weekend')


ax1.tick_params(axis='x', rotation=15)
ax1.set_title('Daily AT Times')

y2 = 'proportion'

# ç»˜åˆ¶å †å� æŸ±çŠ¶å›¾
ax2 = axes[1]
# è½¬ä¸‹pandas polarså¥½åƒ�ä¸�å¤ªæ”¯æŒ�
stack_data = (
daily_AID_distribution.pivot(index='date', columns='ActionId', values='proportion').fill_null(0).to_pandas().set_index('date')
)

action_types = daily_AID_distribution['ActionId'].unique().sort().to_list()
colors = sns.color_palette("tab10", n_colors=len(action_types))

stack_data.plot(
    kind='bar', 
    stacked=True, 
    ax=ax2,
    color=colors,
    width=0.8,
    edgecolor='white'
)

ax2.set_title('Daily AT Times proportion', fontsize=14, pad=12)
ax2.set_xlabel('date')
ax2.set_ylabel('proportion')
ax2.tick_params(axis='x', rotation=15)
ax2.grid(True, axis='y', linestyle='--', alpha=0.7)

date_labels = stack_data.index.strftime('%m-%d').tolist()
ax2.set_xticklabels(date_labels, rotation=90)

ax1.legend([])
ax2.legend([])

plt.tight_layout()
plt.show()


daily_AID_distribution_34 = act_logs.filter(
    pl.col("ActionType").is_in([3,4])
).group_by(['date','ActionId']).agg(
    pl.count().alias('daily_at_times'),
).with_columns(
    (pl.col("daily_at_times") / pl.sum("daily_at_times").over("date")).alias("proportion")
).sort(["date", "ActionId"])
daily_AID_distribution.head()


fig , axes = plt.subplots(2, 1, figsize = (20, 15))
axes = axes.flatten()

axes[0].grid(True, linestyle='--', alpha=0.7)

x1 = 'date'
y1 = 'daily_at_times'
ax1 = sns.lineplot(data=daily_AID_distribution_34, x=x1, y=y1, ax=axes[0],hue = 'ActionId',palette = 'tab10',legend=False)

#å›½åº†åŒºé—´
plot_axvspan(ax1)
# ç»˜åˆ¶å‘¨æœ«ç«–çº¿
plot_axvline(ax1, weekend_dates, color='green', label='Weekend')


ax1.tick_params(axis='x', rotation=15)
ax1.set_title('Daily AT Times')

y2 = 'proportion'

# ç»˜åˆ¶å †å� æŸ±çŠ¶å›¾
ax2 = axes[1]
# è½¬ä¸‹pandas polarså¥½åƒ�ä¸�å¤ªæ”¯æŒ�
stack_data = (
daily_AID_distribution_34.pivot(index='date', columns='ActionId', values='proportion').fill_null(0).to_pandas().set_index('date')
)

action_types = daily_AID_distribution_34['ActionId'].unique().sort().to_list()
colors = sns.color_palette("tab10", n_colors=len(action_types))

stack_data.plot(
    kind='bar', 
    stacked=True, 
    ax=ax2,
    color=colors,
    width=0.8,
    edgecolor='white'
)

ax2.set_title('Daily AT Times proportion', fontsize=14, pad=12)
ax2.set_xlabel('date')
ax2.set_ylabel('proportion')
ax2.tick_params(axis='x', rotation=15)
ax2.grid(True, axis='y', linestyle='--', alpha=0.7)

date_labels = stack_data.index.strftime('%m-%d').tolist()
ax2.set_xticklabels(date_labels, rotation=90)

ax1.legend([])
ax2.legend([])

plt.tight_layout()
plt.show()


# E means the earliest, L means latest
daily_EL_AID = act_logs.sort('Timestamp').group_by(['date']).agg(
    pl.col('ActionId').first().alias('daily_E_AID'),
    pl.col('ActionId').last().alias('daily_L_AID')
).sort('date')
daily_EL_AID.head()


daily_EL_AID.group_by('daily_E_AID').agg(
    pl.count()
).sort('count')


daily_EL_AID.group_by('daily_L_AID').agg(
    pl.count()
).sort('count')


daily_AID_frequency =act_logs.sort('Timestamp').group_by(['date','ActionId']).agg(
    pl.col('Timestamp').diff(1).dt.total_seconds().mean().alias('time_diff(sec)')
)


daily_AID_frequency.group_by('ActionId').agg(
    pl.col('time_diff(sec)').mean()
).sort('time_diff(sec)').head(10)


daily_AID_frequency.group_by('ActionId').agg(
    pl.col('time_diff(sec)').mean()
).sort('time_diff(sec)').tail(10)


fig , axes = plt.subplots(10,1,figsize = (12, 40))
axes = axes.flatten()

top_action_ids = daily_AID_frequency.group_by("ActionId").agg(
    pl.col("time_diff(sec)").mean()
).sort("time_diff(sec)",descending=True).head(10).get_column("ActionId").to_list()

data = daily_AID_frequency.filter(
    pl.col("ActionId").is_in(top_action_ids)
)

x = 'date'
y = 'time_diff(sec)'
colors = sns.color_palette("tab10", n_colors=10)

for i,id in enumerate(data['ActionId'].unique()):
    temp = data.filter(
        pl.col('ActionId') == id
    ).sort('date')

    axes[i].grid(True, linestyle='--', alpha=0.7)
    plot_line(temp,ax = axes[i],x = x ,y = y , color = colors[i])

    #å›½åº†åŒºé—´
    plot_axvspan(axes[i])
    # ç»˜åˆ¶å‘¨æœ«ç«–çº¿
    plot_axvline(axes[i], weekend_dates, color='green', label='Weekend')


    axes[i].tick_params(axis='x', rotation=15)
    axes[i].set_title(f'Daily AT {id} frequency')

plt.tight_layout()
plt.show()


daily_AID_frequency.group_by("ActionId").agg(
    pl.col("time_diff(sec)").mean()
).sort("time_diff(sec)")


fig , axes = plt.subplots(10,1,figsize = (12, 40))
axes = axes.flatten()

bottom_action_ids = daily_AID_frequency.group_by("ActionId").agg(
    pl.col("time_diff(sec)").mean()
).sort("time_diff(sec)").head(10).get_column("ActionId").to_list()

data = daily_AID_frequency.filter(
    pl.col("ActionId").is_in(bottom_action_ids)
)

x = 'date'
y = 'time_diff(sec)'
colors = sns.color_palette("tab10", n_colors=10)

for i,id in enumerate(data['ActionId'].unique()):
    temp = data.filter(
        pl.col('ActionId') == id
    ).sort('date')

    axes[i].grid(True, linestyle='--', alpha=0.7)
    plot_line(temp,ax = axes[i],x = x ,y = y , color = colors[i])

    #å›½åº†åŒºé—´
    plot_axvspan(axes[i])
    # ç»˜åˆ¶å‘¨æœ«ç«–çº¿
    plot_axvline(axes[i], weekend_dates, color='green', label='Weekend')


    axes[i].tick_params(axis='x', rotation=15)
    axes[i].set_title(f'Daily AT {id} frequency')

plt.tight_layout()
plt.show()


ActionId_by_type = act_logs.group_by("ActionType").agg(
    pl.col("ActionId").unique().alias("unique_ids"),
    pl.col("ActionId").n_unique().alias("n_unique_ids"),
    # å¯¹æ¯�ä¸ªå”¯ä¸€ ActionId è®¡ç®—å­—ç¬¦ä¸²é•¿åº¦
    pl.col("ActionId").unique().cast(pl.Utf8).str.len_chars().alias("id_lengths")
).sort("ActionType")


ActionId_by_type


#from collections import Counter
#ActionId_by_type.with_columns(
#    pl.col('id_lengths').map_elements(Counter()).alias('id_lengths_count')
#)


pivot_table = (
    ActionId_by_type.explode("id_lengths")
    .group_by("ActionType", "id_lengths")
    .count()
    .pivot(index="ActionType", columns="id_lengths", values="count")
).sort('ActionType').fill_null(0).to_pandas().set_index("ActionType")   
pivot_table = pivot_table[sorted(pivot_table.columns)]



sns.heatmap(pivot_table,annot=True,linewidths=.5,fmt='d',cmap='YlGnBu')
plt.xlabel('id_lengths')
plt.ylabel('ActionType')
plt.show()


pivot_table = act_logs.group_by(['ActionType','ActionId']).agg(
    pl.n_unique('ID').alias('Id_unique')
).pivot(index = 'ActionType',columns = 'ActionId' , values='Id_unique').fill_null(0).sort('ActionType').to_pandas().set_index("ActionType") 
pivot_table = pivot_table[sorted(pivot_table.columns)]


pivot_table


fig , ax  = plt.subplots(figsize=(60,4))
annot_labels = np.where(pivot_table == 0, "", pivot_table)
sns.heatmap(pivot_table,annot=annot_labels,linewidths=.5,fmt='',cmap='YlGnBu',annot_kws={"rotation": 30})
plt.xlabel('ActionId')
plt.ylabel('ActionType')
plt.show()


pivot_table.stack().nlargest(10)


pivot_table.replace(0, np.nan).stack().nsmallest(10)


pivot_table = act_logs.group_by(['date','ActionType']).agg(
    pl.n_unique('ID').alias('Id_unique')
).pivot(index = 'ActionType',columns = 'date' , values='Id_unique').fill_null(0).sort('ActionType').to_pandas().set_index("ActionType") 
pivot_table = pivot_table[sorted(pivot_table.columns)]


fig , ax  = plt.subplots(figsize=(60,4))
annot_labels = np.where(pivot_table == 0, "", pivot_table)
sns.heatmap(pivot_table,annot=annot_labels,linewidths=.5,fmt='',cmap='YlGnBu',annot_kws={"rotation": 30})
plt.xlabel('date')
plt.ylabel('ActionType')
plt.show()


temp = act_logs.group_by(['date','ActionType']).agg(
    pl.n_unique('ID').alias('Id_unique')
).sort('date')


fig , ax1 = plt.subplots(figsize=(12, 6))
ax1.grid(True, linestyle='--', alpha=0.7)

data = temp
color1 = 'tab:blue'
x1 = 'date'
y1 = 'Id_unique'

ax1 = sns.lineplot(data = data,ax = ax1,x = x1,y = y1,hue = 'ActionType',palette = 'tab10')

#å›½åº†åŒºé—´
plot_axvspan(ax1)
# ç»˜åˆ¶å‘¨æœ«ç«–çº¿
plot_axvline(ax1, weekend_dates, color='green', label='Weekend')

fig.autofmt_xdate()
ax1.legend(loc='upper right')

plt.title('Daily Users By ActionType')
plt.show()


temp.filter(
    pl.col('ActionType').is_in([3,4])
).sort('date').tail(10)


#%whos


import gc
global_vars = globals()

to_delete = [
    var_name for var_name in global_vars 
    if (isinstance(global_vars[var_name], pl.DataFrame) or isinstance(global_vars[var_name], pd.DataFrame))
    and var_name != 'act_logs'  
]

for var_name in to_delete:
    del global_vars[var_name]
    
gc.collect()


df_sorted = act_logs.sort(["ID", "Timestamp"])


# è�·å�–ä¸‹ä¸€ä¸ªåŠ¨ä½œ
df_pairs = df_sorted.with_columns(
    next_action=pl.col("ActionType").shift(-1).over("ID")
).filter(
    pl.col("next_action").is_not_null()
)

# ç»Ÿè®¡åŠ¨ä½œå¯¹é¢‘ç�‡
action_pairs = df_pairs.group_by(["ActionType", "next_action"]).agg(
    pl.count().alias("count")
).sort("count", descending=True)
action_pairs.head(10)


action_pairs.shape


df_unique = (
    df_sorted
    .with_columns(
        # æ ‡è®°æ˜¯å�¦ä¸�å‰�ä¸€ä¸ªåŠ¨ä½œç›¸å�Œ
        is_same_as_prev=pl.col("ActionType") == pl.col("ActionType").shift(1).over("ID")
    )
    .filter(~pl.col("is_same_as_prev"))  # è¿‡æ»¤æ�‰è¿�ç»­é‡�å¤�
    .drop("is_same_as_prev")
)

df_pairs = (
    df_unique
    .with_columns(
        next_action=pl.col("ActionType").shift(-1).over("ID")
    )
    .filter(pl.col("next_action").is_not_null())  # æ�’é™¤æœ€å��ä¸€ä¸ªåŠ¨ä½œ
)

transition_stats = (
    df_pairs
    .group_by(["ActionType", "next_action"])
    .agg(pl.count().alias("count"))
    .sort("count", descending=True)
)

print("é��è¿�ç»­åŠ¨ä½œè½¬ç§»ç»Ÿè®¡:")
transition_stats.head(10)


transition_stats.shape


'''
# è�·å�–å��ç»­ä¸¤ä¸ªåŠ¨ä½œ
df_sequences = df_sorted.with_columns(
    next_action1=pl.col("ActionType").shift(-1).over("ID"),
    next_action2=pl.col("ActionType").shift(-2).over("ID")
).filter(
    pl.col("next_action2").is_not_null()
)

# ç»Ÿè®¡ä¸‰å…ƒç»„é¢‘ç�‡
triples = df_sequences.group_by(
    ["ActionType", "next_action1", "next_action2"]
).agg(pl.count().alias("count")).sort("count", descending=True)
print(triples.head())
'''


invalid_order = df_sorted.with_columns(
    (pl.col("ActionType").diff() < 0).over("ID").alias("invalid")
).filter(pl.col("invalid")).select(pl.count())
print("å­˜åœ¨ActionTypeé€’å‡�çš„è®°å½•æ•°:", invalid_order.item())


# è½¬ä¸ºäº¤å�‰çŸ©é˜µæ ¼å¼�
transfer_matrix = action_pairs.pivot(
    values="count",
    index="ActionType",
    columns="next_action",
    aggregate_function="sum"
).fill_null(0)

print("\nçŠ¶æ€�è½¬ç§»çŸ©é˜µ:")
transfer_matrix


# è½¬æ�¢ä¸º Pandas DataFrameï¼ˆå…¼å®¹ Plotlyï¼‰
action_pairs_pd = action_pairs.to_pandas()


all_actions = pd.unique(
    action_pairs_pd[["ActionType", "next_action"]].values.ravel("K")
).tolist()

# åˆ›å»ºèŠ‚ç‚¹æ˜ å°„å­—å…¸
node_indices = {action: idx for idx, action in enumerate(all_actions)}

# ç”Ÿæˆ�æ¡‘åŸºå›¾é“¾è·¯æ•°æ�®
link_data = dict(
    source=action_pairs_pd["ActionType"].map(node_indices),
    target=action_pairs_pd["next_action"].map(node_indices),
    value=action_pairs_pd["count"],  
    color="rgba(31,119,180,0.6)"  
)

fig = go.Figure(go.Sankey(
    arrangement="snap",
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=all_actions,
        color="lightblue"
    ),
    link=link_data
))

fig.update_layout(title_text="Actionè½¬ç§»æ¡‘åŸºå›¾")
fig.show()


def build_transition_matrix(df,col):

    transitions = (
        df.sort(['ID', 'Timestamp'])
        .with_columns(
            next_action=pl.col(col).shift(-1).over("ID").cast(str)
        )
        .filter(pl.col('next_action').is_not_null())
        .select([
            pl.col(col).cast(str).alias('current_action'),
            pl.col('next_action')
        ])
    )

    transition_counts = (
        transitions
        .group_by(['current_action', 'next_action'])
        .agg(pl.count().alias("count"))
        .pivot(
            values='count',
            index='current_action',
            columns='next_action',
            aggregate_function='sum'
        )
        .fill_null(0)
    )

    all_actions = sorted(
        set(transition_counts['current_action']) |
        set(transition_counts.columns[1:])  
    )
    n = len(all_actions)
    action_index = {action: idx for idx, action in enumerate(all_actions)}

    prob_matrix = np.zeros((n, n))
    for row in transition_counts.iter_rows(named=True):
        current = row['current_action']
        row_total = sum(v for k, v in row.items() if k != 'current_action')
        
        if row_total == 0:
            current_idx = action_index[current]
            prob_matrix[current_idx, current_idx] = 1.0
            continue
        
        current_idx = action_index[current]
        for next_action in all_actions:
            count = row.get(next_action, 0)
            next_idx = action_index[next_action]
            prob_matrix[current_idx, next_idx] = count / row_total

    np.fill_diagonal(prob_matrix, np.where(prob_matrix.sum(axis=1) == 0, 1.0, np.diagonal(prob_matrix)))

    return pl.DataFrame(
        prob_matrix,
        schema={action: pl.Float64 for action in all_actions},
    ).with_columns(
        current_action=pl.Series(all_actions)
    ).select(['current_action'] + all_actions)

prob_df = build_transition_matrix(act_logs,'ActionType')
print('çŠ¶æ€�è½¬ç§»æ¦‚ç�‡çŸ©é˜µ:')
prob_df.head(10)


# ğŸ˜„è¿™é‡Œè¦�ç‰¹åˆ«æ„Ÿè°¢deepseekï¼Œæˆ‘çŸ¥è¯†éƒ½è¿˜ç»™è€�å¸ˆå•¦
actions = prob_df['current_action'].to_list()
prob_matrix = prob_df.drop('current_action').to_numpy()

def print_markov_properties(matrix, states):
    """åˆ†æ��é©¬å°”å�¯å¤«é“¾å…³é”®æ€§è´¨"""
    n = len(states)
    
    # 1. å�¸æ”¶æ€�æ£€æµ‹ï¼ˆå…�è®¸å¾®å°�è¯¯å·®ï¼‰
    absorbing_states = []
    for i in range(n):
        diagonal_ok = np.isclose(matrix[i, i], 1.0, atol=1e-6)
        others_zero = np.allclose(matrix[i, :], np.eye(n)[i], atol=1e-6)
        if diagonal_ok and others_zero:
            absorbing_states.append(states[i])
    
    print('\n[å�¸æ”¶æ€�æ£€æµ‹]')
    if absorbing_states:
        print(f'å�‘ç�° {len(absorbing_states)} ä¸ªå�¸æ”¶æ€�: {absorbing_states}')
    else:
        print('æœªå�‘ç�°å�¸æ”¶æ€�')

    # 2. ä¸�å�¯çº¦æ€§éªŒè¯�ï¼ˆç®€åŒ–ç‰ˆï¼‰
    # æ�„å»ºé‚»æ�¥çŸ©é˜µï¼ˆå­˜åœ¨è½¬ç§»å�³è§†ä¸ºå�¯è¾¾ï¼‰
    adj_matrix = (matrix > 1e-6).astype(int)
    
    # è®¡ç®—å�¯è¾¾çŸ©é˜µçš„å¹‚ï¼ˆå¸ƒå°”åŠ æ³•ï¼‰
    reachable = adj_matrix.copy()
    for _ in range(n-1):  
        reachable = (reachable @ adj_matrix) | reachable
        reachable = reachable.astype(bool)
    
    # æ£€æŸ¥æ˜¯å�¦æ‰€æœ‰çŠ¶æ€�å¯¹äº’ç›¸å�¯è¾¾
    is_irreducible = np.all(reachable)
    
    print('\n[ä¸�å�¯çº¦æ€§éªŒè¯�]')
    print("è¯¥é“¾æ˜¯'ä¸�å�¯çº¦çš„'" if is_irreducible else "è¯¥é“¾æ˜¯'å�¯çº¦çš„'")

    # 3. ç¨³æ€�åˆ†å¸ƒè®¡ç®—
    # å¯»æ‰¾ç‰¹å¾�å€¼1å¯¹åº”çš„å·¦ç‰¹å¾�å�‘é‡�
    eigenvalues, eigenvectors = np.linalg.eig(matrix.T)
    stationary_idx = np.where(np.isclose(eigenvalues, 1.0))[0]
    
    if len(stationary_idx) == 0:
        print('\n[ç¨³æ€�åˆ†å¸ƒ] ä¸�å­˜åœ¨å”¯ä¸€ç¨³æ€�åˆ†å¸ƒ')
        return
    
    # å�–å®�éƒ¨å¹¶å½’ä¸€åŒ–
    stationary = np.real(eigenvectors[:, stationary_idx[0]])
    stationary_dist = stationary / np.sum(stationary)
    
    # æ ¼å¼�åŒ–ä¸ºå�¯è¯»å­—å…¸
    stationary_dict = {
        states[i]: f"{prob:.4f}" 
        for i, prob in enumerate(stationary_dist)
    }
    
    print('\n[ç¨³æ€�åˆ†å¸ƒ] å�„çŠ¶æ€�é•¿æœŸæ¦‚ç�‡:')
    for state, prob in stationary_dict.items():
        print(f"{state}: {prob}")

    # 4. é¦–è¾¾æ—¶é—´åˆ†æ��ï¼ˆç¤ºä¾‹ï¼šä»�ç¬¬ä¸€ä¸ªçŠ¶æ€�åˆ°å…¶ä»–çŠ¶æ€�ï¼‰
    if not absorbing_states:
        print('\n[é¦–è¾¾æ—¶é—´] éœ€è¦�æ›´å¤�æ�‚çš„ç�¬æ€�åˆ†æ��ï¼Œå»ºè®®ä½¿ç”¨ä¸“ç”¨åº“')
    else:
        print('\n[é¦–è¾¾æ—¶é—´] å­˜åœ¨å�¸æ”¶æ€�æ—¶é¦–è¾¾æ—¶é—´è®¡ç®—éœ€è¦�ç‰¹æ®Šå¤„ç�†')

    return stationary_dist
# æ‰§è¡Œåˆ†æ��
_ = print_markov_properties(prob_matrix, actions)


from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth

# æ��å�–æ¯�ä¸ªç”¨æˆ·çš„åŠ¨ä½œåº�åˆ—
sequences = df_sorted.group_by("ID").agg(
    pl.col("ActionType").alias("sequence")
)["sequence"].to_list()

# ä½¿ç”¨FP-Growthç®—æ³•æ‰¾é¢‘ç¹�æ¨¡å¼�
te = TransactionEncoder()
te_ary = te.fit(sequences).transform(sequences)
df_te = pd.DataFrame(te_ary, columns=te.columns_)
frequent_itemsets = fpgrowth(df_te, min_support=0.1, use_colnames=True)
print(frequent_itemsets.sort_values("support", ascending=False))


# è�·å�–ä¸‹ä¸€ä¸ªåŠ¨ä½œ
df_pairs = df_sorted.with_columns(
    next_action=pl.col("ActionId").shift(-1).over("ID")
).filter(
    pl.col("next_action").is_not_null()
)

# ç»Ÿè®¡åŠ¨ä½œå¯¹é¢‘ç�‡
action_pairs = df_pairs.group_by(["ActionId", "next_action"]).agg(
    pl.count().alias("count")
).sort("count", descending=True)
action_pairs.head(10)


action_pairs.shape


print(f'æ‰€æœ‰å�¯èƒ½åº”è¯¥{130*130}ç§�æ‰�å¯¹ï¼Œå°‘äº†{130*130-11510}ç§�')


'''
df_sequences = df_sorted.with_columns(
    next_action1=pl.col("ActionId").shift(-1).over("ID"),
    next_action2=pl.col("ActionId").shift(-2).over("ID")
).filter(
    pl.col("next_action2").is_not_null()
)

triples = df_sequences.groupby(
    ["ActionId", "next_action1", "next_action2"]
).agg(pl.count().alias("count")).sort("count", descending=True)
triples.head()
'''


df_unique = (
    df_sorted
    .with_columns(
        is_same_as_prev=pl.col("ActionId") == pl.col("ActionId").shift(1).over("ID")
    )
    .filter(~pl.col('is_same_as_prev')) 
    .drop("is_same_as_prev")
)

df_pairs = (
    df_unique
    .with_columns(
        next_action=pl.col("ActionId").shift(-1).over("ID")
    )
    .filter(pl.col("next_action").is_not_null())  
)

transition_stats = (
    df_pairs
    .group_by(["ActionId", "next_action"])
    .agg(pl.count().alias("count"))
    .sort("count", descending=True)
)

print("é��è¿�ç»­åŠ¨ä½œè½¬ç§»ç»Ÿè®¡:")
transition_stats.head()


invalid_order = df_sorted.with_columns(
    (pl.col("ActionId").diff() < 0).over("ID").alias("invalid")
).filter(pl.col("invalid")).select(pl.count())
print("å­˜åœ¨ActionIdé€’å‡�çš„è®°å½•æ•°:", invalid_order.item())


transfer_matrix = action_pairs.pivot(
    values="count",
    index="ActionId",
    columns="next_action",
    aggregate_function="sum"
).fill_null(0)

print("\nçŠ¶æ€�è½¬ç§»çŸ©é˜µ:")
transfer_matrix.head()


action_pairs_pd = action_pairs.to_pandas()
prob_df = build_transition_matrix(act_logs,'ActionId')
print("çŠ¶æ€�è½¬ç§»æ¦‚ç�‡çŸ©é˜µ:")
prob_df.head()


actions = prob_df["current_action"].to_list()
prob_matrix = prob_df.drop("current_action").to_numpy()
stable_state = print_markov_properties(prob_matrix, actions)


pd.DataFrame(stable_state).sort_values(by=0,ascending=False)


# ç»Ÿè®¡æ¯�ä¸ªæ“�ä½œä½œä¸ºæœ€å��ä¸€æ�¡è®°å½•çš„æ¬¡æ•°
terminal_actions = (
    act_logs.sort(['ID', 'Timestamp'])
    .group_by('ID')
    .agg(pl.last('ActionId').alias('last_action'))
    .group_by('last_action')
    .agg(pl.count().alias('terminal_count'))
    .sort('terminal_count', descending=True)
)

print('ç–‘ä¼¼ç»ˆæ­¢æ“�ä½œå�Šå‡ºç�°æ¬¡æ•°:')
terminal_actions.head(10)


# æ‰§è¡Œæ—¶é—´è¿‡é•¿ï¼Œæ”¾å¼ƒ
#sequences = df_sorted.group_by("ID").agg(
#    pl.col("ActionId").alias("sequence")
#)["sequence"].to_list()
#
#te = TransactionEncoder()
#te_ary = te.fit(sequences).transform(sequences)
#df_te = pd.DataFrame(te_ary, columns=te.columns_)
#frequent_itemsets = fpgrowth(df_te, min_support=0.1, use_colnames=True)
#print(frequent_itemsets.sort_values("support", ascending=False))


display_opinion_box("""
1. æ¯�å¤©çš„å�Œä¸€ä¸ªç”¨æˆ·é‡Œé�¢ï¼Œæ“�ä½œç±»å�‹ä¹‹é—´æ²¡æœ‰éš”ç¦»ï¼Œå�¯ä»¥äº’ç›¸åˆ°è¾¾<br>
2. æ“�ä½œidå­˜åœ¨ä¸€äº›ç»ˆç‚¹ï¼Œå�¯èƒ½æ˜¯åŠŸèƒ½æ¨¡å�—çš„ç»“æ�Ÿç‚¹ï¼Œæ•°æ�®å�ªæœ‰ä¸€ä¸ªæœˆï¼Œè¿˜æ— æ³•å‡†ç¡®æ‰¾åˆ°è¿™äº›ç»“æ�Ÿç‚¹ã€‚<br>
3. é•¿æœŸæ�¥çœ‹ï¼Œç”¨æˆ·å°†62%çš„æ—¶é—´åœ¨æ“�ä½œç±»å�‹1é‡Œã€‚<br>
""")


opinion_text = """
ä¸šåŠ¡å±‚é�¢ä¸Šçœ‹<br>  
1. ç”¨æˆ·æœ‰äº›é›†ä¸­æŸ�äº›ç±»å�‹çš„æ“�ä½œï¼Œè€ƒè™‘è¿™äº›ç±»å�‹æ˜¯ä¸�æ˜¯å½“å‰�æƒ³è¦�çš„ï¼Œå¦‚æ�œä¸�æ˜¯ï¼Œåº”è¯¥è€ƒè™‘ä¸€äº›è®¾ç½®å°†ç”¨æˆ·å¼•å¯¼åˆ°å…¶ä»–ç±»å�‹çš„æ“�ä½œã€‚<br>
2. å­˜åœ¨ä¸€äº›åŠŸèƒ½ä¸Šçš„éš”ç¦»ï¼Œä½¿å¾—ç”¨æˆ·æ— æ³•è§¦è¾¾æŸ�äº›åŠŸèƒ½ã€‚æ˜¯å�¦åº”è¯¥è€ƒè™‘å¢�åŠ ä¸€äº›è·³è½¬å…¥å�£ã€‚<br>
"""


html_content = f"""
<div style="background-color: #DAA520; border-radius: 10px; padding: 15px; border: 1px solid ; color:white;">
  <strong>ğŸ’¡ Insights <br></strong> {opinion_text}
</div>
"""
display(HTML(html_content))


global_vars = globals()
print(len(global_vars))

to_delete = [
    var_name for var_name in global_vars 
    if (isinstance(global_vars[var_name], pl.DataFrame) or isinstance(global_vars[var_name], pd.DataFrame))
    and var_name != 'act_logs'
]

for var_name in to_delete:
    del global_vars[var_name]
    
gc.collect()


from datetime import datetime

import xgboost as xgb 
import lightgbm as lgb 
import catboost as cat 

from tqdm import tqdm 
from sklearn.model_selection import KFold,TimeSeriesSplit
from sklearn.preprocessing import OrdinalEncoder

import optuna 
import os 


sub = pl.read_csv(os.path.join(DATA_PATH, 'submit_sample.csv'))
daily_act_with_label = pl.read_csv(os.path.join('/kaggle/input/supplementary-data/', 'daily_act_with_label_2.csv'))#æ ‡ç­¾è®¡ç®—å�‚è€ƒè¿™ä¸ª https://www.kaggle.com/code/linjianhang/user-s-daily-label-compute


daily_act_with_label = daily_act_with_label.with_columns(
    pl.col('Date').cast(pl.Date),
)
daily_act_with_label.head()


top_action_ids = (
    act_logs["ActionId"].value_counts()
    .sort("count", descending=True)["ActionId"]
    .head(10).to_list()
)


daily_stats = act_logs.group_by(["ID", "date"]).agg(
    pl.count().alias("Daily_total_act_times"),
    pl.n_unique("ActionType").alias("Daily_unique_act_types"),
    pl.n_unique("ActionId").alias("Daily_unique_act_id"),
    
    # æœ€å��ä¸€æ¬¡æ´»è·ƒçš„è¯¦ç»†ä¿¡æ�¯
    pl.last("ActionType").alias("last_activity_type"),
    pl.last("ActionId").alias("last_activity_id"),
    
    # ç¬¬ä¸€æ¬¡æ´»è·ƒçš„è¯¦ç»†ä¿¡æ�¯
    pl.first("ActionType").alias("first_activity_type"),
    pl.first("ActionId").alias("first_activity_id"),

    # æœ€å¤§æ—¶é—´é—´éš”
    (pl.col("Timestamp").max() - pl.col("Timestamp").min()).dt.total_seconds().fill_null(-1).alias("total_time_spent"),
    
    # å¹³å�‡æ—¶é—´é—´éš”
    pl.col("Timestamp").sort().diff().mean().dt.total_seconds().fill_null(-1).alias("avg_time_gap_between_act"),

    # å�„ç±»æ´»è·ƒè®¡æ•°
    *(
        pl.col("ActionType").filter(pl.col("ActionType") == act_type).count().alias(f"act_type_{act_type}_count")
        for act_type in act_logs["ActionType"].unique()
    ),
    #*(
    #    pl.col("ActionType").filter(pl.col("ActionType") == act_type).n_unique().alias(f"act_type_{act_type}_unique")
    #    for act_type in act_logs["ActionType"].unique()
    #),

    *(
        pl.col("ActionId").filter(pl.col("ActionId") == act_id).count().alias(f"act_id_{act_id}_count")
        for act_id in top_action_ids
    ),
    #*(
    #    pl.col("ActionId").filter(pl.col("ActionId") == act_id).n_unique().alias(f"act_id_{act_id}_unique")
    #    for act_id in top_action_ids
    #)
)

daily_stats.head()


act_logs = act_logs.sort(['ID',"Timestamp"])
act_date = act_logs.select(
    pl.col('ID'),
    pl.col('date')
).unique()


daily_stats2 = (
    act_date.sort(["ID", "date"])  
    .with_columns(
        *[
            (pl.col("date") - pl.col("date").shift(i).over("ID")).dt.total_days()
            .fill_null(-1)  
            .alias(f"Days_Since_Last_{i}_Active")
            for i in range(1, 8)  # è®¡ç®— å‰�1-7 æ¬¡æ´»è·ƒçš„æ´»è·ƒé—´éš”å¤©æ•°
        ]
    )
)


daily_act_with_label = daily_act_with_label.drop_nulls()


test = sub.with_columns(
    Date = datetime(2018,10,20)
).with_columns(
    pl.col('Date').cast(pl.Date)
)
test.head()


test = test.join(
    daily_stats,
    left_on=["ID", "Date"],
    right_on=["ID", "date"],
    how="left"
)

daily_act_with_label =  daily_act_with_label.join(
    daily_stats,
    left_on=["ID", "Date"],
    right_on=["ID", "date"],
    how="left"
)


test = test.join(
    daily_stats2,
    left_on=["ID", "Date"],
    right_on=["ID", "date"],
    how="left"
)

daily_act_with_label =  daily_act_with_label.join(
    daily_stats2,
    left_on=["ID", "Date"],
    right_on=["ID", "date"],
    how="left"
)


daily_act_with_label = daily_act_with_label.with_columns(
    pl.col('Date').dt.month().alias('month'),
    pl.col('Date').dt.day().alias('day'),
    pl.col('Date').dt.weekday().alias('weekday'),
    pl.col('Date').dt.week().alias('week'),
)

test = test.with_columns(
    pl.col('Date').dt.month().alias('month'),
    pl.col('Date').dt.day().alias('day'),
    pl.col('Date').dt.weekday().alias('weekday'),
    pl.col('Date').dt.week().alias('week'),
)


daily_act_with_label = daily_act_with_label.sort('Date')
daily_act_with_label = daily_act_with_label.drop('Date')
test = test.drop(['Date','pred'])


daily_act_with_label = daily_act_with_label.to_pandas()
test = test.to_pandas()
sub = sub.to_pandas()


encode_cols = [
    'last_activity_type',
    'last_activity_id',
    'first_activity_type',
    'first_activity_id'
]

oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

oe.fit(daily_act_with_label[encode_cols])
encoded_train = oe.transform(daily_act_with_label[encode_cols])
encoded_train_df = pd.DataFrame(encoded_train, columns=encode_cols, index=daily_act_with_label.index)
daily_act_with_label[encode_cols] = encoded_train_df

encoded_test = oe.transform(test[encode_cols])
encoded_test_df = pd.DataFrame(encoded_test, columns=encode_cols, index=test.index)
test[encode_cols] = encoded_test_df


features = list(test.columns)
features.remove('ID')
X = daily_act_with_label[features]
y = daily_act_with_label['Label']
test = test[features]
TAREGT = 'Label'


def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred))
    # é�¿å…�é™¤ä»¥é›¶ï¼šå½“çœŸå®�å€¼å’Œé¢„æµ‹å€¼éƒ½ä¸ºé›¶æ—¶ï¼Œè¯¥æ ·æœ¬çš„è¯¯å·®è§†ä¸º 0
    smape_value = np.where(denominator == 0, 0.0, 2 * np.abs(y_pred - y_true) / denominator)
    return np.mean(smape_value) * 100  


def cross_validationR(estimator, X, y, sub = sub,test=None, target=None, if_output=False, if_plot_importance=False):
    train_scores = []
    valid_scores = []
    oof = np.zeros(X.shape[0])
    output = pd.DataFrame()
    thresholds=np.cumsum([0.11310252, 0.06577263, 0.04719162, 0.05515901, 0.05475995,0.07644968, 0.12103906, 0.46647806])
    
    #KF = KFold(n_splits=5)
    KF = TimeSeriesSplit(n_splits=5)
    pbar = tqdm(KF.split(X, y), total=KF.get_n_splits(), desc="Cross Validation")
    for i, (train_idx, valid_idx) in enumerate(pbar):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

        estimator.fit(X_train, y_train)
        train_pred = np.round(estimator.predict(X_train))
        valid_pred = np.round(estimator.predict(X_valid))

        train_score = smape(y_train, train_pred)
        valid_score = smape(y_valid, valid_pred)

        train_scores.append(train_score)
        valid_scores.append(valid_score)

        oof[valid_idx] = valid_pred

        if if_output and test is not None:
            output[i] = estimator.predict(test)
            
    print(f'The train score : {np.mean(train_scores)}')
    print(f'The valid score : {np.mean(valid_scores)}')
    
    oof_score = smape(y, oof)
    print(f'The oof score: {oof_score}')

    if if_output and test is not None:
        output = output.mean(axis=1)
        #å��å¤„ç�† å�‚è€ƒè¿™ä¸ª ï¼š https://www.kaggle.com/code/dc5e964768ef56302a32/4-catboost-regression
        sub['pred'] = output
        sub["pred"]=sub["pred"].rank()
        sub["pred"]=sub["pred"]/(sub["pred"].max())
        sub["pred"]=np.digitize(sub["pred"], thresholds).clip(0,7)
        
        output_dir = f'./ç»“æ�œé›†/{type(estimator).__name__}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        sub.to_csv(f'{output_dir}/{type(estimator).__name__}_{oof_score:.6f}.csv', index=False)

    def plot_importances():
        importances = pd.DataFrame({'cols': X.columns, 'importance': estimator.feature_importances_})
        importances = importances.sort_values('importance', ascending=False)
        
        n_row = importances.shape[0]

        fig , ax = plt.subplots(figsize=(12, 0.2 * n_row))
        sns.barplot(data=importances, x='importance', y='cols' , ax = ax)
        plt.title('Feature Importances')
        plt.show()

    if if_plot_importance:
        plot_importances()

    return np.mean(oof_score)


def optuna_paramsR(estimator_class, params, n_trials=30):
    def objective(trial):
        trial_params = {}
        for key, value in params.items():
            if isinstance(value, tuple) and len(value) == 2:
                if isinstance(value[0], int) and isinstance(value[1], int):
                    trial_params[key] = trial.suggest_int(key, value[0], value[1])
                else:
                    trial_params[key] = trial.suggest_float(key, value[0], value[1])
            elif isinstance(value, list):
                trial_params[key] = trial.suggest_categorical(key, value)
            else:
                trial_params[key] = value

        model = estimator_class(**trial_params)
        return cross_validationR(
            estimator=model, X=X, y=y, test=test, target=TAREGT
        )

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    print(f'Best trial: {study.best_trial.value}')
    print(f'Best params: {study.best_trial.params}')

    return study.best_trial.params


default_xgb_setting = {
    'device':'cuda',
    'verbose':-1
}

'''
default_lgb_setting = {
    'device':'gpu',
    'verbose':-1
}

default_cat_setting = {
    'bootstrap_type': 'Bernoulli',
    'task_type':'GPU',
    'verbose':0
}
'''
#default_xgb = xgb.XGBClassifier (**default_xgb_setting)
#default_lgb = lgb.LGBMClassifier(**default_lgb_setting)
#default_cat = cat.CatBoostClassifier(**default_cat_setting)

default_xgb = xgb.XGBRegressor(**default_xgb_setting)
#default_lgb = lgb.LGBMRegressor(**default_lgb_setting)
#default_cat = cat.CatBoostRegressor(**default_cat_setting)


params_xgb = {
    'n_estimators': (100, 2000),  
    'max_depth': (1, 5),  
    'learning_rate': (0.01, 0.2),  
    'subsample': (0.4, 1.0),  
    'colsample_bytree': (0.4, 1.0), 
    'min_child_weight': (1, 15),
    
}
params_xgb.update(default_xgb_setting)

xgb_best_params = optuna_paramsR(xgb.XGBRegressor, params_xgb,n_trials=5)
xgb_best_params.update(default_xgb_setting)

best_xgb = xgb.XGBRegressor(**xgb_best_params)
cross_validationR(best_xgb, X=X, y=y,target=TAREGT,test=test,if_output=True,if_plot_importance=True)


# çœ‹ä¸‹è®­ç»ƒé›†æ€»ä½“åˆ†å¸ƒ
train_dis = y.value_counts()/y.shape[0]
pre_dis = sub['pred'].value_counts()/sub.shape[0]

fig , ax = plt.subplots(figsize=(12,6))
ax.bar(train_dis.index , train_dis.values , color='b')
ax.bar(pre_dis.index , -pre_dis.values , color='g')


# å��å¤„ç�†å��çš„åˆ†å¸ƒ
pre_dis.sort_index()




