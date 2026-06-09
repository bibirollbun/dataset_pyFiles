!pip install --upgrade plotly


# é€™å€‹ Python 3 ç’°å¢ƒé �è£�äº†è¨±å¤šå¯¦ç”¨çš„åˆ†æ��å‡½å¼�åº«
# å®ƒæ˜¯ä»¥ kaggle/python Docker æ˜ åƒ�æª”ç‚ºåŸºç¤�æ‰€å®šç¾©ï¼š[https://github.com/kaggle/docker-python](https://github.com/kaggle/docker-python)
# ä¾‹å¦‚ï¼Œä»¥ä¸‹æ˜¯ä¸€äº›è¼‰å…¥çš„å¯¦ç”¨å¥—ä»¶

import numpy as np # ç·šæ€§ä»£æ•¸
import pandas as pd # è³‡æ–™è™•ç�†ã€�CSV æª”æ¡ˆè¼¸å…¥/è¼¸å‡º (ä¾‹å¦‚ pd.read_csv)

# è¼¸å…¥è³‡æ–™æª”æ¡ˆä½�æ–¼å”¯è®€çš„ "../input/" ç›®éŒ„ä¸‹
# ä¾‹å¦‚ï¼ŒåŸ·è¡Œæ­¤è™• (é»�æ“Š "run" æˆ–æŒ‰ä¸‹ Shift+Enter) å°‡æœƒåˆ—å‡ºè¼¸å…¥ç›®éŒ„ä¸‹çš„æ‰€æœ‰æª”æ¡ˆ

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# ä½ å�¯ä»¥å°‡æœ€å¤š 20GB çš„è³‡æ–™å¯«å…¥ç›®å‰�ç›®éŒ„ (/kaggle/working/)ï¼Œç•¶ä½ ä½¿ç”¨ "Save & Run All" å»ºç«‹ç‰ˆæœ¬æ™‚ï¼Œé€™äº›è³‡æ–™æœƒè¢«å„²å­˜ä¸‹ä¾†
# ä½ ä¹Ÿå�¯ä»¥å°‡æš«å­˜æª”æ¡ˆå¯«å…¥ /kaggle/temp/ï¼Œä½†é€™äº›æª”æ¡ˆåœ¨ç›®å‰�å·¥ä½œéš�æ®µçµ�æ�Ÿå¾Œä¸�æœƒè¢«å„²å­˜


import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("âœ… æ‰€æœ‰å¥—ä»¶å°�å…¥å®Œæˆ�ï¼�")


# è¼‰å…¥æ•¸æ“š
train = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')
test = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')
submit = pd.read_csv('/kaggle/input/bike-sharing-demand/sampleSubmission.csv')

print(f"è¨“ç·´é›†å¤§å°�: {train.shape}")
print(f"æ¸¬è©¦é›†å¤§å°�: {test.shape}")
print(f"æ��äº¤æª”æ¡ˆå¤§å°�: {submit.shape}")

# é¡¯ç¤ºè¨“ç·´é›†å‰�å¹¾è¡Œ
print("\nè¨“ç·´é›†å‰�5è¡Œ:")
display(train.head())

# æª¢æŸ¥ç¼ºå¤±å€¼
print(f"\nè¨“ç·´é›†ç¼ºå¤±å€¼: {train.isnull().sum().sum()}")
print(f"æ¸¬è©¦é›†ç¼ºå¤±å€¼: {test.isnull().sum().sum()}")


# ç›®æ¨™è®Šæ•¸åˆ†å¸ƒåœ–
fig = px.histogram(train, x='count', nbins=50,
                   title='å…±äº«å–®è»Šéœ€æ±‚é‡�åˆ†å¸ƒ',
                   labels={'count': 'éœ€æ±‚é‡�', 'y': 'é »ç�‡'})
fig.update_layout(showlegend=False)
fig.show()


# è¨“ç·´é›†åŸºæœ¬çµ±è¨ˆ
print("è¨“ç·´é›†åŸºæœ¬çµ±è¨ˆ:")
display(train.describe())


def detect_outliers_iqr(data, column):
    """ä½¿ç”¨IQRæ–¹æ³•æª¢æ¸¬é›¢ç¾¤å€¼"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = (data[column] < lower_bound) | (data[column] > upper_bound)
    return outliers, lower_bound, upper_bound

# æª¢æ¸¬é›¢ç¾¤å€¼
outliers, lower, upper = detect_outliers_iqr(train, 'count')
print(f"é›¢ç¾¤å€¼æ•¸é‡�: {outliers.sum()}")
print(f"é›¢ç¾¤å€¼æ¯”ä¾‹: {outliers.sum()/len(train)*100:.2f}%")

# é›¢ç¾¤å€¼è¦–è¦ºåŒ–
fig = px.box(train, y='count', title='éœ€æ±‚é‡�ç®±å�‹åœ– - é›¢ç¾¤å€¼æª¢æ¸¬')
fig.add_hline(y=lower, line_dash="dash", line_color="red",
              annotation_text=f"ä¸‹ç•Œ: {lower:.0f}")
fig.add_hline(y=upper, line_dash="dash", line_color="red",
              annotation_text=f"ä¸Šç•Œ: {upper:.0f}")
fig.show()


# ç§»é™¤é›¢ç¾¤å€¼
print(f"ç§»é™¤é›¢ç¾¤å€¼å‰�: {train.shape}")
train_clean = train[~outliers].copy()
print(f"ç§»é™¤é›¢ç¾¤å€¼å¾Œ: {train_clean.shape}")

# ç§»é™¤é›¢ç¾¤å€¼å¾Œçš„åˆ†å¸ƒ
fig = px.histogram(train_clean, x='count', nbins=50,
                   title='ç§»é™¤é›¢ç¾¤å€¼å¾Œçš„éœ€æ±‚é‡�åˆ†å¸ƒ',
                   labels={'count': 'éœ€æ±‚é‡�', 'y': 'é »ç�‡'})
fig.update_layout(showlegend=False)
fig.show()


# å�ˆä½µè¨“ç·´å’Œæ¸¬è©¦æ•¸æ“šä»¥é€²è¡Œå®Œæ•´åˆ†æ��
all_data = pd.concat([train_clean, test], ignore_index=True)

# æº«åº¦åˆ†å¸ƒ
fig = px.histogram(all_data, x='temp', nbins=30,
                   title='æº«åº¦åˆ†å¸ƒ',
                   labels={'temp': 'æº«åº¦', 'y': 'é »ç�‡'})
fig.update_layout(showlegend=False)
fig.show()


# é«”æ„Ÿæº«åº¦åˆ†å¸ƒ
fig = px.histogram(all_data, x='atemp', nbins=30,
                   title='é«”æ„Ÿæº«åº¦åˆ†å¸ƒ',
                   labels={'atemp': 'é«”æ„Ÿæº«åº¦', 'y': 'é »ç�‡'})
fig.update_layout(showlegend=False)
fig.show()


# æ¿•åº¦åˆ†å¸ƒ
fig = px.histogram(all_data, x='humidity', nbins=30,
                   title='æ¿•åº¦åˆ†å¸ƒ',
                   labels={'humidity': 'æ¿•åº¦', 'y': 'é »ç�‡'})
fig.update_layout(showlegend=False)
fig.show()


# é¢¨é€Ÿåˆ†å¸ƒ
fig = px.histogram(all_data, x='windspeed', nbins=30,
                   title='é¢¨é€Ÿåˆ†å¸ƒ',
                   labels={'windspeed': 'é¢¨é€Ÿ', 'y': 'é »ç�‡'})
fig.update_layout(showlegend=False)
fig.show()


# æª¢æŸ¥é¢¨é€Ÿç‚º0çš„æƒ…æ³�
wind_zero_count = (all_data['windspeed'] == 0).sum()
print(f"é¢¨é€Ÿç‚º0çš„è¨˜éŒ„æ•¸: {wind_zero_count}")
print(f"é¢¨é€Ÿç‚º0çš„æ¯”ä¾‹: {wind_zero_count/len(all_data)*100:.2f}%")

# é¢¨é€Ÿç‚º0çš„åˆ†å¸ƒè¦–è¦ºåŒ–
wind_status = all_data['windspeed'].apply(lambda x: 'é¢¨é€Ÿç‚º0' if x == 0 else 'é¢¨é€Ÿå¤§æ–¼0')
fig = px.histogram(all_data, x='windspeed', color=wind_status,
                   title='é¢¨é€Ÿåˆ†å¸ƒ - å�€åˆ†æ˜¯å�¦ç‚º0',
                   labels={'windspeed': 'é¢¨é€Ÿ', 'y': 'é »ç�‡'})
fig.show()


def fill_windspeed_zero(data):
    """ä½¿ç”¨éš¨æ©Ÿæ£®æ�—å¡«è£œé¢¨é€Ÿç‚º0çš„å€¼"""
    # åˆ†é›¢æœ‰é¢¨é€Ÿå’Œç„¡é¢¨é€Ÿçš„æ•¸æ“š
    data_with_wind = data[data['windspeed'] != 0].copy()
    data_without_wind = data[data['windspeed'] == 0].copy()

    if len(data_without_wind) > 0:
        # é�¸æ“‡å¡«è£œç‰¹å¾µ
        wind_features = ['season', 'weather', 'humidity', 'temp', 'atemp']

        # è¨“ç·´éš¨æ©Ÿæ£®æ�—æ¨¡å�‹
        rf_wind = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_wind.fit(data_with_wind[wind_features], data_with_wind['windspeed'])

        # é �æ¸¬é¢¨é€Ÿ
        predicted_windspeed = rf_wind.predict(data_without_wind[wind_features])
        data_without_wind.loc[:, 'windspeed'] = predicted_windspeed

        # å�ˆä½µæ•¸æ“š
        filled_data = pd.concat([data_with_wind, data_without_wind], ignore_index=True)
        filled_data = filled_data.sort_values('datetime').reset_index(drop=True)

        return filled_data, rf_wind
    else:
        return data, None

# å¡«è£œé¢¨é€Ÿ
all_data_filled, wind_model = fill_windspeed_zero(all_data)

# å¡«è£œå¾Œçš„é¢¨é€Ÿåˆ†å¸ƒ
fig = px.histogram(all_data_filled, x='windspeed', nbins=30,
                   title='å¡«è£œå¾Œçš„é¢¨é€Ÿåˆ†å¸ƒ',
                   labels={'windspeed': 'é¢¨é€Ÿ', 'y': 'é »ç�‡'})
fig.update_layout(showlegend=False)
fig.show()


def extract_time_features(data):
    """æ��å�–æ™‚é–“ç›¸é—œç‰¹å¾µ"""
    data = data.copy()

    # åˆ†å‰²æ—¥æœŸæ™‚é–“
    data['date'] = data['datetime'].apply(lambda x: x.split()[0])
    data['hour'] = data['datetime'].apply(lambda x: int(x.split()[1].split(':')[0]))
    data['year'] = data['datetime'].apply(lambda x: int(x.split()[0].split('-')[0]))
    data['month'] = data['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').month)
    data['weekday'] = data['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').weekday())

    # é€²éš�æ™‚é–“ç‰¹å¾µ
    data['is_weekend'] = data['weekday'].apply(lambda x: 1 if x >= 5 else 0)
    data['is_rush_hour'] = data['hour'].apply(lambda x: 1 if x in [7,8,9,17,18,19] else 0)
    data['is_working_day'] = ((data['weekday'] < 5) & (data['holiday'] == 0)).astype(int)

    return data

# æ��å�–æ™‚é–“ç‰¹å¾µ
all_data_featured = extract_time_features(all_data_filled)

# æ¯�å°�æ™‚éœ€æ±‚é‡�è®ŠåŒ–ï¼ˆåƒ…ä½¿ç”¨è¨“ç·´æ•¸æ“šï¼‰
train_featured = all_data_featured[pd.notnull(all_data_featured['count'])]
hourly_demand = train_featured.groupby('hour')['count'].mean().reset_index()

fig = px.line(hourly_demand, x='hour', y='count',
              title='æ¯�å°�æ™‚å¹³å�‡éœ€æ±‚é‡�è®ŠåŒ–',
              labels={'hour': 'å°�æ™‚', 'count': 'å¹³å�‡éœ€æ±‚é‡�'})
fig.update_traces(mode='lines+markers')
fig.show()


# æ¯�æœˆéœ€æ±‚é‡�è®ŠåŒ–
monthly_demand = train_featured.groupby('month')['count'].mean().reset_index()
month_names = ['ä¸€æœˆ', 'äºŒæœˆ', 'ä¸‰æœˆ', 'å››æœˆ', 'äº”æœˆ', 'å…­æœˆ',
               'ä¸ƒæœˆ', 'å…«æœˆ', 'ä¹�æœˆ', 'å��æœˆ', 'å��ä¸€æœˆ', 'å��äºŒæœˆ']
monthly_demand['month_name'] = monthly_demand['month'].apply(lambda x: month_names[x-1])

fig = px.bar(monthly_demand, x='month_name', y='count',
             title='æ¯�æœˆå¹³å�‡éœ€æ±‚é‡�è®ŠåŒ–',
             color='month_name', # ä¿®æ­£é€™è£¡ï¼Œä½¿ç”¨ 'month_name' æ¬„ä½�
             labels={'month_name': 'æœˆä»½', 'count': 'å¹³å�‡éœ€æ±‚é‡�'})
fig.update_layout(legend=dict(x=0.5, y=1.0, xanchor='center', yanchor='bottom', orientation='h'))
fig.show()


# å·¥ä½œæ—¥vsé€±æœ«éœ€æ±‚é‡�æ¯”è¼ƒ
weekend_comparison = train_featured.groupby('is_weekend')['count'].mean().reset_index()
weekend_comparison['day_type'] = weekend_comparison['is_weekend'].apply(
    lambda x: 'é€±æœ«' if x == 1 else 'å·¥ä½œæ—¥')

fig = px.bar(weekend_comparison, x='day_type', y='count',
             title='å·¥ä½œæ—¥ vs é€±æœ«å¹³å�‡éœ€æ±‚é‡�',
             color='day_type', # ä¿®æ­£é€™è£¡ï¼Œä½¿ç”¨ 'day_type' æ¬„ä½�
             labels={'day_type': 'æ—¥æœŸé¡�å�‹', 'count': 'å¹³å�‡éœ€æ±‚é‡�'})
fig.update_layout(legend=dict(x=0.5, y=1.0, xanchor='center', yanchor='bottom', orientation='h'))
fig.show()


# å­£ç¯€èˆ‡éœ€æ±‚é‡�é—œä¿‚
season_names = {1: 'æ˜¥å­£', 2: 'å¤�å­£', 3: 'ç§‹å­£', 4: 'å†¬å­£'}
train_featured['season_name'] = train_featured['season'].map(season_names)
season_demand = train_featured.groupby('season_name')['count'].mean().reset_index()

fig = px.bar(season_demand, x='season_name', y='count',
             title='ä¸�å�Œå­£ç¯€å¹³å�‡éœ€æ±‚é‡�',
             color='season_name', # ä¿®æ­£é€™è£¡ï¼Œä½¿ç”¨ 'season_name' æ¬„ä½�
             labels={'season_name': 'å­£ç¯€', 'count': 'å¹³å�‡éœ€æ±‚é‡�'})
fig.update_layout(legend=dict(x=0.5, y=1.0, xanchor='center', yanchor='bottom', orientation='h'))
fig.show()


# å¤©æ°£ç‹€æ³�èˆ‡éœ€æ±‚é‡�é—œä¿‚
weather_names = {1: 'æ™´æœ—', 2: 'è–„é›²/è–„éœ§', 3: 'å°�é›¨/å°�é›ª', 4: 'å¤§é›¨/æš´é›ª'}
train_featured['weather_name'] = train_featured['weather'].map(weather_names)
weather_demand = train_featured.groupby('weather_name')['count'].mean().reset_index()

fig = px.bar(weather_demand, x='weather_name', y='count',
             title='ä¸�å�Œå¤©æ°£ç‹€æ³�å¹³å�‡éœ€æ±‚é‡�',
             color='weather_name', # ä¿®æ­£é€™è£¡ï¼Œä½¿ç”¨ 'weather_name' æ¬„ä½�
             labels={'weather_name': 'å¤©æ°£ç‹€æ³�', 'count': 'å¹³å�‡éœ€æ±‚é‡�'})
fig.update_layout(legend=dict(x=0.5, y=1.0, xanchor='center', yanchor='bottom', orientation='h'))
fig.show()


# æº«åº¦èˆ‡éœ€æ±‚é‡�æ•£é»�åœ–
fig = px.scatter(train_featured, x='temp', y='count',
                 title='æº«åº¦ vs éœ€æ±‚é‡�é—œä¿‚',
                 labels={'temp': 'æº«åº¦', 'count': 'éœ€æ±‚é‡�'},
                 trendline='ols')
fig.show()


# æ¿•åº¦èˆ‡éœ€æ±‚é‡�æ•£é»�åœ–
fig = px.scatter(train_featured, x='humidity', y='count',
                 title='æ¿•åº¦ vs éœ€æ±‚é‡�é—œä¿‚',
                 labels={'humidity': 'æ¿•åº¦', 'count': 'éœ€æ±‚é‡�'},
                 trendline='ols')
fig.show()


# é�¸æ“‡æ•¸å€¼ç‰¹å¾µé€²è¡Œç›¸é—œæ€§åˆ†æ��
numeric_features = ['temp', 'atemp', 'humidity', 'windspeed', 'count']
correlation_matrix = train_featured[numeric_features].corr()

# ç›¸é—œæ€§ç†±åŠ›åœ–
fig = px.imshow(correlation_matrix,
                title='ç‰¹å¾µç›¸é—œæ€§ç†±åŠ›åœ–',
                labels=dict(x="ç‰¹å¾µ", y="ç‰¹å¾µ", color="ç›¸é—œä¿‚æ•¸"),
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                color_continuous_scale='RdBu_r',
                aspect="auto")
fig.update_layout(width=600, height=500)
fig.show()


def prepare_modeling_data(data):
    """æº–å‚™å»ºæ¨¡ç”¨çš„æ•¸æ“š"""
    # åˆ†é›¢è¨“ç·´å’Œæ¸¬è©¦æ•¸æ“š
    train_data = data[pd.notnull(data['count'])].copy()
    test_data = data[~pd.notnull(data['count'])].copy()

    # ä¿�å­˜æ¸¬è©¦æ•¸æ“šçš„datetime
    test_datetime = test_data['datetime'].copy()

    # æº–å‚™æ¨™ç±¤ï¼ˆå°�æ•¸è½‰æ�›ï¼‰
    y_labels = train_data['count']
    y_labels_log = np.log(y_labels)

    # é�¸æ“‡ç‰¹å¾µ
    feature_columns = ['season', 'holiday', 'workingday', 'weather', 'temp',
                      'atemp', 'humidity', 'windspeed', 'hour', 'year',
                      'month', 'weekday', 'is_weekend', 'is_rush_hour',
                      'is_working_day']

    X_train = train_data[feature_columns]
    X_test = test_data[feature_columns]

    return X_train, X_test, y_labels, y_labels_log, test_datetime

# æº–å‚™æ•¸æ“š
X_train, X_test, y_labels, y_labels_log, test_datetime = prepare_modeling_data(all_data_featured)

print(f"è¨“ç·´ç‰¹å¾µç¶­åº¦: {X_train.shape}")
print(f"æ¸¬è©¦ç‰¹å¾µç¶­åº¦: {X_test.shape}")
print(f"ç‰¹å¾µåˆ—è¡¨: {list(X_train.columns)}")


# å»ºç«‹éš¨æ©Ÿæ£®æ�—æ¨¡å�‹
rf_model = RandomForestRegressor(
    n_estimators=1000,
    random_state=42,
    n_jobs=-1,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2
)

# è¨“ç·´æ¨¡å�‹
print("é–‹å§‹è¨“ç·´æ¨¡å�‹...")
rf_model.fit(X_train, y_labels_log)
print("æ¨¡å�‹è¨“ç·´å®Œæˆ�ï¼�")

# ç‰¹å¾µé‡�è¦�æ€§
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("ç‰¹å¾µé‡�è¦�æ€§æ�’åº�:")
print(feature_importance.head(10))


# ç‰¹å¾µé‡�è¦�æ€§åœ“é¤…åœ–ï¼ˆå‰�10å€‹ç‰¹å¾µï¼‰
top_10_features = feature_importance.head(10)
fig = px.pie(top_10_features, values='importance', names='feature',
             title='å‰�10é‡�è¦�ç‰¹å¾µä½”æ¯”')
fig.show()


# ç‰¹å¾µé‡�è¦�æ€§é•·æ¢�åœ–
fig = px.bar(feature_importance.head(15),
             x='importance', y='feature',
             orientation='h',
             title='å‰�15å€‹ç‰¹å¾µé‡�è¦�æ€§æ�’åº�',
             labels={'importance': 'é‡�è¦�æ€§', 'feature': 'ç‰¹å¾µ'},
             color='feature'
             )  # ä½¿ç”¨ 'feature' æ¬„ä½�ä½œç‚ºé¡�è‰²ï¼Œå�³ä½¿é¡�è‰²æ˜¯é �è¨­çš„ï¼Œä¹Ÿæœƒç”Ÿæˆ� Legend
fig.update_layout(yaxis={'categoryorder':'total ascending'},
         legend=dict(x=0.5, y=1.0, xanchor='center', yanchor='bottom', orientation='h')         )
fig.show()


def evaluate_model(model, X, y_true, y_log_true):
    """è©•ä¼°æ¨¡å�‹æ€§èƒ½"""
    # äº¤å�‰é©—è­‰
    cv_scores = cross_val_score(model, X, y_log_true, cv=5,
                               scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores)

    # è¨“ç·´é›†é �æ¸¬
    y_pred_log = model.predict(X)
    y_pred = np.exp(y_pred_log)

    # è¨ˆç®—è©•ä¼°æŒ‡æ¨™
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    return cv_rmse, rmse, mae, y_pred

# è©•ä¼°æ¨¡å�‹
cv_rmse, train_rmse, train_mae, train_pred = evaluate_model(rf_model, X_train, y_labels, y_labels_log)


# è©•ä¼°çµ�æ�œè¦–è¦ºåŒ–
metrics_data = pd.DataFrame({
    'è©•ä¼°æŒ‡æ¨™': ['äº¤å�‰é©—è­‰RMSE', 'è¨“ç·´é›†RMSE', 'è¨“ç·´é›†MAE'],
    'æ•¸å€¼': [cv_rmse.mean(), train_rmse, train_mae]
})

fig = px.bar(metrics_data, x='è©•ä¼°æŒ‡æ¨™', y='æ•¸å€¼',
             title='æ¨¡å�‹è©•ä¼°æŒ‡æ¨™',
             color='è©•ä¼°æŒ‡æ¨™', # ä¿®æ­£é€™è£¡ï¼Œä½¿ç”¨ 'è©•ä¼°æŒ‡æ¨™' æ¬„ä½�
             labels={'è©•ä¼°æŒ‡æ¨™': 'è©•ä¼°æŒ‡æ¨™', 'æ•¸å€¼': 'æ•¸å€¼'})

fig.update_layout(legend=dict(x=0.5, y=1.0, xanchor='center', yanchor='bottom', orientation='h'))

fig.show()

print(f"äº¤å�‰é©—è­‰ RMSE: {cv_rmse.mean():.4f} (+/- {cv_rmse.std() * 2:.4f})")
print(f"è¨“ç·´é›† RMSE: {train_rmse:.4f}")
print(f"è¨“ç·´é›† MAE: {train_mae:.4f}")


# å¯¦éš›å€¼ vs é �æ¸¬å€¼æ•£é»�åœ–
pred_comparison = pd.DataFrame({
    'actual': y_labels,
    'predicted': train_pred
})

fig = px.scatter(pred_comparison, x='actual', y='predicted',
                 title='å¯¦éš›å€¼ vs é �æ¸¬å€¼æ¯”è¼ƒ',
                 labels={'actual': 'å¯¦éš›éœ€æ±‚é‡�', 'predicted': 'é �æ¸¬éœ€æ±‚é‡�'},
                 trendline='ols')
# æ·»åŠ å®Œç¾�é �æ¸¬ç·š
min_val = min(pred_comparison['actual'].min(), pred_comparison['predicted'].min())
max_val = max(pred_comparison['actual'].max(), pred_comparison['predicted'].max())
fig.add_shape(type='line', x0=min_val, y0=min_val, x1=max_val, y1=max_val,
              line=dict(color='red', dash='dash', width=2))
fig.show()


# é �æ¸¬èª¤å·®åˆ†å¸ƒ
residuals = y_labels - train_pred
fig = px.histogram(x=residuals, nbins=50,
                   title='é �æ¸¬èª¤å·®åˆ†å¸ƒ',
                   labels={'x': 'é �æ¸¬èª¤å·®', 'y': 'é »ç�‡'})
fig.update_layout(showlegend=False)
fig.show()


# é �æ¸¬æ¸¬è©¦é›†
print("é–‹å§‹é �æ¸¬æ¸¬è©¦é›†...")
test_pred_log = rf_model.predict(X_test)
test_pred = np.exp(test_pred_log)
test_pred = np.maximum(test_pred, 0)  # ç¢ºä¿�é �æ¸¬å€¼é��è² 

# å‰µå»ºæ��äº¤æª”æ¡ˆ
submission = pd.DataFrame({
    'datetime': test_datetime,
    'count': test_pred
})

# ä¿�å­˜é �æ¸¬çµ�æ�œ
submission.to_csv('bike_predictions_optimized.csv', index=False)
print("é �æ¸¬çµ�æ�œå·²ä¿�å­˜è‡³: bike_predictions_optimized.csv")

# é �æ¸¬çµ�æ�œåˆ†å¸ƒ
fig = px.histogram(submission, x='count', nbins=50,
                   title='æ¸¬è©¦é›†é �æ¸¬çµ�æ�œåˆ†å¸ƒ',
                   labels={'count': 'é �æ¸¬éœ€æ±‚é‡�', 'y': 'é »ç�‡'})
fig.update_layout(showlegend=False)
fig.show()


# é �æ¸¬çµ�æ�œæ™‚é–“åº�åˆ—åœ–ï¼ˆå‰�100å€‹é»�ä½œç‚ºç¤ºä¾‹ï¼‰
sample_predictions = submission.head(100).copy()
sample_predictions['datetime'] = pd.to_datetime(sample_predictions['datetime'])

fig = px.line(sample_predictions, x='datetime', y='count',
              title='æ¸¬è©¦é›†é �æ¸¬çµ�æ�œæ™‚é–“åº�åˆ— (å‰�100å€‹é �æ¸¬é»�)',
              labels={'datetime': 'æ™‚é–“', 'count': 'é �æ¸¬éœ€æ±‚é‡�'})
fig.show()


print("="*50)
print("ğŸ“Š å…±äº«å–®è»Šéœ€æ±‚é �æ¸¬åˆ†æ��ç¸½çµ�")
print("="*50)
print(f"âœ… è¨“ç·´æ•¸æ“š: {X_train.shape[0]} ç­†")
print(f"âœ… æ¸¬è©¦æ•¸æ“š: {X_test.shape[0]} ç­†")
print(f"âœ… ç‰¹å¾µæ•¸é‡�: {X_train.shape[1]} å€‹")
print(f"âœ… æ¨¡å�‹é¡�å�‹: éš¨æ©Ÿæ£®æ�—")
print(f"âœ… äº¤å�‰é©—è­‰RMSE: {cv_rmse.mean():.4f}")
print(f"âœ… è¨“ç·´é›†RMSE: {train_rmse:.4f}")
print(f"âœ… è¨“ç·´é›†MAE: {train_mae:.4f}")
print(f"âœ… æœ€é‡�è¦�ç‰¹å¾µ: {feature_importance.iloc[0]['feature']}")
print(f"âœ… é �æ¸¬æª”æ¡ˆ: bike_predictions_optimized.csv")
print("="*50)

# æœ€çµ‚æ¨¡å�‹è¡¨ç�¾ç¸½çµ�åœ–
summary_data = pd.DataFrame({
    'éš�æ®µ': ['æ•¸æ“šè¼‰å…¥', 'ç‰¹å¾µå·¥ç¨‹', 'æ¨¡å�‹è¨“ç·´', 'æ¨¡å�‹è©•ä¼°', 'é �æ¸¬å®Œæˆ�'],
    'å®Œæˆ�ç‹€æ…‹': ['âœ…', 'âœ…', 'âœ…', 'âœ…', 'âœ…'],
    'é€²åº¦': [100, 100, 100, 100, 100]
})

fig = px.bar(summary_data, x='éš�æ®µ', y='é€²åº¦',
             title='ğŸ�‰ å…±äº«å–®è»Šéœ€æ±‚é �æ¸¬å°ˆæ¡ˆå®Œæˆ�é€²åº¦',
             labels={'éš�æ®µ': 'å°ˆæ¡ˆéš�æ®µ', 'é€²åº¦': "å®Œæˆ�é€²åº¦ (%)"}, # ä½¿ç”¨é›™å¼•è™Ÿ
             color='é€²åº¦',
             color_continuous_scale='Greens')
fig.update_layout(showlegend=False)
fig.show()

print("ğŸ�‰ æ�­å–œï¼�å…±äº«å–®è»Šéœ€æ±‚é �æ¸¬åˆ†æ��å·²å…¨éƒ¨å®Œæˆ�ï¼�")

