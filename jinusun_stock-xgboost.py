pip install ta==0.10.2


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import os
import ta
import gc
import warnings
warnings.filterwarnings('ignore')


# 시드 설정
np.random.seed(42)

# 환경 확인 (Kaggle 또는 로컬)
IN_KAGGLE = os.path.exists('/kaggle/input')

# 데이터 로드
if IN_KAGGLE:
    train_data = pd.read_csv('/kaggle/input/stack-1/stock_data_train.csv')
    test_data = pd.read_csv('/kaggle/input/stack-1/stock_data_test.csv')
    submission = pd.read_csv('/kaggle/input/stack-1/sample_submission.csv')
else:
    train_data = pd.read_csv('data/stock_data_train.csv')
    test_data = pd.read_csv('data/stock_data_test.csv')
    submission = pd.read_csv('data/sample_submission.csv')

# 날짜 형식 변환
train_data['Date'] = pd.to_datetime(train_data['Date'])
test_data['Date'] = pd.to_datetime(test_data['Date'])
submission['Id'] = pd.to_datetime(submission['Id'])

# 날짜 정렬
train_data = train_data.sort_values('Date')
test_data = test_data.sort_values('Date')

# 기본 기술적 지표 추가 함수
def add_technical_indicators(df, close_col='Close'):
    df_copy = df.copy()
    
    # 종가가 없는 경우 (테스트 데이터)
    if close_col not in df_copy.columns:
        # Open, Low, High의 평균을 임시 종가로 사용
        df_copy['temp_close'] = (df_copy['Open'] + df_copy['Low'] + df_copy['High']) / 3
        close_col = 'temp_close'
    
    # 이동평균
    for window in [5, 10, 20, 30, 50]:
        df_copy[f'SMA_{window}'] = ta.trend.SMAIndicator(close=df_copy[close_col], window=window).sma_indicator()
        df_copy[f'EMA_{window}'] = ta.trend.EMAIndicator(close=df_copy[close_col], window=window).ema_indicator()
    
    # RSI (상대강도지수)
    for window in [7, 14, 21]:
        df_copy[f'RSI_{window}'] = ta.momentum.RSIIndicator(close=df_copy[close_col], window=window).rsi()
    
    # 추세 지표
    macd = ta.trend.MACD(close=df_copy[close_col])
    df_copy['MACD'] = macd.macd()
    df_copy['MACD_signal'] = macd.macd_signal()
    df_copy['MACD_diff'] = macd.macd_diff()
    
    # 볼린저밴드
    for window in [10, 20, 30]:
        bollinger = ta.volatility.BollingerBands(close=df_copy[close_col], window=window, window_dev=2)
        df_copy[f'bollinger_upper_{window}'] = bollinger.bollinger_hband()
        df_copy[f'bollinger_lower_{window}'] = bollinger.bollinger_lband()
        df_copy[f'bollinger_pct_{window}'] = bollinger.bollinger_pband()
    
    # 스토캐스틱 오실레이터
    for window in [5, 14]:
        stoch = ta.momentum.StochasticOscillator(high=df_copy['High'], low=df_copy['Low'], close=df_copy[close_col], window=window)
        df_copy[f'stoch_{window}'] = stoch.stoch()
        df_copy[f'stoch_signal_{window}'] = stoch.stoch_signal()
    
    # ATR (Average True Range) - 변동성 지표
    for window in [7, 14, 21]:
        atr = ta.volatility.AverageTrueRange(high=df_copy['High'], low=df_copy['Low'], close=df_copy[close_col], window=window)
        df_copy[f'ATR_{window}'] = atr.average_true_range()
    
    # OBV (온발런스볼륨)
    if 'Volume' in df_copy.columns:
        df_copy['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=df_copy[close_col], volume=df_copy['Volume']).on_balance_volume()
        
        # 추가 거래량 지표
        for window in [5, 10, 20]:
            df_copy[f'Volume_SMA_{window}'] = ta.trend.SMAIndicator(close=df_copy['Volume'], window=window).sma_indicator()
            df_copy[f'Volume_EMA_{window}'] = ta.trend.EMAIndicator(close=df_copy['Volume'], window=window).ema_indicator()
    
    # 가격 변동성 지표
    if 'High' in df_copy.columns and 'Low' in df_copy.columns:
        df_copy['High_Low_Ratio'] = df_copy['High'] / df_copy['Low']
        df_copy['Daily_Range'] = df_copy['High'] - df_copy['Low']
        df_copy['Daily_Range_Pct'] = (df_copy['High'] - df_copy['Low']) / df_copy[close_col]
    
    # 시가와 고가/저가의 비율
    if 'Open' in df_copy.columns and 'High' in df_copy.columns:
        df_copy['Open_High_Ratio'] = df_copy['Open'] / df_copy['High']
        df_copy['Open_Low_Ratio'] = df_copy['Open'] / df_copy['Low']
    
    # 날짜 기반 특성
    df_copy['day_of_week'] = df_copy['Date'].dt.dayofweek
    df_copy['month'] = df_copy['Date'].dt.month
    df_copy['quarter'] = df_copy['Date'].dt.quarter
    
    # 이전 n일 변동률 추가
    if close_col in df_copy.columns:
        for n in [1, 3, 5, 10, 20]:
            df_copy[f'return_{n}d'] = df_copy[close_col].pct_change(n)
    
    # 임시 종가 컬럼 제거
    if 'temp_close' in df_copy.columns:
        df_copy.drop(columns=['temp_close'], inplace=True)
    
    # NaN 값 처리
    df_copy = df_copy.replace([np.inf, -np.inf], np.nan)
    df_copy.fillna(method='bfill', inplace=True)
    df_copy.fillna(method='ffill', inplace=True)
    df_copy.fillna(0, inplace=True)
    
    return df_copy

# 기술적 지표 추가
print("기술적 지표 추가 중...")
train_data_with_indicators = add_technical_indicators(train_data)
test_data_with_indicators = add_technical_indicators(test_data)

# 시차 변수 생성
def add_lag_features(df, columns, lags):
    df_copy = df.copy()
    for col in columns:
        for lag in lags:
            df_copy[f'{col}_lag{lag}'] = df_copy[col].shift(lag)
    return df_copy

# 특성 선택
print("XGBoost 모델 준비 중...")
# 날짜, Close를 제외한 모든 특성 사용
all_features = [col for col in train_data_with_indicators.columns 
                if col not in ['Date', 'Close']]

# 시차 변수 추가
important_features = ['Open', 'High', 'Low', 'Volume', 'SMA_5', 'SMA_20', 'EMA_5', 'EMA_20', 
                      'RSI_14', 'MACD', 'bollinger_pct_20', 'High_Low_Ratio']
lag_periods = [1, 2, 3, 5, 7, 10, 14, 21]

train_with_lags = add_lag_features(train_data_with_indicators, important_features, lag_periods)
test_with_lags = add_lag_features(test_data_with_indicators, important_features, lag_periods)

# NaN 값 처리
train_with_lags = train_with_lags.replace([np.inf, -np.inf], np.nan)
test_with_lags = test_with_lags.replace([np.inf, -np.inf], np.nan)
train_with_lags.fillna(method='bfill', inplace=True)
train_with_lags.fillna(method='ffill', inplace=True)
train_with_lags.fillna(0, inplace=True)
test_with_lags.fillna(method='bfill', inplace=True)
test_with_lags.fillna(method='ffill', inplace=True)
test_with_lags.fillna(0, inplace=True)

# 모든 특성 + 시차 변수 결합
feature_columns = all_features + [f'{col}_lag{lag}' for col in important_features for lag in lag_periods]
feature_columns = [col for col in feature_columns if col in train_with_lags.columns]

# 시차 변수 추가로 처음 몇 행은 NaN이므로 제거
max_lag = max(lag_periods)
X_train = train_with_lags.iloc[max_lag:][feature_columns]
y_train = train_with_lags.iloc[max_lag:]['Close']
X_test = test_with_lags[feature_columns]

print(f"사용된 특성 수: {len(feature_columns)}")

# 최적화된 XGBoost 모델 생성 및 학습
print("XGBoost 모델 학습 중...")
model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=9,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1.0,
    gamma=0.1,
    random_state=42,
    early_stopping_rounds=50,
    n_jobs=-1
)

# 데이터 분할 없이 모델 학습
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train)],
    verbose=100
)

# 예측
print("예측 중...")
predictions = model.predict(X_test)

# 특성 중요도 시각화
plt.figure(figsize=(10, 8))
top_n = 20
xgb.plot_importance(model, max_num_features=top_n, importance_type='weight', title=f"Top {top_n} Important Features")
plt.tight_layout()
importance_path = '/kaggle/working/feature_importance.png' if IN_KAGGLE else 'feature_importance.png'
plt.savefig(importance_path)
plt.close()

# 예측 결과 스무딩
def smooth_predictions(predictions, alpha=0.9):
    smoothed = [predictions[0]]
    for i in range(1, len(predictions)):
        smoothed.append(alpha * predictions[i] + (1 - alpha) * smoothed[i-1])
    return np.array(smoothed)

smoothed_predictions = smooth_predictions(predictions, alpha=0.9)

# 제출 파일 생성
submission['Close'] = smoothed_predictions
submit_path = '/kaggle/working/submission_xgboost.csv' if IN_KAGGLE else 'submission_xgboost.csv'
submission.to_csv(submit_path, index=False)

# 메모리 정리
del model
gc.collect()

# 예측 결과 시각화
plt.figure(figsize=(15, 7))
if not IN_KAGGLE:
    plt.plot(train_data['Date'][-100:], train_data['Close'][-100:], label='Training Data (Last 100 days)')
plt.plot(test_data['Date'], smoothed_predictions, label='XGBoost Predictions')
plt.title('Stock Price Prediction with XGBoost')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True, alpha=0.3)
fig_path = '/kaggle/working/xgboost_prediction.png' if IN_KAGGLE else 'xgboost_prediction.png'
plt.savefig(fig_path)
plt.close()

print("XGBoost 모델의 예측이 완료되었습니다!")
print(f"결과가 {submit_path}에 저장되었습니다.")
print(f"특성 중요도가 {importance_path}에 저장되었습니다.") 

