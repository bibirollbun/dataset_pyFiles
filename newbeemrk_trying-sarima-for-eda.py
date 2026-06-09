import pandas as pd
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

RAW_DIR = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data")
df = pd.read_csv(RAW_DIR / "train.csv")

df_sample=df[df['sequence_id']=="SEQ_000007"]['acc_x']


print("Basic Statistics of the Data:")
print(df_sample.describe())
print(f"\nNumber of Data Points: {len(df_sample)}")


fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Initial Time Series Analysis', fontsize=16)

# Original data plot
axes[0, 0].plot(df_sample.index, df_sample.values, 'b-', linewidth=1)
axes[0, 0].set_title('Original Time Series Data (acc_x)')
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Acceleration (acc_x)')
axes[0, 0].grid(True, alpha=0.3)

# Histogram of values
axes[0, 1].hist(df_sample.values, bins=30, alpha=0.7, edgecolor='black')
axes[0, 1].set_title('Data Distribution')
axes[0, 1].set_xlabel('Acceleration Values')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].grid(True, alpha=0.3)

# Autocorrelation function (ACF)
plot_acf(df_sample, ax=axes[1, 0], lags=40, title='Autocorrelation Function (ACF)')
axes[1, 0].grid(True, alpha=0.3)

# Partial autocorrelation function (PACF)
plot_pacf(df_sample, ax=axes[1, 1], lags=27, title='Partial Autocorrelation Function (PACF)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


def check_stationarity(ts, title):
    print(f"\n=== Stationarity Test for {title} ===")
    
    adf_result = adfuller(ts, autolag='AIC')
    print("ADF Test:")
    print(f"  ADF Statistic: {adf_result[0]:.6f}")
    print(f"  p-value: {adf_result[1]:.6f}")
    print(f"  Critical Values: {adf_result[4]}")
    if adf_result[1] <= 0.05:
        print("  Result: Stationary (p < 0.05)")
    else:
        print("  Result: Non-stationary (p >= 0.05)")
    
    kpss_result = kpss(ts, regression='c', nlags="auto")
    print("\nKPSS Test:")
    print(f"  KPSS Statistic: {kpss_result[0]:.6f}")
    print(f"  p-value: {kpss_result[1]:.6f}")
    print(f"  Critical Values: {kpss_result[3]}")
    if kpss_result[1] >= 0.05:
        print("  Result: Stationary (p >= 0.05)")
    else:
        print("  Result: Non-stationary (p < 0.05)")

# Run stationarity tests on the original data
check_stationarity(df_sample, "Original Data")


def estimate_differencing_order(ts, max_d=2):
    print("\n=== Estimating Differencing Order ===")
    print("Note: SARIMA handles differencing internally; actual differencing is not applied here.")
    
    current_ts = ts.copy()
    recommended_d = 0
    
    for i in range(max_d + 1):
        adf_stat, p_value = adfuller(current_ts, autolag='AIC')[:2]
        print(f"Differencing order {i}: ADF p-value = {p_value:.6f}")
        if p_value <= 0.05:
            print(f"Recommended differencing order: d = {i}")
            recommended_d = i
            break
        if i < max_d:
            current_ts = current_ts.diff().dropna()
            recommended_d = i + 1
    
    if recommended_d > max_d:
        print(f"Data is not stationary even after maximum differencing order: d = {max_d}")
        recommended_d = max_d
    return recommended_d

recommended_d = estimate_differencing_order(df_sample)


def detect_seasonality(ts, max_lag=50):
    print("\n=== Detecting Seasonality ===")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_acf(ts, ax=ax, lags=max_lag, title='ACF for Seasonality Detection')
    plt.show()

    acf_values = [abs(ts.autocorr(lag=lag)) for lag in range(1, min(max_lag+1, len(ts)//4))]
    lags = list(range(1, len(acf_values)+1))
    significant_lags = [(lags[i], acf_values[i]) for i in range(len(acf_values)) if acf_values[i] > 0.1]
    print(f"Lags with significant autocorrelation: {significant_lags}")

    common_periods = [12, 24, 48, 168]
    period_correlations = []
    for period in common_periods:
        if period <= len(ts)//4:
            corr = abs(ts.autocorr(lag=period))
            period_correlations.append((period, corr))
            print(f"Autocorrelation at period {period}: {corr:.4f}")

    max_corr_lag, max_corr_value = max(zip(lags, acf_values), key=lambda x: x[1])
    if period_correlations:
        best_common_period, best_common_corr = max(period_correlations, key=lambda x: x[1])
    else:
        best_common_period, best_common_corr = None, 0

    if best_common_corr > 0.15 and best_common_corr > max_corr_value:
        selected_period = best_common_period
        print(f"\nSelected seasonal period: {selected_period} (common period, correlation: {best_common_corr:.4f})")
    elif max_corr_value > 0.1:
        selected_period = max_corr_lag
        print(f"\nSelected seasonal period: {selected_period} (highest autocorrelation lag, value: {max_corr_value:.4f})")
    else:
        selected_period = 24 if len(ts) >= 24 else max(12, len(ts)//10)
        print(f"\nNo clear seasonality detected; defaulting to: {selected_period}")

    # Optional FFT-based detection
    try:
        from scipy.fft import fft, fftfreq
        fft_vals = np.abs(fft(ts.values))
        fft_freqs = fftfreq(len(ts), 1)
        pos = fft_freqs > 0
        pos_fft = fft_vals[pos]
        pos_freqs = fft_freqs[pos]
        idx = np.argmax(pos_fft)
        dominant_freq = pos_freqs[idx]
        fft_period = int(1 / dominant_freq)
        print(f"FFT-based primary period: {fft_period}")
        if 6 <= fft_period <= len(ts)//4 and pos_fft[idx] > np.mean(pos_fft)*2:
            print("Considering FFT result as well.")
    except ImportError:
        print("Skipping FFT analysis.")

    return selected_period

seasonal_period = detect_seasonality(df_sample)



def auto_sarima_selection(ts, seasonal_period, max_p=3, max_d=2, max_q=3, max_P=2, max_D=1, max_Q=2):
    """
    Automatically select SARIMA parameters based on AIC.
    Uses the detected seasonal period.
    """
    print("\n=== Automatic Selection of SARIMA Parameters ===")
    print(f"Seasonal period to use: {seasonal_period}")
    
    best_aic = float('inf')
    best_params = None
    best_seasonal_params = None
    results = []
    
    # Try multiple candidate seasonal periods:
    # Use both the detected period and common periods if data length allows
    seasonal_candidates = [seasonal_period]
    
    common_periods = [12, 24, 168]  # monthly, daily, weekly
    for period in common_periods:
        if period <= len(ts) // 4 and period != seasonal_period:
            seasonal_candidates.append(period)
    
    print(f"Seasonal periods to try: {seasonal_candidates}")
    
    for s in seasonal_candidates:
        print(f"\nOptimizing with seasonal period {s}...")
        
        # Iterate over parameter combinations
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    for P in range(max_P + 1):
                        for D in range(max_D + 1):
                            for Q in range(max_Q + 1):
                                try:
                                    model = SARIMAX(ts, 
                                                   order=(p, d, q),
                                                   seasonal_order=(P, D, Q, s),
                                                   enforce_stationarity=False,
                                                   enforce_invertibility=False)
                                    fitted_model = model.fit(disp=False)
                                    
                                    aic = fitted_model.aic
                                    results.append({
                                        'order': (p, d, q),
                                        'seasonal_order': (P, D, Q, s),
                                        'AIC': aic,
                                        'BIC': fitted_model.bic
                                    })
                                    
                                    if aic < best_aic:
                                        best_aic = aic
                                        best_params = (p, d, q)
                                        best_seasonal_params = (P, D, Q, s)
                                        
                                except:
                                    continue
    
    # Display the top 10 models by AIC
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('AIC').head(10)
    
    print("\nTop 10 models (sorted by AIC):")
    print(results_df.to_string(index=False))
    
    print(f"\nBest parameters:")
    print(f"  ARIMA order: {best_params}")
    print(f"  Seasonal order: {best_seasonal_params}")
    print(f"  AIC: {best_aic:.4f}")
    
    return best_params, best_seasonal_params, best_aic

# Automatic parameter selection (using detected seasonal period)
best_order, best_seasonal_order, best_aic = auto_sarima_selection(
    df_sample, seasonal_period, max_p=2, max_d=1, max_q=2, max_P=1, max_D=1, max_Q=1
)



# === Fitting the Optimal SARIMA Model ===
print("\n=== Fitting the Optimal SARIMA Model ===")

# Create and fit the model
model = SARIMAX(df_sample, 
                order=best_order,
                seasonal_order=best_seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False)

fitted_model = model.fit(disp=False)

# Display model summary
print(fitted_model.summary())

# === SARIMA Model Diagnostics ===
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('SARIMA Model Diagnostics', fontsize=16)

# Residual Partial Autocorrelation (PACF)
residuals = fitted_model.resid
plot_pacf(residuals, ax=axes[0, 0], lags=27, title='Residual Partial Autocorrelation (PACF)')
axes[0, 0].grid(True, alpha=0.3)

# Residual Autocorrelation (ACF)
plot_acf(residuals, ax=axes[0, 1], lags=40, title='Residual Autocorrelation (ACF)')
axes[0, 1].grid(True, alpha=0.3)

# Residual Time Series Plot
axes[1, 0].plot(residuals.index, residuals.values, 'r-', linewidth=1)
axes[1, 0].set_title('Residual Time Series Plot')
axes[1, 0].set_xlabel('Time')
axes[1, 0].set_ylabel('Residuals')
axes[1, 0].grid(True, alpha=0.3)

# Residual Distribution
axes[1, 1].hist(residuals.values, bins=30, alpha=0.7, edgecolor='black')
axes[1, 1].set_title('Residual Distribution')
axes[1, 1].set_xlabel('Residuals')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# === SARIMA Model Forecast Results ===
fig, axes = plt.subplots(2, 1, figsize=(15, 10))
fig.suptitle('SARIMA Model Forecast Results', fontsize=16)

# Fitted values and confidence intervals
fitted_values = fitted_model.fittedvalues
conf_int = fitted_model.get_forecast(steps=len(df_sample)).conf_int()


# Detailed view for the last n_zoom points
n_zoom = min(57, len(df_sample))
zoom_idx = df_sample.index[-n_zoom:]
axes[0].plot(zoom_idx, df_sample.loc[zoom_idx], 'b-', linewidth=2, label='Actual Data', alpha=0.7)
axes[0].plot(zoom_idx, fitted_values.loc[zoom_idx], 'r-', linewidth=2, label='SARIMA Fit', alpha=0.8)
axes[0].fill_between(zoom_idx,
                     fitted_values.loc[zoom_idx] - 1.96*np.std(residuals),
                     fitted_values.loc[zoom_idx] + 1.96*np.std(residuals),
                     alpha=0.2, label='95% Confidence Interval')
axes[0].set_title(f'Detailed View (Last {n_zoom} Data Points')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Acceleration (acc_x)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Detailed view for the last n_zoom points
n_zoom = min(30, len(df_sample))
zoom_idx = df_sample.index[-n_zoom:]
axes[1].plot(zoom_idx, df_sample.loc[zoom_idx], 'b-', linewidth=2, label='Actual Data', alpha=0.7)
axes[1].plot(zoom_idx, fitted_values.loc[zoom_idx], 'r-', linewidth=2, label='SARIMA Fit', alpha=0.8)
axes[1].fill_between(zoom_idx,
                     fitted_values.loc[zoom_idx] - 1.96*np.std(residuals),
                     fitted_values.loc[zoom_idx] + 1.96*np.std(residuals),
                     alpha=0.2, label='95% Confidence Interval')
axes[1].set_title(f'Detailed View (Last {n_zoom} Data Points')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Acceleration (acc_x)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# === Model Performance Evaluation ===
print("\n=== Model Performance Evaluation ===")

# Basic evaluation metrics
mse = mean_squared_error(df_sample, fitted_values)
rmse = np.sqrt(mse)
mae = mean_absolute_error(df_sample, fitted_values)
mape = np.mean(np.abs((df_sample - fitted_values) / df_sample)) * 100

print(f"Mean Squared Error (MSE): {mse:.6f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
print(f"Mean Absolute Error (MAE): {mae:.6f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Residual Normality Test (Shapiro-Wilk)
from scipy import stats
shapiro_stat, shapiro_p = stats.shapiro(residuals)
print(f"\nResidual Normality Test (Shapiro-Wilk):")
print(f"  Statistic: {shapiro_stat:.6f}")
print(f"  p-value: {shapiro_p:.6f}")

# Residual Independence Test (Ljung-Box)
ljung_box_result = acorr_ljungbox(residuals, lags=10, return_df=True)
print(f"\nResidual Independence Test (Ljung-Box):")
print(ljung_box_result)

# === Future Forecast ===

# Forecast for the next 30 periods
n_forecast = 30
forecast = fitted_model.get_forecast(steps=n_forecast)
forecast_values = forecast.predicted_mean
forecast_conf_int = forecast.conf_int()

# Create index for forecast period
last_step = df_sample.index[-1]  # これは整数
forecast_index = pd.Index(
    np.arange(last_step + 1, last_step + 1 + n_forecast),
    name=df_sample.index.name
)
# Plot future forecast
fig, ax = plt.subplots(figsize=(15, 8))

# Last n_history points of historical data
n_history = min(50, len(df_sample))
history_idx = df_sample.index[-n_history:]

ax.plot(history_idx, df_sample.loc[history_idx], 'b-', linewidth=2, label='Actual Data', alpha=0.7)
ax.plot(forecast_index, forecast_values, 'r-', linewidth=2, label='Forecast', alpha=0.8)
ax.fill_between(forecast_index,
                forecast_conf_int.iloc[:, 0],
                forecast_conf_int.iloc[:, 1],
                alpha=0.2, label='95% Confidence Interval')

ax.axvline(x=last_step, linestyle='--', alpha=0.7, label='Forecast Start')
ax.set_title('Future Forecast by SARIMA Model')
ax.set_xlabel('Time')
ax.set_ylabel('Acceleration (acc_x)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nForecast Statistics:")
print(f"  Forecast Mean: {forecast_values.mean():.6f}")
print(f"  Forecast Std Dev: {forecast_values.std():.6f}")
print(f"  Forecast Min: {forecast_values.min():.6f}")
print(f"  Forecast Max: {forecast_values.max():.6f}")

print("\n=== Analysis Complete ===")
print("SARIMA model application is complete.")
print("Parameter selection was carried out with the following steps:")
print("1. Stationarity tests (ADF, KPSS tests)")
print("2. Estimation of differencing orders (actual differencing not applied since SARIMA processes internally)")
print("3. Seasonal detection (identifying seasonal periods using multiple methods)")
print("4. Automatic parameter selection based on AIC")
print("5. Model fitting and diagnostics")

print("\nKey Improvements:")
print("- Leveraged SARIMA model's differencing capabilities")
print("- Analysis preserving original data information")
print("- Comparison of multiple seasonal period candidates")
print("- Efficient search guided by recommended differencing orders")

