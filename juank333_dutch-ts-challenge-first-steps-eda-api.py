import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from statsmodels.tsa.seasonal import MSTL, STL, seasonal_decompose
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import warnings
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import ccf
from statsmodels.tsa.seasonal import MSTL, STL, seasonal_decompose
from matplotlib.ticker import MaxNLocator
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import pacf
from scipy import stats
warnings.filterwarnings('ignore')


df = pd.read_csv("/kaggle/input/dutch-energy-supplier-load-forecasting-challenge/train_expanded.csv")
df.head()


res_daily = STL(df["net_load_kwh"], period=365).fit()

# Plot
plt.rc("figure", figsize=(20, 12))
plt.rc("font", size=10)
fig = res_daily.plot()
plt.suptitle('STL Decomposition', 
             fontsize=16, y=1.02)
plt.tight_layout()
plt.show()




# ==========================================
# CONFIGURACIÃ“N INICIAL
# ==========================================
print("ğŸ“Š ANÃ�LISIS EXHAUSTIVO DE LA TENDENCIA - net_load_kwh")
print("=" * 60)

# Asegurar que el Ã­ndice es datetime
df_analysis = df.copy()
if not isinstance(df_analysis.index, pd.DatetimeIndex):
    df_analysis.index = pd.to_datetime(df_analysis.index)

df_analysis = df_analysis.sort_index()



# ==========================================
# 2. ANÃ�LISIS DE TENDENCIA TEMPORAL
# ==========================================
print("\nğŸ�¯ 2. ANÃ�LISIS DE TENDENCIA TEMPORAL")
print("=" * 40)

# Crear figura para anÃ¡lisis temporal
fig = plt.figure(figsize=(20, 16))
gs = plt.GridSpec(4, 3, figure=fig)

# SUBPLOT 1: Serie temporal completa
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(df_analysis.index, df_analysis['net_load_kwh'], linewidth=0.8, alpha=0.7, color='blue')
ax1.set_title('SERIE TEMPORAL COMPLETA - net_load_kwh', fontsize=14, fontweight='bold')
ax1.set_ylabel('Net Load kWh')
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# SUBPLOT 2: Tendencia por diferentes ventanas
ax2 = fig.add_subplot(gs[1, 0])

# Calcular medias mÃ³viles para diferentes ventanas
windows = [24, 96, 672]  # 1 dÃ­a, 4 dÃ­as, 1 semana
colors = ['red', 'green', 'purple']
labels = ['1 dÃ­a', '4 dÃ­as', '1 semana']

for i, window in enumerate(windows):
    rolling_mean = df_analysis['net_load_kwh'].rolling(window=window, center=True).mean()
    ax2.plot(df_analysis.index, rolling_mean, linewidth=2, color=colors[i], label=labels[i], alpha=0.8)

ax2.set_title('TENDENCIA - Medias MÃ³viles', fontsize=12, fontweight='bold')
ax2.set_ylabel('Net Load kWh')
ax2.legend()
ax2.grid(True, alpha=0.3)

# SUBPLOT 3: DescomposiciÃ³n estacional (usando 1 semana como perÃ­odo)
ax3 = fig.add_subplot(gs[1, 1])

try:
    # Tomar muestra para descomposiciÃ³n (Ãºltimos 2 meses para mejor visualizaciÃ³n)
    sample_data = df_analysis['net_load_kwh'].last('60D')
    
    if len(sample_data) > 672:  # Al menos 1 semana completa
        decomposition = seasonal_decompose(sample_data, model='additive', period=672)
        
        ax3.plot(decomposition.trend.index, decomposition.trend.values, linewidth=2, color='red')
        ax3.set_title('COMPONENTE DE TENDENCIA (DescomposiciÃ³n)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Tendencia')
        ax3.grid(True, alpha=0.3)
        
        # Guardar componentes para anÃ¡lisis posterior
        trend_component = decomposition.trend
        seasonal_component = decomposition.seasonal
        residual_component = decomposition.resid
        
except Exception as e:
    ax3.text(0.5, 0.5, f'Error en descomposiciÃ³n:\n{str(e)}', 
             transform=ax3.transAxes, ha='center', va='center')
    ax3.set_title('DESCOMPOSICIÃ“N - No disponible', fontsize=12, fontweight='bold')

# SUBPLOT 4: Test de estacionariedad (ADF)
ax4 = fig.add_subplot(gs[1, 2])
ax4.axis('off')

# Test de Dickey-Fuller aumentado
adf_result = adfuller(df_analysis['net_load_kwh'].dropna())
adf_text = f"""TEST DE ESTACIONARIEDAD (ADF):

EstadÃ­stico ADF: {adf_result[0]:.4f}
p-value: {adf_result[1]:.4f}
Lags utilizados: {adf_result[2]}
NÃºmero de observaciones: {adf_result[3]}

Valores crÃ­ticos:
  1%: {adf_result[4]['1%']:.4f}
  5%: {adf_result[4]['5%']:.4f}
  10%: {adf_result[4]['10%']:.4f}

InterpretaciÃ³n: {'ESTACIONARIA' if adf_result[1] < 0.05 else 'NO ESTACIONARIA'}"""

ax4.text(0.1, 0.9, adf_text, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
ax4.set_title('TEST DE ESTACIONARIEDAD', fontsize=12, fontweight='bold')

# ==========================================
# 3. ANÃ�LISIS DE TENDENCIA POR COMPONENTES TEMPORALES
# ==========================================
print("\nğŸ�¯ 3. ANÃ�LISIS DE TENDENCIA POR COMPONENTES TEMPORALES")
print("=" * 40)

# SUBPLOT 5: Tendencia por meses
ax5 = fig.add_subplot(gs[2, 0])

monthly_trend = df_analysis.groupby(df_analysis.index.month)['net_load_kwh'].mean()
month_names = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']

ax5.bar(range(1, 13), [monthly_trend.get(i, np.nan) for i in range(1, 13)], alpha=0.7, color='orange')
ax5.set_title('TENDENCIA MENSUAL - Promedio por Mes', fontsize=12, fontweight='bold')
ax5.set_xlabel('Mes')
ax5.set_ylabel('Net Load kWh Promedio')
ax5.set_xticks(range(1, 13))
ax5.set_xticklabels(month_names, rotation=45)
ax5.grid(True, alpha=0.3)

# SUBPLOT 6: Tendencia por dÃ­a de la semana
ax6 = fig.add_subplot(gs[2, 1])

dow_trend = df_analysis.groupby(df_analysis.index.dayofweek)['net_load_kwh'].mean()
dow_names = ['Lun', 'Mar', 'MiÃ©', 'Jue', 'Vie', 'SÃ¡b', 'Dom']

ax6.bar(dow_names, dow_trend.values, alpha=0.7, color='green')
ax6.set_title('TENDENCIA SEMANAL - Promedio por DÃ­a', fontsize=12, fontweight='bold')
ax5.set_xlabel('DÃ­a de la Semana')
ax6.set_ylabel('Net Load kWh Promedio')
ax6.grid(True, alpha=0.3)

# SUBPLOT 7: Tendencia por hora del dÃ­a
ax7 = fig.add_subplot(gs[2, 2])

hourly_trend = df_analysis.groupby(df_analysis.index.hour)['net_load_kwh'].mean()

ax7.plot(hourly_trend.index, hourly_trend.values, linewidth=3, marker='o', color='red')
ax7.set_title('TENDENCIA DIARIA - PatrÃ³n Horario', fontsize=12, fontweight='bold')
ax7.set_xlabel('Hora del DÃ­a')
ax7.set_ylabel('Net Load kWh Promedio')
ax7.set_xticks(range(0, 24, 2))
ax7.grid(True, alpha=0.3)

# ==========================================
# 4. ANÃ�LISIS DE CAMBIO EN LA TENDENCIA
# ==========================================
print("\nğŸ�¯ 4. ANÃ�LISIS DE CAMBIO EN LA TENDENCIA")
print("=" * 40)

# SUBPLOT 8: AnÃ¡lisis de cambios estructurales
ax8 = fig.add_subplot(gs[3, 0])

# Dividir la serie en segmentos y calcular estadÃ­sticas
n_segments = 4
segment_length = len(df_analysis) // n_segments
segment_stats = []

for i in range(n_segments):
    start_idx = i * segment_length
    end_idx = start_idx + segment_length if i < n_segments - 1 else len(df_analysis)
    segment = df_analysis['net_load_kwh'].iloc[start_idx:end_idx]
    
    segment_stats.append({
        'segment': i + 1,
        'mean': segment.mean(),
        'std': segment.std(),
        'min': segment.min(),
        'max': segment.max(),
        'start_date': df_analysis.index[start_idx],
        'end_date': df_analysis.index[end_idx - 1] if end_idx > 0 else df_analysis.index[-1]
    })

# Graficar medias por segmento
segments = [f"Seg {i+1}" for i in range(n_segments)]
means = [stat['mean'] for stat in segment_stats]

ax8.bar(segments, means, alpha=0.7, color='purple')
ax8.set_title('CAMBIOS ESTRUCTURALES - Media por Segmento', fontsize=12, fontweight='bold')
ax8.set_ylabel('Net Load kWh Promedio')
ax8.grid(True, alpha=0.3)

# SUBPLOT 9: AnÃ¡lisis de volatilidad (desviaciÃ³n estÃ¡ndar mÃ³vil)
ax9 = fig.add_subplot(gs[3, 1])

rolling_std = df_analysis['net_load_kwh'].rolling(window=672, center=True).std()  # 1 semana
ax9.plot(df_analysis.index, rolling_std, linewidth=1, color='brown', alpha=0.7)
ax9.set_title('VOLATILIDAD - DesviaciÃ³n EstÃ¡ndar MÃ³vil (1 semana)', fontsize=12, fontweight='bold')
ax9.set_ylabel('DesviaciÃ³n EstÃ¡ndar')
ax9.grid(True, alpha=0.3)
ax9.tick_params(axis='x', rotation=45)

# SUBPLOT 10: Resumen de hallazgos
ax10 = fig.add_subplot(gs[3, 2])
ax10.axis('off')

# Calcular mÃ©tricas de cambio
total_change = df_analysis['net_load_kwh'].iloc[-1] - df_analysis['net_load_kwh'].iloc[0]
percent_change = (total_change / abs(df_analysis['net_load_kwh'].iloc[0])) * 100
volatility_ratio = df_analysis['net_load_kwh'].std() / abs(df_analysis['net_load_kwh'].mean())

findings_text = f"""HALLAZGOS PRINCIPALES:

CAMBIO TOTAL:
â€¢ Absoluto: {total_change:.1f} kWh
â€¢ Porcentual: {percent_change:.1f}%

VOLATILIDAD:
â€¢ Coeficiente: {volatility_ratio:.3f}
â€¢ Estacionariedad: {'SÃ­' if adf_result[1] < 0.05 else 'No'}

PATRONES DETECTADOS:
â€¢ Estacionalidad: {'Fuerte' if 'seasonal_component' in locals() and seasonal_component.std() > 50 else 'Moderada/DÃ©bil'}
â€¢ Tendencia: {'Clara' if abs(total_change) > 50 else 'Estable'}
â€¢ Ciclicidad: {'Presente' if 'trend_component' in locals() and len(trend_component.dropna()) > 10 else 'Por analizar'}"""

ax10.text(0.1, 0.9, findings_text, transform=ax10.transAxes, fontsize=9,
          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
ax10.set_title('RESUMEN DE HALLAZGOS', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('exhaustive_trend_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ==========================================
# 5. ANÃ�LISIS ESTADÃ�STICO AVANZADO
# ==========================================
print("\nğŸ�¯ 5. ANÃ�LISIS ESTADÃ�STICO AVANZADO")
print("=" * 40)

# Test de normalidad
normality_test = stats.normaltest(df_analysis['net_load_kwh'].dropna())
print(f"ğŸ“Š TEST DE NORMALIDAD (D'Agostino):")
print(f"   â€¢ EstadÃ­stico: {normality_test.statistic:.4f}")
print(f"   â€¢ p-value: {normality_test.pvalue:.4f}")
print(f"   â€¢ InterpretaciÃ³n: {'Normal' if normality_test.pvalue > 0.05 else 'No Normal'}")

# AutocorrelaciÃ³n
from statsmodels.tsa.stattools import acf

lags = 96  # 1 dÃ­a de autocorrelaciÃ³n
autocorr = acf(df_analysis['net_load_kwh'].dropna(), nlags=lags, fft=True)

print(f"\nğŸ“Š AUTOCORRELACIÃ“N (primeros 10 lags):")
for i in range(1, 11):
    print(f"   â€¢ Lag {i}: {autocorr[i]:.4f}")

# AnÃ¡lisis de outliers
Q1 = df_analysis['net_load_kwh'].quantile(0.25)
Q3 = df_analysis['net_load_kwh'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df_analysis[(df_analysis['net_load_kwh'] < lower_bound) | (df_analysis['net_load_kwh'] > upper_bound)]
print(f"\nğŸ“Š ANÃ�LISIS DE OUTLIERS:")
print(f"   â€¢ LÃ­mite inferior: {lower_bound:.2f}")
print(f"   â€¢ LÃ­mite superior: {upper_bound:.2f}")
print(f"   â€¢ Outliers detectados: {len(outliers):,} ({len(outliers)/len(df_analysis)*100:.2f}%)")

# ==========================================
# 6. ANÃ�LISIS DE TENDENCIA CON MODELOS
# ==========================================
print("\nğŸ�¯ 6. MODELADO DE TENDENCIA")
print("=" * 40)

# Modelo de tendencia lineal
X = np.arange(len(df_analysis)).reshape(-1, 1)
y = df_analysis['net_load_kwh'].values

# Ajustar modelo lineal
from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(X, y)
linear_trend = linear_model.predict(X)
linear_slope = linear_model.coef_[0]

print(f"ğŸ“ˆ MODELO DE TENDENCIA LINEAL:")
print(f"   â€¢ Pendiente: {linear_slope:.6f} kWh/periodo")
print(f"   â€¢ Intercepto: {linear_model.intercept_:.2f}")
print(f"   â€¢ RÂ²: {linear_model.score(X, y):.4f}")

# Tendencia polinÃ³mica (grado 2)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

poly_model = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())
])
poly_model.fit(X, y)
poly_trend = poly_model.predict(X)
poly_r2 = poly_model.score(X, y)

print(f"ğŸ“ˆ MODELO DE TENDENCIA POLINÃ“MICA (grado 2):")
print(f"   â€¢ RÂ²: {poly_r2:.4f}")

# ==========================================
# 7. VISUALIZACIÃ“N DE MODELOS DE TENDENCIA
# ==========================================
fig2, (ax11, ax12) = plt.subplots(2, 1, figsize=(15, 10))

# SUBPLOT 1: ComparaciÃ³n de modelos de tendencia
ax11.plot(df_analysis.index, df_analysis['net_load_kwh'], linewidth=0.5, alpha=0.5, color='blue', label='Datos Originales')
ax11.plot(df_analysis.index, linear_trend, linewidth=2, color='red', label=f'Tendencia Lineal (RÂ²: {linear_model.score(X, y):.3f})')
ax11.plot(df_analysis.index, poly_trend, linewidth=2, color='green', label=f'Tendencia PolinÃ³mica (RÂ²: {poly_r2:.3f})')

ax11.set_title('MODELOS DE TENDENCIA AJUSTADOS', fontsize=14, fontweight='bold')
ax11.set_ylabel('Net Load kWh')
ax11.legend()
ax11.grid(True, alpha=0.3)
ax11.tick_params(axis='x', rotation=45)

# SUBPLOT 2: Residuales del modelo lineal
residuals = y - linear_trend
ax12.plot(df_analysis.index, residuals, linewidth=0.8, alpha=0.7, color='purple')
ax12.axhline(y=0, color='red', linestyle='--', linewidth=1)
ax12.set_title('RESIDUALES - Modelo Lineal', fontsize=14, fontweight='bold')
ax12.set_ylabel('Residuales (kWh)')
ax12.set_xlabel('Fecha')
ax12.grid(True, alpha=0.3)
ax12.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('trend_modeling_analysis.png', dpi=300, bbox_inches='tight')
plt.show()




import requests
from datetime import datetime, timedelta

print("\nğŸš€ EXECUTING WEATHER DATA FETCH WITH BOUNDING BOXES - 15 MINUTE DATA")
print("=" * 70)

# 1. Definir bounding boxes
city_bboxes = {
    'Amsterdam': {
        'north': 52.4300, 'south': 52.3050, 
        'east': 5.0500, 'west': 4.7500,
        'points_per_side': 2,
        'weight': 0.5
    },
    'Rotterdam': {
        'north': 51.9800, 'south': 51.8500,
        'east': 4.6000, 'west': 4.4000,
        'points_per_side': 2,
        'weight': 0.25
    },
    'Utrecht': {
        'north': 52.1500, 'south': 52.0500,
        'east': 5.2000, 'west': 5.0500,
        'points_per_side': 2,
        'weight': 0.125}
   ,
    
    'The_Hague': {
        'north': 52.1200, 'south': 52.0300,
        'east': 4.3800, 'west': 4.2500,
        'points_per_side': 2,
        'weight': 0.125
    }
}

def generate_grid_points(bbox, city_name, points_per_side):
    """Generate grid points within bounding box"""
    lat_points = np.linspace(bbox['south'], bbox['north'], points_per_side)
    lon_points = np.linspace(bbox['west'], bbox['east'], points_per_side)
    
    points = []
    for i, lat in enumerate(lat_points):
        for j, lon in enumerate(lon_points):
            point_name = f"{city_name}_point_{i+1}_{j+1}"
            points.append((lat, lon, point_name))
    
    return points

def create_bbox_locations(city_bboxes):
    """Create locations list with bounding box grid points"""
    all_locations = []
    
    for city_name, bbox in city_bboxes.items():
        print(f"ğŸ“� Generating {bbox['points_per_side']}x{bbox['points_per_side']} grid for {city_name}")
        
        grid_points = generate_grid_points(bbox, city_name, bbox['points_per_side'])
        weight_per_point = bbox['weight'] / len(grid_points)
        
        for lat, lon, point_name in grid_points:
            all_locations.append((lat, lon, point_name, weight_per_point))
            
        print(f"   Added {len(grid_points)} points with {weight_per_point:.3f} weight each")
    
    return all_locations

# 2. FUNCIÃ“N RESAMPLE 
def resample_to_15min(df):
    """
    Resample hourly data to 15-minute frequency with appropriate methods
    """
    if df.empty:
        return df
    
    print(f"    Resampling from hourly to 15-minute frequency...")
    print(f"    Original shape: {df.shape}, frequency: {pd.infer_freq(df.index)}")
    
    # Different interpolation methods for different variable types
    linear_cols = [col for col in df.columns if any(x in col for x in 
                   ['temperature', 'humidity', 'dew_point', 'apparent_temperature', 
                    'pressure', 'vapour_pressure_deficit', 'et0_fao_evapotranspiration'])]
    
    forward_fill_cols = [col for col in df.columns if any(x in col for x in 
                         ['weather_code', 'precipitation', 'rain'])]
    
    zero_fill_cols = [col for col in df.columns if 'radiation' in col]
    
    wind_cols = [col for col in df.columns if 'wind' in col]
    
    cloud_cols = [col for col in df.columns if 'cloud' in col]
    
    resampled_dfs = []
    
    # Linear interpolation for continuous variables
    if linear_cols:
        linear_df = df[linear_cols].resample('15min').interpolate(method='linear')
        resampled_dfs.append(linear_df)
        print(f"      Linear interpolation: {len(linear_cols)} variables")
    
    # Forward fill for categorical and precipitation
    if forward_fill_cols:
        ff_df = df[forward_fill_cols].resample('15min').ffill()
        resampled_dfs.append(ff_df)
        print(f"      Forward fill: {len(forward_fill_cols)} variables")
    
    # Radiation - linear interpolation but zero at night
    if zero_fill_cols:
        rad_df = df[zero_fill_cols].resample('15min').interpolate(method='linear')
        # Set radiation to zero during night hours
        night_hours = (rad_df.index.hour < 6) | (rad_df.index.hour >= 21)
        rad_df[night_hours] = 0
        resampled_dfs.append(rad_df)
        print(f"      Radiation with night zero: {len(zero_fill_cols)} variables")
    
    # Wind variables - linear interpolation
    if wind_cols:
        wind_df = df[wind_cols].resample('15min').interpolate(method='linear')
        resampled_dfs.append(wind_df)
        print(f"      Wind interpolation: {len(wind_cols)} variables")
    
    # Cloud cover - linear interpolation
    if cloud_cols:
        cloud_df = df[cloud_cols].resample('15min').interpolate(method='linear')
        resampled_dfs.append(cloud_df)
        print(f"      Cloud cover interpolation: {len(cloud_cols)} variables")
    
    # Combine all resampled data
    result_df = pd.concat(resampled_dfs, axis=1)
    
    # Ensure all original columns are present
    missing_cols = set(df.columns) - set(result_df.columns)
    if missing_cols:
        print(f"      Default interpolation: {len(missing_cols)} variables")
        for col in missing_cols:
            result_df[col] = df[col].resample('15min').interpolate(method='linear')
    
    # Reorder to match original
    result_df = result_df[df.columns]
    
    print(f"    Final shape: {result_df.shape}, samples: {len(result_df):,}")
    return result_df

# 3. FUNCIÃ“N  PARA DATOS SINTÃ‰TICOS EN 15 MINUTOS
def generate_synthetic_15min_data(start_date, end_date, location_name, lat, lon):
    """Generate synthetic weather data at 15-minute frequency when API fails"""
    dates = pd.date_range(start=start_date, end=end_date, freq='15min', tz='Europe/Amsterdam')
    n = len(dates)
    np.random.seed(abs(hash(location_name)) % 10000)
    
    synthetic_df = pd.DataFrame(index=dates)
    
    # Time-based patterns at 15-minute resolution
    hour_of_day = dates.hour + dates.minute / 60.0  # Hora fraccional para mayor precisiÃ³n
    day_of_year = dates.dayofyear
    
    # Basic weather simulation at 15-min resolution
    base_temp = 10 + 8 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    diurnal_temp = 6 * np.sin(2 * np.pi * (hour_of_day - 14) / 24)
    
    synthetic_df['temperature_2m'] = base_temp + diurnal_temp + np.random.normal(0, 0.8, n)
    synthetic_df['relative_humidity_2m'] = np.clip(
        70 + 10 * np.sin(2 * np.pi * (hour_of_day - 4) / 24) + np.random.normal(0, 5, n), 
        40, 95
    )
    synthetic_df['precipitation'] = np.random.exponential(0.05, n) * (np.random.random(n) < 0.025)
    synthetic_df['wind_speed_10m'] = np.abs(4 + np.random.normal(0, 1.0, n))
    
    # Add more variables for completeness
    synthetic_df['apparent_temperature'] = synthetic_df['temperature_2m'] + np.random.normal(0, 0.5, n)
    synthetic_df['pressure_msl'] = 1015 + np.random.normal(0, 5, n)
    synthetic_df['cloud_cover'] = np.clip(50 + np.random.normal(0, 20, n), 0, 100)
    
    return synthetic_df

# 4. FUNCIÃ“N PRINCIPAL 
def fetch_weather_data_bbox(start_date, end_date, bbox_locations):
    """Fetch weather data using bounding box grid points and resample to 15 minutes"""
    
    weather_features = [
        'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'apparent_temperature',
        'precipitation', 'rain', 'weather_code',
        'pressure_msl', 'surface_pressure',
        'cloud_cover', 'cloud_cover_low', 'cloud_cover_mid', 'cloud_cover_high',
        'shortwave_radiation', 'direct_radiation', 'diffuse_radiation',
        'et0_fao_evapotranspiration', 'vapour_pressure_deficit',
        'wind_speed_10m', 'wind_direction_10m', 'wind_gusts_10m'
    ]
    
    all_weather_data = {}
    location_weights = {}
    
    print(f"\nğŸŒ¤ï¸� FETCHING WEATHER DATA FOR {len(bbox_locations)} GRID POINTS")
    print(f"   Date range: {start_date} to {end_date}")
    print(f"   Target frequency: 15 minutes")
    
    for lat, lon, location_name, weight in bbox_locations:
        print(f"  ğŸ“� {location_name} (weight: {weight:.3f})...")
        location_weights[location_name] = weight
        
        try:
            response = requests.get(
                "https://archive-api.open-meteo.com/v1/archive",
                params={
                    'latitude': lat,
                    'longitude': lon,
                    'start_date': start_date,
                    'end_date': end_date,
                    'hourly': ','.join(weather_features),
                    'timezone': 'Europe/Amsterdam',
                    'models': 'era5'
                },
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            # Process hourly data
            hourly_df = pd.DataFrame(data['hourly'])
            hourly_df['timestamp_local'] = pd.to_datetime(hourly_df['time'])
            hourly_df = hourly_df.drop('time', axis=1)
            hourly_df = hourly_df.set_index('timestamp_local')
            
            # Resample to 15 minutes
            df_15min = resample_to_15min(hourly_df)
            
            # Add location identifier to column names
            df_15min.columns = [f"{col}_{location_name}" for col in df_15min.columns]
            
            all_weather_data[location_name] = df_15min
            print(f"     âœ… {len(df_15min.columns)} variables at 15-min frequency")
            
        except Exception as e:
            print(f"     â�Œ API Error: {str(e)[:80]}...")
            print(f"     Generating synthetic 15-min data for {location_name}")
            all_weather_data[location_name] = generate_synthetic_15min_data(
                start_date, end_date, location_name, lat, lon
            )
    
    return all_weather_data, location_weights

# 5. FUNCIÃ“N DE CONSOLIDACIÃ“N 
def consolidate_bbox_weather_corrected(weather_data_dict, location_weights):
    """
    CORRECTED VERSION: Consolidate weather data from multiple grid points
    """
    print(f"\nğŸ”„ CONSOLIDATING WEATHER DATA FROM {len(weather_data_dict)} GRID POINTS")
    print(f"   Target frequency: 15 minutes")
    
    # Get first location to initialize the dataframe
    first_location = list(weather_data_dict.keys())[0]
    consolidated_df = pd.DataFrame(index=weather_data_dict[first_location].index)
    
    # Verificar frecuencia
    freq = pd.infer_freq(consolidated_df.index)
    print(f"   Data frequency: {freq}")
    
    # Lista conocida de variables base que esperamos
    expected_base_variables = [
        'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'apparent_temperature',
        'precipitation', 'rain', 'weather_code', 'pressure_msl', 'surface_pressure',
        'cloud_cover', 'cloud_cover_low', 'cloud_cover_mid', 'cloud_cover_high',
        'shortwave_radiation', 'direct_radiation', 'diffuse_radiation',
        'et0_fao_evapotranspiration', 'vapour_pressure_deficit',
        'wind_speed_10m', 'wind_direction_10m', 'wind_gusts_10m'
    ]
    
    print(f"   Consolidating {len(expected_base_variables)} expected variables...")
    
    # Consolidar cada variable usando promedio ponderado
    variables_added = 0
    
    for base_var in expected_base_variables:
        weighted_sum = None
        total_weight = 0
        locations_with_data = 0
        
        for location_name, location_df in weather_data_dict.items():
            location_col = f"{base_var}_{location_name}"
            if location_col in location_df.columns:
                location_weight = location_weights[location_name]
                
                if weighted_sum is None:
                    weighted_sum = location_df[location_col] * location_weight
                else:
                    weighted_sum += location_df[location_col] * location_weight
                
                total_weight += location_weight
                locations_with_data += 1
        
        if weighted_sum is not None and total_weight > 0:
            consolidated_df[base_var] = weighted_sum / total_weight
            variables_added += 1
            print(f"     âœ… {base_var:25} | {locations_with_data:2d} locations | weight: {total_weight:.3f}")
        else:
            print(f"     â�Œ {base_var:25} | no data available")
    
    print(f"âœ… CONSOLIDATION COMPLETE:")
    print(f"   Variables added: {variables_added}/{len(expected_base_variables)}")
    print(f"   Final dataset shape: {consolidated_df.shape}")
    print(f"   Time period: {consolidated_df.index.min()} to {consolidated_df.index.max()}")
    print(f"   Frequency: {pd.infer_freq(consolidated_df.index)}")
    
    return consolidated_df

# 6. EJECUCIÃ“N PRINCIPAL
print("\nğŸ“‹ EXECUTION PLAN:")
print("   1. Create bounding box locations")
print("   2. Fetch weather data for each grid point") 
print("   3. Resample hourly data to 15-minute frequency")
print("   4. Consolidate all data into single DataFrame")

# Crear localizaciones con bounding boxes
bbox_locations = create_bbox_locations(city_bboxes)

# Fetch datos meteorolÃ³gicos
start_date = "2025-05-08"
end_date = "2025-09-01"

print(f"\nğŸ“… Date range: {start_date} to {end_date}")

# Fetch datos de la API y resample a 15 minutos
weather_data_dict, location_weights = fetch_weather_data_bbox(
    start_date, end_date, bbox_locations
)

# Consolidar datos en un Ãºnico DataFrame
netherlands_weather_bbox = consolidate_bbox_weather_corrected(weather_data_dict, location_weights)

# 7. VERIFICACIÃ“N FINAL
print(f"\nğŸ�¯ FINAL 15-MINUTE WEATHER DATASET:")
print(f"   Shape: {netherlands_weather_bbox.shape}")
print(f"   Variables: {len(netherlands_weather_bbox.columns)}")
print(f"   Samples: {len(netherlands_weather_bbox):,}")
print(f"   Memory: {netherlands_weather_bbox.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
print(f"   Frequency: {pd.infer_freq(netherlands_weather_bbox.index)}")

if len(netherlands_weather_bbox.columns) > 0:
    print(f"\nğŸ“Š VARIABLES IN DATAFRAME:")
    for i, col in enumerate(netherlands_weather_bbox.columns, 1):
        dtype = netherlands_weather_bbox[col].dtype
        nan_count = netherlands_weather_bbox[col].isna().sum()
        print(f"   {i:2d}. {col:<30} | {str(dtype):<10} | NaN: {nan_count:3d}")
    
    print(f"\nğŸ”� FIRST 3 ROWS:")
    print(netherlands_weather_bbox.head(3))


