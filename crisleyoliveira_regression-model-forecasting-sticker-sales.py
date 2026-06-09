# Essential Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import holidays
from colorama import Fore, Style
from category_encoders import TargetEncoder
from statsmodels.tsa.seasonal import seasonal_decompose
import requests
import os

# Preprocessing
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from scipy.stats import boxcox
from sklearn.base import BaseEstimator, TransformerMixin

# Templates and optimization
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import optuna

# Metrics and utilities
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import time
from sklearn.base import clone
from typing import Tuple


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


train_df = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
sample = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')


print(Fore.GREEN + Style.BRIGHT + "INITIAL INFORMATION OF EACH DATASET" + Style.RESET_ALL)
print(Fore.BLUE + "------TRAIN DF------" + Style.RESET_ALL)
print(train_df.info())
print(Fore.BLUE + "------TEST DF------" + Style.RESET_ALL)
print(test_df.info())


print(Fore.BLUE + Style.BRIGHT + "-----TRAINING SET DESCRIPTIVE STATISTICS-----" + Style.RESET_ALL)
print(train_df.describe())

print(Fore.BLUE + Style.BRIGHT + "------TEST SET DESCRIPTIVE STATISTICS-----" + Style.RESET_ALL)
print(test_df.describe())


print(print(Fore.BLUE + Style.BRIGHT + "-----TRAINING SET MISSING VALUES-----" + Style.RESET_ALL))
print(train_df.isnull().sum())
print(print(Fore.BLUE + Style.BRIGHT + "-----TEST SET MISSING VALUES-----" + Style.RESET_ALL))
print(test_df.isnull().sum())


# Initial setup
plt.style.use('seaborn-v0_8-darkgrid')
palette = sns.color_palette("husl", 8)


# Select category columns
cat_cols = train_df.select_dtypes(include=['object']).columns

# Analyze distribution of categorical columns
categorical_distribution = {
    col: {
        'Unique Values': train_df[col].nunique(),
        'Top Value': train_df[col].mode()[0],
        'Top Frequency': train_df[col].value_counts().iloc[0]
    } for col in cat_cols
}

# Format results into a DataFrame
categorical_distribution_df = pd.DataFrame.from_dict(categorical_distribution, orient='index')
categorical_distribution_df.reset_index(inplace=True)
categorical_distribution_df.rename(columns={'index': 'Column'}, inplace=True)

categorical_distribution_df


# Distribution of column `num_sold`
plt.figure(figsize=(10, 6))
plt.hist(train_df['num_sold'].dropna(), bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Products Sold')
plt.xlabel('num_sold')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


# Comparison of sales by country
plt.figure(figsize=(8,6))
sns.boxplot(x='country', y='num_sold', data=train_df)
plt.title('Distribution of num_sold by Country')
plt.xlabel('Country')
plt.ylabel('Sales (num_sold)')
plt.tight_layout()
plt.show()


# Comparison of sales by product
plt.figure(figsize=(8,6))
sns.boxplot(x='product', y='num_sold', data=train_df)
plt.title('Distribution of num_sold by Product')
plt.xlabel('Product')
plt.ylabel('Sales (num_sold)')
plt.tight_layout()
plt.show()


# Daily aggregation
daily_sales = train_df.groupby('date')['num_sold'].sum().reset_index()
daily_sales['date'] = pd.to_datetime(daily_sales['date'])

# Time series with moving average
plt.figure(figsize=(20, 6))
plt.plot(daily_sales['date'], daily_sales['num_sold'], alpha=0.3, label='Daily Sales')
plt.plot(daily_sales['date'], daily_sales['num_sold'].rolling(7).mean(),
         color='red', linewidth=2, label='7-day moving average')
plt.title('Sales Over Time with Trend')
plt.xlabel('Date')
plt.ylabel('Number of Sales')
plt.legend()
plt.show()


# Additive decomposition
result = seasonal_decompose(daily_sales.set_index('date')['num_sold'],
                            model='additive',
                            period=365) # Annual period

plt.figure(figsize=(20,12))
result.plot()
plt.suptitle('Seasonal Sales Decomposition', y=1.02)
plt.tight_layout()
plt.show()


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.figure(figsize=(20, 6))
plot_acf(daily_sales['num_sold'], lags=365, alpha=0.05)
plt.title('Autocorrelation of Sales (1-Year Lag)')
plt.show()


# Sales by country over time
plt.figure(figsize=(20, 8))
sns.lineplot(data=train_df, x='date', y='num_sold', hue='country',
             estimator='sum', errorbar=None, palette=palette)
plt.title('Sales by Country Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.legend(title='Country')
plt.show()

# Sales by product and country
plt.figure(figsize=(18, 10))
sns.catplot(data=train_df, x='product', y='num_sold', col='country',
            kind='box', col_wrap=3, palette=palette, height=5, aspect=1.2)
plt.suptitle('Sales Distribution by Product and Country', y=1.05)
plt.show()


# Check temporal pattern of missing values
missing_dates = train_df[train_df['num_sold'].isnull()]['date']
plt.figure(figsize=(12, 4))
sns.histplot(pd.to_datetime(missing_dates), bins=50, kde=False)
plt.title('Temporal Distribution of Missing Values')
plt.show()

# Check distribution by country/store/product
for col in ['country', 'store', 'product']:
    print(f"Missing por {col}:")
    print(train_df[col][train_df['num_sold'].isnull()].value_counts(normalize=True))


train_df['date'] = pd.to_datetime(train_df['date'])
train_df['is_test'] = False
test_df['date'] = pd.DatetimeIndex(test_df['date'])
test_df['is_test'] = True


from dataclasses import dataclass

@dataclass
class DataConfig:
    # References to the original DataFrames
    train_data: pd.DataFrame
    test_data: pd.DataFrame
    
    # Validation and processing parameters
    validation_year: int = 2015
    fft_window: int = 8
    holiday_effect_duration: int = 10
    
    # Auxiliary dictionaries
    """Store standard ISO country codes for integration with external systems and consistent processing of geographic data"""
    country_standards = {
        'iso3_codes': {
            'Finland': 'FIN', 'Canada': 'CAN', 'Italy': 'IT',
            'Kenya': 'KEN', 'Singapore': 'SGP', 'Norway': 'NOR'
        },
        'iso2_codes': {
            'Finland': 'FI', 'Canada': 'CA', 'Italy': 'IT',
            'Kenya': 'KE', 'Singapore': 'SG', 'Norway': 'NO'
        }
    }
    
    @property
    def temporal_scope(self):
        """Defines the full temporal scope of the data"""
        train_years = self.train_data['date'].dt.year.unique()
        test_years = self.test_data['date'].dt.year.unique()
        return {
            'train': train_years,
            'test': test_years,
            'full': np.union1d(train_years, test_years)
        }
    
    @property
    def categorical_entities(self):
        """Extracts unique categories from training data"""
        return {
            'countries': self.train_data['country'].unique().tolist(),
            'stores': self.train_data['store'].unique().tolist(),
            'products': self.train_data['product'].unique().tolist()
        }
    
    @property
    def processing_params(self):
        """Parameters for data transformations"""
        return {
            'fft_filter_width': self.fft_window,
            'holiday_response_length': self.holiday_effect_duration
        }

    @property
    def sincos_features(self):
        return {
            'base': ['sin t', 'cos t', 'sin t/2', 'cos t/2'],
            'extended': ['sin 2t', 'cos 2t', 'sin t', 'cos t', 'sin t/2', 'cos t/2']
        }

config = DataConfig(
    train_data=train_df,
    test_data=test_df
)

print("Years of training:", config.temporal_scope['train'])
print("Available countries::", config.categorical_entities['countries'])
print("ISO code of Norway:", config.country_standards['iso3_codes']['Norway'])


class TimeSeriesPreprocessor:
    """
    Encapsulates the time series preprocessing pipeline.
    """

    def __init__(self, config: DataConfig):
        self.config = config

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineers time-based features.

        Args:
            df: DataFrame with the 'date' column.

        Returns:
            DataFrame with added temporal features.
        """
        df = df.copy()

        # Convert and validate datetime
        df['date'] = pd.to_datetime(df['date'], errors='coerce', format='%Y-%m-%d')
        if df['date'].isnull().any():
            raise ValueError("Invalid dates found in the dataset")

        # Basic temporal features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['dayofyear'] = df['date'].dt.dayofyear

        # Relative time references
        base_date = pd.Timestamp('2000-01-01')
        df['daynum'] = (df['date'] - base_date).dt.days
        df['weeknum'] = df['daynum'] // 7

        return df

    def calculate_year_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates year-specific metrics considering data density.

        Args:
            df: DataFrame with the 'year' column.

        Returns:
            DataFrame with added year metrics.
        """
        # Days per year calculation
        yearly_obs = df.groupby('year')['date'].nunique().rename('daysinyear')
        divisor = (
            len(self.config.categorical_entities['countries']) *
            len(self.config.categorical_entities['stores']) *
            len(self.config.categorical_entities['products'])
        )
        df = df.merge((yearly_obs / divisor).astype(int), on='year', how='left')

        # Year progression features
        df['partofyear'] = (df['dayofyear'] - 1) / df['daysinyear']
        df['partof2year'] = df['partofyear'] + df['year'] % 2

        return df

    def create_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates sinusoidal representations for cyclical patterns.

        Args:
            df: DataFrame with 'partofyear' and 'partof2year' columns.

        Returns:
            DataFrame with added cyclical features.
        """
        frequency_map = {
            '_4t': (8, False),
            '_3t': (6, False),
            '_2t': (4, False),
            '_t': (2, False),
            '_t_half': (1, True)
        }

        for suffix, (multiplier, use_alt) in frequency_map.items():
            angle_source = df['partof2year'] if use_alt else df['partofyear']
            angle = multiplier * np.pi * angle_source

            df[f'sin{suffix}'] = np.sin(angle)
            df[f'cos{suffix}'] = np.cos(angle)

        return df.drop(columns=['daysinyear', 'partofyear', 'partof2year'])

    def full_pipeline(self, train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        End-to-end preprocessing pipeline.

        Args:
            train: Training DataFrame.
            test: Test DataFrame.

        Returns:
            Tuple containing the processed training and test DataFrames.
        """
        # Process datasets separately
        train_processed = self.create_temporal_features(train)
        test_processed = self.create_temporal_features(test)

        # Combine for year metrics calculation
        temp_df = pd.concat([train_processed, test_processed])
        temp_df = self.calculate_year_metrics(temp_df)

        # Split back and add cyclical features
        train_final = self.create_cyclical_features(temp_df[temp_df['is_test'] == False].copy())
        test_final = self.create_cyclical_features(temp_df[temp_df['is_test'] == True].copy())

        return train_final, test_final


# --------------------------------------------------
# Run pipeline
# --------------------------------------------------

preprocessor = TimeSeriesPreprocessor(config)
train_df, test_df = preprocessor.full_pipeline(train_df, test_df)

# Combine final datasets
df = pd.concat([train_df, test_df])


def fetch_gdp_data(config: DataConfig) -> pd.DataFrame:
    """Create GDP DataFrame using DataConfig configuration"""
    
CACHE_DIR = "gdp_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def fetch_gdp_from_api(iso3_code, year):
    url = f"https://api.worldbank.org/v2/country/{iso3_code}/indicator/NY.GDP.PCAP.CD?date={year}&format=json"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        return data[1][0]['value'] if data[1] else np.nan
    except Exception as e:
        print(f"Error fetching GDP for {iso3_code} {year}: {str(e)}")
        return np.nan

def fetch_gdp_data(config: DataConfig) -> pd.DataFrame:
    """Creates GDP DataFrame using DataConfig configuration with caching."""
    countries = config.categorical_entities['countries']
    years = config.temporal_scope['full']
    gdp_matrix = []

    for country in countries:
        country_gdp = []
        iso3_code = config.country_standards['iso3_codes'][country]

        for year in years:
            cache_file = os.path.join(CACHE_DIR, f"{iso3_code}_{year}.csv")
            if os.path.exists(cache_file):
                gdp_value = pd.read_csv(cache_file, header=None).iloc[0, 0]
            else:
                gdp_value = fetch_gdp_from_api(iso3_code, year)
                pd.DataFrame([gdp_value]).to_csv(cache_file, header=None, index=None)

            country_gdp.append(gdp_value)

        gdp_matrix.append(country_gdp)

    return pd.DataFrame(
        gdp_matrix,
        index=countries,
        columns=years
    )


# Generate GDP DF
gdp_data = fetch_gdp_data(config)

# Fill missing values
#gdp_data.ffill(axis=1, inplace=True)  # Forward fill
gdp_data.bfill(axis=1, inplace=True)  # Backward fill

# Merge with master data
df = df.merge(
    gdp_data.stack().reset_index().rename(columns={'level_0':'country', 'level_1':'year', 0:'gdp'}),
    on=['country', 'year'],
    how='left'
)

# Check result
print(df[['country', 'year', 'gdp']].sample(5))


df.columns


# Correlation between GDP and sales
gdp_sales = (
    df[df['is_test'] == False]
    .groupby(['country', 'year'])
    .agg(
        total_sales=('num_sold', 'sum'), 
        gdp=('gdp', 'first')
    )
    .reset_index()
)

gdp_sales = gdp_sales.rename(columns={'gdp': 'gdp_per_capita'})

# Plot usando colunas nomeadas corretamente
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=gdp_sales,
    x='gdp_per_capita',  # Nome explÃ­cito
    y='total_sales',
    hue='country',
    palette='viridis',  # Palette definida
    s=100
)

plt.title('GDP per Capita vs Total Sales')
plt.xlabel('GDP per Capita (USD)')
plt.ylabel('Total Sales')
plt.legend(title='Country', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()


# Heatmap of sales by month and year
month_year_sales = train_df.groupby(['year', 'month'])['num_sold'].sum().unstack()
#print(month_year_sales)
#print(month_year_sales.shape)
#print(month_year_sales.isnull().sum())
plt.figure(figsize=(15, 8))
sns.heatmap(month_year_sales, cmap='YlGnBu', annot=True, fmt=".0f")
plt.title('Sales by Month and Year')
plt.xlabel('Month')
plt.ylabel('Year')
plt.show()


def plot_holiday_impact(countries, ncols=2):
    nrows = (len(countries) + ncols - 1) // ncols  
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12 * ncols, 6 * nrows))
    
    # Ensure axes is a numpy array for easy indexing
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.ravel()  # # Flatten the array for linear indexing

    for i, country_name in enumerate(countries):
        ax = axes[i]
        country_code = config.country_standards['iso3_codes'].get(country_name)
        if not country_code:
            print(f"ISO3 code not found for {country_name}")
            continue

        try:
            years_list = config.temporal_scope['full'].astype(int).tolist()
            country_holidays = holidays.CountryHoliday(country_code, years=years_list)
        except NotImplementedError:
            print(f"No holidays available for {country_name} ({country_code})")
            continue

        holiday_dates = [pd.Timestamp(date) for date in country_holidays.keys()]
        holiday_effect = []

        for h_date in holiday_dates:
            start_date = h_date - pd.Timedelta(days=7)
            end_date = h_date + pd.Timedelta(days=7)

            period_data = train_df[
                (train_df['date'].between(start_date, end_date)) &
                (train_df['country'] == country_name)
            ].copy()

            if not period_data.empty:
                period_data['days_from_holiday'] = (period_data['date'] - h_date).dt.days
                holiday_effect.append(
                    period_data.groupby('days_from_holiday')['num_sold'].mean()
                )

        if holiday_effect:
            holiday_df = pd.concat(holiday_effect).groupby('days_from_holiday').mean()
            holiday_df = holiday_df.to_frame(name="num_sold").reset_index()

            sns.lineplot(data=holiday_df, x='days_from_holiday', y='num_sold', ax=ax)
            ax.axvline(0, color='red', linestyle='--', label='Dia do Feriado')
            ax.set_title(f'Holiday Impact on Sales - {country_name}')
            ax.set_xlabel('Holiday Days')
            ax.set_ylabel('Average Sales')
            ax.legend()
        else:
            ax.set_visible(False)  # Hide empty subplotss

    # Oculta subplots nÃ£o utilizados restantes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()

plot_holiday_impact(config.categorical_entities['countries'], ncols=2)


class HolidayFeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, countries):
        self.countries = countries 
        self.country_holidays = {}

    def fit(self, X, y=None):
        """Learn holidays from training data ONLY"""
        # Extrair anos APENAS do treino
        train_years = X[X['is_test'] == False]['date'].dt.year.unique()
        
        for country in self.countries:
            iso_code = config.country_standards['iso3_codes'][country]
            
            # Criar dicionÃ¡rio apenas com anos de treino
            self.country_holidays[country] = {
                pd.Timestamp(date): name
                for year in train_years
                for date, name in holidays.CountryHoliday(iso_code, years=year).items()
            }
        return self

    def _days_from_nearest_holiday(self, date, country):
        """Calculates days until the nearest holiday (considering history)"""
        if country not in self.country_holidays:
            return np.nan
            
        holidays_list = self.country_holidays[country]
        if not holidays_list:
            return np.nan

        past_holidays = [h for h in holidays_list if h <= date]
        
        if not past_holidays:
            return np.nan
        
        deltas = [abs((date - h_date).days) for h_date in past_holidays]
        
        return min(deltas) if deltas else np.nan

    def transform(self, X):
        X = X.copy()
        # Converter para datetime se necessÃ¡rio
        if not pd.api.types.is_datetime64_any_dtype(X['date']):
            X['date'] = pd.to_datetime(X['date'])
            
        # Criar features
        X['is_holiday'] = X.apply(
            lambda row: row['date'] in self.country_holidays.get(row['country'], {}), 
            axis=1
        ).astype(int)
        
        X['days_from_holiday'] = X.apply(
            lambda row: self._days_from_nearest_holiday(row['date'], row['country']),
            axis=1
        )
        return X


# Instantiate with countries from dataset
countries = config.categorical_entities['countries']
holiday_gen = HolidayFeatureGenerator(countries=countries)

# Apply in training to learn holidays
holiday_gen.fit(df[df['is_test'] == False])  # Garante uso apenas de dados de treino

# Transform dataset
df = holiday_gen.transform(df)


df = df.dropna(subset=['num_sold'])


# Separate training and testing
train = df[df['is_test'] == False].copy()
test = df[df['is_test'] == True].copy()

# Ensure temporal order
train = train.sort_values('date')
test = test.sort_values('date')

# Remove auxiliary columns
X_train = train.drop(columns=['id', 'date', 'num_sold', 'is_test'])
y_train = train['num_sold']
X_test = test.drop(columns=['id', 'date', 'num_sold', 'is_test'])

# Configure temporal cross-validation
tscv = TimeSeriesSplit(n_splits=5)


# Identify categorical columns
cat_features = ['country', 'store', 'product']
num_features = [col for col in X_train.columns if col not in cat_features]

# Convert types
X_train[cat_features] = X_train[cat_features].astype('category')
X_test[cat_features] = X_test[cat_features].astype('category')


from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# Configure model with temporal processing
model_cb = CatBoostRegressor(
    cat_features=cat_features,
    iterations=1500,
    learning_rate=0.05,
    depth=8,
    random_seed=42,
    early_stopping_rounds=100,
    verbose=0
)

# Temporal cross-validation
time_split = TimeSeriesSplit(n_splits=5)
maes = []

for train_idx, val_idx in time_split.split(X_train):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    model_cb.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    
    preds = model_cb.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    maes.append(mae)
    print(f"Fold MAE: {mae:.2f}")

print(f"\nMAE MÃ©dio: {np.mean(maes):.2f}")


#test_predictions = best_model.predict(X_test_scaled)

# Ensure predictions are within expected range
#test_predictions = np.clip(test_predictions, 5, 5939)

# Create submission DataFrame
#submission = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')
#submission['num_sold'] = test_predictions

# Save the submission file
#submission.to_csv("submission.csv", index=False)

#print("Submission file generated successfully!")

