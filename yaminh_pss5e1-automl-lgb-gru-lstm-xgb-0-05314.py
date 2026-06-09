import holidays
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


train.date = pd.DatetimeIndex(train.date)
train['test'] = 0
test.date = pd.DatetimeIndex(test.date)
test['test'] = 1


class CFG:
    # Extract unique years from the datasets
    years_train = train['date'].dt.year.unique()
    years_test = test['date'].dt.year.unique()
    years = np.concatenate((years_train, years_test))

    validation_year = 2015

    # Unique entities in the training dataset
    countries = train['country'].unique()
    stores = train['store'].unique()
    products = train['product'].unique()

    # Country code mappings
    country_iso_codes = {
        'alpha3': {  # ISO 3166-1 alpha-3 codes
            'Finland': 'FIN', 
            'Canada': 'CAN', 
            'Italy': 'IT', 
            'Kenya': 'KEN', 
            'Singapore': 'SGP', 
            'Norway': 'NOR'
        },
        'alpha2': {  # ISO 3166-1 alpha-2 codes
            'Finland': 'FI', 
            'Canada': 'CA', 
            'Italy': 'IT', 
            'Kenya': 'KE', 
            'Singapore': 'SG', 
            'Norway': 'NO'
        }
    }
    
    # FFT filter configuration
    fft_filter_width = 8  # Smoothing window width for FFT filtering

    # Holiday response settings
    holiday_response_len = 10  # Days for holiday response period


# Combine train and test datasets
df = pd.concat((train, test))
df['date'] = pd.to_datetime(df['date'])

# Create date-related features
df['year'] = df['date'].dt.year
df['weekday'] = df['date'].dt.weekday
df['dayofyear'] = df['date'].dt.dayofyear
df['daynum'] = (df['date'] - df['date'].iloc[0]).dt.days
df['weeknum'] = df['daynum'] // 7
df['month'] = df['date'].dt.month

# Calculate days in each year
daysinyear = (
    df.groupby('year')['id'].count() / len(CFG.countries) / len(CFG.stores) / len(CFG.products)
).rename('daysinyear').astype(int).to_frame()
df = df.join(daysinyear, on='year', how='left')

# Create normalized yearly features
df['partofyear'] = (df['dayofyear'] - 1) / df['daysinyear']
df['partof2year'] = df['partofyear'] + df['year'] % 2

# Generate sinusoidal features for periodicity
for factor in [4, 3, 2]:
    df[f'sin {factor}t'] = np.sin(factor * 2 * np.pi * df['partofyear'])
    df[f'cos {factor}t'] = np.cos(factor * 2 * np.pi * df['partofyear'])

df['sin t'] = np.sin(2 * np.pi * df['partofyear'])
df['cos t'] = np.cos(2 * np.pi * df['partofyear'])
df['sin t/2'] = np.sin(np.pi * df['partof2year'])
df['cos t/2'] = np.cos(np.pi * df['partof2year'])

# Drop intermediate columns
df.drop(['daysinyear', 'partofyear', 'partof2year'], axis=1, inplace=True)


df.head()


import requests

# Function to fetch GDP per capita for a given country and year
def get_gdp_per_capita(country_code, year):
    url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/NY.GDP.PCAP.CD?date={year}&format=json"
    response = requests.get(url).json()
    try:
        return response[1][0]['value']
    except (KeyError, IndexError, TypeError):
        return None  # Return None if data is unavailable

# Fetch GDP data for all countries and years
gdp_data = {}
for country, code in CFG.country_iso_codes['alpha3'].items():
    gdp_data[country] = {
        year: get_gdp_per_capita(code, year) for year in CFG.years
    }

# Create a DataFrame for GDP data
gdp_df = pd.DataFrame(gdp_data).T  # Transpose to align countries as rows, years as columns
gdp_df.columns = CFG.years
gdp_df.index.name = 'country'

# Reshape gdp_df to align it with df
gdp_long = gdp_df.stack().reset_index()  # Reshape wide format to long
gdp_long.columns = ['country', 'year', 'gdp_factor']  # Rename columns for clarity
gdp_long['year'] = gdp_long['year'].astype(int)  # Ensure 'year' is an integer

# Ensure 'year' column exists in df
if 'year' not in df.columns:
    df['year'] = df['date'].dt.year  # Extract 'year' from 'date'

# Merge GDP data into df
df = pd.merge(
    df,
    gdp_long,
    how='left',
    on=['country', 'year']
)


df.head()


# Exclude rows where the country is 'Canada' or 'Kenya'
df_no_can_ken = df[~df['country'].isin(['Canada', 'Kenya'])]

# Calculate the mean 'num_sold' for each store
store_df = (
    df_no_can_ken
    .groupby('store')['num_sold']
    .mean()
    .rename('store_factor')
    .to_frame()
)

# Drop existing 'store_factor' column if it exists and merge the new one
df = (
    df.drop(columns=['store_factor'], errors='ignore')  # Remove old 'store_factor' if present
    .merge(store_df, on='store', how='left')            # Add the new 'store_factor'
)


from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

# Filter out rows for Canada and Kenya
df_no_can_ken = df[~df['country'].isin(['Canada', 'Kenya'])].copy()

# Calculate total daily sales
df_no_can_ken['num_sold_total'] = (
    df_no_can_ken.groupby('date')['num_sold'].transform('sum')
)

# Calculate sales ratio
df_no_can_ken['num_sold_ratio'] = (
    df_no_can_ken['num_sold'] / df_no_can_ken['num_sold_total']
)

# Initialize product_factor and plot
df['product_factor'] = None
plt.figure(figsize=(24, 6))

sinusoidal_columns = ['sin 4t', 'cos 4t', 'sin 3t', 'cos 3t', 'sin 2t', 'cos 2t', 'sin t', 'cos t', 'sin t/2', 'cos t/2']

for product in df_no_can_ken['product'].unique():
    product_data = df_no_can_ken[
        (df_no_can_ken['product'] == product) & (df_no_can_ken['test'] == 0)
    ].groupby('date')

    x = product_data[sinusoidal_columns].mean().to_numpy()
    y = product_data['num_sold_ratio'].sum().to_numpy()

    reg = Ridge()
    reg.fit(x, y)
    predictions = reg.predict(x)

    product_rows = df['product'] == product
    df.loc[product_rows, 'product_factor'] = reg.predict(
        df.loc[product_rows, sinusoidal_columns].to_numpy()
    )

    plt.plot(y, label=f'Actual ({product})', linestyle='--', alpha=0.7)
    plt.plot(predictions, label=f'Predicted ({product})')

plt.legend()
plt.title("Actual vs Predicted Sales Ratios per Product")
plt.show()



import holidays

countries_2l = {'Finland': 'FI', 'Canada': 'CA', 'Italy': 'IT', 'Kenya': 'KE', 'Singapore': 'SG', 'Norway': 'NO'}

# Add a holiday column based on country-specific holidays
df['holiday'] = 0
for country in CFG.countries:
    holiday_dates = [
        str(day) for day in holidays.CountryHoliday(countries_2l[country], years=CFG.years)
    ]
    df.loc[(df['country'] == country) & (df['date'].isin(holiday_dates)), 'holiday'] = 1

# Aggregate sales data by week, country, and weekday
weekly_sales = (
    df.groupby(['weeknum', 'country', 'weekday'])['num_sold']
    .sum()
    .reset_index()
    .pivot(index=['weeknum', 'country'], columns='weekday')
)

# Calculate sales ratio for each weekday
sales_ratio_per_weekday = weekly_sales.apply(lambda row: row / row.sum(), axis=1).reset_index()

# Calculate median weekday ratios for each country
ratio_weekday = pd.DataFrame(index=range(7), columns=CFG.countries)
for country in CFG.countries:
    for day in range(7):
        country_day_data = sales_ratio_per_weekday.loc[
            sales_ratio_per_weekday['country'] == country, ('num_sold', day)
        ][:-60]  # Exclude last 60 weeks (test set)
        ratio_weekday.loc[day, country] = country_day_data.median()

# Compute mean weekday ratio across countries
ratio_weekday['mean'] = ratio_weekday.mean(axis=1)

# Assign weekday factors to the dataset
df['weekday_factor'] = df['weekday'].map(ratio_weekday['mean'])

# Compute total ratio considering all factors
df['ratio'] = (
    df['gdp_factor'] *
    df['product_factor'] *
    df['store_factor'] *
    df['weekday_factor']
)

# Calculate total predicted sales
df['total'] = df['num_sold'] / df['ratio']


df.head()


import holidays
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Step 1: Identify holiday response periods
df_holidays = df.copy()
df_holidays['holiday_response'] = 0

for country in CFG.countries:
    holiday_dates = holidays.CountryHoliday(countries_2l[country], years=CFG.years)
    for holiday_date in holiday_dates:
        response_period = pd.date_range(holiday_date, periods=CFG.holiday_response_len)
        df_holidays.loc[
            (df_holidays['country'] == country) & (df_holidays['date'].isin(response_period)),
            'holiday_response'
        ] = 1

# Step 2: Calculate median sales excluding holidays
fig, ax = plt.subplots(figsize=(24, 6))
data = pd.DataFrame()

for country in CFG.countries:
    country_data = (
        df_holidays[
            (df_holidays['country'] == country) & (df_holidays['holiday_response'] == 0)
        ].groupby('dayofyear')['total'].median()
    )
    data[country] = country_data
    ax.plot(country_data, label=country)

# Compute the overall median across all countries
data['median'] = data.median(axis=1)

# Step 3: Fit a Fourier series to the median data
x = data.index.to_numpy()
y = data['median'].to_numpy()

# Fourier series basis functions
fourier = lambda t: np.array([np.sin(2 * np.pi / 365 * t), np.cos(2 * np.pi / 365 * t)])

# Fit ridge regression to the Fourier series
ridge_model = Ridge(alpha=0.01)
ridge_model.fit(fourier(x).T, y)
year_ratio = ridge_model.predict(fourier(np.arange(1, 366)).T)

# Extend the ratio to account for leap years
year_ratio = np.append(year_ratio, year_ratio[-1])

# Step 4: Map day-of-year factor to the dataset
df['dayofyear_factor'] = df['dayofyear'].map(dict(zip(np.arange(1, 367), year_ratio)))

# Step 5: Update total ratio and predicted total
df['ratio'] = (
    df['gdp_factor'] *
    df['product_factor'] *
    df['store_factor'] *
    df['weekday_factor'] *
    df['dayofyear_factor']
)

df['total'] = df['num_sold'] / df['ratio']

# Step 6: Visualize the results
ax.plot(year_ratio, 'k', linewidth=4, label='Seasonal Adjustment (Fourier)')
ax.legend()
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Define a color palette
colors = sns.color_palette("husl", len(CFG.countries))  # Using 'husl' for a distinct palette

# Initialize the figure
fig, ax = plt.subplots(figsize=(24, 6))

# DataFrame to store median sales per country
data = pd.DataFrame()

# Plot for each country
for n, (country, color) in enumerate(zip(CFG.countries, colors)):
    dt = (
        df_holidays[
            (df_holidays.test == 0) & 
            (df_holidays.country == country) & 
            (df_holidays.holiday_response == 0)
        ]
        .groupby(['date'])
        .total.median()
    )
    data[country] = dt
    
    # Plot with custom styles
    ax.plot(
        dt, 
        label=country, 
        color=color, 
        marker='o', 
        linestyle='-', 
        linewidth=2, 
        markersize=4, 
        alpha=0.8
    )

# Calculate and plot the overall median
data['median'] = data.median(axis=1)
ax.plot(
    data['median'], 
    label="Overall Median", 
    color="black", 
    linestyle='--', 
    linewidth=3, 
    alpha=0.9
)

# Add labels, title, and legend
ax.set_title("Median Daily Sales by Country (Excluding Holidays)", fontsize=18, weight='bold')
ax.set_xlabel("Date", fontsize=14)
ax.set_ylabel("Median Sales", fontsize=14)
ax.legend(title="Country", fontsize=12, title_fontsize=14, loc='upper left')
ax.grid(True, linestyle='--', alpha=0.6)

# Show the plot
plt.tight_layout()
plt.show()


from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

# Define sinusoidal columns
CFG.sincoscol2 = [
    'sin 4t', 'cos 4t', 
    'sin 3t', 'cos 3t', 
    'sin 2t', 'cos 2t', 
    'sin t', 'cos t', 
    'sin t/2', 'cos t/2'
]

# Prepare data for regression
dfsc = df[df.test == 0].groupby('date')[CFG.sincoscol2].mean()
dfsc['median'] = data['median']  # Add median sales to the grouped data

# Extract features (x) and target (y)
valid_data = ~pd.isna(dfsc['median'])  # Filter out rows with NaN values
x = dfsc.loc[valid_data, CFG.sincoscol2].to_numpy()
y = dfsc.loc[valid_data, 'median'].to_numpy()

# Perform linear regression using Ridge
reg = Ridge(alpha=0.01, fit_intercept=True)
reg.fit(x, y)

# Plot actual vs predicted values
fig, ax = plt.subplots(figsize=(24, 6))

# Plot actual median values
ax.plot(y, label='Actual Median', color='blue', linestyle='-', linewidth=2)

# Plot predicted median values
ax.plot(reg.predict(x), label='Predicted Median', color='orange', linestyle='--', linewidth=2)

# Customize the plot
ax.set_title("Actual vs Predicted Median Values Using Sinusoidal Features", fontsize=18, weight='bold')
ax.set_xlabel("Time Index", fontsize=14)
ax.set_ylabel("Median Sales", fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, linestyle='--', alpha=0.6)

# Add Ridge regression factors to the main DataFrame
df['sincos_factor'] = reg.intercept_ + (df[CFG.sincoscol2] * reg.coef_).sum(axis=1)

# Display the plot
plt.tight_layout()
plt.show()


# Calculate the total ratio and total sold items accounting for all factors
df['ratio'] = (
    df['gdp_factor'] *
    df['product_factor'] *
    df['store_factor'] *
    df['weekday_factor'] *
    df['sincos_factor']
)

df['total'] = df['num_sold'] / df['ratio']  # Calculate the adjusted total sales

# Plot the total sales for each country for the "Kaggle" product
fig, ax = plt.subplots(figsize=(24, 6))

# Iterate through each country to plot its total sales
for country in CFG.countries:
    # Filter data for the country and product "Kaggle"
    df_country = df[(df.country == country) & (df['product'] == 'Kaggle')]
    
    # Group by date and calculate total sales
    df_total_sales = df_country.groupby('date')['total'].sum().to_numpy()
    
    # Plot the data
    ax.plot(df_total_sales, label=country)

# Customize the plot
ax.set_title("Total Sales for 'Kaggle' Product by Country", fontsize=18, weight='bold')
ax.set_xlabel("Date (Index)", fontsize=14)
ax.set_ylabel("Total Sales (Adjusted)", fontsize=14)
ax.legend(title="Countries", fontsize=12)
ax.grid(True, linestyle='--', alpha=0.6)

# Improve layout and show the plot
plt.tight_layout()
plt.show()


# Calculate country-specific adjustment factor for the product "Kaggle"
country_factor = (
    df[df['product'] == 'Kaggle']
    .groupby('country')['total']
    .sum()
    .div(df[df['product'] == 'Kaggle'].groupby('country')['total'].sum().median())
    .rename('country_factor')
)

# Add the country factor to the main DataFrame
df = df.merge(country_factor, on='country', how='left')


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from scipy.optimize import minimize

# Ensure no NaN values in the ratio column
df['ratio'] = (
    df['gdp_factor']
* df['product_factor']
* df['store_factor']
* df['weekday_factor']
* df['sincos_factor']
* df['country_factor']
)

# Replace NaN ratios with a default value (e.g., 1.0) or interpolate
df['ratio'].fillna(1.0, inplace=True)  # Default to 1.0 if ratio is missing

# Filter the training data
train_data = df[df['test'] == 0]
train_data = train_data[train_data['num_sold'].notna()]  # Ensure no NaN in num_sold

# Function to calculate MAPE for a given scaling factor
def calculate_mape_for_factor(factor):
    """
    Calculate MAPE for a given constant factor.
    """
    predicted = factor * train_data['ratio']
    actual = train_data['num_sold']
    return mean_absolute_percentage_error(actual, predicted)

# Generate a range of constant factors (e.g., from 0.5 to 2.0)
factor_range = np.linspace(0.5, 2.0, 30)  # 30 values between 0.5 and 2.0
mape_values = []

# Calculate MAPE for each factor in the range
for factor in factor_range:
    mape = calculate_mape_for_factor(factor)
    mape_values.append(mape)

# Plot MAPE for each constant factor with detailed labels
plt.figure(figsize=(12, 6))
plt.plot(factor_range, mape_values, label="MAPE", color="purple", marker='o', linestyle='-', linewidth=2)

# Annotate each point with its corresponding MAPE value
for i, (factor, mape) in enumerate(zip(factor_range, mape_values)):
    plt.text(factor, mape, f"{mape:.4f}", ha='center', va='bottom', fontsize=8, color='black')

# Titles and labels
plt.title("MAPE vs Scaling Factor (Constant Factor)")
plt.xlabel("Scaling Factor")
plt.ylabel("MAPE")
plt.grid(True)

# Highlight the best scaling factor
best_factor = factor_range[np.argmin(mape_values)]
best_mape = min(mape_values)

plt.axvline(best_factor, color='red', linestyle='--', label=f"Optimized Factor: {best_factor:.4f}")
plt.axhline(best_mape, color='green', linestyle='--', label=f"Minimum MAPE: {best_mape:.4f}")

plt.legend()
plt.show()

# Print the optimized scaling factor and its corresponding MAPE
print(f"Optimized Scaling Factor: {best_factor:.4f}")
print(f"Minimum MAPE: {best_mape:.4f}")


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from scipy.optimize import minimize

# Ensure no NaN values in the ratio column
df['ratio'] = (
    df['gdp_factor']
* df['product_factor']
* df['store_factor']
* df['weekday_factor']
* df['sincos_factor']
* df['country_factor']
)

# Replace NaN ratios with a default value (e.g., 1.0) or interpolate
df['ratio'].fillna(1.0, inplace=True)  # Default to 1.0 if ratio is missing

# Filter the training data
train_data = df[df['test'] == 0]
train_data = train_data[train_data['num_sold'].notna()]  # Ensure no NaN in num_sold

# Function to calculate MAPE for a given scaling factor
def calculate_mape_for_factor(factor):
    """
    Calculate MAPE for a given constant factor.
    """
    predicted = factor * train_data['ratio']
    actual = train_data['num_sold']
    return mean_absolute_percentage_error(actual, predicted)

# Generate a range of constant factors (e.g., from 1.0 to 1.2)
factor_range_1_2 = np.linspace(1.0, 1.2, 30)  # 30 values between 1.0 and 1.2
mape_values_1_2 = []

# Calculate MAPE for each factor in the range [1.0, 1.2]
for factor in factor_range_1_2:
    mape = calculate_mape_for_factor(factor)
    mape_values_1_2.append(mape)

# Plot MAPE for each constant factor in the range [1.0, 1.2] with detailed labels
plt.figure(figsize=(12, 6))
plt.plot(factor_range_1_2, mape_values_1_2, label="MAPE (1.0 to 1.2)", color="blue", marker='o', linestyle='-', linewidth=2)

# Annotate each point with its corresponding MAPE value
for i, (factor, mape) in enumerate(zip(factor_range_1_2, mape_values_1_2)):
    plt.text(factor, mape, f"{mape:.4f}", ha='center', va='bottom', fontsize=8, color='black')

# Titles and labels
plt.title("MAPE vs Scaling Factor (1.0 to 1.2)")
plt.xlabel("Scaling Factor")
plt.ylabel("MAPE")
plt.grid(True)

# Highlight the best scaling factor (between 1.0 and 1.2)
best_factor_1_2 = factor_range_1_2[np.argmin(mape_values_1_2)]
best_mape_1_2 = min(mape_values_1_2)

plt.axvline(best_factor_1_2, color='red', linestyle='--', label=f"Optimized Factor: {best_factor_1_2:.4f}")
plt.axhline(best_mape_1_2, color='green', linestyle='--', label=f"Minimum MAPE: {best_mape_1_2:.4f}")

plt.legend()
plt.show()

# Print the optimized scaling factor and its corresponding MAPE
print(f"Optimized Scaling Factor (1.0 to 1.2): {best_factor_1_2:.4f}")
print(f"Minimum MAPE (1.0 to 1.2): {best_mape_1_2:.4f}")


from sklearn.metrics import mean_absolute_percentage_error

# Calculate the ratio by combining all contributing factors
df['ratio'] = (
    df['gdp_factor']
    * df['product_factor']
    * df['store_factor']
    * df['weekday_factor']
    * df['sincos_factor']
    * df['country_factor']
)

# Calculate total sales using the ratio
df['total'] = df['num_sold'] / df['ratio']

# Determine a constant factor for prediction (scaled slightly for adjustment)
const_factor = df['total'].median() * 1.0678

# Generate predictions based on the constant factor and ratio
df['prediction'] = const_factor * df['ratio']

# Calculate Mean Absolute Percentage Error (MAPE) for the training data
mape_train = mean_absolute_percentage_error(
    df.loc[(df['test'] == 0) & (df['num_sold'].notna()), 'num_sold'],
    df.loc[(df['test'] == 0) & (df['num_sold'].notna()), 'prediction']
)

print(f"MAPE for training data: {mape_train:.4f}")


df['prediction'] = np.round(df['prediction'].astype(float)).astype(int)


submission = df[(df.test == 1)][['id', 'prediction']].rename(columns={'prediction': 'num_sold'})
submission.to_csv('submission.csv', index=False)

submission.head()

