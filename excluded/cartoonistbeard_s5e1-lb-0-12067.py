%%time

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
sns.set_style('whitegrid')


%%time

train_df = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv",parse_dates=['date'])
test_df = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv", parse_dates=['date'])


train_df.head()


%%time

print("start training date == ",train_df['date'].min())
print("start testing date == ", test_df['date'].min())

print("last training date == ",train_df['date'].max())
print("last testing date == ",test_df['date'].max())


%%time

gdp_per_capita = pd.read_csv("/kaggle/input/world-gdpgdp-gdp-per-capita-and-annual-growths/gdp_per_capita.csv")
gdp_per_capita.head()


def plot_value_counts(df,col):
    val_df = df[col].value_counts().reset_index()
    val_df.columns = [col,"values"]
    plt.figure(figsize=(10,8))
    ax = sns.barplot(x=col,y="values",data = val_df)
    for container in ax.containers:
        ax.bar_label(container,fmt="%d",label_type="edge")
    ax.set_title(f"Value count of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("values")
    return ax


plot_value_counts(train_df,'country')


plot_value_counts(train_df,'store')


plot_value_counts(train_df,"product")


cnt_df = train_df.groupby(["country","product",'store'])['id'].count().reset_index()
cnt_df


counts = train_df.groupby(["country","store","product"])["num_sold"].count().rename("num_rows")
missing_data = counts.loc[counts != 2557]
missing_data_df = missing_data.reset_index()
missing_data_df["num_missing_rows"] = 2557 - missing_data_df["num_rows"]
missing_data_df


f, axs = plt.subplots(9, 1, figsize=(20, 50))
# Loop through the unique combinations of country, store, and product
for i, (country, store, product) in enumerate(missing_data.index):
    # Filter the DataFrame for the specific combination
    plot_df = train_df.loc[
        (train_df["country"] == country) &
        (train_df["store"] == store) &
        (train_df["product"] == product)
    ]
    
    # Identify rows with missing "num_sold" values
    missing_vals = plot_df.loc[plot_df["num_sold"].isna()]
    
    # Plot the valid "num_sold" data (excluding NaN)
    sns.lineplot(data=plot_df.dropna(subset=["num_sold"]), x="date", y="num_sold", ax=axs[i])
    
    # Highlight the missing dates with vertical lines
    for missing_date in missing_vals["date"]:
        axs[i].axvline(missing_date, color="red", linestyle="--", linewidth=1, alpha=0.8)
    
    # Add titles and labels
    axs[i].set_title(f"{country} - {store} - {product}")
    axs[i].set_xlabel("Date")
    axs[i].set_ylabel("Number Sold")

plt.tight_layout()
plt.show()


weekly_df = train_df.groupby(["country","product","store", pd.Grouper(key="date",freq="W")])["num_sold"].sum().reset_index()
monthly_df = train_df.groupby(["country","product",'store',pd.Grouper(key="date",freq="M")])["num_sold"].sum().reset_index()


def plot_all(df):
    f,axes = plt.subplots(3,2,figsize=(25,25), sharex = True, sharey=True)
    f.tight_layout()
    for n,prod in enumerate(df["product"].unique()):
        plot_df = df.loc[df["product"] == prod]
        sns.lineplot(data=plot_df, x="date", y="num_sold", hue="country", style="store",ax=axes[n//2,n%2])
        axes[n//2,n%2].set_title("Product: "+str(prod))


plot_all(weekly_df)


plot_all(monthly_df)


country_weights = train_df.groupby('country')["num_sold"].sum()/train_df["num_sold"].sum()
country_ratio = (train_df.groupby(['date','country'])["num_sold"].sum()/train_df.groupby("date")["num_sold"].sum()).reset_index()
f,ax = plt.subplots(figsize=(20,10))
sns.lineplot(x="date",y="num_sold",hue="country",data=country_ratio)
ax.set_title("Contribution of tatoal sales of each country over time")


#gdp_per_capita.head()
train_countries = list(train_df['country'].unique())
req_years = ["Country Name","2010","2011","2012","2013","2014","2015","2016","2017","2018","2019","2020"]
req_gdp = gdp_per_capita.loc[gdp_per_capita['Country Name'].isin(train_countries)][req_years]
gdp_melted = req_gdp.melt(id_vars="Country Name", var_name="Year", value_name="Sales")
gdp_melted["Year"] = gdp_melted["Year"].astype(int)

plt.figure(figsize=(12, 6))
for country in train_countries:
    country_data = gdp_melted[gdp_melted["Country Name"] == country]
    sns.lineplot(x=country_data["Year"], y=country_data["Sales"], hue=country_data["Country Name"])

plt.xlabel("Year")
plt.ylabel("Sales")
plt.title("Time Series of Sales for Each Country")
plt.legend(title="Country")
plt.grid()
plt.show()


# Assuming train_df and gdp_per_capita DataFrames are already defined
# Prepare data for Plot 1 (restricting dates to 2010–2020)
country_weights = train_df.groupby('country')["num_sold"].sum() / train_df["num_sold"].sum()
country_ratio = (
    train_df.groupby(['date', 'country'])["num_sold"].sum() /
    train_df.groupby("date")["num_sold"].sum()
).reset_index()
country_ratio['year'] = pd.to_datetime(country_ratio['date']).dt.year
country_ratio = country_ratio[(country_ratio['year'] >= 2010) & (country_ratio['year'] <= 2020)]

# Prepare data for Plot 2 (restricting years to 2010–2020)
train_countries = list(train_df['country'].unique())
req_years = ["Country Name", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"]
req_gdp = gdp_per_capita.loc[gdp_per_capita['Country Name'].isin(train_countries)][req_years]
gdp_melted = req_gdp.melt(id_vars="Country Name", var_name="Year", value_name="Sales")
gdp_melted["Year"] = gdp_melted["Year"].astype(int)
gdp_melted = gdp_melted[(gdp_melted["Year"] >= 2010) & (gdp_melted["Year"] <= 2020)]

# Normalize values for better comparison
country_ratio["normalized_num_sold"] = country_ratio["num_sold"] / country_ratio["num_sold"].max()
gdp_melted["normalized_sales"] = gdp_melted["Sales"] / gdp_melted["Sales"].max()

# Create subplots
fig, axes = plt.subplots(2, 1, figsize=(16, 12), sharex=True)

# Plot sales contributions
sns.lineplot(
    x="year", y="normalized_num_sold", hue="country", 
    data=country_ratio, linewidth=2, ax=axes[0], alpha=0.7
)
axes[0].set_title("Normalized Sales Contributions Over Time")
axes[0].set_ylabel("Normalized Sales Contributions")
axes[0].legend(title="Country")

# Plot GDP trends
for country in train_countries:
    country_data = gdp_melted[gdp_melted["Country Name"] == country]
    axes[1].plot(
        country_data["Year"], country_data["normalized_sales"], 
        linestyle="--", marker="o", label=f"{country}"
    )
axes[1].set_title("Normalized GDP Trends Over Time")
axes[1].set_ylabel("Normalized GDP")
axes[1].legend(title="Country")
axes[1].grid()

# Common x-axis
axes[1].set_xlabel("Year (2010–2020)")

plt.tight_layout()
plt.show()


store_ratio = (train_df.groupby(['date','store'])["num_sold"].sum()/train_df.groupby('date')["num_sold"].sum()).reset_index()
ax = sns.lineplot(x="date",y="num_sold",data = store_ratio, hue="store")
ax.set_xlabel("Date")
ax.set_ylabel("Ratio Of Sales")
ax.set_title("Distribution of sales across all stores over the year")


product_ratio = (train_df.groupby(['date','product'])["num_sold"].sum()/train_df.groupby("date")["num_sold"].sum()).reset_index()
ax = sns.lineplot(data=product_ratio,x="date",y="num_sold",hue="product")
ax.set_xlabel("dates")
ax.set_ylabel("Propotion of sales")
ax.set_title("Propotion of sales of each product across all years")


%%time

years = []

for i in range(2010,2021):
    years.append(str(i))

filtered_gdp_df = gdp_per_capita.loc[gdp_per_capita['Country Name'].isin(train_df['country'].unique()),['Country Name']+ years].set_index('Country Name')
filtered_gdp_df.head()


for year in years:
    filtered_gdp_df[f'{year}_ratio'] = filtered_gdp_df[year]/filtered_gdp_df.sum()[year]

filtered_gdp_df.head()


%%time

gdp_ratio = filtered_gdp_df[[f"{i}_ratio" for i in years]]
gdp_ratio.columns = [int(i) for i in years]
gdp_ratio = gdp_ratio.unstack().reset_index().rename(
    columns={"level_0": "year", 0: "ratio", "Country Name": "country"}
)
gdp_ratio['year'] = pd.to_datetime(gdp_ratio['year'],format="%Y")
gdp_ratio.head()


%%time

gdp_ratio['year'] = gdp_ratio['year'].dt.year
train_df_imputed = train_df.copy()
train_df_imputed['year'] = train_df_imputed['date'].dt.year
print("number of missing values == ", train_df_imputed['num_sold'].isna().sum())


%%time

for year in train_df_imputed['year'].unique():
    
    target_ratio = gdp_ratio.loc[
                (gdp_ratio["year"] == year) & 
                (gdp_ratio["country"] == "Norway"), 
                "ratio"
                ].values[0]
    
    current_raito = gdp_ratio.loc[
                (gdp_ratio["year"] == year) & 
                (gdp_ratio["country"] == "Canada"), 
                "ratio"
                ].values[0]
    
    ratio_can = current_raito / target_ratio
    
    train_df_imputed.loc[
    (train_df_imputed["country"] == "Canada") & 
    (train_df_imputed["store"] == "Discount Stickers") & 
    (train_df_imputed["product"] == "Holographic Goose") & 
    (train_df_imputed["year"] == year), 
    "num_sold"
    ] = (
    train_df_imputed.loc[
        (train_df_imputed["country"] == "Norway") & 
        (train_df_imputed["store"] == "Discount Stickers") & 
        (train_df_imputed["product"] == "Holographic Goose") & 
        (train_df_imputed["year"] == year), 
        "num_sold"
    ] * ratio_can
    ).values
    
    current_ts = train_df_imputed.loc[
    (train_df_imputed["country"] == "Canada") & 
    (train_df_imputed["store"] == "Premium Sticker Mart") & 
    (train_df_imputed["product"] == "Holographic Goose") & 
    (train_df_imputed["year"] == year)
    ]
    
    missing_ts_dates = current_ts.loc[current_ts["num_sold"].isna(), "date"]
    
    train_df_imputed.loc[
    (train_df_imputed["country"] == "Canada") & 
    (train_df_imputed["store"] == "Premium Sticker Mart") & 
    (train_df_imputed["product"] == "Holographic Goose") & 
    (train_df_imputed["year"] == year) & 
    (train_df_imputed["date"].isin(missing_ts_dates)), 
    "num_sold"
    ] = (
    train_df_imputed.loc[
        (train_df_imputed["country"] == "Norway") & 
        (train_df_imputed["store"] == "Premium Sticker Mart") & 
        (train_df_imputed["product"] == "Holographic Goose") & 
        (train_df_imputed["year"] == year) & 
        (train_df_imputed["date"].isin(missing_ts_dates)), 
        "num_sold"
    ] * ratio_can
    ).values
    
    current_ts = train_df_imputed.loc[
    (train_df_imputed["country"] == "Canada") & 
    (train_df_imputed["store"] == "Stickers for Less") & 
    (train_df_imputed["product"] == "Holographic Goose") & 
    (train_df_imputed["year"] == year)
    ]
    
    missing_ts_dates = current_ts.loc[current_ts["num_sold"].isna(), "date"]
    
    train_df_imputed.loc[
    (train_df_imputed["country"] == "Canada") & 
    (train_df_imputed["store"] == "Stickers for Less") & 
    (train_df_imputed["product"] == "Holographic Goose") & 
    (train_df_imputed["year"] == year) & 
    (train_df_imputed["date"].isin(missing_ts_dates)), 
    "num_sold"
    ] = (
    train_df_imputed.loc[
        (train_df_imputed["country"] == "Norway") & 
        (train_df_imputed["store"] == "Stickers for Less") & 
        (train_df_imputed["product"] == "Holographic Goose") & 
        (train_df_imputed["year"] == year) & 
        (train_df_imputed["date"].isin(missing_ts_dates)), 
        "num_sold"
    ] * ratio_can
    ).values
    
    current_raito = gdp_ratio.loc[
    (gdp_ratio["year"] == year) & 
    (gdp_ratio["country"] == "Kenya"), 
    "ratio"
    ].values[0]
    
    ratio_ken = current_raito / target_ratio
    
    train_df_imputed.loc[
    (train_df_imputed["country"] == "Kenya") & 
    (train_df_imputed["store"] == "Discount Stickers") & 
    (train_df_imputed["product"] == "Holographic Goose") & 
    (train_df_imputed["year"] == year), 
    "num_sold"
    ] = (
    train_df_imputed.loc[
        (train_df_imputed["country"] == "Norway") & 
        (train_df_imputed["store"] == "Discount Stickers") & 
        (train_df_imputed["product"] == "Holographic Goose") & 
        (train_df_imputed["year"] == year), 
        "num_sold"
    ] * ratio_ken - 0.0007
    ).values
    
    current_ts = train_df_imputed.loc[
    (train_df_imputed["country"] == "Kenya") &
    (train_df_imputed["store"] == "Premium Sticker Mart") &
    (train_df_imputed["product"] == "Holographic Goose") &
    (train_df_imputed["year"] == year)
    ]
    
    missing_ts_dates = current_ts.loc[current_ts["num_sold"].isna(), "date"]
    
    train_df_imputed.loc[
    (train_df_imputed["country"] == "Kenya") &
    (train_df_imputed["store"] == "Premium Sticker Mart") &
    (train_df_imputed["product"] == "Holographic Goose") &
    (train_df_imputed["year"] == year) &
    (train_df_imputed["date"].isin(missing_ts_dates)),
    "num_sold"
    ] = (
    train_df_imputed.loc[
        (train_df_imputed["country"] == "Norway") &
        (train_df_imputed["store"] == "Premium Sticker Mart") &
        (train_df_imputed["product"] == "Holographic Goose") &
        (train_df_imputed["year"] == year) &
        (train_df_imputed["date"].isin(missing_ts_dates)),
        "num_sold"
    ] * ratio_ken - 0.0007
    ).values
    
    current_ts = train_df_imputed.loc[
    (train_df_imputed["country"] == "Kenya") &
    (train_df_imputed["store"] == "Stickers for Less") &
    (train_df_imputed["product"] == "Holographic Goose") &
    (train_df_imputed["year"] == year)
    ]
    
    missing_ts_dates = current_ts.loc[current_ts["num_sold"].isna(), "date"]
    
    train_df_imputed.loc[
    (train_df_imputed["country"] == "Kenya") &
    (train_df_imputed["store"] == "Stickers for Less") &
    (train_df_imputed["product"] == "Holographic Goose") &
    (train_df_imputed["year"] == year) &
    (train_df_imputed["date"].isin(missing_ts_dates)),
    "num_sold"
    ] = (
    train_df_imputed.loc[
        (train_df_imputed["country"] == "Norway") &
        (train_df_imputed["store"] == "Stickers for Less") &
        (train_df_imputed["product"] == "Holographic Goose") &
        (train_df_imputed["year"] == year) &
        (train_df_imputed["date"].isin(missing_ts_dates)),
        "num_sold"
    ] * ratio_ken-0.0007
    ).values
    
    current_ts = train_df_imputed.loc[
    (train_df_imputed["country"] == "Kenya") &
    (train_df_imputed["store"] == "Discount Stickers") &
    (train_df_imputed["product"] == "Kerneler") &
    (train_df_imputed["year"] == year)
    ]
    
    missing_ts_dates = current_ts.loc[current_ts["num_sold"].isna(), "date"]
    
    train_df_imputed.loc[
    (train_df_imputed["country"] == "Kenya") &
    (train_df_imputed["store"] == "Discount Stickers") &
    (train_df_imputed["product"] == "Kerneler") &
    (train_df_imputed["year"] == year) &
    (train_df_imputed["date"].isin(missing_ts_dates)),
    "num_sold"
    ] = (
    train_df_imputed.loc[
        (train_df_imputed["country"] == "Norway") &
        (train_df_imputed["store"] == "Discount Stickers") &
        (train_df_imputed["product"] == "Kerneler") &
        (train_df_imputed["year"] == year) &
        (train_df_imputed["date"].isin(missing_ts_dates)),
        "num_sold"
    ] * ratio_ken - 0.0007
    ).values

print("The missing values are",train_df_imputed['num_sold'].isna().sum())


train_df_imputed.loc[train_df_imputed["id"] == 23719, "num_sold"] = 4
train_df_imputed.loc[train_df_imputed["id"] == 207003, "num_sold"] = 195

print(f"Missing values remaining: {train_df_imputed['num_sold'].isna().sum()}")


train_df_imputed.head()


train_df_org = train_df_imputed.copy()
data = train_df_imputed.groupby(['date'])["num_sold"].sum().reset_index()
ax = sns.lineplot(x="date",y="num_sold",data=data)
ax.set_xlabel("Dates")
ax.set_ylabel("Sales")
ax.set_title("Data to be Forecasted")


weekly_df_imp = train_df_imputed.groupby([pd.Grouper(key ="date",freq="W")])["num_sold"].sum().reset_index()
monthly_df_imp = train_df_imputed.groupby([pd.Grouper(key="date", freq="M")])["num_sold"].sum().reset_index()


ax = sns.lineplot(x="date",y="num_sold",data=weekly_df_imp)
ax.set_title("Weekly Sales")


ax= sns.lineplot(x="date",y="num_sold",data=monthly_df_imp)
ax.set_title("Monthly Sales")


def plot_seasonality(df):
    """
    Plot seasonality trends for monthly, day-of-week, and day-of-year from a dataframe.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'date' (datetime) and 'num_sold' (numeric) columns.
    """
    # Ensure the 'date' column is in datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Extract temporal features
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek  # Monday=0, Sunday=6
    df['day_of_year'] = df['date'].dt.dayofyear

    # Create a figure with multiple subplots
    fig, axes = plt.subplots(3, 1, figsize=(15, 18))

    # Plot monthly seasonality
    sns.lineplot(data=df.groupby('month')['num_sold'].mean().reset_index(), 
                 x='month', y='num_sold', ax=axes[0], marker="o", linewidth=2)
    axes[0].set_title('Monthly Seasonality')
    axes[0].set_xlabel('Month')
    axes[0].set_ylabel('Average Sales')
    axes[0].set_xticks(range(1, 13))
    axes[0].grid()

    # Plot day-of-week seasonality
    sns.lineplot(data=df.groupby('day_of_week')['num_sold'].mean().reset_index(),
                 x='day_of_week', y='num_sold', ax=axes[1], marker="o", linewidth=2)
    axes[1].set_title('Day-of-Week Seasonality')
    axes[1].set_xlabel('Day of the Week')
    axes[1].set_ylabel('Average Sales')
    axes[1].set_xticks(range(7))
    axes[1].set_xticklabels(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    axes[1].grid()

    # Plot day-of-year seasonality
    sns.lineplot(data=df.groupby('day_of_year')['num_sold'].mean().reset_index(),
                 x='day_of_year', y='num_sold', ax=axes[2], marker="", linewidth=1)
    axes[2].set_title('Day-of-Year Seasonality')
    axes[2].set_xlabel('Day of the Year')
    axes[2].set_ylabel('Average Sales')
    axes[2].grid()

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()

# Example Usage
# Assuming you have a DataFrame 'df' with 'date' and 'num_sold' columns
# plot_seasonality(df)


plot_seasonality(train_df_imputed)


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from xgboost import DMatrix, XGBRegressor
from catboost import Pool, CatBoostRegressor
from lightgbm import LGBMRegressor, early_stopping


train_df_imputed.drop(columns=['id'],inplace=True)
test_df.drop(columns=['id'],inplace=True)


train_df_imputed.drop(columns=['year','month','day_of_week','day_of_year'],inplace=True)
train_df_imputed.head()


test_df.head()


from sklearn.preprocessing import StandardScaler

def normalize_features(df, cols):
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df


import numpy as np
import pandas as pd

def normalize_features(df, cols):
    """Normalize specified columns in the dataframe."""
    for col in cols:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df

def Feature_eng(train, test):
    # Copy datasets
    train = train.copy()
    test = test.copy()

    # Date Feature Engineering
    for df in [train, test]:
        df['day'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['sine_day'] = np.sin(2 * np.pi * df['day'] / 31)
        df['cos_day'] = np.cos(2 * np.pi * df['day'] / 31)
        df['sine_month'] = np.sin(2 * np.pi * df['month'] / 12)
        df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
        df['sine_year'] = np.sin(2 * np.pi * df['year'])
        #df['cos_year'] = np.cos(2 * np.pi * df['year'])
        df.drop(columns=['date'], inplace=True)

    # Normalize sine and cosine features
    sine_cos_cols = ['sine_day', 'cos_day', 'sine_month', 'cos_month', 'sine_year']
    train = normalize_features(train, sine_cos_cols)
    test = normalize_features(test, sine_cos_cols)

    # Encode categorical features using one-hot encoding
    categorical_cols = ['country', 'store', 'product']
    train = pd.get_dummies(train, columns=categorical_cols, drop_first=False)
    test = pd.get_dummies(test, columns=categorical_cols, drop_first=False)

    # Align test dataset with train dataset
    test = test.reindex(columns=train.columns, fill_value=0)

    return train, test


%%time

train_encoded,test_encoded = Feature_eng(train_df_imputed,test_df)


test_encoded.drop(columns=['num_sold'],inplace=True)


display(train_encoded.head(2))
print(train_encoded.shape)
display(test_encoded.head(2))
print(test_encoded.shape)


train_encoded["num_sold"].plot(kind="hist")


from sklearn.metrics import mean_squared_error

def train_catboost_kfold(X, y, test, n_splits=5, random_state=42, verbose=0, early_stopping_rounds=200):
    """
    Train a CatBoost model using K-Fold cross-validation.

    Parameters:
        X (np.ndarray or pd.DataFrame): Feature matrix for training.
        y (np.ndarray or pd.Series): Target values.
        test (np.ndarray or pd.DataFrame): Test feature matrix.
        n_splits (int): Number of K-Fold splits.
        random_state (int): Random state for reproducibility.
        verbose (int): Verbosity level for CatBoost.
        early_stopping_rounds (int): Number of rounds for early stopping.

    Returns:
        tuple: (mean RMSE, test predictions averaged across folds, OOF predictions)
    """
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    print("=" * 80)
    print("Training CatBoost Model with K-Fold Cross-Validation")
    print("=" * 80 + "\n")

    oof_preds = np.zeros(len(y))
    test_preds = np.zeros(len(test))
    rmse_scores = []

    for fold, (train_idx, valid_idx) in enumerate(kfold.split(X)):
        print(f"Fold {fold + 1}/{n_splits}")
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y[valid_idx]

        # Prepare Pool objects for CatBoost
        train_pool = Pool(X_train, y_train)
        valid_pool = Pool(X_valid, y_valid)
        test_pool = Pool(test)

        # Initialize and train the CatBoost model
        model = CatBoostRegressor(
            iterations=2000,
            learning_rate=0.05,
            depth=6,
            eval_metric='RMSE',
            random_seed=random_state,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose
        )
        model.fit(train_pool, eval_set=valid_pool)

        # Generate predictions
        oof_preds[valid_idx] = model.predict(valid_pool)
        test_preds += model.predict(test_pool) / n_splits

        # Calculate RMSE for the current fold
        fold_rmse = mean_squared_error(y_valid, oof_preds[valid_idx], squared=False)
        print(f"Fold {fold + 1} RMSE: {fold_rmse:.4f}\n")
        rmse_scores.append(fold_rmse)

    # Print overall RMSE
    mean_rmse = np.mean(rmse_scores)
    std_rmse = np.std(rmse_scores)
    print(f"Mean RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}\n")
    print("=" * 80)

    return mean_rmse, test_preds, oof_preds


%%time

target = 'num_sold'
X = train_encoded.copy()
y = X.pop(target)


%%time

# Example Usage
mean_rmse, test_preds, oof_preds = train_catboost_kfold(
    X, y, test_encoded, n_splits=5, random_state=42, verbose=100
)


train_encoded['oof'] = oof_preds
test_encoded['oof'] = test_preds


display(train_encoded.head(2))
display(test_encoded.head(2))


def model_trainer(model, X, y, test, n_splits=5, random_state=42, verbose=0, model_name=None):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    print("="*80)
    model_name_ = model[-1].__class__.__name__ if isinstance(model, Pipeline) else model.__class__.__name__
    print(f"Model: {model_name_}")
    print("="*80 + '\n')

    oof_mape = []
    oof_test_preds = np.zeros(len(test))
    oof_train_preds = np.zeros(len(y))
    
    for fold, (train_idx, valid_idx) in enumerate(kfold.split(X)):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y[valid_idx]

        if model_name == 'xgb':
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=verbose)
            booster = model.get_booster()
            y_pred = booster.predict(DMatrix(X_valid), iteration_range=(0, model.best_iteration+1))
            test_pred = booster.predict(DMatrix(test), iteration_range=(0, model.best_iteration+1))
            oof_train_preds[train_idx] = booster.predict(DMatrix(X_train), iteration_range=(0, model.best_iteration+1))

        elif model_name == 'cat':
            trainPool = Pool(X_train ,y_train)
            testPool = Pool(test)
            validPool = Pool(X_valid, y_valid)

            model.fit(X=trainPool, eval_set=validPool, verbose=verbose, early_stopping_rounds=200)
            y_pred = model.predict(validPool)
            test_pred = model.predict(testPool)
            oof_train_preds[train_idx] = model.predict(Pool(X_train))

        elif model_name == 'lgb':
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric='rmse', callbacks=[early_stopping(200, verbose=0)])
            y_pred = model.predict(X_valid, num_iteration=model.best_iteration_)
            test_pred = model.predict(test, num_iteration=model.best_iteration_)
            oof_train_preds[train_idx] = model.predict(X_train, num_iteration=model.best_iteration_)

        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_valid)
            test_pred = model.predict(test)
            oof_train_preds[train_idx] = model.predict(X_train)

        oof_test_preds += test_pred
        mape = mean_absolute_percentage_error(np.expm1(y_valid), np.expm1(y_pred))
        print(f"Fold {fold+1} --> MAPE: {mape:.4f}")
        oof_mape.append(mape)
    
    print()
    print(f"Average Fold MAPE: {np.mean(oof_mape):.4f} \xb1 {np.std(oof_mape):.4f}")
    return oof_test_preds/n_splits, oof_train_preds


test_preds, train_preds = pd.DataFrame(), pd.DataFrame()
X_train = train_encoded.copy()
y_train = np.log1p(X_train.pop("num_sold"))
X_test  = test_encoded.copy()


display(X_train.head(2))
display(X_test.head(2))


%%time

xgb_params = {
    'n_estimators': 3000,
    'learning_rate': 0.00990161328639894,
    'max_depth': 17,
    'min_child_weight': 58,
    'subsample': 0.7373527286687829,
    'colsample_bytree': 0.4544157822113165,
    'gamma': 0.0019767061497068528,
    'reg_alpha': 0.7647218923252306,
    'device': 'cuda',
    'tree_method': 'hist',
    'random_state': 0,
    'early_stopping_rounds': 200
}

xgb_reg = XGBRegressor(**xgb_params)

test_preds['xgb'], train_preds['xgb'] = model_trainer(xgb_reg, X_train, y_train, 
                                                      X_test, 
                                                      random_state=0, verbose=0, model_name='xgb')
# with catbooost encoder ----> 0.5
# with OHE ---> 0.12067


%%time

cat_params = {
    'n_estimators': 10000,
    'learning_rate': 0.05, 
    'task_type': 'GPU', 
    'verbose': False, 
    'allow_writing_files': False,
}

cat_reg = CatBoostRegressor(**cat_params)

test_preds['cat'], train_preds['cat'] = model_trainer(
    cat_reg,
    X_train, y_train, X_test, random_state=0, model_name='cat'
)

# with catboost encoder = 1.16
# with OHE = 0.12616


%%time


lgb_reg = LGBMRegressor(verbosity=-1, device='gpu',
                        n_estimators=5000, learning_rate=0.1
                       )



test_preds['lgb'], train_preds['lgb'] = model_trainer(
    lgb_reg,
    X_train, y_train, X_test, random_state=42, model_name='lgb'
)

#with catboost encoder ---> 1.01
#with OHE ----> 0.12574


test_pred, _ = model_trainer(
    XGBRegressor(
        device='cuda', 
        tree_method='hist', 
        n_estimators=1000,
        learning_rate=0.01,
        early_stopping_rounds=200
    ),
    test_preds,
    y_train,
    test_preds,
    model_name='xgb'
)

#with catboost encoder ---> 
#with OHE ---> 0.15957
#with OHE+OOF--> 0.15234


sub_df = pd.read_csv("/kaggle/input/playground-series-s5e1/sample_submission.csv")
sub_df['num_sold'] = np.expm1(test_pred)
sub_df.head()


sub_df.to_csv("PS5E1ENSBV2.csv",index=False)

