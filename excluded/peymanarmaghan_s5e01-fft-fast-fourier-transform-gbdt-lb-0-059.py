import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error
warnings.filterwarnings('ignore')



# Basic configurations
TARGET = 'num_sold'
RND_SEED = 42
# We will use GDP data from 2010 to 2020
GDP_YEARS = [str(y) for y in range(2010, 2021)]


# Load main train/test datasets and GDP data
train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv', parse_dates=['date'])
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv', parse_dates=['date'])
gdp_per_capita = pd.read_csv('/kaggle/input/world-gdpgdp-gdp-per-capita-and-annual-growths/gdp_per_capita.csv')


# Filter GDP data to match countries in the train set
gdp_filtered = gdp_per_capita[gdp_per_capita["Country Name"].isin(train["country"].unique())]

# Convert each country's GDP to a ratio of the total (across all countries)
gdp_ratios = gdp_filtered.set_index("Country Name")[GDP_YEARS].div(
    gdp_filtered.set_index("Country Name")[GDP_YEARS].sum(axis=0), 
    axis=1
)

# For each row in train/test, assign the GDP ratio based on country and year
train['gdp'] = train.apply(lambda x: gdp_ratios.loc[x.country, str(x.date.year)], axis=1)
test['gdp'] = test.apply(lambda x: gdp_ratios.loc[x.country, str(x.date.year)], axis=1)


#imputatios
target_mask = (
    (train["country"] == "Kenya") & 
    (train["store"] == "Discount Stickers") & 
    (train["product"] == "Holographic Goose")
)
source_mask = (
    (train["country"] == "Finland") & 
    (train["store"] == "Discount Stickers") & 
    (train["product"] == "Holographic Goose")
)

# Calculate a ratio of Kenya's total sales to Finland's, used to fill missing Kenya sales
kenya_sums = train.loc[train["country"] == "Kenya"].groupby("date")[TARGET].sum()
finland_sums = train.loc[train["country"] == "Finland"].groupby("date")[TARGET].sum()
kenya_finland_ratio = (kenya_sums / finland_sums).mean()

# Fill Kenya's missing data using Finland's data multiplied by the ratio
train.loc[target_mask, "num_sold"] = train.loc[target_mask, "num_sold"].where(
    train.loc[target_mask, "num_sold"].notna(),
    train.loc[source_mask, "num_sold"].values * kenya_finland_ratio
)

# Repeat process for Canada, but this time using Finland's values directly
target_mask = (
    (train["country"] == "Canada") & 
    (train["store"] == "Discount Stickers") & 
    (train["product"] == "Holographic Goose")
)
source_mask = (
    (train["country"] == "Finland") & 
    (train["store"] == "Discount Stickers") & 
    (train["product"] == "Holographic Goose")
)

train.loc[target_mask, "num_sold"] = train.loc[target_mask, "num_sold"].where(
    train.loc[target_mask, "num_sold"].notna(),
    train.loc[source_mask, "num_sold"].values
)

# Create a list of other masks where we want to forward-fill missing values
masks = [
    (train["country"] == "Canada") & 
    (train["store"] == "Premium Sticker Mart") & 
    (train["product"] == "Holographic Goose"),

    (train["country"] == "Canada") & 
    (train["store"] == "Stickers for Less") & 
    (train["product"] == "Holographic Goose"),

    (train["country"] == "Kenya") & 
    (train["store"] == "Premium Sticker Mart") & 
    (train["product"] == "Holographic Goose"),

    (train["country"] == "Kenya") & 
    (train["store"] == "Stickers for Less") & 
    (train["product"] == "Holographic Goose"),

    (train["country"] == "Kenya") & 
    (train["store"] == "Discount Stickers") & 
    (train["product"] == "Kerneler")
]

# Use backward fill (bfill) to fill missing values for those rows
for mask in masks:
    train.loc[mask, "num_sold"] = train.loc[mask, "num_sold"].fillna(method="bfill")

# Find rows still missing values
missing_rows = train.loc[train["num_sold"].isna()]

# Manually fill these rows based on known IDs
train.loc[train["id"] == 23719, "num_sold"] = 4
train.loc[train["id"] == 207003, "num_sold"] = 195


# FFT Functions (Core Component)
def compute_fft(group, target_col, extended_length, train_agg_size, is_train):
    """Compute FFT and extend signal using top frequencies"""
    y = group[target_col].values
    fft = np.fft.fft(y)
    magnitude = np.abs(fft)

    # Extract top 2 frequencies
    best_freqs = np.argsort(magnitude)[-2:]
    phases = np.angle(fft[best_freqs])

    # Reconstruct signal
    time_index = np.arange(extended_length)
    extended_signal = np.sum([
        np.cos(2 * np.pi * freq_idx/len(fft) * time_index + phase)
        for freq_idx, phase in zip(best_freqs, phases)
    ], axis=0) / len(best_freqs)

    return extended_signal[:train_agg_size] if is_train else extended_signal[train_agg_size:]

def apply_fft_processing(train_data, test_data, group_cols, target_col, date_col):
    """Main FFT processing pipeline"""
    # Calculate aggregation parameters
    train_agg_size = len(train_data) // 90
    test_agg_size = len(test_data) // 90
    extended_length = train_agg_size + test_agg_size

    # Group-wise FFT computation
    grouped = train_data.groupby(group_cols)

    # Train FFT features
    train_fft = grouped.apply(compute_fft, target_col, extended_length, train_agg_size, True)
    train_fft = train_fft.explode().reset_index(name='fft_feature')

    # Test FFT features
    test_fft = grouped.apply(compute_fft, target_col, extended_length, train_agg_size, False)
    test_fft = test_fft.explode().reset_index(name='fft_feature')

    train_num_repeats = len(train_fft) // len(train_data['date'].unique())
    test_num_repeats = len(test_fft) // len(test_data['date'].unique())
    train_date_list = np.tile(train_data['date'].unique(), train_num_repeats)
    test_date_list = np.tile(test_data['date'].unique(), test_num_repeats)

    # Assign repeated dates to the exploded DataFrame
    train_fft['date'] = train_date_list
    test_fft['date'] = test_date_list


    # Merge FFT results with original data
    train_data = pd.merge(train_data, train_fft, on=group_cols + ['date'], how='left')
    test_data = pd.merge(test_data, test_fft, on=group_cols + ['date'], how='left')
    train_data['fft_feature'] = train_data['fft_feature'].astype(float)
    test_data['fft_feature'] = test_data['fft_feature'].astype(float)

    return train_data, test_data



# --------------- Demonstration of FFT for specific groups ---------------

# We'll look at these groups to show how the FFT behaves
group_cols = ["country", "store", "product"]
demo_train,demo_test = apply_fft_processing(train, test, group_cols, TARGET, 'date')


demo_country = ['Kenya', 'Canada','Norway','Finland','Italy','Singapore']
demo_store = ['Discount Stickers']
demo_product = ['Kerneler', 'Holographic Goose']


num_plots = len(demo_country) * len(demo_store) * len(demo_product)
fig, axs = plt.subplots(num_plots, 1, figsize=(20, 25))

index = 0
for country in demo_country:
    for store in demo_store:
        for product in demo_product:
            # Filter the demonstration train data for plotting
            y = demo_train.loc[
                (demo_train['country'] == country) &
                (demo_train['store'] == store) &
                (demo_train['product'] == product),
                ['date', TARGET, 'fft_feature']
            ]

            # Plot the actual sales
            axs[index].plot(y['date'], y[TARGET], label=f'{country}_{store}_{product}')
            axs[index].set_title(f'{country}_{store}_{product}')
            axs[index].legend(loc="upper left")

            # Plot the FFT feature on a second axis
            ax_1 = axs[index].twinx()
            ax_1.plot(y['date'], y['fft_feature'], color='red', label='FFT')
            ax_1.legend(loc="upper right")

            index += 1

plt.tight_layout()
plt.show()


# --------------- Feature Engineering ---------------
def feature_eng(df, ts):
    df = df.copy()

    # Create date features
    df['day_of_year'] = df['date'].dt.dayofyear
    df['day_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year

    ts['day_of_year'] = ts['date'].dt.dayofyear
    ts['day_week'] = ts['date'].dt.dayofweek
    ts['month'] = ts['date'].dt.month
    ts['year'] = ts['date'].dt.year

    # Encode country and store with numbers
    le = LabelEncoder()
    df['country'] = le.fit_transform(df['country'])
    ts['country'] = le.transform(ts['country'])

    df['store'] = le.fit_transform(df['store'])
    ts['store'] = le.transform(ts['store'])

    # Calculate product popularity (product share of total daily sales)
    temp = train.groupby(['date','product'])[TARGET].sum() / train.groupby('date')[TARGET].sum()
    temp = temp.reset_index()

    product_map = {}
    for prd in train['product'].unique():
        product_map[prd] = temp[temp['product'] == prd][TARGET].mean()

    # Replace product column with the average share for that product
    df['product'] = df['product'].map(product_map)
    ts['product'] = ts['product'].map(product_map)

    # Drop ID columns
    df.drop(['id'], axis=1, inplace=True)
    ts.drop(['id'], axis=1, inplace=True)

    return df, ts


# --------------- Simple Modeling Class ---------------
class TSModel:
    """
    1) Holds a regression model class and parameters.
    2) Trains using time-series cross-validation.
    """
    def __init__(self, model_class, params):
        self.model_class = model_class
        self.params = params

    def train(self, X, n_splits=4):
        """
        1) Splits data into n parts (years).
        2) Uses each part in turn as validation, trains on the rest.
        3) Prints MAPE for each fold.
        4) Returns predictions and the trained model for each fold.
        """
        # Use time-based splitting
        tscv = TimeSeriesSplit(n_splits=n_splits)
        years = np.arange(2010, 2017)

        oof_preds = []
        validation_years = []
        model_fold = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(years)):
            # Split by year
            X_train = X.loc[X.year.isin(years[train_idx])]
            X_val = X.loc[X.year.isin(years[val_idx])]

            # Compute FFT features for each split
            group_cols = ["country", "store", "product"]
            X_train, X_val = apply_fft_processing(X_train, X_val, group_cols, TARGET, 'date')

            # Pick columns to use
            feature_cols = [col for col in X_train.columns if col not in [TARGET, 'date', 'gdp', 'store_weight']]

            # Separate target
            y_train = X_train[TARGET]
            y_val = X_val[TARGET]

            # Keep only the feature columns
            X_train = X_train[feature_cols]
            X_val = X_val[feature_cols]

            # Collect validation years
            validation_years.extend(years[val_idx])

            # Train the chosen model
            model = self.model_class(**self.params)
            model.fit(X_train, y_train)
            model_fold.append(model)

            # Validate on the fold
            pred = model.predict(X_val)
            oof_preds.extend(pred)

            # Compute mean absolute percentage error
            fold_score = mean_absolute_percentage_error(y_val, np.round(pred))

            print(f'\nfold no. {fold+1}:')
            print('training years:', years[train_idx])
            print('validation years:', years[val_idx])
            print(f"MAPE: {fold_score:.4f}")

        # Overall validation (all folds)
        overall_score = mean_absolute_percentage_error(
            X.loc[X.year.isin(validation_years), TARGET], 
            np.round(oof_preds)
        )
        print(f"\nOverall Validation MAPE: {overall_score:.4f}")
        return oof_preds, model_fold



train, test = feature_eng(train, test)

# Compute store weights (each store's share of total sales)
store_weight = train.groupby('store')[TARGET].sum() / train[TARGET].sum()
train['store_weight'] = train['store'].map(store_weight)
test['store_weight'] = test['store'].map(store_weight)

train[TARGET] = train[TARGET] / train['gdp'] / train['store_weight']


param_lgb = {
    'n_estimators': 500,
    'objective': 'mape',
    'learning_rate': 0.08,
    'max_depth': 5,
    'verbose': -1,
    'random_state': RND_SEED
}

# Create and train the model (with cross-validation)
model = TSModel(lgb.LGBMRegressor, param_lgb)
oof_lgb, model_lgb = model.train(train, n_splits=4)


# After we finish training, apply FFT processing to the full train and test for final predictions
train_agg_size = int(len(train) / 90)
test_agg_size = int(len(test) / 90)
extended_length = train_agg_size + test_agg_size

group_cols = ["country", "store", "product"]
train, test = apply_fft_processing(train, test, group_cols, TARGET, 'date')
feature_cols = [col for col in train.columns if col not in [TARGET, 'date', 'gdp', 'store_weight']]

# Use all fold models to make predictions on the test set
pred = 0
for i in range(len(model_lgb)):
    pred += model_lgb[i].predict(test[feature_cols]) / len(model_lgb)

# Undo the earlier normalization by multiplying by GDP and store weight
pred = pred * test['gdp'] * test['store_weight']*1.06
pred = np.round(pred)  # Round to nearest integer

# Make submission file
submission = pd.read_csv("/kaggle/input/playground-series-s5e1/sample_submission.csv")
submission["num_sold"] = np.round(pred)
display(submission.head(5))
submission.to_csv("submission.csv", index=False)

