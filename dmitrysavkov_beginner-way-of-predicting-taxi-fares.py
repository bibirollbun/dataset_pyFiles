# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
import holidays

# ML libraries
from sklearn import preprocessing, metrics, model_selection, ensemble
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRFRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

# Configure visualization settings
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")


# Constants and configuration
DATA_PATH = '/kaggle/input/'
DATASET_PATH = f"{DATA_PATH}reduced-taxi-drive-data" 
COMPETITION_PATH = f"{DATA_PATH}new-york-city-taxi-fare-prediction"

USING_TIME_SERIES_DATA = True # instead of the entire 55M rows dataset
LOAD_ALL_DATA = False
target_column = "fare_amount"

# Specific version of data to use in the current run
TRAIN_FILENAME = "train_1013043"
VAL_FILENAME = "val_19999"


# Some helper functions.

def split_df(df, train_size=0.8):
    train_data, val_data = model_selection.train_test_split(df, train_size=train_size, random_state=42)
    return train_data, val_data

def sep_target(x: pd.DataFrame):
    return [x.drop(target_column, axis=1), x[target_column]]

def get_current_time():
    return datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d_%H-%M-%S")

# Remove key field (irrelevant for modelling)
def drop_key(dfs):
    for df in dfs:
        if "key" in df: df.drop("key", axis=1, inplace=True)


def load_data_init(filename, feather=False, path=COMPETITION_PATH):
    """
    Load one file with support for both CSV and Feather formats.
    Feather provides faster loading for repeated access.
    """
    path = path + f"/{filename}."
    if feather and os.path.exists(path + "feather"):
        print(f"Reading {filename}.feather ")
        df = pd.read_feather(path + "feather")
    
    else: 
        print(f"Reading {filename}.csv ")
        df = pd.read_csv(path + "csv")
        df.columns = df.columns.str.lower()
    
    return df


# Load time-series stratified samples for development
if USING_TIME_SERIES_DATA:
    train = load_data_init(filename=TRAIN_FILENAME, feather=True, path=DATASET_PATH)
    val = load_data_init(filename=VAL_FILENAME, feather=True, path=DATASET_PATH)

    if LOAD_ALL_DATA:
        all_train_data = load_data_init(filename="train")

else:
    all_train_data = load_data_init(filename="train")
    train, val = split_df(all_train_data, train_size=0.999) 

test_data = load_data_init(filename="test")

drop_key([train, val])


def extract_time_series_data(data, frac=0.005, val_size=10000):
    """
    Extract a balanced subset from large temporal data by stratifying across
    years, weeks, and days. This maintains the time distribution patterns.
    The function is a bit inefficiently implemented (multiple groupbys, etc).
    
    Args:
        data: Full dataset
        frac: Fraction to sample from each stratum
        val_size: Size of validation set
    """
    # Sort chronologically
    data = data.sort_values(by=["pickup_year", "pickup_week", "pickup_weekday"])
    
    # Group by temporal features
    grouped = data.groupby(['pickup_year', 'pickup_week', 'pickup_weekday'])

    # Sample proportionally from each time bin
    sampled = grouped.apply(lambda x: x.sample(frac=frac, random_state=42)) 
    sampled.index = sampled.index.droplevel([0,1,2])

    # Create validation set with recent data and samples across years
    # We sample validation data too because test set contains data from
    # various years (instead of only newest).
    n_years = len(data["pickup_year"].unique())
    last_year_size = 0 # if you want to leave more newer data samples
    size_per_year = max(1, (val_size - last_year_size) // n_years)

    # Take most recent data plus stratified samples
    val = sampled.iloc[-last_year_size:]
    sampled = sampled.iloc[:-last_year_size]

    # Get additional validation samples from each year
    year_grouped = sampled.groupby(["pickup_year"])
    sample_val = year_grouped.apply(
        lambda x: x.sample(n=min(x.shape[0], size_per_year), random_state=42))
    val = pd.concat([data.loc[sample_val.index.droplevel([0])], val])

    # Create training set with remaining samples
    train_indices = sampled.loc[~sampled.index.isin(val.index)].index
    train = data.loc[train_indices]
    
    return train, val


# US Holidays for detecting holiday dates
us_holidays = holidays.US()

def format_pickup_time(time_string):
    """
    Extract useful temporal components from a datetime string.
    
    Args:
        time_string (str): Datetime string in format 'YYYY-MM-DD HH:MM:SS UTC'
        
    Returns:
        tuple: (year, week_number, weekday, time_minutes, is_holiday)
            - year: The year component
            - week_number: Week number in the year (1-53)
            - weekday: Day of week (1=Monday, 7=Sunday)
            - time_minutes: Minutes since midnight (0-1439)
            - is_holiday: Boolean indicating if date is a US holiday
    """
    time_string = time_string[:-4] # Remove UTC timezone indicator 
    time_format = "%Y-%m-%d %H:%M:%S"
    time = datetime.datetime.strptime(time_string, time_format)

    # Extract date components
    date = time.date()
    year, week_number, weekday = time.isocalendar()
    
    # Check if date is a US holiday
    is_holiday = date in us_holidays

    # Calculate minutes from midnight (0-1439)
    time_minutes = time.hour * 60 + time.minute
    
    return year, week_number, weekday, time_minutes, is_holiday

def clean_pickup_time(df):
    """
    Process datetime column in a dataframe to extract temporal features.
    
    This function extracts year, week number, weekday, minutes from midnight,
    and holiday status from the pickup_datetime column, and removes the 
    original datetime column.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'pickup_datetime' column
        
    Returns:
        None: The function modifies the dataframe in-place
    """
    column = "pickup_datetime"
    if column not in df:
        return

    # Extract temporal features and create new columns
    new_columns = ["pickup_year", "pickup_week", 
                  "pickup_weekday", "pickup_time_minutes", "pickup_is_holiday"]
    df[new_columns] = df[column].apply(format_pickup_time).apply(pd.Series)

    # NOTE: the whole function is implemented inplace on a dataset!
    df.drop(columns=[column], inplace=True)

# Simple test
def test_pickup_time():
    """Test the datetime processing functions with sample data"""
    # Test holiday detection
    print(f"2015-01-01 is a holiday: {datetime.date(2015, 1, 1) in us_holidays}")
    
    # Test datetime parsing
    print(f"Sample parsing: {format_pickup_time('2010-01-16 20:06:00 UTC')}")
    
    # Test on actual dataframe
    temporal_df = train.iloc[:50, :].copy()
    print("Original datetime values:")
    print(temporal_df.head()["pickup_datetime"])
    
    # Apply transformation
    clean_pickup_time(temporal_df)
    print("\nAfter transformation:")
    print(temporal_df.head(20))

# test_pickup_time()


def plot_important_features(columns, importances):
    """ Plots the most important for RF features. """
    pairs = list(zip(columns, importances))
    df = pd.DataFrame(pairs, columns=["feature", "importance"])
    df = df.sort_values(by="importance", ascending=False)

    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x="importance", y="feature", orient="h")
    plt.title("Feature Importances")
    plt.tight_layout()
    plt.show()

def evaluate(x, y, model):
    """ Evaluates prediction on several scores. """
    yhat = model.predict(x)
    rmse = metrics.mean_squared_error(y, yhat)
    r2 = metrics.r2_score(y, yhat)
    print(f"MSE: {rmse}. \nR2: {r2}")


    if hasattr(model, "feature_importances_"):
        plot_important_features(x.columns, model.feature_importances_)

def baseline_model_workflow(X, y, Xval, yval):
    model = ensemble.RandomForestRegressor(random_state=42, oob_score=True).fit(X, y)
    print(model.oob_score_)

    evaluate(X, y, model)
    evaluate(Xval, yval, model)
    # Overfits

    return model

'''
Xval, yval = sep_target(val)
all_train = pd.concat([X, Xval])
all_target = pd.concat([y, yval])

model = baseline_model_workflow(X, y, Xval, yval)'''


print("Training set shape:", train.shape)
train.info()


display(train.describe())
print("Missing values:", train.isnull().sum())

# Check for zero values in each column
for c in train.columns:
    zero = (train[c] == 0).any()
    print(f"Column {c} has zero values: {zero}")

# Check for negative values
for c in train.columns:
    neg = (train[c] < 0).any()
    print(f"Column {c} has negative values: {neg}")


train["passenger_count"].unique()


train["pickup_year"].value_counts()
train["pickup_week"].value_counts()


print(train["pickup_year"].unique())
print(train["pickup_week"].unique())
print(val["pickup_year"].unique())
print(val["pickup_week"].unique())


# Pickup holiday doesn't matter (1. feature importances, 2. means)?
print(train[train["pickup_is_holiday"] == True][target_column].mean())
print(train[train["pickup_is_holiday"] == False][target_column].mean())


print(train.skew(), "\n")
print(val.skew())

# Pretty big skeweness.


def plot_heatmap(df):
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), 
        annot=True, 
        cmap='coolwarm', 
        vmin=-1, vmax=1, 
        center=0,
        square=True, 
        linewidths=.5, 
        cbar_kws={"shrink": .8})
    plt.title("Feature Correlations", fontsize=16)
    plt.tight_layout()
    plt.show()


plot_heatmap(train)


# Plot fare amount distribution
plt.figure(figsize=(10, 6))
sns.histplot(train["fare_amount"], bins=50)
plt.title("Fare Amount Distribution", fontsize=16)
plt.xlabel("Fare Amount ($)")
plt.ylabel("Frequency")
plt.show()


def all_pairplot(data):
    sns.pairplot(data)
    plt.tight_layout()
    plt.show()


def plot_distribution(x):
    sns.histplot(x, bins=50)
    plt.show()


def print_bounds(feature):
    print(feature.sort_values(ascending=False)[:60])
    print(feature.sort_values(ascending=True)[:60])


def iqr_analysis(df: pd.DataFrame):
    """Calculate percentage of outliers in each column using IQR method"""
    N = df.shape[0]
    results = {}
    
    for c in df.columns:
        if c != "pickup_is_holiday":
            q1 = df[c].quantile(0.25)
            q3 = df[c].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = df[(df[c] < lower_bound) | (df[c] > upper_bound)].shape[0] / N * 100
            results[c] = outliers
            print(f"{c} outliers: {outliers:.2f}%")
    
    return results

outlier_percentages = iqr_analysis(train)

# Plot outlier percentages
plt.figure(figsize=(12, 6))
plt.bar(outlier_percentages.keys(), outlier_percentages.values())
plt.title("Percentage of Outliers by Feature", fontsize=16)
plt.xlabel("Features")
plt.ylabel("Outlier Percentage")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Sample for visualization
# Possible improvement: plot coordinates on the real map image
sample = train.sample(n=10000, random_state=42)

plt.figure(figsize=(10, 10))
plt.scatter(sample["pickup_longitude"], sample["pickup_latitude"], 
           alpha=0.5, s=1, label="Pickup")
plt.scatter(sample["dropoff_longitude"], sample["dropoff_latitude"], 
           alpha=0.5, s=1, label="Dropoff")
plt.title("NYC Taxi Pickup and Dropoff Locations", fontsize=16)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


print(train["pickup_longitude"].describe())


#all_pairplot(sample) # Pairplot is pretty long, even with using sampled data


print_bounds(train[target_column])


def add_euclidean_distance(df):
    """Add Euclidean distance between pickup and dropoff coordinates"""
    df["euclidean_distance"] = np.sqrt(
        (df["pickup_longitude"] - df["dropoff_longitude"])**2 + 
        (df["pickup_latitude"] - df["dropoff_latitude"])**2
    )
    return df


train = add_euclidean_distance(train)


print(train[train["euclidean_distance"] == 0][target_column].describe())
plot_distribution(train[train["euclidean_distance"] == 0][target_column])


temp = train[(train[target_column] > 0) & (train[target_column] <250) & (train["euclidean_distance"] > 0) & (train["euclidean_distance"] <1)]
temp2 = train[(train[target_column] >= 0) & (train[target_column] <200) & (train["euclidean_distance"] >= 0) & (train["euclidean_distance"] <0.5)]
temp3 = train[train["euclidean_distance"] <100]

fig, ax = plt.subplots(3, 1, figsize=(10, 15))

sns.scatterplot(
    data=temp,
    x="euclidean_distance",
    y=target_column,
    alpha=0.2,
    ax=ax[0]
)

sns.scatterplot(
    data=temp2,
    x="euclidean_distance",
    y=target_column,
    alpha=0.2,
    ax=ax[1]
)

sns.scatterplot(
    data=temp3,
    x="euclidean_distance",
    y=target_column,
    alpha=0.2,
    ax=ax[2]
)

train_eu_tar = train[train["euclidean_distance"] <= 0.02][["euclidean_distance", target_column]]
#print(train_eu_tar.describe())
#print(train_eu_tar.skew())
#print(train_eu_tar.sort_values(by=[target_column, "euclidean_distance"], ascending=False)[:60])



def fix_coordinates(x: pd.DataFrame):
    """Filter coordinates to NYC bounding box"""
    # NYC bounding box (approximate)
    lt1, lg1, lt2, lg2 = 40.4774, -74.2591, 40.9176, -73.7004

    mask = (
        (x["pickup_latitude"].between(lt1, lt2)) &
        (x["dropoff_latitude"].between(lt1, lt2)) &
        (x["pickup_longitude"].between(lg1, lg2)) &
        (x["dropoff_longitude"].between(lg1, lg2))
    )
    return x[mask]


temp = train.copy()
temp = fix_coordinates(temp)
temp = add_euclidean_distance(temp)
print(temp[temp["euclidean_distance"] > 15].shape)

# After cleaning coordinates points with extreme euclidean 
# distance vanished too.


def filter_passenger_count(x: pd.DataFrame):
    """Filter passenger count to valid range (0-6)"""
    # Could be zero because taxi can simply wait for a passenger
    # or it may transporting something.
    return x[x["passenger_count"].between(0, 6)]


def filter_target(x: pd.DataFrame):
    """Remove negative and extremely low fares"""

    # 2.5 is the minimal possible taxi price
    return x[(x[target_column] > 2.5) & (x[target_column] < 200)]


def filter_euclidean_distance(x):
    """Remove anomalous fare-distance combinations"""
    if "euclidean_distance" in x:
        # Remove high fares for very short distances
        # The exact bounding values are chosen by eye
        x = x[~((x["fare_amount"] >= 30) & (x["euclidean_distance"] <= 0.001))]
        x = x[~((x["fare_amount"] >= 6) & (x["euclidean_distance"] == 0))]
        
        # Remove low fares for long distances
        # This is also not so accurate. At least this cleaning should
        # be done diagonally rather than horizontally.
        x = x[~((x["fare_amount"] <= 4) & (x["euclidean_distance"] >= 0.05))]
        x = x[~((x["fare_amount"] <= 10) & (x["euclidean_distance"] > 0.1))]
        
        # Remove specific fare anomalies (data clusterings) identified in EDA
        # Not all, but inconsistency (same fare amoung with different eu. 
        # distances were deleted).
        # This code snippet is commented because these values correspond with
        # real fixed prices in NYC and prediction scores show that my assumption
        # can be true. So I leave it this way even if they clearly contradict
        # with euclidean distance.
        '''anomalous_fares = [45, 57.33, 49.80, 49.57, 49.15, 52]
        for v in anomalous_fares:
            i = x[x["fare_amount"] == v].index
            x = x.drop(i)'''
    
    return x


temp = filter_euclidean_distance(temp)

sns.scatterplot(
    data=temp,
    x="euclidean_distance",
    y=target_column,
    alpha=0.2
)


def remove_outliers(data: pd.DataFrame):
    """Apply all cleaning steps sequentially"""
    # Handle missing values
    data.dropna(axis=0, inplace=True)
    
    # Apply all filters
    data = fix_coordinates(data)
    data = filter_target(data)
    data = filter_passenger_count(data)
    data = filter_euclidean_distance(data)
    
    # Removed approximately 5% of data.
    return data


filtered_train = remove_outliers(train)


filtered_train.describe()


plot_heatmap(filtered_train)


# By the way, scaling and encoding are mostly irrelevant for tree based methods.

def scale(X):
    """Scale features appropriately based on their distributions"""
    # Normal distribution features use StandardScaler
    normal_features = ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]
    
    # Uniform distribution features use MinMaxScaler
    uniform_features = ["pickup_time_minutes"]

    # Apply transformations
    std_scaler = preprocessing.StandardScaler()
    X[normal_features] = std_scaler.fit_transform(X[normal_features])

    u_scaler = preprocessing.MinMaxScaler()
    X[uniform_features] = u_scaler.fit_transform(X[uniform_features])

    # Store scalers for later use
    scalers = {
        "std_scaler": {
            "column": normal_features,
            "transformer": std_scaler,
        },
        "u_scaler": {
            "column": uniform_features,
            "transformer": u_scaler,
        }
    }

    return X, scalers

def encode_(X):
    """Encode categorical features"""
    label_features = ["pickup_year"]
    encoders = {}

    for i, c in enumerate(label_features):
        lbl_encoder = preprocessing.LabelEncoder()
        X[c] = lbl_encoder.fit_transform(X[c])
        encoders[f"lbl_encoder{i}"] = {
            "column": c,
            "transformer": lbl_encoder
        }

    return X, encoders


def transform_test(transformers, test):
    test = add_euclidean_distance(test)

    #print(transformers)
    for group in transformers.values():
        #print(group)
        for transformer in group.values():
            #print(transformer)
            if test is None: raise ValueError("Test is None. ")
            
            column = transformer["column"]
            tr = transformer["transformer"]
            test[column] = tr.transform(test[column])
    
    return test


def clean(train, val):
    """Complete preprocessing pipeline for training and validation data"""
    train = add_euclidean_distance(train)
    train = remove_outliers(train)
    val = remove_outliers(val)

    # Separate target from features
    X, y = sep_target(train)
    if target_column in val:
        Xval, yval = sep_target(val)
    else: Xval = val
    
    # Apply scaling and encoding
    Xscaled, scalers = scale(X)
    X_sc_enc, encoders = encode_(Xscaled)

    # Store transformations for later use
    transformers = {
        "scalers": scalers,
        "encoders": encoders
    }

    # Apply same transformations to validation data
    Xval_sc_enc = transform_test(transformers, Xval)

    # Restore target to dataframes for convenience
    X_sc_enc[target_column] = y
    if target_column in val:
        Xval_sc_enc[target_column] = yval

    return X_sc_enc, Xval_sc_enc, transformers


cleaned_train, cleaned_val, transformers = clean(train, val)


cleaned_train.describe()


plot_heatmap(cleaned_train)


X, y = sep_target(cleaned_train)
Xval, yval = sep_target(cleaned_train)
#baseline_model_workflow(X, y, Xval, yval)


train_eu_tar1 = cleaned_train[cleaned_train[target_column] <= 100][["euclidean_distance", target_column]]
train_eu_tar2 = cleaned_train[(cleaned_train[target_column] <= 50)][["euclidean_distance", target_column]]
train_eu_tar21 = cleaned_train[(cleaned_train[target_column] >= 50) & (cleaned_train["euclidean_distance"] <= 0.001)][["euclidean_distance", target_column]]
train_eu_tar3 = cleaned_train[(cleaned_train[target_column] <= 10) & (cleaned_train["euclidean_distance"] > 0.1)][["euclidean_distance", target_column]]
train_eu_tar4 = cleaned_train[(cleaned_train[target_column] <= 4) & (cleaned_train["euclidean_distance"] > 0.05)][["euclidean_distance", target_column]]
train_eu_tar31 = cleaned_train[(cleaned_train[target_column] > 10) & (cleaned_train["euclidean_distance"] > 0.1)][["euclidean_distance", target_column]]
train_eu_tar41 = cleaned_train[(cleaned_train[target_column] > 4) & (cleaned_train["euclidean_distance"] > 0.05)][["euclidean_distance", target_column]]
train_eu_tar5 = cleaned_train[(cleaned_train[target_column] >= 25) & (cleaned_train["euclidean_distance"] <= 0.001)][["euclidean_distance", target_column]]
'''print(train_eu_tar.shape)
print(train_eu_tar.describe())
print(train_eu_tar.sort_values(by=["euclidean_distance", target_column], ascending=False)[:60])'''

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 18))

sns.scatterplot(data=train_eu_tar1, x="euclidean_distance", y=target_column, alpha=0.3, ax=ax[0])
sns.scatterplot(data=train_eu_tar2, x="euclidean_distance", y=target_column, alpha=0.3, ax=ax[1])

print(train_eu_tar21.shape)
print(train_eu_tar3.shape)
print(train_eu_tar4.shape)
print(train_eu_tar31.shape)
print(train_eu_tar41.shape)
print(train_eu_tar5.shape)


# Define models to evaluate
models = {
    "RandomForestRegressor": RandomForestRegressor(random_state=42),
    "XGBRFRegressor": XGBRFRegressor(random_state=42),
    "Ridge": Ridge(random_state=42),
    "MLPRegressor": MLPRegressor(random_state=42),
    # "SVR": SVR() # performed poorly and takes lots of time to run
}


# Prepare data for modeling
cleaned_train, cleaned_val, transformers = clean(train, val)
X_train, y_train = sep_target(cleaned_train)
X_val, y_val = sep_target(cleaned_val)


'''all_init_model_scores = {}

for name, m in models.items():
    c = model_selection.cross_val_score(
        m, X=X_train, y=y_train, cv=5, scoring="neg_root_mean_squared_error", n_jobs=8, verbose=1
    )

    # Using additional testing on my val set.
    m.fit(X_train, y_train)
    yhat = m.predict(X_val)
    rmse = metrics.root_mean_squared_error(yval, yhat)

    all_init_model_scores[name] = {
        "cross_val_score": c,
        "my_val_score": rmse
    }'''


'''for model_name, scores in all_init_model_scores.items():
    all_init_model_scores[model_name]["cross_val_mean"] = scores["cross_val_score"].mean()

for model_name, scores in all_init_model_scores.items():
    print(model_name + ":", scores["cross_val_mean"], scores["my_val_score"])'''


# Tuned on GridSearch best parameters.
tuned_xgbrfr_params = {
    "colsample_bytree": 1, 
    "gamma": 0.04, 
    "learning_rate": 1, 
    "max_depth": 16, 
    "min_child_weight": 3, 
    "n_estimators": 75, 
    "subsample": 0.7
}

# Create and train model with optimal parameters
final_model = XGBRFRegressor(**tuned_xgbrfr_params, objective='reg:squarederror', random_state=42)
final_model.fit(X_train, y_train)


# Visualize feature importances
def plot_important_features(columns, importances):
    pairs = list(zip(columns, importances))
    df = pd.DataFrame(pairs, columns=["feature", "importance"])
    df = df.sort_values(by="importance", ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=df, x="importance", y="feature", orient="h")
    plt.title("Feature Importances", fontsize=16)
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

# Plot feature importances
plot_important_features(X_train.columns, final_model.feature_importances_)


# Final model evaluation
evaluate(X_val, y_val, final_model)
evaluate(X_train, y_train, final_model)

# Possible improvement: handling overfitting (or extreme outliers in the 
# validation set like horizontal clusterings with a fixed fare_amount)
# Although it showed a good result on the test set 
# (without fitting to test set!). So I just left it.


# Visualize predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_train, final_model.predict(X_train), alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
plt.xlabel("Actual Fare Amount")
plt.ylabel("Predicted Fare Amount")
plt.title("Predicted vs Actual Fare Amounts (Training Set)")
plt.tight_layout()
plt.show()


# Possible improvement: Fixing underpredicting for larger fare_amounts 
# and undersample spike with fare_amount ~ 12.5


# Visualize predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_val, final_model.predict(X_val), alpha=0.5)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
plt.xlabel("Actual Fare Amount")
plt.ylabel("Predicted Fare Amount")
plt.title("Predicted vs Actual Fare Amounts (Validation Set)")
plt.tight_layout()
plt.show()


def test_prediction(transformers, test_data, model):
    key, test_data = test_data["key"], test_data.drop("key", axis=1)
    clean_pickup_time(test_data)
    test_data = transform_test(transformers, test_data)

    #print(test_data.columns)
    yhat = model.predict(test_data)
    result = {
        "key": key,
        "fare_amount": yhat
    }
    yhat = pd.DataFrame(result)

    yhat.to_csv(path_or_buf=f"/kaggle/working/nyc_taxi_fare_submission2.csv", index=False)
    print("Submission file created")


test_prediction(transformers, test_data, final_model)

