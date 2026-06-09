import gc
import lightgbm as lgb
import logging
import matplotlib.pyplot as plt
import numpy as np
import os, warnings
import pandas as pd
import seaborn as sns
import time

from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Configure logging to show time and log level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Start timer for Cell 1
start_time = time.time()

# (No complex operation here – just setting up the environment)

elapsed = time.time() - start_time
logging.info(f"Importing executed - time elapsed {elapsed:.2f} seconds")

warnings.filterwarnings("ignore")


# Define file paths (adjust these paths if needed)
train_path = '/kaggle/input/playground-series-s5e4/train.csv'
test_path = '/kaggle/input/playground-series-s5e4/test.csv'
sample_submission_path = '/kaggle/input/playground-series-s5e4/sample_submission.csv'

# Load the datasets
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_submission_path)

# Log dataset shapes for verification
logging.info(f"Train dataset shape: {train_df.shape}")
logging.info(f"Test dataset shape: {test_df.shape}")
logging.info(f"Submission dataset shape: {sample_submission.shape}")

elapsed = time.time() - start_time
logging.info(f"Loading executed - time elapsed {elapsed:.2f} seconds")


# Check for duplicate rows in the training dataset
duplicate_train = train_df.duplicated().sum()
logging.info(f"Duplicate rows in train dataset: {duplicate_train}")


# Check for duplicate rows in the test dataset
duplicate_test = test_df.duplicated().sum()
logging.info(f"Duplicate rows in test dataset: {duplicate_test}")

elapsed = time.time() - start_time
logging.info(f"Loading executed - time elapsed {elapsed:.2f} seconds")


# Display the first few rows of the training data
display(train_df.head())


# Print information about the dataset (data types, non-null counts, etc.)
logging.info("Train dataset info:")
display(train_df.info())


# Display summary statistics for numerical features in the training dataset
logging.info("Train dataset summary statistics:")
display(train_df.describe())

elapsed = time.time() - start_time
logging.info(f"Cell 3 executed in {elapsed:.2f} seconds")


# Calculate missing values for the train dataset
missing_train = train_df.isnull().sum()
logging.info("Missing values in the train dataset:")
print(missing_train)


# Calculate missing values for the test dataset
missing_test = test_df.isnull().sum()
logging.info("Missing values in the test dataset:")
print(missing_test)


# Visualize missing values (if any exist) for the train dataset
if missing_train.sum() > 0:
    plt.figure()
    missing_train[missing_train > 0].plot(kind='bar')
    plt.title("Missing Values in Train Dataset")
    plt.xlabel("Columns")
    plt.ylabel("Number of Missing Values")
    plt.show()
else:
    logging.info("No missing values found in the train dataset.")

elapsed = time.time() - start_time
logging.info(f"Cell 4 executed in {elapsed:.2f} seconds")


# Missingness Pattern Heatmap

# Create a boolean mask
mask = train_df.isnull()
plt.figure(figsize=(12, 4))
sns.heatmap(mask.T, cbar=False, cmap='viridis')
plt.title("Missingness Heatmap")
plt.xlabel("Row index")
plt.ylabel("Feature")
plt.show()

elapsed = time.time() - start_time
logging.info(f"Missingness pattern executed – time elapsed {elapsed:.2f} seconds")


# Identify numeric columns in the training dataset
numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
logging.info(f"Numeric columns in the train dataset: {numeric_cols}")

# Plot histograms for each numeric column
for col in numeric_cols:
    plt.figure()
    plt.hist(train_df[col].dropna(), bins=50, alpha=0.7)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

elapsed = time.time() - start_time
logging.info(f"Loading executed - time elapsed {elapsed:.2f} seconds")


# Identify the categorical columns
categorical_cols = train_df.select_dtypes(include = ['object']).columns.tolist()
logging.info(f"Categorical columns in the train dataset : {categorical_cols}")

# Plot count for each categorical columns
for col in categorical_cols:
    plt.figure()
    train_df[col].value_counts().plot(kind = 'bar')
    plt.title(f"Count plot of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.show()

elapsed = time.time() - start_time
logging.info(f"Loading executed - time elapsed {elapsed:.2f} seconds")


# Compute the correlation matrix for numerical features
corr_matrix = train_df[numeric_cols].corr()
logging.info("Computed correlation matrix")

# Visualise correlation matrix using heatmap
plt.figure()
plt.imshow(corr_matrix, interpolation = 'none')
plt.colorbar()
plt.title("Correlation Matrix Heatmap")
plt.xticks(range(len(corr_matrix)), corr_matrix.columns, rotation = 90)
plt.yticks(range(len(corr_matrix)), corr_matrix.columns)
plt.show()


display(corr_matrix)


elapsed = time.time() - start_time
logging.info(f"Loading executed - time elapsed {elapsed:.2f} seconds")


for col in train_df.columns:
    if col != 'Listening_Time_minutes':
        plt.figure()
        plt.scatter(train_df[col], train_df['Listening_Time_minutes'], alpha = 0.5)
        plt.title(f"{col} vs Listening Time")
        plt.xlabel(col)
        plt.xticks(rotation = 90)
        plt.ylabel("Listening Time (Minutes)")
        plt.show()

elapsed = time.time() - start_time
logging.info(f"Loading executed - time elapsed {elapsed:.2f} seconds")


# Get unique categories from Publication_Day and prepare the data for a boxplot
for col in categorical_cols:
    categories = train_df[col].unique()
    data = [train_df[train_df[col] == cat]['Listening_Time_minutes'] for cat in categories]

    plt.figure()
    plt.boxplot(data, labels=categories)
    plt.title(f"Listening Time vs. {col}")
    plt.xlabel(col)
    plt.xticks(rotation = 90)
    plt.ylabel("Listening Time (minutes)")
    plt.show()

elapsed = time.time() - start_time
logging.info(f"Cell 9 executed in {elapsed:.2f} seconds")


# Group by Genre and calculate mean Listening_Time_minutes
for col in categorical_cols:
    col_means = train_df.groupby(col)['Listening_Time_minutes'].mean().reset_index()
    logging.info(f"Mean Listening Time by {col}:")
    print(col_means)

    plt.figure()
    plt.bar(col_means[col], col_means['Listening_Time_minutes'])
    plt.title(f"Average Listening Time by {col}")
    plt.xlabel(col)
    plt.ylabel("Average Listening Time (minutes)")
    plt.xticks(rotation=90)
    plt.show()

elapsed = time.time() - start_time
logging.info(f"Cell 12 executed in {elapsed:.2f} seconds")


for col in numeric_cols:
    plt.figure()
    plt.boxplot(train_df[col].dropna())
    plt.title(f"Box plot of {col}")
    plt.xlabel(col)
    plt.show()


outlier_counts = {}

for feat in numeric_cols:
    col = train_df[feat].dropna()
    Q1 = col.quantile(0.25)
    Q3 = col.quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    cnt = ((col < lower) | (col > upper)).sum()
    outlier_counts[feat] = cnt
    logging.info(f"{feat}: {cnt} outliers (below {lower:.2f} or above {upper:.2f})")

elapsed = time.time() - start_time
logging.info(f"Outlier detection executed – time elapsed {elapsed:.2f} seconds")


# Creating a feature 
train_df['Title_Word_Count'] = train_df['Episode_Title'].str.split().str.len()

plt.figure()
plt.scatter(train_df['Title_Word_Count'], train_df['Listening_Time_minutes'], alpha = 0.5)
plt.title("Title Word Count vs. Listening Time")
plt.xlabel("Title Word Count")
plt.ylabel("Listening Time (minutes)")
plt.show()

elapsed = time.time() - start_time
logging.info(f"Title word‑count analysis executed – time elapsed {elapsed:.2f} seconds")


train_df['Episode_Title']


drift_pvals = {}
for col in numeric_cols:
    if col != 'Listening_Time_minutes':
        a = train_df[col].dropna()
        b = test_df[col].dropna()
        stat, p = ks_2samp(a, b)
        drift_pvals[col] = p
        logging.info(f"KS p‑value for {col}: {p:.3f}")

elapsed = time.time() - start_time
logging.info(f"Drift testing executed – time elapsed {elapsed:.2f} seconds")


# Prepare data (drop rows with missing in numeric_feats)
df_num = train_df[numeric_cols].dropna()
scaler = StandardScaler()
Z = scaler.fit_transform(df_num)

pca = PCA(n_components=2)
coords = pca.fit_transform(Z)

plt.figure()
plt.scatter(coords[:,0], coords[:,1],
            c=train_df.loc[df_num.index,'Listening_Time_minutes'],
            alpha=0.5)
plt.colorbar(label='Listening Time')
plt.title("PCA (2 comps) colored by Listening Time")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.show()

elapsed = time.time() - start_time
logging.info(f"PCA executed – time elapsed {elapsed:.2f} seconds")


train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv', index_col='id')
test  = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv',  index_col='id')

for col in ['Episode_Length_minutes','Guest_Popularity_percentage']:
    med = train[col].median()
    train[col].fillna(med, inplace=True)
    test[col].fillna(med,  inplace=True)

train = train[train['Number_of_Ads'] < 10].dropna()

cats = ['Podcast_Name','Genre','Publication_Day','Publication_Time','Episode_Sentiment']
encoders = {c: LabelEncoder().fit(train[c]) for c in cats}

for c in cats:
    # get the integer labels as a numpy array
    arr_train = encoders[c].transform(train[c])
    arr_test  = encoders[c].transform(test[c])
    
    # wrap them in pd.Categorical so pandas knows they’re categories
    train[c] = pd.Categorical(arr_train, categories=np.arange(len(encoders[c].classes_)))
    test[c]  = pd.Categorical(arr_test,  categories=np.arange(len(encoders[c].classes_)))

train['Episode_Num'] = train['Episode_Title'].str[8:].astype('category')
test ['Episode_Num'] = test ['Episode_Title'].str[8:].astype('category')

train.drop(columns=['Episode_Title'], inplace=True)
test .drop(columns=['Episode_Title'], inplace=True)

X = train.drop(columns='Listening_Time_minutes')
y = train['Listening_Time_minutes']


# CV loop: collect OOF predictions
oof_preds = np.zeros(len(train))
cv = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (trn_idx, val_idx) in enumerate(cv.split(X,y), 1):
    X_tr, y_tr = X.iloc[trn_idx], y.iloc[trn_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

    model = lgb.LGBMRegressor(
        n_estimators=1000, learning_rate=0.03, num_leaves=1024,
        subsample=0.7, colsample_bytree=0.7, max_bin=1024,
        objective='regression', random_state=42
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )

    oof_preds[val_idx] = model.predict(X_va)
    gc.collect()


# Compute residuals & overall OOF RMSE
residuals = y.values - oof_preds
rmse = np.sqrt(mean_squared_error(y, oof_preds))
logging.info(f"OOF RMSE: {rmse:.3f}")


# Residual distribution
plt.figure(figsize=(6,4))
plt.hist(residuals, bins=50, alpha=0.7)
plt.title("Residual Distribution (y - ŷ)")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.show()


# Residuals vs. Predictions
plt.figure(figsize=(6,4))
plt.scatter(oof_preds, residuals, alpha=0.3)
plt.axhline(0, color='red', linewidth=1)
plt.title("Residuals vs. Predicted Listening Time")
plt.xlabel("Predicted Listening Time")
plt.ylabel("Residual")
plt.show()


# Residuals vs. Episode Length
plt.figure(figsize=(6,4))
plt.scatter(train['Episode_Length_minutes'], residuals, alpha=0.3)
plt.axhline(0, color='red', linewidth=1)
plt.title("Residuals vs. Episode Length")
plt.xlabel("Episode Length (min)")
plt.ylabel("Residual")
plt.show()


# Boxplot: Residuals by Genre
plt.figure(figsize=(8,4))
genres = train['Genre'].cat.categories
data = [residuals[train['Genre']==g] for g in genres]
plt.boxplot(data, labels=genres, showfliers=False)
plt.title("Residuals by Genre")
plt.xlabel("Genre")
plt.ylabel("Residual")
plt.xticks(rotation=45)
plt.show()

elapsed = time.time() - start_time
logging.info(f"Residual analysis executed - time elapsed {elapsed:.2f} seconds")


pred_test_base = np.mean(oof_preds, axis=0)

sub_base = sample_submission.copy()
sub_base["Listening_Time_minutes"] = pred_test_base
sub_base.to_csv("submission_base.csv", index=False)

logging.info("submission_base.csv written")


train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv", index_col="id")
test  = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv",  index_col="id")

num_na_fill = {
    "Episode_Length_minutes": train["Episode_Length_minutes"].median(),
    "Guest_Popularity_percentage": train["Guest_Popularity_percentage"].median(),
}
for col, med in num_na_fill.items():
    train[col].fillna(med, inplace=True)
    test[col].fillna(med,  inplace=True)

# Clip extreme ad count & drop remaining NA rows
train = train[train["Number_of_Ads"] < 10].dropna()

# Episode number from title
train["Episode_Num"] = (
    train["Episode_Title"].str.extract(r"(\d+)", expand=False).astype(int)
)
test["Episode_Num"] = (
    test["Episode_Title"].str.extract(r"(\d+)", expand=False).astype(int)
)

elapsed = time.time() - start_time
logging.info(f"Cleaning executed - time elapsed {elapsed:.2f} seconds")


df_list = [train, test]
for df in df_list:
    # ---- 1) polynomial / transforms of length ----
    df["len_log"]      = np.log1p(df["Episode_Length_minutes"])
    df["len_sq"]       = df["Episode_Length_minutes"] ** 2
    df["len_cubert"]   = np.cbrt(df["Episode_Length_minutes"])

    # ---- 2) host/guest transforms ----
    df["host_log"]     = np.log1p(df["Host_Popularity_percentage"])
    df["host_sq"]      = df["Host_Popularity_percentage"] ** 2
    df["guest_log"]    = np.log1p(df["Guest_Popularity_percentage"])
    df["guest_sq"]     = df["Guest_Popularity_percentage"] ** 2
    df["pop_diff"]     = df["Host_Popularity_percentage"] - df["Guest_Popularity_percentage"]
    df["pop_sum"]      = df["Host_Popularity_percentage"] + df["Guest_Popularity_percentage"]
    df["pop_ratio"]    = df["Host_Popularity_percentage"] / (df["Guest_Popularity_percentage"] + 1)
    df["pop_diff_sign"] = (df["pop_diff"] > 0).astype(int)

    # ---- 3) Number_of_Ads transforms ----
    df["ads_log"]      = np.log1p(df["Number_of_Ads"])
    df["ads_zero"]     = (df["Number_of_Ads"] == 0).astype(int)
    df["ads_high"]     = (df["Number_of_Ads"] >= 3).astype(int)
    df["ads_per_min"]  = df["Number_of_Ads"] / (df["Episode_Length_minutes"] + 1)

    # ---- 4) interactions ----
    df["len_host_mul"] = df["Episode_Length_minutes"] * df["Host_Popularity_percentage"]
    df["len_guest_mul"]= df["Episode_Length_minutes"] * df["Guest_Popularity_percentage"]
    df["ads_len_mul"]  = df["Number_of_Ads"] * df["Episode_Length_minutes"]

    # ---- 5) buckets ----
    df["len_bucket"]   = pd.cut(df["Episode_Length_minutes"],
                                bins=[-1,20,40,60,80,120,1e9],
                                labels=["A","B","C","D","E","F"])
    df["ads_bucket"]   = pd.cut(df["Number_of_Ads"],
                                bins=[-1,0,1,2,3,5,10],
                                labels=["0","1","2","3","4-5","6-9"])
    df["host_bucket"]  = pd.qcut(df["Host_Popularity_percentage"], 5, labels=False)
    df["guest_bucket"] = pd.qcut(df["Guest_Popularity_percentage"], 5, labels=False)

    # ---- 6) temporal ordinals & flags ----
    day_ord = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,
               "Friday":4,"Saturday":5,"Sunday":6}
    time_ord= {"Morning":0,"Afternoon":1,"Evening":2,"Night":3}
    sent_ord= {"Negative":0,"Neutral":1,"Positive":2}
    df["pub_day_ord"]  = df["Publication_Day"].map(day_ord)
    df["pub_time_ord"] = df["Publication_Time"].map(time_ord)
    df["sent_ord"]     = df["Episode_Sentiment"].map(sent_ord)
    df["is_weekend"]   = df["Publication_Day"].isin(["Saturday","Sunday"]).astype(int)
    df["is_night"]     = (df["Publication_Time"]=="Night").astype(int)

    # ---- 7) episode number derivatives ----
    df["ep_mod10"]     = df["Episode_Num"] % 10
    df["ep_quartile"]  = pd.qcut(df["Episode_Num"], 4, labels=False)

    # ---- 8) rare-category flags ----
    df["rare_podcast"] = (df["Podcast_Name"].map(df["Podcast_Name"].value_counts()) < 10000).astype(int)

# ---- 9) Target-encoding (KFold mean) for 3 high-cardinality cats ----
for col in ["Podcast_Name","Genre","Publication_Day"]:
    global_mean = train["Listening_Time_minutes"].mean()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    te_vals = np.zeros(len(train))

    for tr_idx, val_idx in kf.split(train):
        tr_mean = train.iloc[tr_idx].groupby(col)["Listening_Time_minutes"].mean()
        te_vals[val_idx] = train.iloc[val_idx][col].map(tr_mean).fillna(global_mean)

    train[f"{col}_te"] = te_vals
    # mapping computed on full train for test
    full_mean = train.groupby(col)["Listening_Time_minutes"].mean()
    test[f"{col}_te"] = test[col].map(full_mean).fillna(global_mean)

# Episode number from title
train["Episode_Num"] = train["Episode_Title"].str.extract(r"(\d+)", expand=False).astype(int)
test ["Episode_Num"] = test ["Episode_Title"].str.extract(r"(\d+)", expand=False).astype(int)

#  NEW: discard raw text column – we have Episode_Num now
train.drop(columns="Episode_Title", inplace=True)
test .drop(columns="Episode_Title", inplace=True)

# Count how many engineered
engineered_cols = sorted(list(set(train.columns) - set(
    ["Listening_Time_minutes"])))
logging.info(f"Total feature columns (including original): {len(engineered_cols)}")

elapsed = time.time() - start_time
logging.info(f"Feature engineering executed - time elapsed {elapsed:.2f} seconds")


# Encode categoricals for LGBM

cat_cols = [
    "Podcast_Name","Genre","Publication_Day","Publication_Time",
    "Episode_Sentiment","len_bucket","ads_bucket"
]
label_enc = {}
for c in cat_cols:
    le = LabelEncoder().fit(pd.concat([train[c], test[c]]))
    train[c] = pd.Categorical(le.transform(train[c]))
    test[c]  = pd.Categorical(le.transform(test[c]))
    label_enc[c] = le          # save if needed

elapsed = time.time() - start_time
logging.info(f"Encoding executed - time elapsed {elapsed:.2f} seconds")


# Find every non-numeric, non-boolean, non-category column
obj_cols = X.select_dtypes(include="object").columns
if len(obj_cols):
    logging.info(f"Encoding extra object cols → {list(obj_cols)}")
    for col in obj_cols:
        le = LabelEncoder().fit(pd.concat([train[col], test[col]]).astype(str))
        train[col] = pd.Categorical(le.transform(train[col].astype(str)))
        test[col]  = pd.Categorical(le.transform(test[col].astype(str)))
        label_enc[col] = le     # keep for inference

# Re-assemble matrix now that dtypes are safe
X = train.drop(columns="Listening_Time_minutes")
bad = X.select_dtypes(exclude=["number","bool","category"]).columns
assert len(bad) == 0, f"Still non-numeric: {list(bad)}"

elapsed = time.time() - start_time
logging.info(f"Extra encoding executed – time elapsed {elapsed:.2f} seconds")


# Train LGBM with 50+ features

X = train.drop(columns="Listening_Time_minutes")
y = train["Listening_Time_minutes"]

oof = np.zeros(len(train))
cv   = KFold(n_splits=5, shuffle=True, random_state=42)

params = dict(
    n_estimators = 1200,
    learning_rate= 0.03,
    num_leaves   = 1024,
    subsample    = 0.7,
    colsample_bytree = 0.7,
    max_bin      = 1024,
    objective    = "regression",
    random_state = 42
)

for fold,(trn, val) in enumerate(cv.split(X,y),1):
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X.iloc[trn], y.iloc[trn],
        eval_set=[(X.iloc[val], y.iloc[val])],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    oof[val] = model.predict(X.iloc[val])
    gc.collect()

rmse = np.sqrt(mean_squared_error(y,oof))
logging.info(f"OOF RMSE with engineered features: {rmse:.3f}")

elapsed = time.time() - start_time
logging.info(f"Modeling executed - time elapsed {elapsed:.2f} seconds")


# Cell 7 ▸ Residual analysis plots
start_time = time.time()

resid = y - oof

# 1. Histogram
plt.figure(figsize=(6,3))
plt.hist(resid, bins=60, alpha=.75)
plt.title("Residual Histogram")
plt.xlabel("Residual"); plt.ylabel("Freq")
plt.show()

# 2. Residual vs. Prediction
plt.figure(figsize=(6,3))
plt.scatter(oof, resid, alpha=.3)
plt.axhline(0,color='red')
plt.title("Residuals vs. Predicted")
plt.xlabel("Predicted"); plt.ylabel("Residual")
plt.show()

# 3. Residual vs. Episode_Length_minutes
plt.figure(figsize=(6,3))
plt.scatter(train["Episode_Length_minutes"], resid, alpha=.3)
plt.axhline(0,color='red')
plt.title("Residuals vs. Episode Length")
plt.xlabel("Episode Length"); plt.ylabel("Residual")
plt.show()

elapsed = time.time() - start_time
logging.info(f"Residual plots executed - time elapsed {elapsed:.2f} seconds")


pred_test_fe = np.mean(oof, axis=0)

sub_fe = sample_submission.copy()
sub_fe["Listening_Time_minutes"] = pred_test_fe
sub_fe.to_csv("submission.csv", index=False)

logging.info("submission_fe.csv written")




