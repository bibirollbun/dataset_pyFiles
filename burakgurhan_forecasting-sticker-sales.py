import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train_df = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
sample_df = pd.read_csv("/kaggle/input/playground-series-s5e1/sample_submission.csv")
df = train_df.copy()
df.head()


sample_df.head(2)


test_df.head(2)


def basics(df):
    print("\nNumber of columns: ", df.shape[1])
    print("Number of rows: ", df.shape[0])
    print("Number of duplicated rows: ", df.duplicated().sum())
    print("Number of NaNs: ", df.isna().sum().sum())
    print("Dataset begins: ", df["date"].min())
    print("Dataset ends: ", df["date"].max())
    print(f"Dataset contains {len(df['country'].unique())} countries")
    print(f"Dataset contains {len(df['store'].unique())} type of stores")
    print(f"Dataset contains {len(df['product'].unique())} type of products")

basics(df)
basics(test_df)


def create_date_features(df):
    df["date"] = pd.to_datetime(df["date"])

    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df['week_of_year'] = df["date"].dt.isocalendar().week
    df["dayofweek"] = df["date"].dt.dayofweek
    df['is_weekend'] = df["dayofweek"] >= 5

    df["year_sin"] = np.sin(2 * np.pi * df["year"] / 7.0)
    df["year_cos"] = np.cos(2 * np.pi * df["year"] / 7.0)
    df["quarter_sin"] = np.sin(2 * np.pi * df["quarter"] / 4.0)
    df["quarter_cos"] = np.cos(2 * np.pi * df["quarter"] / 4.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df['day_sin'] = np.sin(2 * np.pi + df['day']  / 365.0)
    df['day_cos'] = np.cos(2 * np.pi + df['day'] / 365.0)
    df['dow_sin'] = np.sin(2 * np.pi + df['dayofweek']  / 7)
    df['dow_cos'] = np.cos(2 * np.pi + df['dayofweek'] / 7)
    
create_date_features(df)
create_date_features(test_df)


df.describe()


df.info()


# Then separate columns as numerical and categorical
num_cols = ["num_sold"]
cat_cols = df.select_dtypes("object").columns.tolist()

print("\nNumerical columns:", num_cols)
print("Categorical columns:", cat_cols)

def one_variate_visualization(df):
    n_cols = len(num_cols)
    n_rows = (n_cols + 1) // 2  

    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 4*n_rows))
    plt.style.use('seaborn')
    axes = axes.flatten()

    for idx, col in enumerate(num_cols):
        # Histogram with KDE
        sns.histplot(data=df, x=col, ax=axes[idx], kde=True, bins=50)
        axes[idx].set_title(f'Distribution of {col}', fontsize=12)
        axes[idx].set_xlabel(col, fontsize=10)
        axes[idx].set_ylabel('Count', fontsize=10)

        # Boxplot
        sns.boxplot(data=df, y=col, ax=axes[idx + n_cols])
        axes[idx + n_cols].set_title(f'Boxplot of {col}', fontsize=12)
        axes[idx + n_cols].set_ylabel(col, fontsize=10)

    plt.suptitle('Distribution Analysis of Numerical Features', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

one_variate_visualization(df.drop("id", axis=1))


def categorical_visualization(df):
    n_cats = len(cat_cols)
    
    # Handle single categorical variable case
    if n_cats == 1:
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.countplot(data=df, x=cat_cols[0], ax=ax)
        ax.set_title(f'Distribution of {cat_cols[0]}', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
    else:
        fig, axes = plt.subplots(n_cats, 1, figsize=(6, 3*n_cats))
        
        for idx, col in enumerate(cat_cols):
            sns.countplot(data=df, x=col, ax=axes[idx])
            axes[idx].set_title(f'Distribution of {col}', fontsize=12)
            axes[idx].tick_params(axis='x', rotation=45)

    plt.suptitle('Distribution Analysis of Categorical Features', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

categorical_visualization(df.drop("id", axis=1))


# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# First subplot: Total sales by product
sales_by_product = df.groupby("product")["num_sold"].sum().sort_values(ascending=True).reset_index()
sns.barplot(data=sales_by_product, x="num_sold", y="product", ax=axes[0])
axes[0].set_title("Best Seller Products", fontsize=12, pad=10)
axes[0].set_xlabel("Total Sales", fontsize=10)
axes[0].set_ylabel("Product", fontsize=10)

# Second subplot: Sales over time by product
daily_sales = df.groupby(["date", "product"])["num_sold"].sum().reset_index()
sns.lineplot(data=daily_sales, x="date", y="num_sold", hue="product", ax=axes[1])
axes[1].set_title("Sales Trends Over Time", fontsize=12, pad=10)
axes[1].set_xlabel("Date", fontsize=10)
axes[1].set_ylabel("Sales", fontsize=10)
axes[1].tick_params(axis='x', rotation=45)
axes[1].legend(title="Product", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.suptitle('Product Sales Analysis', fontsize=14, y=1.05)
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(3, 2, figsize=(12, 10))
axes = axes.flatten()
for i, cntry in enumerate(df["country"].unique()):
    sales_by_country = df[df["country"]==cntry].groupby(["date", "product"])["num_sold"].sum().reset_index()
    sns.lineplot(data=sales_by_country, x="date", y="num_sold", hue="product", ax=axes[i])
    axes[i].set_title(f"{cntry} Sales", fontsize=12, pad=10)
    axes[i].set_xlabel("Years", fontsize=10)
    axes[i].set_ylabel("Sales", fontsize=10)

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(12,4))


daily_product_sales = df.groupby(["date", "store"])["num_sold"].mean().reset_index()
sns.lineplot(data=daily_product_sales, x="date", y="num_sold", hue="store", ax=axes[0])
axes[0].set_title("Sales by Date")
axes[0].set_ylabel("Mean Sales")
axes[0].set_xlabel("Date")

country_product_sales = df.groupby(["date", "country"])["num_sold"].mean().reset_index()
sns.lineplot(data=country_product_sales, x="date", y="num_sold", hue="country", ax=axes[1])
axes[1].set_title("Sales by Date")
axes[1].set_ylabel("Mean Sales")
axes[1].set_xlabel("Date")

plt.tight_layout()
plt.show();


null = df.isnull()
sns.heatmap(null)
plt.title("Null value Heat Map")
plt.show();


df.pivot_table(index='country', columns='product', values='num_sold', aggfunc='sum')


# Fill or remove blank rows.
df = df.dropna()


import holidays
years_list = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]

holiday_CA = holidays.CountryHoliday('CA', years = years_list)
holiday_FI = holidays.CountryHoliday('FI', years = years_list)
holiday_IT = holidays.CountryHoliday('IT', years = years_list)
holiday_KE = holidays.CountryHoliday('KE', years = years_list)
holiday_NO = holidays.CountryHoliday('NO', years = years_list)
holiday_SG = holidays.CountryHoliday('SG', years = years_list)

holiday_dict = holiday_CA.copy()
holiday_dict.update(holiday_FI)
holiday_dict.update(holiday_IT)
holiday_dict.update(holiday_KE)
holiday_dict.update(holiday_NO)
holiday_dict.update(holiday_SG)

def map_holydays(df, map_dict = holiday_dict):
    '''
    Describe the function...
    '''
    df['date'] = pd.to_datetime(df['date']) # Convert the date to datetime.
    df['holiday_name'] = df['date'].map(holiday_dict)
    df['is_holiday'] = np.where(df['holiday_name'].notnull(), 1, 0)
    df['holiday_name'] = df['holiday_name'].fillna('Not Holiday')

    return df
    
df = map_holydays(df, holiday_dict)
test_df = map_holydays(test_df, holiday_dict)


# Convert categorical features into numerical features
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

for col in df.select_dtypes("object").columns.tolist():
    df[col] = label_encoder.fit_transform(df[col])

for col in test_df.select_dtypes("object").columns.tolist():
    test_df[col] = label_encoder.fit_transform(test_df[col])


corr = df.corr()

plt.figure(figsize=(16,5))
sns.heatmap(data=corr, annot=True, cmap="coolwarm", fmt='.2f')
plt.title("Correlation of Features")
plt.show();


df = df.sort_values(by="id")


X = df.dropna().drop(["id", "date", "num_sold"], axis=1)
y = df["num_sold"].dropna()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train = df[df["date"]<="2015-12-31"].drop(["id", "date", "num_sold"], axis=1)
X_test = df[df["date"]>"2015-12-31"].drop(["id", "date", "num_sold"], axis=1)
y_train = df[df["date"]<="2015-12-31"]["num_sold"]
y_test  = df[df["date"]>"2015-12-31"]["num_sold"]

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


def model_train(model, X_train, y_train, X_test, y_test):
    # Model training.
    model.fit(X_train, y_train)

    # Making predictions
    y_pred = model.predict(X_test)

    # Performance metrics.
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    print("Test RMSE:", rmse)
    print("Test Mape:", mape)
    print("Test R²:", r2)

params = {'subsample': 0.9, 'n_estimators': 800, 'min_child_weight': 5, 'max_depth': 8, 'learning_rate': 0.1, 'gamma': 0.1, 'colsample_bytree': 0.9}
best_model = XGBRegressor(random_state=42, n_jobs=-1, **params)

model_train(best_model, X_train, y_train, X_test, y_test)


params = {'subsample': 0.9, 'reg_lambda': 0.1, 'reg_alpha': 0.5, 'num_leaves': 50, 'n_estimators': 1500, 'min_child_samples': 70, 'max_depth': 10, 'learning_rate': 0.05, 'colsample_bytree': 0.8}
lgbm = LGBMRegressor(**params, random_state=42, verbose=0)
model_train(lgbm, X_train, y_train, X_test, y_test)


feature_importances = lgbm.feature_importances_
features = X.columns

plt.figure(figsize=(8, 6))
sns.barplot(x=feature_importances, y=features, palette="viridis")
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show();


y_pred = lgbm.predict(X_test)
residuals = y_test - y_pred

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

sns.scatterplot(x=y_test, y=y_pred, color="blue", alpha=0.7, ax=ax1)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--")
ax1.set_title("Actual vs Predicted")
ax1.set_xlabel("Actual Values")
ax1.set_ylabel("Predicted Values")

sns.histplot(residuals, kde=True, color="purple", bins=20, ax=ax2)
ax2.set_title("Residuals Distribution")
ax2.set_xlabel("Residuals")
ax2.set_ylabel("Frequency")

plt.tight_layout()
plt.show();


f_importance = pd.DataFrame({"Features":X_train.columns, "Importance":lgbm.feature_importances_}).sort_values(by="Importance", ascending=False)

new_columns = f_importance[:10]["Features"]

X_new = X[new_columns]
X_train_new = X_train[new_columns]
X_test_new = X_test[new_columns]

model_train(lgbm, X_train_new, y_train, X_test_new, y_test)


# Train a model with full of data.
lgbm.fit(X, y)

# Remove unnecessary columns from test data.
test = test_df[X.columns]

# Prediction with test data.
y_pred = lgbm.predict(test)

# Create submission file.
sample_df["num_sold"] = y_pred
sample_df.to_csv("submission.csv", index=False)


test_df["num_sold"] = y_pred

full_df = pd.concat([df, test_df])

plt.figure(figsize=(12,5))
sns.lineplot(data=full_df, x="date", y="num_sold")
plt.title("Actuals and Predictions")
plt.xlabel("Date")
plt.ylabel("Number of Sales")

plt.axvline(x=pd.Timestamp("2017-01-01"), color="red", linestyle="--", label="Prediction Starts")

plt.grid()
plt.show();




