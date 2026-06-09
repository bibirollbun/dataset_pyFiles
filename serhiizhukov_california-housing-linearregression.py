import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from scipy.stats import zscore


california_housing = fetch_california_housing(as_frame=True)

data = california_housing['frame']
data.head()


target = data.pop('MedHouseVal')
target.head()


data.info()


sns.set_theme()

melted = pd.concat([data, target], axis=1).melt()

g = sns.FacetGrid(melted,
                  col='variable',
                  col_wrap=3,
                  sharex=False,
                  sharey=False)

with warnings.catch_warnings():
    warnings.simplefilter('ignore')

    g.map(sns.histplot, 'value')

g.set_titles(col_template='{col_name}')

g.tight_layout()


features_of_interest = ['AveRooms', 'AveBedrms', 'AveOccup', 'Population']
data[features_of_interest].describe()


# Outlier removal
# Ğ�Ñ‡Ğ¸Ñ�Ñ‚ĞºĞ° Ğ²Ñ–Ğ´ Ğ²Ğ¸ĞºĞ¸Ğ´Ñ–Ğ²
columns_to_clean = ['AveRooms', 'AveBedrms', 'AveOccup', 'Population']
z_scores = data[columns_to_clean].apply(zscore)

# Data filtering: removing rows where at least one Z-score value exceeds 3 (or -3)
# Ğ¤Ñ–Ğ»ÑŒÑ‚Ñ€Ğ°Ñ†Ñ–Ñ� Ğ´Ğ°Ğ½Ğ¸Ñ…: Ğ²Ğ¸Ğ´Ğ°Ğ»Ñ�Ñ”Ğ¼Ğ¾ Ñ€Ñ�Ğ´ĞºĞ¸, Ğ´Ğµ Ñ…Ğ¾Ñ‡Ğ° Ğ± Ğ¾Ğ´Ğ½Ğµ Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ½Ñ� Z-Ğ¾Ñ†Ñ–Ğ½ĞºĞ¸ Ğ¿ĞµÑ€ĞµĞ²Ğ¸Ñ‰ÑƒÑ” 3 (Ğ°Ğ±Ğ¾ -3)

# Creating mask
# Ğ¡Ñ‚Ğ²Ğ¾Ñ€ĞµĞ½Ğ½Ñ� Ğ¼Ğ°Ñ�ĞºĞ¸
outliers_mask = (z_scores.abs() > 3).any(axis=1)

# Removing anomalous rows
# Ğ’Ğ¸Ğ´Ğ°Ğ»ĞµĞ½Ğ½Ñ� Ğ°Ğ½Ğ¾Ğ¼Ğ°Ğ»ÑŒĞ½Ğ¸Ñ… Ñ€Ñ�Ğ´ĞºÑ–Ğ²
data_cleaned = data[~outliers_mask]
target_cleaned = target[~outliers_mask]

print(f'Data before cleaning: {data.shape[0]} rows')
print(f'Data after cleaning: {data_cleaned.shape[0]} rows')


fig, ax = plt.subplots(figsize=(6, 5))

sns.scatterplot(
    data=data_cleaned,
    x='Longitude',
    y='Latitude',
    size=target_cleaned,
    hue=target_cleaned,
    palette='viridis',
    alpha=0.5,
    ax=ax)

plt.legend(
    title='MedHouseVal',
    bbox_to_anchor=(1.05, 0.95),
    loc='upper left')

plt.title('Median house value depending of\n their spatial location')


columns_drop = ['Longitude', 'Latitude']
subset = pd.concat([data_cleaned, target_cleaned], axis=1).drop(columns=columns_drop)

corr_mtx = subset.corr()

mask_mtx = np.zeros_like(corr_mtx)
np.fill_diagonal(mask_mtx, 1)

fig, ax = plt.subplots(figsize=(7, 6))

sns.heatmap(subset.corr(),
            cmap='coolwarm',
            center=0,
            annot=True,
            fmt='.2f',
            linewidth=0.5,
            square=True,
            mask=mask_mtx,
            ax=ax)


# Searching for pairs of features with high correlation (> 0.6)
# ĞŸĞ¾ÑˆÑƒĞº Ğ¿Ğ°Ñ€ Ğ¾Ğ·Ğ½Ğ°Ğº Ñ–Ğ· Ğ²Ğ¸Ñ�Ğ¾ĞºĞ¾Ñ� ĞºĞ¾Ñ€ĞµĞ»Ñ�Ñ†Ñ–Ñ”Ñ� (> 0.6)
high_corr_threshold = 0.6  # High correlation threshold / ĞŸĞ¾Ñ€Ñ–Ğ³ Ğ²Ğ¸Ñ�Ğ¾ĞºĞ¾Ñ— ĞºĞ¾Ñ€ĞµĞ»Ñ�Ñ†Ñ–Ñ—
high_corr_pairs = corr_mtx.unstack().sort_values(ascending=False)

for (col1, col2), corr_value in high_corr_pairs.items():
    if col1 != col2 and abs(corr_value) > high_corr_threshold:
        print(f'High correlation between {col1} Ñ‚Ğ° {col2}: {corr_value:.2f}')


# Removing the AveRooms feature
data_cleaned = data_cleaned.drop(columns=['AveRooms'])


X_train, X_test, y_train, y_test = train_test_split(
    data_cleaned,
    target_cleaned,
    test_size=0.2,
    random_state=42)


scaler = StandardScaler().set_output(transform='pandas').fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


X_train_scaled.describe()


model = LinearRegression().fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
y_train_pred = model.predict(X_train_scaled)

ymin, ymax = y_train.agg(['min', 'max']).values

y_pred = pd.Series(y_pred, index=X_test_scaled.index).clip(ymin, ymax)
y_pred.head()


# Calculating metrics on the training sample / Ğ�Ğ±Ñ‡Ğ¸Ñ�Ğ»ĞµĞ½Ğ½Ñ� Ğ¼ĞµÑ‚Ñ€Ğ¸Ğº Ğ½Ğ° Ñ‚Ñ€ĞµĞ½ÑƒĞ²Ğ°Ğ»ÑŒĞ½Ñ–Ğ¹ Ğ²Ğ¸Ğ±Ñ–Ñ€Ñ†Ñ–
r_sq_train = model.score(X_train_scaled, y_train)
mae_train = mean_absolute_error(y_train, y_train_pred)
mape_train = mean_absolute_percentage_error(y_train, y_train_pred)

print(f'Train Metrics:\nR2: {r_sq_train:.2f} | MAE: {mae_train:.2f} | MAPE: {mape_train:.2f}')


# Calculating metrics on the test sample / Ğ�Ğ±Ñ‡Ğ¸Ñ�Ğ»ĞµĞ½Ğ½Ñ� Ğ¼ĞµÑ‚Ñ€Ğ¸Ğº Ğ½Ğ° Ñ‚ĞµÑ�Ñ‚Ğ¾Ğ²Ñ–Ğ¹ Ğ²Ğ¸Ğ±Ñ–Ñ€Ñ†Ñ–
r_sq = model.score(X_test_scaled, y_test)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f'Test Metrics:\nR2: {r_sq:.2f} | MAE: {mae:.2f} | MAPE: {mape:.2f}')


poly = PolynomialFeatures(3).set_output(transform='pandas')

Xtr = poly.fit_transform(X_train_scaled)
Xts = poly.transform(X_test_scaled)

model_upd = LinearRegression().fit(Xtr, y_train)
y_pred_upd = model_upd.predict(Xts)
y_pred_upd = pd.Series(y_pred_upd, index=Xts.index).clip(ymin, ymax)
y_pred_train_upd = model_upd.predict(Xtr)
y_pred_train_upd = pd.Series(y_pred_train_upd, index=Xtr.index).clip(ymin, ymax)


# Calculating metrics on the training sample / Ğ�Ğ±Ñ‡Ğ¸Ñ�Ğ»ĞµĞ½Ğ½Ñ� Ğ¼ĞµÑ‚Ñ€Ğ¸Ğº Ğ½Ğ° Ñ‚Ñ€ĞµĞ½ÑƒĞ²Ğ°Ğ»ÑŒĞ½Ñ–Ğ¹ Ğ²Ğ¸Ğ±Ñ–Ñ€Ñ†Ñ–
r_sq_upd_train = model_upd.score(Xtr, y_train)
mae_upd_train = mean_absolute_error(y_train, y_pred_train_upd)
mape_upd_train = mean_absolute_percentage_error(y_train, y_pred_train_upd)

print(f'Train Metrics:\nR2: {r_sq_upd_train:.2f} | MAE: {mae_upd_train:.2f} | MAPE: {mape_upd_train:.2f}')


# Calculating metrics on the test sample / Ğ�Ğ±Ñ‡Ğ¸Ñ�Ğ»ĞµĞ½Ğ½Ñ� Ğ¼ĞµÑ‚Ñ€Ğ¸Ğº Ğ½Ğ° Ñ‚ĞµÑ�Ñ‚Ğ¾Ğ²Ñ–Ğ¹ Ğ²Ğ¸Ğ±Ñ–Ñ€Ñ†Ñ–
r_sq_upd = model_upd.score(Xtr, y_train)
mae_upd = mean_absolute_error(y_test, y_pred_upd)
mape_upd = mean_absolute_percentage_error(y_test, y_pred_upd)

print(f'Test Metrics:\nR2: {r_sq_upd:.2f} | MAE: {mae_upd:.2f} | MAPE: {mape_upd:.2f}')

