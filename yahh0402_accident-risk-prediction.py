# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")


df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
df.head()


df = df.drop(columns=['id'])


df.shape


df.info()


df.duplicated().sum()


df.columns


# Identify the target variable
y = df['accident_risk']


# Display the data types of all columns
df.dtypes


# Get the summary statistics
df.describe()


# Check if any column contains null or missing values.
df.isnull().sum()


# Remove any duplicate rows if they exist
df = df.drop_duplicates()


# Convert incorrect data types (e.g., object → int/float/category)
object_col = df.select_dtypes(include=['object']).columns
for col in object_col:
    df[col] = df[col].astype('category')


# Ensure Boolean columns are converted to 0 and 1
bool_col = df.select_dtypes(include=['bool']).columns
for col in bool_col:
    df[col] = df[col].astype(int)


# Are there any outliers in numeric columns? If yes, identify them
num_col = df.select_dtypes(include=['int64','float64']).columns
outlier_info = {}

for col in num_col:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_info[col] = len(outliers)
    # Remove outliers (optional)
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

print("Outliers detected and removed per column:")
for col, count in outlier_info.items():
    print(f"{col}: {count}")

# Check the shape after outlier removal
print("\nNew dataset shape:", df.shape)


# Plot histograms for all numerical columns.
import matplotlib.pyplot as plt
num_col = df.select_dtypes(include=['int64','float64'])
num_col.hist(bins=20, figsize=(15, 10))
plt.suptitle("Histograms of Numerical Columns", fontsize=16)
plt.show()


# Create count plots for all categorical columns
cat_cols = df.select_dtypes(include=['category']).columns

plt.figure(figsize=(15, len(cat_cols) * 4))

for i, col in enumerate(cat_cols, 1):
    plt.subplot(len(cat_cols), 1, i)
    sns.countplot(x=col, data=df, hue=col, palette="viridis")
    plt.legend().set_visible(False)
    plt.title(f"Count Plot of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")

plt.tight_layout()
plt.show()


# Find the correlation between numerical features and visualize it with a heatmap.
plt.figure(figsize=(10, 6))
corr = df[num_col.columns].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Numerical Features")
plt.show()


# Most correlated features
correlations = corr["accident_risk"].sort_values(ascending=False)
print("Most positively correlated with accident_risk:\n", correlations.head(4), "\n")
print("Most negatively correlated with accident_risk:\n", correlations.tail(3), "\n")


# Analyze the impact of weather and lighting on accident risk using bar plots
weather_risk = df.groupby('weather', observed=True)['accident_risk'].mean().reset_index()
lighting_risk = df.groupby('lighting', observed=True)['accident_risk'].mean().reset_index()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(weather_risk['weather'], weather_risk['accident_risk'], color=sns.color_palette("cool", len(weather_risk)))
plt.title("Average Accident Risk by Weather")
plt.xlabel("Weather Condition")
plt.ylabel("Average Accident Risk")

plt.subplot(1, 2, 2)
plt.bar(lighting_risk['lighting'], lighting_risk['accident_risk'], color=sns.color_palette("magma", len(lighting_risk)))
plt.title("Average Accident Risk by Lighting")
plt.xlabel("Lighting Condition")
plt.ylabel("Average Accident Risk")

plt.tight_layout()
plt.show()


# Is accident risk higher during holidays or school seasons? (Use groupby/mean)
print("Average accident risk during holidays:\n", df.groupby('holiday')['accident_risk'].mean(), "\n")
print("Average accident risk during school season:\n", df.groupby('school_season')['accident_risk'].mean(), "\n")


# Does speed_limit affect accident risk significantly
plt.figure(figsize=(8, 5))
sns.scatterplot(x='speed_limit', y='accident_risk', data=df, alpha=0.6)
sns.regplot(x='speed_limit', y='accident_risk', data=df, scatter=False, color='red')
plt.title("Speed Limit vs Accident Risk")
plt.show()


# average accident risk by road_type
road_risk = df.groupby('road_type', observed=True)['accident_risk'].mean().sort_values(ascending=False)
print("Average accident risk by road type:\n", road_risk, "\n")

plt.figure(figsize=(8, 5))
sns.barplot(x=road_risk.index, y=road_risk.values, hue=road_risk.index, palette="cubehelix")
plt.title("Average Accident Risk by Road Type")
plt.ylabel("Average Accident Risk")
plt.xlabel("Road Type")
plt.legend().set_visible(False)
plt.show()


# Apply label encoding or one-hot encoding to categorical variables
from sklearn.preprocessing import OneHotEncoder, StandardScaler

cat_col = df.select_dtypes(include=['category', 'object']).columns

df = pd.get_dummies(df, columns=cat_col, dtype=int, drop_first=True)


# Normalize or standardize numeric columns if needed
num_col = df.select_dtypes(include=['int64','float64']).columns.drop('accident_risk')

scaler = StandardScaler()
df[num_col] = scaler.fit_transform(df[num_col])


# Create a new feature: risk_per_lane = num_reported_accidents / num_lanes
df['risk_per_line'] = df['num_reported_accidents'] / (df['num_lanes'] + 1e-6)
df['accidents_per_speed'] = df['num_reported_accidents'] / (df['speed_limit'] + 1e-6)
df['curvature_speed_ratio'] = df['curvature'] * df['speed_limit']
df['lanes_speed_ratio'] = df['num_lanes'] * df['speed_limit']
df['holiday_school'] = df['holiday'] * df['school_season']


# Display the final prepared dataframe before modeling
df.head()


# Split the dataset into features (X) and target (y)
x = df.drop('accident_risk', axis=1)
y = df['accident_risk']


# Perform a train-test split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)


# Model train
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=42)
model.fit(x_train,y_train)


from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(random_state=42)
gbr.fit(x_train, y_train)


y_predrf = model.predict(x_test)
y_predgbr = gbr.predict(x_test)


results = {
    "Model": ["Random Forest", "Gradient Boosting"],
    "MAE": [
        mean_absolute_error(y_test, y_predrf),
        mean_absolute_error(y_test, y_predgbr)
    ],
    "MSE": [
        mean_squared_error(y_test, y_predrf),
        mean_squared_error(y_test, y_predgbr)
    ],
    "RMSE": [
        np.sqrt(mean_squared_error(y_test, y_predrf)),
        np.sqrt(mean_squared_error(y_test, y_predgbr))
    ],
    "R² Score": [
        r2_score(y_test, y_predrf),
        r2_score(y_test, y_predgbr)
    ]
}

results_df = pd.DataFrame(results)
print(results_df)



# Feature importance plot
rf_importances = model.feature_importances_
gbr_importances = gbr.feature_importances_
features = x.columns

sorted_idx_rf = np.argsort(rf_importances)
sorted_idx_gbr = np.argsort(gbr_importances)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.barplot(x=rf_importances[sorted_idx_rf], y=features[sorted_idx_rf], palette="viridis")
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Features")

plt.subplot(1, 2, 2)
sns.barplot(x=gbr_importances[sorted_idx_gbr], y=features[sorted_idx_gbr], palette="magma")
plt.title("Feature Importance - Gradient Boosting")
plt.xlabel("Importance Score")
plt.ylabel("")

plt.tight_layout()
plt.show()


model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation='relu'),
    BatchNormalization(),

    Dense(1, activation='linear')
])


optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])


early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)


history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=150,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)


y_predann = model.predict(X_test)


# Train the model and check performance metrics
print("r2_score",r2_score(y_test,y_predann))
print("mean_absolute_error",mean_absolute_error(y_test,y_predann))
print("mean_squared_error",mean_squared_error(y_test,y_predann))
print("root_mean_squared_error",np.sqrt(mean_squared_error(y_test,y_predann)))


# Plot training and validation loss and MAE
plt.figure(figsize=(12, 5))

# --- Plot Loss (MSE) ---
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Model Loss (MSE)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# --- Plot MAE ---
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE', color='green')
plt.plot(history.history['val_mae'], label='Validation MAE', color='red')
plt.title('Model Mean Absolute Error (MAE)')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.show()





