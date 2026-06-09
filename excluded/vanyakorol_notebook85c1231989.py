import pandas as pd
import numpy as np


# VIZ
import matplotlib.pylab as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


import warnings

pd.set_option('display.max_column', 200)
warnings.filterwarnings('ignore')
plt.style.use('ggplot')


train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


print(train.shape, test.shape)


train.head()


train.info()


for col in train.columns:
    print(f'Column: {col} has ---> missing values: {train[col].isnull().sum() / train[col].count()}%')


train.describe()


print(train['Podcast_Name'].nunique(), train['Episode_Title'].nunique())


cols2drop = ['Podcast_Name', 'id']

train = train.drop(cols2drop, axis=1)


train.head()


train['Episode_Title'] = train['Episode_Title'].apply(lambda x: int(x.split(" ")[1]))


train.info()


train.rename(columns={'Episode_Title': 'Episode_Number'}, inplace=True)


train.describe()


plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
sns.histplot(data = train, x = "Listening_Time_minutes",kde = True, bins=50)
plt.title('Listening Time minutes Distribution')
plt.subplot(1,2,2)
sns.boxplot(x=train["Listening_Time_minutes"])
plt.title('Listening Time minutes Distribution')
plt.tight_layout()
plt.show() 


# List of numerical features
num_col = [
    "Episode_Length_minutes",
    "Host_Popularity_percentage",
    "Guest_Popularity_percentage",
    "Number_of_Ads",
    "Listening_Time_minutes",
    "Episode_Number"
]

# Plot histograms and box plots for each numerical feature
for feature in num_col:
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    sns.histplot(train[feature], kde=True, bins=50)
    plt.title(f"Histogram of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")

    # Box plot to identify outliers
    plt.subplot(1, 2, 2)
    sns.boxplot(x=train[feature])
    plt.title(f"Box Plot of {feature}")

    plt.tight_layout()
    plt.show()

    # Print additional statistics
    print(f"\nStatistics for {feature}:")
    print(f"Skewness: {train[feature].skew():.2f}")


cat_col = [
    "Genre",
    "Publication_Day",
    "Publication_Time",
    'Episode_Sentiment'
]

for feature in cat_col:
    plt.figure(figsize=(20, 10))

    sns.countplot(
        x=train[feature], order=train[feature].value_counts().index
    )
    plt.title(f"Distribution of {feature}")

    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()

    # Print the number of unique values
    print(f"Number of Unique {feature}: {train[feature].nunique()}")


# Scatter plots for numerical features vs. Label
for feature in num_col[:-1]:  # Exclude Label itself
    plt.figure(figsize=(12, 8))
    if feature != 'Listening_Time_minutes':
        sns.scatterplot(
            x=train[feature], y=train["Listening_Time_minutes"], alpha=0.5
        )
        plt.title(f"{feature} vs. Listening_Time_minutes")
        plt.xlabel(feature)
        plt.ylabel("Listening_Time_minutes")
        plt.show()

# Correlation matrix for numerical features
correlation_matrix = train[num_col].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Numerical Features")
plt.show()


for feature in cat_col:
    plt.figure(figsize=(12, 8))

    
    sns.boxplot(x=train[feature], y=train["Listening_Time_minutes"])
    plt.title(f"{feature} vs. Listening_Time_minutes")
    plt.xlabel(feature)
    plt.ylabel("Listening_Time_minutes")
    plt.xticks(rotation=45)
    plt.show()


train.head()


train['Episode_Length_minutes'].fillna(train['Episode_Length_minutes'].median(), inplace=True)
train['Guest_Popularity_percentage'].fillna(train['Guest_Popularity_percentage'].median(), inplace=True)
train['Number_of_Ads'].fillna(train['Number_of_Ads'].median(), inplace=True)


for col in cat_col:
    print(col)
    if col in ['Genre', 'Publication_Day']:
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col])
    else:
        one_hot = pd.get_dummies(train[col])
        train = train.drop(col,axis = 1)
        train = train.join(one_hot)


train.info()


train_target = train.pop('Listening_Time_minutes')


X_train, X_valid, y_train, y_valid = train_test_split(train, train_target, test_size=0.25, random_state=42)


X_train.info()


model = LinearRegression()
model.fit(X_train, y_train)


feature_weight_df = pd.DataFrame(list(zip(model.feature_names_in_, model.coef_)))
feature_weight_df.columns = ['Feature', 'Weight']
feature_weight_df['abs_weight'] = np.abs(feature_weight_df['Weight'])


feature_weight_df.sort_values(by='abs_weight', ascending=True).head(25).plot.barh(x='Feature', y='Weight', figsize=(10, 10))


predicted = model.predict(X_valid)
actual = y_valid
prediction = pd.DataFrame(list(zip(predicted, actual)), columns=['Predicted', 'Actual'])


plt.figure(figsize=(15, 10))
plt.scatter(actual, predicted, alpha=0.5)
plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], color='blue', linestyle='--', linewidth=2)
plt.title('Predicted vs Actual Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.grid(True)
plt.show()


prediction.plot(style='.', figsize=(15, 5), linewidth=2)
plt.title('Predicted vs Actual Prices')
plt.xlabel('Index')
plt.ylabel('Price')
plt.legend(['Predicted', 'Actual'])
plt.show()


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(actual, predicted)
mse = mean_squared_error(actual, predicted)
rmse = np.sqrt(mse)
r2 = r2_score(actual, predicted)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R-squared (R²): {r2:.4f}")


residuals = actual - predicted
sns.histplot(residuals, kde=True)
plt.title("Distribution of Residuals")
plt.xlabel("Error")
plt.show()


rfr_model = RandomForestRegressor(n_estimators=100,
                               random_state=42,
                               n_jobs=-1,
                               verbose=1)


rfr_model


rfr_model.fit(X_train, y_train)


rfr_features = pd.DataFrame(list(zip(rfr_model.feature_names_in_, rfr_model.feature_importances_)), columns=['Feature', 'Importance']).sort_values('Importance', ascending=True)
rfr_features.plot.barh(x='Feature', y='Importance', edgecolor='k', figsize=(10, 6))


rfr_predicted = rfr_model.predict(X_valid)


mae = mean_absolute_error(y_valid, predicted)
mse = mean_squared_error(y_valid, predicted)
rmse = np.sqrt(mse)
r2 = r2_score(y_valid, predicted)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R^2 Score: {r2:.2f}")


plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_valid, y=predicted, alpha=0.3)
plt.xlabel("Actual Listening Time")
plt.ylabel("Predicted Listening Time")
plt.title("Actual vs Predicted Listening Time")
plt.plot([0, max(y_valid)], [0, max(y_valid)], color='blue', linestyle='--')  # Лінія ідеального передбачення
plt.show()


gb_model = GradientBoostingRegressor(n_estimators=100,
                                   random_state=42,
                                   verbose=1)


gb_model.fit(X_train, y_train)


gb_features = pd.DataFrame(list(zip(gb_model.feature_names_in_, gb_model.feature_importances_)), columns=['Feature', 'Importance']).sort_values('Importance', ascending=True)
gb_features.plot.barh(x='Feature', y='Importance', edgecolor='k', figsize=(10, 6))


predicted = gb_model.predict(X_valid)


mae = mean_absolute_error(y_valid, predicted)
mse = mean_squared_error(y_valid, predicted)
rmse = np.sqrt(mse)
r2 = r2_score(y_valid, predicted)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R^2 Score: {r2:.2f}")


plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_valid, y=predicted, alpha=0.3)
plt.xlabel("Actual Listening Time")
plt.ylabel("Predicted Listening Time")
plt.title("Actual vs Predicted Listening Time")
plt.plot([0, max(y_valid)], [0, max(y_valid)], color='blue', linestyle='--')  # Лінія ідеального передбачення
plt.show()


features = gb_features.merge(rfr_features, on='Feature', suffixes=('_gb', '_rfr'))


features.plot.barh(x='Feature', y=['Importance_rfr', 'Importance_gb'], figsize=(15, 8))
plt.title('Feature Importance Comparison')
plt.ylabel('Importance')
plt.xlabel('Feature')
plt.xticks(rotation=45, ha='right')
plt.legend(['Random Forest', 'Gradient Boosting'])
plt.tight_layout()
plt.show()




