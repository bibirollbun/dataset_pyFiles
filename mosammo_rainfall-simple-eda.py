import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import yeojohnson
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


warnings.filterwarnings("ignore", category=FutureWarning)


def interpret_thi(thi):
    if thi < 75:
        return "No heat stress"
    elif 75 <= thi < 79:
        return "Mild heat stress"
    elif 79 <= thi < 84:
        return "Moderate heat stress"
    else:
        return "Severe heat stress"


def separate_skewed_features(df):
    # Initialize lists to store column names
    positive_skew_features = []
    negative_skew_features = []

    # Calculate skewness for each column
    for col in df.columns:
        skewness = skew(df[col])
        if skewness > 0:
            positive_skew_features.append(col)
        elif skewness < 0:
            negative_skew_features.append(col)

    return positive_skew_features, negative_skew_features


def outlier_cut(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    return np.where(df > upper_bound, upper_bound, np.where(df < lower_bound, lower_bound, df))


def bar_plot(df, column, hue=None, size=(6, 6), title=None):
    if not title:
        title = f'Distribution of {column}'
    plt.figure(figsize=size)
    ax = sns.countplot(data=df, x=column, hue=hue)
    total = len(df)
    for p in ax.patches:
        height = p.get_height()
        percentage = f'{100 * height / total:.1f}%'
        ax.text(p.get_x() + p.get_width() / 2, height / 2, percentage, ha='center', va='center', fontsize=10, color='white')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.title(title)
    plt.show()


data = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
data.columns = data.columns.str.replace(' ', '', regex=True)
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data['rainfall'] = data['rainfall'].replace({1: 'yes', 0: 'no'})
data.head()


data.info()


data.describe()


data.isnull().sum()


data['winddirection'] = data['winddirection'].fillna(data['winddirection'].mean())
data['windspeed'] = data['windspeed'].fillna(data['windspeed'].mean())


data.isnull().sum()


data['month'] = pd.cut(
    data.index,
    bins=[0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365],
    labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
)


season_map = {
    1: 'Winter',
    2: 'Winter',
    3: 'Spring',
    4: 'Spring',
    5: 'Spring',
    6: 'Summer',
    7: 'Summer',
    8: 'Summer',
    9: 'Fall',
    10: 'Fall',
    11: 'Fall',
    12: 'Winter'
}
data['season'] = data['month'].map(season_map)


data['temperature_range'] = data['maxtemp'] - data['mintemp']
data['mean_temperature'] = (data['maxtemp'] + data['mintemp']) / 2
data['dewpoint_depression'] = data['temparature'] - data['dewpoint']
data['rolling_mean_temperature'] = data['temparature'].rolling(window=7).mean()


data['THI'] = ((data['temparature'] * 9/5 + 32) - (0.55 - 0.55 * data['humidity']) * ((data['temparature'] * 9/5 + 32) - 58)).apply(interpret_thi)


data['wind_direction_category'] = pd.cut(data['winddirection'], bins=[0, 45, 90, 135, 180, 225, 270, 315, 360], labels=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])


data = data.bfill()


bar_plot(data, 'rainfall', size=(12,5))


numerical_columns = data.select_dtypes(include=['number']).drop(columns=['day']).columns
categorical_columns = data.select_dtypes(exclude=['number']).drop(columns=['month', 'season', 'rainfall']).columns


for column in categorical_columns:
    bar_plot(data, column, size=(12,5))


for column in numerical_columns:
    plt.figure(figsize=(12, 4))  # Create a new figure for each feature

    # Subplot 1: KDE Plot
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
    sns.kdeplot(np.log(data[column]), fill=True)
    plt.title(f'KDE Plot of {column}')
    plt.xlabel(column)
    plt.ylabel('Density')

    # Subplot 2: Box Plot
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
    sns.boxplot(x=data[column])
    plt.title(f'Box Plot of {column}')
    plt.xlabel(column)

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()  # Display the plot


for feat in numerical_columns:
    data[feat], _ = yeojohnson(data[feat] + 1)  


for column in categorical_columns:
    bar_plot(data, column, hue='rainfall', size=(12,5))


true_data = data[data['rainfall'] == 'yes']
false_data = data[data['rainfall'] == 'no']


for column in numerical_columns:
    plt.figure(figsize=(10, 4))  # Create a new figure for each feature

    # Subplot 1: KDE Plot
    sns.kdeplot(true_data[column], color='red', fill=True)
    sns.kdeplot(false_data[column], color='blue', fill=True)
    plt.legend(['yes', 'no'], loc='upper right')
    plt.title(f'KDE Plot of {column}')
    plt.xlabel(column)
    plt.ylabel('Density')

    plt.show()  # Display the plot


data = data.replace({'yes': 1, 'no': 0})


data = pd.get_dummies(data, dtype=int)


X = data.drop(columns=['rainfall', 'id', 'day'])
y = data['rainfall']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)


# Train Random Forest model
model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_resampled, y_resampled)

# Predict on val set
y_pred = model.predict(X_val)

# Evaluate
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)

print(f"Random Forest - MAE: {mae}, MSE: {mse}, RMSE: {rmse}")


importances = model.feature_importances_
feature_names = X.columns

sorted_indices = np.argsort(importances)[::-1]  
sorted_importances = importances[sorted_indices]
sorted_feature_names = feature_names[sorted_indices]

plt.figure(figsize=(10, 6))
plt.barh(sorted_feature_names, sorted_importances)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance from Random Forest (Sorted)')
plt.gca().invert_yaxis()
plt.show()


test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


test.columns = test.columns.str.replace(' ', '', regex=True)
test.replace([np.inf, -np.inf], np.nan, inplace=True)


test['winddirection'] = test['winddirection'].fillna(test['winddirection'].mean())


test['month'] = pd.cut(
    test.index,
    bins=[0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365],
    labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
)


season_map = {
    1: 'Winter',
    2: 'Winter',
    3: 'Spring',
    4: 'Spring',
    5: 'Spring',
    6: 'Summer',
    7: 'Summer',
    8: 'Summer',
    9: 'Fall',
    10: 'Fall',
    11: 'Fall',
    12: 'Winter'
}
test['season'] = test['month'].map(season_map)


test['temperature_range'] = test['maxtemp'] - test['mintemp']
test['mean_temperature'] = (test['maxtemp'] + test['mintemp']) / 2
test['dewpoint_depression'] = test['temparature'] - test['dewpoint']
test['rolling_mean_temperature'] = test['temparature'].rolling(window=7).mean()


test['THI'] = ((test['temparature'] * 9/5 + 32) - (0.55 - 0.55 * test['humidity']) * ((test['temparature'] * 9/5 + 32) - 58)).apply(interpret_thi)


test['wind_direction_category'] = pd.cut(test['winddirection'], bins=[0, 45, 90, 135, 180, 225, 270, 315, 360], labels=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])


test = test.bfill()


for feat in numerical_columns:
    test[feat], _ = yeojohnson(test[feat] + 1)  


test = pd.get_dummies(test, dtype=int)


test = test.reindex(columns=X_train.columns, fill_value=0)


pred = model.predict(test)


sub = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
sub['rainfall'] = np.round(pred)
sub.to_csv('sub_file.csv', index=False)




