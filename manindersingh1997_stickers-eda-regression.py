import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


train_data = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
sub_data = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')


train_data.head()


train_data.info()



non_null_count = train_data.notnull().sum()
is_null_count = train_data.isnull().sum()
print("Non-Null count is :\n",non_null_count)
print("\n Null count is :\n",is_null_count)


sns.heatmap(train_data.isnull(),cbar = False, cmap = 'viridis')
plt.show()


plt.figure(figsize=(10, 5))
sns.histplot(train_data['num_sold'], kde=True)
plt.xlim(0, 500)
plt.title("Histogram of 'num_sold' (0-100)")
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(train_data['num_sold'], kde=True)

plt.title("Histogram of 'num_sold' (Full Range)")
plt.show()


train_data = train_data.dropna(subset=['num_sold'])

train_data['flag'] = np.where((train_data['num_sold'] > 0) & (train_data['num_sold'] <= 100), 100,
                    np.where((train_data['num_sold'] > 100) & (train_data['num_sold'] <= 1000), 1000,
                    np.where(train_data['num_sold'] > 1000, 
                             ((train_data['num_sold'] // 1000).astype(int) + 1) * 1000, 
                             None)))
bucket_distribution = train_data.groupby('flag')['num_sold'].count()

print(bucket_distribution)


categorical_columns = ['country', 'store', 'product']

for col in categorical_columns:
    plt.figure(figsize=(10, 6))
    sns.catplot(
        data=train_data,
        x=col,
        y="num_sold",
        kind="boxen",
        aspect=1.5,
        palette="Set2"
    )
    plt.xlabel(col, fontsize=12)
    plt.ylabel("num_sold", fontsize=12)
    plt.title(f"Distribution of num_sold by {col}", fontsize=15)
    plt.xticks(rotation=45)
    plt.show()



cat_cols = ["country", "store", "product"]

def plot_categorical_column(dataframe, column):
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x=column, data=dataframe, palette="Set1")
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.title(f'Distribution of {column}')
    plt.xticks(rotation=45)
    
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.0f'),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 10), textcoords='offset points')
    
    plt.tight_layout()
    plt.show()

for column in cat_cols:
    plot_categorical_column(train_data, column)



train_data['date'] = pd.to_datetime(train_data['date'], errors='coerce')
train_data = train_data.dropna(subset=['date'])
train_data['month_year'] = pd.to_datetime(train_data['date'].dt.strftime('%Y-%m') + '-01')
train_data['year'] = train_data['month_year'].dt.year

grouped_data = train_data.groupby(['product', 'month_year'])['num_sold'].sum().reset_index()
plt.figure(figsize=(12, 8))
sns.lineplot(data=grouped_data, x='month_year', y='num_sold', hue='product', marker='o')

plt.title('Product-wise Trends: num_sold over Year (Monthly Data)', fontsize=15)
plt.xlabel('Year', fontsize=12)
plt.ylabel('num_sold', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Product', fontsize=10)
plt.tight_layout()
plt.show()



test_data['date'] = pd.to_datetime(test_data['date'], errors='coerce')
test_data = test_data.dropna(subset=['date'])
test_data['month_year'] = pd.to_datetime(test_data['date'].dt.strftime('%Y-%m') + '-01')
test_data['year'] = test_data['month_year'].dt.year
test_data['date'] = pd.to_datetime(test_data['date'], errors='coerce')
test_data = test_data.dropna(subset=['date'])
test_data['day'] = test_data['date'].dt.day
test_data['month'] = test_data['date'].dt.month
test_data['year'] = test_data['date'].dt.year
test_data['day_of_week'] = test_data['date'].dt.dayofweek  # Day of the week (0=Monday, 6=Sunday)




train_data['date'] = pd.to_datetime(train_data['date'], errors='coerce')
train_data = train_data.dropna(subset=['date'])
train_data['day'] = train_data['date'].dt.day
train_data['month'] = train_data['date'].dt.month
train_data['year'] = train_data['date'].dt.year
train_data['day_of_week'] = train_data['date'].dt.dayofweek  # Day of the week (0=Monday, 6=Sunday)




train_data.info()



train_data_enc = pd.get_dummies(train_data, columns=['country', 'store', 'product', 'day_of_week'], drop_first=True)
train_data_enc = train_data_enc.drop(columns=['date','flag'])

f, ax = plt.subplots(figsize = (10,8))
corr = train_data_enc.corr()
sns.heatmap(corr, 
            mask = np.zeros_like(corr, dtype = bool),
            cmap = sns.diverging_palette(240,10, as_cmap = True),
            square = True, ax = ax
           )


train_data_enc.info()


train_data = train_data.drop(['flag'], axis=1)
sns.pairplot(train_data)

plt.suptitle('Pair Plot of All Numerical Features', y=1.02, fontsize=16)
plt.tight_layout()
plt.show()


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.ensemble import RandomForestRegressor


train_data_enc.head()


test_data['date'] = pd.to_datetime(test_data['date'], errors='coerce')
test_data = test_data.dropna(subset=['date'])
test_data['month_year'] = pd.to_datetime(test_data['date'].dt.strftime('%Y-%m') + '-01')
test_data['year'] = test_data['month_year'].dt.year
test_data['date'] = pd.to_datetime(test_data['date'], errors='coerce')
test_data = test_data.dropna(subset=['date'])
test_data['day'] = test_data['date'].dt.day
test_data['month'] = test_data['date'].dt.month
test_data['year'] = test_data['date'].dt.year
test_data['day_of_week'] = test_data['date'].dt.dayofweek  # Day of the week (0=Monday, 6=Sunday)


test_data_enc = pd.get_dummies(test_data, columns=['country', 'store', 'product', 'day_of_week'], drop_first=True)
test_data_enc = test_data_enc.drop(columns=['date'])

print(test_data_enc.head())



x = train_data_enc.drop(['num_sold', 'month_year'], axis=1)
y = train_data_enc['num_sold']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
numerical_cols = x.select_dtypes(include=['float64', 'int64']).columns

scaler = StandardScaler()
x_train[numerical_cols] = scaler.fit_transform(x_train[numerical_cols])
x_test[numerical_cols] = scaler.transform(x_test[numerical_cols])

lr = LinearRegression().fit(x_train, y_train)

y_train_predict = lr.predict(x_train)
y_test_predict = lr.predict(x_test)

train_r2_score = lr.score(x_train, y_train)
print(f"Training R² Score: {train_r2_score}")

test_r2_score = lr.score(x_test, y_test)
print(f"Test R² Score: {test_r2_score}")


X = train_data_enc.drop(['num_sold','month_year'], axis = 1)
Y = train_data_enc['num_sold']

quad = PolynomialFeatures(degree = 3)
x_quad = quad.fit_transform(X)
X_train,X_test,Y_train,Y_test = train_test_split(x_quad,Y, random_state = 0)
plr = LinearRegression().fit(X_train,Y_train)

Y_train_pred = plr.predict(X_train)
Y_test_pred = plr.predict(X_test)

print("R-squared score on the test data: ", plr.score(X_test, Y_test))

plt.figure(figsize=(10, 6))
plt.scatter(Y_test, Y_test_pred, color='blue', label='Predicted vs Actual')
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='red', lw=2, label='Perfect fit')
plt.xlabel('Actual Trip Price')
plt.ylabel('Predicted Trip Price')
plt.title('Polynomial Regression: Actual vs Predicted')
plt.legend()
plt.show()


from sklearn.metrics import r2_score, mean_squared_error

forest = RandomForestRegressor(n_estimators=100, random_state=42)
forest.fit(x_train, y_train)

forest_train_pred = forest.predict(x_train)
forest_test_pred = forest.predict(x_test)

# R²
train_r2 = r2_score(y_train, forest_train_pred)
test_r2 = r2_score(y_test, forest_test_pred)

n_train = len(y_train)
n_test = len(y_test)
p = x_train.shape[1]

# Adjusted R²
adjusted_train_r2 = 1 - ((1 - train_r2) * (n_train - 1)) / (n_train - p - 1)
adjusted_test_r2 = 1 - ((1 - test_r2) * (n_test - 1)) / (n_test - p - 1)

print('MSE train data: %.3f, MSE test data: %.3f' % (
    mean_squared_error(y_train, forest_train_pred, squared=False),
    mean_squared_error(y_test, forest_test_pred, squared=False)
))

print('R2 train data: %.3f, R2 test data: %.3f' % (
    train_r2,
    test_r2
))

print('Adjusted R² train data: %.3f, Adjusted R² test data: %.3f' % (
    adjusted_train_r2,
    adjusted_test_r2
))



train_r2_scores = [
    r2_score(y_train, forest_train_pred),  # Random Forest
    r2_score(Y_train, Y_train_pred),        # Polynomial 
    r2_score(y_train, y_train_predict)      # Linear Regression
]

models = ['Random Forest', 'Polynomial Regression', 'Linear Regression']

fig, ax = plt.subplots(figsize=(10, 6))

x_axis = range(len(models))
ax.bar(x_axis, train_r2_scores, width=0.4, label='Train R²', align='center')

ax.set_xlabel('Models')
ax.set_ylabel('R² Score')
ax.set_title('R² Score Comparison Across Models')
ax.set_xticks(x_axis)
ax.set_xticklabels(models)
ax.legend()

plt.show()


train_data_enc.info()


test_data_enc.info()


test_data_enc = test_data_enc.drop(['month_year'], axis=1)


test_predictions = forest.predict(test_data_enc)
sub_data['num_sold'] = test_predictions
sub_data.to_csv('sub_data.csv', index=False)

print("Submission file saved as 'submission.csv'.")

