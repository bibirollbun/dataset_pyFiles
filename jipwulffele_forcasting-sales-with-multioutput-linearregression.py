# import the basis
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt



df_train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv', index_col='date')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv', index_col='date')

#print(df_train.head())
#print(df_test.head())


# print the column names of the training data
print('The column names are: {}'.format(df_train.columns.tolist()))

# print the unique entries in the country column
print('The "country" column contains the following categories: {}'.format(df_train['country'].unique()))

# print the unique entries in the store column
print('The "store" column contains the following categories: {}'.format(df_train['store'].unique()))

# print the unique entries in the product column
print('The "product" column contains the following categories: {}'.format(df_train['product'].unique()))


# Check the distributions of the different categories

def plot_pivot_table(ind_name, col_name, val_name):
    
    # Pivot the data to create a matrix for the stacked bar plot
    pivot_table = df_train.pivot_table(index=ind_name, columns=col_name, values=val_name, aggfunc='sum')
    
    # Plotting
    pivot_table.plot(kind='bar', stacked=True, figsize=(10, 6))
    
    # Customizing the plot
    plt.title('Sticker sales by {0} and {1}'. format(ind_name, col_name))
    plt.ylabel('Sticker sales')
    plt.tight_layout()
    
    # Show the plot
    plt.show()

plot_pivot_table('country', 'product', 'num_sold')
plot_pivot_table('country', 'store', 'num_sold')
plot_pivot_table('store', 'product', 'num_sold')


# Reorganize the data pivot the data

# Pivot the DataFrame
pivoted_df = df_train.pivot_table(
    index=df_train.index,
    columns=['country', 'product', 'store'],
    values='num_sold'
)

# Flatten MultiIndex columns and join them with underscores
pivoted_df.columns = ['_'.join(col).strip() for col in pivoted_df.columns.values]

# save date information
date_list = pivoted_df.index
# Reset index because otherwide the plotting is causing errors
pivoted_df.reset_index(inplace=True)

#print(pivoted_df.head())


# look for missing values
total_missing = pivoted_df.isnull().sum().sum()
print("\nTotal missing values in DataFrame:", total_missing)

# Total missing values per column
missing_per_column = pivoted_df.isnull().sum()
columns_with_missing = missing_per_column[missing_per_column > 0]

print("Missing values per column:")
print([columns_with_missing])


# visualize missing values with mnso
import missingno as msno 

pivoted_df_with_missing = pivoted_df[columns_with_missing.index]

msno.matrix(pivoted_df_with_missing)


# Plot the sales 

fig, axes = plt.subplots(7,1, figsize=(12, 24))

j = 0

for col in pivoted_df_with_missing.columns:
    axes[j].plot(pivoted_df_with_missing.index, pivoted_df_with_missing[col], label=col)
    axes[j].set_title(col)
    j+=1

# Show the plot
plt.show()


# fill the missing values
pivoted_df_with_missing.fillna(method='ffill', inplace=True)

fig, axes = plt.subplots(7,1, figsize=(12, 24))

j = 0

for col in pivoted_df_with_missing.columns:
    axes[j].plot(pivoted_df_with_missing.index, pivoted_df_with_missing[col], label=col)
    axes[j].set_title(col)
    j+=1

# Show the plot
plt.show()


# fill missing values in the full dataset

pivoted_df.fillna(method='ffill', inplace=True)
total_missing = pivoted_df.isnull().sum().sum()
print("\nTotal missing values in DataFrame:", total_missing)


# Have a look at the test data
# What do we need to predict?
# 2 years of sales for all combinations of store, product and country?

# Pivot the DataFrame
df_test['num_sold'] = 0

pivoted_test_df = df_test.pivot_table(
    index=df_test.index,
    columns=['country', 'product', 'store'],
    values = ['num_sold']
)

# Flatten MultiIndex columns and join them with underscores
pivoted_test_df.columns = ['_'.join(col).strip() for col in pivoted_test_df.columns.values]



import math

fig, axes = plt.subplots(math.ceil(pivoted_df.shape[1]/2), 2, figsize=(12, 164))

i = 0
j = 0

for col in pivoted_df.columns:
    axes[i, j].plot(pivoted_df[col], label=col)
    axes[i, j].set_title(col)
    if j < 1:
        j+=1
    else:
        i+=1
        j=0

# Show the plot
plt.show()


# there are also weekly cycles for some stickers (weekends!)


from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Pivot the DataFrame
pivoted_df = df_train.pivot_table(
    index=df_train.index,
    columns=['country', 'product', 'store'],
    values='num_sold'
)

# Flatten MultiIndex columns and join them with underscores
pivoted_df.columns = ['_'.join(col).strip() for col in pivoted_df.columns.values]
pivoted_df.index = pd.to_datetime(pivoted_df.index)

# fill missing values
pivoted_df.fillna(method='ffill', inplace=True)

y = pivoted_df.copy()

y = y.reindex(pd.date_range(start=y.index.min(), end=y.index.max(), freq='D'))
y.fillna(0, inplace=True) 

fourier = CalendarFourier(freq="A", order=1)  # 10 sin/cos pairs for "A"nnual seasonality
# Create trend features
dp = DeterministicProcess(
    index=y.index,  # dates from the training data
    constant=True,  # the intercept
    order=2,        # trend
    seasonal=True, # weekly seasonality (indicators)
    additional_terms=[fourier], # anual seasonality (fourier)
    drop=True,      # drop terms to avoid collinearity
)
X = dp.in_sample()  # features for the training data

#split of the last 500 days as test data
idx_train, idx_test = train_test_split(
    y.index, test_size=500, shuffle=False,
)
X_train, X_test = X.loc[idx_train, :], X.loc[idx_test, :]
y_train, y_test = y.loc[idx_train], y.loc[idx_test]

# Fit trend model
model = LinearRegression(fit_intercept=False)

model.fit(X_train, y_train)

# Make predictions
y_fit = pd.DataFrame(
    model.predict(X_train),
    index=y_train.index,
    columns=y_train.columns,
)
y_pred = pd.DataFrame(
    model.predict(X_test),
    index=y_test.index,
    columns=y_test.columns,
)


# visulaize fit

fig, axes = plt.subplots(math.ceil(pivoted_df.shape[1]/2), 2, figsize=(12, 164))

i = 0
j = 0

for col in pivoted_df.columns:
    axes[i, j].plot(pivoted_df[col], label=col)
    axes[i, j].plot(y_fit[col], label=col)
    axes[i, j].plot(y_pred[col], label=col)
    axes[i, j].set_title(col)
    if j < 1:
        j+=1
    else:
        i+=1
        j=0

# Show the plot
plt.show()



# visulaize fit: zoom in week

fig, axes = plt.subplots(math.ceil(pivoted_df.shape[1]/2), 2, figsize=(12, 164))

i = 0
j = 0

for col in pivoted_df.columns:
    axes[i, j].plot(pivoted_df[col], label=col)
    axes[i, j].plot(y_fit[col], label=col)
    axes[i, j].plot(y_pred[col], label=col)
    axes[i, j].set_title(col)
    axes[i, j].set_xlim((pd.to_datetime('2010-01-01'), pd.to_datetime('2010-02-01') ))

    if j < 1:
        j+=1
    else:
        i+=1
        j=0

# Show the plot
plt.show()



# train the model on the full data set

model.fit(X, y)

# Make predictions
y_fit = pd.DataFrame(
    model.predict(X),
    index=y.index,
    columns=y.columns,
)


# visulaize fit

fig, axes = plt.subplots(math.ceil(pivoted_df.shape[1]/2), 2, figsize=(12, 164))

i = 0
j = 0

for col in pivoted_df.columns:
    axes[i, j].plot(pivoted_df[col], label=col)
    axes[i, j].plot(y_fit[col], label=col)
    axes[i, j].set_title(col)
    if j < 1:
        j+=1
    else:
        i+=1
        j=0

# Show the plot
plt.show()



# make predictions on the test data and save for submission to the competition

# Pivot the DataFrame
df_test['num_sold'] = 0

pivoted_test_df = df_test.pivot_table(
    index=df_test.index,
    columns=['country', 'product', 'store'],
    values = 'num_sold'
)

# Flatten MultiIndex columns and join them with underscores
pivoted_test_df.columns = ['_'.join(col).strip() for col in pivoted_test_df.columns.values]
pivoted_test_df.index = pd.to_datetime(pivoted_test_df.index)

# make X
fourier = CalendarFourier(freq="A", order=1)  # 10 sin/cos pairs for "A"nnual seasonality
# Create trend features
dp_test = DeterministicProcess(
    index=pivoted_test_df.index,  # dates from the training data
    constant=True,  # the intercept
    order=2,        # trend
    seasonal=True, # weekly seasonality (indicators)
    additional_terms=[fourier], # anual seasonality (fourier)
    drop=True,      # drop terms to avoid collinearity
)

X_test = dp_test.in_sample()  # features for the training data

y_test = pd.DataFrame(
    model.predict(X_test),
    index=pivoted_test_df.index,
    columns=y.columns,
)


# make predictions on the test data and save for submission to the competition

# Pivot the DataFrame
df_test['num_sold'] = 0

pivoted_test_df = df_test.pivot_table(
    index=df_test.index,
    columns=['country', 'product', 'store'],
    values = 'num_sold'
)

# Flatten MultiIndex columns and join them with underscores
pivoted_test_df.columns = ['_'.join(col).strip() for col in pivoted_test_df.columns.values]
pivoted_test_df.index = pd.to_datetime(pivoted_test_df.index)

X_test = dp.out_of_sample(steps=366*3)  # features for the training data

y_test = pd.DataFrame(
    model.predict(X_test),
    index=X_test.index,
    columns=y.columns,
)


# visulaize the forcast

fig, axes = plt.subplots(math.ceil(pivoted_df.shape[1]/2), 2, figsize=(12, 164))

i = 0
j = 0

for col in pivoted_df.columns:
    axes[i, j].plot(pivoted_df[col], label=col)
    axes[i, j].plot(y_fit[col], label=col)
    axes[i, j].plot(y_test[col], label=col)
    axes[i, j].set_title(col)
    if j < 1:
        j+=1
    else:
        i+=1
        j=0

# Show the plot
plt.show()



# now put the data in the correct format for submission

# unpivot y_test
y_test.columns = y_test.columns.str.split('_', expand=True)
unpivoted_df = y_test.stack(level=[0, 1, 2]).reset_index()
unpivoted_df.columns = ['date', 'country', 'product', 'store', 'num_sold']
unpivoted_df.set_index('date', inplace=True)
print(unpivoted_df.tail())


# merge the forecast and the df_test

# Create a key for matching rows in the unpivoted df with the forecast
unpivoted_df['key'] = list(zip(unpivoted_df.index, 
                               unpivoted_df['country'], 
                               unpivoted_df['product'],  
                               unpivoted_df['store']))

# Create the same key in the original test data
df_test.index = pd.to_datetime(df_test.index)
df_test['key'] = list(zip(df_test.index, 
                               df_test['country'], 
                               df_test['product'],  
                               df_test['store']))


# Map numeric values from df2 to df1 using the key
df_test['num_sold'] = df_test['key'].map(unpivoted_df.set_index('key')['num_sold'])

# Drop the temporary key column
df_test.drop(columns=['key'], inplace=True)


# there are some nan values in the forecast because some combinations of product-store and country do not exist in the test data!

df_na = df_test[df_test.isna().any(axis=1)]
print(df_na.head())
print(df_na['product'].unique())
print(df_na['country'].unique())
print(df_na['store'].unique())


# pretty ugly solution

df_test.loc[(df_test['country'] == 'Canada') & (df_test['product'] == 'Holographic Goose'), 'store'] = 'Stickers for Less'
df_test.loc[(df_test['country'] == 'Kenya') & (df_test['product'] == 'Holographic Goose'), 'store'] = 'Stickers for Less'

# Create a key for matching rows in the unpivoted df with the forecast
unpivoted_df['key'] = list(zip(unpivoted_df.index, 
                               unpivoted_df['country'], 
                               unpivoted_df['product'],  
                               unpivoted_df['store']))

# Create the same key in the original test data
df_test.index = pd.to_datetime(df_test.index)
df_test['key'] = list(zip(df_test.index, 
                               df_test['country'], 
                               df_test['product'],  
                               df_test['store']))


# Map numeric values from df2 to df1 using the key
df_test['num_sold'] = df_test['key'].map(unpivoted_df.set_index('key')['num_sold'])

# Drop the temporary key column
df_test.drop(columns=['key'], inplace=True)


# check there are no more missing values

df_na = df_test[df_test.isna().any(axis=1)]
print(df_na.head())
print(df_na['product'].unique())
print(df_na['country'].unique())
print(df_na['store'].unique())


# prepare for submission

y_submit = df_test[['id', 'num_sold']]
y_submit.to_csv('submission.csv', index=False)

