!pip install sweetviz


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sweetviz as sv

from IPython.display import IFrame


# Load the datasets
train_df = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")

# Display basic info for train_df
print("train_df:\n")
print(train_df.info())
print(train_df.describe())
print()
print(train_df.isna().sum())
print()

# Display basic info for test_df
print("test_df:\n")
print(test_df.info())
print(test_df.describe())
print()
print(test_df.isna().sum())
print()


# Categorical Features
print(train_df['country'].value_counts())
print()
print(train_df['store'].value_counts())
print()
print(train_df['product'].value_counts())


# Create the sweetviz report
report = sv.analyze(train_df)
report.show_html('train_report.html')


# Display the report
IFrame('train_report.html', width=900, height=600)


# Convert date to datetime
train_df['date'] = pd.to_datetime(train_df['date'])
test_df['date'] = pd.to_datetime(test_df['date'])

# Extract features from the date
for df in [train_df, test_df]:
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday

# Add seasonal features
for df in [train_df, test_df]:
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)


# Plot time series of num_sold
train_df.groupby('date')['num_sold'].sum().plot(figsize=(28, 6))
plt.title('Time Series of num_sold')
plt.show()


# Group by country, store, and product
grouped = train_df.groupby(['country', 'store', 'product'])['num_sold'].mean().reset_index()

# Visualize
sns.barplot(data=grouped, x='country', y='num_sold', hue='store')
plt.title('Average num_sold by Country and Store')
plt.show()


# Create a mapping for weekdays (0=Mon, 1=Tue, ..., 6=Sun)
weekday_labels = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}

# Pivot table
heatmap_data = train_df.pivot_table(index='month', columns='weekday', values='num_sold', aggfunc='mean')

# Rename columns 
heatmap_data.columns = [weekday_labels[col] for col in heatmap_data.columns]

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='.2f', annot_kws={"size": 10})
plt.title('Average num_sold by Month and Weekday')
plt.show()


counts = train_df.groupby(["country","store","product"])["num_sold"].count()
missing_data = counts.loc[counts != 2557]
missing_data_df = missing_data.reset_index()
missing_data_df["num_sold_missing"] = 2557 - missing_data_df["num_sold"]
missing_data_df


missing_data = train_df[train_df['num_sold'].isna()]
triplets_with_nan = missing_data.groupby(['country', 'store', 'product']).size().reset_index(name='missing_count')
triplets_with_nan = triplets_with_nan[triplets_with_nan['missing_count'] > 0]

# Visualization of missing "num_sold" values
for _, triplet in triplets_with_nan.iterrows():
    country, store, product = triplet['country'], triplet['store'], triplet['product']
    
    triplet_data = train_df[(train_df['country'] == country) & 
                            (train_df['store'] == store) & 
                            (train_df['product'] == product)]
    
    missing_dates = triplet_data[triplet_data['num_sold'].isna()]['date']
    
    # Plot the time series
    plt.figure(figsize=(12, 6))
    plt.plot(triplet_data['date'], triplet_data['num_sold'], 
             label='num_sold', linewidth=0.8, alpha=0.8, color='blue')
    plt.title(f'Time Series for {country}, {store}, {product}')
    plt.xlabel('Date')
    plt.ylabel('num_sold')
    
    # Add vertical red lines for missing dates
    for missing_date in missing_dates:
        plt.axvline(missing_date, color='red', linestyle='--', linewidth=0.2, alpha=0.7, label='Missing Data')
    
    # Adjust legend to avoid duplicates
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

