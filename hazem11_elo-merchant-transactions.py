import pandas as pd
import numpy as np
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime
import statistics
from sklearn.preprocessing import OneHotEncoder


def IQROutlierCheck(df, col):
    
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q1 + 1.5 * IQR
    critic_score_outliers = df[(df[col] < lower) | (df[col] > upper)]
    
    return critic_score_outliers


def OutliersInfo(df, cols):

    outlier_dict=dict()
    for col in cols: 
        print(f"\n{col}")
        print("-"*35)
        critic_outliers = IQROutlierCheck(df, col)
        outlier_dict[col] = critic_outliers.index
        print(f"Number of outlier samples produced by IQR is {critic_outliers[col].shape[0]}")
        for i in range(0, 2):

            print("{}% percentile value is {:3.3f}".format(i, np.percentile(df[col], i)))
        for i in range(98, 101):
            print("{}% percentile value is {:3.3f}".format(i, np.percentile(df[col], i)))

        percent = np.percentile(df[col], 1)
        count = len(df[df[col]<percent])    
        print(f"\n\t- Number of values less than {percent} is {count}")
        percent = np.percentile(df[col], 99)
        count = len(df[df[col]>percent]) 
        print(f"\t- Number of values greater than {percent} is {count}")


# Define data types for each column to reduce memory usage
dtypes = {
    'card_id': 'category',
    'merchant_id': 'category',
    'month_lag': 'int8',
    'authorized_flag': 'category',
    'category_1': 'category',
    'installments': 'int16',
    'purchase_amount': 'float32',
    'city_id': 'int16',
    'state_id': 'int8',
    'subsector_id': 'int8',
    'merchant_category_id': 'int16',
    'category_2': 'float32',  
    'category_3': 'category'
}



# Define chunk size
chunk_size = 500000

# Initialize lists to store chunks
history_chunks = []
new_merchant_chunks = []

# Read historical transactions in chunks
for chunk in pd.read_csv('/kaggle/input/elo-merchant-category-recommendation/historical_transactions.csv', chunksize=chunk_size, dtype=dtypes):
    history_chunks.append(chunk)

# Read new merchant transactions in chunks
for chunk in pd.read_csv('/kaggle/input/elo-merchant-category-recommendation/new_merchant_transactions.csv', chunksize=chunk_size, dtype=dtypes):
    new_merchant_chunks.append(chunk)

# Combine all chunks from both datasets
combined_transactions = pd.concat(history_chunks + new_merchant_chunks, axis=0)

# Clean up memory
del history_chunks, new_merchant_chunks
gc.collect()

print(combined_transactions.shape)


# Displaying the first 5 rows of the combined transactions

combined_transactions.head()


# Summary of combined transactions DataFrame including data types, non-null counts, and memory usage

combined_transactions.info()


# Check for missing values in the combined transactions

combined_transactions.isnull().sum()


# Calculating and Printing the Mode for 'Category_2', 'Category_3', and 'Merchant_id' columns

category_2_mode = combined_transactions['category_2'].mode()[0]
category_3_mode = combined_transactions['category_3'].mode()[0]
merchant_id_mode = combined_transactions['merchant_id'].mode()[0]

print(f"Most Common in category_2 : {category_2_mode}")
print(f"Most Common in category_3 : {category_3_mode}")
print(f"Most Common in merchant_id : {merchant_id_mode}")


# Fill the missing values in 'mercahnt_id', 'category_2', and 'category_3' columns

combined_transactions['merchant_id'].fillna(merchant_id_mode, inplace=True)
combined_transactions['category_3'].fillna(category_3_mode, inplace=True)
combined_transactions['category_2'].fillna(category_2_mode, inplace=True)


# Check for missing values in combined transactions after filling them

combined_transactions.isnull().sum()


# Check for duplicate rows in combined transactions

combined_transactions.duplicated().sum()


# Generate descriptive statistics for numerical columns in combined transactions

combined_transactions.describe().T


# Display information about outliers

col = ['installments', 'month_lag', 'purchase_amount','category_2']
OutliersInfo(combined_transactions, col)


# Count unique value in 'installments' column

combined_transactions['installments'].value_counts()


# Count the number of rows where 'purchase_amount' is greater than 2 

combined_transactions[combined_transactions['purchase_amount'] > 2].count()


# Drop the rows where 'purchase_amount' is greater than 2 

combined_transactions = combined_transactions.drop(combined_transactions[combined_transactions['purchase_amount'] > 2].index)
combined_transactions.shape


# Drop rows where 'installments' is -1 or 999

Remove_Values = [-1,999]
combined_transactions = combined_transactions[~combined_transactions['installments'].isin(Remove_Values)]
combined_transactions.shape


# Convert the 'Purchase_date' column to datatime Type

combined_transactions['purchase_date']  = pd.to_datetime(combined_transactions['purchase_date'], format='%Y-%m-%d %H:%M:%S')


colors = sns.cubehelix_palette(20,reverse = True, light= 0.01,dark = 0.5, gamma= 0.7)
palette_color  = sns.color_palette("RdBu",10)
sns.set_theme(style="whitegrid", palette=palette_color)

def bar_plot(counts, column, ax, orient='v'):

    if orient=='h':
        x_col = 'count'
        y_col = column
    else:
        x_col = column
        y_col = 'count'

    sns.barplot(data = counts, x=x_col, y=y_col, ax=ax, orient=orient)


    if orient == 'v':
        # Annotate each bar with its height (number of occurrences)
        for p in ax.patches:
            x_coor = p.get_x() + 0.5 * p.get_width()
            y_coor = p.get_height()
            hight = int(p.get_height())

            ax.annotate(hight,            # Text to be displayed (converted to int for formatting)
                        (x_coor, y_coor), # Coordinates of the annotation (x, y)
                        ha='center',      # Horizontal alignment of the text ('center' aligns it at the center of the x-coordinate)
                        va='bottom',      # Vertical alignment of the text ('bottom' aligns it at the bottom of the bar)
                        color='black'     # Color of the text
                        )
        ax.set_xlabel(column, weight = "bold",  fontsize = 14, labelpad = 20)
        ax.set_ylabel('Number of Occurrences', weight = "bold", fontsize = 14, labelpad = 20)

    ax.tick_params(axis = 'both', labelsize = 12)

    return ax


def plot_categorical_feature(counts, figsize=(12,5)):
   column = counts.columns[0]
    
   # Create a figure with two subplots
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

   # Bar chart
   bar_plot(counts, column, ax1)

   # Pie chart
   ax2.pie(counts['count'], labels=counts[column], autopct = '%1.1f%%')
   ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
   ax2.legend(counts[column], loc="best") # Add legned with labels

   # Adjust layout
   plt.suptitle(f'Distribution of {column}', weight = "bold", fontsize = 16)
   plt.tight_layout()
   plt.show()


authorized_flag_counts = combined_transactions['authorized_flag'].value_counts().reset_index()
authorized_flag_counts


plot_categorical_feature(authorized_flag_counts)


Card_ID_Counts = combined_transactions['card_id'].value_counts().head(25).sort_values().reset_index()
Card_ID_Counts


plt.figure(figsize=(10,8))
sns.barplot(data=Card_ID_Counts,x='count',y='card_id' ,orient='h')
plt.title('Top 25 Card')
plt.xlabel('Count')
plt.ylabel('Card ID')
plt.show()


Merchant_ID_Counts = combined_transactions['merchant_id'].value_counts().head(25).sort_values().reset_index()
Merchant_ID_Counts


plt.figure(figsize=(10,8))
sns.barplot(data=Merchant_ID_Counts,x='count',y='merchant_id' ,orient='h')
plt.title('Top 25 Merchant')
plt.xlabel('Count')
plt.ylabel('Merchant ID')
plt.show()


Category_1_Counts = combined_transactions['category_1'].value_counts().reset_index()
Category_1_Counts


plot_categorical_feature(Category_1_Counts)


Installments_Counts = combined_transactions['installments'].value_counts().reset_index()
Installments_Counts


plt.figure(figsize=(10,8))
sns.barplot(data=Installments_Counts,x='count',y='installments' ,orient='h')
plt.title('Distribution of Installments')
plt.xlabel('Count')
plt.ylabel('Months')
plt.show()


Category_3_Counts = combined_transactions['category_3'].value_counts().reset_index()
Category_3_Counts


plot_categorical_feature(Category_3_Counts)


Month_Lag_Counts = combined_transactions['month_lag'].value_counts().reset_index()
Month_Lag_Counts


plt.figure(figsize=(10,8))
sns.barplot(data=Month_Lag_Counts,x='count',y='month_lag' ,orient='h')
plt.title('Distribution of Month Lag')
plt.xlabel('Count')
plt.ylabel('Months')
plt.show()


Category_2_Counts = combined_transactions['category_2'].value_counts().reset_index()
Category_2_Counts


plot_categorical_feature(Category_2_Counts)


grouped = combined_transactions.groupby('card_id')['purchase_amount'].sum().reset_index().sort_values(by='purchase_amount',ascending=False).head(25)
grouped


top_25_card_Purchased = grouped['card_id'].tolist()
filtered_trans = combined_transactions[combined_transactions['card_id'].isin(top_25_card_Purchased)]
filtered_trans


fig = px.box(filtered_trans,x='card_id',y='purchase_amount')

font_config = {
            'family': 'Arial',                # Font family
            'size': 24,                       # Font size
            'color': 'Black'          # Font color
        }

title_config = {
        'text': "Purchase Amount For Top 25 Card",               # The title text
        'x': 0.5,                             # x-position of the title (0 to 1, where 0.5 is centered)
        'xanchor': 'center',                  # Anchor point for the title's x position
        'y': 0.95,                            # y-position of the title (0 to 1)
        'yanchor': 'top',                     # Anchor point for the title's y position
        'font': font_config
        }

xaxis_config={
        'title': 'Card ID',                              # Axis title text
        'titlefont': {'size': 16, 'color': 'DarkBlue'}, # Font properties for the axis title
        'tickangle': -45,                                    # Angle of tick labels (degrees)
        'gridcolor': 'LightGray',                            # Grid line color
    }

yaxis_config={
        'title': 'Purchase Amount',
        'titlefont': {'size': 16, 'color': 'DarkBlue'},
    }

fig.update_layout(title= title_config,xaxis=xaxis_config, yaxis=yaxis_config)
fig.show()


Most_Purchased_Month = combined_transactions.groupby(combined_transactions['purchase_date'].dt.month)['card_id'].count().reset_index()
Most_Purchased_Month


sns.lineplot(data=Most_Purchased_Month,x='purchase_date',y='card_id')
plt.title('Transactions Over Months')
plt.xlabel('Month')
plt.ylabel('Count of Transactions')
plt.show()


sns.histplot(data=combined_transactions,x='purchase_amount',kde=True,bins=20)
plt.title('Distribution of Purchased Amount')
plt.xlabel('Purchased Amount')
plt.show()


sns.displot(combined_transactions['purchase_date'].dt.year,kind='kde')
plt.title('Kde plot for purchase date')
plt.show()


# Encode categorical features to numerical 

combined_transactions['authorized_flag'] = combined_transactions['authorized_flag'].map({'Y': 1 , 'N': 0})
combined_transactions['category_1'] = combined_transactions['category_1'].map({'Y': 1 , 'N': 0})
combined_transactions['category_3'] = combined_transactions['category_3'].map({'A': 1 , 'B': 2 , 'C': 3})


# Extract the year from 'purchase_date' column and create 'purchase_year' column

combined_transactions['purchase_year'] = combined_transactions['purchase_date'].dt.year


# Extract the month from 'purchase_date' column and create 'purchase_month' column


combined_transactions['purchase_month'] = combined_transactions['purchase_date'].dt.month


# Extract the day from 'purchase_date' column and create 'purchase_day' column


combined_transactions['purchase_day'] = combined_transactions['purchase_date'].dt.day


# Extract the day of week from 'purchase_date' column and create 'purchase_dow' column


combined_transactions['purchase_dow'] = combined_transactions['purchase_date'].dt.day_of_week


# Extract the hour from 'purchase_date' column and create 'purchase_hour' column


combined_transactions['purchase_hour'] = combined_transactions['purchase_date'].dt.hour


# Create 'is_weekend' column: 1 if 'purchase_dow' is Saturday (5) or Sunday (6), 0 otherwise

combined_transactions['is_weekend'] = np.where(combined_transactions['purchase_dow'].isin([5, 6]), 1, 0)


# Define time intervals based on the 'purchase_hour' column to Create 'purchase_at' column

time_day = [
    (combined_transactions['purchase_hour'] >= 6) & (combined_transactions['purchase_hour'] < 12),
    (combined_transactions['purchase_hour'] >= 12) & (combined_transactions['purchase_hour'] < 15),
    (combined_transactions['purchase_hour'] >= 15) & (combined_transactions['purchase_hour'] < 18),
    (combined_transactions['purchase_hour'] >= 18) & (combined_transactions['purchase_hour'] < 24)
]


# Define labels for different time periods of the day

purchase_at = ['Morning', 'Noon', 'Afternoon', 'Evening']

# Create 'purchase_at' column based on 'time_day' conditions and 'purchase_at' labels

combined_transactions['purchase_at'] = np.select(time_day, purchase_at, default='Night')


# Get today's date

today = np.datetime64(datetime.today())


# Calculate the difference in days between today and 'purchase_date'

days_difference = (today - combined_transactions['purchase_date'].values).astype('timedelta64[D]').astype(int)


# Calculate the months difference, assuming 30 days per month

combined_transactions['month_difference'] = (days_difference // 30).astype('int16')


# Adjust 'month_difference' by substracting 'month_lag'

combined_transactions['month_difference'] = combined_transactions['month_difference'] - combined_transactions['month_lag']


# Convert 'purchase_date' column to datetime objects

combined_transactions['purchase_date'] = pd.to_datetime(combined_transactions['purchase_date'])


# Display the first rows of the Data

pd.set_option('display.max_columns', None)
combined_transactions.head()


# Create Aggregation dictionary to define how to aggregate columns

aggregate_tbl = {
    'purchase_date':['min','max'],
    'purchase_year': [statistics.mode],
    'purchase_month': ['mean',statistics.mode],
    'purchase_day': [statistics.mode],
    'purchase_dow': [statistics.mode],
    'purchase_hour': ['min','max',statistics.mode],
    'is_weekend': ['mean','sum',statistics.mode],
    'month_difference': ['sum','mean','min','max'] 
    }



# Group the 'combined_transactions' by 'card_id' and apply the aggregations 

aggregate_purchase_date = combined_transactions.groupby('card_id').agg(aggregate_tbl).reset_index()
aggregate_purchase_date.head()


# Rename the columns of the aggregated Dataframe with 'trans_' prefix and concatenated column names

aggregate_purchase_date.columns = ['trans_'+'_'.join(col).strip() for col in aggregate_purchase_date.columns.values]
aggregate_purchase_date = aggregate_purchase_date.rename(columns={'trans_card_id_':'card_id'})
aggregate_purchase_date.head()


# Calculate the number of days since the first and last transactions for each card_id 

aggregate_purchase_date['first_transaction'] = (datetime.today() - aggregate_purchase_date['trans_purchase_date_min']).dt.days
aggregate_purchase_date['last_transaction'] = (datetime.today() - aggregate_purchase_date['trans_purchase_date_max']).dt.days


# Intialize OneHotEncoder to convert categorical 'purchase_at' to numerical feature

encoder = OneHotEncoder()
one_hot = encoder.fit_transform(combined_transactions[['purchase_at']])


# Create Dataframe from the one-hot encoded array with feature names

df = pd.DataFrame(one_hot.toarray(),columns=encoder.get_feature_names_out())
df.head()


# Reset the index of combined_transactions
combined_transactions = combined_transactions.reset_index(drop=True)

#Reset index of df
df = df.reset_index(drop=True)

# Concatenate the 'combined_transactions' DataFrame with the one-hot encoded DataFrame
combined_transactions = pd.concat([combined_transactions, df], axis=1)
combined_transactions.head()


# Group the 'combined_transactions' DataFrame by 'card_id' and aggregate 'purchase_amount'

aggregate_purchase_amount = combined_transactions.groupby('card_id')['purchase_amount'].agg(['sum','max','min','mean','median']).reset_index()
aggregate_purchase_amount.head()


# Rename the columns of the 'aggregate_purchase_amount' DataFrame

aggregate_purchase_amount.columns = ['trans_purchase_amount_'+''.join(col).strip() for col in aggregate_purchase_amount.columns.values]
aggregate_purchase_amount = aggregate_purchase_amount.rename(columns={'trans_purchase_amount_card_id':'card_id'})
aggregate_purchase_amount.head()


# Define an aggregation dictionary

aggregate_tbl2 = {
    'authorized_flag': ['sum','mean'],
    'city_id': [statistics.mode,'nunique'],
    'category_1': ['sum','mean',statistics.mode],
    'category_2': ['sum','mean',statistics.mode],
    'category_3': ['sum','mean',statistics.mode],
    'installments': ['mean','sum','min','max',statistics.mode],
    'merchant_category_id': [statistics.mode,'nunique'],
    'merchant_id': [statistics.mode,'nunique'],
    'state_id': [statistics.mode,'nunique'],
    'subsector_id': ['nunique',statistics.mode],
    'purchase_at_Afternoon': [statistics.mode,'sum','mean'],
    'purchase_at_Evening': [statistics.mode,'sum','mean'],
    'purchase_at_Morning': [statistics.mode,'sum','mean'],
    'purchase_at_Night': [statistics.mode,'sum','mean'],
    'purchase_at_Noon': [statistics.mode,'sum','mean']
}


# Dropping 'purchase_at' column

combined_transactions = combined_transactions.drop(columns='purchase_at',axis=1)


# Convert category columns to object

combined_transactions['category_1'] = combined_transactions['category_1'].astype('object')
combined_transactions['category_3'] = combined_transactions['category_3'].astype('object')


# Group 'combined_transactions' by 'card_id' and apply aggregations defined 

aggregate_features = combined_transactions.groupby('card_id').agg(aggregate_tbl2).reset_index()
aggregate_features.head()


# Rename the columns of the 'aggregate_features' DataFrame

aggregate_features.columns = [str(uppers)+'_'+str(lowers) for uppers, lowers in aggregate_features.columns.values]
aggregate_features.head()


# Concatenate 'aggregate_purchase_date', 'aggregate_purchase_amount' and 'aggregate_features'

aggregate_transactions = pd.concat([aggregate_purchase_date,aggregate_purchase_amount,aggregate_features],axis=1)


# Display the number of rows and columns of 'aggregate_transactions'

aggregate_transactions.shape


# Checking the number of null values in the 'aggregate_transactions' DataFrame

aggregate_transactions.isnull().sum().sum()


# Perform garbage collection to free up memory 

gc.collect()


# Save the 'aggregate_transactions' DataFrame to CSV file without the Index
aggregate_transactions.to_csv('/kaggle/working/Aggregate_Transactions.csv', index=False)

print("aggregate_transactions saved to /kaggle/working/aggregate_transactions.csv")

