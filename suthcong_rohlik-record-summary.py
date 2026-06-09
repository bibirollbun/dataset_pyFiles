import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import warnings
import seaborn as sns

warnings.filterwarnings("ignore")


# Load data
train=pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv")
test=pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv")
inventory=pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv")
calendar=pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv")
total=pd.concat((train,test))
total['date'] = pd.to_datetime(total['date'])
total = total.sort_values(by='date')
total


# total records by year
total['year'] = total['date'].dt.year
total.year.value_counts().reset_index().sort_values(by='year')


# Thống kê record theo năm và cửa hàng
pivot = total.pivot_table(
    values='unique_id', 
    index='year', 
    columns='warehouse', 
    aggfunc='count',
    fill_value=0  # Fill NaN with 0
)

pivot['Total'] = pivot.sum(axis=1)
pivot.loc['Total'] = pivot.sum(axis=0)
pivot


# Visualize
grouped = total.groupby(['date', 'warehouse'])['unique_id'].count().unstack()
# Create the plot
plt.figure(figsize=(18, 6))
sns.set_style("whitegrid")  # Use seaborn's grid style
palette = sns.color_palette("husl", n_colors=len(grouped.columns))  # Use a color palette

for warehouse, color in zip(grouped.columns, palette):
    sns.lineplot(x=grouped.index, y=grouped[warehouse], label=f'{warehouse}', color=color)

# Add vertical lines to separate years
for year in sorted(set(total['date'].dt.year)):
    plt.axvline(pd.Timestamp(f'{year}-01-01'), color='gray', linestyle='--', alpha=0.7)
    plt.annotate(
        str(year), xy=(pd.Timestamp(f'{year}-01-01'), grouped.max().max()), xytext=(pd.Timestamp(f'{year}-01-01'), grouped.max().max() -10),
        arrowprops=dict(facecolor='blue', arrowstyle="->"), fontsize=12, color='gray'
    )
# Add annotations for start and end time
start_date = grouped.index.min()
plt.axvline(start_date, color='blue', linestyle='--', alpha=0.7)
end_date = grouped.index.max()
plt.axvline(end_date, color='blue', linestyle='--', alpha=0.7)

start='Start date:\n'+str(start_date.date())
plt.annotate(
    start, xy=(start_date, grouped.max().max()), xytext=(start_date, grouped.max().max() -10),
    arrowprops=dict(facecolor='blue', arrowstyle="->"), fontsize=12, color='blue'
)
end='End date:\n'+str(end_date.date())
plt.annotate(
    end, xy=(end_date, grouped.max().max()), xytext=(end_date, grouped.max().max() -10),
    arrowprops=dict(facecolor='blue', arrowstyle="->"), fontsize=12, color='blue'
)

# Labels and title
plt.xlabel("Date")
plt.ylabel("Count Records")
plt.title("Records by days per warehouse", fontsize=14)
plt.legend(title="Warehouse", fontsize=12)
plt.xticks(rotation=0)  # Rotate x-axis labels for better readability
plt.grid(True, linestyle='--', alpha=0.6)

# Show plot
plt.show()


# Xét trong thành phố Budapest_1
total['month'] = total['date'].dt.month

city = total[total['warehouse']=='Budapest_1']
pivot = city.pivot_table(
    values='unique_id', 
    index='year', 
    columns='month', 
    aggfunc='count',
    fill_value=0  # Fill NaN with 0
)

pivot


# Xét trong thành phố Munich_1
total['month'] = total['date'].dt.month

city = total[total['warehouse']=='Munich_1']
pivot = city.pivot_table(
    values='unique_id', 
    index='year', 
    columns='month', 
    aggfunc='count',
    fill_value=0  # Fill NaN with 0
)

pivot


# Xét trong thành phố Frankfurt_1
total['month'] = total['date'].dt.month

city = total[total['warehouse']=='Frankfurt_1']
pivot = city.pivot_table(
    values='unique_id', 
    index='year', 
    columns='month', 
    aggfunc='count',
    fill_value=0  # Fill NaN with 0
)

pivot


# Xét trong thành phố Brno_1
total['month'] = total['date'].dt.month

city = total[total['warehouse']=='Brno_1']
pivot = city.pivot_table(
    values='unique_id', 
    index='year', 
    columns='month', 
    aggfunc='count',
    fill_value=0  # Fill NaN with 0
)

pivot


# Xét trong thành phố Prague_1
total['month'] = total['date'].dt.month

city = total[total['warehouse']=='Prague_1']
pivot = city.pivot_table(
    values='unique_id', 
    index='year', 
    columns='month', 
    aggfunc='count',
    fill_value=0  # Fill NaN with 0
)

pivot


# Xét trong thành phố Prague_2
total['month'] = total['date'].dt.month

city = total[total['warehouse']=='Prague_2']
pivot = city.pivot_table(
    values='unique_id', 
    index='year', 
    columns='month', 
    aggfunc='count',
    fill_value=0  # Fill NaN with 0
)

pivot


# Xét trong thành phố Prague_3
total['month'] = total['date'].dt.month

city = total[total['warehouse']=='Prague_3']
pivot = city.pivot_table(
    values='unique_id', 
    index='year', 
    columns='month', 
    aggfunc='count',
    fill_value=0  # Fill NaN with 0
)

pivot




