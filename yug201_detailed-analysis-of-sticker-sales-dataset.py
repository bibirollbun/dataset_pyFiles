import pandas as pd

# Load the dataset
df1 = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')

# Display the DataFrame's info()
print("DataFrame Info:")
df1.info()

# Display the first 5 rows of the DataFrame
print("\nFirst 5 Rows:")
print(df1.head())

# Display the number of rows and columns
print("\nDataFrame Shape:")
print(df1.shape)

# Display the describe() output of the DataFrame
print("\nDataFrame Description:")
print(df1.describe())



# Calculate the median of the 'num_sold' column
median_num_sold = df1['num_sold'].median()

# Impute the missing values in the 'num_sold' column with the calculated median
df1['num_sold'].fillna(median_num_sold, inplace=True)

# Print the number of missing values in the 'num_sold' column after imputation
print("Number of missing values in 'num_sold' after imputation:")
print(df1['num_sold'].isnull().sum())



# Convert the 'date' column to datetime objects
df1['date'] = pd.to_datetime(df1['date'])

# Verify the conversion by printing df1.info()
print("DataFrame Info after converting 'date' column:")
print(df1.info())




import matplotlib.pyplot as plt

# Set 'date' as the index
df1 = df1.set_index('date')

# Resample to monthly frequency and sum num_sold
monthly_sales = df1['num_sold'].resample('M').sum()

# Create a line plot
plt.figure(figsize=(12, 6))
monthly_sales.plot(kind='line')
plt.title('Overall Sales Trends Over Time (Monthly)')
plt.xlabel('Date')
plt.ylabel('Total Number of Items Sold')
plt.grid(True)
plt.tight_layout()

# Save the plot
plt.savefig('monthly_sales_trend.png')
img = 'monthly_sales_trend.png'


# Print the resampled data
print(monthly_sales)




# Reset index to use 'date' for grouping
df1 = df1.reset_index()

# Calculate total sales by country
sales_by_country = df1.groupby('country')['num_sold'].sum().sort_values(ascending=False)

# Create a bar chart
plt.figure(figsize=(10, 6))
sales_by_country.plot(kind='bar')
plt.title('Total Sales by Country')
plt.xlabel('Country')
plt.ylabel('Total Number of Items Sold')
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot
plt.savefig('sales_by_country.png')
img = 'sales_by_country.png'
# Print the sales by country data
print(sales_by_country)




# Reset index to use 'date' for grouping (if not already reset)
# df1 = df1.reset_index() #Potentially redundant, but doesn't hurt.

# Calculate total sales by product
sales_by_product = df1.groupby('product')['num_sold'].sum().sort_values(ascending=False)

# Create a bar chart
plt.figure(figsize=(10, 6))
sales_by_product.plot(kind='bar')
plt.title('Total Sales by Product')
plt.xlabel('Product')
plt.ylabel('Total Number of Items Sold')
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot
plt.savefig('sales_by_product.png')
img = 'sales_by_product.png'

# Print the sales by product data
print(sales_by_product)




# Calculate total sales by store
sales_by_store = df1.groupby('store')['num_sold'].sum().sort_values(ascending=False)

# Create a bar chart
plt.figure(figsize=(10, 6))
sales_by_store.plot(kind='bar')
plt.title('Total Sales by Store')
plt.xlabel('Store')
plt.ylabel('Total Number of Items Sold')
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot
plt.savefig('sales_by_store.png')
img = 'sales_by_store.png'

# Print the sales by store data
print(sales_by_store)


import seaborn as sns

# Reset index to use 'date' for grouping.
df1 = df1.reset_index()

# Calculate the sum of 'num_sold' for each combination of country and store.
sales_by_country_store = df1.groupby(['country', 'store'])['num_sold'].sum().reset_index()

# Create a pivot table for better visualization
pivot_table = sales_by_country_store.pivot(index='country', columns='store', values='num_sold')

# Create a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="viridis", linewidths=.5)
plt.title('Total Sales by Country and Store')
plt.xlabel('Store')
plt.ylabel('Country')
plt.tight_layout()

# Save the heatmap
plt.savefig('sales_by_country_store_heatmap.png')

# Create a bar chart for total sales per Country and Store combination
plt.figure(figsize=(14, 7))
sns.barplot(x='country', y='num_sold', hue='store', data=sales_by_country_store, palette='viridis')
plt.title('Total Sales by Country and Store')
plt.xlabel('Country')
plt.ylabel('Total Number of Items Sold')
plt.xticks(rotation=45)
plt.legend(title='Store')
plt.tight_layout()
plt.savefig('sales_by_country_store.png')

# Combine images

from PIL import Image

# Open images
img_heatmap = Image.open('sales_by_country_store_heatmap.png')
img_bar = Image.open("sales_by_country_store.png")

# Get dimensions for combined image
width = max(img_heatmap.width, img_bar.width)
height = img_heatmap.height + img_bar.height

# Create new image
combined_image = Image.new('RGB', (width, height), color='white')

# Paste images
combined_image.paste(img_heatmap, (0, 0))
combined_image.paste(img_bar, (0, img_heatmap.height))

# Save combined image
combined_image.save('sales_by_country_and_store_combined.png')
img = 'sales_by_country_and_store_combined.png'



# Print the sales by country and store data
print(sales_by_country_store)



# Reset index to use 'date' for grouping.  (Should already be reset, but this is safe)
df1 = df1.reset_index()

# Calculate the sum of 'num_sold' for each combination of product and country.
sales_by_product_country = df1.groupby(['product', 'country'])['num_sold'].sum().reset_index()

# Create a pivot table for the heatmap
pivot_table = sales_by_product_country.pivot(index='product', columns='country', values='num_sold')

# Create a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="viridis", linewidths=.5)
plt.title('Total Sales by Product and Country')
plt.xlabel('Country')
plt.ylabel('Product')
plt.tight_layout()

# Save the heatmap
plt.savefig('sales_by_product_country_heatmap.png')

# Create a bar chart for total sales per Product and Country combination
plt.figure(figsize=(14, 7))
sns.barplot(x='product', y='num_sold', hue='country', data=sales_by_product_country, palette='viridis')
plt.title('Total Sales by Product and Country')
plt.xlabel('Product')
plt.ylabel('Total Number of Items Sold')
plt.xticks(rotation=45, ha="right")  # Rotate labels for readability
plt.legend(title='Country')
plt.tight_layout()
plt.savefig('sales_by_product_country.png')

# Combine images
from PIL import Image

# Open images
img_heatmap = Image.open('sales_by_product_country_heatmap.png')
img_bar = Image.open('sales_by_product_country.png')

# Get dimensions for combined image
width = max(img_heatmap.width, img_bar.width)
height = img_heatmap.height + img_bar.height

# Create new image
combined_image = Image.new('RGB', (width, height), color='white')

# Paste images
combined_image.paste(img_heatmap, (0, 0))
combined_image.paste(img_bar, (0, img_heatmap.height))

# Save combined image
combined_image.save('sales_by_product_and_country_combined.png')
img = 'sales_by_product_and_country_combined.png'


# Print the sales by product and country data
print(sales_by_product_country)



# Calculate the sum of 'num_sold' for each combination of product and store.
sales_by_product_store = df1.groupby(['product', 'store'])['num_sold'].sum().reset_index()

# Create a pivot table for the heatmap
pivot_table = sales_by_product_store.pivot(index='product', columns='store', values='num_sold')

# Create a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="viridis", linewidths=.5)
plt.title('Total Sales by Product and Store')
plt.xlabel('Store')
plt.ylabel('Product')
plt.tight_layout()

# Save the heatmap
plt.savefig('sales_by_product_store_heatmap.png')

# Create a bar chart for total sales per Product and Store combination
plt.figure(figsize=(14, 7))
sns.barplot(x='product', y='num_sold', hue='store', data=sales_by_product_store, palette='viridis')
plt.title('Total Sales by Product and Store')
plt.xlabel('Product')
plt.ylabel('Total Number of Items Sold')
plt.xticks(rotation=45, ha="right")  # Rotate labels for readability
plt.legend(title='Store')
plt.tight_layout()
plt.savefig('sales_by_product_store.png')

# Combine images
from PIL import Image

# Open images
img_heatmap = Image.open('sales_by_product_store_heatmap.png')
img_bar = Image.open('sales_by_product_store.png')

# Get dimensions for combined image
width = max(img_heatmap.width, img_bar.width)
height = img_heatmap.height + img_bar.height

# Create new image
combined_image = Image.new('RGB', (width, height), color='white')

# Paste images
combined_image.paste(img_heatmap, (0, 0))
combined_image.paste(img_bar, (0, img_heatmap.height))

# Save combined image
combined_image.save('sales_by_product_and_store_combined.png')
img = 'sales_by_product_and_store_combined.png'


# Print the sales by product and store data
print(sales_by_product_store)



import pandas as pd
import matplotlib.pyplot as plt

# Reset index to use 'date' for grouping.
df1 = df1.reset_index(drop=True)

# Ensure 'date' is a datetime object (it should be from previous steps)
df1['date'] = pd.to_datetime(df1['date'])

# Group by 'country' and 'date', then calculate the sum of 'num_sold'
sales_by_country_date = df1.groupby(['country', 'date'])['num_sold'].sum().reset_index()

# Resample to monthly frequency and sum num_sold
sales_by_country_date['date'] = pd.to_datetime(sales_by_country_date['date'])
sales_by_country_date.set_index('date', inplace=True)
monthly_sales = sales_by_country_date.groupby('country')['num_sold'].resample('M').sum()

# Create a figure and a set of subplots
countries = df1['country'].unique()
fig, axes = plt.subplots(nrows=len(countries), ncols=1, figsize=(14, 20))

# Loop through each country and create a time series plot
for i, country in enumerate(countries):
    country_data = monthly_sales.loc[country]
    country_data.plot(ax=axes[i], title=f'Monthly Sales Trend for {country}', grid=True)
    axes[i].set_xlabel('Date')
    axes[i].set_ylabel('Number of Items Sold')

# Adjust layout
plt.tight_layout()

# Save the combined plot
plt.savefig('monthly_sales_trends_by_country.png')
img = 'monthly_sales_trends_by_country.png'

# Print the resampled data for each country
print(monthly_sales)




# Create a Markdown file for the report
with open('sales_analysis_report.md', 'w') as f:
    f.write("# Sales Analysis Report\n\n")

    # Overall Sales Trends
    f.write("## Overall Sales Trends\n\n")
    f.write("The overall sales trend shows a yearly seasonal pattern, with peaks typically observed at the beginning and end of each year. There is a slight general upward trend in sales over the observed period.\n\n")
    f.write("![Overall Sales Trend](monthly_sales_trend.png)\n\n")

    # Sales by Country
    f.write("## Sales by Country\n\n")
    f.write("Norway has the highest total sales, significantly outperforming other countries. Singapore and Canada follow, forming a second tier. Finland and Italy constitute a third tier, while Kenya has considerably lower sales.\n\n")
    f.write("![Sales by Country](sales_by_country.png)\n\n")

    # Sales by Store
    f.write("## Sales by Store\n\n")
    f.write("Premium Sticker Mart records the highest sales among all stores, followed by Stickers for Less. Discount Stickers shows the lowest sales performance.\n\n")
    f.write("![Sales by Store](sales_by_store.png)\n\n")

    # Sales by Product
    f.write("## Sales by Product\n\n")
    f.write("Kaggle and Kaggle Tiers are the top-selling products. Kerneler Dark Mode and Kerneler follow, while Holographic Goose has the lowest sales.\n\n")
    f.write("![Sales by Product](sales_by_product.png)\n\n")
    
    # Sales by Country and Store
    f.write("## Sales by Country and Store\n\n")
    f.write("Premium Sticker Mart consistently leads in sales across all countries. Norway's Premium Sticker Mart shows the highest sales, while Kenya's Discount Stickers records the lowest. The combined graph of Heatmap and Bar chart shows that Premium Sticker Mart consistently outperformed other stores within each country, with Norway's branch achieving the highest sales. Singapore and Canada's Premium Sticker Marts also performed strongly. Conversely, Discount Stickers consistently had the lowest sales in each country. While Norway demonstrated significantly higher overall sales than other countries, Kenya's Premium Sticker Mart had the lowest total sales.\n\n")
    f.write("![Sales by Country and Store](sales_by_country_and_store_combined.png)\n\n")

    # Sales by Product and Country
    f.write("## Sales by Product and Country\n\n")
    f.write("Kaggle products are particularly dominant in Norway. Kenya exhibits the lowest sales for all products. Product preferences vary across countries; for example, Norway favors Kaggle over Kaggle Tiers more than other countries do. The combined graph of Heatmap and Bar chart shows that the Kaggle and Kaggle Tiers are the top-selling products. Norway is a dominant market for Kaggle products, exhibiting significantly higher sales compared to other countries. Kenya consistently shows the lowest sales across all products. Holographic Goose is the poorest performing product overall.\n\n")
    f.write("![Sales by Product and Country](sales_by_product_and_country_combined.png)\n\n")

    # Sales by Product and Store
    f.write("## Sales by Product and Store\n\n")
    f.write("Premium Sticker Mart records the highest sales for most products, except for Holographic Goose, where Stickers for Less shows slightly better results. Kaggle and Kaggle Tiers are top sellers across all stores. The combined graph of Heatmap and Bar chart shows that the sales data was analyzed by product and store, revealing key trends. Premium Sticker Mart dominated sales for most products, except Holographic Goose, where Stickers for Less performed slightly better. Kaggle and Kaggle Tiers were the top-selling products across all stores. Holographic Goose consistently underperformed, and Discount Stickers was the lowest-performing store overall.\n\n")
    f.write("![Sales by Product and Store](sales_by_product_and_store_combined.png)\n\n")


    # Time Series Analysis by Country
    f.write("## Time Series Analysis by Country\n\n")
    f.write("All countries show a seasonal sales pattern with peaks near the year's end. Canada, Finland, Norway, and Singapore demonstrate upward sales trends, with Norway showing the most significant increase. Italy's upward trend is less prominent, and Kenya uniquely shows a downward trend.\n\n")
    f.write("![Time Series Analysis by Country](monthly_sales_trends_by_country.png)\n\n")


