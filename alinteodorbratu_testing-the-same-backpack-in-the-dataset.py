import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



df1 = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
df2 = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
df3 = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")


df1.set_index('id', inplace = True)
df2.set_index('id', inplace = True)
df3.set_index('id', inplace = True)


# Define columns to compare, excluding 'Price'
# This is how we can find identical backpacks with different prices
compare_cols = [col for col in df1.columns if col != 'Price']

# Merge df1 and df2 on all columns except 'Price'
merged = df1.merge(df2, on=compare_cols, suffixes=('_df1', '_df2'))

# Rows where 'Price' is different
price_diff_rows = merged[merged['Price_df1'] != merged['Price_df2']]

# Rows where 'Price' is the same
price_same_rows = merged[merged['Price_df1'] == merged['Price_df2']]

# Check which of these exist in df3 (ignoring 'Price')
exists_in_df3 = price_diff_rows[compare_cols].merge(df3, on=compare_cols, how='inner')

# Display results
print("Rows with different prices in df1 and df2:")
print(price_diff_rows)

print("\nRows also found in df3:")
print(exists_in_df3)

print("\nRows with the same price in df1 and df2:")
print(price_same_rows)


# Check if rows from df1 exist in df3
df1_in_df3 = df1[compare_cols].merge(df3, on=compare_cols, how='inner')

# Check if rows from df2 exist in df3
df2_in_df3 = df2[compare_cols].merge(df3, on=compare_cols, how='inner')

# Print results
print("Rows from df1 found in df3:")
print(df1_in_df3)

print("\nRows from df2 found in df3:")
print(df2_in_df3)


price_diff_rows["Price Difference"] = abs(price_diff_rows["Price_df1"] - price_diff_rows["Price_df2"])
average_difference = price_diff_rows["Price Difference"].mean()
print("Average difference in price between datasets {:.2f}".format(average_difference))


average_train_price = df1["Price"].mean()
average_train_extra_price = df2["Price"].mean()

print(f"The average price in the train data is {average_train_price:.2f}")
print(f"The average price in the train extra data is {average_train_extra_price:.2f}")

