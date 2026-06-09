import pandas as pd
df=pd.read_csv('/kaggle/input/DontGetKicked/training.csv')


# step 1: exclude some of the features 
exclude_col=["PurchDate", "VehYear", "Model", "Trim", "SubModel", "WheelTypeID", "BYRNO", "VNZIP1", "VNST"]
df=df.drop(columns=exclude_col)


# step 2: Set "RefId" as index
df=df.set_index('RefId')


df['IsBadBuy'] = df['IsBadBuy'].astype('category')
df.info()


# step3: define y (target) and X (inputs)
target= df['IsBadBuy']
inputs=df.drop(columns='IsBadBuy')
# partition the data into training and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(inputs, target, test_size=0.2, random_state=42, shuffle=True)


# step4: convert any out-of-range values to null
import numpy as np
continuous_field=[
    'VehicleAge',
    'VehOdo',
    'MMRAcquisitionAuctionAveragePrice',
    'MMRAcquisitionAuctionCleanPrice',
    'MMRAcquisitionRetailAveragePrice',
    'MMRAcquisitonRetailCleanPrice',
    'MMRCurrentAuctionAveragePrice',
    'MMRCurrentAuctionCleanPrice',
    'MMRCurrentRetailAveragePrice',
    'MMRCurrentRetailCleanPrice',
    'VehBCost',
    'WarrantyCost'
]

continuous_dict={
    'VehOdo': (0,120000),
    'MMRAcquisitionAuctionAveragePrice': (800,46000),
    'MMRAcquisitionAuctionCleanPrice': (1000,46000),
    'MMRAcquisitionRetailAveragePrice': (1000,46000),
    'MMRAcquisitonRetailCleanPrice': (1000,46000),
    'MMRCurrentAuctionAveragePrice': (300,46000),
    'MMRCurrentAuctionCleanPrice': (400,46000),
    'MMRCurrentRetailAveragePrice': (800,46000),
    'MMRCurrentRetailCleanPrice': (1000,46000),
    'VehBCost': (1000,46000),
    'WarrantyCost': (400,8000)}
# convering to null:
for col, (min_val, max_val) in continuous_dict.items():
    inputs[col] = inputs[col].apply(lambda x: x if min_val <= x <= max_val else None)


inputs
inputs['MMRAcquisitionAuctionAveragePrice'].min()


# convert the value 'NOT AVAIL' in the 'color' variable to null
import numpy as np
def frequency_table(variable):
    unique_colors, counts= np.unique(variable.dropna(), return_counts=True)
    percentages = (counts / len(variable)) * 100

    # Create a dictionary to store the value counts and percentages
    value_counts_and_percentages = zip(unique_colors, counts, percentages)

    # Print the value counts and percentages
    for i, j, k in value_counts_and_percentages:
        print(f"{i}: Count: {j}, Percentage: {k:.2f}%")
    return


frequency_table(inputs['Color'])


inputs['Color'] = inputs['Color'].replace("MAROON", 'RED')
inputs['Color'] = inputs['Color'].replace(["SILVER", "GREY"], 'GRAY')
inputs['Color'] = inputs['Color'].replace("GOLD", 'YELLOW')
inputs['Color'] = inputs['Color'].replace("BEIGE", 'BROWN')
inputs['Color'] = inputs['Color'].replace('NOT AVAIL', np.nan)

inputs['Color'].unique()


# step 6
import pandas as pd

def group_rare_categories(series, threshold=0.01):
    """
    Replace categories with frequency < threshold by 'OTHER'
    """
    freq = series.value_counts(normalize=True)  # percentages as proportions
    rare_categories = freq[freq < threshold].index
    
    return series.apply(lambda x: 'OTHER' if x in rare_categories else x)

inputs['Color'] = group_rare_categories(inputs['Color'], threshold=0.01)
inputs['Make'] = group_rare_categories(inputs['Make'], threshold=0.01)
frequency_table(inputs['Color'])
frequency_table(inputs['Make'])


# step 7:
# Define a minimum value for coefficient of variation
min_cv = 0.1

# Calculate the coefficient of variation for each column
cv_values = x_train[continuous_field].std() / x_train[continuous_field].mean()

# Filter out columns with CV less than 0.1
selected_columns =  cv_values[cv_values < 0.1].index

# Create a new DataFrame with only the selected columns
filtered_con = inputs[selected_columns]

# Print the resulting DataFrame
inputs_con = inputs[continuous_field].drop(selected_columns, axis=1)
print(inputs_con)


columns=x_train.columns
categorical_fields=[j for j in columns if j not in continuous_field]
# Define a threshold for the dominant category percentage
threshold = 99

# Calculate the percentage of the mode category for each column
mode_category = (x_train[categorical_fields].apply(lambda x: x.value_counts().max() / len(x)) * 100)

# Select columns where the mode category percentage is greater than the threshold
selected_categorical_columns = mode_category[mode_category > threshold].index

# Create a new DataFrame with only the selected columns
mode_filtered_inputs = inputs[selected_categorical_columns]

# Filter out selected columns and print the resulting DataFrame
inputs_cat = inputs[categorical_fields].drop(selected_categorical_columns, axis=1)
print(inputs_cat)


# Set a threshold for excluding columns 
threshold = 90

# Calculate the percentage of distinct categories in categorical variables
distinct_percentage = (x_train[categorical_fields].apply(lambda x: x.dropna().nunique() / x.count()) * 100)

# Select categorical columns based on distinct percentage threshold
selected_categorical_columns = distinct_percentage[distinct_percentage > threshold].index

# Create a new DataFrame with only the selected columns
distinct_filtered_inputs = inputs_cat[selected_categorical_columns]

# Filter out selected columns and print the resulting DataFrame
inputs_cat = inputs_cat.drop(selected_categorical_columns, axis=1)
print(inputs_cat)


inputs = pd.concat([inputs_cat, inputs_con], axis=1)


import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact

def handle_high_missing_categorical(x_train, y_train, features, target_col, alpha=0.05):
    """
    بررسی ارتباط ویژگی‌های categorical با target.
    اگر ارتباط معنی‌دار بود، nullها را 'unknown' می‌کنیم.
    اگر ارتباط معنی‌دار نبود، ستون حذف می‌شود.
    
    Parameters:
    -----------
    x_train : pd.DataFrame
        دیتاست ویژگی‌ها
    y_train : pd.Series
        ستون هدف
    features : list
        لیست ستون‌های categorical با missing زیاد
    target_col : str
        نام ستون هدف
    alpha : float
        سطح معنی‌داری (پیش‌فرض 0.05)
    
    Returns:
    --------
    x_train : pd.DataFrame
        دیتاست بعد از اعمال تغییرات
    """
    for feature in features:
        # ترکیب feature و target در یک DataFrame موقت
        temp = pd.concat([x_train[feature], y_train], axis=1).dropna()
        
        if temp.shape[0] < 30:
            print(f"{feature}: Too few non-null values → Dropping column")
            x_train = x_train.drop(columns=[feature])
            continue
        
        # جدول فراوانی
        crosstab = pd.crosstab(temp[feature], temp[target_col])
        
        # اگر جدول 2x2 و مقادیر کوچک، از fisher_exact استفاده کن
        if crosstab.shape == (2,2):
            _, p_val = fisher_exact(crosstab)
        else:
            _, p_val, _, expected = chi2_contingency(crosstab)
            if (expected < 5).any():
                print(f"{feature}: Some expected counts <5, p-value may be unreliable")
        
        # تصمیم‌گیری
        if p_val < alpha:
            print(f"{feature}: Significant (p-value={p_val:.4f}) → Keep & fill nulls as 'unknown'")
            x_train[feature] = x_train[feature].fillna('unknown')
        else:
            print(f"{feature}: Not significant (p-value={p_val:.4f}) → Dropping column")
            x_train = x_train.drop(columns=[feature])
            
    return x_train

# مثال استفاده:
features_to_check = ['PRIMEUNIT', 'AUCGUART']
x_train = handle_high_missing_categorical(x_train, y_train, features_to_check, target_col='IsBadBuy')



x_train.info()


columns = x_train.columns

categorical_fields_x = [
    'Auction',
    'Make',
    'Color',
    'Transmission',
    'WheelType',
    'Nationality',
    'Size',
    'TopThreeAmericanName',
    'AUCGUART'
]

# Create a new list of columns excluding categorical_fields (continuous)
continuous_fields_x = [j for j in columns if j not in categorical_fields_x]


import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder

x_train_iso = x_train.copy()

# Drop rows with NaN values
x_train_iso = x_train_iso.dropna()

# Scale numeric columns
scaler = StandardScaler()
x_train_iso[continuous_fields_x] = scaler.fit_transform(x_train_iso[continuous_fields_x])

# Encode categorical columns
for col in categorical_fields_x:
    le = LabelEncoder()
    x_train_iso[col] = le.fit_transform(x_train_iso[col])

# Fit Isolation Forest
clf = IsolationForest(contamination=0.01, random_state=42)
clf.fit(x_train_iso)

# Predict outliers
outliers = clf.predict(x_train_iso)
x_train_iso['outlier'] = outliers

# Remove outliers
x_train_clean = x_train_iso[outliers == 1].reset_index(drop=True)

# Percentage of outliers
percentage_outliers = (outliers[outliers == -1].shape[0] / len(outliers)) * 100
print(f"Percentage of outliers: {percentage_outliers:.2f}%")
print(f"Shape after removing outliers: {x_train_clean.shape}")


# Create a new column with the number of missing values in each row
x_train_clean['Num_Missing_Values'] = x_train_clean.isnull().sum(axis=1)

# Count and percentage of rows with missing values
rows_with_missing_values = x_train_clean[x_train_clean['Num_Missing_Values'] > 0]

total_rows = len(x_train_clean)
rows_with_missing_count = len(rows_with_missing_values)
percentage_rows_with_missing = (rows_with_missing_count / total_rows) * 100

# Display the report
#print("Report on Rows with Missing Values:")
#print(f"Total Rows: {total_rows}")
#print(f"Rows with Missing Values: {rows_with_missing_count} ({percentage_rows_with_missing:.2f}%)")


# Display the DataFrame with the new column
#print("\nDataFrame with Num_Missing_Values column:")
#print(train_outprep.sort_values(by='Num_Missing_Values', ascending = False))

# Discard rows with missing values
x_train_clean_no_missing = x_train_clean.dropna()

# Define the threshold for maximum allowable missing values per row
max_missing_values_threshold = 4

# Filter rows based on the 'Num_Missing_Values' column
x_train_clean = x_train_clean[x_train_clean['Num_Missing_Values'] <= max_missing_values_threshold].iloc[:, :-1]



# Report on count and percentage of missing values in each column
missing_values_report = pd.DataFrame({
    'Column': x_train_clean.columns,
    'Missing Values': x_train_clean.isnull().sum(),
    'Percentage Missing': x_train_clean.isnull().mean() * 100
})

# Display the missing values report
print("Missing Values Report:")
print(missing_values_report)

