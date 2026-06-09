# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv('/kaggle/input/Predict-the-Customer-Satisfaction-CSE-22/train_dataset.csv')
test = pd.read_csv('/kaggle/input/Predict-the-Customer-Satisfaction-CSE-22/test_dataset.csv')



pd.set_option('display.max_columns', None)  # Display all columns


df


test


duplicates = df[df.drop(columns=['id']).duplicated()]
duplicates


# Remove rows that are duplicates when excluding the 'id' column
df = df[~df.drop(columns=['id']).duplicated()]


df


df.dtypes


# Remove '****' prefix to convert to integer
df['user_id'] = df['user_id'].str.replace(r'\*+', '', regex=True).astype(int)
test['user_id'] = test['user_id'].str.replace(r'\*+', '', regex=True).astype(int)


desc_df = pd.DataFrame(index=df.columns.to_list())
desc_df['type'] = df.dtypes
desc_df['count'] = df.count()
desc_df['nunique'] = df.nunique()
desc_df['null'] = df.isnull().sum()
desc_df['min'] = df.min()
desc_df['max'] = df.max()
desc_df


desc_test = pd.DataFrame(index=test.columns.to_list())
desc_test['type'] = df.dtypes
desc_test['count'] = df.count()
desc_test['nunique'] = df.nunique()
desc_test['null'] = df.isnull().sum()
desc_test['min'] = df.min()
desc_test['max'] = df.max()
desc_test


# List of columns to drop
columns_to_drop = ['tracking_number', 'transaction_id' , 'order_id']

# Drop the columns from df and test
df = df.drop(columns=columns_to_drop, errors='ignore')
test = test.drop(columns=columns_to_drop, errors='ignore')



# List of columns to replace null values with 0
columns_to_fill = [
    'loyalty_tier',
    'Received_tier_discount_percentage',
    'Received_card_discount_percentage'
]

# Replace null values with 0 in the specified columns
df[columns_to_fill] = df[columns_to_fill].fillna(0).astype(int)
test[columns_to_fill] = test[columns_to_fill].fillna(0).astype(int)



# List of columns to display unique categories
columns_to_check = [
    'Gender',
    'Is_current_loyalty_program_member',
    'product_category',
    'payment_method',
    'purchase_medium',
    'shipping_method',
    'customer_experience'
]

# Display unique categories for each column
for col in columns_to_check:
    unique_values = df[col].dropna().unique()
    print(f"Unique categories in '{col}': {unique_values}")



# Define a function to encode categorical columns
def encode_categorical_columns(dataframe):
    dataframe['Gender'] = pd.Categorical(dataframe['Gender'], categories=['O', 'F', 'M'], ordered=True)
    dataframe['Gender'] = dataframe['Gender'].cat.codes

    dataframe['Is_current_loyalty_program_member'] = pd.Categorical(
        dataframe['Is_current_loyalty_program_member'],
        categories=['NO', 'YES'],
        ordered=True
    )
    dataframe['Is_current_loyalty_program_member'] = dataframe['Is_current_loyalty_program_member'].cat.codes

    dataframe['product_category'] = pd.Categorical(
        dataframe['product_category'],
        categories=[
            'office supplies', 'electronics', 'pet supplies', 'clothing', 'books',
            'appliances', 'groceries', 'home', 'health', 'music', 'tools',
            'automotive', 'toys', 'sports', 'video games', 'beauty', 'movies',
            'jewelry', 'garden', 'furniture'
        ],
        ordered=True
    )
    dataframe['product_category'] = dataframe['product_category'].cat.codes

    dataframe['payment_method'] = pd.Categorical(
        dataframe['payment_method'],
        categories=[
            'visa_c', 'amex', 'mastercard_c', 'coinsph', 'visa_d', 'gcash', 'maya',
            'cash', 'bank_transfer', 'shopeepay', 'otc', 'grabpay', 'mastercard_d'
        ],
        ordered=True
    )
    dataframe['payment_method'] = dataframe['payment_method'].cat.codes

    dataframe['purchase_medium'] = pd.Categorical(
        dataframe['purchase_medium'],
        categories=['online', 'in-store'],
        ordered=True
    )
    dataframe['purchase_medium'] = dataframe['purchase_medium'].cat.codes

    dataframe['shipping_method'] = pd.Categorical(
        dataframe['shipping_method'],
        categories=['standard', 'express'],
        ordered=True
    )
    dataframe['shipping_method'] = dataframe['shipping_method'].cat.codes

    dataframe['customer_experience'] = pd.Categorical(
        dataframe['customer_experience'],
        categories=['bad', 'neutral', 'good'],
        ordered=True
    )
    dataframe['customer_experience'] = dataframe['customer_experience'].cat.codes

    return dataframe

# Concatenate df and test temporarily
combined = pd.concat([df, test])

# Encode combined data
combined = encode_categorical_columns(combined)

# Split back into df and test
df = combined.iloc[:len(df)].copy()
test = combined.iloc[len(df):].copy()

# Drop 'customer_experience' column from the test DataFrame
test.drop(columns=['customer_experience'], inplace=True)


import seaborn as sns
import matplotlib.pyplot as plt

# Extract numerical columns excluding `customer_experience`
numerical_columns = df.select_dtypes(include=['number']).columns

# Create box plots for each numerical column
plt.figure(figsize=(16, 6 * len(numerical_columns)))

for i, col in enumerate(numerical_columns, 1):
    plt.subplot(len(numerical_columns), 2, i)
    sns.boxplot(data=df, x='customer_experience', y=col, palette="muted")
    plt.title(f'Box Plot of {col} by Customer Experience', fontsize=14)
    plt.xlabel('Customer Experience', fontsize=12)
    plt.ylabel(col, fontsize=12)
    plt.tight_layout()

plt.show()



import seaborn as sns
import matplotlib.pyplot as plt

# Extract numerical columns excluding `customer_experience`
numerical_columns = df.select_dtypes(include=['number']).columns

# Create violin plots for each numerical column
plt.figure(figsize=(16, 6 * len(numerical_columns)))

for i, col in enumerate(numerical_columns, 1):
    plt.subplot(len(numerical_columns), 2, i)
    sns.violinplot(data=df, x='customer_experience', y=col, palette="muted")
    plt.title(f'Violin Plot of {col} by Customer Experience', fontsize=14)
    plt.xlabel('Customer Experience', fontsize=12)
    plt.ylabel(col, fontsize=12)
    plt.tight_layout()

plt.show()



# Define the outlier conditions
outlier_conditions_df = (
    (df['age'] == 0) |
    (df['Received_card_discount_percentage'] > 40) |
    (df['Received_coupon_discount_percentage'] > 40) |
    (df['Product_value'] > 12000)
)

outlier_conditions_test = (
    (test['age'] == 0) |
    (test['Received_card_discount_percentage'] > 40) |
    (test['Received_coupon_discount_percentage'] > 40) |
    (test['Product_value'] > 10000)
)

# Count the total number of outliers in df and test
outlier_count_df = outlier_conditions_df.sum()
outlier_count_test = outlier_conditions_test.sum()

# Print the outlier counts
print(f"Total outliers in df: {outlier_count_df}")
print(f"Total outliers in test: {outlier_count_test}")



df = df[df['age'] != 0]

# Define the conditions to filter rows that do not satisfy the criteria
remaining_rows_df = ~(
    (df['Received_card_discount_percentage'] > 40) |
    (df['Received_coupon_discount_percentage'] > 40) |
    (df['Product_value'] > 12000)
)

# Filter the dataframe to keep only the rows that do not satisfy the conditions
df = df[remaining_rows_df]

# Reset the index of the filtered dataframe (optional)
df.reset_index(drop=True, inplace=True)

# Print the shape of the updated dataframe
print(f"Updated df shape after removing rows: {df.shape}")



df.info(memory_usage='deep')


import seaborn as sns
import matplotlib.pyplot as plt

# Extract numerical columns excluding `customer_experience`
numerical_columns = df.select_dtypes(include=['number']).columns

# Create violin plots for each numerical column
plt.figure(figsize=(16, 6 * len(numerical_columns)))

for i, col in enumerate(numerical_columns, 1):
    plt.subplot(len(numerical_columns), 2, i)
    sns.violinplot(data=df, x='customer_experience', y=col, palette="muted")
    plt.title(f'Violin Plot of {col} by Customer Experience', fontsize=14)
    plt.xlabel('Customer Experience', fontsize=12)
    plt.ylabel(col, fontsize=12)
    plt.tight_layout()

plt.show()



def add_time_features(df):
    # Ensure datetime conversion
    df['received_date'] = pd.to_datetime(df['received_date'], errors='coerce')
    df['purchased_datetime'] = pd.to_datetime(df['purchased_datetime'], errors='coerce')
    df['released_date'] = pd.to_datetime(df['released_date'], errors='coerce')
    df['estimated_delivery_date'] = pd.to_datetime(df['estimated_delivery_date'], errors='coerce')

    # Received weekday feature
    df['received_weekday'] = df['received_date'].dt.weekday.fillna(0).astype(int)

    # Discount features
    df['total_discount_percentage'] = (
        df['Received_tier_discount_percentage'].fillna(0) +
        df['Received_card_discount_percentage'].fillna(0) +
        df['Received_coupon_discount_percentage'].fillna(0)
    ).astype(int) 

    df['discount_amount'] = (df['Product_value'] - df['final_payment']).fillna(0)
    df['discount_percentage'] = (
        (df['discount_amount'] / (df['Product_value'] + 1e-9)) * 100
    ).fillna(0)

    df['delivery_time'] = (
        (df['received_date'] - df['released_date']).dt.days).fillna(0)

    df['total_order_time'] = (
        (df['received_date'] - df['purchased_datetime']).dt.total_seconds() / 3600
    ).fillna(0)

    df['delivery_delay'] = (
        (df['received_date'] - df['estimated_delivery_date']).dt.days
    ).fillna(0)

    # Additional time-based features
    purchase_datetime = pd.to_datetime(df['purchased_datetime'])
    # df['purchase_hour'] = purchase_datetime.dt.hour.fillna(0).astype(int)
    df['is_weekend'] = purchase_datetime.dt.weekday.isin([5, 6]).astype(int)
    
    
    return df.fillna(0).replace([np.inf, -np.inf], 0)

# Apply features
df = add_time_features(df)
test = add_time_features(test)





# List of date columns to convert
date_columns = ['Date_Registered', 'purchased_datetime', 'released_date', 
                'estimated_delivery_date', 'received_date', 'payment_datetime']

# Convert dates to numeric (days) in one go
for col in date_columns:
    # Convert to datetime first
    df[col] = pd.to_datetime(df[col], errors='coerce')
    test[col] = pd.to_datetime(test[col], errors='coerce')
    
    # Get minimum date and calculate days difference
    min_date_df = df[col].min()
    df[col] = (df[col] - min_date_df).dt.days
    test[col] = (test[col] - min_date_df).dt.days
    
    # Convert to int
    df[col] = df[col].astype(int)
    test[col] = test[col].astype(int)

# Calculate additional datetime features
df['Purchase_Registration_Difference'] = (df['purchased_datetime'] - df['Date_Registered']).astype(int)
test['Purchase_Registration_Difference'] = (test['purchased_datetime'] - test['Date_Registered']).astype(int)

# df['received_weekday'] = pd.to_datetime(df['received_date']).dt.weekday.astype(int)
# test['received_weekday'] = pd.to_datetime(test['received_date']).dt.weekday.astype(int)

print("All date conversions completed successfully")
print("\nSample of converted date columns in training data:")
print(df[date_columns].head())


def add_optimized_features(df):
    # Discount amount features (third highest MI score)
    df['discount_amount_ratio'] = df['discount_amount'] / (df['Product_value'] + 1e-9)
    df['high_discount_order'] = (df['discount_amount'] > df['discount_amount'].median()).astype(int)
    
    # Customer age and tenure features (fourth highest MI score)
    df['customer_tenure_bins'] = pd.qcut(df['Purchase_Registration_Difference'], q=5, labels=False, duplicates='drop')
    
    # Time efficiency features
    df['delivery_time_bins'] = pd.qcut(df['delivery_time'], q=5, labels=False, duplicates='drop')
    # df['processing_time_bins'] = pd.qcut(df['processing_time'], q=5, labels=False, duplicates='drop')
    
    # Interaction features from high MI scores
    df['time_discount_interaction'] = df['total_order_time'] * df['discount_amount_ratio']

    return df.fillna(0).replace([np.inf, -np.inf], 0)

# Apply optimized features
df = add_optimized_features(df)
test = add_optimized_features(test)


from sklearn.feature_selection import mutual_info_classif
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def check_mi_scores(df):
    # Get features and target
    X = df.drop(columns=['customer_experience'])
    y = df['customer_experience']
    
    # Get only numeric columns
    numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns
    X_numeric = X[numeric_columns]
    
    # Calculate MI scores
    mi_scores = mutual_info_classif(X_numeric, y, random_state=42)
    
    # Create dataframe of scores
    mi_df = pd.DataFrame({
        'Feature': X_numeric.columns,
        'MI_Score': mi_scores
    }).sort_values('MI_Score', ascending=False)
    
    return mi_df

# Check scores
mi_scores = check_mi_scores(df)
print("\nMutual Information Scores:")
print(mi_scores)

# Plot the MI scores
plt.figure(figsize=(12, 8))
sns.barplot(data=mi_scores, x='MI_Score', y='Feature', palette='viridis')
plt.title('Mutual Information Scores by Feature', fontsize=16)
plt.xlabel('MI Score', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.tight_layout()
plt.show()



from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Function to apply one-hot encoding
def one_hot_encode_and_add(df, column, one_hot_encoder=None):
    if one_hot_encoder is None:
        one_hot_encoder = OneHotEncoder(sparse_output=False)
        one_hot_encoded = one_hot_encoder.fit_transform(df[[column]])
    else:
        one_hot_encoded = one_hot_encoder.transform(df[[column]])
    
    encoded_columns = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out([column]))
    encoded_columns.index = df.index  # Ensure index alignment
    df = pd.concat([df, encoded_columns], axis=1)  # Add the encoded columns
    df = df.drop(columns=[column])  # Drop the original column
    return df, one_hot_encoder

# Combine df and test temporarily
combined_df = pd.concat([df, test], axis=0)

# List of columns to one-hot encode
columns_to_encode = [
    'Is_current_loyalty_program_member',
    'product_category',
    'payment_method',
    'shipping_method',
]

# Perform one-hot encoding on each column in `combined_df`
one_hot_encoders = {}
for col in columns_to_encode:
    combined_df, encoder = one_hot_encode_and_add(combined_df, col)
    one_hot_encoders[col] = encoder  # Store the encoder for future use

# Separate the combined_df back into df and test
df = combined_df.iloc[:len(df)].reset_index(drop=True)
test = combined_df.iloc[len(df):].reset_index(drop=True)

# Drop the 'customer_experience' column from the test dataframe
test = test.drop(columns=['customer_experience'], errors='ignore') 


from sklearn.model_selection import train_test_split

X = df.drop(columns=['customer_experience'], axis=1)  
y = df['customer_experience']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.metrics import f1_score
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Step 3: Initialize the classifiers
# lgb_model = LGBMClassifier(objective='multiclass', num_class=3, random_state=42)
lgb_model = LGBMClassifier(objective='multiclass', num_class=3,learning_rate=0.1,n_estimators=90,num_leaves=64,feature_fraction=0.9,bagging_fraction=0.9,lambda_l1=0.1,lambda_l2=0.1,random_state=42)
catboost_model = CatBoostClassifier(loss_function='MultiClass', random_state=42, verbose=0)
xgb_model = XGBClassifier(objective='multi:softmax', num_class=3, random_state=42)
mlp_model = MLPClassifier(random_state=42)
rf_model = RandomForestClassifier(random_state=42)
gb_model = GradientBoostingClassifier(random_state=42)
# Train the model
lgb_model.fit(X_train, y_train)

# Step 5: Make predictions on the test set
y_pred = lgb_model.predict(X_test)

# Step 6: Evaluate the model using F1 score
f1 = f1_score(y_test, y_pred, average='weighted')  # Use 'weighted' for multi-class classification
print(f"F1 Score: {f1:.4f}")


lgb_model.fit(X, y)
predictions = lgb_model.predict(test)
predictions


submission = pd.read_csv('/kaggle/input/Predict-the-Customer-Satisfaction-CSE-22/sample_submission.csv')
submission


submission['customer_experience'] = predictions


submission['customer_experience'] = submission['customer_experience'].replace(0,'bad')
submission['customer_experience'] = submission['customer_experience'].replace(1,'neutral')
submission['customer_experience'] = submission['customer_experience'].replace(2,'good')


submission.to_csv("submission.csv", index=False)

