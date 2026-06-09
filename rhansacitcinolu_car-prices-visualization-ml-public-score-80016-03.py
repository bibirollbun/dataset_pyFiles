import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import re 
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score


# Change the settings to show all rows
pd.set_option('display.max_rows', None)


# # Load datasets
train_df = pd.read_csv('/kaggle/input/playground-series-s4e9/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s4e9/test.csv')

# Veri çerçevelerini birleştirin
df = pd.concat([train_df, test_df], ignore_index=True)


# first 5 rows
df.head()


test_df.head()


# last 2 rows
df.tail(2)


# check the number of rows and columns
df.shape


test_df.shape


# Check general information about data
df.info()


# statistics 
df.describe() 


# check empty data
df.isnull().sum()


df['fuel_type'].value_counts()


# Filtering vehicles with 'not supported' in the 'fuel_type' column
not_supported_cars = train_df[df['fuel_type'] == 'not supported']
not_supported_cars.head()


df['ext_col'].value_counts()


df['accident'].value_counts()


df['clean_title'].value_counts()


# Fill missing values in 'accident' and 'clean_title' columns
df['accident'].fillna('None reported', inplace=True)
df['clean_title'].fillna('No', inplace=True)


# Get the most common fuel type for each brand
most_common_fuel = df.groupby('brand')['fuel_type'].agg(lambda x: x.mode()[0] if not x.mode().empty else None)

# Fill missing values in 'fuel_type' based on the brand's most common fuel type
df['fuel_type'] = df.apply(
    lambda row: most_common_fuel[row['brand']] if pd.isna(row['fuel_type']) else row['fuel_type'], axis=1)


# Fill missing values in train_df
train_df['accident'].fillna('None reported', inplace=True)
train_df['clean_title'].fillna('No', inplace=True)

# Get the most common fuel type for each brand in train_df
most_common_fuel = train_df.groupby('brand')['fuel_type'].agg(lambda x: x.mode()[0] if not x.mode().empty else None)

# Fill missing values in 'fuel_type' for train_df
train_df['fuel_type'] = train_df.apply(
    lambda row: most_common_fuel[row['brand']] if pd.isna(row['fuel_type']) else row['fuel_type'], axis=1)

# Repeat the same for test_df
test_df['accident'].fillna('None reported', inplace=True)
test_df['clean_title'].fillna('No', inplace=True)

most_common_fuel_test = test_df.groupby('brand')['fuel_type'].agg(lambda x: x.mode()[0] if not x.mode().empty else None)

test_df['fuel_type'] = test_df.apply(
    lambda row: most_common_fuel_test[row['brand']] if pd.isna(row['fuel_type']) else row['fuel_type'], axis=1)

# Combine the DataFrames
df = pd.concat([train_df, test_df], ignore_index=True)


# Horsepower sütununu oluşturma
def extract_horsepower(engine):
    match = re.search(r'(\d+\.?\d*)HP', engine)
    if match:
        return float(match.group(1))
    return np.nan

# Engine_Size sütununu oluşturma
def extract_engine_size(engine):
    match_l = re.search(r'(\d+\.?\d*)L', engine)
    if match_l:
        return float(match_l.group(1))
    
    match_liter = re.search(r'(\d+\.?\d*)\sLiter', engine)
    if match_liter:
        return float(match_liter.group(1))
    
    return np.nan

# Number_of_Cylinders sütununu oluşturma
def extract_number_of_cylinders(engine):
    # "X Cylinder" formatı
    match_cylinder = re.search(r'(\d+)\sCylinder', engine)
    if match_cylinder:
        return int(match_cylinder.group(1))
    
    # "V6" veya "V8" formatı
    match_v = re.search(r'V(\d+)', engine)
    if match_v:
        return int(match_v.group(1))
    
    return np.nan

# train_df için sütunları ekleme
train_df['Horsepower'] = train_df['engine'].apply(extract_horsepower)
train_df['Engine_Size'] = train_df['engine'].apply(extract_engine_size)
train_df['Number_of_Cylinders'] = train_df['engine'].apply(extract_number_of_cylinders)

# test_df için sütunları ekleme
test_df['Horsepower'] = test_df['engine'].apply(extract_horsepower)
test_df['Engine_Size'] = test_df['engine'].apply(extract_engine_size)
test_df['Number_of_Cylinders'] = test_df['engine'].apply(extract_number_of_cylinders)


test_df.isnull().sum()


train_df.isnull().sum()


# Function to fill missing values
def fill_missing_values(df):
    le_brand = LabelEncoder()
    le_fuel_type = LabelEncoder()
    
    # Encode the categorical features
    df['brand_encoded'] = le_brand.fit_transform(df['brand'])
    df['fuel_type_encoded'] = le_fuel_type.fit_transform(df['fuel_type'])
    
    # Horsepower
    if df['Horsepower'].notna().sum() > 0:
        hp_train = df[df['Horsepower'].notna()]
        hp_test = df[df['Horsepower'].isna()]

        X_hp = hp_train[['brand_encoded', 'fuel_type_encoded']]
        y_hp = hp_train['Horsepower']
        
        model_hp = RandomForestRegressor(n_estimators=100, random_state=42)
        model_hp.fit(X_hp, y_hp)
        
        if not hp_test.empty:
            df.loc[df['Horsepower'].isna(), 'Horsepower'] = model_hp.predict(hp_test[['brand_encoded', 'fuel_type_encoded']])
    
    # Engine Size
    if df['Engine_Size'].notna().sum() > 0:
        es_train = df[df['Engine_Size'].notna()]
        es_test = df[df['Engine_Size'].isna()]

        X_es = es_train[['brand_encoded', 'fuel_type_encoded']]
        y_es = es_train['Engine_Size']
        
        model_es = RandomForestRegressor(n_estimators=100, random_state=42)
        model_es.fit(X_es, y_es)
        
        if not es_test.empty:
            df.loc[df['Engine_Size'].isna(), 'Engine_Size'] = model_es.predict(es_test[['brand_encoded', 'fuel_type_encoded']])
    
    # Number of Cylinders
    if df['Number_of_Cylinders'].notna().sum() > 0:
        nc_train = df[df['Number_of_Cylinders'].notna()]
        nc_test = df[df['Number_of_Cylinders'].isna()]

        X_nc = nc_train[['brand_encoded', 'fuel_type_encoded']]
        y_nc = nc_train['Number_of_Cylinders']
        
        model_nc = RandomForestRegressor(n_estimators=100, random_state=42)
        model_nc.fit(X_nc, y_nc)
        
        if not nc_test.empty:
            df.loc[df['Number_of_Cylinders'].isna(), 'Number_of_Cylinders'] = model_nc.predict(nc_test[['brand_encoded', 'fuel_type_encoded']])
    
    # Restore original values
    df['brand'] = le_brand.inverse_transform(df['brand_encoded'])
    df['fuel_type'] = le_fuel_type.inverse_transform(df['fuel_type_encoded'])
    
    # Remove temporary columns
    df.drop(columns=['brand_encoded', 'fuel_type_encoded'], inplace=True)
    
    return df

# Fill missing values
train_df = fill_missing_values(train_df)
test_df = fill_missing_values(test_df)


# Remove the 'engine' column
train_df.drop(columns=['engine'], inplace=True)
test_df.drop(columns=['engine'], inplace=True)


# Apply get_dummies for categorical variables
train_dummies = pd.get_dummies(train_df, drop_first=True)
test_dummies = pd.get_dummies(test_df, drop_first=True)


sns.pairplot(train_df)
plt.title("Pairplot of Features")
plt.show()


sns.boxplot(x='fuel_type', y='Horsepower', data=train_df)
plt.title("Horsepower Distribution by Fuel Type")
plt.xticks(rotation=45)
plt.show()


# Calculate the distribution of fuel types
fuel_counts = train_df['fuel_type'].value_counts()

# Create a pie chart
plt.figure(figsize=(8, 8))  # Set the figure size
plt.pie(fuel_counts, labels=fuel_counts.index, autopct='%1.1f%%', startangle=140)

# Add a legend
plt.legend(fuel_counts.index, title="Fuel Types", loc="upper right", bbox_to_anchor=(1.3, 1))

plt.title("Fuel Type Distribution")
plt.axis('equal')  # Equal aspect ratio for a circular pie chart
plt.tight_layout()  # Adjust layout for better spacing
plt.show()


plt.hist(train_df['Engine_Size'].dropna(), bins=10, color='skyblue', edgecolor='black')
plt.title("Engine Size Distribution")
plt.xlabel("Engine Size (L)")
plt.ylabel("Frequency")
plt.show()


plt.scatter(train_df['Engine_Size'], train_df['Horsepower'], alpha=0.5)
plt.title("Horsepower vs Engine Size")
plt.xlabel("Engine Size (L)")
plt.ylabel("Horsepower")
plt.show()


# Calculate the average horsepower by brand
avg_hp_by_brand = train_df.groupby('brand')['Horsepower'].mean().reset_index()

# Create the bar plot
plt.figure(figsize=(12, 6))  # Set width and height
sns.barplot(x='brand', y='Horsepower', data=avg_hp_by_brand)
plt.title("Average Horsepower by Brand")
plt.xticks(rotation=45)
plt.xlabel("Brand")
plt.ylabel("Average Horsepower")
plt.tight_layout()  # Prevent overlapping of text
plt.show()


# Select numeric columns
numeric_df = train_df.select_dtypes(include=[np.number])

# Calculate the correlation matrix
correlation_matrix = numeric_df.corr()

# Create the heatmap
plt.figure(figsize=(10, 8))  # Set the figure size
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True)
plt.title("Correlation Heatmap")
plt.tight_layout()  # Adjust layout for better spacing
plt.show()


sns.violinplot(x='fuel_type', y='Horsepower', data=train_df)
plt.title("Horsepower Distribution by Fuel Type (Violin Plot)")
plt.xticks(rotation=45)
plt.show()


# Calculate the average horsepower by fuel type
avg_hp_by_fuel = train_df.groupby('fuel_type')['Horsepower'].mean()

# Create the area chart
plt.figure(figsize=(10, 6))  # Set the figure size
avg_hp_by_fuel.plot(kind='area', alpha=0.5)
plt.title("Average Horsepower by Fuel Type")
plt.ylabel("Average Horsepower")
plt.xlabel("Fuel Type")
plt.xticks(rotation=45)
plt.tight_layout()  # Adjust layout for better spacing
plt.show()


sns.countplot(x='fuel_type', data=train_df)
plt.title("Count of Cars by Fuel Type")
plt.xticks(rotation=45)
plt.show()


# Apply get_dummies for categorical variables
train_dummies = pd.get_dummies(train_df, drop_first=True)
test_dummies = pd.get_dummies(test_df, drop_first=True)


# Features and target variable
x = train_dummies.drop(columns=['id', 'price'])  # Features
y = train_dummies['price']  # Target variable

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x, y)

# Make predictions on the test data
# Ensure the test data has the same structure as the training data
x_test_final = test_dummies.reindex(columns=x.columns, fill_value=0)

# Make predictions
predictions = model.predict(x_test_final)

# Save results to a DataFrame
results = pd.DataFrame({
    'id': test_df['id'],  # Include the ID column
    'price': predictions
})

# Save results to a CSV file
# results.to_csv('test_prices.csv', index=False)

# print("Prediction results have been saved to 'test_prices.csv'.")




