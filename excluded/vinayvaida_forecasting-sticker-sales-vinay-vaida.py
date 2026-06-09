import pandas as pd
import seaborn as sns
df = pd.read_csv(r"/kaggle/input/playground-series-s5e1/train.csv")
df


df.describe(include="all").T


import matplotlib.pyplot as plt
import seaborn as sns

# Subplot layout: 1 row, 2 columns
fig, axes = plt.subplots(1, 2, figsize=(18, 6))  # 1 row, 2 columns

# ===== Subplot 1: KDE for All Stores =====
# Print store value counts for reference
print("Store value counts (All Stores):")
print(df['store'].value_counts())

# Plot KDE for each unique store in the dataset
for col in df['store'].unique().tolist():
    sns.kdeplot(df[df['store'] == col]['num_sold'], label=col, ax=axes[0])

# Subplot 1 settings
axes[0].legend(title="Store")
axes[0].set_xlabel("Number Sold")
axes[0].set_ylabel("Density")
axes[0].set_title("KDE Plot: All Stores")


# ===== Subplot 2: KDE Excluding "Discount Stickers" =====
# Filter out rows where store is "Discount Stickers"
df_notDiscount = df.query('store != "Discount Stickers"')



# Plot KDE for each remaining store
for col in df_notDiscount['store'].unique().tolist():
    sns.kdeplot(df_notDiscount[df_notDiscount['store'] == col]['num_sold'], label=col, ax=axes[1])

# Subplot 2 settings
axes[1].legend(title="Store")
axes[1].set_xlabel("Number Sold")
axes[1].set_ylabel("Density")
axes[1].set_title("KDE Plot: Excluding 'Discount Stickers'")


# ===== Adjust layout and show the plots =====
plt.tight_layout()  # Adjust spacing between subplots
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Subplot layout: 1 row, 2 columns
fig, axes = plt.subplots(1, 2, figsize=(18, 6))  # 1 row, 2 columns

# ===== Subplot 1: KDE for All Products =====
# Print value counts for reference
print("Product value counts (All Products):")
print(df['product'].value_counts())

# Plot KDE for each unique product in the dataset
for col in df['product'].unique().tolist():
    sns.kdeplot(df[df['product'] == col]['num_sold'], label=col, ax=axes[0])

# Subplot 1 settings
axes[0].legend(title="Product")
axes[0].set_xlabel("Number Sold")
axes[0].set_ylabel("Density")
axes[0].set_title("KDE Plot: All Products")


# ===== Subplot 2: KDE Excluding "Holographic Goose" =====
# Filter out rows where product is "Holographic Goose"
df_NotHoloGoose = df.query('product != "Holographic Goose"')


# Plot KDE for each remaining product
for col in df_NotHoloGoose['product'].unique().tolist():
    sns.kdeplot(df_NotHoloGoose[df_NotHoloGoose['product'] == col]['num_sold'], label=col, ax=axes[1])

# Subplot 2 settings
axes[1].legend(title="Product")
axes[1].set_xlabel("Number Sold")
axes[1].set_ylabel("Density")
axes[1].set_title("KDE Plot: Excluding 'Holographic Goose'")


# ===== Adjust layout and show the plots =====
plt.tight_layout()  # Adjust spacing between subplots
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

# Subplot layout: 1 row, 2 columns
fig, axes = plt.subplots(1, 2, figsize=(18, 6))  # 1 row, 2 columns, larger figure size for better readability

# ===== Subplot 1: KDE for All Countries =====
# Print value counts for reference
print(df.country.value_counts())

# Plot KDE for each unique country in the dataset
for col in df['country'].unique().tolist():
    sns.kdeplot(df[df['country'] == col]['num_sold'], label=col, ax=axes[0])

# Subplot 1 settings
axes[0].legend(title="Country")
axes[0].set_xlabel("Number Sold")
axes[0].set_ylabel("Density")
axes[0].set_title("KDE Plot: All Countries")


# ===== Subplot 2: KDE Excluding Kenya and Norway =====
# Filter out rows where country is 'Kenya' or 'Norway'
df_NotKenyaNorway = df.query('country != "Kenya" and country != "Norway"')

# Plot KDE for each remaining country
for col in df_NotKenyaNorway['country'].unique().tolist():
    sns.kdeplot(df_NotKenyaNorway[df_NotKenyaNorway['country'] == col]['num_sold'], label=col, ax=axes[1])

# Subplot 2 settings
axes[1].legend(title="Country")
axes[1].set_xlabel("Number Sold")
axes[1].set_ylabel("Density")
axes[1].set_title("KDE Plot: Excluding Kenya and Norway")


# ===== Adjust layout and show the plots =====
plt.tight_layout()  # Adjust spacing between subplots
plt.show()



for col in ["Kenya"]:
    sns.kdeplot(df[df['country'] == col]['num_sold'], label=col)


import matplotlib.pyplot as plt
import seaborn as sns



# Filter data with updated names for Kenya
df_Kenya = df.query('country == "Kenya"')  # Kenya (all products)
df_Kenya_Holo = df.query('country == "Kenya" and product == "Holographic Goose"')  # Kenya and Holographic Goose
df_Kenya_NotHolo = df.query('country == "Kenya" and product != "Holographic Goose"')  # Kenya excluding Holographic Goose

# Create a figure with 6 subplots (3 rows, 2 columns)
fig, axes = plt.subplots(3, 2, figsize=(14, 15))  # 3 rows, 2 columns
# Define a vibrant, eye-catching color palette
palette = sns.color_palette("Paired", n_colors=len(df_Kenya['store'].unique()) + len(df_Kenya['product'].unique()))
### Row 1: All products in Kenya
# KDE for stores in df_Kenya
for i, col in enumerate(df_Kenya['store'].unique().tolist()):
    sns.kdeplot(df_Kenya[df_Kenya['store'] == col]['num_sold'], label=col, ax=axes[0, 0], color=palette[i])
axes[0, 0].legend(title="Store")
axes[0, 0].set_title("KDE: Stores ( Products = All Products)")
axes[0, 0].set_xlabel("num_sold")
axes[0, 0].set_ylabel("Density")

# KDE for products in df_Kenya
for i, col in enumerate(df_Kenya['product'].unique().tolist()):
    sns.kdeplot(df_Kenya[df_Kenya['product'] == col]['num_sold'], label=col, ax=axes[0, 1], color=palette[i + len(df_Kenya['store'].unique())])
axes[0, 1].legend(title="Product")
axes[0, 1].set_title("KDE: Products ( Products = All Products)")
axes[0, 1].set_xlabel("num_sold")
axes[0, 1].set_ylabel("Density")

### Row 2: Kenya with Holographic Goose
# KDE for stores in df_Kenya_Holo
for i, col in enumerate(df_Kenya_Holo['store'].unique().tolist()):
    sns.kdeplot(df_Kenya_Holo[df_Kenya_Holo['store'] == col]['num_sold'], label=col, ax=axes[1, 0], color=palette[i])
axes[1, 0].legend(title="Store")
axes[1, 0].set_title("KDE: Stores ( Products = Only Holographic Goose)")
axes[1, 0].set_xlabel("num_sold")
axes[1, 0].set_ylabel("Density")

# KDE for products in df_Kenya_Holo
for i, col in enumerate(df_Kenya_Holo['product'].unique().tolist()):
    sns.kdeplot(df_Kenya_Holo[df_Kenya_Holo['product'] == col]['num_sold'], label=col, ax=axes[1, 1], color=palette[i + len(df_Kenya_Holo['store'].unique())])
axes[1, 1].legend(title="Product")
axes[1, 1].set_title("KDE: Products ( Products = only Holographic Goose)")
axes[1, 1].set_xlabel("num_sold")
axes[1, 1].set_ylabel("Density")

### Row 3: Kenya excluding Holographic Goose
# KDE for stores in df_Kenya_NotHolo
for i, col in enumerate(df_Kenya_NotHolo['store'].unique().tolist()):
    sns.kdeplot(df_Kenya_NotHolo[df_Kenya_NotHolo['store'] == col]['num_sold'], label=col, ax=axes[2, 0], color=palette[i])
axes[2, 0].legend(title="Store")
axes[2, 0].set_title("KDE: Stores ( Products = not Holographic Goose)")
axes[2, 0].set_xlabel("num_sold")
axes[2, 0].set_ylabel("Density")

# KDE for products in df_Kenya_NotHolo
for i, col in enumerate(df_Kenya_NotHolo['product'].unique().tolist()):
    sns.kdeplot(df_Kenya_NotHolo[df_Kenya_NotHolo['product'] == col]['num_sold'], label=col, ax=axes[2, 1], color=palette[i + len(df_Kenya_NotHolo['store'].unique())])
axes[2, 1].legend(title="Product")
axes[2, 1].set_title("KDE: Products ( Products = not Holographic Goose)")
axes[2, 1].set_xlabel("num_sold")
axes[2, 1].set_ylabel("Density")

# Adjust layout for better spacing
plt.tight_layout()

# Show the plots
plt.show()



for col in ["Norway"]:
    sns.kdeplot(df[df['country'] == col]['num_sold'], label=col)


import matplotlib.pyplot as plt
import seaborn as sns



# Filter data with updated names for Norway
df_norway = df.query('country == "Norway"')  # Norway (all products)
df_norway_Holo = df.query('country == "Norway" and product == "Holographic Goose"')  # Norway and Holographic Goose
df_norway_NotHolo = df.query('country == "Norway" and product != "Holographic Goose"')  # Norway excluding Holographic Goose

# Create a figure with 6 subplots (3 rows, 2 columns)
fig, axes = plt.subplots(3, 2, figsize=(14, 15))  # 3 rows, 2 columns
# Define a vibrant, eye-catching color palette
palette = sns.color_palette("Paired", n_colors=len(df_norway['store'].unique()) + len(df_norway['product'].unique()))
### Row 1: All products in Norway
# KDE for stores in df_norway
for i, col in enumerate(df_norway['store'].unique().tolist()):
    sns.kdeplot(df_norway[df_norway['store'] == col]['num_sold'], label=col, ax=axes[0, 0], color=palette[i])
axes[0, 0].legend(title="Store")
axes[0, 0].set_title("KDE: Stores (Country = Norway, Products = All Products)")
axes[0, 0].set_xlabel("num_sold")
axes[0, 0].set_ylabel("Density")

# KDE for products in df_norway
for i, col in enumerate(df_norway['product'].unique().tolist()):
    sns.kdeplot(df_norway[df_norway['product'] == col]['num_sold'], label=col, ax=axes[0, 1], color=palette[i + len(df_norway['store'].unique())])
axes[0, 1].legend(title="Product")
axes[0, 1].set_title("KDE: Products (Country = Norway, Products = All Products)")
axes[0, 1].set_xlabel("num_sold")
axes[0, 1].set_ylabel("Density")

### Row 2: Norway with Holographic Goose
# KDE for stores in df_norway_Holo
for i, col in enumerate(df_norway_Holo['store'].unique().tolist()):
    sns.kdeplot(df_norway_Holo[df_norway_Holo['store'] == col]['num_sold'], label=col, ax=axes[1, 0], color=palette[i])
axes[1, 0].legend(title="Store")
axes[1, 0].set_title("KDE: Stores (Country = Norway, Products = Only Holographic Goose)")
axes[1, 0].set_xlabel("num_sold")
axes[1, 0].set_ylabel("Density")

# KDE for products in df_norway_Holo
for i, col in enumerate(df_norway_Holo['product'].unique().tolist()):
    sns.kdeplot(df_norway_Holo[df_norway_Holo['product'] == col]['num_sold'], label=col, ax=axes[1, 1], color=palette[i + len(df_norway_Holo['store'].unique())])
axes[1, 1].legend(title="Product")
axes[1, 1].set_title("KDE: Products (Country = Norway, Products = only Holographic Goose)")
axes[1, 1].set_xlabel("num_sold")
axes[1, 1].set_ylabel("Density")

### Row 3: Norway excluding Holographic Goose
# KDE for stores in df_norway_NotHolo
for i, col in enumerate(df_norway_NotHolo['store'].unique().tolist()):
    sns.kdeplot(df_norway_NotHolo[df_norway_NotHolo['store'] == col]['num_sold'], label=col, ax=axes[2, 0], color=palette[i])
axes[2, 0].legend(title="Store")
axes[2, 0].set_title("KDE: Stores (Country = Norway, Products = not Holographic Goose)")
axes[2, 0].set_xlabel("num_sold")
axes[2, 0].set_ylabel("Density")

# KDE for products in df_norway_NotHolo
for i, col in enumerate(df_norway_NotHolo['product'].unique().tolist()):
    sns.kdeplot(df_norway_NotHolo[df_norway_NotHolo['product'] == col]['num_sold'], label=col, ax=axes[2, 1], color=palette[i + len(df_norway_NotHolo['store'].unique())])
axes[2, 1].legend(title="Product")
axes[2, 1].set_title("KDE: Products (Country = Norway, Products = not Holographic Goose)")
axes[2, 1].set_xlabel("num_sold")
axes[2, 1].set_ylabel("Density")

# Adjust layout for better spacing
plt.tight_layout()

# Show the plots
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

# Filter data for Norway and Kenya (excluding Holographic Goose and Discount Stickers)
df_norway_NotHolo = df.query('country == "Norway" and product != "Holographic Goose" and store != "Discount Stickers"')
df_kenya_NotHolo = df.query('country == "Kenya" and product != "Holographic Goose" and store != "Discount Stickers"')

# Create a figure with 3 rows and 2 columns
fig, axes = plt.subplots(3, 2, figsize=(16, 18))  # 3 rows, 2 columns

### Row 1: KDE for Stores ###
# Norway
for col in df_norway_NotHolo['store'].unique().tolist():
    sns.kdeplot(df_norway_NotHolo[df_norway_NotHolo['store'] == col]['num_sold'], label=col, ax=axes[0, 0])
axes[0, 0].legend(title="Store")
axes[0, 0].set_title("KDE: Stores (Norway, Excluding Holographic Goose)")
axes[0, 0].set_xlabel("num_sold")
axes[0, 0].set_ylabel("Density")

# Kenya
for col in df_kenya_NotHolo['store'].unique().tolist():
    sns.kdeplot(df_kenya_NotHolo[df_kenya_NotHolo['store'] == col]['num_sold'], label=col, ax=axes[0, 1])
axes[0, 1].legend(title="Store")
axes[0, 1].set_title("KDE: Stores (Kenya, Excluding Holographic Goose)")
axes[0, 1].set_xlabel("num_sold")
axes[0, 1].set_ylabel("Density")

### Row 2: KDE for Products ###
# Norway
for col in df_norway_NotHolo['product'].unique().tolist():
    sns.kdeplot(df_norway_NotHolo[df_norway_NotHolo['product'] == col]['num_sold'], label=col, ax=axes[1, 0])
axes[1, 0].legend(title="Product")
axes[1, 0].set_title("KDE: Products (Norway, Excluding Holographic Goose)")
axes[1, 0].set_xlabel("num_sold")
axes[1, 0].set_ylabel("Density")

# Kenya
for col in df_kenya_NotHolo['product'].unique().tolist():
    sns.kdeplot(df_kenya_NotHolo[df_kenya_NotHolo['product'] == col]['num_sold'], label=col, ax=axes[1, 1])
axes[1, 1].legend(title="Product")
axes[1, 1].set_title("KDE: Products (Kenya, Excluding Holographic Goose)")
axes[1, 1].set_xlabel("num_sold")
axes[1, 1].set_ylabel("Density")

### Row 3: KDE for Countries ###
# Norway
sns.kdeplot(df_norway_NotHolo['num_sold'], label="Norway", ax=axes[2, 0])
axes[2, 0].legend(title="Country")
axes[2, 0].set_title("KDE: num_sold (Norway, Excluding Holographic Goose)")
axes[2, 0].set_xlabel("num_sold")
axes[2, 0].set_ylabel("Density")

# Kenya
sns.kdeplot(df_kenya_NotHolo['num_sold'], label="Kenya", ax=axes[2, 1])
axes[2, 1].legend(title="Country")
axes[2, 1].set_title("KDE: num_sold (Kenya, Excluding Holographic Goose)")
axes[2, 1].set_xlabel("num_sold")
axes[2, 1].set_ylabel("Density")

# Adjust layout for better spacing
plt.tight_layout()

# Show the plots
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

# Filtered data
df_NotHolo_notDiscount = df.query('country != "Norway" and country != "Kenya" and product != "Holographic Goose" and store != "Discount Stickers"')

# Create a figure with 3 rows for country, product, and store
fig, axes = plt.subplots(3, 1, figsize=(12, 15))  # 3 rows, 1 column

### Row 1: KDE for Countries ###
for col in df_NotHolo_notDiscount['country'].unique().tolist():
    sns.kdeplot(df_NotHolo_notDiscount[df_NotHolo_notDiscount['country'] == col]['num_sold'], label=col, ax=axes[0])
axes[0].legend(title="Country")
axes[0].set_title("KDE: num_sold by Country (Excluding Norway, Kenya, Holographic Goose & Discount Stickers)")
axes[0].set_xlabel("num_sold")
axes[0].set_ylabel("Density")

### Row 2: KDE for Products ###
for col in df_NotHolo_notDiscount['product'].unique().tolist():
    sns.kdeplot(df_NotHolo_notDiscount[df_NotHolo_notDiscount['product'] == col]['num_sold'], label=col, ax=axes[1])
axes[1].legend(title="Product")
axes[1].set_title("KDE: num_sold by Product (Excluding Norway, Kenya, Holographic Goose & Discount Stickers)")
axes[1].set_xlabel("num_sold")
axes[1].set_ylabel("Density")

### Row 3: KDE for Stores ###
for col in df_NotHolo_notDiscount['store'].unique().tolist():
    sns.kdeplot(df_NotHolo_notDiscount[df_NotHolo_notDiscount['store'] == col]['num_sold'], label=col, ax=axes[2])
axes[2].legend(title="Store")
axes[2].set_title("KDE: num_sold by Store (Excluding Norway, Kenya, Holographic Goose & Discount Stickers)")
axes[2].set_xlabel("num_sold")
axes[2].set_ylabel("Density")

# Adjust layout for better spacing
plt.tight_layout()

# Show the plots
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

# Correct the filters
# Data for countries other than Norway and Kenya, excluding Holographic Goose and Discount Stickers
df_NotHolo_notDiscount = df.query('country != "Norway" and country != "Kenya" and product != "Holographic Goose" and store != "Discount Stickers"')

# Data for only Norway and Kenya, excluding Holographic Goose and Discount Stickers
df_Norway_Kenya_NotHolo_notDiscount = df.query('country == "Norway" or country == "Kenya" and product != "Holographic Goose" and store != "Discount Stickers"')

# Create a figure with 3 subplots (country, product, and store KDE plots)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 1 row, 3 columns

### Plot 1: KDE for Country ###
sns.kdeplot(data=df_Norway_Kenya_NotHolo_notDiscount, x='num_sold', hue='country', ax=axes[0], fill=True)
axes[0].set_title("KDE: num_sold by Country")
axes[0].set_xlabel("num_sold")
axes[0].set_ylabel("Density")

### Plot 2: KDE for Product ###
sns.kdeplot(data=df_Norway_Kenya_NotHolo_notDiscount, x='num_sold', hue='product', ax=axes[1], fill=True)
axes[1].set_title("KDE: num_sold by Product")
axes[1].set_xlabel("num_sold")
axes[1].set_ylabel("Density")

### Plot 3: KDE for Store ###
sns.kdeplot(data=df_Norway_Kenya_NotHolo_notDiscount, x='num_sold', hue='store', ax=axes[2], fill=True)
axes[2].set_title("KDE: num_sold by Store")
axes[2].set_xlabel("num_sold")
axes[2].set_ylabel("Density")

# Adjust layout for better spacing
plt.tight_layout()

# Show the plots
plt.show()



## Machine Learning Model 


df_train = pd.read_csv(r"/kaggle/input/playground-series-s5e1/train.csv")
df_test = pd.read_csv(r"/kaggle/input/playground-series-s5e1/test.csv")
df_test


import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

# Convert 'date' to datetime
df_train['date'] = pd.to_datetime(df_train['date'])
# Extract the features
df_train['month'] = df_train['date'].dt.month
df_train['weekday'] = df_train['date'].dt.weekday  # 0: Monday, 6: Sunday
df_train['dayofweek'] = df_train['date'].dt.dayofweek  # Same as 'weekday
df_train['quarter'] = df_train['date'].dt.quarter
df_train['is_weekend'] = df_train['weekday'] >= 5  # True for Saturday and Sunday


# Convert 'date' to datetime
df_test['date'] = pd.to_datetime(df_test['date'])
# Extract the features
df_test['month'] = df_test['date'].dt.month
df_test['weekday'] = df_test['date'].dt.weekday  # 0: Monday, 6: Sunday
df_test['dayofweek'] = df_test['date'].dt.dayofweek  # Same as 'weekday
df_test['quarter'] = df_test['date'].dt.quarter
df_test['is_weekend'] = df_test['weekday'] >= 5  # True for Saturday and Sunday

# Initialize the KNNImputer with 5 neighbors
imputer = KNNImputer(n_neighbors=5)
# Apply the imputer to the 'num_sold' column
df_train['num_sold'] = imputer.fit_transform(df_train[['num_sold']])

# Apply one-hot encoding to categorical variables
categorical_cols = ['country', 'store', 'product']
df_train_encoded = pd.get_dummies(df_train, columns=categorical_cols, drop_first=True)

# Apply one-hot encoding to categorical variables
categorical_cols = ['country', 'store', 'product']
df_test_encoded = pd.get_dummies(df_test, columns=categorical_cols, drop_first=True)

# Drop unnecessary column
df_train_encoded = df_train_encoded.drop(columns=["id"])
df_test_encoded = df_test_encoded.drop(columns=["id"])

# Display only the first few rows of the transformed dataset
df_train_encoded


df_test_encoded


sns.boxplot(x=df_train_encoded['num_sold'])
plt.title("Box plot for num_sold")
plt.show()


Q1 = df_train_encoded.num_sold.quantile(0.25)
Q3 = df_train_encoded.num_sold.quantile(0.75)
IQR = Q3 - Q1
lower_limit = Q1 - 1.5*IQR
upper_limit = Q3 + 1.5*IQR
df_train_encoded_no_outlier = df_train_encoded[(df_train_encoded.num_sold>lower_limit)&(df_train_encoded.num_sold<upper_limit)]
df_train_encoded_no_outlier


sns.boxplot(x=df_train_encoded_no_outlier['num_sold'])
plt.title("Box plot for num_sold")
plt.show()


plt.figure(figsize=(10, 6))
sns.histplot(df_train_encoded_no_outlier['num_sold'], kde=True, bins=50, color='blue')
plt.title('Distribution of num_sold', fontsize=16)
plt.xlabel('num_sold', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.show()


X_train=df_train_encoded_no_outlier.drop(labels=['num_sold',"date"], axis=1).astype(int)
y_train=df_train_encoded_no_outlier['num_sold'].astype(int)


X_train


y_train


X_test =df_test_encoded.drop(labels=["date"], axis=1).astype(int)
X_test


from sklearn.preprocessing import RobustScaler
scaler =  RobustScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled


print(X_train_scaled.shape)
print(X_test_scaled.shape)


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Create a dictionary of models
models = {
    "Random Forest": RandomForestRegressor(random_state=42),
}

# Assuming X_train and y_train are already defined (training data)
# Assuming X_test_scaled is defined (for prediction)

# Fit the RandomForest model
model = models["Random Forest"]
model.fit(X_train, y_train)

# Calculate accuracy score on training set
train_accuracy = model.score(X_train_scaled, y_train)

# Print the model name and training accuracy
print("Model:", "Random Forest")
print("Accuracy score on training set: {:.4f}".format(train_accuracy))

# Predict on the test set
y_pred_test = model.predict(X_test_scaled)
y_pred_test


df_test["num_sold"] = y_pred_test
k = df_test[['id','num_sold']]
k.to_csv("submission.csv", index=False)
print("submission saved!")

