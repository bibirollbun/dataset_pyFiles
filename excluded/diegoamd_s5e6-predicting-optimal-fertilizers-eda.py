import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


train_set = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
train_set.head()


train_set.info()


train_set.describe()


train_set.describe(include = "O")


train_set["Temperature"] = train_set["Temparature"]
train_set = train_set.drop(columns = ["Temparature"])


sns.boxplot(data = train_set, x = "Temperature")
plt.title("Temperature Boxplot")
plt.show()


sns.countplot(data = train_set, x = "Temperature", color = "lightgray")
plt.title("Temperature Distribution")
plt.show()


sns.boxplot(data = train_set, x = "Humidity")
plt.title("Humidity Boxplot")
plt.show()


sns.countplot(data = train_set, x = "Humidity", color = "lightgray")
plt.title("Humidity Distribution")
plt.show()


sns.boxplot(data = train_set, x = "Moisture")
plt.title("Moisture Boxplot")
plt.show()


sns.histplot(data = train_set, x = "Moisture", color = "lightgray")
plt.title("Moisture Distribution")
plt.show()


sns.boxplot(data = train_set, x = "Nitrogen")
plt.title("Nitrogen Boxplot")
plt.show()


sns.histplot(data = train_set, x = "Nitrogen", color = "lightgray")
plt.title("Nitrogen Distribution")
plt.show()


sns.boxplot(data = train_set, x = "Potassium")
plt.title("Potassium Boxplot")
plt.show()


sns.countplot(data = train_set, x = "Potassium", color = "lightgray")
plt.title("Potassium Distribution")
plt.show()


sns.boxplot(data = train_set, x = "Phosphorous")
plt.title("Phosphorous Boxplot")
plt.show()


sns.histplot(data = train_set, x = "Phosphorous", color = "lightgray")
plt.title("Phosphorous Distribution")
plt.show()


sns.countplot(data = train_set, x = "Soil Type", color = "lightgray")
plt.title("Soil Type Distribution")
plt.show()


fig, ax = plt.subplots(figsize = (10, 4))
sns.countplot(data = train_set, x = "Crop Type", color = "lightgray")
plt.title("Crop Type Distribution")
plt.show()


sns.countplot(data = train_set, x = "Soil Type", color = "lightgray")
plt.title("Soil Type Distribution")
plt.show()


column_names = train_set.columns
print(column_names)


numerical_features = ['Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous', 'Temperature']
categorical_features = ['Soil Type', 'Crop Type', 'Fertilizer Name']


num_corr_matrix = train_set[numerical_features].corr()

sns.heatmap(num_corr_matrix, cmap = "coolwarm", annot = True)
plt.title("Numerical Features Correlation Matrix")
plt.show()


def violin(num_feat, cat_feat, data):
    sns.violinplot(x = num_feat, y = cat_feat, data = data)
    plt.title(f"{num_feat} Distribution by {cat_feat}")
    plt.show()


for num_feat in numerical_features:
    for cat_feat in categorical_features:
        violin(num_feat, cat_feat, train_set)


print(categorical_features)


sns.countplot(y = 'Soil Type', hue = 'Crop Type', data = train_set)
plt.title("Soil Type Count by Crop Type")
plt.show()


sns.countplot(y = 'Crop Type', hue = 'Soil Type', data = train_set)
plt.title("Crop Type Count by Soil Type")
plt.show()


sns.countplot(y = 'Crop Type', hue = 'Fertilizer Name', data = train_set)
plt.title("Crop Type Count by Fertilizer Name")
plt.show()


sns.countplot(y = 'Fertilizer Name', hue = 'Crop Type', data = train_set)
plt.title("Fertilizer Name Count by Crop Type")
plt.show()


sns.countplot(y = 'Soil Type', hue = 'Fertilizer Name', data = train_set)
plt.title("Soil Type Count by Fertilizer Name")
plt.show()


sns.countplot(y = 'Fertilizer Name', hue = 'Soil Type', data = train_set)
plt.title("Fertilizer Name Count by Soil Type")
plt.show()


# Get unique fertilizers
fertilizers = train_set['Fertilizer Name'].unique()

# Create a pivot table for each fertilizer
for fertilizer in fertilizers:
    # Filter data for the current fertilizer
    subset = train_set[train_set['Fertilizer Name'] == fertilizer]
    
    # Create pivot table
    pivot = pd.pivot_table(subset, 
                           values='Fertilizer Name', 
                           index='Nitrogen', 
                           columns='Potassium', 
                           aggfunc='count', 
                           fill_value=0)
    
    print(f"Fertilizer Name: {fertilizer}")
    sns.heatmap(pivot)
    plt.show()


train_set.min()


train_set.max()


# Bin the Nitrogen and Potassium columns
train_set['Nitrogen_binned'] = pd.cut(train_set['Nitrogen'], bins=4, include_lowest=True, right=False)
train_set['Potassium_binned'] = pd.cut(train_set['Potassium'], bins=4, include_lowest=True, right=False)

# Get unique fertilizers
fertilizers = train_set['Fertilizer Name'].unique()

# Create a pivot table for each fertilizer
for fertilizer in fertilizers:
    # Filter data for the current fertilizer
    subset = train_set[train_set['Fertilizer Name'] == fertilizer]
    
    # Create pivot table
    pivot = pd.pivot_table(subset, 
                           values='Fertilizer Name', 
                           index='Nitrogen_binned', 
                           columns='Potassium_binned', 
                           aggfunc='count', 
                           fill_value=0,
                           observed=False)
    
    print(f"Fertilizer Name: {fertilizer}")
    sns.heatmap(pivot, annot = True, cmap = "coolwarm", fmt='g')
    plt.show()


# Bin the Nitrogen and Potassium columns
train_set['Nitrogen_binned'] = pd.cut(train_set['Nitrogen'], bins=4, include_lowest=True, right=False)
train_set['Phosphorous_binned'] = pd.cut(train_set['Phosphorous'], bins=4, include_lowest=True, right=False)

# Get unique fertilizers
fertilizers = train_set['Fertilizer Name'].unique()

# Create a pivot table for each fertilizer
for fertilizer in fertilizers:
    # Filter data for the current fertilizer
    subset = train_set[train_set['Fertilizer Name'] == fertilizer]
    
    # Create pivot table
    pivot = pd.pivot_table(subset, 
                           values='Fertilizer Name', 
                           index='Nitrogen_binned', 
                           columns='Phosphorous_binned', 
                           aggfunc='count', 
                           fill_value=0,
                           observed=False)
    
    print(f"Fertilizer Name: {fertilizer}")
    sns.heatmap(pivot, annot = True, cmap = "coolwarm", fmt='g')
    plt.show()


# Bin the Nitrogen and Potassium columns
train_set['Potassium_binned'] = pd.cut(train_set['Potassium'], bins=4, include_lowest=True, right=False)
train_set['Phosphorous_binned'] = pd.cut(train_set['Phosphorous'], bins=4, include_lowest=True, right=False)

# Get unique fertilizers
fertilizers = train_set['Fertilizer Name'].unique()

# Create a pivot table for each fertilizer
for fertilizer in fertilizers:
    # Filter data for the current fertilizer
    subset = train_set[train_set['Fertilizer Name'] == fertilizer]
    
    # Create pivot table
    pivot = pd.pivot_table(subset, 
                           values='Fertilizer Name', 
                           index='Potassium_binned', 
                           columns='Phosphorous_binned', 
                           aggfunc='count', 
                           fill_value=0,
                           observed=False)
    
    print(f"Fertilizer Name: {fertilizer}")
    sns.heatmap(pivot, annot = True, cmap = "coolwarm", fmt='g')
    plt.show()




