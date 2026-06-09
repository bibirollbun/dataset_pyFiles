# from google.colab import files
# import os

# # Check if kaggle.json already exists
# if not os.path.exists('kaggle.json'):
#     uploaded = files.upload()
#     if 'kaggle.json' not in uploaded:
#         print("Error: kaggle.json not uploaded. Please upload the file.")
#     else:
#         print("kaggle.json uploaded successfully.")

# # Create kaggle directory if not exists
# if not os.path.exists('/root/.kaggle'):
#     !mkdir -p ~/.kaggle

# # Copy kaggle.json file to kaggle directory, handling potential errors
# try:
#     !cp kaggle.json ~/.kaggle/
#     !chmod 600 ~/.kaggle/kaggle.json
#     print("kaggle.json configuration successful.")
# except FileNotFoundError:
#     print("Error: kaggle.json file not found. Please upload the file.")




# try:
#     import kaggle
#     print("Kaggle API successfully imported.")
# except ImportError:
#     print("Error: Kaggle API not installed. Please install it using: !pip install kaggle")
#     # Install kaggle
#     !pip install kaggle

# try:
#     !kaggle competitions download -c sf-crime -p /content/sf-crime-data  # Or use !kaggle datasets download -d <dataset_owner>/<dataset_name> -p /content/dataset_name

#     # Optional: Unzip the downloaded file (modify the file name as needed)
#     !unzip -o -q /content/sf-crime-data/sf-crime.zip -d /content/sf-crime-data

#     sf_crime_path = "/content/sf-crime-data"  # Update the path accordingly
#     print(f"Dataset downloaded to: {sf_crime_path}")

# except Exception as e:
#     print(f"Error downloading dataset: {e}")



# import os
# import zipfile

# def unzip_all(directory):
#     """
#     Recursively find and unzip every .zip file under `directory`,
#     deleting each archive after successful extraction.
#     """
#     if not os.path.isdir(directory):
#         print(f"Error: Directory '{directory}' not found.")
#         return

#     while True:
#         zip_paths = []
#         for root, _, files in os.walk(directory):
#             for fname in files:
#                 if fname.lower().endswith('.zip'):
#                     zip_paths.append(os.path.join(root, fname))
#         if not zip_paths:
#             print("No more zip files found. Extraction complete.")
#             break
#         for zip_path in zip_paths:
#             try:
#                 with zipfile.ZipFile(zip_path, 'r') as zf:
#                     extract_dir = os.path.dirname(zip_path)
#                     zf.extractall(extract_dir)
#                 print(f"Unzipped: {zip_path}")
#                 os.remove(zip_path)
#                 print(f"Removed:   {zip_path}")
#             except zipfile.BadZipFile:
#                 print(f"Bad zip file, skipping: {zip_path}")
#             except Exception as e:
#                 print(f"Error with {zip_path}: {e}")
# unzip_all("/content/sf-crime-data")


# Requirements
!pip install numpy pandas seaborn tensorflow keras plotly matplotlib scikit-learn folium imbalanced-learn
from IPython.display import clear_output
clear_output()
print("All packages successfully installed and updated.")



# !pip install -U scikit-learn imbalanced-learn --quiet
# import os
# os.kill(os.getpid(), 9)  # Force restart the notebook runtime



import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras import layers
import plotly.express as px
from tensorflow import keras
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
from folium import Map, Marker, CircleMarker
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from folium.plugins import HeatMap, FastMarkerCluster, MarkerCluster
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")


def plot_crime_distribution(data, x_column, hue_column=None, title=None, order=None, hue_order=None):
    """
    Plots a distribution of crime data using seaborn's countplot.

    Args:
        data: DataFrame containing crime data.
        x_column: Name of the column to plot on the x-axis.
        hue_column: Optional name of the column to use for color encoding (hue).
        title: Optional title for the plot.
        order: Optional list specifying the order of categories on the x-axis.
        hue_order: Optional list specifying the order of categories for the hue.
    """
    plt.figure(figsize=(12, 6))
    sns.countplot(
        data=data, x=x_column, hue=hue_column, order=order, hue_order=hue_order
    )
    plt.title(title)
    plt.xlabel(x_column)
    plt.ylabel("Number of Crimes")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title=hue_column, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.show()


def plot_crime_distribution_by_day(df):
    dow_order = ['Saturday','Sunday','Monday','Tuesday','Wednesday','Thursday','Friday']
    categories = df['Category'].unique()
    num_categories = len(categories)
    cols = 2
    rows = math.ceil(num_categories / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 4), sharex=True, sharey=False)
    axes = axes.flatten()
    for ax, cat in zip(axes, categories):
        sub = df[df['Category'] == cat]
        sns.countplot(x='DayOfWeek', data=sub, order=dow_order, ax=ax)
        ax.set_title(f'{cat} by Day of Week', fontsize=12)
        ax.set_xlabel('Day of Week', fontsize=10)
        ax.set_ylabel('Number of Crimes', fontsize=10)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
    for i in range(num_categories, len(axes)):
        fig.delaxes(axes[i])
    fig.tight_layout()
    plt.show()



def plot_heatmap(data, x_column, y_column, title=None, cmap="viridis"):
    """
    Plots a heatmap of crime data using seaborn's heatmap function.

    Args:
        data: DataFrame containing crime data.
        x_column: Name of the column to plot on the x-axis.
        y_column: Name of the column to plot on the y-axis.
        title: Optional title for the plot.
        cmap: Optional colormap for the heatmap.
    """
    crime_by_xy = data.groupby([x_column, y_column]).size().unstack()
    plt.figure(figsize=(12, 6))
    sns.heatmap(crime_by_xy, cmap=cmap)
    plt.title(title)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.show()


def preprocess_data(df, categorical_cols, numerical_cols):
    """
    Preprocesses the data by label encoding categorical features and scaling numerical features.

    Args:
        df: DataFrame containing the data.
        categorical_cols: List of categorical columns to encode.
        numerical_cols: List of numerical columns to scale.
    """
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df, label_encoders, scaler


data_path = (
    "/kaggle/input/sf-crime/train.csv.zip"
    if os.path.exists("/kaggle/input/sf-crime/train.csv.zip")
    else "/content/sf-crime-data/train.csv"
)

df = pd.read_csv(data_path)


df.head()


df.describe()


df.shape


# to check missing value in dataframe
print(df.isnull().sum())


# List all columns with their data types and non-null counts
Data_info = pd.DataFrame({
    'Data Type': df.dtypes,
    'Non-Null Count': df.count()
})
Data_info


print(df.duplicated().sum())
df.drop_duplicates(keep="first", inplace=True)
df = df.drop(df[(df.X > -122) | (df.X < -123) | (df.Y > 38) | (df.Y < 37)].index)


crime_counts = df['Category'].value_counts()
category_percentages = (crime_counts / len(df)) * 100
# List all columns with their data types and non-null counts
Data_info2 = pd.DataFrame({
    'counts': crime_counts,
    'percentages': category_percentages
})

Data_info2



# Analyze the distribution of crime over time
# Convert 'Dates' to datetime objects
df['Dates'] = pd.to_datetime(df['Dates'])
# Extract year, month, day, hour from Dates
df['Year'] = df['Dates'].dt.year
df['Month'] = df['Dates'].dt.month
df['Day'] = df['Dates'].dt.day
df['Hour'] = df['Dates'].dt.hour
df.drop('Dates', axis=1, inplace=True)


top_10_categories = df['Category'].value_counts().nlargest(10).index
print(top_10_categories.values)


fig = px.bar(crime_counts,
             x=crime_counts.values,
             y=crime_counts.index,
             orientation='h',
             labels={'x': "Number of Crimes", 'y': "Crime Category"},
             title="Distribution of Crime Categories",
             text=crime_counts.values)
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout(yaxis={'categoryorder':'total ascending'})
fig.show()


# Distribution of Crime Categories percentages
fig = px.bar(category_percentages,
             x=category_percentages.values,
             y=category_percentages.index,
             orientation='h',
             labels={'x': 'Percentage of Total Crimes', 'y': 'Crime Category'},
             title='Percentage Distribution of Crime Categories',
             text=category_percentages.values)
fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
fig.update_layout(yaxis={'categoryorder':'total ascending'})
fig.show()



plot_crime_distribution(
    df,
    x_column='DayOfWeek',
    hue_column='Category',
    title='Crime Category Distribution by Day of the Week',
    order=df['DayOfWeek'].value_counts().index,
    hue_order=df['Category'].value_counts().index
)


plot_crime_distribution(
    df,
    x_column='DayOfWeek',
    hue_column='Category',
    title='Top 10 Crime Categories Distribution by Day of the Week',
    order=df['DayOfWeek'].value_counts().index,
    hue_order=top_10_categories
)


plot_crime_distribution(
    df,
    x_column='Day',
    hue_column='Category',
    title='Crime Categories by Day of the Month',
    order=df['Day'].value_counts().index,
    hue_order=df['Category'].value_counts().index
)


plot_crime_distribution(
    df,
    x_column='Day',
    hue_column='Category',
    title='Top 10 Crime Categories by Day of the Month',
    order=range(1, 32),  # Ensure days are in order
    hue_order=top_10_categories
)



plot_crime_distribution(
    df,
    x_column='Month',
    hue_column='Category',
    title='Crime Categories Distribution by Month',
    order=range(1, 13),  # Ensure months are in order
    hue_order=df['Category'].value_counts().index
)


plot_crime_distribution(
    df,
    x_column='Month',
    hue_column='Category',
    title='Top 10 Crime Categories Distribution by Month',
    order=range(1, 13),  # Ensure months are in order
    hue_order=top_10_categories
)


plot_crime_distribution(
    df,
    x_column='Year',
    hue_column='Category',
    title='Crime Categories Distribution by Year',
    order=df['Year'].value_counts().index,
    hue_order=df['Category'].value_counts().index
)


plot_crime_distribution(
    df,
    x_column='Year',
    hue_column='Category',
    title='Top 10 Crime Categories Distribution by Year',
    order=df['Year'].value_counts().index,
    hue_order=top_10_categories
)


plot_crime_distribution(
    df,
    x_column='Year',
    title='Number of Crimes per Year'
)


#Crime Category and PdDistrict
fig = px.histogram(df, x="PdDistrict", color="Category",
                   labels={"PdDistrict":"Police District", "Category":"Crime Category"},
                   title="Crime Category Distribution by Police District",
                   category_orders={"PdDistrict": df["PdDistrict"].value_counts().index})
fig.show()


plot_crime_distribution(
    df,
    x_column='PdDistrict',
    hue_column='Category',
    title='Crime Category Distribution by Police District',
    order=df['PdDistrict'].value_counts().index,
    hue_order=df['Category'].value_counts().index
)


plot_crime_distribution(
    df,
    x_column='PdDistrict',
    hue_column='Category',
    title='Top 10 Crime Category Distribution by Police District',
    order=df['PdDistrict'].value_counts().index,
    hue_order=top_10_categories
)


correlation_matrix = df[['X', 'Y', 'Year', 'Month', 'Day', 'Hour']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()



plot_crime_distribution(
    df,
    x_column='PdDistrict',
    title='Crime Counts per Police District'
)


hourly_crime_counts = df['Hour'].value_counts().sort_index()
plt.figure(figsize=(12, 6))
plt.plot(hourly_crime_counts.index, hourly_crime_counts.values)
plt.title('Hourly Crime Distribution')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Crimes')
plt.xticks(range(24))
plt.grid(True)
plt.show()



plot_crime_distribution(
    df,
    x_column='Hour',
    title='Number of Crimes per Hour'
)


plot_heatmap(df, 'DayOfWeek', 'Hour', title='Crime Counts by Day of Week and Hour')


plot_crime_distribution_by_day(df)


#   map showing crime locations
# Use a smaller subset of the data for the map to improve performance
sample_df = df.sample(n=10000, random_state=42) # Adjust sample size as needed


fig = px.scatter_mapbox(sample_df, lat="Y", lon="X", color="Category",
                        mapbox_style="carto-positron", zoom=10,
                        labels={"Category": "Crime Category"},
                        title="Interactive Crime Map (Sampled Data)",
                        height=600) # Adjust height for better visibility

fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0}) # Reduce margins for a cleaner look
fig.show()



# # center on SF
# m = Map(location=[37.7749, -122.4194], zoom_start=12)

# # build a list of [lat, lon] pairs
# heat_data = df[['Y','X']].values.tolist()

# # add the HeatMap layer
# HeatMap(
#     heat_data,
#     radius=8,    # cluster radius of each â€œpointâ€�
#     blur=15,     # smoothness
#     max_zoom=12
# ).add_to(m)
# m


# # FastMarkerCluster takes a list of [lat, lon] or [lat, lon, popup_html]
# # Hereâ€™s how to include popups (e.g. crime Category):
# cluster_data = df.apply(lambda r: [r['Y'], r['X'], r['Category']], axis=1).tolist()

# FastMarkerCluster(data=cluster_data).add_to(m)
# m
# m.save('crime_map1.html')


# marker_cluster = MarkerCluster().add_to(m)

# for lat, lon, cat in df[['Y','X','Category']].itertuples(index=False):
#     Marker(
#         location=(lat, lon),
#         popup=str(cat),
#         radius=2
#     ).add_to(marker_cluster)
# m.save('crime_map2.html')


# # Add crime locations as markers to the map
# for index, row in df.iterrows():
#     CircleMarker(
#         location=[row['Y'], row['X']],
#         radius=2,  # Adjust the radius as needed
#         color='red',
#         fill=True,
#         fill_color='red',
#         fill_opacity=0.6,
#         popup=row['Category'] # Add a popup with the crime category
#     ).add_to(m)

# # Display the map
# m.save('crime_map3.html')


import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight


# Load your data
data_path = (
    "/kaggle/input/sf-crime/train.csv.zip"
    if os.path.exists("/kaggle/input/sf-crime/train.csv.zip")
    else "/content/sf-crime-data/train.csv"
)

df = pd.read_csv(data_path)

# Feature Engineering (if needed)
# Convert 'Dates' to datetime objects
df['Dates'] = pd.to_datetime(df['Dates'])
# Extract year, month, day, hour from Dates
df['Year'] = df['Dates'].dt.year
df['Month'] = df['Dates'].dt.month
df['Day'] = df['Dates'].dt.day
df['Hour'] = df['Dates'].dt.hour
df.drop('Dates', axis=1, inplace=True)

# ... (Add any feature engineering steps you want to perform here) ...

# Select relevant features and target
X = df[['DayOfWeek', 'PdDistrict', 'X', 'Y', 'Year', 'Month', 'Day', 'Hour']]
y = df['Category']

# Define categorical and numerical columns
categorical_cols = ['DayOfWeek', 'PdDistrict']
numerical_cols = ['X', 'Y', 'Year', 'Month', 'Day', 'Hour']

# Preprocessing Function
def preprocess_data(df, categorical_cols, numerical_cols):
    """
    Preprocesses the data by label encoding categorical features and scaling numerical features.

    Args:
        df: DataFrame containing the data.
        categorical_cols: List of categorical columns to encode.
        numerical_cols: List of numerical columns to scale.
    """
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df, label_encoders, scaler

# Apply preprocessing
X_preprocessed, label_encoders, scaler = preprocess_data(X.copy(), categorical_cols, numerical_cols)

# Calculate class weights
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y),
    y=y
)
class_weights = dict(enumerate(class_weights))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_preprocessed, y, test_size=0.2, random_state=42, stratify=y
)

# Encode target variable using LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Convert target variables to categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Now you have X_train, X_test, y_train, y_test, and class_weights ready for model training


df_low = df.sample(n=15000, random_state=42)


df_low


df_low['Category'].unique()


from sklearn.preprocessing import StandardScaler, LabelEncoder
le = LabelEncoder()
df_low['Crime_Type'] = le.fit_transform(df_low['Category'])
df_low.head()


# Split the data into features (X) and target (y)
X = df_low.drop(['Crime_Type','Category','Descript', 'Resolution', 'Address'], axis=1)  # Drop unnecessary columns
y = df_low['Crime_Type']
sm = SMOTE(random_state=42, k_neighbors=min(2, len(np.unique(y)) - 1)) # Dynamically adjust k_neighbors


# Check for classes with too few samples and remove them
from collections import Counter

class_counts = Counter(y)
# Identify classes with less than (k_neighbors + 1) samples
minority_classes_to_remove = [
    class_label
    for class_label, count in class_counts.items()
    if count < sm.k_neighbors + 1
]

# Filter out the classes to remove before applying SMOTE
df_filtered = df_low[~df_low['Crime_Type'].isin(minority_classes_to_remove)]

# Reset the index
df_filtered = df_filtered.reset_index(drop=True)

# Separate features and target variable
X_filtered = df_filtered.drop(['Crime_Type', 'Category', 'Descript', 'Resolution', 'Address'], axis=1)
y_filtered = df_filtered['Crime_Type']

#Preprocess the data
categorical_cols = X_filtered.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X_filtered.select_dtypes(include=['number']).columns.tolist()

X_preprocessed, label_encoders, scaler = preprocess_data(X_filtered.copy(), categorical_cols, numerical_cols)

# Apply SMOTE
X_res, y_res = sm.fit_resample(X_preprocessed, y_filtered)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y_res)


import xgboost as xgb

try:
    # Ensure y_train has consecutive labels starting from 0
    unique_labels = np.unique(y_res)
    label_mapping = {label: i for i, label in enumerate(unique_labels)}
    y_res_mapped = np.array([label_mapping[label] for label in y_res])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res_mapped, test_size=0.2, random_state=42
    )

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42,
        num_class=len(np.unique(y_res_mapped)),
        objective='multi:softprob'
    )

    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)


# y_pred = model.predict(X_test)
# acc = accuracy_score(y_test, y_pred)
# print("Accuracy:", acc)
# print("\nClassification Report ğŸ“Š :\n")
# # Get the unique class labels present in y_res (or y_train/y_test after remapping)
unique_classes_after_smote = np.unique(y_res_mapped)  # Use y_res_mapped here
# # Filter le.classes_ to only include the classes present after SMOTE
target_names_filtered = [le.classes_[i] for i in unique_classes_after_smote]
# # Use the filtered target names in the classification_report
# print(classification_report(y_test, y_pred, target_names=target_names_filtered))
plt.figure(figsize=(15, 10))  # Increased figure size

cm = confusion_matrix(y_test, y_pred)

# Use filtered target names for the heatmap and adjust font size
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    xticklabels=target_names_filtered,
    yticklabels=target_names_filtered,
    cmap='Blues',
    annot_kws={"size": 10},  # Adjust annotation font size
)

plt.title("Confusion Matrix", fontsize=16)  # Increased title font size
plt.xlabel("Predicted", fontsize=12)  # Increased x-axis label font size
plt.ylabel("Actual", fontsize=12)  # Increased y-axis label font size

# Rotate x-axis labels for better visibility if needed
plt.xticks(rotation=50, ha="right", rotation_mode="anchor")

plt.tight_layout()  # Adjust layout to prevent overlapping elements
plt.show()



sample = X_test.iloc[0]
sample_df = pd.DataFrame([sample])
# Preprocess the sample
sample_df, _, _ = preprocess_data(sample_df, categorical_cols, numerical_cols)
prediction = model.predict(sample_df)
predicted_class_index = np.argmax(prediction)
predicted_class_name = le.classes_[predicted_class_index]

# Actual crime type and predicted:
actual_class_index = np.argmax(y_test[0]) # Gets the highest probability index
actual_class_name = le.classes_[actual_class_index]
print("Actual Crime Type:", actual_class_name)
print("Predicted Crime Type:", predicted_class_name)  # Print here after prediction


actual_class_index


import random
from collections import defaultdict

# Create a dictionary to store samples for each class
class_samples = defaultdict(list)

# Iterate through the test set and collect samples for each class
for index in range(len(X_test)):
    actual_class_index = np.argmax(y_test[index])  # Gets the highest probability index
    actual_class_name = le.classes_[actual_class_index]
    
    # Store the sample index for the corresponding class
    if len(class_samples[actual_class_name]) < 3:  # Limit to 3 samples per class
        class_samples[actual_class_name].append(index)

# Select samples from different classes
selected_indices = []
for class_name, indices in class_samples.items():
    selected_indices.extend(indices)  # Extend selected_indices with all collected indices

# Shuffle the selected indices
random.shuffle(selected_indices)

# Process and predict for the selected samples
for index in selected_indices:
    sample = X_test.iloc[index]
    sample_df = pd.DataFrame([sample])

    # Preprocess the sample
    sample_df, _, _ = preprocess_data(sample_df, categorical_cols, numerical_cols)

    # Make the prediction
    prediction = model.predict(sample_df)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = le.classes_[predicted_class_index]

    # Get the actual crime type
    actual_class_index = np.argmax(y_test[index])  # Gets the highest probability index
    actual_class_name = le.classes_[actual_class_index]

    # Print both actual and predicted
    print(f"Sample {index + 1}:")  # Add sample number
    print("Actual Crime Type:", actual_class_name)
    print("Predicted Crime Type:", predicted_class_name)
    print("-" * 20)  # Add a separator for better readability


# Split the data into features (X) and target (y)
X = df_preprocessed.drop(['PdDistrict', 'Descript', 'Resolution', 'Address'], axis=1)  # Drop unnecessary columns
y = df_preprocessed['PdDistrict']


le = LabelEncoder()
df['Crime_Type'] = le.fit_transform(df['Primary Type'])


categorical_cols = ['DayOfWeek', 'PdDistrict', 'Category']
numerical_cols = ['X', 'Y', 'Year', 'Month', 'Day', 'Hour']

# Preprocess the data
df_preprocessed, label_encoders, scaler = preprocess_data(df.copy(), categorical_cols, numerical_cols)

# Display the preprocessed data
df_preprocessed.head()



# Split the data into features (X) and target (y)
X = df_preprocessed.drop(['PdDistrict', 'Descript', 'Resolution', 'Address'], axis=1)  # Drop unnecessary columns
y = df_preprocessed['PdDistrict']

# Convert target variable to numerical using LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


# Class imbalance handling with SMOTE
smote = SMOTE(random_state=1) # Comment later
X_resampled, y_resampled = smote.fit_resample(X, y) # Comment later

# One-hot encode the target variable
y_resampled = to_categorical(y_resampled) # Comment later
#y = to_categorical(y) # UnComment later
# Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=36) # UnComment later


# Calculate the percentage of each category
category_percentages = df['PdDistrict'].value_counts(normalize=True) * 100

# Set a threshold for the minimum percentage (e.g., 0.5%)
threshold = 0.5

# Identify categories below the threshold
categories_to_drop = category_percentages[category_percentages < threshold].index

# Drop rows with categories below the threshold
df_filtered = df[~df['PdDistrict'].isin(categories_to_drop)]


df.head()


df.info()


df.describe()


# Number of rows and columns
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

# Data types of all columns
print("\nData types:\n", df.dtypes)

# Non-null counts for each column
print("\nNon-null counts:\n", df.count())

# Specific column's data type and non-null count
print("\n'Category' column data type:", df['Category'].dtype)
print("'Category' column non-null count:", df['Category'].count())


le = LabelEncoder()
df_preprocessed['Category_Encoded'] = le.fit_transform(df_preprocessed['Category'])

# Split the data into features (X) and target (y)
X = df_preprocessed.drop(['Category', 'Category_Encoded', 'Descript', 'Resolution', 'Address'], axis=1)
y = df_preprocessed['Category_Encoded'] # Use the encoded target
   
# Apply SMOTE for oversampling ( Class imbalance handling )
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# Convert target variables to categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# Number of rows and columns
print("Number of rows:", df_preprocessed.shape[0])
print("Number of columns:", df_preprocessed.shape[1])

# Data types of all columns
print("\nData types:\n", df_preprocessed.dtypes)

# Non-null counts for each column
print("\nNon-null counts:\n", df_preprocessed.count())

# Specific column's data type and non-null count
print("\n'Category' column data type:", df_preprocessed['Category'].dtype)
print("'Category' column non-null count:", df_preprocessed['Category'].count())



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# Convert target variables to categorical using to_categorical
from tensorflow.keras.utils import to_categorical 
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Print the shapes of the resulting datasets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# Convert data to numpy array
X_train_np = np.array(X_train)
X_test_np = np.array(X_test)
y_train_np = np.array(y_train)
y_test_np = np.array(y_test)

# Print the shapes of the numpy arrays
print("X_train_np shape:", X_train_np.shape)
print("X_test_np shape:", X_test_np.shape)
print("y_train_np shape:", y_train_np.shape)
print("y_test_np shape:", y_test_np.shape)


from tensorflow.keras.layers import Dense, Dropout , BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2

# Model Building and Training (Improved)
# Initialize the model
model = Sequential()

# Block 1 - First hidden layer
model.add(Dense(1024, activation='relu', input_dim=X_train.shape[1]))  # Increased number of units
model.add(BatchNormalization())
model.add(Dropout(0.5))  # Increased dropout to 50% for better regularization

# Block 2 - Second hidden layer
model.add(Dense(512, activation='relu'))  # Increased number of units
model.add(BatchNormalization())
model.add(Dropout(0.5))  # Dropout for regularization

# Block 3 - Third hidden layer
model.add(Dense(256, activation='relu'))  # Increased number of units
model.add(BatchNormalization())
model.add(Dropout(0.5))  # Dropout

# Block 4 - Fourth hidden layer
model.add(Dense(128, activation='relu'))  # Increased number of units
model.add(BatchNormalization())
model.add(Dropout(0.5))  # Dropout

# Block 5 - Fifth hidden layer (New addition)
model.add(Dense(64, activation='relu'))  # Increased number of units
model.add(BatchNormalization())
model.add(Dropout(0.5))  # Dropout

# Block 6 - Sixth hidden layer (New addition)
model.add(Dense(32, activation='relu'))  # Added another layer to increase model depth
model.add(BatchNormalization())
model.add(Dropout(0.5))  # Dropout

# Output layer - For multi-class classification (softmax)
model.add(Dense(y_train.shape[1], activation='softmax'))  # Output layer for multi-class classification

# # Define the optimizer with learning rate scheduling
# initial_learning_rate = 0.001
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate,
#     decay_steps=10000,
#     decay_rate=0.96,
#     staircase=True
# )
# optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
# Compile the model with the class weights
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])


from tensorflow.keras.callbacks import ReduceLROnPlateau , ModelCheckpoint, EarlyStopping

# Define and Configure callback functions to enhance model training efficiency and prevent overfitting.

# EarlyStopping:
# Stops training if the validation loss ('val_loss') does not improve for 5 consecutive epochs.
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1,
    restore_best_weights=True
)

# ModelCheckpoint:
# Saves the model only when an improvement in validation loss ('val_loss') is observed.
# This ensures that the best-performing model is retained for deployment.
model_checkpoint = ModelCheckpoint(
    'best_model.keras',
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    verbose=1
)

# ReduceLROnPlateau:
# Dynamically adjusts the learning rate when the validation loss ('val_loss') plateaus.
# If no improvement is observed for 3 consecutive epochs, the learning rate is reduced by a factor of 0.5.
# This technique helps fine-tune the learning process and avoid local minima.
lr_reduction = ReduceLROnPlateau(
    monitor='val_loss',
    patience=3,
    verbose=1,
    factor=0.5,
    min_lr=1e-4
)


from sklearn.utils import class_weight

# Convert one-hot encoded y_train to class indices if needed
if len(y_train.shape) > 1 and y_train.shape[1] > 1:
    y_train_labels = np.argmax(y_train, axis=1)
else:
    y_train_labels = y_train

# Compute class weights
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_labels),
    y=y_train_labels
)

# Convert to dictionary
class_weights = dict(enumerate(weights))





# Train the model with the callbacks
history = model.fit(X_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_data=(X_test, y_test),
                    #class_weight=class_weights,
                    callbacks=[early_stopping, model_checkpoint, lr_reduction],
                    verbose='auto',
)


from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Model Evaluation
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Make predictions
y_pred = model.predict(X_test_np)
y_pred_classes = np.argmax(y_pred, axis=1)  # Get the class with the highest probability
y_true_classes = np.argmax(y_test_np, axis=1)

# Evaluate the model
# Compute the classification report and confusion matrix
print(classification_report(y_true_classes, y_pred_classes))
print(confusion_matrix(y_true_classes, y_pred_classes))

# Plot the confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix")
plt.show()



# Load the best model
best_model = keras.models.load_model(checkpoint_filepath)

# Evaluate the best model
loss, accuracy = best_model.evaluate(X_test, y_test, verbose=0)
print(f"Best Model Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

# Save the model
best_model.save("final_model.h5")



cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

