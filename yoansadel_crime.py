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


# Requirments
!pip install numpy pandas seaborn tensorflow keras plotly matplotlib scikit-learn folium
from IPython.display import clear_output
clear_output()
print("All installed")


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
from folium import Map, Marker, CircleMarker
from sklearn.preprocessing import MinMaxScaler
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



# center on SF
m = Map(location=[37.7749, -122.4194], zoom_start=12)

# build a list of [lat, lon] pairs
heat_data = df[['Y','X']].values.tolist()

# add the HeatMap layer
HeatMap(
    heat_data,
    radius=8,    # cluster radius of each â€œpointâ€�
    blur=15,     # smoothness
    max_zoom=12
).add_to(m)
m


# FastMarkerCluster takes a list of [lat, lon] or [lat, lon, popup_html]
# Hereâ€™s how to include popups (e.g. crime Category):
cluster_data = df.apply(lambda r: [r['Y'], r['X'], r['Category']], axis=1).tolist()

FastMarkerCluster(data=cluster_data).add_to(m)
m


# marker_cluster = MarkerCluster().add_to(m)

# for lat, lon, cat in df[['Y','X','Category']].itertuples(index=False):
#     Marker(
#         location=(lat, lon),
#         popup=str(cat),
#         radius=2
#     ).add_to(marker_cluster)

# m


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
# m


import geopandas as gpd
from shapely.geometry import Point
from geopandas import GeoDataFrame
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


train_data = pd.read_csv(data_path)


train_data.duplicated().sum()
train_data.drop_duplicates(keep="first", inplace=True)
geometry = [Point(xy) for xy in zip(train_data['X'], train_data['Y'])]
gdf = GeoDataFrame(train_data, geometry=geometry)
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))


gdf.plot(ax=world.plot(), marker='o', color='red')
plt.show()


train_data[(train_data.X > -122) | (train_data.X < -123) | (train_data.Y > 38) | (train_data.Y < 37)].count()[0]
train_data = train_data.drop(train_data[(train_data.X > -122) | (train_data.X < -123) | (train_data.Y > 38) | (train_data.Y < 37)].index)
dates = pd.to_datetime(train_data["Dates"])
train_data["Year"] = dates.dt.year
train_data["Month"] = dates.dt.month
train_data["Day"] = dates.dt.day
train_data["Hour"] = dates.dt.hour


# Ensure Dates column is in datetime format
holidays = calendar().holidays(start=train_data["Dates"].min(), end=train_data["Dates"].max())
train_data["Holiday"] = train_data["Dates"].isin(holidays)



def weekend(weekday):
    return weekday == "Saturday" or weekday == "Sunday"

train_data["Weekend"] = train_data["DayOfWeek"].map(weekend)


le = LabelEncoder()
train_data["Category_Label"] = le.fit_transform(train_data["Category"])
train_data["DayOfWeek_Label"] = le.fit_transform(train_data["DayOfWeek"])
train_data["PdDistrict_Label"] = le.fit_transform(train_data["PdDistrict"])

train_data = train_data.drop(columns=["Category", "DayOfWeek", "PdDistrict"])


train_data = train_data.drop(columns=["Dates", "Address", "Resolution", "Descript"])
train_data.dropna(inplace=True)


X_train, X_test, y_train, y_test = train_test_split(
    train_data.drop("Category_Label", axis=1),
    train_data["Category_Label"],
    stratify=train_data["Category_Label"],
    test_size=0.3,
    random_state=1
)


random_forest_model = RandomForestClassifier(
    n_estimators=60,
    max_depth=32,
    random_state=1
)


knn_model = KNeighborsClassifier(n_neighbors=9)
logistic_regression_model = LogisticRegression()


def model_result(model, model_name="Model"):
    train_acc = model.score(X_train, y_train)
    train_loss = log_loss(y_train, model.predict_proba(X_train))
    test_acc = model.score(X_test, y_test)
    test_loss = log_loss(y_test, model.predict_proba(X_test))
    
    print(f"\nğŸ“Š {model_name} Performance:")
    print(f"  - Train Accuracy : {train_acc:.4f}")
    print(f"  - Train Log Loss : {train_loss:.4f}")
    print(f"  - Test Accuracy  : {test_acc:.4f}")
    print(f"  - Test Log Loss  : {test_loss:.4f}")


random_forest_model.fit(X_train, y_train)
model_result(random_forest_model, "Random Forest")

knn_model.fit(X_train, y_train)
model_result(knn_model, "K-Nearest Neighbors")

logistic_regression_model.fit(X_train, y_train)
model_result(logistic_regression_model, "Logistic Regression")


