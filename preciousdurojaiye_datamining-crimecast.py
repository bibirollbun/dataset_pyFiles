# ! pip install -r requirements.txt


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
from datetime import datetime as dt
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

# import torch
# import torchvision

# Data Processing
from scipy import sparse
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix



df = pd.read_csv("/kaggle/input/crime-cast-forecasting-crime-categories/train.csv")
test = pd.read_csv("/kaggle/input/crime-cast-forecasting-crime-categories/test.csv")

df.info()
df.head()


print(df.isnull().sum())


# df.drop(columns=['Cross_Street', 'Weapon_Used_Code', 'Weapon_Description', 'Status'], inplace=True)
df.dropna(subset=['Victim_Sex', 'Victim_Descent', 'Premise_Description'], inplace=True)


# Remove X and H Victim_ Sex
indexAge = df[(df['Victim_Sex'] == 'X') | (df['Victim_Sex'] == 'H')].index
df.drop(indexAge, inplace=True)



df.head()


print(df.isnull().sum())


df['Time_Occurred'].isnull().values.any()


df.info()
df.head()


df.drop_duplicates(inplace=True)


# df.to_csv("data/cleaned/cleaned_train.csv", index=False)
# test.to_csv("data/cleaned/cleaned_test.csv", index=False)
# df_clean = pd.read_csv("data/cleaned/cleaned_train.csv")


def cleanTrainData(df):
    df.drop(columns=['Cross_Street', 'Area_Name' 'Premise_Description', 'Weapon_Description', 'Status_Description'])


df_clean = df.drop(columns=['Cross_Street', 'Area_Name', 'Premise_Description', 'Weapon_Description', 'Status_Description'])

# Convert Time_Occurred column to string and pad with zeros if necessary
df_clean['Time_Occurred'] = df_clean['Time_Occurred'].astype(str).str.zfill(4)  # Ensures 4-digit time format
df_clean['Date_Reported'] = pd.to_datetime(df_clean['Date_Reported'], errors='coerce')
df_clean['Date_Occurred'] = pd.to_datetime(df_clean['Date_Occurred'], errors='coerce')
# Convert to HH:MM format
# df_clean['Time_Occurred'] = pd.to_datetime(df['Time_Occurred'], format='%H%M').dt.time

print(df_clean)

df_clean.info()
df_clean.head(5)


plt.figure(figsize=(12, 6))
sns.countplot(y=df_clean['Crime_Category'], order=df_clean['Crime_Category'].value_counts().index, palette="viridis")
plt.title("Crime Category Frequency")
plt.xlabel("Count")
plt.ylabel("Crime Category")
plt.show()


crime_counts = df_clean['Crime_Category'].value_counts()
plt.figure(figsize=(8, 8))
crime_counts[:10].plot(kind='pie', autopct='%1.1f%%', cmap='tab10', legend=True)
plt.title("Top 10 Crime Categories Distribution")
plt.ylabel("")
plt.show()


plt.figure(figsize=(8, 5))
sns.histplot(df_clean['Victim_Age'], bins=30, kde=False, color='teal')
plt.title("Distribution of Victim Age")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()


print(df_clean['Victim_Age'].value_counts()[0])


df_clean = df_clean[df_clean['Victim_Age'] > 0]


plt.figure(figsize=(8, 5))
sns.histplot(df_clean['Victim_Age'], bins=30, kde=False, color='teal')
plt.title("Distribution of Victim Age")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()


age_counts = df_clean['Victim_Age'].value_counts().sort_index()
print(age_counts)


df_clean.info()


# Analyse crime frequency by year and month


df_clean['YearMonth'] = df_clean['Date_Occurred'].dt.to_period('M')  # Creates "YYYY-MM" format
crime_trend = df_clean.groupby('YearMonth').size().reset_index(name='Crime_Count') # count the number of crimes that occurred per month

crime_trend.head()

plt.figure(figsize=(12, 6))
plt.plot(crime_trend['YearMonth'].astype(str), crime_trend['Crime_Count'], marker='o', linestyle='-', color='b')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.xlabel("Month")
plt.ylabel("Number of Crimes")
plt.title("Crime Trend Over Time (Monthly)")
plt.grid(True)
plt.show()


# Time serires plot to see monthly crime trends

# Group by 'YearMonth' and 'Crime_Category' to count the number of crimes per category per month
crime_trend_category = df_clean.groupby(['YearMonth', 'Crime_Category']).size().reset_index(name='Crime_Count')

# Convert 'YearMonth' to string format to avoid issues when plotting
crime_trend_category['YearMonth'] = crime_trend_category['YearMonth'].astype(str)

# Create a line plot with multiple lines (one for each crime category)
plt.figure(figsize=(15, 13))
sns.lineplot(data=crime_trend_category, x='YearMonth', y='Crime_Count', hue='Crime_Category', marker='o')

# Formatting the plot
plt.xticks(rotation=45)
plt.xlabel("Month")
plt.ylabel("Number of Crimes")
plt.title("Monthly Crime Trends by Category")
plt.legend(title="Crime Category")  # Ensures the legend shows crime categories
plt.show()


# # Group by 'Crime_Category' and 'Area_Name' to count crimes per area
# crime_location = df_clean.groupby(['Crime_Category', 'Area_Name']).size().reset_index(name='Crime_Count')

# # Sort to show the most affected locations for each crime type
# crime_location = crime_location.sort_values(by='Crime_Count', ascending=False)

# # Display the first few rows
# print(crime_location.head(10))


# # Identify areas with high crime rates

# crime_hotspots = df_clean.groupby('Area_Name').size().reset_index(name='Num_Crime')
# crime_hotspots = crime_hotspots.sort_values(by='Num_Crime', ascending=False).head(10)

# # Plot crime count per area
# plt.figure(figsize=(12, 6))
# sns.barplot(data=crime_hotspots, x='Num_Crime', y='Area_Name', palette="Reds_r")
# plt.xlabel("Number of Crimes")
# plt.ylabel("Area Name")
# plt.title("Top 10 Crime Hotspots")
# plt.show()



# # Relationship between 1-3 crime hotspots
# location1 = "77th Street"  # Change this to any area in your dataset

# # Filter dataset for this location
# location_data = crime_location[crime_location['Area_Name'] == location1]

# # Plot crimes in this area
# plt.figure(figsize=(12, 6))
# sns.barplot(data=location_data, x='Crime_Count', y='Crime_Category', palette='coolwarm')

# plt.xlabel("Number of Crimes")
# plt.ylabel("Crime Type")
# plt.title(f"Crime Types in {location1}")
# plt.show()


# Choose a specific location (e.g., "Downtown")
# location2 = "Southwest"  # Change this to any area in your dataset

# # Filter dataset for this location
# location_data = crime_location[crime_location['Area_Name'] == location2]

# # Plot crimes in this area
# plt.figure(figsize=(12, 6))
# sns.barplot(data=location_data, x='Crime_Count', y='Crime_Category', palette='coolwarm')

# plt.xlabel("Number of Crimes")
# plt.ylabel("Crime Type")
# plt.title(f"Crime Types in {location2}")
# plt.show()


# location3 = "Central"  # Change this to any area in your dataset

# # Filter dataset for this location
# location_data = crime_location[crime_location['Area_Name'] == location3]

# # Plot crimes in this area
# plt.figure(figsize=(12, 6))
# sns.barplot(data=location_data, x='Crime_Count', y='Crime_Category', palette='coolwarm')

# plt.xlabel("Number of Crimes")
# plt.ylabel("Crime Type")
# plt.title(f"Crime Types in {location3}")
# plt.show()


# Visualise crime hotspots using heatmaps

# Ensure latitude and longitude columns are numeric
df_clean['Latitude'] = pd.to_numeric(df_clean['Latitude'], errors='coerce')
df_clean['Longitude'] = pd.to_numeric(df_clean['Longitude'], errors='coerce')

# Drop rows with missing coordinates
df_clean = df_clean.dropna(subset=['Latitude', 'Longitude'])

# Create a base map centered around the average location
crime_map = folium.Map(location=[df_clean['Latitude'].mean(), df_clean['Longitude'].mean()], zoom_start=11)

# Add crime heatmap
heat_data = df_clean[['Latitude', 'Longitude']].values.tolist()
HeatMap(heat_data).add_to(crime_map)

# Show map
crime_map


# Categorize victim age and sex into groups (e.g., young men/women, elderly).

# Convert victim age to numeric and handle missing values
df_clean['Victim_Age'] = pd.to_numeric(df_clean['Victim_Age'], errors='coerce')
df_clean = df_clean.dropna(subset=['Victim_Age', 'Victim_Sex'])  # Remove rows with missing age/sex

# Define age categories
def categorize_age_sex(row):
    age, sex = row['Victim_Age'], row['Victim_Sex']
    
    if age <= 12:
        return "Child"
    elif 13 <= age <= 17:
        return "Teenager"
    elif 18 <= age <= 30:
        return "Young Man" if sex == "M" else "Young Woman"
    elif 31 <= age <= 50:
        return "Middle-Aged Man" if sex == "M" else "Middle-Aged Woman"
    else:
        return "Elderly Man" if sex == "M" else "Elderly Woman"

# Apply categorization
df_clean['Age_Group'] = df.apply(categorize_age_sex, axis=1)

df_clean['Age_Group'].head()


age_group_counts = df_clean['Age_Group'].value_counts().reset_index()
age_group_counts.columns = ['Age_Group', 'Crime_Count']



# Plot the distribution
plt.figure(figsize=(10, 5))
sns.barplot(data=age_group_counts, x='Age_Group', y='Crime_Count', palette='viridis')
plt.xticks(rotation=45)
plt.xlabel("Age Group")
plt.ylabel("Number of Crimes")
plt.title("Crime Trends Across Different Age Groups")
plt.show()



df_clean['Victim_Sex'].head()


sex_group_counts = df_clean['Victim_Sex'].value_counts().reset_index()
sex_group_counts.columns = ['Victim_Sex', 'Crime_Count']

# gender_group = df_clean.groupby(['Victim', 'Crime_Category']).size().reset_index(name='Crime_Count')

# Plot the distribution
plt.figure(figsize=(10, 5))
sns.barplot(data=sex_group_counts, x='Victim_Sex', y='Crime_Count', palette='viridis')
plt.xticks(rotation=45)
plt.xlabel("Sex")
plt.ylabel("Number of Crimes")
plt.title("Crime Trends Across Different Genders")
plt.show()



# # Convert Date to datetime and extract Year-Month
# df_clean['YearMonth'] = pd.to_datetime(df_clean['Date_Occurred']).dt.to_period('M')

# Count crimes per age group over time
age_trend = df_clean.groupby(['YearMonth', 'Age_Group']).size().reset_index(name='Crime_Count')

age_trend['YearMonth'] = age_trend['YearMonth'].astype(str)
# Plot the trends
plt.figure(figsize=(20, 12))
sns.lineplot(data=age_trend, x='YearMonth', y='Crime_Count', hue='Age_Group', marker='o')

plt.xticks(rotation=45)
plt.xlabel("Month")
plt.ylabel("Number of Crimes")
plt.title("Crime Trends Over Time by Age Group")
plt.legend(title="Age Group")
plt.show()



# # df_clean['Hour'] = df_clean['Time_Occurred'] // 100  # Extract hour from HHMM format
# # df_clean['Hour'] = df_clean['Hour'].astype(str)
# df_clean['DayOfWeek'] = df_clean['Date_Occurred'].dt.dayofweek  # Monday = 0, Sunday = 6
# df_clean['DayOfWeek'] = df_clean['DayOfWeek'].astype(str)
# # df_clean['Month'] = df_clean['Date_Occurred'].dt.month  # Extract month
# # day_of_week_trend

# plt.figure(figsize=(12, 6))
# sns.countplot(data=df_clean, x="DayOfWeek", hue="Crime_Category", palette="Set2")

# plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
# plt.xlabel("Day of the Week")
# plt.ylabel("Number of Crimes")
# plt.title("Crime Trends by Day of the Week")
# plt.legend(title="Crime Category", bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.show()


train = df_clean
test = pd.read_csv("/kaggle/input/crime-cast-forecasting-crime-categories/test.csv")
train.head(3)


train.info()


def convert_data_types(dataset: pd.DataFrame) -> pd.DataFrame:
    """Convert features to correct dtype and encode strings and categories"""
    df = dataset.copy()

    ### Category ###
    df['Area_ID'] = df['Area_ID'].astype('int64')
    df['Reporting_District_no'] = df['Reporting_District_no'].astype('int64')
    df['Part 1-2'] = df['Part 1-2'].astype("category").cat.codes
    df['Victim_Sex'] = df['Victim_Sex'].astype('category').cat.codes
    df['Victim_Descent'] = df['Victim_Descent'].astype('category').cat.codes
    df['Premise_Code'] = df['Premise_Code'].astype('int64')

    ### Numerical ###
    df['Latitude'] = df['Latitude'].astype('float64')
    df['Longitude'] = df['Longitude'].astype('float64')
    df['Date_Reported'] = df['Date_Reported'] 
    df['Date_Occurred'] = df['Date_Occurred']
    df['Time_Occurred'] = df['Time_Occurred']
    df['Victim_Age'] = df['Victim_Age'].astype('int64')
    
    ### Text ###
    df['Location'] = df['Location'].astype("string")
    df['Modus_Operandi'] = df['Modus_Operandi'] # later

    ## Dependent variable ###
    if 'Crime_Category' in df.columns:
        df['Crime_Category'] = df['Crime_Category'].astype('category').cat.codes

    return df


train = convert_data_types(train)
test = convert_data_types(test)


def fixFeatures(df):
    df['Victim_Age'] = df['Victim_Age'].apply(lambda x: -1 if x <= 0 else x)
    df['Victim_Sex'] = df['Victim_Sex'].apply(lambda x: 'Null' if x == 'H' else x)
    df['Victim_Sex'] = df['Victim_Sex'].fillna('Null')
    df['Victim_Descent'] = df['Victim_Descent'].fillna('Null')
    df['Weapon_Used_Code'] = df['Weapon_Used_Code'].fillna('N/A')
    return df




def dateFeatures(df):
    #Coverts into strings of HHMM, adding 0s to times like 210 -> 0210
    df['Time_Occurred'] = df['Time_Occurred'].apply(lambda x: str(int(x)).zfill(4))
    df['Time_Occurred'] = df['Time_Occurred'].str[:2] + ':' + df['Time_Occurred'].str[2:]
    #Date and time occurred as datetime object
    df['DateTime_Occurred'] = pd.to_datetime(df['Date_Occurred'].astype("string") + " " + df['Time_Occurred'])
    df['DateTime_Reported'] = pd.to_datetime(df['Date_Reported'])
    # No. of days between occurance and reporting as int
    #df['Occurred_Reported_Diff'] = (df['Date_Reported'] - df['Date_Occurred']).dt.days
    
    # Day of the week as int (Monday=0, Tuesday=1 ...)
    df['Weekday_Occurred'] = df['DateTime_Occurred'].dt.weekday
    df['Year_Occurred'] = df['DateTime_Occurred'].dt.year # Not needed, everything is 2020
    df['Month_Occurred'] = df['DateTime_Occurred'].dt.month
    df['Day_Occurred'] = df['DateTime_Occurred'].dt.day

    df['Weekday_Reported'] = df['DateTime_Reported'].dt.weekday
    df['Year_Reported'] = df['DateTime_Reported'].dt.year # Not needed, everything is 2020
    df['Month_Reported'] = df['DateTime_Reported'].dt.month
    df['Day_Reported'] = df['DateTime_Reported'].dt.day

    # Putting the time into buckets
    df['O-Night'] = df['DateTime_Occurred'].apply(lambda x: 0 < x.hour & x.hour <= 6 )
    df['O-Morning'] = df['DateTime_Occurred'].apply(lambda x: 6 < x.hour & x.hour <= 12 )
    df['O-Afternoon'] = df['DateTime_Occurred'].apply(lambda x: 12 < x.hour & x.hour <= 18 )
    df['O-Evening'] = df['DateTime_Occurred'].apply(lambda x: 18 < x.hour & x.hour <= 23 )
    df['O-Hour'] = df['DateTime_Occurred'].apply(lambda x: x.hour)
    
    return df


raw = df
dateFeatures(convert_data_types(raw)).info()


def interpolateCoords(df):
    avg_longitudes = df.groupby('Area_ID')['Longitude'].mean()
    avg_latitudes = df.groupby('Area_ID')['Latitude'].mean()
    df['Longitude'] = df.apply(lambda x: x['Longitude'] if x['Longitude'] != 0 else avg_longitudes[x['Area_ID']], axis=1)
    df['Latitude'] = df.apply(lambda x: x['Latitude'] if x['Latitude'] != 0 else avg_latitudes[x['Area_ID']], axis=1)
    return df
interpolateCoords(train)
interpolateCoords(test)


df = train
# Process the "Modus_Operandi" column
mo_code_series = df["Modus_Operandi"].dropna().astype(str).str.split()
mo_code_list = [code for sublist in mo_code_series for code in sublist]  # Flatten list
mo_code_counts = Counter(mo_code_list)
num_unique_mo_codes = len(mo_code_counts)
mo_counts_per_crime = mo_code_series.apply(len)
print("Total unique MO codes:", num_unique_mo_codes)
print("MO codes per crime statistics:\n", mo_counts_per_crime.describe())
print("Top 10 most common MO codes:", mo_code_counts.most_common(10))


def process_modus_operandi(df: pd.DataFrame) -> pd.DataFrame:
    """Creates multilabel modus operandi labels and adds their PCA to the main dataframe."""
    df["Modus_Operandi"] = df["Modus_Operandi"].apply(lambda x: str(x).split())
    mlb = MultiLabelBinarizer()
    MO_encoded = pd.DataFrame((mlb.fit_transform(df["Modus_Operandi"])), columns=mlb.classes_)

    n_components = 80  # Number of components for PCA
    pca = PCA(n_components=n_components)

    mo_pca = pca.fit_transform(MO_encoded)
    mo_pca_df = pd.DataFrame(mo_pca, columns=[f"MO_PCA_{i+1}" for i in range(n_components)])

    # Merge PCA features into original dataset
    df = pd.concat([df, mo_pca_df], axis=1)
    
    return df, MO_encoded

train = process_modus_operandi(train)
test = process_modus_operandi(test)
train[0].head(5)


# Fit PCA on the Many-Hot Encoded MO codes
pca = PCA()
pca.fit(train[1])  # Replace with your Many-Hot Encoded DataFrame

# Calculate cumulative explained variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Plot the variance
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker="x")
plt.xlabel("Components")
plt.ylabel("Explained Variance")
plt.title("Explained Variance vs. Number of Components")
plt.minorticks_on()
plt.grid(which='both')
plt.show()


def engineerFeatures(df):
    df = fixFeatures(df)
    df = convert_data_types(df)
    df = dateFeatures(df)
    df = interpolateCoords(df)
    df, MO_encoded = process_modus_operandi(df)
    return df

rawTrain = pd.read_csv("/kaggle/input/crime-cast-forecasting-crime-categories/train.csv")
rawTest = pd.read_csv("/kaggle/input/crime-cast-forecasting-crime-categories/test.csv")

train = engineerFeatures(rawTrain)
test = engineerFeatures(rawTest)

#test['Victim_Descent'].head(50)
#test['Victim_Descent'].unique()
#print(rawTrain.columns.tolist())

#for col in test.columns:
#    print(col)


# Categrory Mapping for final output 
original_categories = rawTrain['Crime_Category'].astype('category')
category_mapping = dict(enumerate(original_categories.cat.categories))
print(category_mapping)



X_train_unencoded = train.drop(columns=['Crime_Category'])
y_train_unencoded = (train['Crime_Category']).to_frame()
X_test_unencoded = test

print(X_train_unencoded)


# Define column categories
categorical = ['Area_ID','Reporting_District_no','Part 1-2','Victim_Sex','Victim_Descent','Premise_Code','Weekday_Occurred',
               'Year_Occurred','Month_Occurred','Day_Occurred','Weekday_Reported','Year_Reported','Month_Reported','Day_Reported',
               'O-Night','O-Morning','O-Afternoon','O-Evening','O-Hour']
targetCategorical = ['Crime_Category']
numerical_stanscal = ['Latitude','Longitude']
numerical_minmax = ['Victim_Age']
textual = []

# Pipelines
cat_pipeline = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))])
num_pipeline1 = Pipeline([('stanscal', StandardScaler())])
num_pipeline2 = Pipeline([('minmaxscal', MinMaxScaler())])
text_pipeline = Pipeline([('tfidf', TfidfVectorizer())])

cat_pipeline2 = Pipeline([('manyhot', MultiLabelBinarizer())])

# Column transformers
transformer1 = ColumnTransformer([('cat', cat_pipeline, categorical)])
transformer2 = ColumnTransformer([('num1', num_pipeline1, numerical_stanscal), ('num2', num_pipeline2, numerical_minmax)])
transformer3 = ColumnTransformer([('text', text_pipeline, textual)])

transformer4 = ColumnTransformer([('catgoal', cat_pipeline, targetCategorical)])


#
union = FeatureUnion([('tran1', transformer1), ('tran2', transformer2), ('tran3', transformer3)])



# Transformations
X_train = union.fit_transform(X_train_unencoded)
X_test = union.transform(X_test_unencoded)
y_train = transformer4.fit_transform(y_train_unencoded)


print(X_train.shape)
print(X_test.shape)


# Save the transformed data
sparse.save_npz("/kaggle/working/X_train.npz", X_train)
sparse.save_npz("/kaggle/working/X_test.npz", X_test)
sparse.save_npz("/kaggle/working/y_train.npz", y_train)

#np.savez("data/processed/y_train.npz", y=y_train.to_numpy())
#np.savez("data/processed/y_test.npz", y=y_test.to_numpy()) # Cannot do this as the test data doesn't have labels


X_train = sparse.load_npz("/kaggle/working/X_train.npz")
X_test = sparse.load_npz("/kaggle/working/X_test.npz")
y_train = np.load("/kaggle/working/y_train.npz")



# y_test = np.load("data/processed/y_test.npz")["y"] -> there is no y_test


# Load the NPZ file properly
y_train = np.load("/kaggle/working/y_train.npz")

# Access keys from the file object (not the array)
print(f"Keys in NPZ file: {list(y_train.keys())}")

# Extract the array using the appropriate key
y_train = y_train['y'] if 'y' in y_train.keys() else y_train[list(y_train.keys())[0]]

# Now y_train is a NumPy array
print(f"Shape of y_train: {y_train.shape}")


print("X_train shape: ", X_train.shape)
print("X_test shape: ", X_test.shape)
# print("y_train shape: ", y_train.shape)


# Create and train the model
rf_model = RandomForestClassifier(
    n_estimators=100,         # Number of trees
    max_depth=None,           # Maximum depth of trees (None means unlimited)
    min_samples_split=2,      # Minimum samples required to split
    min_samples_leaf=1,       # Minimum samples at leaf node
    random_state=42,          # For reproducibility
    n_jobs=-1                 # Use all CPU cores
)

# Train the model
rf_model.fit(X_train, y_train)  

# Make predictions on test data
# If you have labels for test data:
# y_test_pred = rf_model.predict(X_test)
# print(classification_report(y_test, y_test_pred))

# For submission file if this is a competition:
predictions = rf_model.predict(X_test)
original_predictions = [category_mapping[int(pred)] for pred in predictions]

submission_df = pd.DataFrame({
    'ID': range(len(original_predictions)),
    'Crime_Category': original_predictions
})

# Save to CSV

submission_df.to_csv('submission.csv',index=False)
print("Predictions saved to submission.csv")

# print(submission_df.head(5))

