import numpy as np
import pandas as pd


path_data = "/kaggle/input/ashrae-dataset/"

train_data = pd.read_csv(path_data + 'train.csv')
test_data = pd.read_csv(path_data + 'test.csv')
building_metadata = pd.read_csv(path_data + 'building_metadata.csv')
weather_train = pd.read_csv(path_data + 'weather_train.csv')
weather_test = pd.read_csv(path_data + 'weather_test.csv')
sample_submission = pd.read_csv(path_data + 'sample_submission.csv')


display(train_data.head())
display(test_data.head())
display(sample_submission)


display(building_metadata.head())


display(weather_train.head())


display(weather_test.head())


display(sample_submission.head())


def prepare_data(X, building_data, weather_data, test=False):
    """
    Preparing final dataset with all features.
    """
    
    X = X.merge(building_data, on="building_id", how="left")
    X = X.merge(weather_data, on=["site_id", "timestamp"], how="left")
    
    X.timestamp = pd.to_datetime(X.timestamp, format="%Y-%m-%d %H:%M:%S")
    X.square_feet = np.log1p(X.square_feet)
    
    if not test:
        X.sort_values("timestamp", inplace=True)
        X.reset_index(drop=True, inplace=True)
        
    holidays = ["2016-01-01", "2016-01-18", "2016-02-15", "2016-05-30", "2016-07-04",
                "2016-09-05", "2016-10-10", "2016-11-11", "2016-11-24", "2016-12-26",
                "2017-01-01", "2017-01-16", "2017-02-20", "2017-05-29", "2017-07-04",
                "2017-09-04", "2017-10-09", "2017-11-10", "2017-11-23", "2017-12-25",
                "2018-01-01", "2018-01-15", "2018-02-19", "2018-05-28", "2018-07-04",
                "2018-09-03", "2018-10-08", "2018-11-12", "2018-11-22", "2018-12-25",
                "2019-01-01"]
    
    X["hour"] = X.timestamp.dt.hour
    X["weekday"] = X.timestamp.dt.weekday
    X["is_holiday"] = (X.timestamp.dt.date.astype("str").isin(holidays)).astype(int)
    
    drop_features = ["timestamp", "sea_level_pressure", "wind_direction", "wind_speed"]

    X.drop(drop_features, axis=1, inplace=True)

    if test:
        row_ids = X.row_id
        X.drop("row_id", axis=1, inplace=True)
        return X, row_ids
    else:
        y = np.log1p(X.meter_reading)
        X.drop("meter_reading", axis=1, inplace=True)
        return X, y


train_full, y_train_full = prepare_data(train_data, building_metadata, weather_train)

display(y_train_full)


test_full, row_ids = prepare_data(test_data, building_metadata, weather_test, test=True)

display(test_full)





import seaborn as sns
import matplotlib.pyplot as plt

# Encode the categorical column 'primary_use'
train_full['primary_use'] = train_full['primary_use'].astype('category').cat.codes

# Calculate the correlation matrix
corr_matrix = train_full.corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.1f')
plt.title('Correlation Heatmap')
plt.show()



train_data = train_data.dropna()


mean_meter_reading = y_train_full.mean()
print(mean_meter_reading)


num_rows = len(test_full)  # Get the number of rows in test_full
mean_meter_reading = y_train_full.mean()  # Compute the mean meter reading

# Create the submission DataFrame
submission_df = pd.DataFrame({
    "row_id": range(num_rows),  # Sequential row IDs
    "meter_reading": mean_meter_reading  # Fill with mean meter reading
})

# Save to CSV file
submission_df.to_csv("submission.csv", index=False)

print("CSV file 'submission.csv' created successfully!")

