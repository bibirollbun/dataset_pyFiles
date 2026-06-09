import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sys


def score(solution: pd.DataFrame, submission: pd.DataFrame) -> float:
    """
    Computes the Kaggle competition metric for predicting forest fire sizes.

    Parameters:
        solution (pd.DataFrame): A DataFrame with columns ["ID", "STATE", "month", "total_fire_size"].
        submission (pd.DataFrame): A DataFrame with columns ["ID", "STATE", "month", "total_fire_size"],
                                  where "ID" is formatted as "STATE_month".

    Returns:
        float: The mean of min(abs(log(pred / true)), 10) over all valid entries.
    """
    # Merge submission with ground truth on (STATE, month)
    merged = solution.merge(submission, on=["STATE", "month"], how="left", suffixes=("_true", "_pred"))

    # Identify missing predictions and assign a score of 10 for them
    missing_pred_mask = merged["total_fire_size_pred"].isna()
    zero_pred_mask = merged["total_fire_size_pred"] <= 0

    # Compute log error where prediction is valid
    valid_pred_mask = ~missing_pred_mask & ~zero_pred_mask
    log_errors = np.full(len(merged), 10.0)  # Default to max penalty

    # Compute actual log error only for valid predictions
    log_errors[valid_pred_mask] = np.abs(np.log(merged.loc[valid_pred_mask, "total_fire_size_pred"] /
                                                 merged.loc[valid_pred_mask, "total_fire_size_true"]))

    # Apply the min operation
    final_scores = np.minimum(log_errors, 10)

    # Return the mean score (if no valid entries, return 10)
    return np.mean(final_scores) if len(final_scores) > 0 else 10.0




# Loading data

merged_state_data = pd.read_csv("data/merged_state_data.csv")
weather_data = pd.read_csv("data/weather_monthly_state_aggregates.csv")
wildfire_sizes = pd.read_csv("data/wildfire_sizes_before_2010.csv")

trainData = pd.read_csv("train.csv")
df_imputed = pd.read_csv("imputedTest.csv")


plt.hist(wildfire_sizes["total_fire_size"])
plt.show()


sys.pairplot(merged_state_data)



# First need to make the data readable

all_states = merged_state_data['State']

stateIdToNum = {v: k for k, v in all_states.items()}

trainData = pd.read_csv("train.csv")



combined = pd.read_csv("data/combined.csv")

combined.set_index('month', inplace=True)
months = range(1, 13)
years = range(2011, 2016)
allc = pd.MultiIndex.from_product([years, months], names=['year', 'month']).to_frame(index=False)
combined = pd.merge(allc, combined, on=['year', 'month'], how='left')

wildfire_sizes = pd.read_csv("data/wildfire_sizes_before_2010.csv")
wildfire_sizes['month'] = pd.to_datetime(wildfire_sizes['month'], format='%Y-%m')
wildfire_sizes['year'] = wildfire_sizes['month'].dt.year
wildfire_sizes['month'] = wildfire_sizes['month'].dt.month

states = pd.DataFrame({'State': range(50)})

combined = pd.concat([combined.assign(State=state) for state in states['State']], ignore_index=True)

weather_data = pd.read_csv("data/weather_monthly_state_aggregates.csv")
weather_data['month'] = pd.to_datetime(weather_data['year_month'], format='%Y-%m')
weather_data['year'] = weather_data['month'].dt.year
weather_data['month'] = weather_data['month'].dt.month

state_mapping = {abbr: num for num, abbr in enumerate(weather_data['State'].unique())}

weather_data['State'] = weather_data['State'].map(state_mapping)

combined = pd.merge(combined, weather_data, on=['State', 'year', 'month'], how='left')

combined.drop(columns=['prcp', 'evap', 'tmin', 'tmax', 'year_month'], inplace=True)
combined.drop(columns=['elev', 'lArea', 'wArea', 'tArea', 'fLand', 'urban'], inplace=True)

merged_state_data = pd.read_csv("data/merged_state_data.csv")

all_states = merged_state_data['State']

stateIdToNum = {v: k for k, v in all_states.items()}

merged_state_data['State'] = merged_state_data['State'].map(stateIdToNum)

combined = pd.merge(combined, merged_state_data, on=['State'], how='left')

combined['fire'] = 0

combined = combined.rename(columns={'PRCP':'prcp','EVAP':'evap','TMIN':'tmin','TMAX':'tmax', 'total_fire_size': 'fire', 'mean_elevation': 'elev', 'Land Area (sq mi)' : 'lArea','Water Area (sq mi)' : 'wArea','Total Area (sq mi)' : 'tArea','Percentage of Federal Land' : "fLand",'Urbanization Rate (%)' : 'urban'})
combined['fLand']=combined['fLand'].str.rstrip('%').astype(float)/100
combined['urban']=combined['urban'].astype(float)/100

combined = combined[['State', 'year', 'month', 'prcp', 'evap', 'tmin', 'tmax', 'elev',
       'lArea', 'wArea', 'tArea', 'fLand', 'urban', 'fire']]

combined.head(100)



# Now we impute missing values:

from sklearn.impute import SimpleImputer,KNNImputer

# imputer = SimpleImputer(strategy='mean')  # You can change the strategy to 'median', 'most_frequent', or 'constant'

imputer = KNNImputer(n_neighbors=2)
# Fit and transform the dataframe
df_imputed = pd.DataFrame(imputer.fit_transform(combined), columns=combined.columns)
df_imputed['State'] = df_imputed['State'].astype(int)
df_imputed['month'] = df_imputed['month'].astype(int)
df_imputed['year'] = df_imputed['year'].astype(int)

df_imputed = df_imputed[['State', 'year', 'month', 'prcp', 'evap', 'tmin', 'tmax', 'elev',
       'lArea', 'wArea', 'tArea', 'fLand', 'urban', 'fire']]

df_imputed.head(100)


data = pd.read_csv('train.csv')


from sklearn.preprocessing import StandardScaler
# Initialize the StandardScaler
scaler = StandardScaler()
columns_to_normalize = ["prcp", "evap", "tmin", "tmax", "elev", "lArea", "wArea", "tArea", "fLand", "urban"]
# Fit and transform the specified columns
data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])

# data.to_csv("normalised.csv", index=False)

trainData = data


# Normalising the columns. 

grouped = trainData.groupby('State')
# Create a dictionary where the key is the 'id' and the value is a smaller dataframe
result_dict = {str(key): group.drop(columns='State').reset_index(drop=True) for key, group in grouped}

print(result_dict.keys())




from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
def acc(y_pred, y_test) -> float:
    assert len(y_pred) == len(y_test)
    N = len(y_pred)
    log_errors = np.full(N, 10.0)
    log_errors = np.abs(np.log( y_pred / y_test ))
    final_scores = np.minimum(log_errors, 10)
    return np.mean(final_scores)


# 1 model on everything

X = trainData.drop(columns=['prcp', 'evap', 'tmin', 'tmax', 'elev','lArea', 'wArea', 'tArea', 'fLand', 'urban', 'fire'])
y = trainData['fire']
# Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf.fit(X, y)

Xt = df_imputed.drop(columns=['prcp', 'evap', 'tmin', 'tmax', 'elev','lArea', 'wArea', 'tArea', 'fLand', 'urban', 'fire'])

y_pred = rf.predict(Xt)


# Now make a RandomForestRegression model per state:

modelPerState = {}

for k in result_dict.keys():
    # Define features and target variable
    dataK = result_dict[k]
    X = dataK.drop(columns=['fire','tmax', 'elev','lArea', 'wArea', 'tArea', 'fLand', 'urban'])
    y = dataK['fire']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Train the model
    rf.fit(X_train, y_train)

    modelPerState[k] = rf

    # y_pred = rf.predict(X_test)

    # print(str(k) + " : " + str(acc(y_pred, y_test)))



from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

modelPerState = {}

for k in result_dict.keys():
    # Define features and target variable
    dataK = result_dict[k]
    X = dataK.drop(columns=['fire','tmax', 'elev','lArea', 'wArea', 'tArea', 'fLand', 'urban'])
    y = dataK['fire']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


    rf = Pipeline([
        ('poly', PolynomialFeatures(degree=10)),  # Adjust degree as needed
        ('linear', LinearRegression())
    ])    
    
    # rf = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Train the model
    rf.fit(X_train, y_train)

    modelPerState[k] = rf



# solution = pd.read_csv("data/zero_submission.csv")
# solution['month'] = pd.to_datetime(solution['month'], format='%Y-%m')
# solution['year'] = solution['month'].dt.year
# solution['month'] = solution['month'].dt.month

# solution = pd.get_dummies(solution, columns=['STATE'])

# newX = solution[['STATE','year','month']]
# newY = getPrediction(newX['STATE'], newX['year'], newX['month'])

solution = pd.read_csv("data/zero_submission.csv")

df = df_imputed.drop(columns=['fire','State','tmax', 'elev','lArea', 'wArea', 'tArea', 'fLand', 'urban'])
y_pred = []
for i in range(len(solution)):
    model = modelPerState[str(stateIdToNum[solution['STATE'][i]])]
    y_pred.append(model.predict(df.iloc[[i]])[0])


solution['total_fire_size'] = y_pred
solution.to_csv("data/models4dimsNorm_submission.csv", index=False)


df_imputed.to_csv("imputedTest.csv", index=False)


print(df_imputed.columns)


def score(y_pred, y_test) -> float:
    assert len(y_pred) == len(y_test)
    N = len(y_pred)
    log_errors = np.full(N, 10.0)
    log_errors = np.abs(np.log( y_pred / y_test ))
    final_scores = np.minimum(log_errors, 10)
    return np.mean(final_scores)


wildfire_sizes = pd.read_csv("data/wildfire_sizes_before_2010.csv")
wildfire_sizes['month'] = pd.to_datetime(wildfire_sizes['month'], format='%Y-%m')
wildfire_sizes['year'] = wildfire_sizes['month'].dt.year
wildfire_sizes['month'] = wildfire_sizes['month'].dt.month

wildfire_sizes = pd.get_dummies(wildfire_sizes, columns=['STATE'])
# merged_state_data = pd.read_csv("data/merged_state_data.csv")
# all_states = merged_state_data['State']
# stateIdToNum = {v: k for k, v in all_states.items()}
# wildfire_sizes['STATE'] = wildfire_sizes['STATE'].map(stateIdToNum)

# Define features and target variable
X = wildfire_sizes.drop(columns=['total_fire_size'])
y = wildfire_sizes['total_fire_size']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)


score(y_pred, y_test)


solution = pd.read_csv("data/zero_submission.csv")
solution['month'] = pd.to_datetime(solution['month'], format='%Y-%m')
solution['year'] = solution['month'].dt.year
solution['month'] = solution['month'].dt.month
  
solution = pd.get_dummies(solution, columns=['STATE'])

newX = solution.drop(columns=['ID', 'total_fire_size'])
newY = rf.predict(newX)

solution = pd.read_csv("data/zero_submission.csv")

solution['total_fire_size'] = newY
solution.to_csv("data/first_submission.csv", index=False)

