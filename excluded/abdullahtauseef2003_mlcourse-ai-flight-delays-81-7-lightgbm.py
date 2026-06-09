import pandas as pd

train_df = pd.read_csv("/kaggle/input/flight-delays-fall-2018/flight_delays_train.csv.zip")
train_df.head()


train_df.info()


train_df.describe()


train_df.isnull().sum()


train_df.nunique()


train_df1 = train_df.copy()
train_df1["Month"] = train_df1["Month"].apply(lambda x: x.split("-")[1])
train_df1["DayofMonth"] = train_df1["DayofMonth"].apply(lambda x: x.split("-")[1])
train_df1["DayOfWeek"] = train_df1["DayOfWeek"].apply(lambda x: x.split("-")[1])
train_df1.head()


def weekClass(df):
    df["DayOfWeek"] = df["DayOfWeek"].astype(float)
    df["DayType"] = df["DayOfWeek"].apply(lambda x: "Weekend" if x > 5 else "Weekday")
    return df

train_df2 = train_df1.copy()
train_df2 = weekClass(train_df2)
train_df2.head()


def dayTime(x):
    if 459 < x <= 1159:  # Corrected range for "Morning"
        return "Morning"
    elif 1159 < x <= 1659:  # Corrected range for "Afternoon"
        return "Afternoon"
    elif 1659 < x <= 2059:  # Corrected range for "Evening"
        return "Evening"
    else:
        return "Night"

train_df3 = train_df2.copy()
train_df3["TimeType"] = train_df3["DepTime"].apply(dayTime)
train_df3.head()


def is_us_or_international_holiday(month, day):
    # Define US-specific holidays as (month, day) tuples
    us_specific_holidays = {
        (1, 15),  # Example: Martin Luther King Jr. Day (use fixed date or dynamic logic for third Monday)
        (7, 4),   # Independence Day
        (11, 11), # Veterans Day
    }
    
    # Define internationally recognized holidays observed in the US
    international_holidays = {
        (1, 1),   # New Year's Day
        (9, 1),   # Labor Day (use dynamic logic for first Monday in September if needed)
        (12, 25), # Christmas Day
    }
    
    # Combine both sets for a comprehensive check
    all_holidays = us_specific_holidays.union(international_holidays)
    
    # Return the holiday type
    if (month, day) in us_specific_holidays:
        return "US-Specific Holiday"
    elif (month, day) in international_holidays:
        return "Internationally Recognized Holiday"
    else:
        return "Non-Holiday"

# Example usage
train_df4 = train_df3.copy()
train_df4["HolidayType"] = train_df3.apply(lambda row: is_us_or_international_holiday(row["Month"], row["DayofMonth"]), axis=1)

train_df4.head()


train_df4.HolidayType.nunique()


train_df4 = train_df4.drop(["HolidayType"], axis = 1)
train_df4.head()


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
train_df5 = train_df4.copy()
train_df5["DayType"] = le.fit_transform(train_df5["DayType"])
train_df5["TimeType"] = le.fit_transform(train_df5["TimeType"])
train_df5["dep_delayed_15min"] = le.fit_transform(train_df5["dep_delayed_15min"])
train_df5.head()


train_df5["UniqueCarrier"].unique()


# Count occurrences of each unique value in the "UniqueCarrier" column
carrier_counts = train_df5["UniqueCarrier"].value_counts()
# Display the counts
print(carrier_counts)


def carrierNum(carrier):
    if carrier_counts[carrier] < 1000:
        return "Other"
    else:
        return carrier

# Apply the function to the column
train_df6 = train_df5.copy()
train_df6["UniqueCarrier"] = train_df6["UniqueCarrier"].apply(carrierNum)

# Display the updated DataFrame
train_df6.head()


train_df6["UniqueCarrier"].unique()


originCount = train_df6["Origin"].value_counts()
print(originCount)


originCount.describe()


def originNum(origin):
    if originCount[origin] < 50:
        return "Other"
    else:
        return origin

train_df7 = train_df6.copy()
train_df7["Origin"] = train_df7["Origin"].apply(originNum)
train_df7.head()


train_df7["Origin"].nunique()


destCount = train_df6["Dest"].value_counts()
print(destCount)


destCount.describe()


def destNum(dest):
    if destCount[dest] < 50:
        return "Other"
    else:
        return dest

train_df8 = train_df7.copy()
train_df8["Dest"] = train_df8["Dest"].apply(destNum)
train_df8.head()


train_df8["Dest"].nunique()


dummies = pd.get_dummies(train_df8["UniqueCarrier"])
dummies.head()


train_df9 = pd.concat([train_df8, dummies], axis = "columns")
train_df9 = train_df9.drop(["UniqueCarrier", "YV", "HP"], axis = 1)
train_df9


new_le = LabelEncoder()
train_df10 = train_df9.copy()
train_df10["Origin"] = new_le.fit_transform(train_df10["Origin"])
train_df10["Dest"] = new_le.fit_transform(train_df10["Dest"])
train_df10.head()


train_df10.dtypes


train_df10["Month"] = train_df10["Month"].astype("int64")
train_df10["DayofMonth"] = train_df10["DayofMonth"].astype("int64")
train_df10.dtypes


cr = train_df10.corr()


import matplotlib.pyplot as plt
import seaborn as sns

# Set the figure size
plt.figure(figsize=(20, 10))  # Change (12, 8) to your desired dimensions

# Plot the heatmap
sns.heatmap(cr, xticklabels=cr.columns, yticklabels=cr.columns, annot=True)

# Show the plot
plt.show()


def plot_boxplots(df):
    num_features = df.select_dtypes(include=['number']).columns  # Select only numerical columns
    for feature in num_features:
        plt.figure(figsize=(6, 4))
        sns.catplot(x="dep_delayed_15min", y=feature, data=df, kind="box", height=4, aspect=1.5)
        plt.title(f"Boxplot of {feature} by dep_delayed_15min")
        plt.grid()
        plt.show()

plot_boxplots(train_df10)


# Assuming 'df' is your dataset and 'Distance' is the column to process
# Calculate the IQR
Q1 = train_df10['Distance'].quantile(0.25)  # 25th percentile
Q3 = train_df10['Distance'].quantile(0.75)  # 75th percentile
IQR = Q3 - Q1

# Define the lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Winsorize (Cap) the outliers
train_df10['Distance'] = train_df10['Distance'].apply(lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x))

# Print summary after capping
print("After Winsorizing:")
print(train_df10['Distance'].describe())


x = train_df10.drop("dep_delayed_15min", axis = 1)
y = train_df10["dep_delayed_15min"]


from sklearn.feature_selection import mutual_info_classif, RFE

mi_scores = mutual_info_classif(x, y)
mi_scores = pd.Series(mi_scores, index=x.columns).sort_values(ascending=False)
print("\nMutual Information Scores:")
print(mi_scores)


selected_features_mi = mi_scores.index[:5].tolist()
print(f"\nTop Features Based on Mutual Information: {selected_features_mi}")


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
rfe = RFE(model, n_features_to_select=5)
rfe.fit(x, y)

selected_features_rfe = x.columns[rfe.support_].tolist()
print(f"\nTop Features Based on RFE: {selected_features_rfe}")


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(x, y)
feature_importance = pd.Series(rf.feature_importances_, index=x.columns).sort_values(ascending=False)

print("\nFeature Importance Scores (Random Forest):")
print(feature_importance)

# Select top 5 features
selected_features_rf = feature_importance.index[:5].tolist()
print(f"\nTop Features Based on Random Forest: {selected_features_rf}")

### FINAL SELECTED FEATURES ###
final_selected_features = list(set(selected_features_mi + selected_features_rfe + selected_features_rf))
print(f"\nFinal Selected Features: {final_selected_features}")


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_filtered = x[final_selected_features]
x_scaled = sc.fit_transform(x_filtered)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size = 0.2, random_state = 10)


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score


# Define models and parameter grids
models = {
    "LogisticRegression": (LogisticRegression(max_iter=1000, solver='liblinear'), {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2']
    }),
    "DecisionTree": (DecisionTreeClassifier(), {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'criterion': ['gini', 'entropy']
    }),
    "RandomForest": (RandomForestClassifier(), {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'criterion': ['gini', 'entropy']
    }),
    "XGBoost": (XGBClassifier(use_label_encoder=False, eval_metric='logloss'), {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 6, 9]
    }),
    "LightGBM": (LGBMClassifier(), {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'num_leaves': [31, 50, 100]
    })
}

# Train and tune each model using GridSearchCV
best_estimators = {}
for model_name, (model, param_grid) in models.items():
    print(f"Tuning {model_name}...")
    grid_search = GridSearchCV(
        model, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1
    )
    grid_search.fit(x_train, y_train)
    best_estimators[model_name] = grid_search.best_estimator_
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best ROC-AUC score (CV): {grid_search.best_score_:.4f}")

# Test the best models on the test set
for model_name, model in best_estimators.items():
    y_pred_proba = model.predict_proba(x_test)[:, 1]  # Get probabilities for positive class
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, model.predict(x_test))
    print(f"Test ROC-AUC for {model_name}: {roc_auc:.4f}")
    print(f"Test accuracy for {model_name}: {acc:.4f}")


test_df = pd.read_csv("/kaggle/input/flight-delays-fall-2018/flight_delays_test.csv.zip")
test_df.head()


test_df1 = test_df.copy()
test_df1["Month"] = test_df1["Month"].apply(lambda x: x.split("-")[1])
test_df1["DayofMonth"] = test_df1["DayofMonth"].apply(lambda x: x.split("-")[1])
test_df1["DayOfWeek"] = test_df1["DayOfWeek"].apply(lambda x: x.split("-")[1])
test_df1.head()


test_df2 = test_df1.copy()
test_df2 = weekClass(test_df2)
test_df2.head()


test_df3 = test_df2.copy()
test_df3["TimeType"] = test_df3["DepTime"].apply(dayTime)
test_df3.head()


test_df4 = test_df3.copy()
test_df4["DayType"] = le.fit_transform(test_df4["DayType"])
test_df4["TimeType"] = le.fit_transform(test_df4["TimeType"])
test_df4.head()


# List of valid unique carrier names
valid_carriers = ['AA', 'US', 'XE', 'OO', 'WN', 'NW', 'DL', 'OH', 'AS', 'UA',
                  'MQ', 'CO', 'EV', 'Other', 'YV', 'F9', 'HP', 'B6', 'FL']
test_df5 = test_df4.copy()
test_df5['UniqueCarrier'] = test_df5['UniqueCarrier'].apply(lambda x: x if x in valid_carriers else 'Other')
test_df5.head()


test_df5.UniqueCarrier.unique()


# Get the unique list of Origin values in test_df
valid_origins = train_df7['Origin'].unique()
test_df6 = test_df5.copy()
test_df6['Origin'] = test_df5['Origin'].apply(lambda x: x if x in valid_origins else 'Other')
test_df6.head()


test_df6.Origin.unique()


# Get the unique list of Origin values in test_df
valid_origins = train_df8['Dest'].unique()
test_df7 = test_df6.copy()
test_df7['Dest'] = test_df7['Dest'].apply(lambda x: x if x in valid_origins else 'Other')
test_df7.head()


test_df7.Dest.unique()


test_dummies = pd.get_dummies(test_df7.UniqueCarrier)
test_dummies.head()


test_df8 = pd.concat([test_df7, test_dummies], axis = "columns")
test_df8 = test_df8.drop(["UniqueCarrier", "YV"], axis = 1)
test_df8


test_df9 = test_df8.copy()
test_df9["Origin"] = new_le.fit_transform(test_df9["Origin"])
test_df9["Dest"] = new_le.fit_transform(test_df9["Dest"])
test_df9.head()


test_df9.dtypes


test_df9["Month"] = test_df9["Month"].astype("int64")
test_df9["DayofMonth"] = test_df9["DayofMonth"].astype("int64")
test_df9.dtypes


# Correct way to select multiple columns
test_df10 = test_df9[final_selected_features]
test_df10.head()


pred_scaled = sc.fit_transform(test_df10)


# Add an ID column to the test DataFrame
test_df['id'] = range(1, len(test_df) + 1)  # Create a sequential ID starting from 1

# Predict prices for the test dataset using the best model
test_predictions = best_estimators["LightGBM"].predict(pred_scaled)  # Replace 'YourBestModelName' with the chosen model

# Save predictions to a submission file
submission = pd.DataFrame({
    "id": test_df["id"],  # Use the newly created 'id' column
    "price": test_predictions
})
submission.to_csv("submission.csv", index=False)

print("Submission file saved as 'submission.csv'.")


import pickle

# Save the model properly
with open("flight_delay_lightgbm.pkl", "wb") as f:
    pickle.dump(best_estimators["LightGBM"], f)

