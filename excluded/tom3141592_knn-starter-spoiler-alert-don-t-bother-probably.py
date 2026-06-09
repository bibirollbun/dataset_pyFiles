import numpy as np
import pandas as pd


from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_log_error


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


def calculate_max_hr(row):
    age = row["Age"]
    sex = row["Sex"].lower()

    if sex == "male":
        return 208 - (0.8*age)
    elif sex == "female":
        return 206 - 0.88 * age
    else:
        return 220 - age

def calculate_fthr(max_hr):
    return 0.85 * max_hr if max_hr else None

def calculate_percentage_of_fthr(row):
    hr = row["Heart_Rate"]
    fthr = row["fthr"]

    return hr/fthr

def calculate_tss(row):
    avg_hr = row["Heart_Rate"]
    duration_min = row["Duration"]
    fthr = row["fthr"]
    if not fthr or fthr == 0:
        return None
    intensity = avg_hr / fthr
    duration_hr = duration_min / 60
    return round(duration_hr * (intensity ** 2) * 100, 1)


train["max_hr"] = train.apply(calculate_max_hr, axis=1)
train["fthr"] = train["max_hr"].apply(calculate_fthr)
train["perc_fthr"] = train.apply(calculate_percentage_of_fthr, axis=1)
train["tss"] = train.apply(calculate_tss, axis=1)

test["max_hr"] = test.apply(calculate_max_hr, axis=1)
test["fthr"] = test["max_hr"].apply(calculate_fthr)
test["perc_fthr"] = test.apply(calculate_percentage_of_fthr, axis=1)
test["tss"] = test.apply(calculate_tss, axis=1)


#train = train.sample(n=100000)
X = train.drop("Calories", axis='columns')
y = train['Calories']
X_test = test


# Encode categorical variable ('Sex')
label_encoder = LabelEncoder()
X['Sex'] = label_encoder.fit_transform(X['Sex'])  # 0 = female, 1 = male

# Define features and target
X = X.drop(columns=['id'])  # Drop ID and target column

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize kNN model
knn = KNeighborsRegressor(n_neighbors=13)

# Perform cross-validation manually
fold = 1
rmsle_scores = []

for train_index, val_index in kf.split(X_scaled):
    X_train, X_val = X_scaled[train_index], X_scaled[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Fit model on training data
    knn.fit(X_train, y_train)

    # Predict on validation set
    y_pred = knn.predict(X_val)

    # Calculate RMSLE
    rmsle = np.sqrt(mean_squared_log_error(y_val, y_pred))
    rmsle_scores.append(rmsle)

    print(f"Fold {fold} - RMSLE: {rmsle:.4f}")
    fold += 1

# Print average RMSLE across folds
print(f"\nAverage RMSLE: {np.mean(rmsle_scores):.4f}")

# Fit model on entire training set
knn.fit(X_scaled, y)

X_test['Sex'] = label_encoder.transform(X_test['Sex'])  # Ensure same encoding
test_X = X_test.drop(columns=['id'])  # Drop ID column
test_X_scaled = scaler.transform(test_X)

# Predict on test data
test_predictions = knn.predict(test_X_scaled)
print(f"\nPredicted Calories for Test Data:\n{test_predictions}")


submission = pd.DataFrame({
    "id": range(750000, 750000 + len(test_predictions)),
    "Calories": test_predictions
})


submission


submission.to_csv("submission.csv", index=False)

