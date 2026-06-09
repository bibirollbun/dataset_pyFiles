import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
import catboost
import xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc


df1 = pd.read_csv("/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv")
df2 = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

df1.columns = df1.columns.str.strip()
df1['rainfall'] = df1['rainfall'].str.lower().map({'yes': 1, 'no': 0})
df2 = df2.drop(columns=['id'])

# Reorder columns to make them match
column_order = ['day', 'pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint',
                'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed', 'rainfall']
df1 = df1[column_order]
df2 = df2[column_order]

train_df = pd.concat([df1, df2], ignore_index=True)



# Drop 'id' column from train and test datasets
train_df.drop(columns=['id'], inplace=True, errors='ignore')



train_df.head()


train_df.describe()


train_df.info()


test_df.info()


# Check for missing values in test and train
print("Does test_df have missing values?", test_df.isnull().sum().any())
print("Does train_df have missing values?", train_df.isnull().sum().any())


print("Missing values in test_df:")
print(test_df.isnull().sum()[test_df.isnull().sum() > 0])

print("\nMissing values in train_df:")
print(train_df.isnull().sum()[train_df.isnull().sum() > 0])



train_df = train_df.fillna(train_df.mean())
test_df = test_df.fillna(test_df.mean())


# Check for missing values in test and train
print("Does test_df have missing values?", test_df.isnull().sum().any())
print("Does train_df have missing values?", train_df.isnull().sum().any())


print("Missing values in test_df:")
print(test_df.isnull().sum()[test_df.isnull().sum() > 0])

print("\nMissing values in train_df:")
print(train_df.isnull().sum()[train_df.isnull().sum() > 0])



def get_season(day):
    if 80 <= day <= 171:
        return "spring"
    elif 172 <= day <= 263:
        return "summer"
    elif 264 <= day <= 354:
        return "fall"
    else:
        return "winter"
        
train_df["season"] = train_df["day"].apply(get_season)
test_df["season"] = test_df["day"].apply(get_season)

train_df["temp_range"] = train_df["maxtemp"] - train_df["mintemp"]
test_df["temp_range"] = test_df["maxtemp"] - test_df["mintemp"]

train_df["dew_humidity_ratio"] = train_df["dewpoint"] / (train_df["humidity"] + 1e-5)
test_df["dew_humidity_ratio"] = test_df["dewpoint"] / (test_df["humidity"] + 1e-5)

train_df["temp_dew_diff"] = train_df["temparature"] - train_df["dewpoint"]
test_df["temp_dew_diff"] = test_df["temparature"] - test_df["dewpoint"]

train_df["cloud_sun_ratio"] = train_df["cloud"] / (train_df["sunshine"] + 1e-5)
test_df["cloud_sun_ratio"] = test_df["cloud"] / (test_df["sunshine"] + 1e-5)

train_df["low_sun"] = (train_df["sunshine"] < 1).astype(int)
test_df["low_sun"] = (test_df["sunshine"] < 1).astype(int)

train_df["cloud_humidity"] = train_df["humidity"] * train_df["cloud"]
test_df["cloud_humidity"] = test_df["humidity"] * test_df["cloud"]

train_df["temp_humidity"] = train_df["humidity"] * train_df["temp_dew_diff"]
test_df["temp_humidity"] = test_df["humidity"] * test_df["temp_dew_diff"]

season_map = {"winter": 0, "spring": 1, "summer": 2, "fall": 3}

train_df["season_num"] = train_df["season"].map(season_map)
test_df["season_num"] = test_df["season"].map(season_map)

train_df["cloud_sun_season"] = train_df["cloud_sun_ratio"] * train_df["season_num"]
test_df["cloud_sun_season"] = test_df["cloud_sun_ratio"] * test_df["season_num"]

train_df["cloud_sun_intersect"] = train_df["cloud"] * train_df["sunshine"]
test_df["cloud_sun_intersect"] = test_df["cloud"] * test_df["sunshine"]

train_df["cloud_humidity_intersect"] = train_df["cloud"] * train_df["humidity"]
test_df["cloud_humidity_intersect"] = test_df["cloud"] * test_df["humidity"]

train_df["cloud_sun_intersect"] = train_df["cloud"] / (train_df["sunshine"] + 1e-3)
test_df["cloud_sun_intersect"] = test_df["cloud"] / (test_df["sunshine"] + 1e-3)

train_df["humidity_dewpoint_intersect"] = train_df["humidity"] * train_df["dewpoint"]
test_df["humidity_dewpoint_intersect"] = test_df["humidity"] * test_df["dewpoint"]

train_df["sun_wind_intersect"] = train_df["sunshine"] / (train_df["windspeed"] + 1e-3)
test_df["sun_wind_intersect"] = test_df["sunshine"] / (test_df["windspeed"] + 1e-3)

train_df["cloud_low_sun_intersect"] = train_df["cloud"] * train_df["low_sun"]
test_df["cloud_low_sun_intersect"] = test_df["cloud"] * test_df["low_sun"]

bool_cols = train_df.select_dtypes(include='bool').columns

for col in bool_cols:
    train_df[col] = train_df[col].astype(int)
    test_df[col] = test_df[col].astype(int)
    
train_df = train_df.drop(["season"], axis=1)
test_df = test_df.drop(["season"], axis=1)

test_df["winddirection"].fillna(test_df["winddirection"].mean(), inplace=True)



X = train_df.drop(columns=['rainfall'])
y = train_df['rainfall']



scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



def purged_cross_validation(X, y, n_splits=5, purge_length=1):
    ts_split = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, val_idx in ts_split.split(X):
        val_idx = val_idx[val_idx >= train_idx[-purge_length]]
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        yield X_train, X_val, y_train, y_val



# Define all tuned models
l1 = LogisticRegression(penalty='l1', solver='saga', C=0.1, max_iter=1000)
l2 = LogisticRegression(penalty='l2', solver='lbfgs', C=1.0, max_iter=1000)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100, max_depth=6, learning_rate=0.1)
cat = CatBoostClassifier(verbose=0, iterations=100, learning_rate=0.1, depth=6)
svc = SVC(kernel='rbf', C=1, gamma=0.01, probability=True)
rf = RandomForestClassifier(n_estimators=100, max_depth=10)
et = ExtraTreesClassifier(n_estimators=100)
ann = MLPClassifier(hidden_layer_sizes=(3, 5), activation='relu', solver='adam', max_iter=1000)

# Define the voting classifier
model = VotingClassifier(
    estimators=[
        ('l1', l1),
        ('l2', l2),
        ('xgb', xgb),
        ('cat', cat),
        ('svc', svc),
        ('rf', rf),
        ('et', et),
        ('ann', ann)
    ],
    voting='soft'
)



# Assuming X_scaled and y are already defined
for fold_idx, (X_train, X_val, y_train, y_val) in enumerate(purged_cross_validation(X_scaled, y)):
    print(f"\n--- Fold {fold_idx + 1} ---")
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    probas = model.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds)
    fpr, tpr, _ = roc_curve(y_val, probas)
    roc_auc = auc(fpr, tpr)

    print("Accuracy:", acc)
    print("F1 Score:", f1)
    print("AUC:", roc_auc)




roc_auc = auc(fpr, tpr)

# Plotting the ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid()
plt.show()



# Load test set
test_ids = test_df['id']
X_test = test_df.drop(columns=["id"])

# Scale
X_test_scaled = scaler.transform(X_test)

# Fit final model on full training data
model.fit(X_scaled, y)

# Ensure X_test_scaled has correct column names
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

# Predict probabilities
preds_test = model.predict_proba(X_test_scaled)[:, 1]

# Create submission DataFrame with probabilities
submission = pd.DataFrame({'id': test_ids, 'rainfall': preds_test})
submission.to_csv('/kaggle/working/submission.csv', index=False)



submission










