import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer


# Load datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv',index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv',index_col='id')

train.shape, test.shape



print(test.head(10))
print(test.isna().sum())


imputer = SimpleImputer(strategy="mean")
test[:] = imputer.fit_transform(test)

print(test.head(10))
print(test.isna().sum())


# Define features and target
X = train.drop(columns=["rainfall"])
y = train["rainfall"]

X.shape, y.shape


# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)


model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, n_jobs=1)

# Train the model
model.fit(X_train,y_train)



y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val,y_pred)

print(f"Random Forest train model accuracy : {accuracy}")


probabilites_test = model.predict_proba(test)
probabilites_classe_1 = probabilites_test[:, 1]

# Convert to DataFrame
submission = pd.DataFrame({"id": test.index, "rainfall": probabilites_classe_1})
print(submission.head(20))
# Save submission file
submission.to_csv("submission.csv", index=False)

