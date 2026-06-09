import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# Config
DATA_DIR = '/kaggle/input/playground-series-s5e9/'
TARGET_COL = "BeatsPerMinute"
ID_COL = "id"

# Load data
train = pd.read_csv(DATA_DIR + "train.csv")
test = pd.read_csv(DATA_DIR + "test.csv")

train = pd.concat([train, train], axis=0, ignore_index=True)

# Then extract X, y again
features = [c for c in train.columns if c not in [ID_COL, TARGET_COL]]
X = train[features]
y = train[TARGET_COL]

# Now the indices are aligned and OLS will work:
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)

X_scaled = sm.add_constant(X_scaled)

model = sm.OLS(y, X_scaled).fit()
print(model.summary())

# Prepare test set
X_test = test[features]
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=features)
X_test_scaled = sm.add_constant(X_test_scaled)

# Predict on test data
test_preds = model.predict(X_test_scaled)

# Prepare submission
submission = pd.DataFrame({
    ID_COL: test[ID_COL],
    TARGET_COL: test_preds
})

submission.to_csv("submission.csv", index=False)
print("Submission saved as submission.csv")



pd.read_csv("/kaggle/working/submission.csv")




