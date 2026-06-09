#Import libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor


train = pd.read_csv("/kaggle/input/kaggle-competition/train.csv")
test = pd.read_csv("/kaggle/input/kaggle-competition/test.csv")
sample = pd.read_csv("/kaggle/input/kaggle-competition/sample_submission.csv")

print(train.shape, test.shape, sample.shape)
train.head()



print(train.columns)



X = train.drop(["BeatsPerMinute", "id"], axis=1)
y = train["BeatsPerMinute"]

X_test = test.drop("id", axis=1)



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor

# --------------------------
# 1. Load data
# --------------------------
train = pd.read_csv("/kaggle/input/kaggle-competition/train.csv")
test = pd.read_csv("/kaggle/input/kaggle-competition/test.csv")
sample = pd.read_csv("/kaggle/input/kaggle-competition/sample_submission.csv")

# --------------------------
# 2. Split features/target
# --------------------------
X = train.drop(["BeatsPerMinute", "id"], axis=1)
y = train["BeatsPerMinute"]
X_test = test.drop("id", axis=1)

# --------------------------
# 3. Train/validation split
# --------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------
# 4. Train model
# --------------------------
model = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    random_state=42
)
model.fit(X_train, y_train)

# --------------------------
# 5. Validation
# --------------------------
val_preds = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_preds))
print("Validation RMSE:", rmse)

# --------------------------
# 6. Predict on test
# --------------------------
final_preds = model.predict(X_test)

submission = pd.DataFrame({
    "ID": test["id"],            # Kaggle expects "ID"
    "BeatsPerMinute": final_preds
})
submission.to_csv("submission.csv", index=False)
print("submission.csv saved!")


