# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from IPython.display import HTML

# Center all plots
HTML("<style>.output_png {display: block; margin: auto;}</style>")

# Set a consistent default figure size
plt.rcParams['figure.figsize'] = (6, 4)  # width=6, height=4 inches


# Load train and test data
train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")


# Check data structure
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print("\nFirst 5 rows of train data:")
display(train.head())


import warnings
warnings.filterwarnings("ignore")

# Target distribution
plt.figure()
sns.histplot(train["BeatsPerMinute"], bins=50, kde=True)
plt.title("Distribution of Beats Per Minute")
plt.show()


import warnings
warnings.filterwarnings("ignore")

# Correlation check
plt.figure()
sns.heatmap(train.corr(), cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.show()


# Prepare features
X = train.drop(["id","BeatsPerMinute"], axis=1)
y = train["BeatsPerMinute"]
X_test = test.drop("id", axis=1)


# Train/Validation split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Random Forest model
model = RandomForestRegressor(
    n_estimators=100,   # limited trees
    max_depth=12,       # cap depth (prevents overfitting + speeds up)
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# Evaluate on validation set
y_pred = model.predict(X_valid)
rmse = mean_squared_error(y_valid, y_pred, squared=False)
print("Validation RMSE:", rmse)


# Train on full data
model.fit(X, y)


# Make predictions
test_preds = model.predict(X_test)


# Create submission file df
submission = pd.DataFrame({
    "id": test["id"],
    "BeatsPerMinute": test_preds
})

# Create csv
submission.to_csv("submission.csv", index=False)
print("Submission file created:", submission.shape)
print(submission.head())

