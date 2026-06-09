# =====================================================
# ğŸ�¶ GWC Intro to ML â€” PetFinder Adoption Speed 
# =====================================================

# STEP 0: sanity-check the data is mounted
import os, textwrap, pandas as pd, numpy as np
from pathlib import Path

BASE = Path("/kaggle/input/petfinder-adoption-prediction")
print("Exists?", BASE.exists())
print("Files:", [p.name for p in BASE.iterdir()][:10])

# =====================================================
# 1) IMPORTS
# =====================================================
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# pretty plots
plt.rcParams["figure.figsize"] = (7,4)
plt.rcParams["figure.dpi"] = 120

# =====================================================
# 2) LOAD DATA
# =====================================================
train_path = BASE/"train/train.csv"
test_path  = BASE/"test/test.csv"
sample_sub = BASE/"test/sample_submission.csv"

data = pd.read_csv(train_path)
print("shape:", data.shape)
data.head()



# =====================================================
# 3) QUICK EXPLORATORY DATA ANALYSIS (EDA)
# =====================================================

print(data[["Type","Age","PhotoAmt","AdoptionSpeed"]].describe())

sns.countplot(x="AdoptionSpeed", data=data)
plt.title("Adoption Speed distribution (0 fast â†’ 4 slow)")
plt.show()

sns.boxplot(x="Type", y="AdoptionSpeed", data=data)
plt.title("Adoption Speed by Type (1=Dog, 2=Cat)")
plt.show()

sns.scatterplot(x="Age", y="AdoptionSpeed", data=data.sample(5000, random_state=42))
plt.title("Age vs Adoption Speed (sample)")
plt.show()



# =====================================================
# 4) DATA ENGINEERING 
# =====================================================
# Keep it beginner-friendly: select a few numeric features
cols = ["Type","Age","PhotoAmt","AdoptionSpeed"]
df = data[cols].dropna()

X = df[["Type","Age","PhotoAmt"]]
y = df["AdoptionSpeed"]  # ordinal 0..4; we'll treat as regression for simplicity

# train/validation split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train.shape, X_valid.shape



# =====================================================
# 5) BASELINE MODEL â€” LINEAR REGRESSION
# =====================================================
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
p_lr = lin_reg.predict(X_valid)

print("Linear Regression R^2:", r2_score(y_valid, p_lr))
print("Linear Regression RMSE:", mean_squared_error(y_valid, p_lr, squared=False))



# =====================================================
# 6) DECISION TREE MODEL (nonlinear)
# =====================================================
tree = DecisionTreeRegressor(max_depth=6, random_state=42)
tree.fit(X_train, y_train)
p_tree = tree.predict(X_valid)

print("Decision Tree R^2:", r2_score(y_valid, p_tree))
print("Decision Tree RMSE:", mean_squared_error(y_valid, p_tree, squared=False))



# =====================================================
# 7) VISUAL â€” PREDICTION COMPARISON
# =====================================================
plt.scatter(y_valid, p_lr, alpha=0.6, label="Linear Reg")
plt.scatter(y_valid, p_tree, alpha=0.6, label="Decision Tree")
plt.xlabel("Actual AdoptionSpeed")
plt.ylabel("Predicted (regression)")
plt.legend()
plt.title("Model predictions vs. actual")
plt.show()



# =====================================================
# BONUS 8) CREATE A SUBMISSION
# =====================================================
# retrain on ALL rows (using our simple features)
X_all = df[["Type","Age","PhotoAmt"]]
y_all = df["AdoptionSpeed"]
tree.fit(X_all, y_all)

test_df = pd.read_csv(test_path)
X_test_simple = test_df[["Type","Age","PhotoAmt"]].fillna(0)

pred_test = tree.predict(X_test_simple)
pred_test = np.clip(np.rint(pred_test), 0, 4).astype(int)

sub = pd.read_csv(sample_sub)  # has 'PetID' and 'AdoptionSpeed' columns
sub["AdoptionSpeed"] = pred_test
sub.head()


# save submission
out_path = Path("/kaggle/working/submission.csv")
sub.to_csv(out_path, index=False)
print("wrote:", out_path, " â€” now click 'Save Version' â†’ 'Submit to Competition'")

