import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")


# Load data
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

train=pd.read_csv(r"/kaggle/input/playground-series-s5e7/train.csv")
test=pd.read_csv(r"/kaggle/input/playground-series-s5e7/test.csv")
submission=pd.read_csv(r"/kaggle/input/playground-series-s5e7/sample_submission.csv")

# Prepare data
X = train.drop(columns=["id", "Personality"])
y = train["Personality"]
test_ids = test["id"]
X_test = test.drop(columns=["id"])

# Encode categorical features
for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    X_test[col] = le.transform(X_test[col])

# Encode target
le_y = LabelEncoder()
y_encoded = le_y.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Train model
model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.85,
    colsample_bytree=0.9,
    gamma=0.1,
    reg_alpha=0.3,
    reg_lambda=0.8,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)
model.fit(X_scaled, y_encoded)

# Predict and create submission
preds = model.predict(X_test_scaled)
pred_labels = le_y.inverse_transform(preds)

submission = pd.DataFrame({
    "id": test_ids,
    "Personality": pred_labels
})
print("Submission Is Complete")
submission.to_csv(r"submissionXF.csv", index=False)

