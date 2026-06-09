import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
submission_df = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


train_df.info()
test_df.info()


train_df.head(10)


test_df.head(10)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# Combine unique values from train and test for 'Soil Type'
combined_soil_types = pd.concat([train_df['Soil Type'], test_df['Soil Type']], axis=0).astype(str).unique()
le.fit(combined_soil_types)

# Transform 'Soil Type' in both dataframes
train_df['Soil Type'] = le.transform(train_df['Soil Type'].astype(str))
test_df['Soil Type'] = le.transform(test_df['Soil Type'].astype(str))

# Repeat the process for 'Crop Type'
combined_crop_types = pd.concat([train_df['Crop Type'], test_df['Crop Type']], axis=0).astype(str).unique()
le.fit(combined_crop_types)

train_df['Crop Type'] = le.transform(train_df['Crop Type'].astype(str))
test_df['Crop Type'] = le.transform(test_df['Crop Type'].astype(str))

# For 'Fertilizer Name', it only exists in the training data, so fitting only on train is sufficient.
# However, it's good practice to ensure consistency in data types.
le.fit(train_df['Fertilizer Name'].astype(str))
train_df['Fertilizer Name'] = le.transform(train_df['Fertilizer Name'].astype(str))



# Feature Engineering: Add interaction terms
train_df['N_P_ratio'] = train_df['Nitrogen'] / (train_df['Phosphorous'] + 1)
train_df['N_K_ratio'] = train_df['Nitrogen'] / (train_df['Potassium'] + 1)
test_df['N_P_ratio'] = test_df['Nitrogen'] / (test_df['Phosphorous'] + 1)
test_df['N_K_ratio'] = test_df['Nitrogen'] / (test_df['Potassium'] + 1)


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, average_precision_score


# Features and target
X = train_df.drop(['id', 'Fertilizer Name'], axis=1)
y = train_df['Fertilizer Name']
X_test = test_df.drop('id', axis=1)

# Split training data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Visualization: Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


# Model Training: Tuned Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42
)
rf.fit(X_train, y_train)


# MAP@3 Calculation
def map_at_3(y_true, y_pred, k=3):
    n = len(y_true)
    ap_sum = 0
    for i in range(n):
        relevant = y_true[i]
        pred = y_pred[i][:k]
        precisions = 0
        rel_count = 0
        for j in range(min(k, len(pred))):
            if pred[j] == relevant:
                rel_count += 1
                precisions += rel_count / (j + 1)
                break
        ap_sum += precisions / min(1, rel_count) if rel_count > 0 else 0
    return ap_sum / n


# Validation: Top 3 predictions
y_pred_proba = rf.predict_proba(X_val)
top3_preds = np.argsort(-y_pred_proba, axis=1)[:, :3]
top3_labels = le.inverse_transform(top3_preds.ravel()).reshape(top3_preds.shape)
y_val_labels = le.inverse_transform(y_val)
map_score = map_at_3(y_val_labels, top3_labels)
print(f"MAP@3 Score: {map_score:.4f}")


# Cross-validation
cv_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


# Test predictions
test_pred_proba = rf.predict_proba(X_test)
top3_test_preds = np.argsort(-test_pred_proba, axis=1)[:, :3]
top3_test_labels = le.inverse_transform(top3_test_preds.ravel()).reshape(top3_test_preds.shape)


# Format submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'Fertilizer Name': [' '.join(preds) for preds in top3_test_labels]
})
submission.to_csv("submissionss.csv", index=False)
print("Submission file saved as submission.csv")




