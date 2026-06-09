import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import average_precision_score, precision_recall_curve


df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
df.head()


df.info()


print("Soil Type:")
print(df['Soil Type'].unique())
print("\nCrop Type:")
print(df['Crop Type'].unique())
print("\nFertilizer Name:")
print(df['Fertilizer Name'].unique())


df_encoded = pd.get_dummies(df, columns=['Soil Type', 'Crop Type'], drop_first=True)
df_encoded.head()


df_encoded.columns


X = df_encoded.drop(['id','Fertilizer Name'], axis=1).astype(int)
y = df['Fertilizer Name']
print("\nShape data fitur (X):", X.shape)
print("Shape data target (y):", y.shape)
X.head()


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Validation set size: {X_val.shape[0]} samples")
print(f"Number of unique classes in target: {len(label_encoder.classes_)}")


model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=len(label_encoder.classes_),
    eval_metric='mlogloss', # Evaluation metrics for multiclass classification
    n_estimators=100,      # Number of boosting rounds (adjustable)
    learning_rate=0.1,     # Learning level (can be adjusted)
    use_label_encoder=False, # disarankan False
    random_state=42,
    tree_method='hist',    # Can 'hist' for faster performance on large datasets
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.005,
    reg_lambda=1
)

print("\nStart training model XGBoost...")
model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          early_stopping_rounds=10, # Stop if performance does not improve for 10 rounds
          verbose=True)

print("\nTraining Complete!")


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
y_pred = model.predict(X_val) # Make predictions on testing data

print("\n--- Model Evaluation Results ---")

# accuracy
accuracy = accuracy_score(y_val, y_pred)
print(f"Model accuracy: {accuracy:.4f}") # Tampilkan 4 angka di belakang koma

# Classification Report (Precision, Recall, F1-Score)
print("\nClassification Report:")
print(classification_report(y_val, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_val, y_pred)
print(cm)


# Probability prediction for validation set
y_pred_proba = model.predict_proba(X_val)

# Take 3 classes with the highest probability for each sample
top_k = 3
y_pred_top_k_indices = np.argsort(y_pred_proba, axis=1)[:, -top_k:][:, ::-1]

# Create a binary matrix for ground truth (true labels)
y_true_binary = np.zeros_like(y_pred_proba)
y_true_binary[np.arange(len(y_val)), y_val] = 1


# Function to calculate mAP@k
def map_at_k(y_true, y_pred_proba, k=3):
    total_precision = 0
    num_samples = y_true.shape[0]

    for i in range(num_samples):
        # Get the class index with the highest probability
        ranked_indices = np.argsort(y_pred_proba[i])[::-1] # Sort by highest probability

        # Take the top k predictions
        top_k_predictions = ranked_indices[:k]

        # Check if the ground truth is in the top k predictions
        # If the ground truth is class 'c', and 'c' is in the top_k_predictions
        # So relevant = 1, if not relevant = 0
        relevant_in_top_k = 0
        if y_true[i] in top_k_predictions:
            # Find the ground truth position among the top-k predictions
            position = np.where(top_k_predictions == y_true[i])[0][0] + 1 # +1 karena posisi 1-indexed
            relevant_in_top_k = 1 / position # precision di posisi itu (1/position)

        total_precision += relevant_in_top_k
    return total_precision / num_samples

# Calculate mAP@3
mAP_3 = map_at_k(y_val, y_pred_proba, k=3)
print(f"\nmAP@3: {mAP_3:.4f}")


# To see the original predictions and labels for example
print("\nExample Prediction vs. Origial Sample:")
for i in range(5):
    actual_label_encoded = y_val[i]
    actual_label_name = label_encoder.inverse_transform([actual_label_encoded])[0]

    predicted_top_3_indices = y_pred_top_k_indices[i]
    predicted_top_3_names = label_encoder.inverse_transform(predicted_top_3_indices)
    predicted_top_3_probabilities = y_pred_proba[i, predicted_top_3_indices]

    print(f"Sample {i+1}:")
    print(f"  Original Sample: {actual_label_name} (Encoded: {actual_label_encoded})")
    print(f"  Top 3 Prediction:")
    for j in range(top_k):
        print(f"    - {predicted_top_3_names[j]} (Prob: {predicted_top_3_probabilities[j]:.4f})")
    print("-" * 30)


df2 = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
df2.head()


df.info()


print("Soil Type:")
print(df2['Soil Type'].unique())
print("\nCrop Type:")
print(df2['Crop Type'].unique())


df2_encoded = pd.get_dummies(df2, columns=['Soil Type', 'Crop Type'], drop_first=True)
df2_encoded.head()


idData = df2['id']
A = df2_encoded.drop('id', axis=1).astype(int)
A.head()


print("\nPerforming predictions on the test dataset...")
# Prediction of probability for each class
y_test_pred_proba = model.predict_proba(A)

# Take 3 class indices with the highest probability for each sample.
top_k = 3
y_test_top_k_indices = np.argsort(y_test_pred_proba, axis=1)[:, -top_k:][:, ::-1]

# Cek dulu isi dari label_encoder.classes_ untuk debugging
print("\nFill from label_encoder.classes_:", label_encoder.classes_)
print("Top 3 prediction index example (before inverse_transform):", y_test_top_k_indices[0])

# Convert class index back to fertilizer name using LabelEncoder
predicted_fertilizer_names = label_encoder.inverse_transform(y_test_top_k_indices.flatten()).astype(str).reshape(-1, top_k)

# Create a result DataFrame
output_df = pd.DataFrame({'id': idData})
output_df['Fertilizer Name'] = [' '.join(row) for row in predicted_fertilizer_names]

output_df.head()


output_df.to_csv('XGBoost Optimal Fertilizer.csv', index=False)
print("\nThe prediction results have been saved")

