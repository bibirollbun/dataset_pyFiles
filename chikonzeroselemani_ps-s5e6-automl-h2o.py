import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from h2o.estimators import H2OXGBoostEstimator
import h2o
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings("ignore")

h2o.init()



# Load datasets
train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv", index_col="id")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv", index_col="id")
external = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")

# Combine datasets
overall_train = pd.concat([train, external], ignore_index=True)
overall_train = overall_train.drop_duplicates().reset_index(drop=True)
overall_train.head(5)


# Convert to H2O frames
trainH2o = h2o.H2OFrame(overall_train)
testH2o = h2o.H2OFrame(test)

# Encode target variable
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(overall_train['Fertilizer Name']).tolist()
targetH2o = h2o.H2OFrame(y_encoded)
trainH2o['Fertilizer Name'] = targetH2o
trainH2o['Fertilizer Name'] = trainH2o['Fertilizer Name'].asfactor()

# Handle categorical features
categorical_features = test.select_dtypes(include="object").columns.tolist()
for col in categorical_features:
    trainH2o[col] = trainH2o[col].asfactor()
    testH2o[col] = testH2o[col].asfactor()


# Define model parameters
x_columns = [col for col in overall_train.columns if col != 'Fertilizer Name']
y_column = "Fertilizer Name"

# Split data
train_split, valid_split = trainH2o.split_frame(ratios=[0.9], seed=42)

# Initialize model
xgb_model = H2OXGBoostEstimator(
   stopping_rounds=20,
    stopping_metric="logloss",
    stopping_tolerance=0.001,
    score_tree_interval=10,
    seed=42,
    keep_cross_validation_predictions=True,
    booster='gbtree',
    ntrees=100,                          
    learn_rate=0.1,  
    gpu_id = 0, 
    max_depth=15,
    min_rows=10,                            
    sample_rate=0.8276149323901826,        
    col_sample_rate=0.2587327850345624,    
    distribution='multinomial',           
    categorical_encoding='auto',           
    quiet_mode=True,
    backend='gpu',
    tree_method="hist",   
    reg_lambda = 0.05656209749983576,  
    reg_alpha = 5.620898657099113

    
)



# Train model
xgb_model.train(
    x=x_columns,
    y=y_column,
    training_frame=train_split,
    validation_frame=valid_split
)




# Get model performance
perf = xgb_model.model_performance(valid_split)

# Plot training history 
scoring_history = xgb_model.scoring_history()
plt.figure(figsize=(12, 10))

# Logloss plot
plt.subplot(2, 1, 1)
plt.plot(scoring_history['number_of_trees'], scoring_history['training_logloss'], 
         label='Training', linewidth=2)
plt.plot(scoring_history['number_of_trees'], scoring_history['validation_logloss'], 
         label='Validation', linewidth=2, linestyle='--')
plt.title('Training History: Logloss', fontsize=14)
plt.xlabel('Number of Trees', fontsize=12)
plt.ylabel('Logloss', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.2)
plt.axvline(x=scoring_history['number_of_trees'].iloc[-1], 
            color='r', linestyle=':', alpha=0.7)


 #MAP@3 Evaluation Function
def map_at_3(y_true, y_pred_proba, k=3):
    top_k = np.argsort(-y_pred_proba, axis=1)[:, :k]
    ranks = np.argwhere(top_k == y_true.reshape(-1, 1))[:, 1] + 1
    map_score = np.sum(1 / ranks) / len(y_true)
    return map_score

# Calculate MAP@3
print("\nEvaluating model performance...")
val_preds = xgb_model.predict(valid_split)
val_preds_df = val_preds.as_data_frame()
n_classes = len(target_encoder.classes_)
prob_cols = [f"p{i}" for i in range(n_classes)]
y_pred_proba = val_preds_df[prob_cols].values
y_true = valid_split[y_column].as_data_frame().values.flatten()

validation_map3 = map_at_3(y_true, y_pred_proba)
logloss = xgb_model.logloss(valid=True)

print(f"Validation MAP@3: {validation_map3:.6f}")
print(f"Validation Logloss: {logloss:.5f}")


 #Generate predictions
print("\nGenerating test predictions...")
test_preds = xgb_model.predict(testH2o)
test_preds_df = test_preds.as_data_frame()

# Prepare submission
print("Creating submission file...")
top3 = test_preds_df[prob_cols].apply(
    lambda row: row.nlargest(3).index.tolist(), 
    axis=1
)

label_map = {i: label for i, label in enumerate(target_encoder.classes_)}
top3_labels = top3.apply(lambda x: [label_map[int(col.split('p')[1])] for col in x])

submission = pd.DataFrame({
    "id": test.index,
    "Fertilizer Name": top3_labels.apply(lambda x: " ".join(x))
})

# Display submission preview
print("\nSubmission preview:")
print(submission.head())

# Save results
submission.to_csv("submission.csv", index=False)
print("\nSubmission saved successfully")

