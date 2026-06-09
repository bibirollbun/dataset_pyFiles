import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow as pa
import xgboost as xgb
import torch
import gc
import numpy as np


def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

clear_gpu_memory()


parquet_file = "/kaggle/input/traintestparquet/train_test.parquet"
table = pq.read_table(parquet_file)


products = [
    "ind_ahor_fin_ult1", "ind_aval_fin_ult1", "ind_cco_fin_ult1",
    "ind_cder_fin_ult1", "ind_cno_fin_ult1", "ind_ctju_fin_ult1",
    "ind_ctma_fin_ult1", "ind_ctop_fin_ult1", "ind_ctpp_fin_ult1",
    "ind_deco_fin_ult1", "ind_deme_fin_ult1", "ind_dela_fin_ult1",
    "ind_ecue_fin_ult1", "ind_fond_fin_ult1", "ind_hip_fin_ult1",
    "ind_plan_fin_ult1", "ind_pres_fin_ult1", "ind_reca_fin_ult1",
    "ind_tjcr_fin_ult1", "ind_valo_fin_ult1", "ind_viv_fin_ult1",
    "ind_nomina_ult1", "ind_nom_pens_ult1", "ind_recibo_ult1"
]

lag_months = [1, 2, 3, 4, 5, 6]
lagged_features = [f"lag_{lag}_{prod}" for lag in lag_months for prod in products]


# ------------------------------------------------------------------------
# 3. Chia train, val, test

train_filter = pc.less(table["month_int"], 15)
val_filter = pc.equal(table["month_int"], 15)
test_filter  = pc.equal(table["month_int"], 16)

train_table = table.filter(train_filter)
val_table  = table.filter(val_filter)
test_table  = table.filter(test_filter)


# ------------------------------------------------------------------------
# 4. Prepare training data

training_features_list = []
training_labels_list = []

for product_index, product in enumerate(products):
    current_month_indicators = train_table.column(product).to_numpy()
    previous_month_indicators = train_table.column(f"lag_1_{product}").to_numpy()

    newly_added_product_mask = (current_month_indicators == 1) & (previous_month_indicators == 0)
    if np.sum(newly_added_product_mask) == 0:
        continue

    lagged_feature_arrays = []
    for feature in lagged_features:
        feature_column = train_table.column(feature).to_numpy()
        lagged_feature_arrays.append(feature_column[newly_added_product_mask].reshape(-1, 1))

    feature_matrix = np.hstack(lagged_feature_arrays)
    training_features_list.append(feature_matrix)

    training_labels_list.append(np.full(feature_matrix.shape[0], product_index, dtype=np.int8))

if len(training_features_list) == 0:
    raise ValueError("No training data found. Check the data and filters.")

training_features = np.vstack(training_features_list)
training_labels = np.concatenate(training_labels_list)

print("Training samples:", training_features.shape[0])
print("Number of features:", training_features.shape[1])
print("Training labels distribution:", {i: int(np.sum(training_labels==i)) for i in np.unique(training_labels)})



# ------------------------------------------------------------------------
# 4.5. Prepare validation data

validation_features_list = []
validation_labels_list = []

for product_index, product in enumerate(products):
    current_month_indicators = val_table.column(product).to_numpy()
    previous_month_indicators = val_table.column(f"lag_1_{product}").to_numpy()

    newly_added_product_mask = (current_month_indicators == 1) & (previous_month_indicators == 0)
    if np.sum(newly_added_product_mask) == 0:
        continue

    lagged_feature_arrays = []
    for feature in lagged_features:
        feature_column = val_table.column(feature).to_numpy()
        lagged_feature_arrays.append(feature_column[newly_added_product_mask].reshape(-1, 1))

    feature_matrix = np.hstack(lagged_feature_arrays)
    validation_features_list.append(feature_matrix)

    validation_labels_list.append(np.full(feature_matrix.shape[0], product_index, dtype=np.int8))

if len(validation_features_list) == 0:
    print("Warning: No validation data found for new product additions. Validation set might be empty.")
    validation_features = np.empty((0, len(lagged_features)))
    validation_labels = np.empty(0, dtype=np.int8)
else:
    validation_features = np.vstack(validation_features_list)
    validation_labels = np.concatenate(validation_labels_list)

print("Validation samples:", validation_features.shape[0])
print("Number of features:", validation_features.shape[1])
print("Validation labels distribution:", {i: int(np.sum(validation_labels==i)) for i in np.unique(validation_labels)})


# ------------------------------------------------------------------------
# 5. Prepare test data

test_feature_arrays = []
for feature in lagged_features:
    arr = test_table.column(feature).to_numpy()
    test_feature_arrays.append(arr.reshape(-1, 1))
test_features = np.hstack(test_feature_arrays)

ncodpers = test_table.column("ncodpers").to_pylist()

print("Test samples:", test_features.shape[0])
print("Test feature shape:", test_features.shape)


# %%
# ------------------------------------------------------------------------
# 6. Build XGBoost DMatrix objects

dtrain = xgb.DMatrix(training_features, label=training_labels, feature_names=lagged_features)
dval = xgb.DMatrix(validation_features, label=validation_labels, feature_names=lagged_features) # Created dval
dtest  = xgb.DMatrix(test_features, feature_names=lagged_features)

# ------------------------------------------------------------------------
# 7. Set XGBoost parameters (multi:softprob for multiclass)
param = {
    "objective": "multi:softprob",
    "eta": 0.1,
    "min_child_weight": 10,
    "max_depth": 8,
    "silent": 1,
    "eval_metric": "mlogloss",
    "colsample_bytree": 0.8,
    "colsample_bylevel": 0.9,
    "num_class": len(products),
    "device": "cuda"
}


# ------------------------------------------------------------------------
# 8. Train the XGBoost model

num_boost_round = 1000
watchlist = [(dtrain, "train"), (dval, "eval")]

print("Training XGBoost model ...")
model = xgb.train(param, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=20)

print("\nFeature Importance:")
feature_importance = model.get_fscore()  
for feat, score in sorted(feature_importance.items(), key=lambda item: item[1], reverse=True):
    print(f"{feat}: {score}")


# -------------------------------------------------------------
# 9. Predict on test data (fixing the AttributeError)

preds = model.predict(dtest)
print("\nPredictions shape:", preds.shape)


# -------------------------------------------------------------
# 10. Define submission function

import csv
import io

def make_submission_file(submission_file, predictions, customer_ids, products):
    writer = csv.writer(submission_file)
    writer.writerow(["ncodpers", "added_products"])
    for customer_id, prediction in zip(customer_ids, predictions):
        predicted_products_indices = np.argsort(prediction)[::-1][:3]
        predicted_product_names = [products[i] for i in predicted_products_indices]
        writer.writerow([int(customer_id), " ".join(predicted_product_names)])




# -------------------------------------------------------------
# 11. Define MAP@3 metric function

def apk(actual, predicted, k=3, default=0.0):
    """
    Calculate the average precision at k (AP@k) for a single instance.
    
    :param actual: List of actual products.
    :param predicted: List of predicted products.
    :param k: The number of predictions to consider.
    :param default: The default value to return if there are no actual products.
    :return: The average precision at k.
    """
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    if not actual:
        return default
    return score / min(len(actual), k)

def mapk(actual, predicted, k=3, default=0.0):
    """
    Calculate the mean average precision at k (MAP@k) for a set of instances.
    
    :param actual: List of lists of actual products.
    :param predicted: List of lists of predicted products.
    :param k: The number of predictions to consider.
    :param default: The default value to return if there are no actual products.
    :return: The mean average precision at k.
    """
    return np.mean([apk(a, p, k, default) for a, p in zip(actual, predicted)])



# -------------------------------------------------------------
# 12. Calculate MAP@3 on the test set

# Prepare actual and predicted products for the test set
actual_products_list = []
predicted_products_list = []

# For each customer in the test set
for i in range(len(ncodpers)):
    # Get the actual products added by the customer
    actual_products = [product for product in products if test_table.column(product).to_numpy()[i] == 1]
    actual_products_list.append(actual_products)

    # Get the top 3 predicted products for the customer
    predicted_indices = np.argsort(preds[i])[::-1][:3]  # Top 3 predictions
    predicted_products = [products[idx] for idx in predicted_indices]
    predicted_products_list.append(predicted_products)

# Calculate MAP@3
map3_score = mapk(actual_products_list, predicted_products_list, k=3)
print(f"MAP@3 on test set: {map3_score:.4f}")


# Assuming you already have the following variables:
# - preds: The predictions from the model (a 2D numpy array of shape [n_samples, n_products])
# - ncodpers: A list of customer IDs (e.g., from test_table.column("ncodpers").to_pylist())
# - products: A list of product names (e.g., the `products` list defined earlier)

# Create a StringIO buffer to hold the submission data
submission_io_buffer = io.StringIO()

# Call the make_submission_file function to write the submission data to the buffer
make_submission_file(submission_io_buffer, preds, ncodpers, products)

# Write the content of the buffer to submission.txt
txt_filename = "submission_MAP3_2nd.txt"
with open(txt_filename, 'w') as txt_file:
    txt_file.write(submission_io_buffer.getvalue())

print(f"Submission written to: {txt_filename}")

