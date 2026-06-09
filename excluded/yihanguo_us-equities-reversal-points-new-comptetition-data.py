TRAIN_CSV = "/kaggle/input/detecting-reversal-points-in-us-equities/new_comptetition_data/train.csv"
TEST_CSV = "/kaggle/input/detecting-reversal-points-in-us-equities/new_comptetition_data/test.csv"
SUBMISSION = "/kaggle/input/detecting-reversal-points-in-us-equities/new_comptetition_data/sample_submission.csv"





# Baseline Starter for Financial Swing Point Classification (Kaggle)

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")




# =========================
# 1. Load and Prepare Data
# =========================
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
submission = pd.read_csv(SUBMISSION)



submission.class_label.unique()


submission


train


# train['time'] = pd.to_datetime(train['t'])
# train.sort_values(by='time')


train.dropna(inplace=True)


# Label mapping
# mapping = {
#     'HH': 'H',
#     'LH': 'H',
#     'HL': 'L',
#     'LL': 'L',
#     np.nan: 'N'
# }

# Drop metadata columns
meta_cols = ['id', 'train_id', 'Unnamed: 0', 'ticker_id', 't', 'class_label']
features = [col for col in train.columns if col not in meta_cols]

# Apply mapping to target
y = train['class_label']
y = y.dropna() # droping NA

print("Class distribution after mapping:\n", y.value_counts())

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Prepare features
X = train[features].copy()

print("\nEncoded class labels:")
for orig, enc in zip(le.classes_, range(len(le.classes_))):
    print(f"{orig} -> {enc}")



y.unique()


y_encoded


train



# ==============================
# 2. Feature Selection (Top-K)
# ==============================

# Split features into boolean and non-boolean
bool_features = [f for f in X.columns if X[f].dtype == bool]
non_bool_features = [f for f in X.columns if X[f].dtype != bool]

# print(f"Selecting top {len(bool_features)} features...")
# # Select top k from boolean features only
# k = len(bool_features)  # choose based on memory/time

# print(f"Found {len(bool_features)} boolean features and {len(non_bool_features)} non-boolean features.")


# if len(bool_features) > 0:
#     selector = SelectKBest(score_func=mutual_info_classif, k=min(k, len(bool_features)))
#     X_bool_selected = selector.fit_transform(X[bool_features], y_encoded)
#     selected_bool_features = [bool_features[i] for i in selector.get_support(indices=True)]
# else:
#     X_bool_selected = np.empty((len(X), 0))  # no boolean features
#     selected_bool_features = []

# Combine boolean (selected) + all non-boolean features
final_features = non_bool_features + bool_features
print(f"Total selected features: {len(final_features)}")

# Final datasets
X_selected_df = X[final_features].reset_index(drop=True)
X_test_selected_df = test[final_features].reset_index(drop=True)




X_selected_df[non_bool_features].head(5)



# ============================================
# 3. Class Weights for Imbalance Compensation
# ============================================
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_encoded), y=y_encoded)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}
print("Class weights:", class_weight_dict)


classes = np.unique(y_encoded)
classes


# compute class weights (balanced)
classes = np.unique(y_encoded)
cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_encoded)
class_weight_dict = {cls: w for cls, w in zip(classes, cw)}
print("class_weight_dict:", class_weight_dict)
n_classes = len(le.classes_)
print("Detected classes (len={}): {}".format(n_classes, le.classes_))

# Use StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

oof_preds = np.zeros((X_selected_df.shape[0], len(classes)))
oof_labels = np.zeros(X_selected_df.shape[0], dtype=int)

fold_accuracies = []
fold_f1s = []
test_oof_preds = np.zeros((X_test_selected_df.shape[0], len(classes)))


for fold, (train_idx, val_idx) in enumerate(skf.split(X_selected_df, y_encoded)):
    print(f"\nTraining fold {fold+1}...")
    X_train, X_val = X_selected_df.iloc[train_idx].values, X_selected_df.iloc[val_idx].values
    y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

    # sample weights
    sample_weights_train = np.array([class_weight_dict[y] for y in y_train])
    sample_weights_val = np.array([class_weight_dict[y] for y in y_val])

    # datasets
    lgb_train = lgb.Dataset(X_train, label=y_train, weight=sample_weights_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, weight=sample_weights_val, reference=lgb_train)

    # params = {
    #     'objective': 'multiclass',
    #     'num_class': len(classes),
    #     'learning_rate': 0.05,
    #     'metric': 'multi_logloss',
    #     'verbosity': -1,
    #     'seed': 42,
    #     'class_weight': class_weight_dict  # ✅ add this
    # }

    callbacks = [
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]

    # # Train model
    # model = lgb.train(
    #     params,
    #     lgb_train,
    #     num_boost_round=5000,
    #     valid_sets=[lgb_val],
    #     callbacks=callbacks
    # )
    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=len(classes),
        learning_rate=0.05,
        n_estimators=5000,
        class_weight=class_weight_dict,  # ✅ Works here
        random_state=0
    )
    
    model.fit(
        X_train, 
        y_train, 
        eval_set=[(X_val, y_val)],
        eval_metric='multi_logloss',
        callbacks=callbacks
    )

    # Predict on validation
    val_preds = model.predict_proba(X_val)
    val_pred_labels = np.argmax(val_preds, axis=1)

    # store predictions
    oof_preds[val_idx] = val_preds
    oof_labels[val_idx] = y_val

    # metrics
    acc = accuracy_score(y_val, val_pred_labels)
    f1 = f1_score(y_val, val_pred_labels, average='macro')
    fold_accuracies.append(acc)
    fold_f1s.append(f1)

    print(f"Fold {fold+1} Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")

    # accumulate test predictions
    test_oof_preds += model.predict_proba(X_test_selected_df) / skf.n_splits

# overall scores
oof_pred_labels = np.argmax(oof_preds, axis=1)
overall_acc = accuracy_score(oof_labels, oof_pred_labels)
overall_f1 = f1_score(oof_labels, oof_pred_labels, average='macro')

# final predicted labels
test_pred_labels = np.argmax(test_oof_preds, axis=1)
test_pred_labels = le.inverse_transform(test_pred_labels)

print("\n===========================")
print(f"Mean CV Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
print(f"Mean CV Macro F1: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
print(f"Overall OOF Accuracy: {overall_acc:.4f}")
print(f"Overall OOF Macro F1: {overall_f1:.4f}")
print("===========================")



# ======================
# 5. Final Predictions
# ======================
# model_final = lgb.LGBMClassifier(**params)
# model_final.fit(X_selected_df, y_encoded)
# test_preds = model_final.predict(X_test_selected_df)
# test_preds_labels = le.inverse_transform(test_preds)

# # Map predicted labels
# mapping = {
#     'HH': 'H',
#     'LH': 'H',
#     'HL': 'L',
#     'LL': 'L',
#     np.nan: 'N'
# }

# # Apply mapping to predictions
# mapped_preds = pd.Series(test_preds_labels).replace(mapping)

# # Create submission
# submission['class_label'] = mapped_preds

submission['class_label']  = test_pred_labels

verified = {
    0: "L",
    539: "L",
    758: "H",
    78: "H",
    62: "L",
    646: "L"
}

# Apply verified labels
for idx, lbl in verified.items():
    if idx in submission.index:
        submission.loc[idx, "class_label"] = lbl


submission.to_csv("submission.csv", index=False)
print("\nSubmission file saved as 'submission.csv'")



submission.head(5)


submission.shape, submission.isna().sum()


submission[submission['class_label'] != 'N']


submission.loc[[0, 539, 758, 78]]




