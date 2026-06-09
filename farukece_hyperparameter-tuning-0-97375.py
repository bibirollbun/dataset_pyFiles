from IPython.display import Image
Image(url='https://upload.wikimedia.org/wikipedia/commons/7/7b/Wassily_Kandinsky%2C_1903%2C_The_Blue_Rider_%28Der_Blaue_Reiter%29%2C_oil_on_canvas%2C_52.1_x_54.6_cm%2C_Stiftung_Sammlung_E.G._B%C3%BChrle%2C_Zurich.jpg')


import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier




train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv", index_col='id')
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv", index_col='id')


orig = pd.read_csv("/kaggle/input/bank-marketing-dataset-full/bank-full.csv", sep=';')
orig['y'] = orig['y'].map({'no': 0, 'yes': 1})


train = pd.concat([train, orig], ignore_index=True)
train = train.drop_duplicates()


X = train.drop('y', axis=1).astype('str')
y = train['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, stratify=y, random_state=0)

# here you need to add and adjust
# selected_features = ['balance', 'duration', 'age', 'day', 'pdays', 'campaign', 'previous']
# X_train = X_train[selected_features]
# X_test = X_test[selected_features] 


# Define and configure the CatBoostClassifier
cat_clf = CatBoostClassifier(
    cat_features=X.columns.to_list(),   # categorical features
    task_type='CPU',                   # use CPU (change to 'GPU' if available)
    devices='1',                       # specific device if needed
    n_estimators=3000,                 # total boosting iterations
    learning_rate=0.068,               # step size shrinkage
    depth=12,                          # depth of trees
    l2_leaf_reg=2,                     # L2 regularization
    subsample=0.8,                     # subsampling ratio for bagging
    allow_writing_files=True,          # allow writing files
    save_snapshot=True,                # snapshot to resume training if interrupted
    snapshot_file='catboost_checkpoint.cb',
    snapshot_interval=600,
    verbose=100                        # log training every 100 iterations
)

# Train the model with early stopping
cat_clf.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    early_stopping_rounds=300
)

# Save the trained model
cat_clf.save_model("catboost_final_model.cbm")

# Reload model (for demonstration / future use)
loaded_model = CatBoostClassifier()
loaded_model.load_model("catboost_final_model.cbm")

# Predict probabilities
y_pred_proba = loaded_model.predict_proba(X_test)[:, 1]

# Evaluate with ROC-AUC
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC Score: {roc_auc:.4f}")


test_pred = cat_clf.predict_proba(test.astype('str'))[:, 1]


sub = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")
sub['y'] = test_pred
sub.to_csv("submission_2000e1.csv", index=False)
sub.head()

