import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.base import clone


TRAIN_PATH = "/kaggle/input/playground-series-s5e7/train.csv"
TEST_PATH = "/kaggle/input/playground-series-s5e7/test.csv"
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
train_df.head()


X = train_df.drop(columns=["Personality", "id"])
y = train_df["Personality"]

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

preprocessor = ColumnTransformer([
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols),
    ("num", SimpleImputer(strategy="mean"), num_cols)
])


class PostProcessor:
    def __init__(self):
        self.dataset_path = "/kaggle/input/extrovert-vs-introvert-behavior-data"
        self.csv_filepath = f"{self.dataset_path}/personality_dataset.csv"
        self.reference_df = pd.read_csv(self.csv_filepath)
        self.feature_cols = self.reference_df.columns.drop("Personality").tolist()
        self.reference_df = self.reference_df.rename(columns={"Personality": "match_p"}).drop_duplicates(subset=self.feature_cols)

    def apply(self, X_to_fix, labels):
        
        match_df = X_to_fix.copy()
        match_df = match_df.merge(self.reference_df, on=self.feature_cols, how="left")

        mask = match_df["match_p"].notna()
        opposite = {"Extrovert": "Introvert", "Introvert": "Extrovert"}
        fixed_labels = pd.Series(labels, dtype="object").copy()
        fixed_labels[mask] = match_df.loc[mask, "match_p"].map(opposite)

        return fixed_labels.tolist()



post_processor = PostProcessor()


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=56)
models = []
oof_preds = np.zeros((len(X), len(y.unique())))

classes = np.sort(y.unique())
class_to_index = {c: i for i, c in enumerate(classes)}
index_to_class = {i: c for c, i in class_to_index.items()}
y_encoded = y.map(class_to_index)

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_encoded), 1):
    print(f"Training Fold {fold}...")

    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold = y_encoded.iloc[train_idx]

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=56))
    ])

    model.fit(X_train_fold, y_train_fold)
    models.append(model)

    oof_preds[val_idx] = model.predict_proba(X_val_fold)


oof_labels = np.argmax(oof_preds, axis=1)
oof_true = y_encoded.values

oof_accuracy = accuracy_score(oof_true, oof_labels)
print(f"\nOOF CV Accuracy: {oof_accuracy:.4f}")


oof_labels_str = pd.Series(oof_labels).map(index_to_class)
X_with_id = X.copy()
X_with_id["id"] = train_df["id"]

post_oof_labels = post_processor.apply(X_with_id, oof_labels_str)

post_oof_labels_int = pd.Series(post_oof_labels).map(class_to_index)
post_oof_accuracy = accuracy_score(oof_true, post_oof_labels_int)
print(f"OOF CV Accuracy After Post-Processing: {post_oof_accuracy:.4f}")



X_test = test_df.drop(columns=["id"])
test_probs = np.array([model.predict_proba(X_test) for model in models])
avg_test_probs = np.mean(test_probs, axis=0)

final_test_preds = np.argmax(avg_test_probs, axis=1)
final_test_labels = [index_to_class[i] for i in final_test_preds]


X_test_with_id = X_test.copy()
X_test_with_id["id"] = test_df["id"]

final_test_labels = post_processor.apply(X_test_with_id, final_test_labels)


submission = pd.DataFrame({
    "id": test_df["id"],
    "Personality": final_test_labels
})

submission.to_csv("submission.csv", index=False)
submission.head()

