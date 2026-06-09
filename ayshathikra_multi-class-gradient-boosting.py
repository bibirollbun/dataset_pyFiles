import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import joblib
import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


print("Dataset shape:", train_df.shape)
print("\nFirst 5 rows of the dataset")
print("------------------------------")
print(train_df.head())

print("\n\nData types")
print("-------------")
print(train_df.dtypes)

print("\n\nMissing values")
print("----------------")
print(train_df.isnull().sum())


print("\nSummary statistics")
print("---------------------")
print(train_df.describe())


def label_encode(series):
    unique = sorted(series.unique())
    mapping = {v: i for i, v in enumerate(unique)}
    return series.map(mapping), mapping, unique

train_df['target'], label_map, label_classes = label_encode(train_df['Fertilizer Name'])

for df in [train_df, test_df]:
    df['Crop_Code'], _, _ = label_encode(df['Crop Type'])
    df['Soil_Code'], _, _ = label_encode(df['Soil Type'])

features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen',
            'Potassium', 'Phosphorous', 'Crop_Code', 'Soil_Code']

X = train_df[features].values
y = train_df['target'].values
X_test = test_df[features].values


def softmax(logits):
    e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        if a in p[:k]:
            return 1.0 / (p[:k].index(a) + 1)
        return 0.0
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def stratified_kfold(X, y, n_splits=5, random_state=42):
    np.random.seed(random_state)
    folds = [[] for _ in range(n_splits)]
    class_indices = defaultdict(list)
    
    for idx, label in enumerate(y):
        class_indices[label].append(idx)
    
    for cls in class_indices:
        idxs = class_indices[cls]
        np.random.shuffle(idxs)
        for i, fold_idxs in enumerate(np.array_split(idxs, n_splits)):
            folds[i].extend(fold_idxs)
    
    return [np.array(fold) for fold in folds]


class ScratchDecisionTreeRegressor:
    def __init__(self, max_depth=3, min_samples_split=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if depth >= self.max_depth or len(X) < self.min_samples_split:
            return np.mean(y)

        best_feature, best_thresh, best_loss = None, None, float('inf')
        split_thresholds = 10

        for feature in range(X.shape[1]):
            thresholds = np.percentile(X[:, feature], np.linspace(0, 100, split_thresholds))
            for threshold in thresholds:
                left_idx = X[:, feature] <= threshold
                right_idx = ~left_idx
                if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
                    continue
                loss = np.var(y[left_idx]) * len(y[left_idx]) + np.var(y[right_idx]) * len(y[right_idx])
                if loss < best_loss:
                    best_feature = feature
                    best_thresh = threshold
                    best_loss = loss

        if best_feature is None:
            return np.mean(y)

        left_idx = X[:, best_feature] <= best_thresh
        right_idx = ~left_idx
        left = self._build_tree(X[left_idx], y[left_idx], depth+1)
        right = self._build_tree(X[right_idx], y[right_idx], depth+1)

        return (best_feature, best_thresh, left, right)

    def predict(self, X):
        return np.array([self._predict_one(row, self.tree) for row in X])

    def _predict_one(self, x, node):
        if not isinstance(node, tuple):
            return node
        feature, threshold, left, right = node
        return self._predict_one(x, left if x[feature] <= threshold else right)


class ScratchXGBoostClassifier:
    def __init__(self, num_classes, n_estimators=30, max_depth=3, learning_rate=0.1):
        self.num_classes = num_classes
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.lr = learning_rate
        self.trees = [[] for _ in range(num_classes)]

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        n_samples = X_train.shape[0]
        base_score = np.zeros((n_samples, self.num_classes))

        for est in range(self.n_estimators):
            probs = softmax(base_score)
            for k in range(self.num_classes):
                g = probs[:, k] - (y_train == k).astype(float)
                tree = ScratchDecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=10)
                tree.fit(X_train, -g)
                base_score[:, k] += self.lr * tree.predict(X_train)
                self.trees[k].append(tree)

        if X_val is not None and y_val is not None:
            val_score = self.score(X_val, y_val)
            print(f"Validation MAP@3: {val_score:.5f}")

    def predict_proba(self, X):
        raw_scores = np.zeros((X.shape[0], self.num_classes))
        for k in range(self.num_classes):
            for tree in self.trees[k]:
                raw_scores[:, k] += self.lr * tree.predict(X)
        return softmax(raw_scores)

    def predict_topk(self, X, k=3):
        proba = self.predict_proba(X)
        return np.argsort(proba, axis=1)[:, -k:][:, ::-1]

    def score(self, X, y_true, k=3):
        top_preds = self.predict_topk(X, k)
        return mapk(y_true.tolist(), top_preds.tolist(), k)


n_splits = 5
folds = stratified_kfold(X, y, n_splits=n_splits)

oof_preds = np.zeros((len(train_df), len(label_classes)))
test_preds = np.zeros((len(test_df), len(label_classes)))
fold_scores = []

for fold_i in range(n_splits):
    print(f"\nTraining fold {fold_i+1}...")
    val_idx = folds[fold_i]
    train_idx = np.concatenate([folds[j] for j in range(n_splits) if j != fold_i])

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    model = ScratchXGBoostClassifier(num_classes=len(label_classes), n_estimators=20, max_depth=3, learning_rate=0.1)
    model.fit(X_train, y_train, X_val, y_val)

    val_proba = model.predict_proba(X_val)
    oof_preds[val_idx] = val_proba

    fold_score = mapk(y_val.tolist(), np.argsort(val_proba, axis=1)[:, -3:][:, ::-1].tolist(), k=3)
    fold_scores.append(fold_score)

    test_preds += model.predict_proba(X_test) / n_splits
    print(f"Fold {fold_i+1} MAP@3: {fold_score:.5f}")

best_fold = np.argmax(fold_scores)
print(f"\n\nBest fold: {best_fold + 1} with MAP@3 = {fold_scores[best_fold]:.5f}")
print(f"\nAverage MAP@3: {np.mean(fold_scores):.5f}")


plt.figure(figsize=(8, 4))
plt.plot(range(1, n_splits + 1), fold_scores, marker='o', label='Fold MAP@3')
plt.axhline(np.mean(fold_scores), linestyle='--', color='gray', label='Average')
plt.title('MAP@3 Score Per Fold (Scratch Model)')
plt.xlabel('Fold')
plt.ylabel('MAP@3')
plt.legend()
plt.grid(True)
plt.show()


y_bin = label_binarize(y, classes=range(len(label_classes)))

plt.figure(figsize=(10, 8))

for i in range(len(label_classes)):
    fpr, tpr, _ = roc_curve(y_bin[:, i], oof_preds[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{label_classes[i]} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Each Class (OOF Predictions)")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()



top3_test_preds = np.argsort(test_preds, axis=1)[:, -3:][:, ::-1]
inv_label_map = {v: k for k, v in label_map.items()}
top3_labels = np.vectorize(inv_label_map.get)(top3_test_preds)

submission = pd.DataFrame({
    'id': test_df['id'],
    'Fertilizer Name': [' '.join(row) for row in top3_labels]
})
submission.to_csv('submission.csv', index=False)
print("✅ Submission file created successfully: submission.csv")


sub_check = submission.copy()
sub_check['Fertilizer Count'] = sub_check['Fertilizer Name'].apply(lambda x: len(set(x.split())))
repeated = sub_check[sub_check['Fertilizer Count'] < 3]
print(f"Rows with duplicate fertilizers: {len(repeated)}")


print("First 5 rows of submission:")
print(pd.read_csv('submission.csv').head())


print("Total rows:", len(submission))
print("Unique IDs:", submission['id'].nunique())


print("Random sample from submission:")
print(pd.read_csv('submission.csv').sample(5))


joblib.dump(model, 'fertilizer_xgb_model.pkl')
print("✅ Model saved successfully using joblib as 'fertilizer_xgb_model.pkl'")

