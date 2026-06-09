import pandas as pd
from sklearn.preprocessing import LabelEncoder,PowerTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


train=pd.read_csv("/kaggle/input/playground-series-s4e2/train.csv")
test=pd.read_csv("/kaggle/input/playground-series-s4e2/test.csv")


train.head()


test.head()


train.info()


test.info()


train.describe()


test.describe()


train=train.drop_duplicates()


sns.set(style="whitegrid")
numerical_columns = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_columns = train.select_dtypes(include=['object']).columns.tolist()
num_cols = len(numerical_columns)
num_rows = -(-num_cols // 4) 

plt.figure(figsize=(20, 6 * num_rows))
for i, col in enumerate(numerical_columns, 1):
    plt.subplot(num_rows, 4, i)
    sns.histplot(train[col], kde=True)
    plt.title(f'{col} Distribution')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


num_cols = len(categorical_columns)
num_rows = -(-num_cols // 4)

plt.figure(figsize=(20,6 *num_rows))
for i ,col in enumerate(categorical_columns,1):
    plt.subplot(num_rows, 4, i)
    sns.countplot(x= col, data=train)
    plt.title(f'{col} Distribution')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


columns=['Gender','family_history_with_overweight','FAVC','SMOKE','SCC']
train=pd.get_dummies(train,columns=columns,drop_first=True,dtype=int)
test=pd.get_dummies(test,columns=columns,drop_first=True,dtype=int)


le=LabelEncoder()
a=['CAEC','CALC','MTRANS']
for i in a:
    train[i]=le.fit_transform(train[i])
    test[i]=le.fit_transform(test[i])

train['NObeyesdad']=le.fit_transform(train['NObeyesdad'])


train = train.drop(columns=['id'])
test_ids = test['id']  # 保存 id 列
test = test.drop(columns=['id'])  # 然后再删


pt = PowerTransformer(method='yeo-johnson')
train_features = train.drop('NObeyesdad', axis=1)
train_features_scaled = pt.fit_transform(train_features)

train_scaled = pd.DataFrame(train_features_scaled, columns=train_features.columns)
train_scaled['NObeyesdad'] = train['NObeyesdad']


plt.figure(figsize=(12, 8))
sns.heatmap(train_scaled.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


for i in train.columns:
    sns.boxplot(train[i])
    plt.title(i)
    plt.show()


for i in test.columns:
    sns.boxplot(test[i])
    plt.title(i)
    plt.show()


def remove_outliers(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
    return data

train_scaled = remove_outliers(train_scaled)


# 去除最不重要的两个特征
b=['SCC_yes', 'SMOKE_yes']
for i in b:
    train_scaled=train_scaled.drop(i,axis=1)



X = train_scaled.drop('NObeyesdad', axis=1)
y = train_scaled['NObeyesdad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


import numpy as np
import pandas as pd
from collections import Counter

# --- 树节点类 ---
class TreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

# --- Gini计算 ---
def gini(y):
    m = len(y)
    if m == 0:
        return 0
    counts = np.bincount(y)
    probabilities = counts / m
    return 1 - np.sum(probabilities ** 2)

# --- 手写决策树 ---
class ManualDecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None

    def fit(self, X, y):
        self.n_classes_ = len(np.unique(y))
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        if (depth >= self.max_depth or num_labels == 1 or
                num_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return TreeNode(value=leaf_value)

        best_feature, best_threshold = self._best_split(X, y, num_features)
        if best_feature is None:
            return TreeNode(value=self._most_common_label(y))

        left_idx = X[:, best_feature] <= best_threshold
        right_idx = X[:, best_feature] > best_threshold

        if (left_idx.sum() < self.min_samples_leaf or right_idx.sum() < self.min_samples_leaf):
            return TreeNode(value=self._most_common_label(y))

        left = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        return TreeNode(feature_index=best_feature, threshold=best_threshold, left=left, right=right)

    def _best_split(self, X, y, num_features):
        best_gain = -1
        split_idx, split_threshold = None, None
        for feature_index in range(num_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_idx = X[:, feature_index] <= threshold
                right_idx = X[:, feature_index] > threshold
                if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
                    continue

                gain = self._information_gain(y, y[left_idx], y[right_idx])
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_index
                    split_threshold = threshold
        return split_idx, split_threshold

    def _information_gain(self, y, y_left, y_right):
        parent_gini = gini(y)
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)
        child_gini = (n_left / n) * gini(y_left) + (n_right / n) * gini(y_right)
        return parent_gini - child_gini

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)


from tqdm import tqdm  # 导入tqdm库用于进度条

class ManualRandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features='sqrt', bootstrap=True, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees = []
        self.features_indices = []

    def _get_bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        rng = np.random.RandomState(self.random_state)
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def _get_max_features(self, n_features):
        if self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2':
            return max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)  # 确保 max_feats 不大于 n_features
        elif isinstance(self.max_features, float):
            return min(int(n_features * self.max_features), n_features)  # 确保 max_feats 不大于 n_features
        else:
            return n_features

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        np.random.seed(self.random_state)
        n_features = X.shape[1]

        # 添加进度条
        for i in tqdm(range(self.n_estimators), desc="Fitting trees", unit="tree"):
            tree = ManualDecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )

            max_feats = self._get_max_features(n_features)
            feature_indices = np.random.choice(n_features, max_feats, replace=False)
            self.features_indices.append(feature_indices)

            if self.bootstrap:
                X_sample, y_sample = self._get_bootstrap_sample(X[:, feature_indices], y)
            else:
                X_sample, y_sample = X[:, feature_indices], y

            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values

        predictions = []
        for i, tree in enumerate(self.trees):
            feature_idx = self.features_indices[i]
            pred = tree.predict(X[:, feature_idx])
            predictions.append(pred)

        predictions = np.array(predictions).T
        final_preds = [Counter(row).most_common(1)[0][0] for row in predictions]
        return np.array(final_preds)

# 假设你已经有 X_train, y_train 和 X_test, y_test 数据
rf = ManualRandomForestClassifier(
    n_estimators=1000,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=16,  # 注意这里是 16，不是 '16'
    bootstrap=True,
    random_state=42
)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print('准确率:', accuracy_score(y_test, y_pred))
print('分类报告:\n', classification_report(y_test, y_pred))
print('混淆矩阵:\n', confusion_matrix(y_test, y_pred))

# 保存模型结果到文件
with open('model_results.txt', 'w', encoding='utf-8') as f:
    f.write(f"准确率: {accuracy_score(y_test, y_pred):.4f}\n\n")
    f.write("分类报告:\n")
    f.write(classification_report(y_test, y_pred))
    f.write("\n混淆矩阵:\n")
    f.write(np.array2string(confusion_matrix(y_test, y_pred)))


y_pred_test = rf.predict(test)
submission = pd.DataFrame({'id': test_ids, 'NObeyesdad': y_pred_test})
submission['NObeyesdad'] = le.inverse_transform(submission['NObeyesdad'])
submission.to_csv('submission.csv', index=False)

