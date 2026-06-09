import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from collections import Counter
from sklearn.model_selection import train_test_split
import time
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


print("\nUnique Crop Types")
print("-------------------")
print(train_df['Crop Type'].unique())

print("\n\nUnique Soil Types")
print("--------------------")
print(train_df['Soil Type'].unique())



fertilizer_names = sorted(train_df['Fertilizer Name'].unique())

print(" All the Unique Fertilizer Names")
print("---------------------------------")
for name in fertilizer_names:
    print(f"  - {name}")


le_fert = LabelEncoder()
le_fert.fit(train_df['Fertilizer Name'])

original_fertilizer_names = le_fert.classes_

print("Original Fertilizer Names")
print("----------------------------")
for name in original_fertilizer_names:
    print(f"- {name}")

print("\n\n----------------------------")
print(f"Total unique fertilizers: {len(original_fertilizer_names)}")


fert_counts = train_df['Fertilizer Name'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(fert_counts, labels=fert_counts.index, autopct='%1.1f%%', 
        startangle=140)
plt.title('Fertilizer Type Distribution (Pie Chart)\n\n')
plt.axis('equal')
plt.tight_layout()
plt.show()



numeric_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen',
                'Phosphorous', 'Potassium']

for col in numeric_cols:
    plt.figure()
    sns.histplot(train_df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()


for col in numeric_cols:
    plt.figure()
    sns.boxplot(x=train_df[col])
    plt.title(f'Boxplot of {col}')
    plt.tight_layout()
    plt.show()


sample_df = train_df.sample(3000, random_state=42)

sns.pairplot(sample_df, vars=['Nitrogen', 'Phosphorous', 'Potassium', 'Moisture'],
             hue='Fertilizer Name', palette='tab10', diag_kind='kde')
plt.suptitle('Pairwise Feature Distributions by Fertilizer', y=1.02)
plt.show()



plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=train_df, 
    x='Moisture', 
    y='Potassium', 
    hue='Crop Type',
    palette='Set2',
    alpha=0.6
)
plt.title('Moisture vs Potassium Colored by Crop Type')
plt.xlabel('Moisture')
plt.ylabel('Potassium')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()



plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=train_df,
    x='Temparature',
    y='Humidity',
    hue='Soil Type',
    palette='coolwarm',
    alpha=0.6
)
plt.title('Temperature vs Humidity Colored by Soil Type')
plt.xlabel('Temperature')
plt.ylabel('Humidity')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()



categorical_cols = ['Soil Type', 'Crop Type']

for col in categorical_cols:
    plt.figure()
    sns.countplot(data=train_df, x=col, order=train_df[col].value_counts()
                  .index)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


plt.figure()
sns.countplot(data=train_df, x='Fertilizer Name', order=train_df
              ['Fertilizer Name'].value_counts().index)
plt.title('Distribution of Fertilizer Types')
plt.xlabel('Fertilizer Name')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
sns.countplot(data=train_df, x='Crop Type', hue='Fertilizer Name',
              order=train_df['Crop Type'].value_counts().index)
plt.title('Fertilizer Usage Across Crop Types')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
sns.countplot(data=train_df, x='Soil Type', hue='Fertilizer Name', 
              order=train_df['Soil Type'].value_counts().index)
plt.title('Fertilizer Usage Across Soil Types')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


train_df['Crop'] = train_df['Crop Type'].astype('category').cat.codes
train_df['Soil'] = train_df['Soil Type'].astype('category').cat.codes
corr = train_df[['Temparature', 'Humidity','Moisture','Soil','Crop','Nitrogen','Potassium','Phosphorous']].corr()

plt.figure(figsize=(20, 20))
sns.heatmap(
    corr,
    cmap='cividis',
    annot=True,
    annot_kws={"size": 16, "weight": "bold"}
)

plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.title("Correlation Heatmap", fontsize=18, fontweight='bold')
plt.show()



train_df = train_df.drop(columns=['id'])


names = [col for col in train_df.columns if col not in ['Fertilizer Name', 'Soil Type', 'Crop Type']]

print(names)


le_fert = LabelEncoder()
le_soil = LabelEncoder()
le_crop = LabelEncoder()

train_df['Fertilizer Name'] = le_fert.fit_transform(train_df['Fertilizer Name'])
train_df['Soil'] = le_soil.fit_transform(train_df['Soil Type'])
train_df['Crop'] = le_crop.fit_transform(train_df['Crop Type'])

test_df['Soil'] = le_soil.transform(test_df['Soil Type'])
test_df['Crop'] = le_crop.transform(test_df['Crop Type'])


features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous', 'Soil', 'Crop']
X = train_df[features].values
y = train_df['Fertilizer Name'].values
X_test_final = test_df[features].values

X_small, _, y_small, _ = train_test_split(X, y, train_size=1000, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_small, y_small, test_size=0.2, random_state=42)


class DecisionNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, max_depth=6, min_size=5):
        self.max_depth = max_depth
        self.min_size = min_size
        self.root = None

    def fit(self, X, y):
        dataset = np.column_stack((X, y))
        self.root = self._build_tree(dataset)

    def predict(self, X):
        return [self._predict(self.root, row) for row in X]

    def _build_tree(self, dataset, depth=0):
        X, y = dataset[:, :-1], dataset[:, -1]
        if len(set(y)) == 1 or len(y) < self.min_size or depth >= self.max_depth:
            return DecisionNode(value=Counter(y).most_common(1)[0][0])
        feature_index, threshold = self._get_best_split(dataset)
        if feature_index is None:
            return DecisionNode(value=Counter(y).most_common(1)[0][0])
        left, right = self._split(dataset, feature_index, threshold)
        left_node = self._build_tree(left, depth + 1)
        right_node = self._build_tree(right, depth + 1)
        return DecisionNode(feature_index, threshold, left_node, right_node)

    def _split(self, dataset, feature_index, threshold):
        left = np.array([row for row in dataset if row[feature_index] < threshold])
        right = np.array([row for row in dataset if row[feature_index] >= threshold])
        return left, right

    def _gini_index(self, groups, classes):
        n_instances = float(sum([len(group) for group in groups]))
        gini = 0.0
        for group in groups:
            if len(group) == 0:
                continue
            score = 0.0
            labels = group[:, -1]
            for class_val in classes:
                p = np.sum(labels == class_val) / len(labels)
                score += p * p
            gini += (1.0 - score) * (len(group) / n_instances)
        return gini

    def _get_best_split(self, dataset):
        class_values = list(set(dataset[:, -1]))
        b_index, b_value, b_score = None, None, float('inf')
        for index in range(dataset.shape[1] - 1):
            for row in dataset:
                groups = self._split(dataset, index, row[index])
                gini = self._gini_index(groups, class_values)
                if gini < b_score:
                    b_index, b_value, b_score = index, row[index], gini
        return b_index, b_value

    def _predict(self, node, row):
        if node.value is not None:
            return node.value
        if row[node.feature_index] < node.threshold:
            return self._predict(node.left, row)
        else:
            return self._predict(node.right, row)

class RandomForest:
    def __init__(self, n_trees=3, max_depth=6, min_size=5, sample_size=0.7, seed=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_size = min_size
        self.sample_size = sample_size
        self.trees = []
        self.seed = seed

    def fit(self, X, y):
        np.random.seed(self.seed)
        self.trees = []
        for _ in range(self.n_trees):
            indices = np.random.choice(len(X), int(len(X) * self.sample_size), replace=True)
            sample_X, sample_y = X[indices], y[indices]
            tree = DecisionTree(max_depth=self.max_depth, min_size=self.min_size)
            tree.fit(sample_X, sample_y)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        final_predictions = []
        for row in predictions.T:
            final_predictions.append(Counter(row).most_common(1)[0][0])
        return final_predictions

    def predict_top3(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        final_predictions = []
        for row in predictions.T:
            counted = Counter(row).most_common()
            seen = set()
            top3_unique = []
            for class_id, _ in counted:
                if class_id not in seen:
                    top3_unique.append(class_id)
                    seen.add(class_id)
                if len(top3_unique) == 3:
                    break
            while len(top3_unique) < 3:
                top3_unique.append(top3_unique[0])
            final_predictions.append(top3_unique)
        return final_predictions



models = []
accuracies = []

print("This might take a while. Training and evaluating 5 models one by one...\n")
for i in range(5):
    print(f"Model {i+1}")
    start = time.time()
    rf = RandomForest(n_trees=3, max_depth=6, min_size=5, sample_size=0.7, seed=42+i)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"â�±ï¸� Time taken: {time.time() - start:.2f} seconds")
    print(f"âœ… Accuracy: {acc:.4f}\n")
    models.append(rf)
    accuracies.append(acc)

best_index = np.argmax(accuracies)
best_model = models[best_index]

print("\n\n----------------------------------------------")
print(f"Best model: Model {best_index+1} with Accuracy: {accuracies[best_index]:.4f}")


def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        if len(p) > k:
            p = p[:k]
        score = 0.0
        for i, pred in enumerate(p):
            if pred == a and pred not in p[:i]:
                score += 1.0 / (i + 1)
        return score

    return round(np.mean([apk(a, p, k) for a, p in zip(actual, predicted)]), 5)

print("\nEvaluating best model with MAP@3 on validation set...")

val_top3_preds = best_model.predict_top3(X_val)
y_val_list = list(y_val)

map3_score = mapk(y_val_list, val_top3_preds, k=3)
print(f"\n\nMAP@3 Score on Validation Set: {map3_score}")


plt.figure(figsize=(10, 5))
plt.bar(range(1, 6), accuracies, color=['skyblue' if i != best_index else 'limegreen' for i in range(5)])
plt.xticks(range(1, 6), [f'Model {i}' for i in range(1, 6)], fontsize=12)
plt.ylabel('Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Model Number', fontsize=14, fontweight='bold')
plt.title('Custom Random Forest Accuracy per Model', fontsize=16, fontweight='bold')
plt.grid(axis='y')
plt.show()



print("\n Loadingâ€¦ \n Predicting top 3 fertilizers for full test set...")

top3_preds_raw = best_model.predict_top3(X_test_final)

top3_labels = []
for row in top3_preds_raw:
    unique_indices = []
    seen = set()
    for val in row:
        if val not in seen:
            unique_indices.append(val)
            seen.add(val)
        if len(unique_indices) == 3:
            break

    while len(unique_indices) < 3:
        for i in range(len(le_fert.classes_)):
            if i not in seen:
                unique_indices.append(i)
                seen.add(i)
                break

    decoded = le_fert.inverse_transform(unique_indices)
    top3_labels.append(decoded)

top3_strings = [' '.join(name_list) for name_list in top3_labels]

submission = pd.DataFrame({
    'id': test_df['id'],
    'Fertilizer Name': top3_strings
})
submission.to_csv('submission.csv', index=False)

print("\n\nâœ… Submission file created successfully: submission.csv")


sub = pd.read_csv("submission.csv")
sub['Fertilizer Count'] = sub['Fertilizer Name'].apply(lambda x: len(x.strip('"').split(' ')))
print(sub['Fertilizer Count'].value_counts())



submission['Fertilizer Count'] = submission['Fertilizer Name'].apply(lambda x: len(set(x.split())))
repeated = submission[submission['Fertilizer Count'] < 3]
print(f"Rows with duplicate fertilizers: {len(repeated)}")



submission = pd.read_csv('submission.csv')
submission.head()


submission = pd.read_csv('submission.csv')
print("First 5 rows:")
print(submission.head())


print("Total rows in submission.csv file:", len(submission))
print("Unique IDs:", submission['id'].nunique())


print("Random sample from submission.csv file:\n")
print(submission.sample(5))

