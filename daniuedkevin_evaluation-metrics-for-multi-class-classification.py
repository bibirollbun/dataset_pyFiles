import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import os
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')
train.head()


train['is_train'] = 1
test['is_train'] = 0
data = pd.concat([train, test], sort=False)

categorical_cols = ['Soil Type', 'Crop Type']
for col in categorical_cols:
    data[col] = data[col].astype('category').cat.codes

numerical_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Phosphorous', 'Potassium']
data[numerical_cols] = (data[numerical_cols] - data[numerical_cols].mean()) / data[numerical_cols].std()

fertilizer_labels = data.loc[data['is_train'] == 1, 'Fertilizer Name'].astype('category')
data.loc[data['is_train'] == 1, 'Fertilizer Name'] = fertilizer_labels.cat.codes
label_map = dict(enumerate(fertilizer_labels.cat.categories))

train_data = data[data['is_train'] == 1].drop(columns=['id', 'is_train'])
test_data = data[data['is_train'] == 0].drop(columns=['Fertilizer Name', 'is_train'])

X = train_data.drop(columns=['Fertilizer Name']).values.astype(float)
y = train_data['Fertilizer Name'].values.astype(int)

X_test_final = test_data.drop(columns=['id']).values.astype(float)

print(f"Training samples: {X.shape[0]}, Features: {X.shape[1]}")


def custom_train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_size = int(len(X) * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

X_train, X_val, y_train, y_val = custom_train_test_split(X, y)
print("Train shape:", X_train.shape, "Validation shape:", X_val.shape)


def one_hot(y, num_classes):
    one_hot_y = np.zeros((len(y), num_classes))
    one_hot_y[np.arange(len(y)), y] = 1
    return one_hot_y

class NeuralNet:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.01):
        self.lr = lr
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros((1, output_dim))

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_deriv(self, Z):
        return (Z > 0).astype(float)

    def softmax(self, Z):
        exps = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m
        return loss

    def backward(self, X, y_true, y_pred):
        m = y_true.shape[0]
        dZ2 = y_pred - y_true
        dW2 = self.A1.T @ dZ2 / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.relu_deriv(self.Z1)
        dW1 = X.T @ dZ1 / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2


num_classes = len(np.unique(y))
model = NeuralNet(input_dim=X_train.shape[1], hidden_dim=64, output_dim=num_classes, lr=0.1)

epochs = 100
y_train_onehot = one_hot(y_train, num_classes)
y_val_onehot = one_hot(y_val, num_classes)

for epoch in range(1, epochs + 1):
    y_pred_train = model.forward(X_train)
    loss = model.compute_loss(y_train_onehot, y_pred_train)
    model.backward(X_train, y_train_onehot, y_pred_train)

    if epoch % 10 == 0 or epoch == 1:
        y_pred_val = model.forward(X_val)
        val_loss = model.compute_loss(y_val_onehot, y_pred_val)
        val_acc = np.mean(np.argmax(y_pred_val, axis=1) == y_val)
        print(f"Epoch {epoch}: Train Loss = {loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")


y_test_pred = model.forward(X_test_final)

# Get top 3 predicted class indices
top3_indices = np.argsort(y_test_pred, axis=1)[:, -3:][:, ::-1]

# Convert indices to fertilizer names
top3_names = []
for row in top3_indices:
    names = [label_map[idx] for idx in row]
    top3_names.append(" ".join(names))

submission['Fertilizer Name'] = top3_names
submission.to_csv("submission.csv", index=False)

# Show first few predictions
print("First 10 test predictions (top 3 fertilizers):")
print(submission.head(10))


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score
from sklearn.preprocessing import label_binarize
import numpy as np
import pandas as pd

# Get predicted probabilities and classes
y_val_pred_probs = model.forward(X_val)
y_val_preds = np.argmax(y_val_pred_probs, axis=1)
class_labels = np.unique(y)

# 1. Confusion Matrix
conf_mat = confusion_matrix(y_val, y_val_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 2. F1 Score per class (bar plot)
f1_scores = f1_score(y_val, y_val_preds, average=None)
plt.figure(figsize=(10, 5))
plt.bar(class_labels, f1_scores)
plt.title("F1 Score per Fertilizer")
plt.xlabel("Fertilizer Class")
plt.ylabel("F1 Score")
plt.xticks(class_labels)
plt.show()

# 3. Prediction Distribution (bar chart)
pred_counts = pd.Series(y_val_preds).value_counts().sort_index()
plt.figure(figsize=(10, 5))
plt.bar(pred_counts.index, pred_counts.values)
plt.title("Predicted Fertilizer Class Distribution")
plt.xlabel("Predicted Fertilizer Class")
plt.ylabel("Count")
plt.show()

# 4. ROC Curve per class (One-vs-Rest)
y_val_bin = label_binarize(y_val, classes=class_labels)
plt.figure(figsize=(10, 8))
for i in range(len(class_labels)):
    fpr, tpr, _ = roc_curve(y_val_bin[:, i], y_val_pred_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {class_labels[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curves (OvR)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid()
plt.show()


from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score

# 1. Accuracy
accuracy = accuracy_score(y_val, y_val_preds)
print(f"ðŸ”¹ Accuracy: {accuracy:.4f}")

# 2. F1 Score (macro & weighted)
f1_macro = f1_score(y_val, y_val_preds, average='macro')
f1_weighted = f1_score(y_val, y_val_preds, average='weighted')
print(f"ðŸ”¹ F1 Score (Macro): {f1_macro:.4f}")
print(f"ðŸ”¹ F1 Score (Weighted): {f1_weighted:.4f}")

# 3. ROC AUC Score (OvR)
roc_auc = roc_auc_score(label_binarize(y_val, classes=np.unique(y)), y_val_pred_probs, average='macro', multi_class='ovr')
print(f"ðŸ”¹ ROC AUC Score (OvR): {roc_auc:.4f}")

# 4. Classification Report
target_names = [label_map[i] for i in np.unique(y)]
print("\\nðŸ”¹ Classification Report:")
print(classification_report(y_val, y_val_preds, target_names=target_names))

