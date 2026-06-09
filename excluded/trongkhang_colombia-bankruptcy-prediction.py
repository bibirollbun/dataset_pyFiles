import pandas as pd 
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv("/kaggle/input/colombia-bankruptcy-prediction/train.csv")

df.info()

df.head()


df = df.rename(columns={
    'Ganancia bruta': 'Gross profit',
    'Ganancia (pérdida)': 'Profit (loss)',
    'Ingresos de actividades ordinarias': 'Revenue from ordinary activities',
    'Costo de ventas': 'Cost of sales',
    'Patrimonio total': 'Total equity',
    'Total pasivos': 'Total liabilities',
    'Total de activos': 'Total assets',
    'Ganancias acumuladas': 'Retained earnings',
    'Pasivos corrientes totales': 'Total current liabilities',
    'Activos corrientes totales': 'Total current assets',
    'Sector': 'Sector',
    'event': 'Event'
})


print("**Missing Value")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("**Duplicate Value")
print(df.duplicated().sum())


import matplotlib.pyplot as plt
import seaborn as sns

corr = df.corr(numeric_only=True)
plt.figure(figsize=(10,8))  # Tùy chỉnh kích thước
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
plt.title("Correlation Heatmap", fontsize=16)
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import math

# Lấy danh sách các cột số
numerical_cols = df.select_dtypes(include='number').columns

n_cols = 4
n_rows = math.ceil(len(numerical_cols) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 4))

axes = axes.flatten()

for i, col in enumerate(numerical_cols):
    sns.histplot(df[col].dropna(), kde=True, bins=30, ax=axes[i])
    axes[i].set_title(f'Histogram of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()



from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

# Chuẩn bị dữ liệu (loại bỏ NaN cho mô hình)
df_model = df.copy().dropna()

# Tách target
y = df_model["Event"]
X = df_model.drop(["Event","index"], axis=1)

# Xử lý dữ liệu
X["Sector"] = X["Sector"].astype('category')
X = pd.get_dummies(X, drop_first=True)

# Huấn luyện mô hình cây quyết định
dt_classifier = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_classifier.fit(X, y)

# Trực quan hóa cây
plt.figure(figsize=(25, 15))
plot_tree(
    dt_classifier,
    feature_names=X.columns,
    class_names=[str(cls) for cls in dt_classifier.classes_],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Cây Quyết định để tìm quy tắc phi tuyến tính")
plt.savefig("/kaggle/working/tree.jpg")
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))

sns.countplot(data=df, x="Sector", hue="Event")

plt.title("Số lượng Event theo Sector")
plt.xlabel("Sector")
plt.ylabel("Số lượng")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



df = df.dropna()


drop_cols = ["index"]

for col in drop_cols:
    df = df.drop(col, axis=1)


df["Sector"] = df["Sector"].astype('category')

print(df["Sector"].unique())

df_dummies = pd.get_dummies(df, drop_first=True)

df_dummies.head()


from sklearn.model_selection import train_test_split

X = df_dummies.drop('Event',axis=1)
y = df_dummies['Event']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print(y_train.value_counts())


from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Định nghĩa model + params
models_params = [
    {
        "name": "Logistic Regression",
        "model": LogisticRegression(max_iter=1000),
        "params": {
            "C": [0.01, 0.1, 1, 10],
            "solver": ["liblinear", "lbfgs"]
        }
    },
    {
        "name": "XGBoost",
        "model": XGBClassifier(eval_metric='logloss'),
        "params": {
            "n_estimators": [50, 100],
            "max_depth": [3, 5],
            "learning_rate": [0.01, 0.1]
        }
    },
    {
        "name": "LightGBM",
        "model": LGBMClassifier(verbosity=-1),
        "params": {
            "n_estimators": [50, 100],
            "max_depth": [3, 5],
            "learning_rate": [0.01, 0.1]
        }
    }
]

# Huấn luyện và đánh giá
for item in models_params:
    print(f"\n================= {item['name']} =================")
    
    grid = GridSearchCV(item["model"], item["params"], cv=5, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    print(f"Best parameters: {grid.best_params_}")
    print(f"Best CV score: {grid.best_score_:.4f}")
    
    # Dự đoán
    y_pred = grid.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    # Ma trận nhầm lẫn
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {item['name']}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    



from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Tính scale_pos_weight cho XGBoost & LightGBM
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos

# Định nghĩa model + params
models_params = [
    {
        "name": "Logistic Regression",
        "model": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "params": {
            "C": [0.01, 0.1, 1, 10],
            "solver": ["liblinear", "lbfgs"]
        }
    },
    {
        "name": "XGBoost",
        "model": XGBClassifier(eval_metric='logloss'),
        "params": {
            "scale_pos_weight": [1, 2, 3, float(scale_pos_weight)],
            "n_estimators": [50, 100],
            "max_depth": [3, 5],
            "learning_rate": [0.01, 0.1]
        }
    },
    {
        "name": "LightGBM",
        "model": LGBMClassifier(verbosity=-1),
        "params": {
            "scale_pos_weight": [1, 2, 3, float(scale_pos_weight)],
            "n_estimators": [50, 100],
            "max_depth": [3, 5],
            "learning_rate": [0.01, 0.1]
        }
    }
]

# Huấn luyện và đánh giá
for item in models_params:
    print(f"\n================= {item['name']} =================")
    
    grid = GridSearchCV(item["model"], item["params"], cv=5, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    print(f"Best parameters: {grid.best_params_}")
    print(f"Best CV score: {grid.best_score_:.4f}")
    
    # Dự đoán
    y_pred = grid.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    # Ma trận nhầm lẫn
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {item['name']}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    



df_test = pd.read_csv("/kaggle/input/colombia-bankruptcy-prediction/test.csv")

df_test_drop = df_test.drop("index",axis=1)

df_test_drop["Sector"] = df_test_drop["Sector"].astype('category')

df_test_drop_dummies = pd.get_dummies(df_test_drop, drop_first=True)

df_test_drop_dummies.head()


df_test_drop_dummies = df_test_drop_dummies.rename(columns={
    'Ganancia bruta': 'Gross profit',
    'Ganancia (pérdida)': 'Profit (loss)',
    'Ingresos de actividades ordinarias': 'Revenue from ordinary activities',
    'Costo de ventas': 'Cost of sales',
    'Patrimonio total': 'Total equity',
    'Total pasivos': 'Total liabilities',
    'Total de activos': 'Total assets',
    'Ganancias acumuladas': 'Retained earnings',
    'Pasivos corrientes totales': 'Total current liabilities',
    'Activos corrientes totales': 'Total current assets',
    'Sector': 'Sector',
    'event': 'Event'
})

df_test_drop_dummies.head()


best_model = grid.best_estimator_

y_predict = best_model.predict(df_test_drop_dummies)

submission = pd.DataFrame({"index":df_test["index"], "prediction":y_predict})

submission.to_csv("submission.csv",index=False)


sub = pd.read_csv("/kaggle/working/submission.csv")

sub.head()

