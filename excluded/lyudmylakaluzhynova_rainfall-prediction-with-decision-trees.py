# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve, roc_curve, auc


train_df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")
train_df = train_df.drop(["id"], axis=1).copy()
test_df = test_df.drop(["id"], axis=1).copy()


train_df.head(10)


test_df.tail(10)


train_df.info()


test_df.info()


train_df.describe()


train_df.shape


train_df.columns


df_melted = train_df.melt(var_name = 'Features', value_name = 'Value')
df_melted.head(10)


df_melted = train_df.melt(var_name = "Features", value_name = "Value")
graph = sns.FacetGrid(df_melted, col= "Features", col_wrap=4, height=3, sharex=False, sharey=False) # seaborn
graph.map(plt.hist, "Value", bins=15, color = '#2424a8')
graph.set_axis_labels("Value", "Frequency")
graph.fig.suptitle("Distribution of Features",fontsize=15)
graph.fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


rain_counts = train_df["rainfall"].value_counts()
plt.figure(figsize = (8, 4))
fig, ax = plt.subplots()
ax.pie(rain_counts.values, labels = rain_counts.index, autopct='%1.1f%%', colors=["#0290ff", "#ffd800"])
colors=["#1e90ff", "#2424a8" ]
ax.set_title("Rainy and Non-Rainy Days")

plt.show()


correlation_matrix = train_df.corr()
plt.figure(figsize = (8, 4))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation matrix')
plt.show()


# Создаём корреляционную матрицу
correlation_matrix = train_df.corr()

# Разворачиваем корреляционную матрицу в серию
correlation_pairs = correlation_matrix.unstack()

# Убираем самокорреляции (где feature1 == feature2)
correlation_pairs = correlation_pairs[correlation_pairs.index.get_level_values(0) != correlation_pairs.index.get_level_values(1)]

# Фильтруем только те пары, где один из признаков — "rainfall"
correlation_pairs = correlation_pairs[correlation_pairs.index.get_level_values(0) == "rainfall"]
correlation_pairs = correlation_pairs.reset_index()
correlation_pairs.columns = ["Feature1", "Feature2", "Correlation"]

# Сортируем по абсолютному значению корреляции
correlation_pairs = correlation_pairs.sort_values(by="Correlation", key=abs, ascending=False)

# Выбираем топ-4 признака, наиболее коррелирующих с "rainfall"
top_pairs = correlation_pairs.head(4)

# Выводим топ-4 признака
print("Top Correlated Features with Rainfall:\n", top_pairs)

# Строим графики рассеяния для топ-4 признаков
plt.figure(figsize=(12, 8))

for idx, row in enumerate(top_pairs.itertuples(index=False), start=1):
    feature2, correlation = row.Feature2, row.Correlation

    plt.subplot(2, 2, idx)  # Две строки, два столбца для 4 графиков
    plt.scatter(train_df["rainfall"], train_df[feature2], alpha=0.7, color='blue', edgecolor='k')
    plt.title(f"Rainfall vs {feature2}\nCorrelation: {correlation:.2f}")
    plt.xlabel("Rainfall")
    plt.ylabel(feature2)
    plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()



#columns_to_drop = ['day', 'mintemp', 'winddirection']
columns_to_drop = ['day', 'winddirection']
train_df = train_df.drop(columns=columns_to_drop, axis=1)
test_df = test_df.drop(columns=columns_to_drop, axis=1)
train_df.head(10)



corr_matrix = train_df.corr().style.background_gradient()
corr_matrix


# Creating a figure with two rows and two columns of plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.subplots_adjust(hspace=0.4)

# Chart 1
sns.boxplot(x='rainfall', y='humidity', data=train_df, ax=axes[0, 0])
axes[0, 0].set_title('Dependence of raainfall on humidity')

# Chart 2
sns.boxplot(x='rainfall', y='cloud', data=train_df, ax=axes[0, 1])
axes[0, 1].set_title('Dependence of raainfall on cloud')

# Chart 3
sns.boxplot(x='rainfall', y='sunshine', data=train_df, ax=axes[1, 0])
axes[1, 0].set_title('Dependence of raainfall on sunshine')

# Chart 4
sns.boxplot(x='rainfall', y='windspeed', data=train_df, ax=axes[1, 1])
axes[1, 1].set_title('Dependence of raainfall on windspeed')

# Showing all graphs
plt.show()


X = train_df.drop(columns=["rainfall"])
y = train_df["rainfall"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, recall_score, precision_score

# Defining the Decision Tree model
model = DecisionTreeClassifier()

# Defining hyperparameters for Random Search
param_dist = {
    'max_depth': [3, 4,  5, 6, 7, 8, 10, 15, 20, 25],
    'min_samples_split': [2, 3, 4, 5, 10],
    'min_samples_leaf': [1, 2, 4, 6,10, 15, 20]
}

# RandomizedSearchCV: testing 20 random combinations
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=100, cv=5, random_state=42)

# Searching for optimal parameters
random_search.fit(X_train, y_train)

# Results
print("Best parameters:", random_search.best_params_)
print("Test set accuracy:", random_search.score(X_test, y_test))

# Metric evaluation
y_pred = random_search.predict(X_test)
f1 = f1_score(y_test, y_pred)
recall_s = recall_score(y_test, y_pred)
precision_s = precision_score(y_test, y_pred)
print("recall:", recall_s)
print("precision:", precision_s)
print("f1:", f1)



from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
# Encoding classes
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Definition of XGBoost model
# Optimized parameters
xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.92,
    gamma=0.03,
    random_state=42
)

# Training the model on the training data
xgb_model.fit(X_train, y_train)

# Prediction on test data
y_pred = xgb_model.predict(X_test)

# Model accuracy assessment
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

f1 = f1_score(y_test, y_pred, average='weighted')
recall_s = recall_score(y_test, y_pred, average='macro')
precision_s = precision_score(y_test, y_pred, average='macro')
print("recall:", recall_s)
print("precission:", precision_s)
print("f1:", f1)


# Visualization - ROC Curve
from sklearn.metrics import roc_curve, auc
y_prob = xgb_model.predict_proba(X_test)[:, 1] 
fpr, tpr, _ = roc_curve(y_test, y_prob)
# roc_auc = auc(fpr, tpr)
roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score: {roc_auc:.3f}")
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()



# Preprocess test data
test_features = test_df.drop(columns=['id'], errors='ignore')  # Drop unnecessary columns


test_features.isnull().sum()


rainfall = xgb_model.predict_proba(test_features)[:, 1]


submission = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
submission.head()


submission["rainfall"] = rainfall
submission.to_csv("submission.csv", index=False)


submission

