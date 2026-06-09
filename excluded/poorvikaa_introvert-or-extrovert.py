import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")



train


train.shape


train.sample(10)


train = train.fillna(train.mean(numeric_only=True))


train.isnull().sum()


for col in train.select_dtypes(include='object'):
    train[col] = train[col].fillna(train[col].mode()[0])


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for col in train.select_dtypes(include='object'):
    train[col] = le.fit_transform(train[col].astype(str))


numeric_cols = train.select_dtypes(include='number').columns

for col in numeric_cols:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'{col} - Histogram and Boxplot')

    #Histogram
    axes[0].hist(train[col].dropna(), bins=20, color='skyblue', edgecolor='black')
    axes[0].set_title('Histogram')
    axes[0].set_xlabel(col)
    axes[0].set_ylabel('Frequency')

    #Boxplot
    axes[1].boxplot(train[col].dropna(), vert=False)
    axes[1].set_title('Boxplot')
    axes[1].set_xlabel(col)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  
    plt.show()


train = train.drop('id', axis=1)



train.info()


numeric_df = train.select_dtypes(include='number')

corr_matrix = numeric_df.corr()

# 3. Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()


X = train.drop('Personality', axis=1)
y = train['Personality']



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    n_jobs=-1,
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Accuracy: {acc:.4f}")
print("\n Classification Report:\n")
print(classification_report(y_test, y_pred))



comparison_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})

mismatches = comparison_df[comparison_df['Actual'] != comparison_df['Predicted']]
print("First few mismatches:")
print(mismatches.head())

print("\n Full Actual vs Predicted:")
print(comparison_df.head(10)) 




actual_counts = y_test.value_counts().sort_index()
predicted_counts = pd.Series(y_pred).value_counts().sort_index()

# Plot side-by-side bars
labels = actual_counts.index.astype(str)

x = range(len(labels))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x, actual_counts, width=width, label='Actual', color='skyblue')
plt.bar([i + width for i in x], predicted_counts, width=width, label='Predicted', color='salmon')

plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Actual vs Predicted Class Counts')
plt.xticks([i + width / 2 for i in x], labels)
plt.legend()
plt.tight_layout()
plt.show()




test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv').fillna(method='ffill')         # or any fill rule you used


test = test.fillna(test.mean(numeric_only=True))

cat_cols = test.select_dtypes(include=['object', 'category']).columns
for col in cat_cols:
    enc = LabelEncoder()
    test[col] = enc.fit_transform(test[col].astype(str))


test = test[X_train.columns]


y_pred_num = model.predict(test)          

target_encoder = LabelEncoder()
target_encoder.fit(y_train)                

label_map = {0: "Extrovert", 1: "Introvert"}
y_pred_label = [label_map[val] for val in y_pred_num]



submission = pd.DataFrame({'Predicted': y_pred_label})
submission.to_csv('submission.csv', index=False)

print(submission.head())
print("'submission.csv' now contains 'Introvert', 'Extrovert', etc.")



output = pd.DataFrame({'Predicted': y_pred_label})
output.to_csv('submission.csv', index=False)



import os
os.listdir()



pd.read_csv('submission.csv').head()





