import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, auc, roc_curve, ConfusionMatrixDisplay, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.utils import resample
from imblearn.over_sampling import BorderlineSMOTE
from xgboost import XGBClassifier


df=pd.read_csv('/kaggle/input/playground-series-s3e2/train.csv')


df.head()


df.shape


df.info()


#stroke column plot
stroke_counts = df['stroke'].value_counts().sort_index()

plt.figure(figsize=(8, 6))
sns.barplot(x=stroke_counts.index, y=stroke_counts.values, palette="coolwarm")
for i in range(len(stroke_counts)):
    plt.text(i, stroke_counts.values[i], str(stroke_counts.values[i]),
             ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')

plt.title('Stroke Column Distribution', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Stroke (0 = No, 1 = Yes)', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


#gender vs stroke
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='gender', hue='stroke', palette='coolwarm')
plt.title('Gender Distribution by Stroke')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()


#age vs stroke
plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='age', hue='stroke', kde=True, bins=10, palette='coolwarm')
plt.title('Age Distribution by Stroke')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


#BMI vs stroke
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='stroke', y='bmi', palette='coolwarm')
plt.title('BMI Distribution by Stroke')
plt.xlabel('Stroke')
plt.ylabel('BMI')
plt.show()


#smoking status ve stroke
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='smoking_status', hue='stroke', palette='coolwarm')
plt.title('Smoking Status Impact on Stroke')
plt.xlabel('Smoking Status')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


#work type vs stroke
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='work_type', hue='stroke', palette='coolwarm')
plt.title('Work Type Distribution by Stroke')
plt.xlabel('Work Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


#avg_glucose_level vs BMI
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='avg_glucose_level', y='bmi', hue='stroke', style='stroke', palette='coolwarm', s=100)
plt.title('Relationship between Avg Glucose Level and BMI')
plt.xlabel('Average Glucose Level')
plt.ylabel('BMI')
plt.show()


#hypertension vs stroke
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='hypertension', hue='stroke')
plt.title('Hypertension vs. Stroke')
plt.xlabel('Hypertension')
plt.ylabel('Count')
plt.show()


#correlation heatmap of numeric columns
numeric_cols = df.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_cols.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True, linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Features', fontsize=16, fontweight='bold', pad=15)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()


x = df.drop(columns=['stroke'])
y = df['stroke']

x = pd.get_dummies(x, drop_first=True)

#combining x and y for easier manipulation
data = pd.concat([x, y], axis=1)


#separating majority and minority classes
majority = data[data['stroke'] == 0]
minority = data[data['stroke'] == 1]

#undersampling the majority class
majority_downsampled = resample(majority, replace=False, n_samples=len(minority), random_state=42)

#combining the undersampled majority class with the minority class
balanced_data = pd.concat([majority_downsampled, minority])

#applying borderline-SMOTE to both classes for augmentation
x_balanced = balanced_data.drop(columns=['stroke'])
y_balanced = balanced_data['stroke']

smote = BorderlineSMOTE(random_state=42)
x_augmented, y_augmented = smote.fit_resample(x_balanced, y_balanced)


x_train, x_test, y_train, y_test = train_test_split(x_augmented, y_augmented, test_size=0.2, random_state=42)

#defining the classifiers with initial hyperparameters
classifiers = {
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=200, max_depth=10, class_weight='balanced'),
    "Logistic Regression": LogisticRegression(random_state=42, class_weight='balanced', max_iter=500),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42, n_estimators=200, learning_rate=0.1, max_depth=5),
    "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', n_estimators=200, max_depth=5, learning_rate=0.1),
    "Support Vector Machine": SVC(probability=True, random_state=42)
}



#results container
results = {}
roc_curves = []

#training the models
for name, clf in classifiers.items():
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    y_prob = clf.predict_proba(x_test)[:, 1] if hasattr(clf, "predict_proba") else None
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
    results[name] = {
        "Classification Report": classification_report(y_test, y_pred, output_dict=True),
        "AUC-ROC": auc
    }
    
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_curves.append((name, fpr, tpr, auc))



#plot ROC Curves
plt.figure(figsize=(10, 8))
for name, fpr, tpr, auc in roc_curves:
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guessing")
plt.title('ROC Curves Comparison', fontsize=16, fontweight='bold')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.legend(loc='lower right', fontsize=10)
plt.grid()
plt.tight_layout()
plt.show()


#classification reports
for name, res in results.items():
    print(f"\n{name} Results:")
    print(f"Classification Report:\n{pd.DataFrame(res['Classification Report'])}")
    print(f"AUC-ROC: {res['AUC-ROC']:.2f}")


#continuing with the gradient boosting model, which performed the best for both classes
gb = GradientBoostingClassifier(random_state=42, n_estimators=200, learning_rate=0.1, max_depth=5)
gb.fit(x_train, y_train)

y_pred = gb.predict(x_test)
y_prob = gb.predict_proba(x_test)[:, 1]


#confusion matrix
cm = confusion_matrix(y_test, y_pred)
cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
cmd.plot(cmap="Blues")
plt.title("Confusion Matrix - Gradient Boosting", fontsize=14)
plt.show()


#precision-recall curve
precision, recall, _ = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color="green", label="Precision-Recall Curve")
plt.title("Precision-Recall Curve - Gradient Boosting", fontsize=14)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower left")
plt.grid()
plt.show()


#classification report
print("Classification Report:\n", classification_report(y_test, y_pred))


test = pd.read_csv("/kaggle/input/playground-series-s3e2/test.csv")
test_features = pd.get_dummies(test, drop_first=True).reindex(columns=x.columns, fill_value=0)
test_predictions = gb.predict_proba(test_features)[:, 1]


#creating submission file
submission = pd.DataFrame({
    "id": test["id"],
    "stroke": test_predictions
})
submission.to_csv("submission.csv", index=False)

