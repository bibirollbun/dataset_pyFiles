import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv("/kaggle/input/loan-approval-predictions/train.csv")
df.head()


df.shape


df = df.drop("id", axis=1)


df.info()


for col in df.columns:
    print(f"\nColumn: {col}")
    print(f"The type data in {col} is {np.dtype(df[col])}")
    print(f"Number of unique values in '{col}': {df[col].nunique()}")
    if pd.api.types.is_numeric_dtype(df[col]):
        print(df[col].value_counts().head(10)) 
    else:
        print(df[col].value_counts())
    print("-------------------------")




counts = df['loan_status'].value_counts()
labels = counts.index
sizes = counts.values


plt.figure(figsize=(6,6))
plt.pie(
    sizes, 
    labels=labels, 
    autopct='%1.1f%%',  
    startangle=90,       
    colors=['skyblue', 'lightcoral']
)
plt.title('Loan Status Distribution', fontsize=14)
plt.axis('equal') 
plt.show()



df.describe()


df_num = df.select_dtypes(include=["float64", "int64"]).drop(columns=["loan_status"])
df_num.head()


df_num.hist(figsize=(20,24),
            bins=50,
            xlabelsize=15,
            ylabelsize=15,
            color="r");


df_cat = df.select_dtypes(include="object")
df_cat


for col in df_cat:
    sns.countplot(data=df, x=col,hue="loan_status",palette="Set2")
    plt.xticks(rotation=45)
    plt.title(f"{col} Distribution")
    plt.show()



for col in df_cat.columns:
    plt.figure(figsize=(10,6))
    ax = sns.countplot(data=df, x=col, hue="loan_status", palette="Set2")
    plt.xticks(rotation=45)
    plt.title(f"{col} Distribution with Loan Status Counts")
    
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.text(
                p.get_x() + p.get_width() / 2,  
                height + 1,                    
                int(height),                   
                ha='center'                   
            )
    plt.show()



for col in df_cat.columns:
    print(pd.crosstab(df[col], df['loan_status'], normalize='index'))
    print("-"*40)


for col in df_num.columns:
    plt.figure(figsize=(8,5))
    sns.boxplot(data=df, x='loan_status', y=col,palette="Set3")
    plt.xlabel("Loan Status")
    plt.ylabel("col")
    plt.title(f"Boxpltot of {col} by loan_status")
    plt.show()


# correlation matrix


corr = df.corr(numeric_only=True)

plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={"shrink": .8})
plt.title("Correlation Matrix")
plt.show()



plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='loan_amnt', y='loan_int_rate', hue='loan_status', palette=['red', 'green'], alpha=0.6)

plt.title('Scatter Plot of Loan Amount vs Interest Rate by Loan Status')
plt.xlabel('Loan Amount')
plt.ylabel('Interest Rate (%)')
plt.legend(title='Loan Status')
plt.show()


plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='loan_percent_income', y='person_income', hue='loan_status', palette=['red', 'green'], alpha=0.6)

plt.title('Scatter Plot of loan percent income vs person income by Loan Status')
plt.xlabel('loan_percent_income')
plt.ylabel('person_income')
plt.legend(title='Loan Status')
plt.show()


df.isnull().sum()


for col in df_num.columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5* IQR) | (df[col]> Q3 + 1.5*IQR)]
    print(f"Number of outliers in {col} column: " ,len(outliers))
    print(f"Q1 = {Q1}")
    print(f"Q3 = {Q3}")
    print("-"*40)



df.describe()


df = df[df['person_age']< 57 ]
df = df[df['person_emp_length']< 30 ]
Q1 = 42000.0
Q3 = 75600.0
IQR = Q3 - Q1

lower_bound = Q1 - 1.50 * IQR
upper_bound = Q3 + 1.50 * IQR

df = df[(df['person_income'] >= lower_bound) & (df['person_income'] <= upper_bound)]



for col in df_num.columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5* IQR) | (df[col]> Q3 + 1.5*IQR)]
    print(f"Number of outliers in {col} column: " ,len(outliers))
    print(f"Q1 = {Q1}")
    print(f"Q3 = {Q3}")
    print("-"*40)



df.shape


df = pd.get_dummies(df, drop_first=True)



df.info()



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[df_num.columns] = scaler.fit_transform(df[df_num.columns])


df.head()


from sklearn.model_selection import train_test_split

X = df.drop('loan_status', axis=1)
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train :", X_train.shape)
print("y_train :", y_train.shape)
print("X_test :", X_test.shape)
print("y_test :", y_test.shape)


# np.random.seed(42)
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score

# logreg_model = LogisticRegression(max_iter=1000)
# dt_model = DecisionTreeClassifier()
# rf_model = RandomForestClassifier()
# gb_model = GradientBoostingClassifier()
# svm_model = SVC()
# knn_model = KNeighborsClassifier()

# models = {
#     'Logistic Regression': logreg_model,
#     'Decision Tree': dt_model,
#     'Random Forest': rf_model,
#     'Gradient Boosting': gb_model,
#     'SVM': svm_model,
#     'KNN': knn_model
# }

# for name, model in models.items():
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     acc = accuracy_score(y_test, y_pred)
#     print(f"{name} Accuracy: {acc:.4f}")


# np.random.seed(42)
# from sklearn.metrics import  classification_report


# for name , model in models.items():
#     print(name)
#     print(classification_report(y_test, y_pred))
#     print("-"*60)



# Fitting Random Forest model
np.random.seed(42)
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)


# ُClassification Report
np.random.seed(42)
from sklearn.metrics import classification_report, confusion_matrix
y_pred = rf_model.predict(X_test)
print("Classification Report")
print(classification_report(y_test, y_pred))



# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.show()


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


y_probs = rf_model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_probs)
auc_score = roc_auc_score(y_test, y_probs)

# Plot ROC Curve
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f'Random Forest (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess') 
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()



pip install scikit-learn==1.5.2



pip install --force-reinstall imbalanced-learn



from imblearn.over_sampling import SMOTE





