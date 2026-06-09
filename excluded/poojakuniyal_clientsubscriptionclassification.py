import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


train_df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
train_df.head() 


print(f"Length of train dataset {train_df.shape[0]} rows and {train_df.shape[1]} columns")


train_df.info()


train_df.describe(include='object')


train_df.describe()


test_df = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
test_df.head()


pd.crosstab(train_df['default'], train_df['y'], normalize='index')


sns.countplot(data= train_df, x='education', hue='y', color='Red', edgecolor='darkred', linewidth=2)
plt.title("How education level impacts subscription to bank term deposit")


sns.countplot(data=train_df, x='default', hue='y', color='Green', edgecolor='darkgreen', linewidth=2)

plt.title("Visualizing how client's default relates to thier subscription to bank term deposit");


plt.figure(figsize=(8, 5))

# custom pink palette 
pink_palette = ['#ffb6c1', '#ff69b4']  # Light pink and hot pink

sns.countplot(data=train_df, x='job', hue='y', palette=pink_palette, edgecolor='#ff69b4', linewidth=1.5)

plt.title('Distribution of Bank term deposit Subscription Across Job Status')
plt.xlabel('Job Status')
plt.ylabel('Count')
plt.legend(title='Target (y)')
plt.xticks(rotation=75)  
plt.tight_layout()
plt.show()


cat_vars = list(train_df.select_dtypes(include='object'))
cat_vars


num_vars = list(train_df.select_dtypes(exclude='object'))
num_vars = [i for i in num_vars if i not in ([cat_vars, 'id','y'])]
num_vars


corr = train_df[num_vars].corr()

plt.figure(figsize=(8, 6))

sns.heatmap(corr, annot=True, cmap='YlOrBr', vmin=-1, vmax=1, linewidths=0.5, linecolor='gray')

plt.title('Correlation Matrix Heatmap', fontsize=14)
plt.tight_layout()
plt.show()


test_id = test_df['id']


train_df.drop(columns=['id'], axis=1, inplace=True)
test_df.drop(columns=['id'], axis=1, inplace=True) 


X= train_df.drop(columns='y')
y = train_df['y']


X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2,
                                                 stratify=y, random_state=77)


X_train.shape, X_test.shape, y_train.shape, y_test.shape


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


# Define transformers
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine into a ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, num_vars),
    ('cat', categorical_transformer, cat_vars)
])


X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)
test_df = preprocessor.transform(test_df)


# Get categorical feature names after transformation
cat_transformer = preprocessor.named_transformers_['cat']
cat_feature_names =cat_transformer.get_feature_names_out(cat_vars)

# Convert to a list for easier use
cat_feature_names = list(cat_feature_names)
print(cat_feature_names)


feature_names = list(num_vars) + cat_feature_names


# Recreate DataFrame with transformed data and original column names
X_train = pd.DataFrame(X_train,columns=feature_names)
X_test = pd.DataFrame(X_test, columns=feature_names)
test_df = pd.DataFrame(test_df, columns=feature_names)


X_train.head(3)


dt_model = DecisionTreeClassifier()


dt_model.fit(X_train,y_train) 


from sklearn.metrics import roc_auc_score

# Get predicted probabilities for the positive class
y_probs1 = dt_model.predict_proba(X_test)[:, 1]

# Calculate ROC AUC
roc_auc = roc_auc_score(y_test, y_probs1)
print(f"ROC AUC Score: {roc_auc:.4f}")


rt_model = RandomForestClassifier()
rt_model.fit(X_train,y_train)


y_probs = rt_model.predict_proba(X_test)[:,1]
roc_auc = roc_auc_score(y_test, y_probs)
print(f"ROC AUC Score: {roc_auc:.4f}")


from sklearn.metrics import roc_curve, auc

# Compute ROC curve and ROC area
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve of RandomForest')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()


from sklearn.ensemble import ExtraTreesClassifier


et_model = ExtraTreesClassifier()
et_model.fit(X_train,y_train)



# Get predicted probabilities for the positive class
y_probs2 = et_model.predict_proba(X_test)[:, 1]

# Calculate ROC AUC
roc_auc = roc_auc_score(y_test, y_probs2)
print(f"ROC AUC Score: {roc_auc:.4f}")


# Compute ROC curve and ROC area
fpr, tpr, thresholds = roc_curve(y_test, y_probs2)
roc_auc = auc(fpr, tpr)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='crimson', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve of ExtraTrees')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()


log_model = LogisticRegression()
log_model.fit(X_train,y_train)

y_probs_log = log_model.predict_proba(X_test)[:,1]


# Calculate ROC AUC
roc_auc = roc_auc_score(y_test, y_probs_log)
print(f"ROC AUC Score: {roc_auc:.4f}")


# Compute ROC curve and ROC area
fpr, tpr, thresholds = roc_curve(y_test, y_probs_log)
roc_auc = auc(fpr, tpr)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='lime', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='green', lw=2, linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve of Logistic Regression')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()


# Get predicted probabilities
y_probs_rf = rt_model.predict_proba(X_test)[:, 1]
y_probs_et = et_model.predict_proba(X_test)[:, 1]
y_probs_lr = log_model.predict_proba(X_test)[:, 1]

# Example weights (adjust as needed)
w_rf = 0.8
w_et = 0.10
w_lr = 0.10

# Weighted average of predicted probabilities
y_probs_weighted = (
    w_rf * y_probs_rf +
    w_et * y_probs_et +
    w_lr * y_probs_lr
)

roc_auc = roc_auc_score(y_test, y_probs_weighted)
print(f"Weighted Ensemble ROC AUC: {roc_auc:.4f}")


# Compute ROC curve and ROC area
fpr, tpr, thresholds = roc_curve(y_test, y_probs_weighted)
roc_auc = auc(fpr, tpr)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='purple', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='lime', lw=2, linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve of Averaged Predicition')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()


# Get predicted probabilities from each model
y_rf_test = rt_model.predict_proba(test_df)[:, 1]
y_et_test = et_model.predict_proba(test_df)[:, 1]
y_lr_test = log_model.predict_proba(test_df)[:, 1] 

# Weighted average (adjust weights as needed)
y_test_avg = (
    0.8 * y_rf_test +
    0.10 * y_et_test +
    0.10 * y_lr_test
)



# Create a submission DataFrame
submission_df = pd.DataFrame({
    'id': test_id, 
    'y': y_test_avg
})
submission_df.head() 


# Save to CSV
submission_df.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' generated successfully.")




