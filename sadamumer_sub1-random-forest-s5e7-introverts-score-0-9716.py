import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


train_df.head()


train_df.info()


train_df.describe()


train_df.shape


train_df.isna().sum()


train_df['Stage_fear'].value_counts()


sns.countplot(data=train_df,x='Stage_fear')
plt.title('Value Counts: Stage_fear')
plt.show()
train_df['Stage_fear'].value_counts()


sns.countplot(data=train_df,x='Drained_after_socializing')
plt.title('Value Counts: Drained_after_socializing')
plt.show()
train_df['Drained_after_socializing'].value_counts()


sns.countplot(data=train_df,x='Personality')
plt.title('Value Counts: Personality')
plt.show()
train_df['Personality'].value_counts()


introHeatmap = train_df[train_df['Personality']=="Introvert"].drop(['id','Stage_fear','Drained_after_socializing','Personality'],axis=1).corr()
extroHeatmap = train_df[train_df['Personality']=="Extrovert"].drop(['id','Stage_fear','Drained_after_socializing','Personality'],axis=1).corr()


plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.heatmap(introHeatmap, annot=True, cmap=sns.cubehelix_palette(as_cmap=True), fmt=".2f")
plt.title("Correlations for  Introvert")

plt.subplot(1,2,2)
sns.heatmap(extroHeatmap, annot=True, cmap=sns.cubehelix_palette(as_cmap=True), fmt=".2f")
plt.title("Correlations for  Extrovert")

plt.tight_layout()

plt.show()


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score


# Separate features and target
X = train_df.drop(['id', 'Personality'], axis=1)
y = train_df['Personality'].map({'Introvert': 0, 'Extrovert': 1})  # Encode target
X_test = test_df.drop(['id'], axis=1)

# Handle categorical variables
categorical_cols = ['Stage_fear', 'Drained_after_socializing']
for col in categorical_cols:
    X[col] = X[col].map({'No': 0, 'Yes': 1})
    X_test[col] = X_test[col].map({'No': 0, 'Yes': 1})
    
    # Impute missing values in categorical columns with mode
    imputer = SimpleImputer(strategy='most_frequent')
    X[col] = imputer.fit_transform(X[[col]]).ravel()
    X_test[col] = imputer.transform(X_test[[col]]).ravel()

# Handle numerical columns
numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                  'Friends_circle_size', 'Post_frequency']

imputer_num = SimpleImputer(strategy='median')
X[numerical_cols] = imputer_num.fit_transform(X[numerical_cols])
X_test[numerical_cols] = imputer_num.transform(X_test[numerical_cols])


# Scale numerical features
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Split training data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Validate model
val_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, val_pred)
print(f'Validation Accuracy: {accuracy:.4f}')

# Cross-validation score
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f'Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')



# Predict on test set
test_pred = model.predict(X_test)
test_pred_labels = np.where(test_pred == 0, 'Introvert', 'Extrovert')


importances = model.feature_importances_
feature_names = X.columns
for name, importance in zip(feature_names, importances):
    print(f'{name}: {importance:.4f}')


# Create submission file
submission = pd.DataFrame({'id': test_df['id'], 'Personality': test_pred_labels})
submission.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")


submission

