import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



train_path= '/kaggle/input/playground-series-s5e7/train.csv'
test_path= '/kaggle/input/playground-series-s5e7/test.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print(f'Train shape: {train.shape}')
print(f'Test shape:  {test.shape}')




print('First five rows Training set')
display(train.head())

print('First five rows Test set')
display(test.head())

print('\nSumarry statistics (numerical and categorical)(Train):')
display(train.describe(include='all'))

missing_val= train.isnull().mean()* 100
print('n\Columns with missing values (%)(Train):')
display(missing_val[missing_val > 0].sort_values (ascending=False))


plt.figure(figsize=(6,4))
sns.countplot(x='Personality', data=train)
plt.title('Personality Class Distribution')
plt.xlabel('Personality')
plt.ylabel('Count')
plt.show()

counts= train['Personality'].value_counts()
print('Personality counts')
print(counts)


features =['Time_spent_Alone','Going_outside','Post_frequency']

for feat in features:
    plt.figure9figsize=(8,4)
    sns.kdeplot(data=train, x=feat, hue= 'Personality', common_norm=False,fill=True,alpha= 0.4)
    plt.title(f'{feat} Distribution by Personality')
    plt.xlabel(feat)
    plt.ylabel('Density')
    plt.show()
    



plt.figure(figsize=(8,4))
sns.countplot(
    data=train,
    x='Stage_fear',
    hue='Personality'
)
plt.title('Stage_fear Counts by Personality')
plt.xlabel('Stage_fear')
plt.ylabel('Count')
plt.show()


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

le = LabelEncoder()
y =le.fit_transform(train['Personality'])

X = train.drop(columns=['Personality'])

X_train,X_val , y_train, y_val = train_test_split(
    X,y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f'X_train: {X_train.shape}, y_train: {y_train.shape}')
print(f'X_val: {X_val.shape}, y _val: {y_val.shape}')


cat_cols =X_train.select_dtypes(include=['object']).columns.tolist()
print('Categorical features:', cat_cols)

X_train_enc = pd.get_dummies(X_train, columns=cat_cols,drop_first=True)
X_val_enc   = pd.get_dummies(X_val, columns=cat_cols,drop_first=True)

X_val_enc = X_val_enc.reindex(columns= X_train_enc.columns,fill_value=0)

print('After encoding:')
print('X_train_enc:',X_train_enc.shape)
print('X_val_enc:',X_val_enc.shape)



print("Total NaNs in X_train_enc:", X_train_enc.isnull().sum().sum())
print("Total NaNs in X_val_enc  :", X_val_enc.isnull().sum().sum())




from sklearn.impute import SimpleImputer

num_cols = X_train_enc.select_dtypes(include=[float, int]).columns
imp = SimpleImputer(strategy='median')
X_train_enc[num_cols] = imp.fit_transform(X_train_enc[num_cols])
X_val_enc  [num_cols] = imp.transform(X_val_enc[num_cols])




print("Total NaNs in X_train_enc:", X_train_enc.isnull().sum().sum())
print("Total NaNs in X_val_enc  :", X_val_enc.isnull().sum().sum())



from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd

def simple_smote(X, y, minority_label=1, k=5, random_state=42):
    np.random.seed(random_state)

   
    X_min = X[y == minority_label]
    n_min, n_feat = X_min.shape
    n_maj = np.sum(y != minority_label)
    n_to_generate = n_maj - n_min

 
    nn = NearestNeighbors(n_neighbors=k+1).fit(X_min)
    neigh_idxs = nn.kneighbors(X_min, return_distance=False)[:, 1:]

   
    synthetic = []
    for _ in range(n_to_generate):
        idx = np.random.randint(0, n_min)
        neighbor = np.random.choice(neigh_idxs[idx])
        diff = X_min[neighbor] - X_min[idx]
        gap = np.random.rand(n_feat)
        synthetic.append(X_min[idx] + gap * diff)
    X_syn = np.vstack(synthetic)
    y_syn = np.array([minority_label] * len(X_syn))

    # 4. Combine
    X_bal = np.vstack([X, X_syn])
    y_bal = np.hstack([y, y_syn])
    return X_bal, y_bal


X_train_sm, y_train_sm = simple_smote(X_train_enc.values, y_train, minority_label=1)


X_train_bal = pd.DataFrame(X_train_sm, columns=X_train_enc.columns)
y_train_bal = y_train_sm


print("New class counts:", np.bincount(y_train_bal))




from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

rf_model = RandomForestClassifier(n_estimators= 200 , random_state= 42)
rf_model.fit(X_train_bal,y_train_bal)

y_pred = rf_model.predict(X_val_enc)

print('Acurracy:',accuracy_score(y_val,y_pred))
print('\nClassification Report:')
print(classification_report(y_val,y_pred,target_names=le.classes_))

print('\nConfusion Matrix')
print(confusion_matrix(y_val,y_pred))



from sklearn.metrics import accuracy_score

y_train_pred = rf_model.predict(X_train_bal)
train_acc = accuracy_score(y_train_bal, y_train_pred)


val_acc = accuracy_score(y_val, y_pred)

print(f"Train accuracy: {train_acc:.4f}")
print(f"Val   accuracy: {val_acc:.4f}")



from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [None, 5, 10, 20],
    'min_samples_leaf': [1, 10, 30],
    'max_features': ['sqrt', 0.5,'log2']
}

rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
grid = GridSearchCV(
    rf,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train_bal, y_train_bal)

print("Best params:", grid.best_params_)
best_rf = grid.best_estimator_

# Evaluate on validation
y_pred = best_rf.predict(X_val_enc)
print("Tuned Val Accuracy:", accuracy_score(y_val, y_pred))




X_test = test.copy().drop(columns=['id'])

X_test_enc = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)
X_test_enc[num_cols] = imp.transform(X_test_enc[num_cols])
y_test_pred = rf_model.predict(X_test_enc)

pred_labels = le.inverse_transform(y_test_pred)

submission = pd.DataFrame({
    'id': test['id'],
    'Personality': pred_labels
})
submission.to_csv('submission.csv', index=False)
print(submission.head())



submission.to_csv('/kaggle/working/submission.csv', index=False)
print("Saved to", '/kaggle/working/submission.csv')


