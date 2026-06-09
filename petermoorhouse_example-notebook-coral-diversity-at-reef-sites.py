import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import hamming_loss, classification_report
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn import tree


train_data_path = '/kaggle/input/coral-diversity-at-reef-sites/train.csv'
test_data_path = '/kaggle/input/coral-diversity-at-reef-sites/test.csv'
df_comp_train = pd.read_csv(train_data_path)
df_comp_test = pd.read_csv(test_data_path)

label_cols = [col for col in df_comp_train.columns if col.startswith('species_')]
feature_cols = [col for col in df_comp_train.columns if col not in label_cols and col != 'id']

X = df_comp_train[feature_cols]
y = df_comp_train[label_cols]

categorical_cols = X.select_dtypes(include=['object']).columns
numeric_cols = X.select_dtypes(include=[np.number]).columns

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y.sum(axis=1)
)


external_data_path = '/kaggle/input/coral-reef-sites/coral_reef_sites.csv'
df_external = pd.read_csv(external_data_path)

X_ext = df_external[feature_cols]
y_ext = df_external[label_cols]

X_train = pd.concat([X_train, X_ext], ignore_index=True)
y_train = pd.concat([y_train, y_ext], ignore_index=True)


num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='most_frequent')
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

X_train_num = num_imputer.fit_transform(X_train[numeric_cols])
X_train_cat = cat_imputer.fit_transform(X_train[categorical_cols])
encoder.fit(X_train_cat)

def preprocess_features(X_raw):
    X_num = num_imputer.transform(X_raw[numeric_cols])
    X_cat = cat_imputer.transform(X_raw[categorical_cols])
    X_cat_encoded = encoder.transform(X_cat)
    return np.hstack([X_num, X_cat_encoded])

X_train_final = preprocess_features(X_train)
X_val_final = preprocess_features(X_val)


base_tree = DecisionTreeClassifier(max_depth=2, random_state=42)
model = MultiOutputClassifier(base_tree)

model.fit(X_train_final, y_train)


y_val_pred = model.predict(X_val_final)

print("Hamming Loss:", hamming_loss(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred, target_names=label_cols))


species_index = label_cols.index('species_Acropora')

plt.figure(figsize=(12, 8))
tree.plot_tree(model.estimators_[species_index],
               filled=True,
               feature_names=list(numeric_cols) + list(encoder.get_feature_names_out(categorical_cols)),
               class_names=["Absent", "Present"])
plt.title("Decision Tree for species_Acropora")
plt.show()


X_comp_test = df_comp_test[feature_cols]
X_comp_test = preprocess_features(X_comp_test)
y_pred_test = model.predict(X_comp_test)


submission = pd.DataFrame(y_pred_test, columns=label_cols)
submission.insert(0, 'id', df_comp_test['id'])

submission.to_csv('submission.csv', index=False)
submission.head()

