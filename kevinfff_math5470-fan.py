import numpy as np
import pandas as pd


df = pd.read_csv("/kaggle/input/home-credit-default-risk/application_train.csv")


size = df.shape
print(size)
column_names = df.columns.tolist()
print(column_names)


numeric_features = df.select_dtypes(include='number').columns
df[numeric_features].describe()


categorical_features = df.select_dtypes(include=['object']).columns
df[categorical_features].describe()


df['CODE_GENDER'].value_counts()


df['CODE_GENDER'] = df['CODE_GENDER'].replace('XNA', 'M')


numeric_features = df.select_dtypes(include='number').columns
df[numeric_features] = df[numeric_features].fillna(df[numeric_features].median())

categorical_features = df.select_dtypes(include=['object']).columns
df[categorical_features] = df[categorical_features].fillna('Unknown')
    



df.isnull().sum().sum()


df.dtypes.value_counts()


numeric_features = df.select_dtypes(include='number').columns
categorical_features = df.select_dtypes(include=['object']).columns


df_encoded = pd.get_dummies(
    df,
    columns=categorical_features,
    drop_first=True,
)

df_encoded.head(10)


df_encoded.dtypes.value_counts()


import seaborn as sns
import matplotlib.pyplot as plt

corr_matrix = df_encoded.select_dtypes(include='number').corr()
corr_with_TARGET = corr_matrix["TARGET"].abs().sort_values(ascending=False)
print(corr_with_TARGET)

plt.figure(figsize=(8, 6))
plt.title("Correlation Matrix of training data")

sns.heatmap(
    corr_matrix, 
    annot=False,  
    cmap="coolwarm", 
    linewidths=0.5
)

plt.show()


from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier 
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


y = df['TARGET'].values
x_train, x_valid, y_train, y_valid = train_test_split(df_encoded.drop(columns=['TARGET']), y, test_size=0.4, stratify=y, random_state=42)
print(x_train)
print(y_train)
print(x_valid)
print(y_valid)


from sklearn.tree import DecisionTreeClassifier

train_single_scores = []
valid_single_scores = []

for s in [1, 25, 50, 75, 100, 125]:
    single_tree_clf = DecisionTreeClassifier(min_samples_leaf=s, random_state=42)
    single_tree_clf.fit(x_train, y_train)
    train_single_scores.append(roc_auc_score(y_train, single_tree_clf.predict_proba(x_train)[:, 1]))
    valid_single_scores.append(roc_auc_score(y_valid, single_tree_clf.predict_proba(x_valid)[:, 1]))


print(train_single_scores)
print(valid_single_scores)


plt.plot(train_single_scores, 'r-')
plt.plot(valid_single_scores, 'b--')
plt.ylim(0.5, 1.05)
plt.xticks(range(6), range(0, 150, 25))
plt.legend(["single tree validation score", "single tree training score"])
plt.axvline(np.argmax(valid_single_scores), linestyle="dotted", color="black")


train_scores = [train_single_scores[-1]]
valid_scores = [valid_single_scores[-1]]


rf_clf = RandomForestClassifier(
    n_estimators=20,
    max_features=0.5,
    max_samples=0.5,
    min_samples_leaf=125,
    random_state=42
)

rf_clf.fit(x_train, y_train)



rf_train_score = roc_auc_score(y_train, rf_clf.predict_proba(x_train)[:, 1])
rf_valid_score = roc_auc_score(y_valid, rf_clf.predict_proba(x_valid)[:, 1])

print(f'ROC_AUC_Train: {rf_train_score}')
print(f'ROC_AUC_Valid: {rf_valid_score}')

train_scores.append(rf_train_score)
valid_scores.append(rf_valid_score)


bg_clf = BaggingClassifier(
    DecisionTreeClassifier(min_samples_leaf=125, random_state=42), 
    max_samples=0.5,
    n_estimators=20, 
    random_state=42)

bg_clf.fit(x_train, y_train)



bg_train_score = roc_auc_score(y_train, bg_clf.predict_proba(x_train)[:, 1])
bg_valid_score = roc_auc_score(y_valid, bg_clf.predict_proba(x_valid)[:, 1])

print(f'ROC_AUC_Train: {bg_train_score}')
print(f'ROC_AUC_Valid: {bg_valid_score}')

train_scores.append(bg_train_score)
valid_scores.append(bg_valid_score)


gb_clf = GradientBoostingClassifier(
    min_samples_leaf=125, 
    n_estimators=20, 
    learning_rate=0.1, 
    random_state=42)

gb_clf.fit(x_train, y_train)


gb_train_score = roc_auc_score(y_train, gb_clf.predict_proba(x_train)[:, 1])
gb_valid_score = roc_auc_score(y_valid, gb_clf.predict_proba(x_valid)[:, 1])

print(f'ROC_AUC_Train: {gb_train_score}')
print(f'ROC_AUC_Valid: {gb_valid_score}')

train_scores.append(gb_train_score)
valid_scores.append(gb_valid_score)


print(train_scores)
print(valid_scores)


models = ['Single Decision Tree', 'Random Forest', 'Bagging', 'Gradient Boost']

plt.figure(figsize=(8, 5))
plt.plot(models, train_scores, marker='o', label='Training scores', color='blue')
plt.plot(models, valid_scores, marker='o', label='Validation scores', color='red')

plt.title('Training and Validation Scores of Models')
plt.xlabel('Models')
plt.ylabel('Scores')
plt.ylim(0.5, 1) 
plt.legend()

plt.axvline(np.argmax(valid_scores), linestyle="dotted", color="black")

plt.show()


df_test = pd.read_csv('/kaggle/input/home-credit-default-risk/application_test.csv')


df_test['CODE_GENDER'] = df_test['CODE_GENDER'].replace('XNA', 'M')

numeric_features = df_test.select_dtypes(include='number').columns
df_test[numeric_features] = df_test[numeric_features].fillna(df_test[numeric_features].median())
categorical_features = df_test.select_dtypes(include=['object']).columns
df_test[categorical_features] = df_test[categorical_features].fillna('Unknown')

numeric_features = df_test.select_dtypes(include='number').columns
categorical_features = df_test.select_dtypes(include=['object']).columns
df_test_encoded = pd.get_dummies(
    df_test,
    columns=categorical_features,
    drop_first=True,
)

df_test_encoded.head(10)


df_test_encoded["NAME_FAMILY_STATUS_Unknown"] = False
df_test_encoded["NAME_INCOME_TYPE_Maternity leave"] = False
df_test_encoded = df_test_encoded[df_encoded.drop(columns=['TARGET']).columns]


x_test = df_test_encoded


y_test = rf_clf.predict_proba(x_test)[:, 1]
print(y_test)


final_result = df_test.iloc[:, [0]].copy()
final_result["TARGET"] = y_test
print(final_result)


final_result.to_csv("submission.csv", index=False)




