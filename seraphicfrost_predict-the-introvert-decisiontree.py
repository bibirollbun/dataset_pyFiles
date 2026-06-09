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


train_data  = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv", index_col='id')
test_data = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv", index_col='id')


# train_data = train_data[['Drained_after_socializing', 'Stage_fear', 'Post_frequency', 'Social_event_attendance', 'Personality']]
# test_data = test_data[['Drained_after_socializing', 'Stage_fear', 'Post_frequency', 'Social_event_attendance']]
# train_data.head()

train_data = train_data[['Drained_after_socializing', 'Stage_fear', 'Post_frequency', 'Personality']]
test_data = test_data[['Drained_after_socializing', 'Stage_fear', 'Post_frequency' ]]
train_data.head()


from sklearn.discriminant_analysis import StandardScaler

scaler = StandardScaler()
num_cols = list(train_data.select_dtypes(exclude=['object']).columns)

train_data[num_cols] = scaler.fit_transform(train_data[num_cols])
test_data[num_cols] = scaler.transform(test_data[num_cols])
test_data.head()


X = train_data.drop('Personality', axis=1)
y = train_data['Personality']
y


X_encoded = pd.get_dummies(X, columns=["Stage_fear", "Drained_after_socializing"])
y_labeled = y.apply(lambda x: 0 if x == "Extrovert" else 1)


from sklearn.impute import KNNImputer


imputer = KNNImputer()
X_imputed = pd.DataFrame(imputer.fit_transform(X_encoded), columns=X_encoded.columns)
X_imputed.head()


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y_labeled, test_size=0.2, random_state=42
);


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


model_dt_not_pruned = DecisionTreeClassifier(random_state=42)
model_dt_not_pruned.fit(X_train, y_train)
model_dt_not_pruned.score(X_test, y_test)

cv_score = cross_val_score(model_dt_not_pruned, X_train, y_train, cv=20)
print(f"accuracy-{model_dt_not_pruned.score(X_test, y_test)}", f"cv-score: {cv_score.mean()}")


from matplotlib import pyplot as plt

path = model_dt_not_pruned.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
ccp_alphas = ccp_alphas[:-1]

clf_dts = []

for ccp_alpha in ccp_alphas:
	clf_dt = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
	clf_dt.fit(X_train, y_train)
	clf_dts.append(clf_dt)

	

pd.DataFrame(path)


train_scores = [clf_dt.score(X_train, y_train) for clf_dt in clf_dts]
test_scores = [clf_dt.score(X_test, y_test) for clf_dt in clf_dts]



from matplotlib.ticker import FormatStrFormatter

plt.plot(ccp_alphas, train_scores, label='Train Accuracy', color='royalblue', linewidth=2)
plt.plot(ccp_alphas, test_scores, label='Test Accuracy', color='darkorange', linewidth=2)

plt.xscale('log')  # Log scale for better spread of small alpha values
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.5f'))
plt.xlabel('Alpha (log scale)', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


from sklearn.model_selection import cross_val_score

alpha_loop_values = []

for ccp_alpha in ccp_alphas:
    clf_dt = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    scores = cross_val_score(clf_dt, X_train, y_train, cv=10)

    alpha_loop_values.append([ccp_alpha, np.mean(scores), np.std(scores)])
 
alpha_results = pd.DataFrame(alpha_loop_values, columns=['ccp_alpha', 'mean_accuracy', 'std'])


alpha_results.plot(x='ccp_alpha', y='mean_accuracy', yerr='std', marker='o', linestyle='--', logx=True  )


# ccp alpha with highest mean accuracy on cross_val

optimal_ccp_alpha = alpha_results.loc[alpha_results['mean_accuracy'].idxmax(), 'ccp_alpha']
print(optimal_ccp_alpha)

clf_dt_pruned = DecisionTreeClassifier(random_state=4)
clf_dt_pruned.fit(X_train, y_train)


cv_score = cross_val_score(clf_dt_pruned, X_train, y_train, cv=10)
print(f"accuracy-{clf_dt_pruned.score(X_test, y_test)}", f"cv-score: {cv_score.mean()}")


test_data.head()


test_data_encoded = pd.get_dummies(test_data, columns=["Stage_fear", "Drained_after_socializing"])

imputer = KNNImputer()
test_data_imputed = pd.DataFrame(imputer.fit_transform(test_data_encoded), columns=test_data_encoded.columns)
test_data_imputed.head()


test_preds = clf_dt_pruned.predict(test_data_imputed)

# Mapping 0 & 1 to Extrovert & Introvert
label_map = {0: 'Extrovert', 1: 'Introvert'}
test_preds_mapped = [label_map[pred] for pred in test_preds]

ids = test_data.index

output = pd.DataFrame({'id': ids,
				   'Personality': test_preds_mapped})
output.to_csv('submission.csv', index=False)

output

