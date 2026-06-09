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


train_dataset = pd.read_csv("/kaggle/input/water-quality-classification/data.csv")
test_dataset = pd.read_csv("/kaggle/input/water-quality-classification/test.csv")


train_dataset.head()


test_dataset.head()


train_dataset.shape


test_dataset.shape


train_dataset.info()


test_dataset.info()


train_dataset.describe()


test_dataset.describe()


from preprocess_utils import missing_data, most_frequent_values, unique_values


missing_data(train_dataset)


missing_data(test_dataset)


most_frequent_values(train_dataset)


most_frequent_values(test_dataset)


unique_values(train_dataset)


unique_values(test_dataset)


from preprocess_utils import set_color_map, plot_distribution_pairs


dataset = pd.concat([train_dataset, test_dataset], axis=0)
dataset['set'] = 'train'
dataset.loc[dataset.Potability.isna(), 'set'] = 'test'
dataset.sample(10)


color_list = ["#A5D7E8", "#576CBC", "#19376D", "#0b2447"]
set_color_map(color_list)


plot_distribution_pairs(dataset, 'ph', 'ph', hue='set', color_list=color_list)


plot_distribution_pairs(dataset, 'ph', 'ph', hue='Potability', color_list=color_list)


plot_distribution_pairs(dataset, 'Sulfate', 'Sulfate', hue='set', color_list=color_list)


plot_distribution_pairs(dataset, 'Sulfate', 'Sulfate', hue='Potability', color_list=color_list)


plot_distribution_pairs(dataset, 'Trihalomethanes', 'Trihalomethanes', hue='set', color_list=color_list)


plot_distribution_pairs(dataset, 'Trihalomethanes', 'Trihalomethanes', hue='Potability', color_list=color_list)


X_train = train_dataset.iloc[:, :-1]
y_train = train_dataset.iloc[:, -1]
X_test = test_dataset.iloc[:, :-1]
test_id = test_dataset.iloc[:, -1]


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(random_state=42)
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)


columns = ['pH', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
           'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']

X_train = pd.DataFrame(X_train, columns=columns)
X_test = pd.DataFrame(X_test, columns=columns)

X_train['solids_per_hardness'] = X_train['Solids'] / (X_train['Hardness'] + 1e-3)
X_train['chloramine_sulfate_ratio'] = X_train['Chloramines'] / (X_train['Sulfate'] + 1e-3)
X_train['organic_density'] = X_train['Organic_carbon'] * X_train['Turbidity']
X_train['conductivity_scaled'] = X_train['Conductivity'] / (X_train['Solids'] + 1e-3)

X_test['solids_per_hardness'] = X_test['Solids'] / (X_test['Hardness'] + 1e-3)
X_test['chloramine_sulfate_ratio'] = X_test['Chloramines'] / (X_test['Sulfate'] + 1e-3)
X_test['organic_density'] = X_test['Organic_carbon'] * X_test['Turbidity']
X_test['conductivity_scaled'] = X_test['Conductivity'] / (X_test['Solids'] + 1e-3)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)


submission = pd.DataFrame({
    'id': test_id,
    'Potability': y_pred
})

submission.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")


submit = pd.read_csv("submission.csv")
submit.sample(10)

