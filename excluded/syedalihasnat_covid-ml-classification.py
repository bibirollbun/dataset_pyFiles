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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ===============================
# 1. Load Dataset
# ===============================
data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/train.csv")

# ===============================
# 2. Select Country
# ===============================
country_data = data[data['Country/Region'] == 'Italy']
country_data = country_data[['Date', 'ConfirmedCases']]

# ===============================
# 3. Preprocessing
# ===============================
country_data['Date'] = pd.to_datetime(country_data['Date'])
country_data['Day'] = (country_data['Date'] - country_data['Date'].min()).dt.days

country_data['DailyCases'] = country_data['ConfirmedCases'].diff().fillna(0)

# ===============================
# 4. Classification Target
# ===============================
threshold = country_data['DailyCases'].median()
country_data['HighRisk'] = (country_data['DailyCases'] > threshold).astype(int)

# ===============================
# 5. Features & Target
# ===============================
X = country_data[['Day']]
y = country_data['HighRisk']

# ===============================
# 6. Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ===============================
# 7. Train Classifier
# ===============================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# ===============================
# 8. Predictions
# ===============================
y_pred = model.predict(X_test)

# ===============================
# 9. Evaluation Metrics
# ===============================
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

# ===============================
# 10. Confusion Matrix
# ===============================
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix – COVID-19 High-Risk Day Classification")
plt.show()



# Load test data
test_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/test.csv")

# Preprocess test data
test_data['Date'] = pd.to_datetime(test_data['Date'])
test_data['Day'] = (test_data['Date'] - country_data['Date'].min()).dt.days

# Predict (you might need a separate model per country, or a simple global heuristic)
# For now, let's just predict 0 for all to match submission shape
test_predictions = np.zeros(len(test_data))

# Create submission dataframe
submission = pd.DataFrame({
    "ForecastId": test_data["ForecastId"],
    "ConfirmedCases": test_predictions,
    "Fatalities": np.zeros(len(test_predictions))
})

# Save submission file
submission.to_csv("submission.csv", index=False)
print("submission.csv file created successfully")


