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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


data = {
    "income": [30000, 50000, 80000, 25000, 65000, 90000],
    "credit_score": [650, 700, 720, 580, 690, 750],
    "loan_amount": [200000, 350000, 500000, 150000, 400000, 600000],
    "eligible": [0, 1, 1, 0, 1, 1]
}

df = pd.DataFrame(data)
df


X = df[["income", "credit_score", "loan_amount"]]
y = df["eligible"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


def calculate_emi(principal, rate, tenure_months):
    r = rate / (12 * 100)
    emi = (principal * r) * ((1+r)**tenure_months) / ((1+r)**tenure_months - 1)
    return round(emi, 2)

calculate_emi(300000, 12, 36)


def loan_agent(income, credit_score, amount):
    scaled = scaler.transform([[income, credit_score, amount]])
    eligible = model.predict(scaled)[0]

    emi = calculate_emi(amount, 12, 36)

    return {
        "Eligible": "Yes" if eligible == 1 else "No",
        "Estimated EMI": emi,
        "Required Documents": [
            "Aadhaar Card",
            "PAN Card",
            "3 Months Salary Slip",
            "Bank Statement (6 months)"
        ]
    }

loan_agent(60000, 710, 300000)


df.to_csv("sample_data.csv", index=False)

