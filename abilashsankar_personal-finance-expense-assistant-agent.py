!pip install scikit-learn --quiet

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

"Cell 1 loaded"



data = [
    {"date":"2025-10-01","description":"Starbucks latte","amount":4.50},
    {"date":"2025-10-01","description":"Uber trip to airport","amount":18.20},
    {"date":"2025-10-02","description":"Salary October","amount":1500.00},
    {"date":"2025-10-03","description":"Walmart groceries milk apples","amount":56.40},
    {"date":"2025-10-03","description":"Netflix subscription","amount":9.99},
    {"date":"2025-10-04","description":"Electricity bill","amount":72.10},
    {"date":"2025-10-05","description":"Bought Python book on Amazon","amount":24.99},
    {"date":"2025-10-06","description":"Gym monthly membership","amount":29.99},
    {"date":"2025-10-07","description":"Dinner at Olive Garden","amount":35.00},
    {"date":"2025-10-08","description":"Transfer to savings","amount":200.00},
    {"date":"2025-10-09","description":"Paid rent","amount":600.00},
    {"date":"2025-10-10","description":"Doctor consultation","amount":45.00},
]

df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df['date']).dt.date
df



labeled = [
    ("Starbucks latte", "Coffee"),
    ("Starbucks", "Coffee"),
    ("Uber trip", "Transport"),
    ("Uber", "Transport"),
    ("Salary", "Income"),
    ("Walmart groceries", "Groceries"),
    ("grocery", "Groceries"),
    ("Netflix subscription", "Entertainment"),
    ("Electricity bill", "Utilities"),
    ("Bought Python book on Amazon", "Education"),
    ("Gym membership", "Health"),
    ("Dinner restaurant", "Dining"),
    ("Paid rent", "Rent"),
    ("Doctor consultation", "Healthcare"),
    ("Transfer to savings", "Transfer"),
]

train_df = pd.DataFrame(labeled, columns=["description", "category"])
train_df



RULES = [
    (["starbucks","coffee","latte"], "Coffee"),
    (["uber","ola","taxi","cab"], "Transport"),
    (["walmart","grocery","groceries"], "Groceries"),
    (["netflix","spotify","hulu"], "Entertainment"),
    (["electricity","water bill","gas bill"], "Utilities"),
    (["rent","paid rent"], "Rent"),
    (["salary","payroll"], "Income"),
    (["transfer","savings"], "Transfer"),
    (["gym","fitness","membership"], "Health"),
    (["doctor","clinic","hospital"], "Healthcare"),
    (["book","amazon","course","udemy"], "Education"),
    (["dinner","restaurant"], "Dining"),
]

def rule_based(desc):
    d = desc.lower()
    for keywords, category in RULES:
        for kw in keywords:
            if kw in d:
                return category
    return None

"Cell 4 loaded"



le = LabelEncoder()
y = le.fit_transform(train_df['category'])
X = train_df['description']

pipeline = make_pipeline(
    TfidfVectorizer(ngram_range=(1,2), max_features=1000),
    LogisticRegression(max_iter=1000)
)

pipeline.fit(X, y)

classification_report(y, pipeline.predict(X), target_names=le.classes_)



def categorize(row):
    desc = row['description']
    # Rule-based first
    r = rule_based(desc)
    if r:
        return r, "rule"
    # ML fallback
    pred = pipeline.predict([desc])[0]
    return le.inverse_transform([pred])[0], "model"

df['category'], df['source'] = zip(*df.apply(categorize, axis=1))
df



summary = df.groupby("category").agg(
    total_spent=("amount","sum"),
    count=("amount","count")
).reset_index().sort_values("total_spent", ascending=False)

summary


plt.figure(figsize=(8,4))
plt.bar(summary['category'], summary['total_spent'])
plt.title("Total Spent by Category")
plt.ylabel("Amount")
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()



