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


# ðŸ§¾ AI-Powered Finance Reconciliation & Insights Agent  

This notebook performs

- Invoice vs Payment reconciliation  
- Anomaly detection  
- Finance insights generation  
- Automated summary reporting  

It simulates how an Agent would process financial data end-to-end.



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("default")



# Sample Invoice Data
invoices = pd.DataFrame({
    "invoice_id": ["INV001","INV002","INV003","INV004","INV005"],
    "customer": ["A","B","A","C","B"],
    "amount": [1200, 500, 900, 300, 700],
    "date": ["2024-01-05","2024-01-07","2024-01-10","2024-01-11","2024-01-12"]
})

# Sample Payment data
payments = pd.DataFrame({
    "payment_id": ["PMT01","PMT02","PMT03","PMT04"],
    "invoice_id": ["INV001","INV003","INV005","INV006"],  
    "amount_paid": [1200, 800, 700, 200],
    "date": ["2024-01-08","2024-01-12","2024-01-13","2024-01-13"]
})

invoices, payments



recon = invoices.merge(payments, on="invoice_id", how="left")

recon["payment_status"] = recon.apply(
    lambda x: "Paid" if x["amount"] == x["amount_paid"] else
              ("Underpaid" if pd.notna(x["amount_paid"]) else "Unpaid"),
    axis=1
)

recon



summary = recon.groupby("payment_status").size().reset_index(name="count")
summary



anomalies = recon[
    (recon["payment_status"] != "Paid") & 
    (recon["amount_paid"].notna())
]

anomalies



total_invoiced = invoices["amount"].sum()
total_paid = payments["amount_paid"].sum()
pending_amount = total_invoiced - total_paid

insights = {
    "Total Invoiced": total_invoiced,
    "Total Paid": total_paid,
    "Pending Amount": pending_amount,
    "Fully Paid Invoices": int((recon["payment_status"] == "Paid").sum()),
    "Unpaid Invoices": int((recon["payment_status"] == "Unpaid").sum())
}

insights



plt.figure(figsize=(6,4))
sns.barplot(x=list(insights.keys())[:3], y=list(insights.values())[:3])
plt.title("Finance Summary")
plt.xticks(rotation=45)
plt.show()



plt.figure(figsize=(6,4))
sns.countplot(data=recon, x="payment_status")
plt.title("Payment Status Distribution")
plt.show()



report_path = "/mnt/data/finance_reconciliation_report.xlsx"
with pd.ExcelWriter(report_path) as writer:
    invoices.to_excel(writer, sheet_name="Invoices", index=False)
    payments.to_excel(writer, sheet_name="Payments", index=False)
    recon.to_excel(writer, sheet_name="Reconciliation", index=False)
    anomalies.to_excel(writer, sheet_name="Anomalies", index=False)

report_path




