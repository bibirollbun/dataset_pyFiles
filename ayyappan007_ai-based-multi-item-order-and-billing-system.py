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


!pip install gspread google-auth



from kaggle_secrets import UserSecretsClient
import json
secret_client = UserSecretsClient()
service_account_info = secret_client.get_secret("GOOGLE_SERVICE_ACCOUNT")
service_account_dict = json.loads(service_account_info)
client = gspread.authorize(creds)


from kaggle_secrets import UserSecretsClient
import json
from google.oauth2.service_account import Credentials
import gspread

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

secret_client = UserSecretsClient()
spreadsheet_id = secret_client.get_secret("SPREADSHEET_ID")
service_account_str = secret_client.get_secret("GOOGLE_SERVICE_ACCOUNT")
service_account_info = json.loads(service_account_str)

creds = Credentials.from_service_account_info(service_account_info, scopes=SCOPES)
client = gspread.authorize(creds)

sheet = client.open_by_key(spreadsheet_id).sheet1





name = input("Customer Name: ")

ITEM_PRICES = {
    "Milk": 40,
    "Egg": 8,
    "Apple": 30,
    "Bread": 20,
    "Banana": 7,
    "Rice": 1800,
}

order_items = []
print("\nAvailable items and prices:\n")
for item, price in ITEM_PRICES.items():
    print(f"  {item:10} : ₹{price}")
n = int(input("How many different items do you want?"))
for i in range(n):
    item = input(f"Item {i+1}: ")
    quantity = int(input(f"Quantity for {item}: "))
    unit_price = ITEM_PRICES.get(item, 0)
    total = unit_price * quantity
    order_items.append({"item": item, "quantity": quantity, "unit_price": unit_price, "total": total})

from datetime import datetime
order_date = datetime.now().strftime("%d-%m-%Y")
grand_total = sum(x['total'] for x in order_items)

def save_order_to_sheet(name, order_items, grand_total):
   
    for x in order_items:
        row = [name, x['item'], x['quantity'], x['total']]
        sheet.append_row(row)
   
    sheet.append_row([name, "Total", "", grand_total])
    print(f"Order saved to Google Sheet for: {name}")

save_order_to_sheet(name, order_items, grand_total)

item_lines = []
for x in order_items:
    item_lines.append(f"{x['item']} x {x['quantity']} units at ₹{x['unit_price']} each (₹{x['total']})")
items_text = "\n".join(item_lines)

order_desc = (
    "Please generate a polite, easy-to-read bill receipt for the customer in the following style:\n"
    "Bill Receipt\n"
    f"Date: {order_date}\n"
    f"Customer: {name}\n"
    "---------------------------\n"
    "Items Purchased:\n"
    + "\n".join([f"- {x['item']} Qty: {x['quantity']} x ₹{x['unit_price']} = ₹{x['total']}" for x in order_items]) +
    "\n---------------------------\n"
    f"Total Amount: ₹{grand_total}\n"
    "---------------------------\n"
    "Thank you for choosing our store, {name}. We appreciate your business. Have a great day!\n"
    "Please visit again!"
)

from kaggle_secrets import UserSecretsClient
import requests

GROQ_API_KEY = UserSecretsClient().get_secret("GROQ_API_KEY")
MODEL = "openai/gpt-oss-120b"  # Use Groq's best supported model

url = "https://api.groq.com/openai/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}
payload = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": "You are an English bill assistant. Always reply with a receipt in professional, friendly tone. Close with a thank-you and friendly wishes."},
        {"role": "user", "content": order_desc}
    ],
    "max_tokens": 250,
    "temperature": 0.6
}

response = requests.post(url, headers=headers, json=payload)
ai_reply = response.json()["choices"][0]["message"]["content"]
print("\nGroq AI Response:\n", ai_reply)




