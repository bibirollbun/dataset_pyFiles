from kaggle_secrets import UserSecretsClient
import google.generativeai as genai

# Get your API key from Kaggle Secrets
user_secrets = UserSecretsClient()
google_api_key = user_secrets.get_secret("GOOGLE_API_KEY")

# Configure Gemini
genai.configure(api_key=google_api_key)

print("Gemini API is configured ✅")


import os
import pandas as pd

PRODUCTS_PATH = "/kaggle/working/products.csv"
ORDERS_PATH = "/kaggle/working/orders.csv"
EARNINGS_PATH = "/kaggle/working/earnings.csv"

def init_files():
    # Products – only create once
    if not os.path.exists(PRODUCTS_PATH):
        products = pd.DataFrame({
            'name': [
                'Rice 1kg', 'Wheat Flour 1kg', 'Sugar 1kg', 'Oil 1L', 'Salt 1kg',
                'Tur Dal 1kg', 'Moong Dal 1kg', 'Tea Powder 250g', 'Coffee 50g',
                'Milk 1L', 'Bread Loaf', 'Eggs (12 pack)', 'Butter 100g',
                'Biscuits Pack', 'Chips Pack', 'Shampoo Sachet', 'Soap Bar',
                'Detergent Powder 1kg', 'Toothpaste 100g', 'Soft Drink 1L',
                'Onion 1kg', 'Potato 1kg', 'Tomato 1kg'
            ],
            'price': [
                50, 40, 45, 120, 20,
                90, 100, 60, 80,
                55, 30, 70, 45,
                20, 15, 2, 25,
                60, 55, 40,
                35, 25, 30
            ],
            'quantity': [
                25, 30, 40, 20, 50,
                15, 18, 12, 10,
                20, 15, 10, 20,
                30, 25, 50, 25,
                18, 22, 15,
                40, 35, 25
            ]
        })
        products.to_csv(PRODUCTS_PATH, index=False)
    
    # Empty orders file (create only if not there)
    if not os.path.exists(ORDERS_PATH):
        orders = pd.DataFrame(columns=['order_id', 'customer_name', 'product_name', 'quantity', 'total_price'])
        orders.to_csv(ORDERS_PATH, index=False)

    # Empty earnings file (create only if not there)
    if not os.path.exists(EARNINGS_PATH):
        earnings = pd.DataFrame(columns=['date', 'amount', 'type'])
        earnings.to_csv(EARNINGS_PATH, index=False)

# Run once to make sure files exist
init_files()

# Show products
pd.read_csv(PRODUCTS_PATH).head()



import pandas as pd

# --- LOAD & SAVE HELPERS ---

def load_products():
    return pd.read_csv(PRODUCTS_PATH)

def save_products(df):
    df.to_csv(PRODUCTS_PATH, index=False)

def load_orders():
    return pd.read_csv(ORDERS_PATH)

def save_orders(df):
    df.to_csv(ORDERS_PATH, index=False)

def load_earnings():
    df = pd.read_csv(EARNINGS_PATH)
    # Make sure date is treated as date
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df

def save_earnings(df):
    df.to_csv(EARNINGS_PATH, index=False)


# --- TOOL 1: CHECK STOCK ---
def check_stock(product_name):
    products = load_products()
    item = products[products['name'].str.lower() == product_name.lower()]
    
    if item.empty:
        return f"'{product_name}' is not found in your shop."
    
    qty = int(item.iloc[0]['quantity'])
    price = int(item.iloc[0]['price'])
    return f"You have {qty} units of {product_name} in stock (₹{price}/unit)."


# --- TOOL 2: ADD PRODUCT ---
def add_product(name, price, quantity):
    products = load_products()
    new_row = pd.DataFrame([[name, price, quantity]], 
                           columns=['name', 'price', 'quantity'])
    products = pd.concat([products, new_row], ignore_index=True)
    save_products(products)
    return f"Added: {name} – Price ₹{price}, Qty {quantity}"


# --- TOOL 3: PLACE ORDER (this updates earnings!) ---
def place_order(customer_name, product_name, quantity):
    products = load_products()
    
    item = products[products['name'].str.lower() == product_name.lower()]
    if item.empty:
        return f"'{product_name}' is not available."

    stock = int(item.iloc[0]['quantity'])
    price = int(item.iloc[0]['price'])

    if stock < quantity:
        return f"Not enough stock. Only {stock} left."

    total_price = price * quantity

    # Update stock
    products.loc[products['name'].str.lower() == product_name.lower(), 'quantity'] -= quantity
    save_products(products)

    # Add order
    orders = load_orders()
    new_order = pd.DataFrame([[len(orders) + 1, customer_name, product_name, quantity, total_price]],
                             columns=['order_id', 'customer_name', 'product_name', 'quantity', 'total_price'])
    orders = pd.concat([orders, new_order], ignore_index=True)
    save_orders(orders)

    # Log earnings
    earnings = load_earnings()
    today = pd.Timestamp.today().date()
    new_earning = pd.DataFrame([[today, total_price, 'sale']],
                               columns=['date', 'amount', 'type'])
    earnings = pd.concat([earnings, new_earning], ignore_index=True)
    save_earnings(earnings)

    return f"Order placed for {customer_name}: {quantity} x {product_name} = ₹{total_price}"


# --- TOOL 4: RECORD EXPENSE ---
def record_expense(amount, description):
    earnings = load_earnings()
    today = pd.Timestamp.today().date()
    new_exp = pd.DataFrame([[today, -abs(amount), description]],
                           columns=['date', 'amount', 'type'])
    earnings = pd.concat([earnings, new_exp], ignore_index=True)
    save_earnings(earnings)
    return f"Recorded expense: ₹{amount} for {description}"


# --- TOOL 5: DAILY SUMMARY ---
def daily_summary():
    earnings = load_earnings()
    today = pd.Timestamp.today().date()
    today_data = earnings[earnings['date'] == today]
    total = today_data['amount'].sum()

    return f"Today's net amount: ₹{total}\n\nDetails:\n{today_data.to_string(index=False)}"



import google.generativeai as genai

SYSTEM_PROMPT = """
You are DukanMitra, an AI agent that helps small shop owners manage their business.

You can:
- Check stock
- Add new products
- Place customer orders
- Record expenses
- Summarize daily earnings

Your job:
- Speak simply and clearly (like talking to a shop owner)
- When needed, reason using the shop data I give you
- If the user asks for something not in the shop data, give suggestions

Always return:
1. Your reply to the user (in simple language)
"""

agent_model = genai.GenerativeModel(
    "models/gemini-pro-latest",
    system_instruction=SYSTEM_PROMPT
)



def safe_get_text(response):
    # Safer way to get text from the model response
    try:
        if hasattr(response, "text") and response.text:
            return response.text
    except Exception:
        pass

    if getattr(response, "candidates", None):
        parts = response.candidates[0].content.parts
        for p in parts:
            if getattr(p, "text", None):
                return p.text

    return "(No text returned by model)"

def ask_agent(message):
    # Load limited data for context
    products = load_products()
    orders = load_orders()
    earnings = load_earnings()

    context = f"""
PRODUCTS DATA (first 10 rows):
{products.head(10).to_markdown(index=False)}

ORDERS DATA (first 10 rows):
{orders.head(10).to_markdown(index=False)}

EARNINGS DATA (first 10 rows):
{earnings.tail(10).to_markdown(index=False)}
"""

    prompt = f"{context}\n\nUser: {message}\nAssistant:"
    response = agent_model.generate_content(prompt)
    print(safe_get_text(response))



print(place_order("Rohan", "Wheat Flour 1kg", 2))



load_earnings().tail()



ask_agent("How much money did I earn today?")


