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


# Create a small example product dataset for the Smart Shopping Assistant
import pandas as pd

data = [
    {"id": 1, "name": "Comfy Running Shoes", "category": "Shoes", "brand": "RunFast", "price": 49.99, "rating": 4.3, "stock": 25},
    {"id": 2, "name": "Classic White Shirt", "category": "Clothing", "brand": "Elegance", "price": 29.99, "rating": 4.6, "stock": 40},
    {"id": 3, "name": "Waterproof Backpack 20L", "category": "Bags", "brand": "TravelPro", "price": 39.50, "rating": 4.1, "stock": 10},
    {"id": 4, "name": "Wireless Earbuds X", "category": "Electronics", "brand": "SoundMax", "price": 79.99, "rating": 4.0, "stock": 5},
    {"id": 5, "name": "LED Desk Lamp", "category": "Home", "brand": "BrightLite", "price": 19.99, "rating": 4.4, "stock": 60},
    {"id": 6, "name": "Stainless Water Bottle 750ml", "category": "Home", "brand": "PureSip", "price": 15.00, "rating": 4.8, "stock": 120},
]

df = pd.DataFrame(data)
df.to_csv('products.csv', index=False)
print("Saved products.csv — here are the rows:")
print(df)


import pandas as pd

df = pd.read_csv('products.csv')
df


# Step 6 - Smart Shopping Assistant: search, filter, compare, shopping list
import pandas as pd

# Load products (created earlier)
df = pd.read_csv('products.csv')
df

# --- Utility functions for the Smart Shopping Assistant ---

def search_products(df, query):
    """Return products whose name or brand contains the query (case-insensitive)."""
    q = str(query).strip().lower()
    mask = df['name'].str.lower().str.contains(q) | df['brand'].str.lower().str.contains(q)
    return df[mask].reset_index(drop=True)

def filter_products(df, category=None, brand=None, price_min=None, price_max=None, min_rating=None):
    """Filter products by category, brand, price range and minimum rating."""
    res = df.copy()
    if category:
        res = res[res['category'].str.lower() == category.strip().lower()]
    if brand:
        res = res[res['brand'].str.lower() == brand.strip().lower()]
    if price_min is not None:
        res = res[res['price'] >= float(price_min)]
    if price_max is not None:
        res = res[res['price'] <= float(price_max)]
    if min_rating is not None:
        res = res[res['rating'] >= float(min_rating)]
    return res.reset_index(drop=True)

def compare_products(df, ids):
    """Return a comparison table for given product ids (list of ints)."""
    comp = df[df['id'].isin(ids)].copy()
    comp = comp[['id','name','brand','category','price','rating','stock']]
    return comp.reset_index(drop=True)

def build_shopping_list(df, ids):
    """Given product ids, create shopping list and total estimate."""
    items = df[df['id'].isin(ids)].copy()
    total = items['price'].sum()
    return items.reset_index(drop=True), float(total)

def recommend_product(df, category=None, budget=None):
    """
    Simple recommendation: choose the highest-rated product in the category
    that is within budget (if provided), breaking ties by lower price.
    """
    cand = df.copy()
    if category:
        cand = cand[cand['category'].str.lower() == category.strip().lower()]
    if budget is not None:
        cand = cand[cand['price'] <= float(budget)]
    if cand.empty:
        return None
    # sort by rating desc then price asc
    cand = cand.sort_values(['rating','price'], ascending=[False, True]).reset_index(drop=True)
    return cand.iloc[0].to_dict()

# --- Example usage: try these in the notebook ---
print("All products\n", df, "\n")

print("Search for 'water' ->")
print(search_products(df, "water"), "\n")

print("Filter: category 'Home', rating >= 4.5 ->")
print(filter_products(df, category="Home", min_rating=4.5), "\n")

print("Compare IDs [1,4] ->")
print(compare_products(df, [1,4]), "\n")

print("Build shopping list for IDs [2,6] ->")
items, total = build_shopping_list(df, [2,6])
print(items)
print("Estimated total:", total, "\n")

print("Recommend best product in category 'Home' under budget 20 ->")
print(recommend_product(df, category="Home", budget=20))


# Smart Shopping Assistant — search, filter, compare, build shopping list
import pandas as pd

# Load the product file we created earlier
df = pd.read_csv('products.csv')

# --- Functions ---
def search_products(df, query=None):
    """Simple text search in product name and brand."""
    if not query:
        return df.copy()
    q = query.lower()
    mask = df['name'].str.lower().str.contains(q) | df['brand'].str.lower().str.contains(q)
    return df[mask].reset_index(drop=True)

def filter_products(df, category=None, brand=None, price_min=None, price_max=None, rating_min=None, in_stock_only=False):
    """Filter by multiple optional criteria."""
    res = df.copy()
    if category:
        res = res[res['category'].str.lower() == category.lower()]
    if brand:
        res = res[res['brand'].str.lower() == brand.lower()]
    if price_min is not None:
        res = res[res['price'] >= price_min]
    if price_max is not None:
        res = res[res['price'] <= price_max]
    if rating_min is not None:
        res = res[res['rating'] >= rating_min]
    if in_stock_only:
        res = res[res['stock'] > 0]
    return res.reset_index(drop=True)

def compare_products(df, ids):
    """Return a side-by-side comparison dataframe for product ids (list)."""
    sel = df[df['id'].isin(ids)].set_index('id')
    # Reorder columns for nicer display
    cols = ['name','category','brand','price','rating','stock']
    return sel[cols]

def build_shopping_list(df, ids):
    """Create shopping list summary and estimated total price for given ids."""
    sel = df[df['id'].isin(ids)].copy()
    sel['qty'] = 1  # default 1 each — you can change later
    sel['line_total'] = sel['price'] * sel['qty']
    total = sel['line_total'].sum()
    summary = {
        'items': sel[['id','name','brand','price','qty','line_total']].to_dict(orient='records'),
        'estimated_total': round(total, 2),
        'num_items': len(sel)
    }
    return summary

def recommendation_text(summary):
    """Generate a short user-friendly recommendation string from shopping-list summary."""
    lines = []
    lines.append(f"Shopping list: {summary['num_items']} item(s). Estimated total: ${summary['estimated_total']}")
    for it in summary['items']:
        lines.append(f"- {it['name']} ({it['brand']}) — ${it['price']} x{it['qty']} = ${round(it['line_total'],2)}")
    return "\n".join(lines)

# --- Example usage (run these lines or call functions in your own cells) ---
print("All products:")
print(df.to_string(index=False))

print("\nSearch for 'water':")
print(search_products(df, 'water').to_string(index=False))

print("\nFilter: category='Home', price_max=20:")
print(filter_products(df, category='Home', price_max=20).to_string(index=False))

print("\nCompare products with ids [1,4]:")
print(compare_products(df, [1,4]).to_string())

print("\nBuild shopping list for ids [2,6,5]:")
summary = build_shopping_list(df, [2,6,5])
print(recommendation_text(summary))

# Save useful outputs (optional)
df.to_csv('products_full.csv', index=False)


# --------------------------
# Simple Text-Based Shopping Agent
# --------------------------

def shopping_agent(query):
    """
    Very simple text interface for our shopping agent.
    The model interprets the query and calls the right function.
    """

    q = query.lower()

    # --- Search ---
    if "search" in q:
        # extract keyword
        keyword = q.replace("search", "").strip()
        results = search_products(df, keyword)
        return results.to_string(index=False)

    # --- Filter by category ---
    elif "category" in q:
        parts = q.split("category")
        cat = parts[1].strip()
        results = filter_products(df, category=cat)
        return results.to_string(index=False)

    # --- Cheap / expensive filters ---
    elif "under" in q:
        # e.g., "show products under 30"
        price_max = float(q.split("under")[1].strip())
        results = filter_products(df, price_max=price_max)
        return results.to_string(index=False)

    # --- Compare products ---
    elif "compare" in q:
        # e.g., "compare 1 and 4"
        nums = [int(x) for x in q.replace("compare", "").replace("and", " ").split() if x.isdigit()]
        results = compare_products(df, nums)
        return results.to_string()

    # --- Build shopping list ---
    elif "shopping list" in q or "build list" in q:
        nums = [int(x) for x in q.split() if x.isdigit()]
        summary = build_shopping_list(df, nums)
        return recommendation_text(summary)

    else:
        return "I can help with search, filter, compare, and shopping lists. Try: 'search shoes', 'category home', 'under 30', 'compare 1 and 4', or 'build list 2 5 6'."


# Save example outputs as a text file for the capstone submission
demo_text = """
Demo Output for Smart Shopping Assistant

Example 1: Search
-----------------
""" + shopping_agent("search water") + """

Example 2: Category Filter
--------------------------
""" + shopping_agent("category home") + """

Example 3: Compare Products
---------------------------
""" + shopping_agent("compare 1 and 4") + """

Example 4: Build Shopping List
------------------------------
""" + shopping_agent("build list 2 5 6")

with open("demo_output.txt", "w") as f:
    f.write(demo_text)

print("Saved demo_output.txt")

