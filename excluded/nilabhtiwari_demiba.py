import pandas as pd
import random

# Define brands per category
phones_brands = ['Xiaomi','Samsung','Apple','Realme','Poco','OnePlus','Vivo','Oppo']
laptops_brands = ['Dell','HP','Lenovo','Asus','Acer','Apple']
smartwatch_brands = ['Apple','Samsung','Fitbit','Garmin','Realme']
earbuds_brands = ['Sony','Bose','JBL','Boat','OnePlus','Realme']

# Function to generate product rows
def generate_product(category, brands, start_id, n):
    products = []
    for i in range(n):
        brand = random.choice(brands)
        name = f"{brand} {category} {i+1}"
        price = random.randint(1000, 200000)  # adjust per category if needed
        availability = random.choice(['In stock','Out of stock'])
        battery = f"{random.choice([150,200,250,300,4000,4500,5000,7000])} mAh"
        camera = random.choice(['No','12 MP','16 MP Dual','48 MP Triple','108 MP'])
        ram = f"{random.choice([512,1024,2048,4,6,8,12,16,32,64])} GB"
        storage = f"{random.choice([4,8,16,32,64,128,256,512,1024,2048])} GB"
        display = random.choice(['No','1.2 inch LCD','1.5 inch AMOLED','6.5 inch AMOLED','15 inch IPS','17 inch LCD'])
        processor = random.choice(['Snapdragon 732G','Snapdragon 888','Intel i5','Intel i7','Apple A15','MediaTek Dimensity'])
        connectivity = ','.join(random.sample(['5G','4G','Wi-Fi 6','Bluetooth 5.0','NFC','LTE','Ethernet'],3))
        waterproof = random.choice(['Yes','No'])
        flashlight = random.choice(['LED','No'])
        port = random.choice(['Type-C','Micro','Lightning','USB 3.0','USB 2.0','HDMI','None'])
        speaker = random.choice(['Mono','Stereo','Dolby','High Bass'])
        weight = f"{random.randint(5,2500)}g"
        color = random.choice(['Black','White','Blue','Red','Silver','Green','Gray'])
        os = random.choice(['Android','iOS','Windows 11','Windows 10','MacOS','WatchOS','WearOS','Tizen','None'])
        features = f"Battery:{battery}; Camera:{camera}; RAM:{ram}; Storage:{storage}; Display:{display}; Processor:{processor}; Connectivity:{connectivity}; Waterproof:{waterproof}; Flashlight:{flashlight}; Port:{port}; Speaker:{speaker}; Weight:{weight}; Color:{color}; OS:{os}"
        products.append([start_id+i+1, name, brand, category, price, availability, features, f"local:{brand.lower()}{start_id+i+1}"])
    return products

# Generate products
phones = generate_product('Phone', phones_brands, 0, 30)
laptops = generate_product('Laptop', laptops_brands, 30, 20)
smartwatches = generate_product('Smartwatch', smartwatch_brands, 50, 15)
earbuds = generate_product('Earbuds', earbuds_brands, 65, 15)

all_products = phones + laptops + smartwatches + earbuds

# Create DataFrame
df_products = pd.DataFrame(all_products, columns=['id','name','brand','category','price','availability','features','source_url'])

# Save CSV
df_products.to_csv("nexusscout_products_80.csv", index=False)

print("CSV created with 80 products: nexusscout_products_80.csv")



import pandas as pd

df = pd.read_csv("/kaggle/working/nexusscout_products_80.csv")
df.head()



def nexus_scout(query):
    query = query.lower()
    results = df.copy()
    
    # Filters
    if "phone" in query:
        results = results[results['category'] == "Phone"]
    if "laptop" in query:
        results = results[results['category'] == "Laptop"]
    if "earbuds" in query:
        results = results[results['category'] == "Earbuds"]
    if "smartwatch" in query:
        results = results[results['category'] == "Smartwatch"]

    # Price filters
    import re
    price_match = re.search(r"under ?₹?(\d+)", query)
    if price_match:
        price_limit = int(price_match.group(1))
        results = results[results["price"] < price_limit]

    # Cheapest
    if "cheapest" in query:
        return results.sort_values("price").head(1)

    # Expensive
    if "most expensive" in query or "costliest" in query:
        return results.sort_values("price", ascending=False).head(1)

    # Compare two products
    if "compare" in query:
        items = query.replace("compare", "").split("and")
        if len(items) == 2:
            item1 = items[0].strip()
            item2 = items[1].strip()
            r1 = df[df["name"].str.contains(item1, case=False)]
            r2 = df[df["name"].str.contains(item2, case=False)]
            return pd.concat([r1, r2])
    
    return results.head(10)   # default fallback



nexus_scout("Find the cheapest phone.")



nexus_scout("Show phones under ₹20000.")




nexus_scout("Show me Samsung phones under ₹30000.")
nexus_scout("Which phone has Type-C port?")





nexus_scout("Display all phones with LED flashlight.")



nexus_scout("Show laptops under ₹50000.")





nexus_scout("Find the most expensive laptop.")



nexus_scout("List all HP laptops.")



nexus_scout("Show earbuds with high bass speaker.")




nexus_scout("Find the cheapest earbuds.")



nexus_scout("Show earbuds with high bass speaker.")
nexus_scout("Find the cheapest earbuds.")





nexus_scout("Compare Apple Laptop 2 and Dell Laptop 7.")




nexus_scout("Show products with Bluetooth 5.0.")




nexus_scout("List phones with 5000 mAh battery.")




nexus_scout("Show devices with waterproof feature.")
nexus_scout("Find items with Dolby speaker.")



nexus_scout("Show all phones.")


nexus_scout("Show all laptops.")


nexus_scout("Show all earbuds.")


nexus_scout("Show all smartwatches.")

