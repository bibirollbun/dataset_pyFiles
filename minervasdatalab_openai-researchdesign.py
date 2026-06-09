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





!pip install --upgrade openai


import openai


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("OpenAI")


import os
from openai import OpenAI
from kaggle_secrets import UserSecretsClient

# Access OpenAI key from Kaggle Secrets
user_secrets = UserSecretsClient()
openai_key = user_secrets.get_secret("OpenAI")  # "OpenAI" should be the secret label

# Create OpenAI client
client = OpenAI(api_key=openai_key)

# Request a research plan
chat_completion = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a world-class remote sensing archaeologist and research strategist."},
        {"role": "user", "content": """
You are participating in a global challenge to find unknown archaeological sites in the Amazon using satellite imagery, LIDAR, and historical documents.

Your tools include:
- High-resolution satellite imagery (Sentinel-2, Landsat, NICFI)
- LIDAR data (1m–5m resolution) from OpenTopography
- Colonial-era texts and Indigenous maps (to be parsed with GPT/NLP)
- Access to geospatial libraries (rasterio, geopandas, openCV)
- Python, Jupyter notebooks, Kaggle environment
- Models: CNN, ViT, GPT-4, o4-mini

The geographic focus is the Santarém region in northern Brazil. You're expected to:
1. Detect potential archaeological settlements hidden beneath forest canopy.
2. Validate coordinates with two independent methods (e.g., LIDAR + text).
3. Package your findings in a reproducible notebook + PDF report.

Now, devise a **step-by-step research plan** including:
- Data acquisition and bounding box
- Data preprocessing and masking
- Feature engineering and ML approach
- Validation strategy
- Expected output format
- Risks and mitigation

Please format the output as markdown with numbered steps.
"""}
    ]
)

# Print the markdown-formatted research plan
print(chat_completion.choices[0].message.content)



import requests

query = "Amazon LiDAR archaeology Sentinel Landsat"
url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit=10&fields=title,authors,year,abstract,url"

response = requests.get(url)
data = response.json()

for i, paper in enumerate(data.get("data", []), start=1):
    print(f"{i}. {paper['title']}")
    print("   Authors:", ", ".join(a["name"] for a in paper["authors"]))
    print("   Year:", paper["year"])
    print("   URL:", paper["url"])
    print()



import requests

query = "Amazon LiDAR archaeology Sentinel"
url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit=10&fields=title,authors,year,abstract,url"

response = requests.get(url)
data = response.json()

for i, paper in enumerate(data.get("data", []), start=1):
    print(f"{i}. {paper['title']}")
    print("   Authors:", ", ".join(a["name"] for a in paper["authors"]))
    print("   Year:", paper["year"])
    print("   URL:", paper["url"])
    print()


query = "Amazon archaeology NDVI"
url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit=10&fields=title,authors,year,abstract,url"

response = requests.get(url)
data = response.json()

for i, paper in enumerate(data.get("data", []), start=1):
    print(f"{i}. {paper['title']}")
    print("   Authors:", ", ".join(a["name"] for a in paper["authors"]))
    print("   Year:", paper["year"])
    print("   URL:", paper["url"])
    print()


import requests
import pandas as pd
from IPython.display import display, HTML

# List of external DOIs
dois = [
    "10.7717/peerj.15137",                   # PeerJ
    "10.1038/s41467-018-03510-7"             # Nature Communications
]

records = []

for doi in dois:
    url = f"https://api.crossref.org/works/{doi}"
    r = requests.get(url)

    if r.status_code == 200:
        data = r.json()["message"]
        title = data.get("title", ["N/A"])[0]
        authors = ", ".join([f"{a.get('given', '')} {a.get('family', '')}".strip() for a in data.get("author", [])])
        year = data.get("published-print", data.get("published-online", {})).get("date-parts", [[None]])[0][0]
        abstract = data.get("abstract", "N/A").replace("<jats:p>", "").replace("</jats:p>", "") if "abstract" in data else "N/A"

        records.append({
            "Title": title,
            "Year": year,
            "DOI": f"<a href='https://doi.org/{doi}' target='_blank'>{doi}</a>",
            "Abstract": abstract,
            "Link": f"<a href='https://doi.org/{doi}' target='_blank'>View Paper</a>",
            "Origin": "CrossRef"
        })
    else:
        records.append({
            "Title": "N/A",
            "Year": "N/A",
            "DOI": doi,
            "Abstract": "Could not fetch from CrossRef",
            "Link": f"<a href='https://doi.org/{doi}' target='_blank'>Link</a>",
            "Origin": "CrossRef"
        })

# Show combined table
df = pd.DataFrame(records)
pd.set_option("display.max_colwidth", None)
display(HTML(df.to_html(escape=False, index=False)))






import requests
import pandas as pd
from IPython.display import display, HTML

# === Step 1: Load Semantic Scholar papers into df_ss ===
paper_ids = [
    "a57a33a75366fcc98ee60167ef5e909e7e01236f",
    "a57a31b8ac9e0d93751374f8e39ef7bb64f362ca",
    "d389b863c924b0a755745bc43d32af7ff16cb532",
    "d9472c92bb31a001c2dbce9a4a25f9bac6be3640",
    "369c2329587b7fc73ac84766ff7dd1ea9abf0816",
    "b70459ec0536d9747b5f00e219daf52ffac8982f",
    "51ea9a2b69c3e48db02da782dbd3b76359428274",
    "5cb1290372394009b9225b010debea5e5c26c2cc",
    "09f3b85cd7fdc9dd3b9f25a55295b5e3d9cad17d",
    "5887a1cdb74d149f4d6f9a5160cff45e6141e7ff",
    "a69b0b75368b3f32136042b8b8341b133f7b7bcf",
    "8dde685ff036b7027fc23fd71311692105de1e01",
    "afae001fc095865ebff60fd6e1f11d4fc1362e2c"
]

records = []
for pid in paper_ids:
    if not pid.strip():
        continue

    url = f"https://api.semanticscholar.org/graph/v1/paper/{pid}?fields=title,year,abstract,url,externalIds"
    r = requests.get(url)
    if r.status_code == 200:
        data = r.json()
        doi = data.get("externalIds", {}).get("DOI", "N/A")
        doi_link = f"<a href='https://doi.org/{doi}' target='_blank'>{doi}</a>" if doi != "N/A" else "N/A"
        records.append({
            "Title": data.get("title", "N/A"),
            "Year": data.get("year", "N/A"),
            "DOI": doi_link,
            "Abstract": data.get("abstract", "N/A"),
            "Link": f"<a href='{data.get('url')}' target='_blank'>View Paper</a>",
            "Source": "Semantic Scholar"
        })
    else:
        records.append({
            "Title": "N/A",
            "Year": "N/A",
            "DOI": "N/A",
            "Abstract": "Failed to fetch",
            "Link": f"<a href='https://www.semanticscholar.org/paper/{pid}' target='_blank'>Link</a>",
            "Source": "Semantic Scholar"
        })

df_ss = pd.DataFrame(records)

# === Step 2: Fetch additional CrossRef papers ===
crossref_dois = [
    "10.7717/peerj.15137",
    "10.1038/s41467-018-03510-7"
]

crossref_records = []

for doi in crossref_dois:
    url = f"https://api.crossref.org/works/{doi}"
    r = requests.get(url)

    if r.status_code == 200:
        data = r.json()["message"]
        title = data.get("title", ["N/A"])[0]
        authors = ", ".join([f"{a.get('given', '')} {a.get('family', '')}".strip() for a in data.get("author", [])])
        year = data.get("published-print", data.get("published-online", {})).get("date-parts", [[None]])[0][0]
        abstract = data.get("abstract", "N/A").replace("<jats:p>", "").replace("</jats:p>", "") if "abstract" in data else "N/A"
        doi_link = f"<a href='https://doi.org/{doi}' target='_blank'>{doi}</a>"

        crossref_records.append({
            "Title": title,
            "Year": year,
            "DOI": doi_link,
            "Abstract": abstract,
            "Link": f"<a href='https://doi.org/{doi}' target='_blank'>View Paper</a>",
            "Source": "CrossRef"
        })

df_crossref = pd.DataFrame(crossref_records)

# === Step 3: Merge both
df_combined = pd.concat([df_ss, df_crossref], ignore_index=True)

# === Step 4: Display
pd.set_option("display.max_colwidth", None)
display(HTML(df_combined.to_html(escape=False, index=False)))








