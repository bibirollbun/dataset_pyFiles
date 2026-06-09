from kaggle_secrets import UserSecretsClient
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
import os
from fpdf import FPDF
import shutil
from PIL import Image
import pytesseract
from sentence_transformers import SentenceTransformer
import numpy as np
import fitz
import folium
from folium.plugins import BeautifyIcon
from IPython.display import display
from geopy.distance import geodesic
import ee
import geemap
from IPython.display import Image


#!pip install imageio-ffmpeg



#!pip install PyMuPDF



#pip install FPDF



#!pip install ffmpeg-python



#!apt-get update && apt-get install -y ffmpeg



user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("API_KEY")
secret_value_1 = user_secrets.get_secret("GEMINI_API_KEY_0")

genai.configure(api_key=secret_value_0)

#models = genai.list_models()

#print("Gemini Models Available (safe to list):")
#for m in models:
#    print("-", m.name)



model = genai.GenerativeModel(model_name="gemini-2.5-flash")

print("Gemini model loaded successfully:")
print("Model name:", model.model_name)



folder = "paititi_articles"
os.makedirs(folder, exist_ok=True)

urls = {
    "jesuit_report": "https://www.paititi.info/discovering-paititi/jesuits-report/",
    "machupicchu_blog": "https://blog.viajesmachupicchu.travel/en/peruvian-legends-paititi-the-lost-gold-city-in-the-amazon/"
}

# Function to clean special characters not supported by fpdf
def clean_text(text):
    replacements = {
        '\u201c': '"', '\u201d': '"',     # curly double quotes
        '\u2018': "'", '\u2019': "'",     # curly single quotes
        '\u2014': '-',  '\u2013': '-',    # em/en dashes
        '\u2026': '...', '\xa0': ' '      # ellipsis, non-breaking space
    }
    for src, target in replacements.items():
        text = text.replace(src, target)
    return text

def txt_to_pdf(txt_path, pdf_path):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            clean_line = clean_text(line.strip())
            pdf.multi_cell(0, 10, clean_line)

    pdf.output(pdf_path)

def scrape_article(name, url):
    print(f"Scraping: {url}")
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract article text
    paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
    full_text = "\n\n".join(paragraphs)

    # Save .txt
    text_path = os.path.join(folder, f"{name}_text.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    # Convert to PDF
    pdf_path = os.path.join(folder, f"{name}_text.pdf")
    txt_to_pdf(text_path, pdf_path)

    # Download all images
    img_tags = soup.find_all("img")
    for i, img in enumerate(img_tags):
        img_url = img.get("src")
        if not img_url or img_url.startswith("data:"):
            continue

        full_img_url = requests.compat.urljoin(url, img_url)
        try:
            img_data = requests.get(full_img_url).content
            img_ext = os.path.splitext(full_img_url)[1].split("?")[0] or ".jpg"
            img_name = f"{name}_img_{i}{img_ext}"
            with open(os.path.join(folder, img_name), "wb") as f:
                f.write(img_data)
        except Exception as e:
            print(f" Failed to download image: {full_img_url} — {e}")

# Run for each URL
for name, url in urls.items():
    scrape_article(name, url)

print("\nScraping complete! Text, images, and PDFs saved in the 'paititi_articles' folder.")



article_folder = "/kaggle/working/paititi_articles"
pdf_paths = [
    "/kaggle/input/library-of-congress/Library of Congress.pdf",
    "/kaggle/input/archsitesintheperuvianamazonusingsatellite/IdentificationofArcheologicalSitesinthePeruvianAmazonUsingSatellite.pdf"
]


os.makedirs(article_folder, exist_ok=True)

# Move the two PDF files into the `paititi_articles` folder
for pdf_path in pdf_paths:
    filename = os.path.basename(pdf_path)
    dest_path = os.path.join(article_folder, filename)
    shutil.copy(pdf_path, dest_path)

print("PDFs moved to:", article_folder)



files_to_delete = [
    "/kaggle/working/paititi_articles/machupicchu_blog_img_1.png",
    "/kaggle/working/paititi_articles/machupicchu_blog_img_11.jpg",
    "/kaggle/working/paititi_articles/machupicchu_blog_img_13.jpg",
    "/kaggle/working/paititi_articles/machupicchu_blog_img_3.webp",
    "/kaggle/working/paititi_articles/machupicchu_blog_img_5.jpg",
    "/kaggle/working/paititi_articles/machupicchu_blog_img_7.jpg",
    "/kaggle/working/paititi_articles/machupicchu_blog_img_9.jpg",
    "/kaggle/working/paititi_articles/jesuit_report_text.txt",
    "/kaggle/working/paititi_articles/jesuit_report_img_25.png",
    "/kaggle/working/paititi_articles/jesuit_report_img_4.png",
    "/kaggle/working/paititi_articles/jesuit_report_img_5.png",
    "/kaggle/working/paititi_articles/jesuit_report_img_7.jpg",
    "/kaggle/working/paititi_articles/machupicchu_blog_text.txt"
]

for filename in files_to_delete:
    file_path = os.path.join(folder, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted: {filename}")
    else:
        print(f" Not found: {filename}")




model_sentence = SentenceTransformer("all-MiniLM-L6-v2")

folder_path = "/kaggle/working/paititi_articles"
file_embeddings = {}

def extract_text(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Extract and embed only PDFs
for file_name in os.listdir(folder_path):
    if file_name.lower().endswith(".pdf"):
        path = os.path.join(folder_path, file_name)
        text = extract_text(path)
        if text.strip():
            embedding = model_sentence.encode(text)
            file_embeddings[file_name] = {
                "path": path,
                "embedding": embedding,
                "text": text
            }

print(f" Embedded {len(file_embeddings)} PDF files.")



def get_top_k_files(user_query, k=3):
    query_embedding = model_sentence .encode(user_query)
    similarity_scores = []

    for file_name, data in file_embeddings.items():
        score = np.dot(query_embedding, data["embedding"])  # cosine similarity
        similarity_scores.append((file_name, score))

    # Sort by highest similarity score
    top_k = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:k]

    print(f"\n Top {k} files for query: \"{user_query}\"")
    for rank, (file_name, score) in enumerate(top_k, 1):
        print(f"{rank}. {file_name} (score: {score:.4f})")

    return [file_embeddings[name]["path"] for name, _ in top_k]



top_files = get_top_k_files("What do the articles say about the history, legends, and possible location of Paititi, especially any clues about gold, lost cities, or ancient reports like the Jesuit manuscripts? ", k=3)
uploaded_files = [genai.upload_file(path) for path in top_files]


query = """
You are an expert historian and archaeologist. Carefully read the attached historical documents related to the legend of Paititi, including colonial-era Jesuit reports, indigenous testimonies, and modern expedition books.

Your task is to extract and analyze all **relevant clues** and **possible references** to:

- The city of Paititi or related Inca cities, sacred regions, or mythical settlements
- Named towns, villages, rivers, lakes, mountains, or natural features
- Routes taken by Jesuit missionaries, Inca nobility, indigenous groups, or explorers
- References to terrain (jungle, cliffs, fog forests, etc.) or unusual ecosystems
- Cultural and ritual clues tied to treasure, exile, worship, or spiritual protection

For each meaningful clue, return the following in a **clean, readable format**:

1. **Summary of the Clue** – Combine direct quote (or a short paraphrase) with its meaning in one brief paragraph.
2. **Modern Region** – Suggest where this refers to geographically.
3. **Estimated Coordinates**:
   - Latitude: [decimal]
   - Longitude: [decimal]
4. **Confidence Level** – High / Medium / Low (based on clarity of the reference)
5. **Source Document** – Name of the book, manuscript, or article
6. **Historical Period** – If known (e.g. 1600s Jesuit, 1800s explorer, Inca legend)

If multiple clues clearly describe the same or adjacent areas, group them into a **Clue Cluster** and explain their connection.

---

### Search Hypothesis Summary:
At the end, give a scholarly, GIS-compatible **summary** with:

- Suggested 1–3 **Target Zones** for modern exploration:
  - Explanation
  - Latitude and Longitude
- Any **clustering patterns** (e.g. near rivers, cloud forests, ruins)
- Highlight how modern tools (e.g. satellite, GIS, or AI) could help

**Keep your format concise, professional, and suitable for direct use in GIS tools, Google Earth, or expedition planning.**"""



combined_text = ""
for file_data in file_embeddings.values():
    combined_text += f"\n\n--- File: {os.path.basename(file_data['path'])} ---\n\n"
    combined_text += file_data["text"]

full_prompt = combined_text + "\n\n" + query
response = model.generate_content(full_prompt)



print(response.text)




manuscript_images = [
    "/kaggle/working/paititi_articles/jesuit_report_img_11.jpg",
    "/kaggle/working/paititi_articles/jesuit_report_img_13.jpg",
    "/kaggle/working/paititi_articles/jesuit_report_img_15.jpg",
    "/kaggle/working/paititi_articles/jesuit_report_img_17.jpg",
    "/kaggle/working/paititi_articles/jesuit_report_img_19.jpg",
    "/kaggle/working/paititi_articles/jesuit_report_img_9.jpg"
]

manuscript_prompt = """
You are a historical linguist and archaeologist with expertise in ancient and colonial documents related to the Inca civilization.

You have been given an image of a handwritten manuscript that may relate to the legend of Paititi — the lost city of gold. Do not transcribe the document.

Instead, please analyze the image holistically and provide insights based on:

1. Stylistic features — What kind of document might this be? A report? A diary? A letter? Does the writing style suggest it’s colonial, missionary, or indigenous in origin?
2. Mentioned names or terms — Can you visually recognize recurring words, especially proper nouns like towns, regions, or rivers?
3. Language or script clues — Does the writing appear Spanish, Quechua, Latin, or a mix? Is the tone formal, poetic, or instructional?
4. Directional or geographic hints — Even if unclear, are there parts that resemble geographic terms or journey descriptions (e.g., directions, distances, places)?
5. Possible connections to Paititi — Based on visible content, could this page relate to Inca exile, sacred gold, lost settlements, or missionary paths into the Amazon?
6. Provide a reasoned hypothesis: If this document is relevant, what does it imply about the location or myth of Paititi?

Your goal is to help archaeologists understand what this manuscript might reveal about the path to Paititi — or how it was imagined by the writer.

Please give your findings as a professional interpretation, not as a transcription.
"""

for img_path in manuscript_images:
    print(f"\n Analyzing: {os.path.basename(img_path)}")
    image = Image.open(img_path)

    response = model.generate_content([image, manuscript_prompt])
    print(response.text)


images = [
    Image.open("/kaggle/working/paititi_articles/jesuit_report_img_11.jpg"),
    Image.open("/kaggle/working/paititi_articles/jesuit_report_img_13.jpg"),
    Image.open("/kaggle/working/paititi_articles/jesuit_report_img_15.jpg"),
    Image.open("/kaggle/working/paititi_articles/jesuit_report_img_17.jpg"),
    Image.open("/kaggle/working/paititi_articles/jesuit_report_img_19.jpg"),
    Image.open("/kaggle/working/paititi_articles/jesuit_report_img_9.jpg")
]

response = model.generate_content([
    "Please transcribe all the following manuscript images together then translate it in English and identify possible connections to the legend of Paititi. Focus on terms like 'Regno', 'Indiani', 'oro', 'Crocifisso', and any references to sacred places or missionary journeys.",
    *images
])

print(response.text)


m = folium.Map(location=[-12.8, -71.4], zoom_start=7)

colors = {
    "Satellite": "red",
    "Historical": "blue",
    "Legendary": "purple",
    "Symbolic": "orange",
    "Geographic": "darkgreen",
    "Other": "gray"
}

def classify_place(name):
    name_lower = name.lower()
    if "aoi" in name_lower or "satellite" in name_lower or "madre de dios" in name_lower:
        return "Satellite"
    elif "crucifix" in name_lower or "paititi" in name_lower or "college" in name_lower or "missionary" in name_lower or "cotahuasi" in name_lower:
        return "Historical"
    elif "sacred" in name_lower or "jungle" in name_lower or "pongo" in name_lower:
        return "Legendary"
    elif "pope" in name_lower or "vatican" in name_lower:
        return "Symbolic"
    elif "mountain" in name_lower or "river" in name_lower or "range" in name_lower or "cluster" in name_lower:
        return "Geographic"
    else:
        return "Other"

locations = [
    {"name": "Nistron River, Manu National Park", "lat": -12.850, "lon": -71.450},
    {"name": "Pantiacolla Jungle (Paititi suspect zone)", "lat": -13.000, "lon": -71.500},
    {"name": "City of Paititi (as referenced in manuscript)", "lat": -12.500, "lon": -72.000},
    {"name": "Location of Crucifix Miracle (as described by Indian 'Chausea')", "lat": -12.600, "lon": -71.700},
    {"name": "Site where gold chapel was allegedly built", "lat": -12.520, "lon": -71.950},
    {"name": "Missionary Route of P. Andrea Lopez", "lat": -12.000, "lon": -76.900},
    {"name": "College of the Company of Jesus (Cusco)", "lat": -13.516, "lon": -71.978},
    {"name": "Corsen Valley, Peru (possibly Cotahuasi area)", "lat": -15.200, "lon": -72.650},
    {"name": "Sacred Stone (Pietra Bazzarra) Healing Site", "lat": -12.700, "lon": -72.100},
    {"name": "Alleged Transfer Route to the Pope (from Paititi)", "lat": 41.9028, "lon": 12.4964},
    {"name": "AOI 3 (Core, Nistron Basin)", "lat": -12.3780, "lon": -71.6960},
    {"name": "AOI 3 – Upper Town", "lat": -12.3750, "lon": -71.6950},
    {"name": "AOI 3 – Lower Town", "lat": -12.3800, "lon": -71.7000},
    {"name": "Manu National Park (Center)", "lat": -12.1800, "lon": -71.9500},
    {"name": "Pongo de Mainique", "lat": -12.2700, "lon": -72.9300},
    {"name": "Madre de Dios Region", "lat": -12.0000, "lon": -70.0000},
    {"name": "Chachapoyas City", "lat": -6.2167, "lon": -77.8667},
    {"name": "Caldeirao do Inferno (Madeira River)", "lat": -9.0069, "lon": -65.2045},
    {"name": "Ribeirao (Madeira River)", "lat": -9.5583, "lon": -65.0486},
    {"name": "Cachoeira das Lages (Madeira River)", "lat": -10.0000, "lon": -65.0500},
    {"name": "Exaltacion (Mamore River)", "lat": -11.9333, "lon": -64.7167},
    {"name": "Mouth of Beni into Madeira", "lat": -10.3300, "lon": -65.3700},
    {"name": "Apolobamba Mountain Range", "lat": -14.6000, "lon": -69.2000},
    {"name": "Guaporé River (Chronicle of Lizarazu)", "lat": -11.9000, "lon": -63.5000},
    {"name": "Amazonian Target Zone (Madeira River Cluster)", "lat": -9.5000, "lon": -65.1000},
]

for loc in locations:
    group = classify_place(loc["name"])
    color = colors[group]
    folium.Marker(
        location=[loc["lat"], loc["lon"]],
        tooltip=loc["name"],
        popup=folium.Popup(f"<b>{loc['name']}</b><br><i>Category:</i> {group}", max_width=300),
        icon=BeautifyIcon(
            icon_shape='marker',
            border_color=color,
            text_color=color,
            background_color="white",
            border_width=2,
            inner_icon_style="font-size:11px;"
        )
    ).add_to(m)


display(m)



m = folium.Map(
    location=[-12.8, -71.4],
    zoom_start=7,
    tiles="OpenStreetMap",
    attr="© OpenStreetMap contributors"
)

locations = [
    {
        "name": "AOI3: Nistron River (Satellite AOI)",
        "lat": -12.850,
        "lon": -71.450,
        "color": "red",
        "radius": 10000,
        "popup": """
        <b>Nistron River, Manu National Park</b><br>
        Identified via satellite remote sensing (AOI3) as having ideal terrain and possible ancient structures.
        """
    },
    {
        "name": "Pantiacolla Jungle (Inkarri Refuge)",
        "lat": -13.000,
        "lon": -71.500,
        "color": "green",
        "radius": 8000,
        "popup": """
        <b>Pantiacolla Region</b><br>
        Said to be where Inkarri found peace. Deep in jungle, linked to Q'ero people and Inca legends.
        """
    },
    {
        "name": "Madre de Dios (Fawcett's Zone)",
        "lat": -12.500,
        "lon": -70.000,
        "color": "blue",
        "radius": 12000,
        "popup": """
        <b>Madre de Dios Region</b><br>
        Percy Fawcett's search area; satellite reveals geometric formations suggesting ancient settlements.
        """
    },
    {
        "name": "Pongo de Mainique (Indigenous Route)",
        "lat": -12.140,
        "lon": -73.400,
        "color": "orange",
        "radius": 7000,
        "popup": """
        <b>Pongo de Mainique</b><br>
        Machiguenga stories speak of ancient gold being hidden past this gorge along the Urubamba River.
        """
    },
    {
        "name": "City of Paititi (as referenced in manuscript)",
        "lat": -12.500,
        "lon": -72.000,
        "color": "purple",
        "radius": 10000,
        "popup": """
        <b>City of Paititi (Historical Reference)</b><br>
        Described in Jesuit manuscript as the capital visited by missionaries and site of conversion.
        """
    },
    {
        "name": "Location of Crucifix Miracle (Indian 'Chausea')",
        "lat": -12.600,
        "lon": -71.700,
        "color": "darkred",
        "radius": 5000,
        "popup": """
        <b>Miracle Site</b><br>
        Site where the crucifix allegedly moved and caused the king’s conversion.
        """
    },
    {
        "name": "Site of Golden Chapel (Built by King)",
        "lat": -12.520,
        "lon": -71.950,
        "color": "gold",
        "radius": 6000,
        "popup": """
        <b>Golden Oratory Site</b><br>
        Location where the King built a chapel entirely adorned in gold and gems.
        """
    },
    {
        "name": "Sacred Stone (Pietra Bazzarra) Healing Site",
        "lat": -12.700,
        "lon": -72.100,
        "color": "darkgreen",
        "radius": 5000,
        "popup": """
        <b>Sacred Stone Worship Site</b><br>
        Healing stone worshipped before conversion; later offered to the Pope.
        """
    },
    {
        "name": "Missionary Route Start (Lima)",
        "lat": -12.0464,
        "lon": -77.0428,
        "color": "cadetblue",
        "radius": 3000,
        "popup": """
        <b>Missionary Origin Point</b><br>
        Likely origin of Jesuit missionary P. Andrea Lopez’s journey into the Amazon.
        """
    },
    {
        "name": "Jesuit College in Cusco",
        "lat": -13.516,
        "lon": -71.978,
        "color": "lightblue",
        "radius": 3000,
        "popup": """
        <b>College of the Company of Jesus</b><br>
        Jesuit headquarters; possibly coordinated missionary efforts to Paititi.
        """
    },
    {
        "name": "Corsen Valley (Possibly Cotahuasi)",
        "lat": -15.200,
        "lon": -72.650,
        "color": "gray",
        "radius": 4000,
        "popup": """
        <b>Corsen Valley</b><br>
        Mentioned in manuscript as Andrea Lopez’s base; likely Cotahuasi region.
        """
    },
    {
        "name": "Alleged Transfer to the Pope (Symbolic)",
        "lat": 41.9028,
        "lon": 12.4964,
        "color": "pink",
        "radius": 20000,
        "popup": """
        <b>Vatican City</b><br>
        The sacred healing stone was reportedly sent here for preservation.
        """
    },
]

for loc in locations:
    folium.Marker(
        [loc["lat"], loc["lon"]],
        popup=folium.Popup(loc["popup"], max_width=300),
        icon=folium.Icon(color=loc["color"], icon="info-sign")
    ).add_to(m)
    folium.Circle(
        radius=loc["radius"],
        location=[loc["lat"], loc["lon"]],
        color=loc["color"],
        fill=True,
        fill_opacity=0.3
    ).add_to(m)

coords = {loc["name"]: (loc["lat"], loc["lon"]) for loc in locations}

route_sequence = [
    "Missionary Route Start (Lima)",
    "Jesuit College in Cusco",
    "Corsen Valley (Possibly Cotahuasi)",
    "Pantiacolla Jungle (Inkarri Refuge)",
    "AOI3: Nistron River (Satellite AOI)",
    "Location of Crucifix Miracle (Indian 'Chausea')",
    "Site of Golden Chapel (Built by King)",
    "City of Paititi (as referenced in manuscript)",
    "Sacred Stone (Pietra Bazzarra) Healing Site",
    "Madre de Dios (Fawcett's Zone)",
    "Alleged Transfer to the Pope (Symbolic)"
]

for i in range(len(route_sequence) - 1):
    start_name = route_sequence[i]
    end_name = route_sequence[i + 1]
    start_coords = coords[start_name]
    end_coords = coords[end_name]
    dist_km = geodesic(start_coords, end_coords).km

    folium.PolyLine(
        [start_coords, end_coords],
        color="purple",
        weight=2.5,
        opacity=0.7
    ).add_to(m)

    mid_lat = (start_coords[0] + end_coords[0]) / 2
    mid_lon = (start_coords[1] + end_coords[1]) / 2

    folium.Marker(
        location=[mid_lat, mid_lon],
        icon=folium.DivIcon(html=f"""<div style="font-size: 10pt; color: black;">{dist_km:.1f} km</div>""")
    ).add_to(m)


display(m)


m = folium.Map(
    location=[-12.8, -71.4],
    zoom_start=7,
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
attr="Tiles © Esri & the GIS community"
)

for loc in locations:
    folium.Marker(
        [loc["lat"], loc["lon"]],
        popup=folium.Popup(loc["popup"], max_width=300),
        icon=folium.Icon(color=loc["color"], icon="info-sign")
    ).add_to(m)
    folium.Circle(
        radius=loc["radius"],
        location=[loc["lat"], loc["lon"]],
        color=loc["color"],
        fill=True,
        fill_opacity=0.3
    ).add_to(m)

coords = {loc["name"]: (loc["lat"], loc["lon"]) for loc in locations}

route_sequence = [
    "Missionary Route Start (Lima)",
    "Jesuit College in Cusco",
    "Corsen Valley (Possibly Cotahuasi)",
    "Pantiacolla Jungle (Inkarri Refuge)",
    "AOI3: Nistron River (Satellite AOI)",
    "Location of Crucifix Miracle (Indian 'Chausea')",
    "Site of Golden Chapel (Built by King)",
    "City of Paititi (as referenced in manuscript)",
    "Sacred Stone (Pietra Bazzarra) Healing Site",
    "Madre de Dios (Fawcett's Zone)",
    "Alleged Transfer to the Pope (Symbolic)"
]

for i in range(len(route_sequence) - 1):
    start_name = route_sequence[i]
    end_name = route_sequence[i + 1]
    start_coords = coords[start_name]
    end_coords = coords[end_name]
    dist_km = geodesic(start_coords, end_coords).km

    # Draw the line
    folium.PolyLine(
        [start_coords, end_coords],
        color="purple",
        weight=2.5,
        opacity=0.7
    ).add_to(m)

    # Add visible distance label at midpoint
    mid_lat = (start_coords[0] + end_coords[0]) / 2
    mid_lon = (start_coords[1] + end_coords[1]) / 2

    folium.Marker(
        location=[mid_lat, mid_lon],
        icon=folium.DivIcon(
            html=f'<div style="font-size: 10pt; color: black;">{dist_km:.1f} km</div>'
        )
    ).add_to(m)

display(m)



#!pip install earthengine-api geemap
#!earthengine unauthenticate



import ee
import geemap
service_account = "ee-service-account@paititi-464021.iam.gserviceaccount.com"
key_path = "/content/paititi-464021-ba440d77faf8.json"

credentials = ee.ServiceAccountCredentials(service_account, key_path)
ee.Initialize(credentials)
print("Earth Engine initialized with service account!")


# regions of AOI3, Paititi, Cusco
region = ee.Geometry.Rectangle([-72.3, -13.6, -71.2, -12.4])
start_date = '2013-01-01'
end_date = '2023-12-31'

def mask_sr(image):
    cloudShadowBitMask = (1 << 3)
    cloudsBitMask = (1 << 5)
    qa = image.select('QA_PIXEL')
    mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0).And(
           qa.bitwiseAnd(cloudsBitMask).eq(0))
    optical_bands = image.select(['SR_B2', 'SR_B3', 'SR_B4']).multiply(0.0000275).add(-0.2)
    return optical_bands.updateMask(mask).copyProperties(image, ['system:time_start'])

collection = (
    ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") # loading and processing Landsat imagery
    .filterBounds(region)
    .filterDate(start_date, end_date)
    .filter(ee.Filter.lt('CLOUD_COVER', 30))
    .map(mask_sr)
)

vis_params = {
    'bands': ['SR_B4', 'SR_B3', 'SR_B2'],
    'min': 0.02,
    'max': 0.35,
    'gamma': 1.4
}

geemap.timelapse.create_timelapse(
    collection=collection,
    region=region,
    start_date=start_date,
    end_date=end_date,
    out_gif='paititi_deforestation.gif',
    vis_params=vis_params,
    frames_per_second=2,
    dimensions=600,
    date_format='YYYY',
    add_text=True,
    font_size=20
)

Image(filename='paititi_deforestation.gif')



points = ee.FeatureCollection([
    ee.Feature(ee.Geometry.Point([-71.450, -12.850]), {'name': 'AOI3: Nistron River'}),
    ee.Feature(ee.Geometry.Point([-71.500, -13.000]), {'name': 'Pantiacolla Jungle'}),
    ee.Feature(ee.Geometry.Point([-72.000, -12.500]), {'name': 'City of Paititi'}),
    ee.Feature(ee.Geometry.Point([-71.700, -12.600]), {'name': 'Location of Crucifix Miracle'}),
    ee.Feature(ee.Geometry.Point([-71.950, -12.520]), {'name': 'Site of Golden Chapel'}),
    ee.Feature(ee.Geometry.Point([-72.100, -12.700]), {'name': 'Sacred Stone Healing Site'}),
])

Map = geemap.Map(center=[-12.7, -71.8], zoom=9)

median_img = collection.median()
Map.addLayer(median_img, vis_params, 'Landsat SR Median')

Map.addLayer(points.style(color='red', pointSize=8), {}, 'Paititi Sites')

Map.addLayer(region, {'color': 'blue'}, 'Region Boundary')

Map


