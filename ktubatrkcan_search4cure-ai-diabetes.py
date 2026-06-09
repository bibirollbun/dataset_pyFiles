import os

# Path to your GCP service account JSON
path_to_credentials = ""  # <-- fill in your JSON path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path_to_credentials

# GCS project & bucket (for PDFs and images)
GCS_PROJECT = ""  # <-- fill in your GCS project ID
GCS_BUCKET = ""   # <-- fill in your GCS bucket name

# BigQuery project, dataset & table (for embeddings)
PROJECT_ID = ""   # <-- fill in your BigQuery project ID


from google.cloud import storage


gcs_client = storage.Client(project=GCS_PROJECT)
gcs_bucket = gcs_client.bucket(GCS_BUCKET)


%pip install langchain-community bigframes tqdm PyMuPDF Pillow PyPDF2 arxiv


import os
import sys
from io import BytesIO
import requests
import fitz
from langchain_community.document_loaders import ArxivLoader




class ArxivPDFLoader:
    def __init__(self, query, max_docs=300, top_k_results = 300):
        self.query = query
        self.max_docs = max_docs
        self.top_k_results = top_k_results
        self.loader = ArxivLoader(query=query, top_k_results=max_docs, load_max_docs=max_docs, load_all_available_meta=True)

    def get_pdf_urls(self):
        docs = self.loader.get_summaries_as_docs()
        pdf_urls = []
        for doc in docs:
            entry_id = doc.metadata.get("Entry ID")
            if entry_id:
                arxiv_id = entry_id.split("/")[-1]
                pdf_urls.append(f"https://arxiv.org/pdf/{arxiv_id}.pdf")
        return pdf_urls

    def download_pdfs(self):
        docs = self.loader.get_summaries_as_docs()
        pdf_docs_with_meta = []

        for doc in docs:
            entry_id = doc.metadata.get("Entry ID")
            title = doc.metadata.get("Title", "untitled").replace(" ", "_").replace("/", "_")
            if entry_id:
                arxiv_id = entry_id.split("/")[-1]
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                try:
                    r = requests.get(pdf_url)
                    if r.status_code == 200:
                        pdf_stream = BytesIO(r.content)
                        pdf = fitz.open(stream=pdf_stream, filetype="pdf")
                        pdf_docs_with_meta.append({
                            "pdf": pdf,
                            "title": title,
                            "url": pdf_url,
                            "arxiv_id": arxiv_id
                        })
                    else:
                        print(f"Failed to download {pdf_url}")
                except Exception as e:
                    print(f"Error downloading {pdf_url}: {e}")

        return pdf_docs_with_meta



# Load from Arxiv
arxiv_loader = ArxivPDFLoader(query="Diabetes")
all_pdfs = arxiv_loader.download_pdfs()


len(all_pdfs)


from google.cloud import storage
from io import BytesIO
from PyPDF2 import PdfReader, PdfWriter


gcs_client = storage.Client(project=GCS_PROJECT)
gcs_bucket = gcs_client.bucket(GCS_BUCKET)


def upload_pdf_to_gcs(key: str, data: bytes) -> None:
    blob = gcs_bucket.blob(key)
    blob.upload_from_string(data, content_type="application/pdf")


for pdf_info in all_pdfs:
    pdf_doc = pdf_info["pdf"]  # fitz.Document
    safe_title = pdf_info['title'].replace(" ", "_").replace("/", "_")
    
    for page_idx in range(pdf_doc.page_count):
        page = pdf_doc[page_idx]
        # Create new PDF with only this page
        pdf_page_doc = fitz.open()
        pdf_page_doc.insert_pdf(pdf_doc, from_page=page_idx, to_page=page_idx)
        page_stream = BytesIO()
        pdf_page_doc.save(page_stream)
        pdf_page_doc.close()
        page_stream.seek(0)

        pdf_page_key = f"multimodal-rag-arxiv-pdf-pages/{safe_title}_page_{page_idx+1}.pdf"
        upload_pdf_to_gcs(pdf_page_key, page_stream.getvalue())


from PIL import Image
zoom = 3.0
mat = fitz.Matrix(zoom, zoom)


def upload_image_to_gcs(key: str, data: bytes) -> None:
    blob = gcs_bucket.blob(key)
    blob.upload_from_string(data, content_type="image/png")


for pdf_info in all_pdfs:
    pdf = pdf_info["pdf"]
    safe_title = pdf_info['title'].replace(" ", "_").replace("/", "_")
    for page_idx in range(pdf.page_count):
        page = pdf[page_idx]
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        gcs_key = f"multimodal-rag-arxiv-pubmed/{safe_title}_page_{page_idx+1}.png"
        upload_image_to_gcs(gcs_key, img_bytes)

    



from google.cloud import storage
import bigframes.pandas as bpd
import bigframes.bigquery as bbq
from google.cloud import bigquery
import bigframes


# Close existing session
bpd.close_session()


# Setup project
bigframes.options.bigquery.project = PROJECT_ID


# Display options
bigframes.options.display.blob_display_width = 300
bigframes.options.display.progress_bar = None


# Create dataframe for images
df_image = bpd.from_glob_path(f"gs://{GCS_BUCKET}/multimodal-rag-arxiv-pubmed/*.png", name="image")
# Extract the URI from the ObjectRef
df_image = df_image.assign(
    uri=df_image["image"].uri
)


len(df_image)


df_image = df_image.assign(
    title=df_image["uri"].str.extract(r"([^/]+)_page_\d+\.png"),
    page_idx=df_image["uri"].str.extract(r"_page_(\d+)\.png").astype(int)
)


df_image.head()


# Create dataframe for PDFs 
df_pdf = bpd.from_glob_path(f"gs://{GCS_BUCKET}/multimodal-rag-arxiv-pdf-pages/*.pdf", name="pdf")


df_pdf.head()


len(df_pdf)


# 1️⃣ Add metadata to df_pdf
df_pdf = df_pdf.assign(
    title=df_pdf["pdf"].uri.str.extract(r"([^/]+)\.pdf"),
    uri = df_pdf["pdf"].uri,
    page_idx=df_pdf["pdf"].uri.str.extract(r"_page_(\d+)\.pdf").astype(int)
)


df_pdf.peek()


df_pdf["page_text"] = df_pdf["pdf"].blob.pdf_extract(engine="pypdf")


df_pdf.columns, df_image.columns


## Join text + image (multimodal dataframe)
df_multimodal = df_pdf.merge(
    df_image,
    on=["title", "page_idx"],
    how="left"  # keeps all text even if image missing
)


df_multimodal.columns


# Rename columns for clarity
df_multimodal = df_multimodal.rename(columns={
    "uri_x": "uri_text",   # from df_chunked
    "uri_y": "uri_image"   # from df_image
})


df_multimodal.columns


from bigframes.ml import llm


import bigframes.bigquery as bbq
from google.cloud import bigquery


embed_model = llm.MultimodalEmbeddingGenerator(model_name="multimodalembedding@001")


# Rename page_text to content globally
df_multimodal = df_multimodal.rename(columns={"page_text": "content"})

# Prepare content + image 
#df_input_full = df_multimodal[["content", "image"]]


# Get first 50 unique PDF titles
first_50_titles = df_multimodal['title'].drop_duplicates().head(50).tolist()


first_50_titles


# Filter df_multimodal to only include these  PDFs
df_50pdfs = df_multimodal[df_multimodal['title'].isin(first_50_titles)]


# Take first 2 pages per PDF
df_multimodal_demo = df_50pdfs.groupby("title").head(2).reset_index(drop=True)


# Prepare input for embedding
df_input_demo = df_multimodal_demo[["content", "image"]]


#  run embedding
df_embeddings = embed_model.predict(df_input_demo)


df_embeddings.columns


#merge back with original df 
df_multimodal_demo = df_multimodal_demo.join(
    df_embeddings["ml_generate_embedding_result"].rename("embedding")
)


# Convert relevant columns to pandas (in-memory)
df_pandas = df_multimodal_demo.to_pandas()
# Check lengths
embedding_lengths = df_pandas["embedding"].apply(lambda x: len(x) if x is not None else 0)
print(embedding_lengths.describe())


# Keep only non-empty embeddings
df_demo_pd = df_pandas[df_pandas["embedding"].map(lambda x: x is not None and len(x) > 0)]



import numpy as np

TARGET_LEN = 1408

def fix_embedding(x):
    if isinstance(x, np.ndarray):  # convert np.array to list first
        x = x.tolist()
    if len(x) < TARGET_LEN:
        return x + [0.0] * (TARGET_LEN - len(x))
    return x[:TARGET_LEN]

df_demo_pd = df_demo_pd.copy()
df_demo_pd.loc[:, "embedding"] = df_demo_pd["embedding"].map(fix_embedding)



df_demo_pd


import pandas_gbq


# Convert complex Arrow struct to plain string safely
df_demo_pd["pdf"] = df_demo_pd["pdf"].map(str)
df_demo_pd["image"] = df_demo_pd["image"].map(str)



# Now convert back to BigFrames
import bigframes.pandas as bpd
df_multimodal_demo = bpd.read_pandas(df_demo_pd)


table_id = df_multimodal_demo.to_gbq()


# Your text query
query_text = "mechanism of insulin resistance"

# Generate embedding using your multimodal embedding model
embedding_df = embed_model.predict([query_text])   # returns a BigFrames DataFrame
# Extract the embedding vector
query_embedding = embedding_df["ml_generate_embedding_result"][0]  # first row


# Wrap it in a DataFrame for BigFrames
search_query = bpd.DataFrame({
    "query_id": [query_text],
    "embedding": [query_embedding]
})

results = bbq.vector_search(
    base_table=f"{table_id}",
    column_to_search="embedding",
    query=search_query,   
    top_k=5
)

print("Top-k results (text + image ObjectRef):")
print(results[["title", "page_idx", "content", "image"]])



from bigframes.ml.llm import GeminiTextGenerator


generator = GeminiTextGenerator()
# Prepare input as a list of strings
pages_text = results["content"].tolist()  
summary = generator.predict(
    X=[f"Please summarize the following pages:\n{page}" for page in pages_text],
    max_output_tokens=500
)

print("Generative summary:")
print(summary)


# Combine all page texts into one string
all_pages_text = "\n".join(results["content"].tolist())

# Generate a single summary
summary = generator.predict(
    X=[f"Please summarize the following pages:\n{all_pages_text}"],
    max_output_tokens=500
)

# Extract the text from the result
summary_text = summary["ml_generate_text_llm_result"][0]
print("Generative summary:")
print(summary_text)




