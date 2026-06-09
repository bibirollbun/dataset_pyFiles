import warnings
warnings.filterwarnings('ignore')


%%capture
!pip install pymupdf


import pandas as pd
import numpy as np
import fitz  # PyMuPDF
import os
from pathlib import Path
import re
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel
import torch
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF, prioritizing abstract and title"""
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        
        # Extract text from first few pages (usually contain abstract)
        for page_num in range(min(3, len(doc))):
            page = doc[page_num]
            text = page.get_text()
            full_text += text + " "
        
        doc.close()
        
        # Extract title (usually first line or largest text)
        lines = full_text.split('\n')
        title = lines[0].strip() if lines else "Unknown Title"

        content = full_text[:10_000].strip()
            
        return title, content
            
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return "Error", ""

# Load train data
train_path = "/kaggle/input/make-data-count-finding-data-references/train/PDF"
pdf_files = list(Path(train_path).glob("*.pdf"))

texts = []
text_previews = []
titles = []
filenames = []

for pdf_file in pdf_files:
    title, text = extract_text_from_pdf(pdf_file)
    if text:
        texts.append(text)
        titles.append(title)
        text_previews.append(text[500:600])  # Limit length: 50 chars
        filenames.append(pdf_file.stem)

print(f"Extracted text from {len(texts)} papers")


import tqdm
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Load SciBERT model
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased').to(DEVICE)

def get_embeddings(texts, batch_size=16):
    """Generate embeddings using SciBERT"""
    embeddings = []
    
    for i in tqdm.tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize
        inputs = tokenizer(batch_texts, padding=True, truncation=True, 
                          max_length=512, return_tensors="pt").to(DEVICE)
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # Use CLS token embedding
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.extend(batch_embeddings)
    
    return np.array(embeddings)

embeddings = get_embeddings(texts)
print(f"Generated embeddings shape: {embeddings.shape}")


# Apply K-Means clustering
n_clusters = 8
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(embeddings)

# Reduce dimensions for visualization
pca = PCA(n_components=2, random_state=42)
embeddings_2d = pca.fit_transform(embeddings)

# Format text for better hover display
def format_text_for_hover(text, chars_per_line=50, max_lines=10):
    """Format text into multiple lines for hover display"""
    # Start from 100th character
    text = text[500:] if len(text) > 100 else text
    
    words = text.split()
    lines = []
    current_line = ""
    
    for word in words:
        if len(current_line + " " + word) <= chars_per_line:
            current_line += " " + word if current_line else word
        else:
            lines.append(current_line)
            current_line = word
            if len(lines) >= max_lines:
                break
    
    if current_line and len(lines) < max_lines:
        lines.append(current_line)
    
    return "<br>".join(lines)

# Create dataframe for visualization
df_viz = pd.DataFrame({
    'x': embeddings_2d[:, 0],
    'y': embeddings_2d[:, 1],
    'cluster': cluster_labels,
    'filename': filenames,
    'title': titles,
    'text_preview': [format_text_for_hover(text) for text in texts]
})

print(f"Clustering completed with {n_clusters} clusters")
print(f"Cluster distribution: {np.bincount(cluster_labels)}")


# Create interactive scatter plot
fig = px.scatter(df_viz, x='x', y='y', color='cluster', 
                hover_data=['text_preview'], 
                hover_name='title',
                title='Science Papers Clustering using SciBERT Embeddings',
                labels={'x': 'PCA Component 1', 'y': 'PCA Component 2'},
                color_continuous_scale='viridis')

fig.update_traces(
    hovertemplate='<b>%{hovertext}</b><br>' +
                  'Text: %{customdata[0]}<br>' +
                  'Cluster: %{marker.color}<br>' +
                  'PCA1: %{x:.2f}<br>' +
                  'PCA2: %{y:.2f}<br>' +
                  '<extra></extra>'
)

fig.update_layout(
    width=800,
    height=600,
    showlegend=True
)

fig.show(renderer='iframe')


# Analyze clusters
for cluster_id in range(n_clusters):
    cluster_papers = df_viz[df_viz['cluster'] == cluster_id]
    print(f"\n=== CLUSTER {cluster_id} ({len(cluster_papers)} papers) ===")
    
    # Get up to 5 papers from this cluster
    sample_papers = cluster_papers.head(5)
    
    for idx, (_, paper) in enumerate(sample_papers.iterrows()):
        print(f"\nPaper {idx+1}:")
        print(f"Title: {paper['title']}")
        print(f"Text: {paper['text_preview'].replace('<br>', ' ')}")
        print("-" * 50)
    
    print(f"END OF CLUSTER {cluster_id}")
    print("=" * 60)

