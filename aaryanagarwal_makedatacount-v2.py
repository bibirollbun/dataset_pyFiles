# Install required packages
!pip install -q torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
!pip install -q transformers==4.24.0 datasets==2.8.0 sentence-transformers==2.2.2
!pip install -q pymupdf==1.21.1 scikit-learn==1.2.0 umap-learn==0.5.3 kneed==0.8.1

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime

print(f"ğŸ•’ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# Import ML libraries (FIXED VERSION)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from umap import UMAP
from kneed import KneeLocator
from sklearn.neighbors import NearestNeighbors

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
# Replacing SentenceTransformer with direct BERT mean pooling
from datasets import Dataset as HFDataset

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"ğŸ”¥ Using device: {device}")

if torch.cuda.is_available():
    print(f"ğŸ�® GPU: {torch.cuda.get_device_name(0)}")
    print(f"ğŸ’¾ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
# Function to replace SentenceTransformer
def create_embeddings(texts, model_name="bert-base-uncased"):
    """Create embeddings using BERT mean pooling"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    
    embeddings = []
    batch_size = 32
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**encoded)
            # Use mean pooling of last hidden states
            attention_mask = encoded["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
            
        embeddings.append(batch_embeddings)
        
    return np.vstack(embeddings)

print("âœ… Fixed imports successful!")


# Define text extraction functions
import fitz  # PyMuPDF
import xml.etree.ElementTree as ET

def extract_text_from_pdf(pdf_path, max_pages=50):
    """Extract text from PDF with error handling"""
    try:
        text = ""
        with fitz.open(pdf_path) as doc:
            # Limit pages to avoid memory issues
            page_count = min(len(doc), max_pages)
            for page_num in range(page_count):
                page = doc[page_num]
                text += page.get_text() + "\n"
        return text
    except Exception as e:
        print(f"âš ï¸� Error processing PDF {pdf_path}: {e}")
        return ""

def extract_text_from_xml(xml_path):
    """Extract text from XML with error handling"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        text = ET.tostring(root, encoding='unicode', method='text')
        return text
    except Exception as e:
        print(f"âš ï¸� Error processing XML {xml_path}: {e}")
        return ""

def extract_dataset_mentions(text, context_window=500):
    """Extract dataset mentions with their context"""
    # Define regex patterns for different dataset identifiers
    patterns = {
        "DOI": re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I),
        "Accession": re.compile(r"\b(GSE\d+|EGAS\d+|PRJNA\d+|SRP\d+|DRP\d+|ERP\d+|PXD\d+|MTBLS\d+)\b", re.I),
        "Handle": re.compile(r"hdl:\s*\d{2,4}/[\w.]+", re.I),
        "URL": re.compile(r"https?://[\w\.-]+(?:/[\w\.-]*)*(?:\?[\w&=%\.-]*)?(?:#[\w\.-]*)?")
    }
    
    mentions = []
    
    for pattern_name, pattern in patterns.items():
        for match in pattern.finditer(text):
            mention = match.group().strip()
            
            # Get context around the mention
            start_pos = max(0, match.start() - context_window)
            end_pos = min(len(text), match.end() + context_window)
            context = text[start_pos:end_pos].strip()
            
            # Clean context (remove extra whitespace)
            context = re.sub(r'\s+', ' ', context)
            
            mentions.append({
                "mention": mention,
                "mention_type": pattern_name,
                "context": context,
                "position": match.start()
            })
    
    return mentions

print("âœ“ Text extraction functions defined")


# Set up data paths
base_path = Path("/kaggle/input/make-data-count-finding-data-references/test")
pdf_dir = base_path / "PDF"
xml_dir = base_path / "XML"

# Create fallback paths for local testing
if not base_path.exists():
    print("âš ï¸� Kaggle paths not found, using local test data")
    pdf_dir = Path("./PDF")
    xml_dir = Path("./XML")

print(f"ğŸ“� PDF directory exists: {pdf_dir.exists()}")
print(f"ğŸ“� XML directory exists: {xml_dir.exists()}")


# Extract dataset mentions from all files
print("ğŸ”� Extracting dataset mentions from all files...")

# Get all file paths
if pdf_dir.exists() and xml_dir.exists():
    pdf_files = {f.stem: f for f in pdf_dir.glob("*.pdf") if f.is_file()}
    xml_files = {f.stem: f for f in xml_dir.glob("*.xml") if f.is_file()}
    all_article_ids = sorted(set(pdf_files.keys()) | set(xml_files.keys()))
    
    print(f"ğŸ“„ Found {len(pdf_files)} PDF files")
    print(f"ğŸ“„ Found {len(xml_files)} XML files")
    print(f"ğŸ“š Total unique articles: {len(all_article_ids)}")
    
    all_mentions = []
    
    # Process each article
    for article_id in tqdm(all_article_ids, desc="Processing articles"):
        # Extract from PDF
        pdf_text = ""
        if article_id in pdf_files:
            pdf_text = extract_text_from_pdf(pdf_files[article_id])
            
        # Extract from XML
        xml_text = ""
        if article_id in xml_files:
            xml_text = extract_text_from_xml(xml_files[article_id])
        
        # Combine texts
        combined_text = pdf_text + "\n\n" + xml_text
        
        # Extract mentions
        if combined_text.strip():
            mentions = extract_dataset_mentions(combined_text)
            
            for mention in mentions:
                mention["article_id"] = article_id
                all_mentions.append(mention)
else:
    print("âš ï¸� Data directories not found. Creating sample data for testing...")
    # Create sample data for testing
    all_mentions = [
        {
            "article_id": "sample_001",
            "mention": "GSE12345",
            "mention_type": "Accession",
            "context": "We downloaded RNA-seq data from the Gene Expression Omnibus database under accession GSE12345. The dataset contains 150 samples from cancer patients."
        },
        {
            "article_id": "sample_001",
            "mention": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE54321",
            "mention_type": "URL",
            "context": "For comparison, we also used a previously published dataset available at https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE54321 which contains similar tissue samples."
        },
        {
            "article_id": "sample_002",
            "mention": "10.5061/dryad.abc123",
            "mention_type": "DOI",
            "context": "All genomic data generated in this study has been deposited in the Dryad repository with identifier 10.5061/dryad.abc123. The data includes raw sequencing files and processed matrices."
        }
    ]

# Create DataFrame
df_mentions = pd.DataFrame(all_mentions)

# Remove duplicates
df_mentions = df_mentions.drop_duplicates(subset=['article_id', 'mention'])

print(f"\nâœ… Extraction completed!")
print(f"ğŸ“Š Total mentions: {len(df_mentions)}")
if len(df_mentions) > 0:
    print(f"ğŸ“Š Mention types: {df_mentions['mention_type'].value_counts().to_dict()}")
    print(f"\nğŸ“‹ Sample mentions:")
    display(df_mentions.head(3))


# Feature engineering with fixed approach
print("ğŸ§ª Engineering features for ML models...")

def extract_features(df):
    """Extract features from dataset mentions"""
    
    print("ğŸ”¤ Encoding text with BERT embeddings...")
    
    # Text features from context
    contexts = df['context'].fillna('').tolist()
    
    # Create embeddings using our custom function
    context_embeddings = create_embeddings(contexts)
    
    print(f"ğŸ”¢ Creating manual features...")
    # Mention type features (one-hot encoding)
    mention_types = pd.get_dummies(df['mention_type'])
    
    # Pattern-based features
    pattern_features = []
    for context in tqdm(contexts, desc="Extracting lexical features"):
        context_lower = context.lower()
        features = {
            # Primary dataset signals
            'has_we_collected': 'we collected' in context_lower or 'our collection' in context_lower,
            'has_we_generated': 'we generated' in context_lower or 'our generated' in context_lower,
            'has_our_study': 'our study' in context_lower or 'this study' in context_lower,
            'has_experiment': 'experiment' in context_lower,
            'has_measured': 'measured' in context_lower or 'measurement' in context_lower,
            'has_patients': 'patients' in context_lower or 'subjects' in context_lower,
            'has_enrolled': 'enrolled' in context_lower or 'recruited' in context_lower,
            
            # Secondary dataset signals
            'has_downloaded': 'downloaded' in context_lower,
            'has_database': 'database' in context_lower,
            'has_repository': 'repository' in context_lower,
            'has_available': 'available' in context_lower or 'accessed' in context_lower,
            'has_previously': 'previously' in context_lower,
            'has_publicly': 'publicly' in context_lower or 'public' in context_lower,
            'has_obtained_from': 'obtained from' in context_lower,
            
            # General features
            'context_length': len(context),
            'sentences': len(re.split(r'[.!?]', context))
        }
        pattern_features.append(list(features.values()))
    
    pattern_array = np.array(pattern_features, dtype=np.float32)
    
    # Extract features from mentions themselves
    mention_features = []
    for mention in df['mention'].fillna('').tolist():
        mention_lower = str(mention).lower()
        features = [
            # Length features
            len(mention),
            mention.count('.'),
            mention.count('/'),
            
            # Content features
            'geo' in mention_lower or 'gse' in mention_lower,
            'github' in mention_lower,
            'zenodo' in mention_lower or 'figshare' in mention_lower,
            'dryad' in mention_lower,
            'ncbi' in mention_lower or 'nih' in mention_lower,
            any(db in mention_lower for db in ['prjna', 'srp', 'pxd', 'mtbls'])
        ]
        mention_features.append(features)
    
    mention_array = np.array(mention_features, dtype=np.float32)
    
    # Combine all features
    numeric_features = np.hstack([
        context_embeddings,           # BERT embeddings (768 dims)
        pattern_array,                # Manual lexical features (~16 dims)
        mention_types.values,         # One-hot mention types (4 dims)
        mention_array                 # Mention features (9 dims)
    ])
    
    print(f"âœ… Feature extraction complete!")
    print(f"ğŸ“Š Feature matrix shape: {numeric_features.shape}")
    
    return numeric_features, context_embeddings

# Extract features
numeric_features, context_embeddings = extract_features(df_mentions)


# Define and train base models
print("ğŸ¤– Training base models...")

# 1. Isolation Forest
def train_isolation_forest(features):
    print("ğŸŒ² Training Isolation Forest...")
    model = IsolationForest(
        n_estimators=100,
        contamination=0.2,  # Assume about 20% are Primary
        random_state=42,
        n_jobs=-1
    )
    
    # -1 for anomaly (Primary), 1 for normal (Secondary)
    predictions = model.fit_predict(features)
    
    # Convert to binary classification (0=Primary, 1=Secondary)
    binary_predictions = (predictions == 1).astype(int)
    
    print(f"âœ… Isolation Forest trained")
    print(f"ğŸ“Š Predicted Primary: {np.sum(binary_predictions == 0)}")
    print(f"ğŸ“Š Predicted Secondary: {np.sum(binary_predictions == 1)}")
    
    return binary_predictions

# 2. Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=32):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_autoencoder(features, encoding_dim=32, epochs=30):
    print("ğŸ§  Training Autoencoder...")
    
    # Normalize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Convert to PyTorch tensor
    features_tensor = torch.tensor(scaled_features, dtype=torch.float32)
    dataset = TensorDataset(features_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Set up autoencoder
    input_dim = features.shape[1]
    model = Autoencoder(input_dim, encoding_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train autoencoder
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            inputs = batch[0].to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    # Calculate reconstruction error
    model.eval()
    reconstruction_errors = []
    
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=128):
            inputs = batch[0].to(device)
            outputs = model(inputs)
            batch_errors = torch.mean((outputs - inputs) ** 2, dim=1).cpu().numpy()
            reconstruction_errors.extend(batch_errors)
    
    reconstruction_errors = np.array(reconstruction_errors)
    
    # Higher error = more anomalous = Primary
    # Use median as threshold
    threshold = np.median(reconstruction_errors) * 1.2  # Add 20% margin
    binary_predictions = (reconstruction_errors <= threshold).astype(int)
    
    print(f"âœ… Autoencoder trained")
    print(f"ğŸ“Š Predicted Primary: {np.sum(binary_predictions == 0)}")
    print(f"ğŸ“Š Predicted Secondary: {np.sum(binary_predictions == 1)}")
    
    return binary_predictions

# 3. DBSCAN
def train_dbscan(embeddings):
    print("ğŸ”� Running DBSCAN clustering...")
    
    # Optional: reduce dimensions for better clustering
    if embeddings.shape[1] > 50:
        print("Reducing dimensions for clustering...")
        umap_model = UMAP(n_components=20, random_state=42)
        reduced_embeddings = umap_model.fit_transform(embeddings)
    else:
        reduced_embeddings = embeddings
    
    # Find optimal epsilon parameter
    nearest_neighbors = NearestNeighbors(n_neighbors=5)
    nearest_neighbors.fit(reduced_embeddings)
    distances, _ = nearest_neighbors.kneighbors(reduced_embeddings)
    
    # Sort and get knee point for epsilon
    sorted_distances = np.sort(distances[:, 4])
    knee_locator = KneeLocator(
        range(len(sorted_distances)), 
        sorted_distances, 
        curve='convex', 
        direction='increasing'
    )
    
    # Use knee point as epsilon or a default value
    if knee_locator.knee is not None:
        epsilon = sorted_distances[knee_locator.knee]
    else:
        epsilon = np.percentile(sorted_distances, 90)  # Fallback
        
    print(f"DBSCAN epsilon parameter: {epsilon:.4f}")
    
    # Run DBSCAN
    dbscan = DBSCAN(
        eps=epsilon,
        min_samples=5,
        metric='euclidean',
        n_jobs=-1
    )
    clusters = dbscan.fit_predict(reduced_embeddings)
    
    # Analyze clusters
    unique_clusters = np.unique(clusters)
    n_noise = np.sum(clusters == -1)
    
    print(f"DBSCAN found {len(unique_clusters)-1} clusters and {n_noise} noise points")
    
    # If all points are noise or one cluster
    if len(unique_clusters) <= 2:
        # Noise points (-1) are likely Primary, cluster points are Secondary
        binary_predictions = (clusters != -1).astype(int)
    else:
        # Calculate cluster statistics
        cluster_stats = {}
        for cluster_id in unique_clusters:
            if cluster_id != -1:  # Skip noise
                cluster_size = np.sum(clusters == cluster_id)
                cluster_stats[cluster_id] = {
                    "size": cluster_size,
                    "percent": cluster_size / len(clusters) * 100
                }
        
        # Smaller clusters and noise points are Primary
        # The largest cluster is Secondary (most papers use secondary data)
        if cluster_stats:
            largest_cluster = max(cluster_stats.keys(), key=lambda k: cluster_stats[k]['size'])
            binary_predictions = np.ones_like(clusters)  # Default Secondary
            
            # Mark small clusters and noise as Primary
            for cluster_id in unique_clusters:
                if cluster_id == -1 or (cluster_id != largest_cluster and 
                                       cluster_stats[cluster_id]['percent'] < 20):  # Clusters < 20% = Primary
                    binary_predictions[clusters == cluster_id] = 0
        else:
            # Fallback: noise points are Primary
            binary_predictions = (clusters != -1).astype(int)
    
    print(f"âœ… DBSCAN clustering completed")
    print(f"ğŸ“Š Predicted Primary: {np.sum(binary_predictions == 0)}")
    print(f"ğŸ“Š Predicted Secondary: {np.sum(binary_predictions == 1)}")
    
    return binary_predictions

# Train all base models
if_predictions = train_isolation_forest(numeric_features)
ae_predictions = train_autoencoder(numeric_features)
db_predictions = train_dbscan(context_embeddings)

# Combine base predictions
base_predictions = {
    'isolation_forest': if_predictions,
    'autoencoder': ae_predictions,
    'dbscan': db_predictions
}

# Display agreement statistics
print("\nğŸ“Š Model Agreement Statistics:")

# Create combined predictions array
combined = np.vstack([if_predictions, ae_predictions, db_predictions]).T
all_agree = np.sum(np.all(combined == combined[:, 0].reshape(-1, 1), axis=1))
all_primary = np.sum(np.sum(combined == 0, axis=1) == 3)
all_secondary = np.sum(np.sum(combined == 1, axis=1) == 3)

print(f"ğŸ¤� All models agree: {all_agree} ({all_agree/len(combined)*100:.1f}%)")
print(f"   - All predict Primary: {all_primary}")
print(f"   - All predict Secondary: {all_secondary}")

for i in range(3):
    for j in range(i+1, 3):
        model_names = list(base_predictions.keys())
        agreement = np.sum(combined[:, i] == combined[:, j])
        print(f"ğŸ¤� {model_names[i]} and {model_names[j]} agree: {agreement} ({agreement/len(combined)*100:.1f}%)")


# Complete Fixed SciBERT Meta-Model with ALL necessary imports
print("ğŸ§  Setting up SciBERT Meta-Model with all imports...")

# IMPORTANT: Disable wandb logging
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification

os.environ["WANDB_DISABLED"] = "true"

def train_scibert_meta_model(df, base_predictions, model_name="allenai/scibert_scivocab_uncased"):
    """Train SciBERT meta-model for final predictions with robust error handling"""
    print(f"ğŸ¤– Loading {model_name}...")
    
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Step 1: Load model and tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        ).to(device)
    except Exception as e:
        print(f"âš ï¸� Failed to load {model_name}: {e}")
        return fallback_ensemble_voting(df, base_predictions)
    
    # Step 2: Check data dimensions
    print("ğŸ“Š Data dimensions:")
    print(f"   DataFrame: {len(df)}")
    for name, preds in base_predictions.items():
        print(f"   {name}: {len(preds)}")
    
    # Step 3: Create aligned training data
    print("ğŸ”„ Creating meta-model inputs and labels...")
    meta_inputs = []
    meta_labels = []
    valid_indices = []
    
    # Single loop to keep arrays aligned
    for idx in range(min(len(df), len(base_predictions['isolation_forest']))):
        # Get base model predictions for this example
        votes = [
            int(base_predictions['isolation_forest'][idx]),
            int(base_predictions['autoencoder'][idx]),
            int(base_predictions['dbscan'][idx])
        ]
        
        # Calculate majority vote (0=Primary, 1=Secondary)
        majority_vote = 1 if sum(votes) >= 2 else 0
        
        # Keep track of valid index
        valid_indices.append(idx)
        
        # Store label
        meta_labels.append(majority_vote)
        
        # Get row data
        try:
            row = df.iloc[idx]
        except:
            print(f"âš ï¸� Failed to get row at index {idx}, skipping")
            # Remove this index from tracking
            valid_indices.pop()
            meta_labels.pop()
            continue
            
        # Create formatted input text for SciBERT
        context = str(row.get('context', ''))[:300]  # Limit context length
        mention = str(row.get('mention', ''))
        mention_type = str(row.get('mention_type', ''))
        
        # Format votes into text
        votes_text = f"IF:{votes[0]} AE:{votes[1]} DB:{votes[2]}"
        
        # Create structured input
        input_text = f"Mention: {mention} Type: {mention_type} Models: {votes_text} Context: {context}"
        meta_inputs.append(input_text)
    
    # Verify arrays are aligned
    print(f"âœ… Created {len(meta_inputs)} examples with matching labels")
    if len(meta_inputs) != len(meta_labels):
        print("âš ï¸� ERROR: Input and label arrays have different lengths!")
        return fallback_ensemble_voting(df, base_predictions)
        
    # Step 4: Sample data to prevent memory issues
    print("ğŸ§® Sampling training data...")
    max_train_samples = 800  # Limit total training samples
    if len(meta_inputs) > max_train_samples:
        # Random sampling without replacement
        sample_indices = np.random.choice(
            len(meta_inputs), 
            size=max_train_samples, 
            replace=False
        )
        
        train_inputs = [meta_inputs[i] for i in sample_indices]
        train_labels = [meta_labels[i] for i in sample_indices]
        print(f"   Sampled {len(train_inputs)} examples for training")
    else:
        train_inputs = meta_inputs
        train_labels = meta_labels
    
    # Step 5: Create dataset
    try:
        # Create training dataset
        train_encodings = tokenizer(
            train_inputs,
            truncation=True,
            padding="max_length",
            max_length=384,  # Reduced from 512 for memory efficiency
            return_tensors="pt"
        )
        
        # Convert labels to PyTorch tensor (not NumPy!)
        train_labels_tensor = torch.tensor(train_labels)
        
        # Create training dataset
        train_dataset = TensorDataset(
            train_encodings['input_ids'],
            train_encodings['attention_mask'],
            train_labels_tensor
        )
        
        # Split train/val
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        
        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size]
        )
        
        print(f"   Training set: {len(train_dataset)} examples")
        print(f"   Validation set: {len(val_dataset)} examples")
        
    except Exception as e:
        print(f"âš ï¸� Failed to create datasets: {e}")
        return fallback_ensemble_voting(df, base_predictions)
    
    # Step 6: Direct training loop (no Trainer)
    print("ğŸš€ Training SciBERT meta-model...")
    
    # Training parameters
    num_epochs = 2
    batch_size = 16
    learning_rate = 2e-5
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size
    )
    
    # Train model
    model.train()
    
    try:
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            epoch_loss = 0
            
            # Training loop
            for batch_idx, batch in enumerate(train_loader):
                input_ids, attention_mask, labels = batch
                
                # Move to device
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                epoch_loss += loss.item()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Print progress
                if batch_idx % 5 == 0:
                    print(f"   Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            # Calculate average loss
            avg_loss = epoch_loss / len(train_loader)
            print(f"   Average loss: {avg_loss:.4f}")
            
            # Validation
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids, attention_mask, labels = batch
                    
                    # Move to device
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    labels = labels.to(device)
                    
                    # Forward pass
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    
                    # Get predictions
                    _, preds = torch.max(outputs.logits, dim=1)
                    
                    # Calculate accuracy
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
            
            val_acc = val_correct / val_total
            print(f"   Validation accuracy: {val_acc:.4f}")
            
            # Back to training mode
            model.train()
        
        print("âœ… Training complete!")
        
    except Exception as e:
        print(f"âš ï¸� Training failed: {e}")
        return fallback_ensemble_voting(df, base_predictions)
    
    # Step 7: Make predictions on all data
    print("ğŸ”® Making predictions with SciBERT...")
    
    # Initialize prediction arrays
    final_predictions = []  # Final class labels (Primary/Secondary)
    confidence_scores = []  # Confidence scores
    
    # Batch prediction for memory efficiency
    try:
        model.eval()
        
        # Process in manageable batches
        batch_size = 32
        
        for i in range(0, len(meta_inputs), batch_size):
            batch_texts = meta_inputs[i:i+batch_size]
            
            # Tokenize
            batch_encodings = tokenizer(
                batch_texts,
                truncation=True,
                padding="max_length",
                max_length=384,
                return_tensors="pt"
            ).to(device)
            
            # Predict
            with torch.no_grad():
                outputs = model(**batch_encodings)
            
            # Get probabilities with softmax
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            
            # Get class predictions (0=Primary, 1=Secondary)
            preds = torch.argmax(probs, dim=1).cpu()
            
            # Get confidence scores
            # Take probability of predicted class as confidence
            confs = torch.gather(
                probs, 1, preds.unsqueeze(1).to(device)
            ).squeeze().cpu()
            
            # Convert to Primary/Secondary labels
            batch_labels = ["Primary" if p == 0 else "Secondary" for p in preds.numpy()]
            
            # Store predictions and confidence scores
            final_predictions.extend(batch_labels)
            confidence_scores.extend(confs.numpy())
            
            # Progress indicator
            if i % 100 == 0:
                print(f"   Processed {i}/{len(meta_inputs)} examples")
        
        print(f"âœ… Predicted {len(final_predictions)} examples with SciBERT")
        
    except Exception as e:
        print(f"âš ï¸� Prediction failed: {e}")
        return fallback_ensemble_voting(df, base_predictions)
    
    # Step 8: Create full predictions for all data points
    print("ğŸ“Š Creating final predictions for all data points...")
    
    # Map valid indices to their predictions
    all_predictions = ["Secondary"] * len(df)  # Default
    all_confidences = [0.7] * len(df)  # Default confidence
    
    # Fill in SciBERT predictions
    for i, idx in enumerate(valid_indices):
        if i < len(final_predictions):
            all_predictions[idx] = final_predictions[i]
            all_confidences[idx] = float(confidence_scores[i])
    
    # Final distribution
    primary_count = all_predictions.count("Primary")
    secondary_count = all_predictions.count("Secondary")
    print(f"ğŸ“Š Final distribution:")
    print(f"   Primary: {primary_count} ({primary_count/len(all_predictions):.2%})")
    print(f"   Secondary: {secondary_count} ({secondary_count/len(all_predictions):.2%})")
    
    return all_predictions, all_confidences, model, tokenizer

def fallback_ensemble_voting(df, base_predictions):
    """Fallback to simple ensemble voting if SciBERT fails"""
    print("âš ï¸� Falling back to ensemble voting!")
    
    final_predictions = []
    confidence_scores = []
    
    # Create predictions for valid indices
    valid_count = min(len(df), len(base_predictions['isolation_forest']))
    
    for i in range(valid_count):
        # Get votes
        votes = [
            int(base_predictions['isolation_forest'][i]),
            int(base_predictions['autoencoder'][i]),
            int(base_predictions['dbscan'][i])
        ]
        
        # Count Secondary votes
        secondary_votes = sum(votes)
        
        # Determine prediction
        if secondary_votes >= 2:
            prediction = "Secondary"
        else:
            prediction = "Primary"
            
        # Calculate confidence based on vote strength
        if secondary_votes == 0 or secondary_votes == 3:
            # All models agree
            confidence = 0.9
        elif secondary_votes == 1 or secondary_votes == 2:
            # Split decision
            confidence = 0.7
        
        final_predictions.append(prediction)
        confidence_scores.append(confidence)
    
    # Pad remaining indices if needed
    if len(final_predictions) < len(df):
        padding = len(df) - len(final_predictions)
        final_predictions.extend(["Secondary"] * padding)
        confidence_scores.extend([0.7] * padding)
    
    print(f"âœ… Created {len(final_predictions)} predictions using ensemble voting")
    
    # Return predictions with dummy model and tokenizer
    return final_predictions, confidence_scores, None, None

# Run the fixed SciBERT meta-model
final_predictions, confidence_scores, scibert_model, scibert_tokenizer = train_scibert_meta_model(df_mentions, base_predictions)


# Add predictions to dataframe
df_mentions['predicted_type'] = final_predictions
df_mentions['confidence'] = confidence_scores

# Calculate agreement with base models
print("\nğŸ“Š Final prediction agreement with base models:")
for model_name, preds in base_predictions.items():
    model_labels = np.array(["Primary" if p == 0 else "Secondary" for p in preds])
    agreement = np.mean(model_labels == np.array(final_predictions))
    print(f"   {model_name}: {agreement:.2%} agreement")

# Analyze results
print("\nğŸ“Š Final prediction distribution:")
pred_counts = df_mentions['predicted_type'].value_counts()
for label, count in pred_counts.items():
    print(f"   {label}: {count} ({count/len(df_mentions):.2%})")

print("\nğŸ“Š Average confidence:")
for label in ['Primary', 'Secondary']:
    mask = df_mentions['predicted_type'] == label
    if mask.sum() > 0:
        avg_conf = df_mentions.loc[mask, 'confidence'].mean()
        print(f"   {label}: {avg_conf:.4f}")

print("\nğŸ“ˆ Confidence distribution:")
high_conf = (df_mentions['confidence'] > 0.9).mean()
medium_conf = ((df_mentions['confidence'] > 0.7) & (df_mentions['confidence'] <= 0.9)).mean()
low_conf = (df_mentions['confidence'] <= 0.7).mean()
print(f"   High (>0.9): {high_conf:.2%}")
print(f"   Medium (0.7-0.9): {medium_conf:.2%}")
print(f"   Low (â‰¤0.7): {low_conf:.2%}")


# Create submission file
print("ğŸ“� Creating submission file...")

# Create submission dataframe
submission_df = df_mentions[['article_id', 'mention', 'predicted_type']].copy()
submission_df = submission_df.rename(columns={
    'mention': 'dataset_id',
    'predicted_type': 'type'
})

# Add row ID
submission_df = submission_df.reset_index(drop=True)
submission_df['row_id'] = submission_df.index

# Reorder columns
submission_df = submission_df[['row_id', 'article_id', 'dataset_id', 'type']]

# Save submission
submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"âœ… Submission file created: {submission_path}")
print(f"ğŸ“Š Shape: {submission_df.shape}")
print(f"ğŸ“Š Distribution:")
print(submission_df['type'].value_counts())


# Show sample predictions - SIMPLE VERSION
print("ğŸ“‹ Sample predictions:")

# Make sure final_predictions is added to the dataframe
df_mentions['predicted_type'] = final_predictions
if 'confidence' not in df_mentions.columns:
    df_mentions['confidence'] = confidence_scores

# Simple display of examples for each class
for pred_type in ['Primary', 'Secondary']:
    examples = df_mentions[df_mentions['predicted_type'] == pred_type].head(3)
    
    print(f"\nğŸ�·ï¸� {pred_type} examples:")
    if len(examples) == 0:
        print(f"  No {pred_type} examples found")
        continue
        
    for _, row in examples.iterrows():
        print(f"\n   â€¢ Article: {row['article_id']}")
        print(f"     Mention: {row['mention']} ({row['mention_type']})")
        print(f"     Confidence: {row['confidence']:.4f}")
        
        # Get a snippet of context
        context = str(row.get('context', ''))[:] 
        print(f"     Context: {context}...")

# Show examples where base models disagree
print("\nğŸ”„ Examples where base models disagreed:")
found_disagreements = 0

# Loop through a limited number of examples
for i in range(min(50, len(df_mentions))):
    # Skip if index is out of range for base predictions
    if i >= len(base_predictions['isolation_forest']):
        continue
        
    # Get base model predictions
    if_pred = base_predictions['isolation_forest'][i]
    ae_pred = base_predictions['autoencoder'][i] 
    db_pred = base_predictions['dbscan'][i]
    
    # Check if there's disagreement
    if not ((if_pred == ae_pred) and (ae_pred == db_pred)):
        found_disagreements += 1
        row = df_mentions.iloc[i]
        
        print(f"\n   â€¢ Article: {row['article_id']}")
        print(f"     Mention: {row['mention']} ({row['mention_type']})")
        
        # Convert numeric to text labels for readability
        if_label = "Primary" if if_pred == 0 else "Secondary"
        ae_label = "Primary" if ae_pred == 0 else "Secondary" 
        db_label = "Primary" if db_pred == 0 else "Secondary"
        
        print(f"     Base Models: IF:{if_label}, AE:{ae_label}, DB:{db_label}")
        print(f"     Final: {row['predicted_type']} (confidence: {row['confidence']:.4f})")
        
        context = str(row.get('context', ''))[:]
        print(f"     Context: {context}...")
        
        # Only show 3 examples
        if found_disagreements >= 3:
            break

if found_disagreements == 0:
    print("  No disagreements found in the first 50 examples")


# F1 Score Calculation for SciBERT Meta-Model
print("ğŸ“Š Evaluating SciBERT Meta-Model Performance...")

from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Convert predictions to binary format for metrics calculation
binary_predictions = np.array([0 if p == "Primary" else 1 for p in final_predictions])

# 1. Calculate metrics against majority voting (as "ground truth")
print("\nğŸ§® Method 1: Using majority voting as reference")

# Create majority vote "ground truth"
majority_votes = []
for i in range(min(len(df_mentions), len(base_predictions['isolation_forest']))):
    votes = [
        int(base_predictions['isolation_forest'][i]),
        int(base_predictions['autoencoder'][i]),
        int(base_predictions['dbscan'][i])
    ]
    majority_vote = 1 if sum(votes) >= 2 else 0
    majority_votes.append(majority_vote)

# Calculate metrics (only for valid indices)
valid_len = len(majority_votes)
report = classification_report(
    majority_votes[:valid_len], 
    binary_predictions[:valid_len],
    target_names=["Primary", "Secondary"],
    output_dict=True
)

print("\nğŸ“‹ Classification Report:")
print(f"F1-score (Primary): {report['Primary']['f1-score']:.4f}")
print(f"F1-score (Secondary): {report['Secondary']['f1-score']:.4f}")
print(f"Macro Avg F1-score: {report['macro avg']['f1-score']:.4f}")
print(f"Weighted Avg F1-score: {report['weighted avg']['f1-score']:.4f}")

# Display confusion matrix
cm = confusion_matrix(
    majority_votes[:valid_len], 
    binary_predictions[:valid_len]
)

print("\nğŸ”„ Confusion Matrix:")
print("                  Predicted")
print("                Primary  Secondary")
print(f"Actual Primary    {cm[0, 0]:5d}    {cm[0, 1]:5d}")
print(f"       Secondary  {cm[1, 0]:5d}    {cm[1, 1]:5d}")

# 2. Calculate metrics against individual base models
print("\nğŸ§® Method 2: Comparing against individual base models")

base_model_f1s = {}
for model_name, preds in base_predictions.items():
    # Only use valid indices
    valid_len = min(len(preds), len(binary_predictions))
    
    # Calculate F1 score against this base model
    model_f1 = f1_score(
        preds[:valid_len], 
        binary_predictions[:valid_len], 
        average='macro'
    )
    
    # Calculate per-class F1 scores
    report = classification_report(
        preds[:valid_len],
        binary_predictions[:valid_len],
        target_names=["Primary", "Secondary"],
        output_dict=True
    )
    
    # Store results
    base_model_f1s[model_name] = {
        'f1_macro': model_f1,
        'f1_primary': report['Primary']['f1-score'],
        'f1_secondary': report['Secondary']['f1-score']
    }
    
    print(f"\nğŸ”� Compared to {model_name}:")
    print(f"   F1-score (Primary): {report['Primary']['f1-score']:.4f}")
    print(f"   F1-score (Secondary): {report['Secondary']['f1-score']:.4f}")
    print(f"   Macro Avg F1-score: {report['macro avg']['f1-score']:.4f}")

# 3. Calculate agreement rates between SciBERT and base models
print("\nğŸ§® Method 3: Agreement rates with base models")

agreements = {}
for model_name, preds in base_predictions.items():
    valid_len = min(len(preds), len(binary_predictions))
    
    # Calculate agreement rate
    agreement_rate = np.mean(
        np.array(preds[:valid_len]) == binary_predictions[:valid_len]
    )
    
    agreements[model_name] = agreement_rate
    
    print(f"   Agreement with {model_name}: {agreement_rate:.4f} ({agreement_rate*100:.1f}%)")

# 4. Calculate metrics on high-confidence predictions
if len(confidence_scores) > 0:
    print("\nğŸ§® Method 4: Performance on high-confidence predictions")
    high_conf_threshold = 0.8
    
    # Find high confidence predictions
    high_conf_indices = [i for i, conf in enumerate(confidence_scores) 
                         if i < valid_len and conf >= high_conf_threshold]
    
    if len(high_conf_indices) > 0:
        # Get predictions and majority votes for high confidence examples
        high_conf_preds = [binary_predictions[i] for i in high_conf_indices]
        high_conf_truth = [majority_votes[i] for i in high_conf_indices]
        
        # Calculate metrics
        report = classification_report(
            high_conf_truth,
            high_conf_preds,
            target_names=["Primary", "Secondary"],
            output_dict=True
        )
        
        high_conf_count = len(high_conf_indices)
        high_conf_percent = high_conf_count / valid_len * 100
        
        print(f"   High confidence predictions: {high_conf_count} ({high_conf_percent:.1f}% of data)")
        print(f"   F1-score (Primary): {report['Primary']['f1-score']:.4f}")
        print(f"   F1-score (Secondary): {report['Secondary']['f1-score']:.4f}")
        print(f"   Macro Avg F1-score: {report['macro avg']['f1-score']:.4f}")
    else:
        print("   No high-confidence predictions found")

# 5. Visual comparison (only if notebook allows visualization)
try:
    # Prepare data for plotting
    model_names = list(base_model_f1s.keys()) + ['SciBERT']
    f1_primary = [scores['f1_primary'] for scores in base_model_f1s.values()] + [report['Primary']['f1-score']]
    f1_secondary = [scores['f1_secondary'] for scores in base_model_f1s.values()] + [report['Secondary']['f1-score']]
    
    # Create DataFrame
    plot_df = pd.DataFrame({
        'Model': model_names * 2,
        'Class': ['Primary'] * len(model_names) + ['Secondary'] * len(model_names),
        'F1 Score': f1_primary + f1_secondary
    })
    
    # Create plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='F1 Score', hue='Class', data=plot_df)
    plt.title('F1 Score Comparison Across Models')
    plt.xlabel('Model')
    plt.ylabel('F1 Score')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"Note: Couldn't create visualization - {e}")


# Final summary
print("ğŸ�¯ FINAL SUMMARY")
print("=" * 50)
print(f"ğŸ“š Articles processed: {len(set(df_mentions['article_id']))}")
print(f"ğŸ”� Total dataset mentions: {len(df_mentions)}")
print("\nğŸ§ª Approach:")
print("   â€¢ Base models: Isolation Forest, Autoencoder, DBSCAN")
print("   â€¢ Meta-model: SciBERT")
print("   â€¢ Feature engineering: Context embeddings + lexical patterns")

print("\nğŸ“Š Results:")
for label, count in df_mentions['predicted_type'].value_counts().items():
    print(f"   â€¢ {label}: {count} ({count/len(df_mentions):.2%})")

print("\nğŸ’ª Model Strengths:")
print("   â€¢ Ensemble approach combines statistical and semantic signals")
print("   â€¢ SciBERT leverages scientific domain knowledge")
print("   â€¢ High-confidence predictions for most mentions")

print("\nğŸ“„ Output file:")
print(f"   â€¢ Path: {submission_path}")
print(f"   â€¢ Rows: {len(submission_df)}")

print(f"\nâ�° End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nğŸ�‰ Done!")

