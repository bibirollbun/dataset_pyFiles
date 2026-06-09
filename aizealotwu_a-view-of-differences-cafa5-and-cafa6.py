!pip install obonet -q
!pip install pyvis -q


!pip install Bio


import os
import json
from PIL import Image
from typing import Dict
from collections import Counter

import random
import cv2
import obonet
import networkx
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
from Bio import SeqIO
from pyvis.network import Network
import plotly.io as pio
pio.renderers.default = 'iframe_connected'


class CFG:
    train_go_obo_path: str = "/kaggle/input/cafa-5-protein-function-prediction/Train/go-basic.obo"
    train_seq_fasta_path: str = "/kaggle/input/cafa-5-protein-function-prediction/Train/train_sequences.fasta"
    train_terms_path: str = "/kaggle/input/cafa-5-protein-function-prediction/Train/train_terms.tsv"
    train_taxonomy_path: str = "/kaggle/input/cafa-5-protein-function-prediction/Train/train_taxonomy.tsv"
    train_ia_path: str = "/kaggle/input/cafa-5-protein-function-prediction/IA.txt"


def plot_dag(graph, term, radius=1):
    # create smaller subgraph
    # radius - include all neighbors of distance<=radius from n (increse it to add further parent's branches).
    ng_graph = networkx.ego_graph(graph, term, radius=radius)

    for n in ng_graph.nodes(data=True):
        # concatenate label of the node with its attribute
        n[1]["label"] = n[0] + " " +n[1]["name"]

    nt = Network(directed=True, notebook=True, cdn_resources="in_line")
    nt.from_nx(ng_graph)
    return nt.show("network.html")


%%time
graph = obonet.read_obo(CFG.train_go_obo_path)


print(f"Number of nodes: {len(graph)}")


print(f"Number of edges: {graph.number_of_edges()}")


term = "GO:0034655"


graph.nodes[term]


def plot_dag(graph, term, radius=1):
    """
    Plots a DAG subgraph around the specified term using pyvis.
    Saves a separate HTML file for each term and radius.
    """
    # Create smaller subgraph
    ng_graph = networkx.ego_graph(graph, term, radius=radius, undirected=False)

    # Prepare node labels
    for n in ng_graph.nodes(data=True):
        n[1]["label"] = n[0] + " " + n[1].get("name", "")

    # Initialize pyvis network
    nt = Network(
        height="800px",
        width="100%",
        directed=True,
        notebook=True,
        bgcolor="#1e1e1e",
        font_color="white",
        cdn_resources="in_line"
    )
    nt.from_nx(ng_graph)

    # Optional styling
    nt.set_options("""
    var options = {
      "nodes": {"color":{"background":"#1f78b4","border":"#ffffff"},"font":{"color":"white","size":14,"face":"Consolas"}},
      "edges": {"color":"white","arrows":{"to":{"enabled":true}}},
      "physics": {"enabled":true,"stabilization":{"iterations":200}}
    }
    """)

    # Dynamic filename
    filename = f"network_{term}_radius{radius}.html"
    return nt.show(filename)



plot_dag(graph, term, radius= 1)


plot_dag(graph, term, radius=50000)


term = "GO:0004842"


graph.nodes[term]


plot_dag(graph, term, radius=3)


plot_dag(graph, term, radius=1000)


print("Sequence example:\n\n", next(iter(SeqIO.parse(CFG.train_seq_fasta_path, "fasta"))))


sequences = SeqIO.parse(CFG.train_seq_fasta_path, "fasta")
num_sequences = sum(1 for seq in sequences)

print("Number of sequences:", num_sequences)


from Bio import SeqIO
import plotly.express as px

# Load sequences
sequences = SeqIO.parse(CFG.train_seq_fasta_path, "fasta")

# Get the length of each sequence
lengths = [len(seq) for seq in sequences]

# Create histogram
fig = px.histogram(
    x=lengths,
    nbins=1000,
    color_discrete_sequence=['goldenrod']
)

# Dark mode layout
fig.update_layout(
    template='plotly_dark',  # Enables dark mode
    title={
        'text': "Distribution of Protein Sequence Lengths",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {'size': 24, 'color': 'goldenrod'}
    },
    xaxis_title="Sequence Length",
    yaxis_title="Count",
    xaxis=dict(showgrid=True, gridcolor='gray', zeroline=False),
    yaxis=dict(showgrid=True, gridcolor='gray', zeroline=False),
    plot_bgcolor='#1e1e1e',  # Dark background
    paper_bgcolor='#1e1e1e', # Dark surrounding
)

fig.show()



np.percentile(lengths, 99)


from Bio import SeqIO
from collections import Counter
import plotly.express as px

# Load sequences
records = SeqIO.parse(CFG.train_seq_fasta_path, "fasta")

# Create a list of all amino acids
aa_list = [aa for record in records for aa in record.seq]

# Count frequency of each amino acid
aa_count = Counter(aa_list)

# Plot horizontal bar chart
fig = px.bar(
    x=list(aa_count.values()),
    y=list(aa_count.keys()),
    orientation='h',
    color_discrete_sequence=['darkslateblue'],
    height=700
)

# Dark mode layout
fig.update_layout(
    template='plotly_dark',
    title={
        'text': "Amino Acid Composition",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {'size': 24, 'color': 'darkorange'}
    },
    xaxis_title="Frequency",
    yaxis_title="Amino Acid",
    xaxis=dict(showgrid=True, gridcolor='gray', zeroline=False),
    yaxis=dict(showgrid=False),
    plot_bgcolor='#1e1e1e',
    paper_bgcolor='#1e1e1e'
)

fig.show()



train_terms_df = pd.read_csv(CFG.train_terms_path, sep="\t")


train_terms_df.head()


train_terms_df.describe()


import plotly.express as px

# Aspect counts
aspect_counts = train_terms_df.aspect.value_counts()

# Pie chart
fig = px.pie(
    values=aspect_counts.values,
    names=aspect_counts.index,
    color_discrete_sequence=px.colors.sequential.Viridis
)

# Dark mode layout
fig.update_traces(
    textposition='inside',
    textfont_size=14,
    textinfo='percent+label',
    marker=dict(line=dict(color='#1e1e1e', width=2))
)
fig.update_layout(
    template='plotly_dark',
    title={
        'text': "Pie Distribution of Aspect Values",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {'size': 22, 'color': 'lightblue'}
    },
    legend_title_text='Aspect:',
    paper_bgcolor='#1e1e1e',
    plot_bgcolor='#1e1e1e'
)

fig.show()



train_taxonomy_df = pd.read_csv(CFG.train_taxonomy_path, sep="\t")


train_taxonomy_df.head()


len(train_taxonomy_df)


limit = 10

with open("/kaggle/input/cafa-5-protein-function-prediction/IA.txt") as f:
    ia_weights = [x.replace("\n", "").split("\t") for x in f.readlines()]

ia_weights[:limit]




# 3. Test Set Analysis
# âšª The test set consists of two main files:
# - testsuperset.fasta: Contains protein sequences for prediction, with headers including UniProt ID and taxon ID
# - testsuperset-taxon-list.tsv: Provides taxon ID information for proteins in the test superset

from Bio import SeqIO
from collections import Counter
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Update config with test set paths
class CFG:
    # ... (ä¿�æŒ�å�Ÿæœ‰è®­ç»ƒé›†è·¯å¾„)
    test_seq_fasta_path: str = "/kaggle/input/cafa-5-protein-function-prediction/Test (Targets)/testsuperset.fasta"
    test_taxon_path: str = "/kaggle/input/cafa-5-protein-function-prediction/Test (Targets)/testsuperset-taxon-list.tsv"
    train_taxonomy_path : str = "/kaggle/input/cafa-5-protein-function-prediction/Train/train_taxonomy.tsv"
    train_seq_fasta_path : str = "/kaggle/input/cafa-5-protein-function-prediction/Train/train_sequences.fasta"
    
# 3.1 Test Sequences Analysis
# â�” Let's first analyze the protein sequences in testsuperset.fasta
# ä¿®æ­£è®­ç»ƒé›†åˆ†ç±»æ•°æ�®çš„åˆ—å��
# train_taxonomy_df = pd.read_csv(
#     CFG.train_taxonomy_path, 
#     sep="\t", 
#     header=None,  # å£°æ˜�æ–‡ä»¶æ²¡æœ‰è¡¨å¤´
#     names=['EntryID', 'Species']  # æ‰‹åŠ¨æŒ‡å®šåˆ—å��
# )
# Count number of test sequences
test_sequences = SeqIO.parse(CFG.test_seq_fasta_path, "fasta")
num_test_sequences = sum(1 for seq in test_sequences)
print(f"Number of test sequences: {num_test_sequences}")

# Sequence length distribution
test_sequences = SeqIO.parse(CFG.test_seq_fasta_path, "fasta")
test_lengths = [len(seq) for seq in test_sequences]

# Plot length distribution
fig = px.histogram(
    x=test_lengths,
    nbins=1000,
    color_discrete_sequence=['crimson']
)

fig.update_layout(
    template='plotly_dark',
    title={
        'text': "Distribution of Test Protein Sequence Lengths",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {'size': 24, 'color': 'crimson'}
    },
    xaxis_title="Sequence Length",
    yaxis_title="Count",
    xaxis=dict(showgrid=True, gridcolor='gray', zeroline=False),
    yaxis=dict(showgrid=True, gridcolor='gray', zeroline=False),
    plot_bgcolor='#1e1e1e',
    paper_bgcolor='#1e1e1e',
)

fig.show()

# 99th percentile of test sequence lengths
print(f"99th percentile of test sequence lengths: {np.percentile(test_lengths, 99)}")

# Amino acid composition in test set
test_sequences = SeqIO.parse(CFG.test_seq_fasta_path, "fasta")
test_aa_list = [aa for record in test_sequences for aa in record.seq]
test_aa_count = Counter(test_aa_list)

# Plot amino acid composition
fig = px.bar(
    x=list(test_aa_count.values()),
    y=list(test_aa_count.keys()),
    orientation='h',
    color_discrete_sequence=['indigo'],
    height=700
)

fig.update_layout(
    template='plotly_dark',
    title={
        'text': "Test Set Amino Acid Composition",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {'size': 24, 'color': 'indigo'}
    },
    xaxis_title="Frequency",
    yaxis_title="Amino Acid",
    xaxis=dict(showgrid=True, gridcolor='gray', zeroline=False),
    yaxis=dict(showgrid=False),
    plot_bgcolor='#1e1e1e',
    paper_bgcolor='#1e1e1e'
)

fig.show()

# 3.2 Test Taxonomy Analysis
# â�” ä»�æµ‹è¯•é›†Fastaå¤´ä¸­æ��å�–ç‰©ç§�ç¼–å�·ï¼ˆTaxon IDï¼‰

# è§£æ��æµ‹è¯•é›†Fastaï¼Œæ��å�–æ¯�ä¸ªåº�åˆ—çš„ç‰©ç§�ç¼–å�·
test_sequences = SeqIO.parse(CFG.test_seq_fasta_path, "fasta")
test_taxon_data = []
for seq in test_sequences:
    entry_id = seq.id
    taxon_id = seq.description.split()[1]  # åº�åˆ—å¤´æ ¼å¼�ï¼š"EntryID TaxonID"ï¼Œåˆ†å‰²å��å�–ç¬¬äºŒä¸ªå…ƒç´ 
    test_taxon_data.append({'EntryID': entry_id, 'Species': taxon_id})

# æ�„å»ºæµ‹è¯•é›†åˆ†ç±»æ•°æ�®DataFrame
test_taxonomy_df = pd.DataFrame(test_taxon_data)
test_taxonomy_df['Species'] = test_taxonomy_df['Species'].astype(str)  # ç»Ÿä¸€ä¸ºå­—ç¬¦ä¸²ç±»å�‹
print("æµ‹è¯•é›†ç‰©ç§�æ•°æ�®ï¼ˆä»�Fastaæ��å�–ï¼‰é¢„è§ˆï¼š")
display(test_taxonomy_df.head())

# Basic statistics
print(f"Number of test proteins in taxon list: {len(test_taxonomy_df)}")
print(f"Number of unique taxon IDs in test set: {test_taxonomy_df['Species'].nunique()}")

# Top 10 most common species in test set
top_test_species = test_taxonomy_df['Species'].value_counts().head(10)

# Plot top species distribution
fig = px.bar(
    x=top_test_species.values,
    y=top_test_species.index,
    orientation='h',
    color_discrete_sequence=['limegreen']
)

fig.update_layout(
    template='plotly_dark',
    title={
        'text': "Top 10 Most Common Species in Test Set",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {'size': 22, 'color': 'limegreen'}
    },
    xaxis_title="Number of Proteins",
    yaxis_title="Species (Taxon ID)",
    xaxis=dict(showgrid=True, gridcolor='gray', zeroline=False),
    yaxis=dict(showgrid=False),
    plot_bgcolor='#1e1e1e',
    paper_bgcolor='#1e1e1e',
    height=600
)

fig.show()

# ==============================================================================
# 2. åŠ è½½è®­ç»ƒé›†åˆ†ç±»æ•°æ�®
# ==============================================================================
train_taxonomy_df = pd.read_csv(
    CFG.train_taxonomy_path, 
    sep="\t", 
    header=None, 
    names=['EntryID', 'Species']
)
# ==============================================================================
# å…³é”®ï¼šç›´æ�¥ä½¿ç”¨å�Ÿå§‹çš„ç‰©ç§�ç¼–å�·ï¼ˆTaxon IDï¼‰è¿›è¡Œå¯¹æ¯”
# ç¡®ä¿�è®­ç»ƒé›†å’Œæµ‹è¯•é›†çš„ç‰©ç§�ç¼–å�·éƒ½æ˜¯å­—ç¬¦ä¸²ç±»å�‹ï¼Œä»¥é�¿å…�ç±»å�‹é”™è¯¯
# ==============================================================================

# ==============================================================================
# å…³é”®ï¼šåœ¨ä½¿ç”¨ Matplotlib ç»˜å›¾å‰�ï¼Œé…�ç½®æ”¯æŒ�ä¸­æ–‡çš„å­—ä½“
# ==============================================================================
import matplotlib.pyplot as plt

# è®¾ç½®å­—ä½“ä»¥æ”¯æŒ�ä¸­æ–‡æ˜¾ç¤º
# 'WenQuanYi Micro Hei' å’Œ 'Heiti TC' æ˜¯ Linux å’Œ macOS ä¸Šå¸¸è§�çš„ä¸­æ–‡å­—ä½“
# å¦‚æ�œæ‰¾ä¸�åˆ°ï¼Œå�¯ä»¥å°�è¯•å…¶ä»–å­—ä½“ï¼Œå¦‚ 'SimHei' (Windows)
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Heiti TC', 'DejaVu Sans']
# è§£å†³è´Ÿå�· '-' åœ¨å›¾è¡¨ä¸­æ˜¾ç¤ºä¸ºæ–¹æ¡†çš„é—®é¢˜
plt.rcParams['axes.unicode_minus'] = False

# ==============================================================================
# å…³é”®ï¼šåœ¨ä½¿ç”¨ Matplotlib ç»˜å›¾å‰�ï¼Œé…�ç½®æ”¯æŒ�ä¸­æ–‡çš„å­—ä½“
# ==============================================================================
import matplotlib.pyplot as plt

# è®¾ç½®å­—ä½“ä»¥æ”¯æŒ�ä¸­æ–‡æ˜¾ç¤º
# 'WenQuanYi Micro Hei' å’Œ 'Heiti TC' æ˜¯ Linux å’Œ macOS ä¸Šå¸¸è§�çš„ä¸­æ–‡å­—ä½“
# å¦‚æ�œæ‰¾ä¸�åˆ°ï¼Œå�¯ä»¥å°�è¯•å…¶ä»–å­—ä½“ï¼Œå¦‚ 'SimHei' (Windows)
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Heiti TC', 'DejaVu Sans']
# è§£å†³è´Ÿå�· '-' åœ¨å›¾è¡¨ä¸­æ˜¾ç¤ºä¸ºæ–¹æ¡†çš„é—®é¢˜
plt.rcParams['axes.unicode_minus'] = False

# ==============================================================================
# 1. é‡�æ–°åŠ è½½å¹¶é¢„å¤„ç�†è®­ç»ƒé›†åˆ†ç±»æ•°æ�®
# ==============================================================================
# ==============================================================================
# 1. é‡�æ–°åŠ è½½å¹¶é¢„å¤„ç�†è®­ç»ƒé›†åˆ†ç±»æ•°æ�®
# ==============================================================================
print("ğŸ”„ Reloading and preprocessing training taxonomy data...")
try:
    train_taxonomy_df = pd.read_csv(
        CFG.train_taxonomy_path, 
        sep="\t", 
        header=None, 
        names=['EntryID', 'Species']
    )
    train_taxonomy_df['Species'] = train_taxonomy_df['Species'].astype(str)
    print("âœ… Training taxonomy data reloaded successfully.")
except Exception as e:
    print(f"â�Œ Failed to reload training taxonomy data. Error: {e}")

# ==============================================================================
# 2. ä»�æµ‹è¯•é›†Fastaæ–‡ä»¶ä¸­æ��å�–å�Ÿå§‹ç‰©ç§�ç¼–å�·
# ==============================================================================
print("\nğŸ”„ Extracting taxon IDs from test FASTA file...")
try:
    test_sequences = SeqIO.parse(CFG.test_seq_fasta_path, "fasta")
    test_taxon_data = []
    for seq in test_sequences:
        entry_id = seq.id
        taxon_id = seq.description.split()[1]
        test_taxon_data.append({'EntryID': entry_id, 'Species': taxon_id})
    
    test_taxonomy_df = pd.DataFrame(test_taxon_data)
    test_taxonomy_df['Species'] = test_taxonomy_df['Species'].astype(str)
    print("âœ… Test taxon IDs extracted successfully.")
except Exception as e:
    print(f"â�Œ Failed to extract test taxon IDs. Error: {e}")

# ==============================================================================
# 3. ç»Ÿè®¡ç‰©ç§�æ•°é‡�å¹¶å�¯è§†åŒ–ï¼ˆå®Œå…¨è‹±æ–‡ç‰ˆæœ¬ï¼‰
# ==============================================================================
try:
    # è®¡ç®—ç‰©ç§�æ•°é‡�
    train_species_count = train_taxonomy_df['Species'].nunique()
    test_species_count = test_taxonomy_df['Species'].nunique()
    train_species_set = set(train_taxonomy_df['Species'].unique())
    test_species_set = set(test_taxonomy_df['Species'].unique())
    common_species_count = len(train_species_set.intersection(test_species_set))
    
    print("\nğŸ“Š Species Count Statistics:")
    print(f"Total species in Training Set: {train_species_count}")
    print(f"Total species in Test Set: {test_species_count}")
    print(f"Common species between both sets: {common_species_count}")
    print(f"Species unique to Training Set: {train_species_count - common_species_count}")
    print(f"Species unique to Test Set: {test_species_count - common_species_count}")
    
    # --- ç»˜å›¾ä»£ç � (å®Œå…¨è‹±æ–‡) ---
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # æŸ±çŠ¶å›¾
    categories = ['Training Set', 'Test Set']
    counts = [train_species_count, test_species_count]
    bars = ax1.bar(categories, counts, color=['royalblue', 'crimson'], width=0.5)
    ax1.set_title('Total Number of Species: Training vs Test Set', fontsize=16, pad=20)
    ax1.set_ylabel('Number of Species', fontsize=14)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 100,
                f'{count}', ha='center', va='bottom', fontsize=14)
    
    # é¥¼å›¾
    labels = ['Common Species', 'Test Set Unique']
    sizes = [common_species_count, test_species_count - common_species_count]
    colors = ['#ff9999', '#99ff99']
    wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors, 
                                       autopct='%1.1f%%', startangle=90,
                                       textprops={'fontsize': 12})
    ax2.set_title('Species Distribution', fontsize=16, pad=20)
    
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"\nâš ï¸� Species count statistics or plotting failed. Detailed error: {type(e).__name__} - {e}")

# ==============================================================================
# 4. è¿›è¡Œç‰©ç§�åˆ†å¸ƒå¯¹æ¯”ï¼ˆPlotlyéƒ¨åˆ†ï¼‰
# ==============================================================================
try:
    top_train_species = train_taxonomy_df['Species'].value_counts().head(10)
    top_test_species = test_taxonomy_df['Species'].value_counts().head(10)
    
    print("\n===== Test Set Top 10 Species Distribution (Taxon ID) =====")
    print(top_test_species)
    print("\n===== Training Set Top 10 Species Distribution (Taxon ID) =====")
    print(top_train_species)
    
    compare_df = pd.DataFrame({
        'Test Set': top_test_species,
        'Training Set': top_train_species
    }).fillna(0)
    
    compare_df['Total'] = compare_df['Test Set'] + compare_df['Training Set']
    compare_df = compare_df.sort_values(by='Total', ascending=True).tail(15)
    
    fig = go.Figure(data=[
        go.Bar(name='Test Set', y=compare_df.index, x=compare_df['Test Set'], orientation='h', marker_color='royalblue'),
        go.Bar(name='Training Set', y=compare_df.index, x=compare_df['Training Set'], orientation='h', marker_color='crimson')
    ])
    
    fig.update_layout(
        template='plotly_dark',
        barmode='group',
        title={
            'text': "Comparison of Species Distribution by Taxon ID (Test vs Training)",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Number of Proteins",
        yaxis_title="Species (Taxon ID)",
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        height=700,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    fig.show()
    print("\nâœ… Comparison plot generated successfully!")

except Exception as e:
    print(f"\nâš ï¸� Comparison failed. Detailed error: {type(e).__name__} - {e}")


# ==============================================================================
# åˆ†æ��è®­ç»ƒé›†å’Œæµ‹è¯•é›†è›‹ç™½åº�åˆ—çš„é‡�å�ˆæ¯”ä¾‹
# ==============================================================================
try:
    # 1. æ��å�–è®­ç»ƒé›†æ‰€æœ‰è›‹ç™½åº�åˆ—
    print("ğŸ”„ Extracting training set sequences...")
    train_sequences = SeqIO.parse(CFG.train_seq_fasta_path, "fasta")
    train_seq_set = set()
    train_seq_count = 0
    
    for seq in train_sequences:
        train_seq_count += 1
        # å°†åº�åˆ—è½¬æ�¢ä¸ºå­—ç¬¦ä¸²å¹¶æ·»åŠ åˆ°é›†å�ˆä¸­ï¼ˆè‡ªåŠ¨å�»é‡�ï¼‰
        train_seq_set.add(str(seq.seq))
    
    print(f"âœ… Training set: {train_seq_count} total sequences, {len(train_seq_set)} unique sequences")
    
    # 2. æ��å�–æµ‹è¯•é›†æ‰€æœ‰è›‹ç™½åº�åˆ—
    print("\nğŸ”„ Extracting test set sequences...")
    test_sequences = SeqIO.parse(CFG.test_seq_fasta_path, "fasta")
    test_seq_set = set()
    test_seq_count = 0
    
    for seq in test_sequences:
        test_seq_count += 1
        test_seq_set.add(str(seq.seq))
    
    print(f"âœ… Test set: {test_seq_count} total sequences, {len(test_seq_set)} unique sequences")
    
    # 3. è®¡ç®—é‡�å�ˆåº�åˆ—
    overlapping_sequences = train_seq_set.intersection(test_seq_set)
    overlapping_count = len(overlapping_sequences)
    
    # 4. è®¡ç®—æ¯”ä¾‹
    train_overlap_ratio = overlapping_count / len(train_seq_set) * 100 if len(train_seq_set) > 0 else 0
    test_overlap_ratio = overlapping_count / len(test_seq_set) * 100 if len(test_seq_set) > 0 else 0
    
    # è¾“å‡ºç»Ÿè®¡ç»“æ�œ
    print("\nğŸ“Š Sequence Overlap Analysis:")
    print(f"Number of overlapping unique sequences: {overlapping_count}")
    print(f"Overlap ratio in training set: {train_overlap_ratio:.2f}%")
    print(f"Overlap ratio in test set: {test_overlap_ratio:.2f}%")
    
    # 5. å�¯è§†åŒ–å±•ç¤º
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    
    # å­�å›¾1ï¼šåº�åˆ—æ€»æ•°å¯¹æ¯”
    categories = ['Training Set', 'Test Set']
    total_counts = [train_seq_count, test_seq_count]
    bars1 = ax1.bar(categories, total_counts, color=['royalblue', 'crimson'], alpha=0.8)
    ax1.set_title('Total Number of Sequences', fontsize=14, pad=15)
    ax1.set_ylabel('Sequence Count')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    for bar, count in zip(bars1, total_counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 500,
                f'{count}', ha='center', va='bottom', fontsize=12)
    
    # å­�å›¾2ï¼šå”¯ä¸€åº�åˆ—æ•°å¯¹æ¯”
    unique_counts = [len(train_seq_set), len(test_seq_set)]
    bars2 = ax2.bar(categories, unique_counts, color=['royalblue', 'crimson'], alpha=0.8)
    ax2.set_title('Number of Unique Sequences', fontsize=14, pad=15)
    ax2.set_ylabel('Unique Sequence Count')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    for bar, count in zip(bars2, unique_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 500,
                f'{count}', ha='center', va='bottom', fontsize=12)
    
    # å­�å›¾3ï¼šé‡�å�ˆåº�åˆ—å� æ¯”é¥¼å›¾
    labels3 = ['Overlapping Sequences', 'Unique to Training']
    sizes3 = [overlapping_count, len(train_seq_set) - overlapping_count]
    colors3 = ['#ff9999', '#66b3ff']
    ax3.pie(sizes3, labels=labels3, colors=colors3, autopct='%1.1f%%', startangle=90)
    ax3.set_title('Sequence Distribution in Training Set', fontsize=14, pad=15)
    
    # å­�å›¾4ï¼šé‡�å�ˆåº�åˆ—å� æ¯”é¥¼å›¾ï¼ˆæµ‹è¯•é›†è§†è§’ï¼‰
    labels4 = ['Overlapping Sequences', 'Unique to Test']
    sizes4 = [overlapping_count, len(test_seq_set) - overlapping_count]
    colors4 = ['#ff9999', '#99ff99']
    ax4.pie(sizes4, labels=labels4, colors=colors4, autopct='%1.1f%%', startangle=90)
    ax4.set_title('Sequence Distribution in Test Set', fontsize=14, pad=15)
    
    plt.tight_layout()
    plt.show()
    
    # 6. é¢�å¤–ç»Ÿè®¡ï¼šè®­ç»ƒé›†å†…éƒ¨é‡�å¤�åº�åˆ—æƒ…å†µ
    train_duplicate_count = train_seq_count - len(train_seq_set)
    test_duplicate_count = test_seq_count - len(test_seq_set)
    
    print("\nğŸ“ˆ Duplicate Sequence Analysis:")
    print(f"Duplicate sequences in training set: {train_duplicate_count} ({train_duplicate_count/train_seq_count*100:.2f}%)")
    print(f"Duplicate sequences in test set: {test_duplicate_count} ({test_duplicate_count/test_seq_count*100:.2f}%)")
    
except Exception as e:
    print(f"\nâš ï¸� Sequence overlap analysis failed. Detailed error: {type(e).__name__} - {e}")

