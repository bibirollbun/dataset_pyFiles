# Gemma 3n Impact Challenge: Multimodal Healthcare Assistant

# SECTION 1: Environment Setup and Documentation

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import requests
import json
import time
import gc
import logging
import warnings
from pathlib import Path
from IPython.display import display, HTML, Image as IPythonImage, Markdown
from datetime import datetime
from PIL import Image
import io
import base64

# Configure logging with timestamp
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Apply notebook-wide styling for a more professional appearance
display(HTML("""
<style>
    /* Custom styling for the entire notebook */
    div.cell {
        margin-bottom: 20px;
    }
    
    /* Code cell styling */
    div.input_area {
        background-color: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #e1e4e8;
    }
    
    /* Output styling */
    div.output_area {
        padding: 8px;
    }
    
    /* Text output styling */
    div.output_text {
        font-size: 15px;
    }
    
    /* Table styling */
    table {
        border-collapse: collapse;
        margin: 15px 0;
        font-size: 14px;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);
    }
    
    th {
        background-color: #4285F4;
        color: white;
        text-align: left;
        padding: 12px 15px;
    }
    
    td {
        padding: 12px 15px;
        border-bottom: 1px solid #dddddd;
    }
    
    tr:nth-child(even) {
        background-color: #f8f9fa;
    }
    
    tr:hover {
        background-color: #e6f2ff;
    }
    
    /* Section headers */
    h1, h2, h3 {
        font-family: 'Google Sans', Arial, sans-serif;
    }
    
    /* Matplotlib output */
    .matplotlib-figure {
        margin: 20px auto;
        text-align: center;
    }
</style>
"""))

# Display notebook header with logo
def display_logo():
    # Using a reliable Google logo
    google_logo_url = "https://cdn.pixabay.com/photo/2017/01/19/09/11/logo-google-1991840_640.png"
    
    try:
        # Try to display the image directly
        display(HTML(f"""
        <div style="background:linear-gradient(90deg, #f8f9fa 0%, #e8f0fe 100%); padding:30px; border-radius:15px; margin-bottom:30px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            <div style="display:flex; align-items:center; justify-content:center; margin-bottom:20px;">
                <img src="{google_logo_url}" style="height:60px; margin-right:20px;">
            </div>
            <h1 style="text-align:center; color:#4285F4; font-size:32px; margin-bottom:10px;">ğŸ§  Gemma 3n Impact Challenge</h1>
            <h2 style="text-align:center; color:#EA4335; font-size:24px; margin-bottom:15px;">Building a Multimodal Healthcare Assistant</h2>
            <p style="text-align:center; color:#5F6368; font-size:16px;">
                Created by <b>AdilShamim8</b> | Last updated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UTC
            </p>
            <div style="display:flex; justify-content:center; margin-top:20px;">
                <span style="background:#E8F0FE; color:#1A73E8; padding:6px 12px; border-radius:20px; font-size:14px; margin:0 5px;">
                    <b>ğŸ�† Kaggle Competition</b>
                </span>
                <span style="background:#E6F4EA; color:#137333; padding:6px 12px; border-radius:20px; font-size:14px; margin:0 5px;">
                    <b>ğŸ¤– Multimodal AI</b>
                </span>
                <span style="background:#FCE8E6; color:#C5221F; padding:6px 12px; border-radius:20px; font-size:14px; margin:0 5px;">
                    <b>âš•ï¸� Healthcare</b>
                </span>
            </div>
        </div>
        """))
    except Exception as e:
        logger.warning(f"Could not display header with image: {e}")
        # Fallback to text-only header
        display(HTML("""
        <div style="background:#f8f9fa; padding:20px; border-radius:10px; margin-bottom:20px">
            <h1 style="text-align:center; color:#4285F4;">ğŸ§  Gemma 3n Impact Challenge</h1>
            <h2 style="text-align:center; color:#EA4335;">Building a Multimodal Healthcare Assistant</h2>
            <p style="text-align:center; color:#5F6368;">
                Created by <b>AdilShamim8</b> | Last updated: 2025-07-06 05:30:00 UTC
            </p>
        </div>
        """))

display_logo()

# Function to create styled section headers
def section_header(title, subtitle, bg_color, text_color, icon):
    display(HTML(f"""
    <div style="background:{bg_color}; padding:15px; border-radius:10px; margin:30px 0 20px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <h2 style="color:{text_color}; margin-bottom:8px;">{icon} {title}</h2>
        <p style="color:{text_color}; opacity:0.8; font-size:16px; margin:0;">{subtitle}</p>
    </div>
    """))

# Print system information with styled output
print("ğŸ“Š System Information:")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")

# Check CUDA availability with visual indicator and styling
if torch.cuda.is_available():
    gpu_info = {
        "CUDA Available": torch.cuda.is_available(),
        "CUDA Device Count": torch.cuda.device_count(),
        "Current CUDA Device": torch.cuda.current_device(),
        "Device Name": torch.cuda.get_device_name(torch.cuda.current_device()),
        "Memory Allocated (MB)": round(torch.cuda.memory_allocated(0)/1024**2, 2),
        "Memory Reserved (MB)": round(torch.cuda.memory_reserved(0)/1024**2, 2),
    }
    
    # Create a visually appealing GPU info box
    gpu_html = f"""
    <div style="background: linear-gradient(90deg, #002b36 0%, #073642 100%); color: #839496; 
         padding: 15px; border-radius: 10px; margin: 20px 0; font-family: monospace; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <span style="font-size: 24px; margin-right: 10px;">ğŸš€</span>
            <span style="font-size: 18px; font-weight: bold; color: #93a1a1;">GPU Detected</span>
        </div>
        <div style="display: grid; grid-template-columns: auto auto; gap: 10px;">
    """
    
    for key, value in gpu_info.items():
        gpu_html += f"""
            <div style="color: #586e75;">{key}:</div>
            <div style="color: #268bd2; font-weight: bold;">{value}</div>
        """
    
    gpu_html += """
        </div>
    </div>
    """
    
    display(HTML(gpu_html))
else:
    display(HTML("""
    <div style="background: #fdf6e3; color: #657b83; padding: 15px; border-radius: 10px; 
         margin: 20px 0; font-family: monospace; border-left: 5px solid #cb4b16; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <div style="display: flex; align-items: center;">
            <span style="font-size: 24px; margin-right: 10px;">âš ï¸�</span>
            <span style="font-size: 18px; font-weight: bold; color: #cb4b16;">No GPU detected. Running on CPU only.</span>
        </div>
        <p style="margin-top: 10px; margin-bottom: 0;">Some operations may be slower, and model loading might take longer.</p>
    </div>
    """))

# ====================================================================
# SECTION 2: Install Required Libraries
# ====================================================================

section_header(
    "Install Required Libraries", 
    "Setting up dependencies for the Gemma 3n model and multimodal processing.",
    "#E8F0FE",
    "#1A73E8",
    "ğŸ“¦"
)

# Function to safely install packages with visual feedback
def install_package(package, version=None):
    """Install a package safely, checking if it's already installed."""
    display(HTML(f"""
    <div style="display: flex; align-items: center; margin: 5px 0;">
        <div style="width: 20px; margin-right: 10px;">â�³</div>
        <div style="color: #5F6368;">Installing {package}{f' {version}' if version else ''}...</div>
    </div>
    """))
    
    try:
        if version:
            if version == "latest":
                !pip install {package} -q
            else:
                !pip install {package}=={version} -q
        else:
            !pip install {package} -q
        
        display(HTML(f"""
        <div style="display: flex; align-items: center; margin: 5px 0;">
            <div style="width: 20px; margin-right: 10px;">âœ…</div>
            <div style="color: #137333;">Successfully installed {package}</div>
        </div>
        """))
        return True
    except Exception as e:
        display(HTML(f"""
        <div style="display: flex; align-items: center; margin: 5px 0;">
            <div style="width: 20px; margin-right: 10px;">â�Œ</div>
            <div style="color: #EA4335;">Failed to install {package}: {str(e)}</div>
        </div>
        """))
        return False

# List of required packages with versions
required_packages = [
    ("timm", "latest"),
    ("accelerate", "latest"),
    ("transformers", "git+https://github.com/huggingface/transformers.git"),
    ("gradio", "latest"),
    ("safetensors", "latest"),
    ("sentencepiece", "latest"),
    ("bitsandbytes", "latest"),
    ("peft", "latest"),
    ("kagglehub", "latest")
]

# Install packages with visual feedback
display(HTML("""
<div style="margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 10px; border-left: 5px solid #4285F4;">
    <h3 style="margin-top: 0; color: #4285F4;">ğŸ“¦ Package Installation</h3>
    <p>Installing required dependencies for model loading and inference...</p>
</div>
"""))

# Install packages
for package, version in required_packages:
    if version == "latest":
        install_package(package)
    elif version.startswith("git+"):
        install_package(version)
    else:
        install_package(package, version)

# ====================================================================
# SECTION 3: License Agreement and Model Setup
# ====================================================================

section_header(
    "License Agreement and Model Setup", 
    "Understanding the Gemma 3n license requirements and setting up the model.",
    "#FCE8E6",
    "#C5221F",
    "ğŸ“�"
)

# Display license information with better styling
display(HTML("""
<div style="background: #fff8e1; border: 1px solid #ffecb3; border-radius: 10px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
    <div style="display: flex; align-items: center; margin-bottom: 15px;">
        <span style="font-size: 24px; margin-right: 15px;">ğŸ“œ</span>
        <h3 style="margin: 0; color: #FB8C00;">Gemma 3n License Agreement</h3>
    </div>
    <p style="margin-bottom: 15px;">Before using Gemma 3n, you must accept the license agreement on Kaggle.</p>
    
    <div style="background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; margin-bottom: 15px;">
        <h4 style="margin-top: 0; color: #616161;">Follow these steps:</h4>
        <ol style="margin-bottom: 0; padding-left: 25px;">
            <li>Visit: <a href="https://www.kaggle.com/models/google/gemma-3n" target="_blank">https://www.kaggle.com/models/google/gemma-3n</a></li>
            <li>Click on the "Accept License" button</li>
            <li>Once accepted, you can download and use the model in this notebook</li>
        </ol>
    </div>
    
    <div style="font-style: italic; color: #757575;">
        The Gemma 3n model is released under specific terms that require acceptance before use.
    </div>
</div>
"""))

# Function to download model with error handling and visual feedback
def get_gemma_model(model_name="google/gemma-3n/transformers/gemma-3n-e2b-it", use_local_fallback=True):
    """
    Downloads and caches the Gemma 3n model with robust error handling.
    
    Args:
        model_name (str): The identifier for the model in Kaggle Hub
        use_local_fallback (bool): Whether to use local fallback if download fails
        
    Returns:
        str: Path to the model directory
    """
    display(HTML(f"""
    <div style="display: flex; align-items: center; margin: 15px 0; background: #f1f3f4; padding: 10px; border-radius: 8px;">
        <div style="width: 30px; margin-right: 10px; text-align: center;">ğŸ”„</div>
        <div>Attempting to download model: <code>{model_name}</code></div>
    </div>
    """))
    
    try:
        import kagglehub
        start_time = time.time()
        model_path = kagglehub.model_download(model_name)
        elapsed = time.time() - start_time
        
        display(HTML(f"""
        <div style="display: flex; align-items: center; margin: 15px 0; background: #e6f4ea; padding: 10px; border-radius: 8px;">
            <div style="width: 30px; margin-right: 10px; text-align: center;">âœ…</div>
            <div>
                <div style="font-weight: bold; color: #137333;">Model downloaded successfully</div>
                <div>Download time: {elapsed:.2f} seconds</div>
                <div>Model path: <code>{model_path}</code></div>
            </div>
        </div>
        """))
        
        logger.info(f"Model downloaded in {elapsed:.2f} seconds")
        logger.info(f"Model path: {model_path}")
        return model_path
    
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error downloading model: {error_message}")
        
        if "You must agree to the license" in error_message:
            display(HTML(f"""
            <div style="display: flex; align-items: flex-start; margin: 15px 0; background: #fce8e6; padding: 15px; border-radius: 8px; border-left: 5px solid #EA4335;">
                <div style="width: 30px; margin-right: 10px; text-align: center;">âš ï¸�</div>
                <div>
                    <div style="font-weight: bold; color: #EA4335; font-size: 16px; margin-bottom: 10px;">LICENSE AGREEMENT REQUIRED</div>
                    <div style="margin-bottom: 10px;">You need to accept the Gemma 3n license before downloading.</div>
                    <div style="margin-bottom: 10px;">Please visit: <a href="https://www.kaggle.com/models/google/gemma-3n" target="_blank">https://www.kaggle.com/models/google/gemma-3n</a></div>
                    <div>Click the 'Accept License' button, then re-run this cell.</div>
                </div>
            </div>
            """))
            
            if use_local_fallback:
                display(HTML(f"""
                <div style="display: flex; align-items: center; margin: 15px 0; background: #e8f0fe; padding: 10px; border-radius: 8px;">
                    <div style="width: 30px; margin-right: 10px; text-align: center;">ğŸ”„</div>
                    <div>Using local fallback path for development...</div>
                </div>
                """))
                # Create a mock path for development purposes
                local_path = "./mock_gemma_model"
                os.makedirs(local_path, exist_ok=True)
                return local_path
            else:
                raise ValueError("License not accepted. Cannot proceed without model.")
        else:
            display(HTML(f"""
            <div style="display: flex; align-items: flex-start; margin: 15px 0; background: #fce8e6; padding: 15px; border-radius: 8px; border-left: 5px solid #EA4335;">
                <div style="width: 30px; margin-right: 10px; text-align: center;">â�Œ</div>
                <div>
                    <div style="font-weight: bold; color: #EA4335; font-size: 16px; margin-bottom: 10px;">Error downloading model</div>
                    <div style="font-family: monospace; background: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 10px;">{error_message}</div>
                </div>
            </div>
            """))
            
            if use_local_fallback:
                display(HTML(f"""
                <div style="display: flex; align-items: center; margin: 15px 0; background: #e8f0fe; padding: 10px; border-radius: 8px;">
                    <div style="width: 30px; margin-right: 10px; text-align: center;">ğŸ”„</div>
                    <div>Using local fallback path for development...</div>
                </div>
                """))
                local_path = "./mock_gemma_model"
                os.makedirs(local_path, exist_ok=True)
                return local_path
            else:
                raise e

# Attempt to download model with visual feedback
GEMMA_PATH = get_gemma_model()

# Check if we're working with a mock path
is_mock_model = "mock_gemma_model" in GEMMA_PATH
if is_mock_model:
    display(HTML("""
    <div style="background: #fff8e1; border-left: 5px solid #FFA000; padding: 15px; border-radius: 8px; margin: 20px 0;">
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <span style="font-size: 24px; margin-right: 15px;">âš ï¸�</span>
            <div style="font-weight: bold; color: #795548;">Using mock model path for development</div>
        </div>
        <p style="margin-bottom: 10px;">Some functionality will be limited until you accept the license and download the actual model.</p>
        <p style="margin-bottom: 0;">The notebook will still run in demonstration mode with simulated responses.</p>
    </div>
    """))
    
    # Create a visual diagram to explain the model architecture - enhanced version
    plt.figure(figsize=(12, 10), facecolor='#f5f5f5')
    
    # Add a background grid
    plt.grid(color='white', linestyle='-', linewidth=1, alpha=0.7)
    
    # Main model box with gradient background
    plt.text(0.5, 0.9, "Gemma 3n Architecture", 
             horizontalalignment='center', fontsize=24, fontweight='bold',
             bbox=dict(facecolor='#e8f0fe', alpha=0.8, boxstyle='round,pad=0.6', 
                      edgecolor='#4285F4'))
    
    # Architecture components with improved styling
    components = [
        {"name": "Text Encoder (Transformer)", "color": "#4285F4", "desc": "Processes natural language inputs"},
        {"name": "Image Encoder (Vision Transformer)", "color": "#EA4335", "desc": "Analyzes visual content"},
        {"name": "Audio Encoder", "color": "#FBBC05", "desc": "Processes speech and sound inputs"},
        {"name": "Multimodal Fusion Layer", "color": "#34A853", "desc": "Combines information across modalities"},
        {"name": "Decoder (Gemma Core)", "color": "#8442f5", "desc": "Generates contextual responses"}
    ]
    
    # Draw connection lines first (so they appear behind the boxes)
    for i in range(len(components) - 1):
        y_start = 0.75 - (i * 0.1)
        y_end = 0.75 - ((i + 1) * 0.1)
        plt.plot([0.5, 0.5], [y_start - 0.03, y_end + 0.03], color='#bdc1c6', linestyle='-', linewidth=2, zorder=1)
    
    # Add components with enhanced boxes and descriptions
    for i, component in enumerate(components):
        y_pos = 0.75 - (i * 0.1)
        plt.text(0.5, y_pos, component["name"], 
                 horizontalalignment='center', fontsize=16, fontweight='bold', color='white',
                 bbox=dict(facecolor=component["color"], alpha=0.8, boxstyle='round,pad=0.5',
                          edgecolor='white', linewidth=1), zorder=2)
        
        # Add description text
        plt.text(0.8, y_pos, component["desc"],
                fontsize=14, horizontalalignment='left', verticalalignment='center',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
    
    # Model capabilities with enhanced styling
    plt.text(0.5, 0.25, "Key Capabilities:", 
             horizontalalignment='center', fontsize=20, fontweight='bold',
             bbox=dict(facecolor='#e0e0e0', alpha=0.5, boxstyle='round,pad=0.5'))
    
    capabilities = [
        {"text": "On-device inference (4B/8B parameters)", "icon": "ğŸ“±"},
        {"text": "Multimodal understanding (text, image, audio)", "icon": "ğŸ”„"},
        {"text": "Memory-efficient (Per-Layer Embeddings)", "icon": "ğŸ’¾"},
        {"text": "Offline operation for privacy", "icon": "ğŸ”’"},
        {"text": "Multilingual support", "icon": "ğŸŒ�"}
    ]
    
    for i, capability in enumerate(capabilities):
        y_pos = 0.18 - (i * 0.06)
        plt.text(0.5, y_pos, f"{capability['icon']} {capability['text']}", 
                 horizontalalignment='center', fontsize=15,
                 bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3',
                          edgecolor='#bdc1c6'))
    
    plt.axis('off')
    plt.tight_layout()
    
    # Save figure to bytes and display as image for better quality
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    img_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    display(HTML(f"""
    <div style="text-align:center; margin:20px 0;">
        <img src="data:image/png;base64,{img_data}" style="max-width:95%; border-radius:10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
    </div>
    """))

# Import transformer components with visual feedback
display(HTML("""
<div style="display: flex; align-items: center; margin: 20px 0; background: #f1f3f4; padding: 10px; border-radius: 8px;">
    <div style="width: 30px; margin-right: 10px; text-align: center;">ğŸ“š</div>
    <div>Importing transformer components...</div>
</div>
"""))

try:
    from transformers import (
        AutoModelForCausalLM, 
        AutoTokenizer, 
        AutoProcessor,
        AutoModelForImageTextToText,
        GenerationConfig,
        BitsAndBytesConfig,
        pipeline
    )
    
    display(HTML("""
    <div style="display: flex; align-items: center; margin: 15px 0; background: #e6f4ea; padding: 10px; border-radius: 8px;">
        <div style="width: 30px; margin-right: 10px; text-align: center;">âœ…</div>
        <div>Successfully imported transformer components</div>
    </div>
    """))
except Exception as e:
    display(HTML(f"""
    <div style="display: flex; align-items: flex-start; margin: 15px 0; background: #fce8e6; padding: 15px; border-radius: 8px; border-left: 5px solid #EA4335;">
        <div style="width: 30px; margin-right: 10px; text-align: center;">â�Œ</div>
        <div>
            <div style="font-weight: bold; color: #EA4335; font-size: 16px; margin-bottom: 10px;">Error importing transformer components</div>
            <div style="font-family: monospace; background: #f8f9fa; padding: 10px; border-radius: 5px;">{str(e)}</div>
        </div>
    </div>
    """))

# ====================================================================
# SECTION 4: Model Loading with Robust Mock Support
# ====================================================================

section_header(
    "Model Loading with Robust Mock Support", 
    "Loading the Gemma 3n model with fallback mechanisms for development.",
    "#E8F0FE",
    "#1A73E8",
    "ğŸ”„"
)

# Function to create quantization config with visual feedback
def get_quantization_config(quantize=True, bits=4):
    """
    Creates a quantization configuration for memory-efficient loading.
    
    Args:
        quantize (bool): Whether to use quantization
        bits (int): Bit precision (4 or 8)
        
    Returns:
        BitsAndBytesConfig or None: Quantization configuration
    """
    if not quantize:
        display(HTML("""
        <div style="display: flex; align-items: center; margin: 10px 0; background: #f1f3f4; padding: 8px; border-radius: 6px;">
            <div style="width: 25px; margin-right: 10px; text-align: center;">â„¹ï¸�</div>
            <div>Quantization disabled. Using full precision model.</div>
        </div>
        """))
        return None
    
    if bits not in [4, 8]:
        display(HTML("""
        <div style="display: flex; align-items: center; margin: 10px 0; background: #fff8e1; padding: 8px; border-radius: 6px;">
            <div style="width: 25px; margin-right: 10px; text-align: center;">âš ï¸�</div>
            <div>Unsupported bit precision. Defaulting to 4-bit.</div>
        </div>
        """))
        bits = 4
    
    display(HTML(f"""
    <div style="display: flex; align-items: center; margin: 10px 0; background: #e6f4ea; padding: 8px; border-radius: 6px;">
        <div style="width: 25px; margin-right: 10px; text-align: center;">ğŸ”§</div>
        <div>Using {bits}-bit quantization for memory efficiency</div>
    </div>
    """))
    
    return BitsAndBytesConfig(
        load_in_4bit=bits == 4,
        load_in_8bit=bits == 8,
        bnb_4bit_quant_type="nf4" if bits == 4 else None,
        bnb_4bit_compute_dtype=torch.float16 if bits == 4 else None,
        bnb_4bit_use_double_quant=True if bits == 4 else False,
    )

# Create a tensor dictionary class for mock models
class TensorDict(dict):
    """A dictionary that mimics tensor methods for mock model compatibility."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = "cpu"
    
    def to(self, device=None, dtype=None):
        """Mimic the to() method of PyTorch tensors."""
        self.device = device if device is not None else self.device
        return self

# Function to load or mock text model with enhanced visuals
def load_text_model(model_path, quantization_config=None, mock_if_needed=True):
    """
    Loads the text generation model with robust mock support.
    
    Args:
        model_path (str): Path to the model
        quantization_config: Quantization configuration
        mock_if_needed (bool): Whether to use a mock model if loading fails
        
    Returns:
        tuple: (tokenizer, model)
    """
    display(HTML("""
    <div style="display: flex; align-items: center; margin: 15px 0; background: #f1f3f4; padding: 10px; border-radius: 8px;">
        <div style="width: 30px; margin-right: 10px; text-align: center;">ğŸ”„</div>
        <div>Loading text generation model and tokenizer...</div>
    </div>
    """))
    
    # Check if we should use a mock model
    if mock_if_needed and "mock_gemma_model" in model_path:
        display(HTML("""
        <div style="display: flex; align-items: center; margin: 15px 0; background: #fff8e1; padding: 10px; border-radius: 8px;">
            <div style="width: 30px; margin-right: 10px; text-align: center;">âš ï¸�</div>
            <div>Using mock text model for development</div>
        </div>
        """))
        
        # Create a simple tokenizer class
        class MockTokenizer:
            def __init__(self):
                self.mock_vocab = {i: f"token_{i}" for i in range(1000)}
            
            def __call__(self, text, return_tensors=None):
                """Return a TensorDict that mimics tensor behavior."""
                return TensorDict({
                    "input_ids": torch.tensor([[1, 2, 3]]), 
                    "attention_mask": torch.tensor([[1, 1, 1]])
                })
            
            def decode(self, token_ids, skip_special_tokens=True):
                if isinstance(token_ids, torch.Tensor):
                    return f"This is a mock response for input: {token_ids.tolist()}"
                return f"This is a mock response for input: {token_ids}"
                
        # Create a simple model class
        class MockTextModel:
            def __init__(self):
                self.device = "cpu"
                
            def generate(self, **kwargs):
                # Extract some common parameters for more realistic responses
                if "input_ids" in kwargs:
                    input_length = len(kwargs["input_ids"][0])
                else:
                    input_length = 3
                
                max_length = kwargs.get("max_new_tokens", 10) + input_length
                return torch.tensor([[i for i in range(1, max_length + 1)]])
            
            def to(self, device):
                self.device = device
                return self
                
        return MockTokenizer(), MockTextModel()
    
    # Try to load the actual model
    try:
        start_time = time.time()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        
        # Determine optimal device map
        if torch.cuda.is_available():
            device_map = "auto"
        else:
            device_map = None
        
        # Load model with optimizations
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map=device_map,
            quantization_config=quantization_config
        )
        
        display(HTML(f"""
        <div style="display: flex; align-items: center; margin: 15px 0; background: #e6f4ea; padding: 10px; border-radius: 8px;">
            <div style="width: 30px; margin-right: 10px; text-align: center;">âœ…</div>
            <div>Text model loaded in {time.time() - start_time:.2f} seconds</div>
        </div>
        """))
        
        return tokenizer, model
    
    except Exception as e:
        display(HTML(f"""
        <div style="display: flex; align-items: flex-start; margin: 15px 0; background: #fce8e6; padding: 15px; border-radius: 8px; border-left: 5px solid #EA4335;">
            <div style="width: 30px; margin-right: 10px; text-align: center;">â�Œ</div>
            <div>
                <div style="font-weight: bold; color: #EA4335; font-size: 16px; margin-bottom: 10px;">Error loading text model</div>
                <div style="font-family: monospace; background: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 10px;">{str(e)}</div>
            </div>
        </div>
        """))
        
        if mock_if_needed:
            display(HTML("""
            <div style="display: flex; align-items: center; margin: 15px 0; background: #fff8e1; padding: 10px; border-radius: 8px;">
                <div style="width: 30px; margin-right: 10px; text-align: center;">ğŸ”„</div>
                <div>Falling back to mock text model</div>
            </div>
            """))
            return load_text_model("./mock_gemma_model", quantization_config, True)
        else:
            raise e

# Function to load or mock multimodal model with enhanced visuals
def load_multimodal_model(model_path, quantization_config=None, mock_if_needed=True):
    """
    Loads the multimodal model with robust mock support.
    
    Args:
        model_path (str): Path to the model
        quantization_config: Quantization configuration
        mock_if_needed (bool): Whether to use a mock model if loading fails
        
    Returns:
        tuple: (processor, model)
    """
    display(HTML("""
    <div style="display: flex; align-items: center; margin: 15px 0; background: #f1f3f4; padding: 10px; border-radius: 8px;">
        <div style="width: 30px; margin-right: 10px; text-align: center;">ğŸ”„</div>
        <div>Loading multimodal model and processor...</div>
    </div>
    """))
    
    # Check if we should use a mock model
    if mock_if_needed and "mock_gemma_model" in model_path:
        display(HTML("""
        <div style="display: flex; align-items: center; margin: 15px 0; background: #fff8e1; padding: 10px; border-radius: 8px;">
            <div style="width: 30px; margin-right: 10px; text-align: center;">âš ï¸�</div>
            <div>Using mock multimodal model for development</div>
        </div>
        """))
        
        # Create a simple processor class
        class MockProcessor:
            def __init__(self):
                pass
            
            def apply_chat_template(self, messages, **kwargs):
                """Return a TensorDict that mimics tensor behavior."""
                return TensorDict({
                    "input_ids": torch.tensor([[1, 2, 3]]), 
                    "attention_mask": torch.tensor([[1, 1, 1]])
                })
            
            def batch_decode(self, outputs, **kwargs):
                # Generate more realistic mock responses based on image content
                global messages
                if len(messages) > 0 and len(messages[0].get("content", [])) > 0:
                    # Check if this is a nutrition-related query
                    content = messages[0]["content"]
                    text_parts = [item["text"] for item in content if item.get("type") == "text"]
                    text = " ".join(text_parts) if text_parts else ""
                    
                    if "nutrition" in text.lower() or "food" in text.lower() or "meal" in text.lower():
                        return ["This appears to be a balanced meal with proteins, vegetables, and complex carbohydrates. The plate contains what looks like grilled chicken, steamed broccoli, and brown rice. This combination provides a good balance of macronutrients with lean protein, fiber-rich vegetables, and whole grains. The meal is approximately 400-500 calories depending on portion sizes and preparation methods."]
                    elif "exercise" in text.lower() or "fitness" in text.lower():
                        return ["The image shows a person performing what appears to be a plank exercise, which is excellent for core strength. The form looks good with a straight line from head to heels. This exercise engages multiple muscle groups including the abdominals, shoulders, and back muscles. For best results, aim to hold this position for 30-60 seconds and repeat 3-5 times."]
                    elif "skin" in text.lower() or "rash" in text.lower():
                        return ["The image shows an area of skin with what appears to be mild redness. Without providing a diagnosis, I can note that the skin has some uneven coloration and possibly slight inflammation. It would be advisable to consult with a healthcare professional for proper evaluation. Keep the area clean, avoid scratching, and monitor for any changes."]
                
                # Default generic response
                return ["This image appears to show [mock description]. Based on what I can see, [mock analysis]. Remember that this is a general observation and not a professional assessment."]
                
        # Create a simple model class
        class MockMultimodalModel:
            def __init__(self):
                self.device = "cpu"
                self.dtype = torch.float32
                
            def generate(self, **kwargs):
                # Extract some common parameters for more realistic responses
                if "input_ids" in kwargs:
                    input_length = len(kwargs["input_ids"][0])
                else:
                    input_length = 3
                
                max_length = kwargs.get("max_new_tokens", 20) + input_length
                return torch.tensor([[i for i in range(1, max_length + 1)]])
                
        return MockProcessor(), MockMultimodalModel()
    
    # Try to load the actual model
    try:
        start_time = time.time()
        
        # Load processor
        processor = AutoProcessor.from_pretrained(model_path)
        
        # Determine optimal device map
        if torch.cuda.is_available():
            device_map = "auto"
            dtype = "auto"
        else:
            device_map = None
            dtype = torch.float32
        
        # Load model with optimizations
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
            quantization_config=quantization_config
        )
        
        display(HTML(f"""
        <div style="display: flex; align-items: center; margin: 15px 0; background: #e6f4ea; padding: 10px; border-radius: 8px;">
            <div style="width: 30px; margin-right: 10px; text-align: center;">âœ…</div>
            <div>Multimodal model loaded in {time.time() - start_time:.2f} seconds</div>
        </div>
        """))
        
        return processor, model
    
    except Exception as e:
        display(HTML(f"""
        <div style="display: flex; align-items: flex-start; margin: 15px 0; background: #fce8e6; padding: 15px; border-radius: 8px; border-left: 5px solid #EA4335;">
            <div style="width: 30px; margin-right: 10px; text-align: center;">â�Œ</div>
            <div>
                <div style="font-weight: bold; color: #EA4335; font-size: 16px; margin-bottom: 10px;">Error loading multimodal model</div>
                <div style="font-family: monospace; background: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 10px;">{str(e)}</div>
            </div>
        </div>
        """))
        
        if mock_if_needed:
            display(HTML("""
            <div style="display: flex; align-items: center; margin: 15px 0; background: #fff8e1; padding: 10px; border-radius: 8px;">
                <div style="width: 30px; margin-right: 10px; text-align: center;">ğŸ”„</div>
                <div>Falling back to mock multimodal model</div>
            </div>
            """))
            return load_multimodal_model("./mock_gemma_model", quantization_config, True)
        else:
            raise e

# Define messages globally for mock processor to access
messages = []

# Get quantization config
quant_config = get_quantization_config(quantize=True, bits=4)

# Load the models (with mock support)
text_tokenizer, text_model = load_text_model(GEMMA_PATH, quant_config)
img_processor, img_model = load_multimodal_model(GEMMA_PATH, quant_config)

# ====================================================================
# SECTION 5: Core Model Capabilities with Robust Error Handling
# ====================================================================

section_header(
    "Core Model Capabilities", 
    "Gemma 3n offers advanced multimodal capabilities that make it ideal for healthcare applications.",
    "#E8F0FE",
    "#1A73E8",
    "ğŸ”�"
)

# Create a visual card to explain model capabilities
display(HTML("""
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 30px 0;">
    <!-- Text Generation Card -->
    <div style="background: linear-gradient(135deg, #E8F0FE 0%, #C2E0FF 100%); border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <div style="display: flex; align-items: center; margin-bottom: 15px;">
            <div style="background: #4285F4; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 15px;">
                <span style="color: white; font-size: 20px;">ğŸ’¬</span>
            </div>
            <h3 style="margin: 0; color: #1A73E8;">Text Generation</h3>
        </div>
        <p style="margin-bottom: 15px; color: #3c4043;">
            Generates coherent, contextually relevant text responses to user queries with control over parameters like temperature and token length.
        </p>
        <div style="background: rgba(255,255,255,0.7); border-radius: 8px; padding: 10px; font-family: monospace; font-size: 12px; color: #3c4043;">
            response = generate_text(prompt, max_new_tokens=512, temperature=0.7)
        </div>
    </div>
    
    <!-- Image Processing Card -->
    <div style="background: linear-gradient(135deg, #FCE8E6 0%, #FFBDAD 100%); border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <div style="display: flex; align-items: center; margin-bottom: 15px;">
            <div style="background: #EA4335; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 15px;">
                <span style="color: white; font-size: 20px;">ğŸ–¼ï¸�</span>
            </div>
            <h3 style="margin: 0; color: #C5221F;">Image Processing</h3>
        </div>
        <p style="margin-bottom: 15px; color: #3c4043;">
            Analyzes images with accompanying text, enabling multimodal understanding for visual health assessments and guidance.
        </p>
        <div style="background: rgba(255,255,255,0.7); border-radius: 8px; padding: 10px; font-family: monospace; font-size: 12px; color: #3c4043;">
            response = process_image_with_text(image_path, text_prompt)
        </div>
    </div>
    
    <!-- Healthcare Insights Card -->
    <div style="background: linear-gradient(135deg, #E6F4EA 0%, #BDEED9 100%); border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <div style="display: flex; align-items: center; margin-bottom: 15px;">
            <div style="background: #34A853; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 15px;">
                <span style="color: white; font-size: 20px;">âš•ï¸�</span>
            </div>
            <h3 style="margin: 0; color: #137333;">Healthcare Insights</h3>
        </div>
        <p style="margin-bottom: 15px; color: #3c4043;">
            Specialized prompts enhance responses with relevant medical knowledge, creating a more helpful healthcare assistant experience.
        </p>
        <div style="background: rgba(255,255,255,0.7); border-radius: 8px; padding: 10px; font-family: monospace; font-size: 12px; color: #3c4043;">
            enhanced_prompt = enhance_medical_prompt(query, category)
        </div>
    </div>
</div>
"""))

# Text generation function with robust error handling and improved logging
def generate_text(
    prompt, 
    model=None, 
    tokenizer=None, 
    max_new_tokens=512, 
    temperature=0.7, 
    top_p=0.9, 
    do_sample=True
):
    """
    Generates text based on the provided prompt with robust error handling.
    
    Args:
        prompt (str): The input prompt
        model: The text generation model
        tokenizer: The tokenizer
        max_new_tokens (int): Maximum number of tokens to generate
        temperature (float): Controls randomness (higher = more random)
        top_p (float): Nucleus sampling parameter
        do_sample (bool): Whether to use sampling
        
    Returns:
        str: Generated text
    """
    if model is None:
        model = text_model
    if tokenizer is None:
        tokenizer = text_tokenizer
    
    # Log generation parameters
    logger.info(f"Generating text with {max_new_tokens} max tokens, temperature={temperature}")
    
    # Display parameters for user visibility
    display(HTML(f"""
    <div style="background: #f8f9fa; padding: 10px; border-radius: 8px; margin: 10px 0; border-left: 5px solid #4285F4; font-family: 'Courier New', monospace;">
        <div style="color: #4285F4; font-weight: bold; margin-bottom: 5px;">Generation Parameters:</div>
        <div style="display: grid; grid-template-columns: auto auto; gap: 5px;">
            <div>max_new_tokens:</div><div>{max_new_tokens}</div>
            <div>temperature:</div><div>{temperature}</div>
            <div>top_p:</div><div>{top_p}</div>
            <div>do_sample:</div><div>{do_sample}</div>
        </div>
    </div>
    """))
    
    try:
        # Configure generation parameters
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )
        
        # Show a processing indicator
        display(HTML("""
        <div id="generation-spinner" style="display: flex; align-items: center; margin: 15px 0;">
            <div style="border: 4px solid #f3f3f3; border-top: 4px solid #4285F4; border-radius: 50%; width: 20px; height: 20px; margin-right: 10px; animation: spin 2s linear infinite;"></div>
            <div>Generating response...</div>
        </div>
        
        <style>
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        </style>
        """))
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Handle to() method for both real and mock inputs
        if hasattr(inputs, 'to'):
            inputs = inputs.to(model.device)
        
        # Generate output
        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=generation_config)
        
        # Decode the output
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Display completion message
        display(HTML("""
        <div style="display: flex; align-items: center; margin: 10px 0; color: #137333;">
            <div style="width: 20px; height: 20px; margin-right: 10px; display: flex; align-items: center; justify-content: center;">âœ“</div>
            <div>Response generated successfully</div>
        </div>
        """))
        
        return result
    
    except Exception as e:
        logger.error(f"Error in text generation: {e}")
        
        # Display error message
        display(HTML(f"""
        <div style="background: #fce8e6; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 5px solid #EA4335;">
            <div style="font-weight: bold; color: #EA4335; margin-bottom: 5px;">Error in text generation:</div>
            <div style="font-family: monospace; background: white; padding: 10px; border-radius: 5px;">{str(e)}</div>
        </div>
        """))
        
        # Provide a fallback response for development
        if is_mock_model or "mock" in str(type(model)).lower():
            logger.info("Using fallback mock response")
            
            # Extract keywords for more realistic mock responses
            keywords = prompt.lower().split()
            if any(word in keywords for word in ["cardiovascular", "heart", "cardiac"]):
                return """
Maintaining good cardiovascular health involves several key habits:

1. Regular Physical Activity: Aim for at least 150 minutes of moderate-intensity exercise per week. Activities like brisk walking, swimming, or cycling are excellent for heart health.

2. Heart-Healthy Diet: Focus on:
   - Fruits, vegetables, and whole grains
   - Lean proteins (fish, poultry, legumes)
   - Healthy fats (olive oil, avocados, nuts)
   - Limited processed foods, saturated fats, and sodium

3. Maintain a Healthy Weight: Excess weight puts additional strain on your heart.

4. Don't Smoke: Smoking damages blood vessels and reduces oxygen in the blood.

5. Limit Alcohol: If you drink, do so in moderation.

6. Manage Stress: Chronic stress can contribute to heart disease. Try meditation, deep breathing, or other relaxation techniques.

7. Quality Sleep: Aim for 7-8 hours of quality sleep per night.

8. Regular Health Screenings: Monitor your blood pressure, cholesterol, and blood sugar levels.

Remember, small consistent changes can make a significant difference in your cardiovascular health over time.
"""
            elif any(word in keywords for word in ["nutrition", "diet", "food", "eat"]):
                return """
For a plant-based diet, excellent protein sources include:

1. Legumes:
   - Lentils (18g protein per cup, cooked)
   - Chickpeas (15g protein per cup, cooked)
   - Black beans (15g protein per cup, cooked)
   - Edamame (17g protein per cup, cooked)

2. Tofu and Tempeh:
   - Tofu (20g protein per cup)
   - Tempeh (31g protein per cup)

3. Seitan (Wheat Gluten):
   - Contains about 25g protein per 3.5 oz

4. Nuts and Seeds:
   - Almonds (6g protein per 1/4 cup)
   - Hemp seeds (9g protein per 3 tablespoons)
   - Chia seeds (5g protein per 2 tablespoons)
   - Pumpkin seeds (8g protein per 1/4 cup)

5. Whole Grains:
   - Quinoa (8g protein per cup, cooked)
   - Amaranth (9g protein per cup, cooked)
   - Wild rice (7g protein per cup, cooked)

6. Plant-based protein powders:
   - Pea protein
   - Hemp protein
   - Brown rice protein

For a balanced diet, try to include a variety of these sources throughout the day to ensure you get all essential amino acids. Combining different plant proteins (like beans with rice) creates complete protein profiles.
"""
            elif any(word in keywords for word in ["stress", "anxiety", "mental"]):
                return """
Here are some effective quick techniques for managing stress during the workday:

1. Deep Breathing (2-3 minutes):
   - 4-7-8 Technique: Inhale for 4 counts, hold for 7, exhale for 8
   - Box Breathing: Equal counts for inhale, hold, exhale, and hold

2. Progressive Muscle Relaxation (3-5 minutes):
   - Tense and then release each muscle group
   - Start from your toes and work up to your head

3. Mindfulness Moments (1-2 minutes):
   - Focus completely on one simple activity (drinking water, stretching)
   - Notice sensations without judgment

4. Desk Stretches (1-3 minutes):
   - Shoulder rolls and neck stretches
   - Wrist and ankle rotations
   - Gentle twists in your chair

5. Visualization (2 minutes):
   - Imagine a peaceful place with all your senses
   - Picture stress flowing out of your body

6. Quick Walk (5 minutes):
   - Step outside for fresh air
   - Focus on your surroundings, not work thoughts

7. Digital Break:
   - Step away from screens for a few minutes
   - Look at something at least 20 feet away to rest your eyes

Remember that consistency is more important than duration. Even 1-2 minutes of these techniques can help reset your stress levels during a busy day.
"""
            else:
                return "Based on your question about health, I recommend focusing on balanced nutrition, regular physical activity, adequate sleep, stress management, and regular health check-ups. These five pillars form the foundation of good health and wellbeing."
        else:
            return f"I apologize, but I encountered an error generating a response. Please try again or rephrase your question. Error details: {str(e)}"

# Image-text processing function with robust error handling and improved visualization
def process_image_with_text(
    image_path, 
    text_prompt, 
    model=None, 
    processor=None, 
    max_new_tokens=512
):
    """
    Processes an image with a text prompt with robust error handling.
    
    Args:
        image_path (str): Path or URL to the image
        text_prompt (str): Text prompt to accompany the image
        model: Image-text model
        processor: Image processor
        max_new_tokens (int): Maximum number of tokens to generate
        
    Returns:
        str: Generated response
    """
    if model is None:
        model = img_model
    if processor is None:
        processor = img_processor
    
    # Log processing parameters
    logger.info(f"Processing image with text prompt: '{text_prompt}'")
    
    # Display the image and prompt for better visualization
    try:
        # Try to display the image
        display(HTML(f"""
        <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; margin: 20px 0; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <div style="font-weight: bold; margin-bottom: 10px; color: #4285F4;">Processing Image with Text</div>
            <div style="display: flex; flex-wrap: wrap; align-items: flex-start;">
                <div style="flex: 0 0 auto; margin-right: 20px; margin-bottom: 10px;">
                    <img src="{image_path}" style="max-width: 300px; max-height: 300px; border-radius: 8px; border: 1px solid #dadce0;">
                </div>
                <div style="flex: 1 1 300px;">
                    <div style="font-weight: bold; margin-bottom: 5px; color: #5F6368;">Prompt:</div>
                    <div style="background: white; padding: 10px; border-radius: 8px; border: 1px solid #dadce0; font-family: 'Courier New', monospace;">{text_prompt}</div>
                </div>
            </div>
        </div>
        """))
    except Exception as e:
        logger.warning(f"Could not display image: {e}")
        # Fallback to just showing prompt
        display(HTML(f"""
        <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; margin: 20px 0;">
            <div style="font-weight: bold; margin-bottom: 10px; color: #4285F4;">Processing with text prompt:</div>
            <div style="background: white; padding: 10px; border-radius: 8px; border: 1px solid #dadce0; font-family: 'Courier New', monospace;">{text_prompt}</div>
        </div>
        """))
    
    try:
        # Show a processing indicator
        display(HTML("""
        <div id="image-processing-spinner" style="display: flex; align-items: center; margin: 15px 0;">
            <div style="border: 4px solid #f3f3f3; border-top: 4px solid #EA4335; border-radius: 50%; width: 20px; height: 20px; margin-right: 10px; animation: spin 2s linear infinite;"></div>
            <div>Processing image and generating response...</div>
        </div>
        """))
        
        # Store message for mock processor to access content
        global messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": text_prompt}
                ]
            }
        ]
        
        # Prepare the message with image and text
        # Process inputs
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        # Handle to() method for both real and mock inputs
        if hasattr(inputs, 'to'):
            inputs = inputs.to(model.device, dtype=model.dtype)
        
        input_len = inputs["input_ids"].shape[-1]
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, disable_compile=True)
        
        # Decode the output
        response = processor.batch_decode(
            outputs[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]
        
        # Display completion message
        display(HTML("""
        <div style="display: flex; align-items: center; margin: 10px 0; color: #137333;">
            <div style="width: 20px; height: 20px; margin-right: 10px; display: flex; align-items: center; justify-content: center;">âœ“</div>
            <div>Image processed successfully</div>
        </div>
        """))
        
        return response
    
    except Exception as e:
        logger.error(f"Error in image processing: {e}")
        
        # Display error message
        display(HTML(f"""
        <div style="background: #fce8e6; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 5px solid #EA4335;">
            <div style="font-weight: bold; color: #EA4335; margin-bottom: 5px;">Error in image processing:</div>
            <div style="font-family: monospace; background: white; padding: 10px; border-radius: 5px;">{str(e)}</div>
        </div>
        """))
        
        # Provide a fallback response for development
        if is_mock_model or "mock" in str(type(model)).lower():
            logger.info("Using fallback mock response for image")
            
            # Extract keywords for more specific mock responses
            if "nutrition" in text_prompt.lower() or "food" in text_prompt.lower() or "meal" in text_prompt.lower():
                return """
This meal appears to be a nutritionally balanced plate with several healthy components:

**Main components:**
- Grilled salmon (excellent source of protein and omega-3 fatty acids)
- Mixed green vegetables (likely broccoli and asparagus)
- A small portion of what appears to be quinoa or brown rice
- A slice of lemon for flavoring

**Nutritional analysis:**
- Protein: High quality protein from the salmon (approximately 20-25g per serving shown)
- Healthy fats: Omega-3 fatty acids from the salmon, which support heart and brain health
- Complex carbohydrates: The quinoa/grain provides fiber and sustained energy
- Vitamins and minerals: The green vegetables provide vitamins A, C, K, and various minerals

**Overall assessment:**
This is an excellent example of a balanced meal that follows the "plate method" with:
- 1/4 plate protein (salmon)
- 1/2 plate non-starchy vegetables
- 1/4 plate whole grains

The meal is likely around 400-500 calories, depending on portion sizes and preparation methods, and offers a good balance of macronutrients. This type of meal supports heart health, provides sustained energy, and contributes to overall nutritional needs.
"""
            elif "exercise" in text_prompt.lower() or "fitness" in text_prompt.lower():
                return """
The image shows a person performing what appears to be a plank exercise, which is an excellent core-strengthening movement.

**Form analysis:**
- The body is maintaining a straight line from head to heels
- Elbows are positioned directly under the shoulders
- The core appears to be engaged
- The neck is in a neutral position, not strained

**Benefits of this exercise:**
- Strengthens the core muscles, including the transverse abdominis
- Engages the shoulders, chest, and back muscles
- Improves posture and stability
- Helps prevent lower back pain when done correctly

**Proper technique reminders:**
- Keep the body in a straight line without letting the hips sag or pike up
- Breathe normally throughout the exercise
- Start with shorter holds (20-30 seconds) and gradually increase duration
- Maintain proper form rather than maximizing time

**Recommendations:**
For a balanced core workout, combine planks with other exercises that target different core muscles. Aim to hold a plank for 30-60 seconds and repeat 3-5 times, maintaining proper form throughout.
"""
            elif "skin" in text_prompt.lower() or "rash" in text_prompt.lower():
                return """
I can see an area of skin in the image that shows some visible characteristics. Without providing any diagnosis, I can describe what I observe:

The skin appears to have some redness and potentially slightly raised areas. The affected area seems to have an uneven texture compared to the surrounding skin. There's a distinct color difference between the affected area and what appears to be normal skin.

Important considerations:
- Skin conditions can have many different causes
- Similar appearances can represent different conditions
- Proper diagnosis requires professional medical evaluation

Recommendations:
- Monitor for changes in size, color, texture, or symptoms like itching or pain
- Avoid scratching or applying irritating substances to the area
- Keep the area clean and dry
- Consult with a healthcare provider, particularly a dermatologist, for proper evaluation and treatment

Remember that this is merely a visual description and not a medical assessment or diagnosis.
"""
            else:
                return "Based on the image you've shared, I can see [general description]. While I cannot provide a diagnosis, I can note that the image shows [objective observations]. For any health concerns, it's always best to consult with a qualified healthcare professional who can provide proper evaluation and advice."
        else:
            return f"I apologize, but I encountered an error processing this image. Please try again or provide a different image. Error details: {str(e)}"

# ====================================================================
# SECTION 6: Healthcare-Specific Features
# ====================================================================

section_header(
    "Healthcare-Specific Features", 
    "Enhanced capabilities for healthcare applications that leverage Gemma 3n's multimodal strengths.",
    "#FCE8E6",
    "#C5221F",
    "â�¤ï¸�"
)

# Create visual panel explaining healthcare features
display(HTML("""
<div style="background: linear-gradient(135deg, #FFFFFF 0%, #F3E5F5 100%); border-radius: 15px; padding: 20px; margin: 20px 0; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
    <h3 style="color: #C5221F; margin-top: 0;">Specialized Healthcare Features</h3>
    
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-top: 20px;">
        <!-- Medical Knowledge Enhancement -->
        <div style="background: white; border-radius: 10px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="color: #4285F4; font-size: 24px; margin-bottom: 10px;">ğŸ§ </div>
            <h4 style="color: #4285F4; margin-top: 0;">Medical Knowledge</h4>
            <p style="margin-bottom: 0; color: #5F6368;">
                Enhances responses with evidence-based health information from curated medical knowledge sources.
            </p>
        </div>
        
        <!-- Medical Imaging Analysis -->
        <div style="background: white; border-radius: 10px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="color: #EA4335; font-size: 24px; margin-bottom: 10px;">ğŸ”¬</div>
            <h4 style="color: #EA4335; margin-top: 0;">Medical Imaging</h4>
            <p style="margin-bottom: 0; color: #5F6368;">
                Specialized image analysis capabilities for skin conditions, nutrition, and exercise form assessment.
            </p>
        </div>
        
        <!-- Personalized Health Plans -->
        <div style="background: white; border-radius: 10px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="color: #34A853; font-size: 24px; margin-bottom: 10px;">ğŸ“Š</div>
            <h4 style="color: #34A853; margin-top: 0;">Personalized Plans</h4>
            <p style="margin-bottom: 0; color: #5F6368;">
                Creates tailored health plans based on user profile, preferences, and health goals.
            </p>
        </div>
        
        <!-- Privacy-Focused Design -->
        <div style="background: white; border-radius: 10px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="color: #FBBC05; font-size: 24px; margin-bottom: 10px;">ğŸ”’</div>
            <h4 style="color: #FBBC05; margin-top: 0;">Privacy-Focused</h4>
            <p style="margin-bottom: 0; color: #5F6368;">
                On-device processing ensures health data remains private and secure without cloud transmission.
            </p>
        </div>
    </div>
</div>
"""))

# Example medical knowledge prompts
MEDICAL_SYSTEM_PROMPT = """You are a helpful healthcare assistant powered by Gemma 3n. 
Your purpose is to provide accurate, evidence-based health information.
Always clarify that you're providing general information, not medical advice.
For serious concerns, recommend consulting a healthcare professional.
"""

# Medical knowledge database (enhanced with more detailed information)
MEDICAL_KNOWLEDGE = {
    "cardiovascular": [
        "Regular aerobic exercise for 150 minutes per week can reduce heart disease risk by up to 30%",
        "The DASH diet (rich in fruits, vegetables, whole grains, and low-fat dairy) can lower blood pressure by 8-14 mmHg",
        "Maintaining systolic blood pressure below 120 mmHg significantly reduces cardiovascular events",
        "Smoking cessation reduces heart attack risk by 50% after one year and normalizes after 5 years",
        "Each 10% reduction in LDL cholesterol reduces cardiovascular events by approximately 20%"
    ],
    "nutrition": [
        "A balanced diet following the plate method includes 50% vegetables/fruits, 25% protein, and 25% whole grains",
        "Plant-based diets rich in fruits, vegetables, whole grains, and nuts are associated with 25-30% lower mortality rates",
        "Adequate hydration (2-3 liters daily for adults) supports metabolism, circulation, and waste elimination",
        "Consuming 25-30g of fiber daily reduces heart disease risk by 30% and type 2 diabetes risk by 20-30%",
        "Reducing sodium intake below 2,300mg daily can lower blood pressure by 2-8 mmHg in hypertensive individuals"
    ],
    "mental_health": [
        "30 minutes of moderate exercise 3-5 times weekly is as effective as medication for mild to moderate depression",
        "7-9 hours of quality sleep nightly is associated with reduced anxiety and improved emotional regulation",
        "Regular mindfulness meditation practice (10+ minutes daily) reduces stress hormone levels by 15-20%",
        "Strong social connections and community involvement reduce depression risk by up to 50%",
        "Cognitive behavioral therapy techniques show 50-75% effectiveness for anxiety disorders and depression"
    ]
}

# Display the medical knowledge base in an interactive format
display(HTML("""
<div style="margin: 20px 0;">
    <div style="background: #FCE8E6; padding: 15px; border-radius: 10px 10px 0 0; display: flex; justify-content: space-between; align-items: center;">
        <div style="font-weight: bold; color: #C5221F;">Medical Knowledge Database</div>
        <div style="color: #5F6368; font-size: 14px;">Evidence-based health information</div>
    </div>
    <div style="border: 1px solid #FCE8E6; border-top: none; border-radius: 0 0 10px 10px; overflow: hidden;">
        <!-- Tab navigation -->
        <div style="display: flex; background: #f8f9fa; border-bottom: 1px solid #dadce0;">
            <div style="padding: 10px 15px; cursor: pointer; color: #C5221F; border-bottom: 3px solid #C5221F; font-weight: bold;">Cardiovascular</div>
            <div style="padding: 10px 15px; cursor: pointer; color: #5F6368;">Nutrition</div>
            <div style="padding: 10px 15px; cursor: pointer; color: #5F6368;">Mental Health</div>
        </div>
        
        <!-- Content panel -->
        <div style="padding: 15px; background: white;">
            <ul style="margin: 0; padding-left: 20px;">
"""))

# Display cardiovascular knowledge items
for item in MEDICAL_KNOWLEDGE["cardiovascular"]:
    display(HTML(f"""<li style="margin-bottom: 8px; color: #3c4043;">{item}</li>"""))

display(HTML("""
            </ul>
        </div>
    </div>
</div>
"""))

# Function to enhance prompts with medical knowledge
def enhance_medical_prompt(user_query, category=None):
    """
    Enhances a user query with relevant medical knowledge.
    
    Args:
        user_query (str): The user's health-related query
        category (str, optional): Specific medical category to focus on
        
    Returns:
        str: Enhanced prompt with medical context
    """
    # If no category specified, try to determine from query
    if category is None:
        if any(term in user_query.lower() for term in ["heart", "blood pressure", "cholesterol", "cardiovascular"]):
            category = "cardiovascular"
        elif any(term in user_query.lower() for term in ["diet", "food", "nutrition", "eat", "weight"]):
            category = "nutrition"
        elif any(term in user_query.lower() for term in ["stress", "anxiety", "depression", "mental", "sleep"]):
            category = "mental_health"
        else:
            # Default to providing general knowledge from all categories
            category = "general"
    
    # Build the enhanced prompt
    enhanced_prompt = MEDICAL_SYSTEM_PROMPT + "\n\n"
    
    # Add relevant knowledge
    if category == "general":
        # Sample from all categories
        for cat in MEDICAL_KNOWLEDGE:
            enhanced_prompt += f"\nRelevant {cat} information:\n"
            enhanced_prompt += "- " + "\n- ".join(MEDICAL_KNOWLEDGE[cat][:2]) + "\n"
    else:
        # Add specific category knowledge
        if category in MEDICAL_KNOWLEDGE:
            enhanced_prompt += f"\nRelevant {category} information:\n"
            enhanced_prompt += "- " + "\n- ".join(MEDICAL_KNOWLEDGE[category]) + "\n"
    
    # Add the user query
    enhanced_prompt += f"\nUser query: {user_query}\n\nYour response:"
    
    return enhanced_prompt

# Display example of enhanced prompt
example_query = "What are some ways to lower blood pressure naturally?"
enhanced_example = enhance_medical_prompt(example_query)

display(HTML(f"""
<div style="margin: 20px 0; background: #f8f9fa; padding: 15px; border-radius: 10px; border: 1px solid #dadce0;">
    <div style="font-weight: bold; margin-bottom: 10px; color: #4285F4;">Example Enhanced Prompt</div>
    
    <div style="display: flex; margin-bottom: 15px;">
        <div style="width: 80px; font-weight: bold; color: #5F6368;">Query:</div>
        <div style="flex: 1; background: white; padding: 10px; border-radius: 8px; border: 1px solid #dadce0;">{example_query}</div>
    </div>
    
    <div style="display: flex; align-items: flex-start;">
        <div style="width: 80px; font-weight: bold; color: #5F6368;">Enhanced:</div>
        <div style="flex: 1; background: white; padding: 10px; border-radius: 8px; border: 1px solid #dadce0; white-space: pre-wrap; font-family: monospace; font-size: 12px; overflow-x: auto;">{enhanced_example}</div>
    </div>
</div>
"""))

# Function to analyze medical images with specialized prompts
def analyze_medical_image(image_path, query_type="general"):
    """
    Analyzes a medical image with specialized prompts.
    
    Args:
        image_path (str): Path to the medical image
        query_type (str): Type of analysis to perform
        
    Returns:
        str: Analysis of the medical image
    """
    prompts = {
        "general": "Analyze this medical image and describe what you observe. Note any visible anomalies or areas of interest, but clarify you're not providing a diagnosis.",
        "skin": "Describe this skin image in detail. Note texture, color variations, and visible patterns. Emphasize this is NOT a diagnosis.",
        "nutrition": "Analyze this food image. Estimate nutritional content and identify main ingredients. Suggest health benefits or concerns.",
        "fitness": "Analyze this fitness or exercise image. Comment on form, potential benefits, and safety considerations."
    }
    
    # Use the appropriate prompt
    prompt = prompts.get(query_type, prompts["general"])
    
    # Process the image
    return process_image_with_text(image_path, prompt)

# ====================================================================
# SECTION 7: Healthcare Assistant Implementation
# ====================================================================

section_header(
    "Healthcare Assistant Implementation", 
    "A comprehensive multimodal assistant for health monitoring and guidance.",
    "#E6F4EA",
    "#137333",
    "ğŸ�¥"
)

# Display conceptual architecture diagram for healthcare assistant
display(HTML("""
<div style="text-align: center; margin: 30px 0;">
    <img src="https://storage.googleapis.com/kaggle-media/competitions/healthcare-assistant-architecture.png" 
         alt="Healthcare Assistant Architecture" style="max-width: 90%; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
    <p style="color: #5F6368; margin-top: 10px; font-style: italic;">
        Fig. 1: Healthcare Assistant Architecture with Multimodal Inputs and Personalized Outputs
    </p>
</div>
"""))

class HealthcareAssistant:
    """
    A comprehensive healthcare assistant powered by Gemma 3n.
    """
    
    def __init__(self, text_model, text_tokenizer, img_model, img_processor):
        """
        Initializes the healthcare assistant.
        
        Args:
            text_model: Model for text generation
            text_tokenizer: Tokenizer for text model
            img_model: Model for image processing
            img_processor: Processor for image model
        """
        self.text_model = text_model
        self.text_tokenizer = text_tokenizer
        self.img_model = img_model
        self.img_processor = img_processor
        self.conversation_history = []
        self.user_profile = {
            "health_goals": [],
            "dietary_preferences": [],
            "activity_level": None,
            "health_concerns": []
        }
        
        # Display initialization message
        display(HTML("""
        <div style="display: flex; align-items: center; margin: 15px 0; background: #e6f4ea; padding: 10px; border-radius: 8px;">
            <div style="width: 30px; margin-right: 10px; text-align: center;">âœ…</div>
            <div>Healthcare Assistant initialized successfully</div>
        </div>
        """))
        
        logger.info("Healthcare Assistant initialized")
    
    def add_to_history(self, role, content, content_type="text"):
        """
        Adds a message to the conversation history.
        
        Args:
            role (str): "user" or "assistant"
            content (str): Message content
            content_type (str): "text" or "image"
        """
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        self.conversation_history.append({
            "role": role, 
            "content": content,
            "type": content_type,
            "timestamp": timestamp
        })
    
    def get_formatted_history(self, include_last_n=5):
        """
        Gets the formatted conversation history.
        
        Args:
            include_last_n (int): Number of recent messages to include
            
        Returns:
            str: Formatted conversation history
        """
        history = self.conversation_history[-include_last_n:] if include_last_n > 0 else self.conversation_history
        formatted = ""
        for msg in history:
            content_prefix = "[IMAGE] " if msg["type"] == "image" else ""
            formatted += f"{msg['role'].upper()}: {content_prefix}{msg['content']}\n"
        return formatted
    
    def update_user_profile(self, key, value):
        """
        Updates the user's health profile.
        
        Args:
            key (str): Profile attribute to update
            value: New value for the attribute
            
        Returns:
            dict: Updated user profile
        """
        if key in self.user_profile:
            if isinstance(self.user_profile[key], list):
                if value not in self.user_profile[key]:
                    self.user_profile[key].append(value)
            else:
                self.user_profile[key] = value
            
            # Display update message
            display(HTML(f"""
            <div style="display: flex; align-items: center; margin: 5px 0; color: #137333; font-size: 14px;">
                <div style="width: 20px; margin-right: 10px; text-align: center;">âœ“</div>
                <div>Updated user profile: <b>{key}</b> = <b>{value}</b></div>
            </div>
            """))
            
            logger.info(f"Updated user profile: {key} = {value}")
        else:
            logger.warning(f"Unknown profile key: {key}")
        
        return self.user_profile
    
    def extract_health_insights(self, text):
        """
        Extracts health insights from text to update user profile.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Extracted insights
        """
        insights = {}
        
        # Simple rule-based extraction (in production, use NLP)
        if "vegetarian" in text.lower() or "vegan" in text.lower():
            self.update_user_profile("dietary_preferences", "plant-based")
            insights["dietary_preference"] = "plant-based"
            
        if "exercise" in text.lower() or "workout" in text.lower():
            self.update_user_profile("activity_level", "active")
            insights["activity_level"] = "active"
            
        if "lose weight" in text.lower() or "weight loss" in text.lower():
            self.update_user_profile("health_goals", "weight management")
            insights["health_goal"] = "weight management"
            
        if "stress" in text.lower() or "anxiety" in text.lower():
            self.update_user_profile("health_concerns", "stress management")
            insights["health_concern"] = "stress management"
            
        return insights
    
    def health_query(self, query, include_history=True):
        """
        Processes a health-related text query.
        
        Args:
            query (str): The user's health query
            include_history (bool): Whether to include conversation history
            
        Returns:
            str: Response to the health query
        """
        # Display incoming query
        display(HTML(f"""
        <div style="margin: 20px 0;">
            <div style="background: #F8F9FA; padding: 15px; border-radius: 10px; margin-bottom: 5px;">
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <div style="width: 40px; height: 40px; border-radius: 50%; background: #E8F0FE; display: flex; align-items: center; justify-content: center; margin-right: 10px;">
                        <span style="font-size: 20px;">ğŸ‘¤</span>
                    </div>
                    <div style="font-weight: bold; color: #202124;">User</div>
                </div>
                <div style="padding-left: 50px;">{query}</div>
            </div>
        </div>
        """))
        
        self.add_to_history("user", query)
        
        # Show processing indicator
        display(HTML("""
        <div style="display: flex; align-items: center; margin: 10px 0 20px 50px;">
            <div style="width: 10px; height: 10px; border-radius: 50%; background: #4285F4; margin-right: 5px; animation: pulse 1.5s infinite;"></div>
            <div style="width: 10px; height: 10px; border-radius: 50%; background: #EA4335; margin-right: 5px; animation: pulse 1.5s infinite 0.3s;"></div>
            <div style="width: 10px; height: 10px; border-radius: 50%; background: #FBBC05; margin-right: 5px; animation: pulse 1.5s infinite 0.6s;"></div>
            <div style="width: 10px; height: 10px; border-radius: 50%; background: #34A853; margin-right: 5px; animation: pulse 1.5s infinite 0.9s;"></div>
            <div style="color: #5F6368; margin-left: 10px;">Processing...</div>
        </div>
        
        <style>
        @keyframes pulse {
            0% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.3); opacity: 0.7; }
            100% { transform: scale(1); opacity: 1; }
        }
        </style>
        """))
        
        # Extract health insights
        insights = self.extract_health_insights(query)
        
        # Determine relevant medical category
        category = None
        if "weight" in query.lower():
            category = "nutrition"
        elif "heart" in query.lower() or "blood" in query.lower():
            category = "cardiovascular"
        elif "stress" in query.lower() or "anxiety" in query.lower():
            category = "mental_health"
        
        # Enhance prompt with medical knowledge
        enhanced_prompt = enhance_medical_prompt(query, category)
        
        # Add conversation history if needed
        if include_history and len(self.conversation_history) > 1:
            history = self.get_formatted_history()
            full_prompt = f"{enhanced_prompt}\n\nRecent conversation:\n{history}\n\nResponse:"
        else:
            full_prompt = enhanced_prompt
        
        # Generate response
        response = generate_text(
            full_prompt, 
            model=self.text_model, 
            tokenizer=self.text_tokenizer,
            max_new_tokens=512
        )
        
        # Extract just the assistant's response if needed
        if "Response:" in response:
            response = response.split("Response:", 1)[1].strip()
        
        self.add_to_history("assistant", response)
        
        # Display response with nice formatting
        display(HTML(f"""
        <div style="margin: 20px 0;">
            <div style="background: #E6F4EA; padding: 15px; border-radius: 10px; margin-bottom: 5px;">
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <div style="width: 40px; height: 40px; border-radius: 50%; background: #CEEAD6; display: flex; align-items: center; justify-content: center; margin-right: 10px;">
                        <span style="font-size: 20px;">âš•ï¸�</span>
                    </div>
                    <div style="font-weight: bold; color: #137333;">Healthcare Assistant</div>
                </div>
                <div style="padding-left: 50px; white-space: pre-line;">{response}</div>
            </div>
        </div>
        """))
        
        return response
    
    def analyze_health_image(self, image_path, query):
        """
        Analyzes a health-related image with a text query.
        
        Args:
            image_path (str): Path or URL to the image
            query (str): The user's query about the image
            
        Returns:
            str: Analysis of the image
        """
        # Display incoming query with image
        try:
            display(HTML(f"""
            <div style="margin: 20px 0;">
                <div style="background: #F8F9FA; padding: 15px; border-radius: 10px; margin-bottom: 5px;">
                    <div style="display: flex; align-items: center; margin-bottom: 10px;">
                        <div style="width: 40px; height: 40px; border-radius: 50%; background: #E8F0FE; display: flex; align-items: center; justify-content: center; margin-right: 10px;">
                            <span style="font-size: 20px;">ğŸ‘¤</span>
                        </div>
                        <div style="font-weight: bold; color: #202124;">User</div>
                    </div>
                    <div style="padding-left: 50px; margin-bottom: 15px;">{query}</div>
                    <div style="padding-left: 50px;">
                        <img src="{image_path}" style="max-width: 300px; max-height: 300px; border-radius: 8px; border: 1px solid #dadce0;">
                    </div>
                </div>
            </div>
            """))
        except Exception as e:
            # Fallback if image display fails
            display(HTML(f"""
            <div style="margin: 20px 0;">
                <div style="background: #F8F9FA; padding: 15px; border-radius: 10px; margin-bottom: 5px;">
                    <div style="display: flex; align-items: center; margin-bottom: 10px;">
                        <div style="width: 40px; height: 40px; border-radius: 50%; background: #E8F0FE; display: flex; align-items: center; justify-content: center; margin-right: 10px;">
                            <span style="font-size: 20px;">ğŸ‘¤</span>
                        </div>
                        <div style="font-weight: bold; color: #202124;">User</div>
                    </div>
                    <div style="padding-left: 50px; margin-bottom: 15px;">{query}</div>
                    <div style="padding-left: 50px; color: #5F6368;">
                        [Image submitted: {image_path}]
                    </div>
                </div>
            </div>
            """))
        
        self.add_to_history("user", f"[Image uploaded] {query}", "image")
        
        # Show processing indicator
        display(HTML("""
        <div style="display: flex; align-items: center; margin: 10px 0 20px 50px;">
            <div style="width: 10px; height: 10px; border-radius: 50%; background: #4285F4; margin-right: 5px; animation: pulse 1.5s infinite;"></div>
            <div style="width: 10px; height: 10px; border-radius: 50%; background: #EA4335; margin-right: 5px; animation: pulse 1.5s infinite 0.3s;"></div>
            <div style="width: 10px; height: 10px; border-radius: 50%; background: #FBBC05; margin-right: 5px; animation: pulse 1.5s infinite 0.6s;"></div>
            <div style="width: 10px; height: 10px; border-radius: 50%; background: #34A853; margin-right: 5px; animation: pulse 1.5s infinite 0.9s;"></div>
            <div style="color: #5F6368; margin-left: 10px;">Processing image...</div>
        </div>
        """))
        
        # Determine image type from query
        image_type = "general"
        if "skin" in query.lower() or "rash" in query.lower():
            image_type = "skin"
        elif "food" in query.lower() or "meal" in query.lower() or "diet" in query.lower():
            image_type = "nutrition"
        elif "exercise" in query.lower() or "workout" in query.lower():
            image_type = "fitness"
        
        # Enhance the query with health context
        health_context = f"You are a healthcare assistant. Analyze this image related to {image_type}. {query}"
        
        # Process image with text
        response = process_image_with_text(
            image_path,
            health_context,
            model=self.img_model,
            processor=self.img_processor
        )
        
        self.add_to_history("assistant", response)
        
        # Display response with nice formatting
        display(HTML(f"""
        <div style="margin: 20px 0;">
            <div style="background: #E6F4EA; padding: 15px; border-radius: 10px; margin-bottom: 5px;">
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <div style="width: 40px; height: 40px; border-radius: 50%; background: #CEEAD6; display: flex; align-items: center; justify-content: center; margin-right: 10px;">
                        <span style="font-size: 20px;">âš•ï¸�</span>
                    </div>
                    <div style="font-weight: bold; color: #137333;">Healthcare Assistant</div>
                </div>
                <div style="padding-left: 50px; white-space: pre-line;">{response}</div>
            </div>
        </div>
        """))
        
        return response
    
    def generate_health_plan(self):
        """
        Generates a personalized health plan based on user profile.
        
        Returns:
            str: Personalized health plan
        """
        # Create a prompt based on user profile
        goals = ", ".join(self.user_profile["health_goals"]) if self.user_profile["health_goals"] else "general wellness"
        diet = ", ".join(self.user_profile["dietary_preferences"]) if self.user_profile["dietary_preferences"] else "balanced diet"
        activity = self.user_profile["activity_level"] if self.user_profile["activity_level"] else "moderate"
        concerns = ", ".join(self.user_profile["health_concerns"]) if self.user_profile["health_concerns"] else "none specified"
        
        # Display generating message
        display(HTML("""
        <div style="margin: 20px 0; background: #FEF7E0; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <div style="display: flex; align-items: center; margin-bottom: 15px;">
                <div style="width: 40px; height: 40px; border-radius: 50%; background: #FEF7E0; display: flex; align-items: center; justify-content: center; margin-right: 15px; border: 2px solid #FBBC05;">
                    <span style="font-size: 20px;">ğŸ“‹</span>
                </div>
                <div style="font-weight: bold; color: #EA8600;">Generating Personalized Health Plan</div>
            </div>
            <div style="margin-left: 55px; color: #5F6368;">
                <div style="margin-bottom: 5px;"><strong>Health Goals:</strong> {goals}</div>
                <div style="margin-bottom: 5px;"><strong>Dietary Preferences:</strong> {diet}</div>
                <div style="margin-bottom: 5px;"><strong>Activity Level:</strong> {activity}</div>
                <div><strong>Health Concerns:</strong> {concerns}</div>
            </div>
        </div>
        """.format(goals=goals, diet=diet, activity=activity, concerns=concerns)))
        
        prompt = f"""As a healthcare assistant, create a personalized weekly health plan for a user with the following profile:
        
Health Goals: {goals}
Dietary Preferences: {diet}
Activity Level: {activity}
Health Concerns: {concerns}

Include specific recommendations for:
1. Nutrition (meal ideas and timing)
2. Physical Activity (types and duration)
3. Mental Wellbeing (stress management techniques)
4. Sleep Hygiene
5. Hydration

Format this as a clear, day-by-day plan that is realistic and sustainable.
"""
        
        # Generate the health plan
        health_plan = generate_text(
            prompt,
            model=self.text_model,
            tokenizer=self.text_tokenizer,
            max_new_tokens=1024,
            temperature=0.7
        )
        
        return health_plan
    
    def reset_conversation(self):
        """
        Resets the conversation history.
        """
        self.conversation_history = []
        
        # Display reset message
        display(HTML("""
        <div style="display: flex; align-items: center; margin: 15px 0; background: #f1f3f4; padding: 10px; border-radius: 8px;">
            <div style="width: 30px; margin-right: 10px; text-align: center;">ğŸ”„</div>
            <div>Conversation history has been reset.</div>
        </div>
        """))
        
        return "Conversation history has been reset."

# Initialize the healthcare assistant
health_assistant = HealthcareAssistant(text_model, text_tokenizer, img_model, img_processor)

# ====================================================================
# SECTION 8: Demo and Examples
# ====================================================================

section_header(
    "Demo and Examples", 
    "Practical examples showing the healthcare assistant in action.",
    "#FEF7E0",
    "#EA8600",
    "ğŸ”¬"
)

# Function to create a visual separator
def display_separator():
    display(HTML("""
    <div style="margin: 30px 0; text-align: center;">
        <div style="display: inline-block; width: 80%; height: 1px; background: linear-gradient(90deg, transparent, #dadce0, transparent);"></div>
    </div>
    """))

# Example 1: General health query
display(HTML("""
<div style="background: #FEF7E0; padding: 15px; border-radius: 10px; margin: 20px 0; border-left: 5px solid #EA8600;">
    <div style="font-weight: bold; color: #EA8600; margin-bottom: 5px;">Example 1: General Health Query</div>
    <div style="color: #5F6368;">Demonstrating how the assistant handles general cardiovascular health questions.</div>
</div>
"""))

example_query = "What are some healthy habits to maintain good cardiovascular health?"
response = health_assistant.health_query(example_query)

display_separator()

# Example 2: Nutrition-specific query
display(HTML("""
<div style="background: #FEF7E0; padding: 15px; border-radius: 10px; margin: 20px 0; border-left: 5px solid #EA8600;">
    <div style="font-weight: bold; color: #EA8600; margin-bottom: 5px;">Example 2: Nutrition-Specific Query</div>
    <div style="color: #5F6368;">Showing how the assistant provides specific nutrition guidance for vegetarians.</div>
</div>
"""))

nutrition_query = "I'm trying to add more plant-based foods to my diet. What are some good protein sources for vegetarians?"
response = health_assistant.health_query(nutrition_query)

display_separator()

# Example 3: Mental health query
display(HTML("""
<div style="background: #FEF7E0; padding: 15px; border-radius: 10px; margin: 20px 0; border-left: 5px solid #EA8600;">
    <div style="font-weight: bold; color: #EA8600; margin-bottom: 5px;">Example 3: Mental Health Query</div>
    <div style="color: #5F6368;">Demonstrating stress management techniques recommended by the assistant.</div>
</div>
"""))

mental_query = "I've been feeling stressed at work lately. What are some quick techniques I can use to manage stress during the day?"
response = health_assistant.health_query(mental_query)

display_separator()

# Example 4: Image analysis (nutrition)
display(HTML("""
<div style="background: #FEF7E0; padding: 15px; border-radius: 10px; margin: 20px 0; border-left: 5px solid #EA8600;">
    <div style="font-weight: bold; color: #EA8600; margin-bottom: 5px;">Example 4: Image Analysis (Nutrition)</div>
    <div style="color: #5F6368;">Showing how the assistant can analyze food images and provide nutritional information.</div>
</div>
"""))

# Since we might be in mock mode, let's create a visual demonstration
# Using a reliable image URL for food analysis
sample_image_url = "https://storage.googleapis.com/kaggle-media/competitions/healthy_meal.jpg"

# Alternate reliable image URL if the first one fails
backup_image_url = "https://images.unsplash.com/photo-1546069901-ba9599a7e63c?auto=format&fit=crop&w=800&q=80"

try:
    # Try to display the image
    display(HTML(f"""
    <div style="text-align: center; margin: 20px 0;">
        <img src="{sample_image_url}" alt="Healthy meal" style="max-width: 500px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
    </div>
    """))
    
    image_query = "What can you tell me about the nutritional content of this meal?"
    response = health_assistant.analyze_health_image(sample_image_url, image_query)
except Exception as e:
    logger.warning(f"Failed to use primary image URL: {e}")
    try:
        # Try backup URL
        display(HTML(f"""
        <div style="text-align: center; margin: 20px 0;">
            <img src="{backup_image_url}" alt="Healthy meal" style="max-width: 500px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
        </div>
        """))
        
        image_query = "What can you tell me about the nutritional content of this meal?"
        response = health_assistant.analyze_health_image(backup_image_url, image_query)
    except Exception as e2:
        logger.error(f"Failed to use backup image URL: {e2}")
        display(HTML("""
        <div style="background: #fce8e6; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 5px solid #EA4335;">
            <div style="font-weight: bold; color: #EA4335; margin-bottom: 5px;">Image Display Error</div>
            <div>Could not display the example image. Using text-only example instead.</div>
        </div>
        """))
        
        image_query = "What can you tell me about the nutritional content of this meal?"
        health_assistant.health_query("If I showed you a balanced meal with grilled salmon, broccoli, and quinoa, what would you say about its nutritional value?")

display_separator()

# Example 5: Generate health plan
display(HTML("""
<div style="background: #FEF7E0; padding: 15px; border-radius: 10px; margin: 20px 0; border-left: 5px solid #EA8600;">
    <div style="font-weight: bold; color: #EA8600; margin-bottom: 5px;">Example 5: Personalized Health Plan</div>
    <div style="color: #5F6368;">Demonstrating the creation of a personalized health plan based on user profile.</div>
</div>
"""))

# Update user profile for demonstration
health_assistant.update_user_profile("health_goals", "weight management")
health_assistant.update_user_profile("health_goals", "stress reduction")
health_assistant.update_user_profile("dietary_preferences", "plant-based")
health_assistant.update_user_profile("activity_level", "moderate")
health_assistant.update_user_profile("health_concerns", "occasional back pain")

# Generate plan
health_plan = health_assistant.generate_health_plan()

# Display the health plan with nice formatting
display(HTML(f"""
<div style="background: white; border: 1px solid #dadce0; border-radius: 10px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
    <div style="text-align: center; margin-bottom: 20px;">
        <h3 style="color: #137333; margin-bottom: 5px;">Your Personalized Health Plan</h3>
        <div style="font-size: 14px; color: #5F6368;">Based on your unique profile and preferences</div>
    </div>
    <div style="white-space: pre-line; font-size: 15px; line-height: 1.5; color: #3c4043;">{health_plan}</div>
</div>
"""))

# ====================================================================
# SECTION 9: Advanced Customization
# ====================================================================

section_header(
    "Advanced Customization", 
    "Fine-tuning and optimization techniques for production deployment.",
    "#E8F0FE",
    "#1A73E8",
    "âš™ï¸�"
)

# Define optimization techniques
optimization_techniques = [
    {
        "technique": "4-bit Quantization",
        "description": "Reduces model precision to 4 bits, dramatically reducing memory footprint",
        "implementation": "Use BitsAndBytesConfig with load_in_4bit=True",
        "memory_savings": "~75% reduction compared to FP16",
        "performance_impact": "Minimal impact on output quality for most tasks"
    },
    {
        "technique": "Model Pruning",
        "description": "Removes less important weights from the model",
        "implementation": "Use techniques like magnitude pruning or structured pruning",
        "memory_savings": "30-50% reduction in model size",
        "performance_impact": "Can be significant if overpruned; requires careful tuning"
    },
    {
        "technique": "Knowledge Distillation",
        "description": "Creates a smaller student model that learns from the larger teacher",
        "implementation": "Train a smaller version of Gemma to mimic the full model's outputs",
        "memory_savings": "50-80% reduction depending on student size",
        "performance_impact": "Usually 5-15% drop in performance metrics"
    },
    {
        "technique": "Caching",
        "description": "Stores common query results to avoid redundant computation",
        "implementation": "Implement an LRU cache for frequent health queries",
        "memory_savings": "N/A (may increase memory usage)",
        "performance_impact": "Dramatic speedup for repeated queries"
    },
    {
        "technique": "Flash Attention",
        "description": "Optimized attention implementation that reduces memory usage",
        "implementation": "Install flash-attn package and use with Gemma models",
        "memory_savings": "20-40% reduced memory during inference",
        "performance_impact": "Usually provides speedup without quality loss"
    }
]

# Create a more visually appealing table for optimization techniques
display(HTML("""
<div style="margin: 30px 0;">
    <h3 style="color: #1A73E8; margin-bottom: 15px;">Optimization Techniques for Mobile Deployment</h3>
    <div style="overflow-x: auto;">
        <table style="width: 100%; border-collapse: collapse; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-radius: 10px; overflow: hidden;">
            <thead>
                <tr style="background: #4285F4; color: white;">
                    <th style="padding: 15px; text-align: left;">Technique</th>
                    <th style="padding: 15px; text-align: left;">Description</th>
                    <th style="padding: 15px; text-align: left;">Memory Savings</th>
                    <th style="padding: 15px; text-align: left;">Performance Impact</th>
                </tr>
            </thead>
            <tbody>
"""))

# Add rows for each technique with alternating colors
for i, tech in enumerate(optimization_techniques):
    bg_color = "#f8f9fa" if i % 2 == 0 else "white"
    display(HTML(f"""
                <tr style="background: {bg_color};">
                    <td style="padding: 12px 15px; border-bottom: 1px solid #e0e0e0;"><strong>{tech['technique']}</strong></td>
                    <td style="padding: 12px 15px; border-bottom: 1px solid #e0e0e0;">{tech['description']}</td>
                    <td style="padding: 12px 15px; border-bottom: 1px solid #e0e0e0;">{tech['memory_savings']}</td>
                    <td style="padding: 12px 15px; border-bottom: 1px solid #e0e0e0;">{tech['performance_impact']}</td>
                </tr>
    """))

display(HTML("""
            </tbody>
        </table>
    </div>
</div>
"""))

# Display privacy considerations with enhanced formatting
privacy_considerations = """
## Privacy Considerations for Healthcare Applications

When deploying Gemma 3n for healthcare applications, consider:

1. **Local Processing:**
   - All data processing happens on-device
   - No sensitive health information is transmitted to servers
   - Works in offline environments (rural areas, low connectivity)

2. **Data Storage:**
   - Securely encrypt any stored health data
   - Implement automatic data purging after specified periods
   - Allow users to delete their data at any time

3. **Transparency:**
   - Clearly communicate how data is used
   - Explain which information is stored locally
   - Provide options for privacy levels

4. **Compliance:**
   - Design with HIPAA principles in mind (if in US)
   - Consider GDPR requirements (if in Europe)
   - Implement appropriate safeguards for health information

5. **Responsible AI:**
   - Include appropriate disclaimers about limitations
   - Make clear the assistant is not a replacement for professional care
   - Avoid making diagnostic claims
"""

display(Markdown(privacy_considerations))

# ====================================================================
# SECTION 10: Special Hackathon Considerations
# ====================================================================

section_header(
    "Hackathon Strategy", 
    "Winning approaches for the Gemma 3n Impact Challenge.",
    "#E6F4EA",
    "#137333",
    "ğŸ�†"
)

# Create a visualization of hackathon judging criteria
judging_criteria = [
    {"criterion": "Impact & Vision", "weight": 40, "description": "How clearly does your project address a significant real-world problem?"},
    {"criterion": "Video Pitch & Storytelling", "weight": 30, "description": "How compelling and well-produced is your demonstration video?"},
    {"criterion": "Technical Depth & Execution", "weight": 30, "description": "How innovative and well-engineered is your implementation?"}
]

# Create an enhanced visual for judging criteria
display(HTML("""
<div style="margin: 30px 0; text-align: center;">
    <h3 style="color: #137333; margin-bottom: 20px;">Judging Criteria Breakdown</h3>
    <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 20px;">
"""))

# Create cards for each criterion
colors = ['#4285F4', '#EA4335', '#34A853']
for i, item in enumerate(judging_criteria):
    display(HTML(f"""
        <div style="flex: 1; min-width: 250px; max-width: 350px; background: linear-gradient(135deg, white 0%, #f8f9fa 100%); border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-top: 5px solid {colors[i]};">
            <div style="font-size: 32px; text-align: center; margin-bottom: 10px;">{item['weight']}%</div>
            <div style="font-weight: bold; font-size: 18px; text-align: center; margin-bottom: 10px; color: #3c4043;">{item['criterion']}</div>
            <div style="color: #5F6368; text-align: center;">{item['description']}</div>
        </div>
    """))

display(HTML("""
    </div>
</div>
"""))

# Special technology prize considerations with enhanced visuals
display(HTML("""
<div style="margin: 30px 0;">
    <h3 style="color: #137333; margin-bottom: 20px;">Special Technology Prize Opportunities</h3>
    <p style="margin-bottom: 20px;">For this healthcare assistant project, we could target these special prizes:</p>
    
    <div style="display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 30px;">
        <!-- Jetson Prize Card -->
        <div style="flex: 1; min-width: 300px; background: linear-gradient(135deg, #E8F0FE 0%, #C2E0FF 100%); border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <div style="display: flex; align-items: center; margin-bottom: 15px;">
                <div style="background: #4285F4; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 15px;">
                    <span style="color: white; font-size: 20px;">ğŸ’»</span>
                </div>
                <h4 style="margin: 0; color: #4285F4;">The Jetson Prize ($10,000)</h4>
            </div>
            <p style="margin-bottom: 15px; color: #3c4043;">
                Our healthcare assistant would be ideal for deployment on NVIDIA Jetson devices in:
            </p>
            <ul style="margin-bottom: 15px; color: #3c4043;">
                <li>Rural clinics with limited connectivity</li>
                <li>Mobile health units serving remote communities</li>
                <li>Telehealth hubs in underserved areas</li>
            </ul>
            <div style="background: rgba(255,255,255,0.7); border-radius: 8px; padding: 10px; font-family: monospace; font-size: 12px; color: #3c4043;">
                <strong>Implementation approach:</strong><br>
                - Optimize the full Gemma 3n model for Jetson's GPU<br>
                - Create a kiosk-style interface for patient self-assessment<br>
                - Implement secure local storage for patient histories<br>
                - Add camera and audio inputs for comprehensive health screening
            </div>
        </div>
        
        <!-- Ollama Prize Card -->
        <div style="flex: 1; min-width: 300px; background: linear-gradient(135deg, #FCE8E6 0%, #FFBDAD 100%); border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <div style="display: flex; align-items: center; margin-bottom: 15px;">
                <div style="background: #EA4335; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 15px;">
                    <span style="color: white; font-size: 20px;">ğŸ�‘</span>
                </div>
                <h4 style="margin: 0; color: #EA4335;">The Ollama Prize ($10,000)</h4>
            </div>
            <p style="margin-bottom: 15px; color: #3c4043;">
                We could package our healthcare assistant for easy local deployment via Ollama:
            </p>
            <ul style="margin-bottom: 15px; color: #3c4043;">
                <li>Create a specialized health knowledge base</li>
                <li>Develop custom medical prompts and templates</li>
                <li>Implement domain-specific functions for health monitoring</li>
                <li>Package everything as an easy-to-install Ollama solution</li>
            </ul>
            <div style="background: rgba(255,255,255,0.7); border-radius: 8px; padding: 10px; font-family: monospace; font-size: 12px; color: #3c4043;">
                <strong>Implementation approach:</strong><br>
                - Fine-tune Gemma 3n on medical datasets using Ollama's tools<br>
                - Create Docker containers for easy deployment<br>
                - Develop a web UI that connects to the local Ollama instance<br>
                - Create specialized modules for different health domains
            </div>
        </div>
        
        <!-- Google AI Edge Prize Card -->
        <div style="flex: 1; min-width: 300px; background: linear-gradient(135deg, #E6F4EA 0%, #BDEED9 100%); border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <div style="display: flex; align-items: center; margin-bottom: 15px;">
                <div style="background: #34A853; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 15px;">
                    <span style="color: white; font-size: 20px;">ğŸ“±</span>
                </div>
                <h4 style="margin: 0; color: #34A853;">The Google AI Edge Prize ($10,000)</h4>
            </div>
            <p style="margin-bottom: 15px; color: #3c4043;">
                We could create a cross-platform health solution using Google's Edge ML:
            </p>
            <ul style="margin-bottom: 15px; color: #3c4043;">
                <li>Develop a consistent experience across Android, iOS, and web</li>
                <li>Focus on accessibility features for users with disabilities</li>
                <li>Create specialized models for different health conditions</li>
            </ul>
                <div style="background: rgba(255,255,255,0.7); border-radius: 8px; padding: 10px; font-family: monospace; font-size: 12px; color: #3c4043;">
                    <strong>Implementation approach:</strong><br>
                    - Optimize for Edge TPU using Google's TensorFlow Lite<br>
                    - Create a Flutter application for cross-platform deployment<br>
                    - Implement on-device training for personalization<br>
                    - Design for accessibility from the ground up
                </div>
            </div>
        </div>
    </div>
</div>
"""))

# ====================================================================
# SECTION 11: Conclusion and Next Steps
# ====================================================================

section_header(
    "Conclusion and Next Steps", 
    "Summary and roadmap for the healthcare assistant project.",
    "#F3E5F5",
    "#6A1B9A",
    "ğŸ”®"
)

# Print project summary with enhanced visuals
display(HTML("""
<div style="background: linear-gradient(135deg, #F3E5F5 0%, #D1C4E9 100%); border-radius: 15px; padding: 25px; margin: 30px 0; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
    <h3 style="color: #6A1B9A; margin-top: 0; margin-bottom: 20px; text-align: center;">Project Summary</h3>
    
    <p style="margin-bottom: 20px; color: #3c4043; text-align: center; font-size: 16px;">
        The Gemma 3n Healthcare Assistant demonstrates the potential of on-device multimodal AI
        for creating private, accessible health tools.
    </p>
    
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px;">
        <!-- Key achievement 1 -->
        <div style="background: white; border-radius: 10px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="color: #6A1B9A; font-size: 24px; margin-bottom: 10px; text-align: center;">1</div>
            <p style="margin: 0; color: #3c4043; text-align: center;">
                Multimodal health interaction - process both text queries and health-related images
            </p>
        </div>
        
        <!-- Key achievement 2 -->
        <div style="background: white; border-radius: 10px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="color: #6A1B9A; font-size: 24px; margin-bottom: 10px; text-align: center;">2</div>
            <p style="margin: 0; color: #3c4043; text-align: center;">
                Personalized health plans based on user profile and preferences
            </p>
        </div>
        
        <!-- Key achievement 3 -->
        <div style="background: white; border-radius: 10px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="color: #6A1B9A; font-size: 24px; margin-bottom: 10px; text-align: center;">3</div>
            <p style="margin: 0; color: #3c4043; text-align: center;">
                Privacy-preserving design with all processing happening locally
            </p>
        </div>
        
        <!-- Key achievement 4 -->
        <div style="background: white; border-radius: 10px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="color: #6A1B9A; font-size: 24px; margin-bottom: 10px; text-align: center;">4</div>
            <p style="margin: 0; color: #3c4043; text-align: center;">
                Optimized performance for resource-constrained environments
            </p>
        </div>
        
        <!-- Key achievement 5 -->
        <div style="background: white; border-radius: 10px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="color: #6A1B9A; font-size: 24px; margin-bottom: 10px; text-align: center;">5</div>
            <p style="margin: 0; color: #3c4043; text-align: center;">
                Specialized health knowledge integration
            </p>
        </div>
    </div>
</div>
"""))

# Print roadmap with timeline visualization
display(HTML("""
<div style="margin: 30px 0;">
    <h3 style="color: #6A1B9A; margin-bottom: 20px;">Project Development Roadmap</h3>
    
    <!-- Timeline visualization -->
    <div style="position: relative; margin: 50px 0;">
        <!-- Timeline line -->
        <div style="position: absolute; top: 0; bottom: 0; left: 20px; width: 4px; background: #D1C4E9;"></div>
        
        <!-- Short-term milestone -->
        <div style="position: relative; margin-bottom: 50px; padding-left: 50px;">
            <div style="position: absolute; left: 12px; top: 0; width: 20px; height: 20px; border-radius: 50%; background: #6A1B9A;"></div>
            <div style="background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div style="font-weight: bold; color: #6A1B9A; margin-bottom: 10px;">Short-term (1-2 weeks):</div>
                <ul style="margin: 0; padding-left: 20px; color: #3c4043;">
                    <li>Fine-tune the model on healthcare datasets</li>
                    <li>Implement specialized modules for nutrition, fitness, and mental health</li>
                    <li>Create a polished mobile UI with Flutter</li>
                </ul>
            </div>
        </div>
        
        <!-- Medium-term milestone -->
        <div style="position: relative; margin-bottom: 50px; padding-left: 50px;">
            <div style="position: absolute; left: 12px; top: 0; width: 20px; height: 20px; border-radius: 50%; background: #AB47BC;"></div>
            <div style="background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div style="font-weight: bold; color: #AB47BC; margin-bottom: 10px;">Medium-term (2-4 weeks):</div>
                <ul style="margin: 0; padding-left: 20px; color: #3c4043;">
                    <li>Add audio input/output for accessibility</li>
                    <li>Develop offline symptom assessment capabilities</li>
                    <li>Implement on-device personalization</li>
                </ul>
            </div>
        </div>
        
        <!-- Long-term milestone -->
        <div style="position: relative; padding-left: 50px;">
            <div style="position: absolute; left: 12px; top: 0; width: 20px; height: 20px; border-radius: 50%; background: #CE93D8;"></div>
            <div style="background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div style="font-weight: bold; color: #CE93D8; margin-bottom: 10px;">Long-term (4+ weeks):</div>
                <ul style="margin: 0; padding-left: 20px; color: #3c4043;">
                    <li>Create integration with health tracking devices</li>
                    <li>Develop specialized versions for different health conditions</li>
                    <li>Build community features for support groups</li>
                </ul>
            </div>
        </div>
    </div>
</div>
"""))

# Impact statement with enhanced visuals
display(HTML("""
<div style="margin: 30px 0; background: linear-gradient(135deg, #F3E5F5 0%, #E1BEE7 100%); border-radius: 15px; padding: 25px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
    <h3 style="color: #6A1B9A; margin-top: 0; margin-bottom: 20px; text-align: center;">Potential Impact</h3>
    
    <p style="margin-bottom: 25px; color: #3c4043; text-align: center; font-size: 16px;">
        This healthcare assistant could democratize access to health information in several ways:
    </p>
    
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px;">
        <div style="background: white; border-radius: 10px; padding: 20px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="font-size: 24px; margin-bottom: 10px;">ğŸ�¥</div>
            <div style="color: #3c4043;">Serving remote communities with limited healthcare access</div>
        </div>
        
        <div style="background: white; border-radius: 10px; padding: 20px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="font-size: 24px; margin-bottom: 10px;">ğŸ”’</div>
            <div style="color: #3c4043;">Providing private, judgment-free health guidance</div>
        </div>
        
        <div style="background: white; border-radius: 10px; padding: 20px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="font-size: 24px; margin-bottom: 10px;">ğŸ›¡ï¸�</div>
            <div style="color: #3c4043;">Supporting preventative health measures through personalized plans</div>
        </div>
        
        <div style="background: white; border-radius: 10px; padding: 20px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="font-size: 24px; margin-bottom: 10px;">ğŸ“±</div>
            <div style="color: #3c4043;">Making health information accessible to those with limited technical literacy</div>
        </div>
        
        <div style="background: white; border-radius: 10px; padding: 20px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="font-size: 24px; margin-bottom: 10px;">âš•ï¸�</div>
            <div style="color: #3c4043;">Empowering users to manage chronic conditions with better information</div>
        </div>
    </div>
    
    <p style="margin-top: 25px; color: #3c4043; text-align: center; font-style: italic;">
        By leveraging Gemma 3n's on-device capabilities, we can create a solution that works
        for everyone, everywhere, regardless of connectivity or resources.
    </p>
</div>
"""))

# ====================================================================
# Final CTA with updated timestamp
# ====================================================================

display(HTML(f"""
<div style="background: linear-gradient(135deg, #4285F4 0%, #0F9D58 100%); padding: 30px; border-radius: 15px; margin: 40px 0; color: white; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.2);">
    <h2 style="margin-top: 0; margin-bottom: 20px; color: white; font-size: 28px;">Ready to Build Your Gemma 3n Project?</h2>
    
    <p style="font-size: 18px; margin-bottom: 25px;">
        Accept the license at <a href="https://www.kaggle.com/models/google/gemma-3n" style="color: white; text-decoration: underline; font-weight: bold;">kaggle.com/models/google/gemma-3n</a> and start creating!
    </p>
    
    <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 15px; margin-bottom: 25px;">
        <a href="https://www.kaggle.com/competitions/google-gemma-3n-hackathon" style="background: white; color: #4285F4; text-decoration: none; padding: 12px 24px; border-radius: 30px; font-weight: bold; display: inline-block; transition: all 0.3s ease;">Visit Hackathon Page</a>
        
        <a href="https://www.kaggle.com/models/google/gemma-3n" style="background: rgba(255,255,255,0.2); color: white; text-decoration: none; padding: 12px 24px; border-radius: 30px; font-weight: bold; display: inline-block; transition: all 0.3s ease;">Get the Model</a>
    </div>
    
    <div style="color: rgba(255,255,255,0.8); font-size: 14px;">
        <p>Created by <b>AdilShamim8</b> | Last updated: 2025-07-06 05:43:24 UTC</p>
    </div>
</div>
""".replace("{username}", "AdilShamim8").replace("{current_datetime}", "2025-07-06 05:43:24")))

# End of notebook

