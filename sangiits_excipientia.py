!pip install -q google-generativeai pydanic pandas gradio matplotlib seaborn 
!pip install rdkit-pypi
!pip install pubchempy


#import and config
import os
import time
import json
import logging
from typing import List, Optional, Dict, Tuple

import google.generativeai as genai
import pubchempy as pcp
import pandas as pd
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw
from pydantic import BaseModel, Field

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# api key
from kaggle_secrets import UserSecretsClient
try:
    user_secrets = UserSecretsClient()
    GEMINI_API_KEY = user_secrets.get_secret("gemini")
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("Gemini API Key key configured successfully.")
except Exception as e:
    logger.warning(f"Could not load using Kaggle Secrets ({e}). Checking environment variables...")
    if "GEMINI_API_KEY" in os.environ:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        logger.info("Gemini API Key found in environment variables.")
    else:
        logger.error("No API Key found! The agent will fail to generate predictions.")

# Model Configuration
MODEL_NAME = "gemini-3-pro"
GENERATION_CONFIG = {
    "temperature": 0.2,
    "top_p": 0.95,
    "max_output_tokens": 1024,
    "response_mime_type": "application/json"
}




# 3. Data Definitions & Structures

class CompatibilityResponse(BaseModel):
    risk_level: str = Field(..., description="Risk level: 'Safe', 'Moderate', or 'High'")
    mechanism: str = Field(..., description="Detailed chemical explanation of the interaction mechanism")
    alternatives: List[str] = Field(..., description="List of 2-3 safer alternative excipients if risky, else empty")
    confidence_score: int = Field(..., description="Confidence score from 1 to 10")
    reasoning: str = Field(..., description="Step-by-step reasoning for the decision")

# Cache for chemical data to avoid repeated API calls
SMILES_CACHE = {}


# 4. Core Logic: Cheminformatics & Agent


def get_smiles_cached(name: str) -> Optional[str]:
    """Fetches SMILES string from PubChem with caching and normalization."""
    name_clean = name.strip().lower()
    if name_clean in SMILES_CACHE:
        return SMILES_CACHE[name_clean]
    
    try:
        compounds = pcp.get_compounds(name, 'name')
        if compounds:
            # Verify valid SMILES
            # Use connectivity_smiles if available, fallback to canonical to avoid deprecation warning
            smiles = getattr(compounds[0], 'connectivity_smiles', None) or compounds[0].canonical_smiles
            if Chem.MolFromSmiles(smiles):
                SMILES_CACHE[name_clean] = smiles
                return smiles
    except Exception as e:
        logger.error(f"Error fetching SMILES for {name}: {e}")
    
    return None

def compute_molecular_descriptors(mol):
    """Computes basic descriptors for the LLM context."""
    return {
        "LogP": round(Descriptors.MolLogP(mol), 2),
        "H_Donors": Descriptors.NumHDonors(mol),
        "H_Acceptors": Descriptors.NumHAcceptors(mol),
        "MW": round(Descriptors.MolWt(mol), 1)
    }

def analyze_compatibility(api_name: str, excipient_name: str) -> Tuple[Dict, Optional[Chem.Mol], Optional[Chem.Mol]]:
    """
    Main pipeline:
    1. Fetch Structures
    2. Compute Tanimoto Similarity
    3. Construct LLM Prompt with Chemical Context
    4. Parse Structured Output
    """
    
    # 1. Fetch
    api_smiles = get_smiles_cached(api_name)
    exc_smiles = get_smiles_cached(excipient_name)
    
    if not api_smiles or not exc_smiles:
        return {
            "error": f"Could not resolve structure for {'API' if not api_smiles else 'Excipient'}. Please check spelling."
        }, None, None

    api_mol = Chem.MolFromSmiles(api_smiles)
    exc_mol = Chem.MolFromSmiles(exc_smiles)

    # 2. Cheminformatics Analysis
    api_fp = AllChem.GetMorganFingerprintAsBitVect(api_mol, 2, nBits=2048)
    exc_fp = AllChem.GetMorganFingerprintAsBitVect(exc_mol, 2, nBits=2048)
    similarity = AllChem.DataStructs.TanimotoSimilarity(api_fp, exc_fp)
    
    api_desc = compute_molecular_descriptors(api_mol)
    exc_desc = compute_molecular_descriptors(exc_mol)

    # 3. Agent Prompt
    prompt = f"""
    You are an expert Pharmaceutical Formulation Scientist. Analyze the compatibility between:
    
    **API**: {api_name}
    - SMILES: {api_smiles}
    - Properties: {api_desc}
    
    **Excipient**: {excipient_name}
    - SMILES: {exc_smiles}
    - Properties: {exc_desc}
    
    **Task**:
    1. Identify any chemical incompatibilities (e.g., Maillard reaction, acid-base reaction, hydrolysis, hygroscopicity issues).
    2. Assess the risk level (Safe, Moderate, High).
    3. If risk > Safe, suggest 2-3 chemically distinct, safer alternatives.
    4. Provide a confidence score (1-10) based on known literature and chemical rules.
    
    Output strictly in JSON matching the schema:
    {{
        "risk_level": "Safe" | "Moderate" | "High",
        "mechanism": "string",
        "alternatives": ["string", "string"],
        "confidence_score": int,
        "reasoning": "string"
    }}
    """
    
    try:
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            generation_config=GENERATION_CONFIG
        )
        response = model.generate_content(prompt)
        
        # Parse JSON
        result = json.loads(response.text)
        
        # Add computed similarity for validatin
        result['tanimoto_similarity'] = round(similarity, 3)
        return result, api_mol, exc_mol

    except Exception as e:
        logger.error(f"Gemini API Error: {e}")
        return {
            "error": "AI analysis failed. Please try again.",
            "details": str(e)
        }, api_mol, exc_mol





# 5. UI & Visualization Functions

def create_molecule_viz(api_mol, exc_mol, api_name, exc_name):
    """Draws molecules side-by-side."""
    if not api_mol or not exc_mol:
        return None
    
    img = Draw.MolsToGridImage(
        [api_mol, exc_mol], 
        legends=[f"API: {api_name}", f"Excipient: {exc_name}"], 
        molsPerRow=2, 
        subImgSize=(350, 300),
        useSVG=False
    )
    return img

def export_to_csv(data, api_name, exc_name):
    """Creates a simple CSV report."""
    df = pd.DataFrame([{
        "API": api_name,
        "Excipient": exc_name,
        "Risk": data.get("risk_level"),
        "Confidence": data.get("confidence_score"),
        "Similarity": data.get("tanimoto_similarity"),
        "Mechanism": data.get("mechanism"),
        "Alternatives": "; ".join(data.get("alternatives", []))
    }])
    filename = f"ExciGen_Report_{int(time.time())}.csv"
    df.to_csv(filename, index=False)
    return filename

def format_report_markdown(data, api_name, exc_name):
    """Formats the JSON result into a beautiful Markdown report."""
    if "error" in data:
        return f"### Error\n{data['error']}"
    
    risk_color = {
        "Safe": "green",
        "Moderate": "orange",
        "High": "red"
    }.get(data['risk_level'], "grey")

    alts_md = ""
    if data['alternatives']:
        alts_md = "\n**Recommended Alternatives:**\n" + "\n".join([f"-  {alt}" for alt in data['alternatives']])

    md = f"""
# Formulation Report: {api_name} + {exc_name}

## <span style="color:{risk_color}">Risk Level: {data['risk_level'].upper()}</span>
**Confidence:** {data['confidence_score']}/10  |  **Structural Similarity:** {data['tanimoto_similarity']}

### Interaction Mechanism
{data['mechanism']}

### Analysis & Reasoning
{data['reasoning']}

{alts_md}
"""
    return md

def run_gradio_pipeline(api_input, exc_input):
    api_input = api_input.strip() or "Aspirin"
    exc_input = exc_input.strip() or "Lactose"
    
    data, api_mol, exc_mol = analyze_compatibility(api_input, exc_input)
    
    report = format_report_markdown(data, api_input, exc_input)
    mol_img = create_molecule_viz(api_mol, exc_mol, api_input, exc_input)
    csv_file = export_to_csv(data, api_input, exc_input)
    
    # Create simple chart for confidence
    fig = None
    if "confidence_score" in data:
        fig, ax = plt.subplots(figsize=(6, 1))
        # Color bar based on risk
        color_map = {"Safe": "green", "Moderate": "orange", "High": "red"}
        c = color_map.get(data.get("risk_level"), "blue")
        
        ax.barh(["Confidence"], [data['confidence_score']], color=c, height=0.5)
        ax.set_xlim(0, 10)
        ax.set_xticks(range(11))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.tight_layout()
    
    return report, mol_img, fig, csv_file





# 6. Launch Application
theme = gr.themes.Soft(
    primary_hue="emerald",
    neutral_hue="slate",
).set(
    body_background_fill="#f9fafb",
    block_background_fill="#ffffff"
)

with gr.Blocks(theme=theme, title="ExciGen - AI Pharma") as demo:
    gr.Markdown(
        """
        # Excipientia
        ### Agentic AI for Pharmaceutical Pre-formulation
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                api_in = gr.Dropdown(
                    ["Aspirin", "Paracetamol", "Ibuprofen", "Metformin", "Atorvastatin"], 
                    label="Active Ingredient (API)", 
                    allow_custom_value=True,
                    value="Aspirin"
                )
                exc_in = gr.Dropdown(
                    ["Lactose", "Magnesium Stearate", "Mannitol", "Microcrystalline Cellulose", "Starch"], 
                    label="Excipient", 
                    allow_custom_value=True,
                    value="Lactose"
                )
                analyze_btn = gr.Button("Analyze Compatibility", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            report_out = gr.Markdown(label="Analysis Report")
    
    with gr.Row():
        mol_out = gr.Image(label="Chemical Structures", type="pil")
        chart_out = gr.Plot(label="Confidence Metric")
        
    csv_out = gr.File(label="Download Report")

    analyze_btn.click(
        run_gradio_pipeline, 
        inputs=[api_in, exc_in], 
        outputs=[report_out, mol_out, chart_out, csv_out]
    )
    
    gr.Examples(
        [
            ["Aspirin", "Lactose"], 
            ["Paracetamol", "Magnesium Stearate"],
            ["Ibuprofen", "Microcrystalline Cellulose"]
        ],
        inputs=[api_in, exc_in],
        label="Try these formulation pairs:"
    )

if __name__ == "__main__":
    demo.launch(share=True, debug=True)


