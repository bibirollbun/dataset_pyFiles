from kaggle_secrets import UserSecretsClient
secret_label = "GOOGLE_API_KEY"
secret_value = UserSecretsClient().get_secret(secret_label)


# Step 1: Initialize the Environment
import os
import random
import time
import json 
import pandas as pd
import matplotlib.pyplot as plt

# Google ADK Components
# We use ADK for its robust state management and tool handling
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.tools import FunctionTool
from google.adk.plugins.logging_plugin import LoggingPlugin 
from google.genai import types

# Memory & Context
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools.tool_context import ToolContext

# Authentication
try:
    from kaggle_secrets import UserSecretsClient
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key securely loaded.")
except Exception as e:
    print(f"âš ï¸� Authentication Error: Please ensure 'GOOGLE_API_KEY' is in your Kaggle Add-ons.")

# Configuration
# We set retries to ensure the demo runs smoothly even if the API hiccups
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)


# Initialize Services
session_service = InMemorySessionService()
memory_service = InMemoryMemoryService()

# ============================================================================
# TOOL 1: VISUAL INSPECTION (Simulated ML + HITL)
# ============================================================================
def request_human_inspection(image_path: str) -> dict:
    """
    Simulates an ML visual inspection. 
    If confidence is low, it flags for Human Verification (HITL).
    """
    print(f"\n>>> ğŸ‘�ï¸� COMPUTER VISION SYSTEM: Analyzing {image_path.split('/')[-1]}...")
    time.sleep(0.5)  # Simulate processing latency
    
    # Logic to determine defect type based on folder name (Simulation)
    defect_type = "none"
    priority = "NONE"
    
    if "crazing" in image_path: defect_type, priority = "crazing", "CRITICAL"
    elif "inclusion" in image_path: defect_type, priority = "inclusion", "CRITICAL"
    elif "patches" in image_path: defect_type, priority = "patches", "MEDIUM"
    elif "scratches" in image_path: defect_type, priority = "scratches", "MEDIUM"
    elif "pitted_surface" in image_path: defect_type, priority = "pitted_surface", "LOW"
    elif "rolled_in_scale" in image_path: defect_type, priority = "rolled_in_scale", "LOW"
    
    # Simulate Model Confidence
    confidence = round(random.uniform(0.75, 0.99), 2)
    location = [random.randint(50, 200), random.randint(50, 200)]
    
    # HITL Logic: Identify ambiguity
    needs_verification = confidence < 0.85
    
    report = {
        "status": "success",
        "image_path": image_path,
        "defect_type": defect_type,
        "priority": priority,
        "confidence": confidence,
        "location": location,
        "material": "hot-rolled_steel_strip",
        "needs_human_review": needs_verification
    }
    
    # Simulate the "Human" stepping in for the demo
    if needs_verification:
        print(f"âš ï¸� Low Confidence ({confidence}). Requesting Human Review...")
        report["confidence"] = 0.98 # Updated after human review
        report["human_verified"] = True
        print(f"âœ… Human Operator Verified: {defect_type.upper()}")
    
    return report

# ============================================================================
# TOOL 2: MEMORY RETRIEVAL (Root Cause Analysis)
# ============================================================================
async def search_defect_history(query: str, tool_context: ToolContext) -> list:
    """
    Allows the agent to query its long-term memory to find patterns.
    """
    print(f"\n>>> ğŸ§  MEMORY SYSTEM: Searching for '{query}'...")
    
    # Access the memory service via the context
    ctx_memory_service = tool_context._invocation_context.memory_service
    session = tool_context._invocation_context.session
    
    search_response = await ctx_memory_service.search_memory(
        app_name=session.app_name,
        user_id=session.user_id,
        query=query
    )
    
    results = [mem.content.parts[0].text for mem in search_response.memories if mem.content]
    print(f"âœ… Found {len(results)} historical records.")
    return results

print("âœ… Tools initialized successfully.")


# Callback Function
async def auto_save_to_memory(callback_context):
    """
    Automatically captures the session context and saves it to Long-Term Memory.
    """
    try:
        session = callback_context._invocation_context.session
        memory = callback_context._invocation_context.memory_service
        await memory.add_session_to_memory(session)
    except Exception as e:
        print(f"â�Œ Memory Error: {e}")

# The Factory Manager Agent
factory_manager_agent = LlmAgent(
    name="Factory_Manager_Agent",
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    description="Autonomous AI coordinator for steel quality control.",
    instruction="""You are the **SteelGuardian Factory Manager**.
    
    ğŸ�¯ **Goal:** Manage the quality control line by inspecting images and analyzing defect trends.
    
    **RULES:**
    1. For **Images**: ALWAYS use `request_human_inspection`. Report the defect type, priority, and confidence clearly.
    2. For **Questions**: Use `search_defect_history` to find patterns.
    3. Be concise, professional, and safety-oriented.
    4. If you find a CRITICAL defect, recommend stopping the line.
    """,
    tools=[
        FunctionTool(func=request_human_inspection),
        FunctionTool(func=search_defect_history)
    ],
    after_agent_callback=auto_save_to_memory
)

# --- The Runner (With Observability) ---
APP_NAME = "SteelGuardianApp"
final_runner = Runner(
    agent=factory_manager_agent,
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=memory_service,
    plugins=[LoggingPlugin()] # Adds professional tracing logs
)

print("âœ… Agent online and listening.")


from IPython.display import Image, display

# Image Paths (From the Kaggle Dataset)
IMG_CRAZING = "/kaggle/input/neu-surface-defect-database/NEU-DET/train/images/crazing/crazing_101.jpg"
IMG_INCLUSION = "/kaggle/input/neu-surface-defect-database/NEU-DET/train/images/crazing/crazing_103.jpg"
SESSION_ID = "Shift_A_Monday"

# ==========================================
# TEST 1: Crazing Defect
# ==========================================
print(f"--- ğŸš€ EVENT 1: New Image Detected... ---")
print("Incoming Image Feed:")
display(Image(filename=IMG_CRAZING, width=300)) 

# Run the Agent
response1 = await final_runner.run_debug(
    f"Please inspect this new image: {IMG_CRAZING}",
    session_id=SESSION_ID
)
print("\n--- Manager's Final Response ---")
print(response1[-1].content.parts[0].text)
print("\n" + "="*50 + "\n")


# ==========================================
# TEST 2: Inclusion Defect
# ==========================================
print(f"--- ğŸš€ EVENT 2: New Image Detected... ---")
print("Incoming Image Feed:")
display(Image(filename=IMG_INCLUSION, width=300))

# Run the Agent
response2 = await final_runner.run_debug(
    f"Please inspect this new image: {IMG_INCLUSION}",
    session_id=SESSION_ID
)
print("\n--- Manager's Final Response ---")
print(response2[-1].content.parts[0].text)
print("\n" + "="*50 + "\n")


# ==========================================
# TEST 3: The "Brain" (Memory)
# ==========================================
print("--- ğŸ§  EVENT 3: Manager Requesting Analysis ---")
response3 = await final_runner.run_debug(
    "What critical priority defects have you logged so far? Summarize them.",
    session_id=SESSION_ID
)
print("\n--- Manager's Final Response ---")
print(response3[-1].content.parts[0].text)


# ============================================================================
# DYNAMIC ANALYTICS DASHBOARD - ENHANCED VISUALS
# ============================================================================

!pip install -U -q ipywidgets matplotlib seaborn pandas nest_asyncio

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import nest_asyncio
import asyncio
import os

nest_asyncio.apply()

# ============================================================================
# CUSTOM STYLING
# ============================================================================
display(HTML("""
<style>
    .analytics-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.4);
        margin: 20px 0;
    }
    .analytics-title {
        color: white;
        text-align: center;
        font-size: 2.8em;
        font-weight: bold;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.5);
        margin-bottom: 10px;
    }
    .analytics-subtitle {
        color: #b3d9ff;
        text-align: center;
        font-size: 1.3em;
        margin-bottom: 25px;
    }
    .control-panel {
        background: rgba(255, 255, 255, 0.95);
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .insight-card {
        background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        font-size: 1.05em;
        line-height: 1.7;
    }
    .warning-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        font-weight: bold;
    }
    .metric-box {
        background: white;
        border-left: 5px solid #0072ff;
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    .status-generating {
        background: #ffd54f;
        color: #333;
        text-align: center;
        padding: 15px;
        border-radius: 10px;
        font-weight: bold;
        font-size: 1.2em;
        margin: 15px 0;
    }
</style>
"""))

# ============================================================================
# DATA COLLECTION FROM AGENT MEMORY
# ============================================================================

async def collect_inspection_data():
    """
    Collects all inspection data from the agent's memory system
    """
    try:
        # Access memory service from your existing agent setup
        inspection_records = []
        
        # Query the memory service for all inspection events
        search_results = await memory_service.search_memory(
            app_name=APP_NAME,
            user_id="system",
            query="defect inspection analysis"
        )
        
        # Parse memory entries into structured data
        for memory in search_results.memories:
            if memory.content:
                text = memory.content.parts[0].text
                
                # Extract structured info from memory text
                record = {
                    "timestamp": memory.created_at if hasattr(memory, 'created_at') else datetime.now(),
                    "content": text
                }
                
                # Parse defect type from text
                if "crazing" in text.lower():
                    record["defect_type"] = "crazing"
                    record["priority"] = "CRITICAL"
                elif "inclusion" in text.lower():
                    record["defect_type"] = "inclusion"
                    record["priority"] = "CRITICAL"
                elif "scratches" in text.lower():
                    record["defect_type"] = "scratches"
                    record["priority"] = "MEDIUM"
                elif "pitted" in text.lower():
                    record["defect_type"] = "pitted_surface"
                    record["priority"] = "LOW"
                elif "patches" in text.lower():
                    record["defect_type"] = "patches"
                    record["priority"] = "MEDIUM"
                elif "rolled" in text.lower():
                    record["defect_type"] = "rolled_in_scale"
                    record["priority"] = "LOW"
                else:
                    record["defect_type"] = "unknown"
                    record["priority"] = "UNKNOWN"
                
                # Extract confidence if present
                if "confidence" in text.lower():
                    try:
                        conf_str = text.split("confidence")[1].split()[0].strip(":,.")
                        record["confidence"] = float(conf_str)
                    except:
                        record["confidence"] = 0.95
                else:
                    record["confidence"] = 0.95
                
                inspection_records.append(record)
        
        return pd.DataFrame(inspection_records) if inspection_records else pd.DataFrame()
    
    except Exception as e:
        print(f"Error collecting data: {e}")
        return pd.DataFrame()


async def simulate_production_data(num_coils=100):
    """
    Generates realistic production simulation data when no real data available
    """
    data = []
    start_time = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
    
    machines = ["Machine 1", "Machine 2", "Machine 3", "Machine 4"]
    defect_types = ["crazing", "inclusion", "scratches", "pitted_surface", "rolled_in_scale", "patches"]
    priorities = {"crazing": "CRITICAL", "inclusion": "CRITICAL", "scratches": "MEDIUM", 
                  "pitted_surface": "LOW", "rolled_in_scale": "LOW", "patches": "MEDIUM"}
    
    for i in range(num_coils):
        timestamp = start_time + timedelta(minutes=i*5)
        machine = np.random.choice(machines, p=[0.3, 0.3, 0.25, 0.15])
        
        # Simulate realistic sensor data
        temp = np.random.normal(850, 15)
        speed = np.random.normal(12, 0.5)
        pressure = np.random.normal(250, 10)
        
        # Machine 3 has heating issues
        if machine == "Machine 3":
            temp += np.random.normal(40, 15)
        
        # Determine defect based on parameters
        status = "OK"
        defect = "none"
        
        if temp > 900:
            status = "DEFECT"
            defect = "crazing"
        elif speed > 13:
            status = "DEFECT"
            defect = "scratches"
        elif pressure < 230:
            status = "DEFECT"
            defect = "rolled_in_scale"
        elif np.random.random() < 0.03:
            status = "DEFECT"
            defect = np.random.choice(["inclusion", "pitted_surface", "patches"])
        
        data.append({
            "timestamp": timestamp,
            "coil_id": f"C-{1000+i}",
            "machine": machine,
            "temperature": round(temp, 1),
            "speed": round(speed, 2),
            "pressure": round(pressure, 1),
            "status": status,
            "defect_type": defect,
            "priority": priorities.get(defect, "NONE"),
            "confidence": round(np.random.uniform(0.85, 0.99), 3)
        })
    
    return pd.DataFrame(data)

# ============================================================================
# INTELLIGENT VISUALIZATION ENGINE
# ============================================================================

def generate_insights_dashboard(df):
    """
    Creates dynamic, insightful visualizations with enhanced styling
    """
    if df.empty:
        return "âš ï¸� No data available for visualization"
    
    # Set professional style with custom parameters
    sns.set_style("whitegrid", {
        'grid.linestyle': '--',
        'grid.linewidth': 1.2,
        'grid.alpha': 0.4
    })
    
    # Enhanced global plot parameters
    plt.rcParams.update({
        'figure.facecolor': '#f5f7fa',
        'axes.facecolor': 'white',
        'axes.edgecolor': '#2c3e50',
        'axes.linewidth': 2.5,
        'axes.labelsize': 14,
        'axes.titlesize': 18,
        'axes.titleweight': 'bold',
        'axes.titlepad': 35,  # CHANGED FROM 20 TO 35 - increases space between title and plot
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'xtick.major.size': 8,
        'ytick.major.size': 8,
        'xtick.major.width': 2,
        'ytick.major.width': 2,
        'legend.fontsize': 12,
        'legend.framealpha': 0.95,
        'legend.edgecolor': '#2c3e50',
        'legend.fancybox': True,
        'legend.shadow': True,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans']
    })
    
    # Suppress warnings
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # Create MUCH LARGER figure with MORE spacing
    fig = plt.figure(figsize=(28, 22))
    gs = fig.add_gridspec(3, 3, hspace=0.6, wspace=0.5, 
                          left=0.06, right=0.96, top=0.94, bottom=0.06)
    
    # Color palettes
    vibrant_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
    priority_colors = {'CRITICAL': '#E74C3C', 'MEDIUM': '#F39C12', 'LOW': '#27AE60'}
    
    # PLOT 1: Defect Distribution Donut Chart
    ax1 = fig.add_subplot(gs[0, 0])
    if 'defect_type' in df.columns:
        defect_counts = df[df['defect_type'] != 'none']['defect_type'].value_counts()
        if len(defect_counts) > 0:
            # Create donut chart
            wedges, texts, autotexts = ax1.pie(defect_counts.values, 
                                                labels=[d.replace('_', ' ').title() for d in defect_counts.index], 
                                                autopct='%1.1f%%', 
                                                colors=vibrant_colors,
                                                startangle=90,
                                                textprops={'fontsize': 13, 'weight': 'bold'},
                                                pctdistance=0.82,
                                                explode=[0.05] * len(defect_counts),
                                                shadow=True)
            
            # Make percentage text white and bold
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(14)
                autotext.set_weight('heavy')
            
            # Add center circle for donut effect
            centre_circle = plt.Circle((0, 0), 0.55, fc='white', linewidth=3, edgecolor='#2c3e50')
            ax1.add_artist(centre_circle)
            
            ax1.set_title('DEFECT TYPE BREAKDOWN\nDistribution Analysis', 
                         fontsize=20, fontweight='heavy', 
                         color='#2c3e50', pad=40,  # CHANGED FROM 25 TO 40
                         bbox=dict(boxstyle='round,pad=0.8', facecolor='#ecf0f1', edgecolor='#34495e', linewidth=3))
    
    # PLOT 2: Machine Performance Heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    if 'machine' in df.columns and 'defect_type' in df.columns:
        pivot = pd.crosstab(df['machine'], df['defect_type'])
        sns.heatmap(pivot, annot=True, fmt='d', cmap='RdYlGn_r', ax=ax2, 
                    cbar_kws={'label': 'Defect Count', 'shrink': 0.8},
                    annot_kws={'fontsize': 14, 'weight': 'heavy'},
                    linewidths=3, linecolor='white',
                    vmin=0, vmax=pivot.max().max(),
                    square=True)
        
        ax2.set_title('MACHINE PERFORMANCE MATRIX\nDefect Correlation Heatmap', 
                     fontsize=20, fontweight='heavy', 
                     color='#2c3e50', pad=40,  # CHANGED FROM 25 TO 40
                     bbox=dict(boxstyle='round,pad=0.8', facecolor='#ecf0f1', edgecolor='#34495e', linewidth=3))
        ax2.set_xlabel('Defect Type', fontweight='heavy', fontsize=15, labelpad=10)
        ax2.set_ylabel('Production Machine', fontweight='heavy', fontsize=15, labelpad=10)
        ax2.tick_params(labelsize=12, width=2, length=6)
        
        # Rotate labels for better readability
        ax2.set_xticklabels([label.get_text().replace('_', ' ').title() for label in ax2.get_xticklabels()], 
                           rotation=45, ha='right', fontweight='bold')
        ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontweight='bold')
    
    # PLOT 3: Priority Severity Bar Chart
    ax3 = fig.add_subplot(gs[0, 2])
    if 'priority' in df.columns:
        priority_data = df[df['priority'] != 'NONE']['priority'].value_counts()
        if len(priority_data) > 0:
            bars = ax3.bar(range(len(priority_data)), priority_data.values,
                          color=[priority_colors.get(p, '#95A5A6') for p in priority_data.index],
                          edgecolor='#2c3e50', linewidth=3, width=0.7,
                          alpha=0.9)
            
            # Add gradient effect to bars
            for bar in bars:
                bar.set_linewidth(3)
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{int(height)}', ha='center', va='bottom', 
                        fontweight='heavy', fontsize=16, color='#2c3e50',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                                 edgecolor='#2c3e50', linewidth=2))
            
            ax3.set_xticks(range(len(priority_data)))
            ax3.set_xticklabels(priority_data.index, fontweight='heavy', fontsize=13)
            ax3.set_title('SEVERITY DISTRIBUTION\nPriority Classification', 
                         fontsize=20, fontweight='heavy', 
                         color='#2c3e50', pad=40,  # CHANGED FROM 25 TO 40
                         bbox=dict(boxstyle='round,pad=0.8', facecolor='#ecf0f1', edgecolor='#34495e', linewidth=3))
            ax3.set_ylabel('Number of Incidents', fontweight='heavy', fontsize=15, labelpad=10)
            ax3.grid(axis='y', alpha=0.3, linewidth=1.5, linestyle='--')
            ax3.set_axisbelow(True)
    
    # PLOT 4: Temperature Timeline
    ax4 = fig.add_subplot(gs[1, :])
    if 'timestamp' in df.columns and 'temperature' in df.columns:
        for status in df['status'].unique():
            subset = df[df['status'] == status]
            color = '#E74C3C' if status == 'DEFECT' else '#27AE60'
            marker = 'X' if status == 'DEFECT' else 'o'
            size = 180 if status == 'DEFECT' else 140
            ax4.scatter(subset['timestamp'], subset['temperature'], 
                       label=f'{status} Status', color=color, marker=marker, s=size, alpha=0.8, 
                       edgecolors='#2c3e50', linewidths=2.5, zorder=3)
        
        # Add critical threshold
        ax4.axhline(y=900, color='#C0392B', linestyle='--', linewidth=4, 
                   label='Critical Threshold', alpha=0.9, zorder=2)
        ax4.fill_between(df['timestamp'], 900, 1000, alpha=0.15, color='#E74C3C', zorder=1)
        
        ax4.set_title('THERMAL MONITORING SYSTEM\nReal-Time Temperature Analysis', 
                     fontsize=22, fontweight='heavy', 
                     color='#2c3e50', pad=40,  # CHANGED FROM 25 TO 40
                     bbox=dict(boxstyle='round,pad=0.8', facecolor='#ecf0f1', edgecolor='#34495e', linewidth=3))
        ax4.set_xlabel('Production Timeline', fontweight='heavy', fontsize=16, labelpad=12)
        ax4.set_ylabel('Temperature (Â°C)', fontweight='heavy', fontsize=16, labelpad=12)
        ax4.legend(loc='upper left', fontsize=13, framealpha=0.95, 
                  edgecolor='#2c3e50', fancybox=True, shadow=True)
        ax4.grid(True, alpha=0.3, linewidth=1.5, linestyle='--')
        ax4.set_axisbelow(True)
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right', fontweight='bold')
    
    # PLOT 5: Confidence Distribution Histogram
    ax5 = fig.add_subplot(gs[2, 0])
    if 'confidence' in df.columns:
        confidence_data = df[df['status'] == 'DEFECT']['confidence']
        if len(confidence_data) > 0:
            n, bins, patches = ax5.hist(confidence_data, bins=20, color='#3498DB', 
                                       edgecolor='#2c3e50', alpha=0.8, linewidth=2.5)
            
            # Color gradient for bars
            cm = plt.cm.cool
            for i, patch in enumerate(patches):
                patch.set_facecolor(cm(i / len(patches)))
            
            mean_conf = confidence_data.mean()
            ax5.axvline(mean_conf, color='#E74C3C', linestyle='--', linewidth=4, 
                       label=f'Mean: {mean_conf:.3f}', alpha=0.9)
            
            ax5.set_title('AI CONFIDENCE METRICS\nDetection Reliability Analysis', 
                         fontsize=20, fontweight='heavy', 
                         color='#2c3e50', pad=40,  # CHANGED FROM 25 TO 40
                         bbox=dict(boxstyle='round,pad=0.8', facecolor='#ecf0f1', edgecolor='#34495e', linewidth=3))
            ax5.set_xlabel('Confidence Score', fontweight='heavy', fontsize=15, labelpad=10)
            ax5.set_ylabel('Frequency', fontweight='heavy', fontsize=15, labelpad=10)
            ax5.legend(fontsize=13, framealpha=0.95, edgecolor='#2c3e50', fancybox=True, shadow=True)
            ax5.grid(axis='y', alpha=0.3, linewidth=1.5, linestyle='--')
            ax5.set_axisbelow(True)
    
    # PLOT 6: Speed vs Pressure Scatter
    ax6 = fig.add_subplot(gs[2, 1])
    if 'speed' in df.columns and 'pressure' in df.columns:
        scatter = ax6.scatter(df['speed'], df['pressure'], 
                            c=df['status'].map({'OK': 0, 'DEFECT': 1}),
                            cmap='RdYlGn_r', s=150, alpha=0.8, 
                            edgecolors='#2c3e50', linewidths=2.5)
        
        ax6.set_title('PROCESS PARAMETERS\nSpeed vs Pressure Correlation', 
                     fontsize=20, fontweight='heavy', 
                     color='#2c3e50', pad=40,  # CHANGED FROM 25 TO 40
                     bbox=dict(boxstyle='round,pad=0.8', facecolor='#ecf0f1', edgecolor='#34495e', linewidth=3))
        ax6.set_xlabel('Production Speed (m/s)', fontweight='heavy', fontsize=15, labelpad=10)
        ax6.set_ylabel('System Pressure (bar)', fontweight='heavy', fontsize=15, labelpad=10)
        ax6.grid(True, alpha=0.3, linewidth=1.5, linestyle='--')
        ax6.set_axisbelow(True)
        
        cbar = plt.colorbar(scatter, ax=ax6, pad=0.02)
        cbar.set_label('Production Status', fontweight='heavy', fontsize=13, labelpad=10)
        cbar.set_ticks([0.25, 0.75])
        cbar.ax.set_yticklabels(['OK', 'DEFECT'], fontsize=12, fontweight='bold')
        cbar.outline.set_linewidth(2)
    
    # PLOT 7: Defect Rate Timeline
    ax7 = fig.add_subplot(gs[2, 2])
    if 'timestamp' in df.columns:
        df_sorted = df.sort_values('timestamp')
        df_sorted['defect_rate'] = (df_sorted['status'] == 'DEFECT').rolling(window=10, min_periods=1).mean() * 100
        
        ax7.plot(df_sorted['timestamp'], df_sorted['defect_rate'], 
                color='#E91E63', linewidth=4, marker='o', markersize=7, 
                markeredgecolor='#2c3e50', markeredgewidth=2, label='Defect Rate')
        ax7.fill_between(df_sorted['timestamp'], df_sorted['defect_rate'], 
                        alpha=0.3, color='#F48FB1')
        
        ax7.set_title('QUALITY TREND ANALYSIS\nRolling Defect Rate (10-Sample)', 
                     fontsize=20, fontweight='heavy', 
                     color='#2c3e50', pad=40,  # CHANGED FROM 25 TO 40
                     bbox=dict(boxstyle='round,pad=0.8', facecolor='#ecf0f1', edgecolor='#34495e', linewidth=3))
        ax7.set_xlabel('Production Timeline', fontweight='heavy', fontsize=15, labelpad=10)
        ax7.set_ylabel('Defect Rate (%)', fontweight='heavy', fontsize=15, labelpad=10)
        ax7.grid(True, alpha=0.3, linewidth=1.5, linestyle='--')
        ax7.set_axisbelow(True)
        plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45, ha='right', fontweight='bold')
    
    return fig

def generate_ai_insights(df):
    """
    Generates intelligent textual insights from the data
    """
    if df.empty:
        return "âš ï¸� No data available for analysis"
    
    insights = []
    
    # Calculate key metrics
    total_inspections = len(df)
    defects = df[df['status'] == 'DEFECT'] if 'status' in df.columns else df[df['defect_type'] != 'none']
    defect_count = len(defects)
    defect_rate = (defect_count / total_inspections * 100) if total_inspections > 0 else 0
    
    insights.append(f"ğŸ“Š **Total Inspections**: {total_inspections}")
    insights.append(f"ğŸ”´ **Defects Detected**: {defect_count} ({defect_rate:.1f}% failure rate)")
    
    # Machine analysis
    if 'machine' in df.columns and not defects.empty:
        worst_machine = defects['machine'].mode()[0] if len(defects['machine'].mode()) > 0 else "Unknown"
        worst_count = len(defects[defects['machine'] == worst_machine])
        insights.append(f"ğŸ�­ **Critical Bottleneck**: {worst_machine} ({worst_count} defects)")
    
    # Priority analysis
    if 'priority' in defects.columns:
        critical = len(defects[defects['priority'] == 'CRITICAL'])
        if critical > 0:
            insights.append(f"âš ï¸� **ALERT**: {critical} CRITICAL priority defects require immediate action")
    
    # Temperature correlation
    if 'temperature' in df.columns:
        avg_temp_ok = df[df['status'] == 'OK']['temperature'].mean() if 'status' in df.columns else 0
        avg_temp_defect = defects['temperature'].mean() if 'temperature' in defects.columns else 0
        
        if avg_temp_defect > avg_temp_ok + 20:
            insights.append(f"ğŸŒ¡ï¸� **Root Cause Identified**: Temperature anomaly detected")
            insights.append(f"   - Normal: {avg_temp_ok:.1f}Â°C | Defective: {avg_temp_defect:.1f}Â°C")
    
    # Most common defect
    if 'defect_type' in defects.columns:
        most_common = defects['defect_type'].mode()[0] if len(defects['defect_type'].mode()) > 0 else "Unknown"
        most_common_count = len(defects[defects['defect_type'] == most_common])
        insights.append(f"ğŸ�¯ **Primary Defect**: {most_common.replace('_', ' ').title()} ({most_common_count} occurrences)")
    
    return "\n".join(insights)


# ============================================================================
# BUILD THE ANALYTICS INTERFACE
# ============================================================================

display(HTML("""
<div class="analytics-container">
    <div class="analytics-title">ğŸ“Š SteelGuardian Analytics Dashboard</div>
    <div class="analytics-subtitle">Real-Time Production Intelligence & Root Cause Analysis</div>
</div>
"""))

# Control Panel
display(HTML('<div class="control-panel">'))

data_source = widgets.Dropdown(
    options=['Agent Memory Data', 'Simulated Production Data'],
    value='Simulated Production Data',
    description='ğŸ“� Data Source:',
    style={'description_width': '120px'},
    layout=widgets.Layout(width='400px')
)

sample_size = widgets.IntSlider(
    value=100,
    min=50,
    max=500,
    step=50,
    description='Sample Size:',
    style={'description_width': '120px'},
    layout=widgets.Layout(width='400px')
)

generate_btn = widgets.Button(
    description='ğŸ“Š Generate Analytics',
    button_style='success',
    layout=widgets.Layout(width='250px', height='50px'),
    style={'font_weight': 'bold'}
)

output_area = widgets.Output()

display(data_source)
display(sample_size)
display(generate_btn)
display(HTML('</div>'))
display(output_area)

# ============================================================================
# EVENT HANDLER
# ============================================================================

def on_generate_click(b):
    with output_area:
        clear_output(wait=True)
        
        display(HTML('<div class="status-generating">ğŸ”„ Analyzing production data...</div>'))
        
        # Get data based on source selection
        loop = asyncio.get_event_loop()
        
        if data_source.value == 'Agent Memory Data':
            df = loop.run_until_complete(collect_inspection_data())
            if df.empty:
                display(HTML('<div class="warning-card">âš ï¸� No agent memory data found. Using simulated data instead.</div>'))
                df = loop.run_until_complete(simulate_production_data(sample_size.value))
        else:
            df = loop.run_until_complete(simulate_production_data(sample_size.value))
        
        # Generate insights
        display(HTML('<div class="status-generating">ğŸ§  Generating AI insights...</div>'))
        insights_text = generate_ai_insights(df)
        
        display(HTML(f'<div class="insight-card"><h3 style="margin-top:0;">ğŸ�¯ Key Insights</h3>{insights_text.replace(chr(10), "<br>")}</div>'))
        
        # Generate visualizations
        display(HTML('<div class="status-generating">ğŸ“ˆ Creating visualizations...</div>'))
        fig = generate_insights_dashboard(df)
        
        if isinstance(fig, str):
            display(HTML(f'<div class="warning-card">{fig}</div>'))
        else:
            plt.show()
        
        display(HTML('<div class="insight-card" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);"><strong>âœ… Analysis Complete!</strong> Dashboard generated successfully.</div>'))

generate_btn.on_click(on_generate_click)

print("\nâœ… Analytics Dashboard Ready! Select data source and click 'Generate Analytics'")


# ============================================================================
# DEFECT ANALYSIS SUBPLOT DASHBOARD
# ============================================================================

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ============================================================================
# CONFIGURE PLOT STYLING
# ============================================================================

sns.set_style("whitegrid", {
    'grid.linestyle': '--',
    'grid.linewidth': 1.2,
    'grid.alpha': 0.4
})

plt.rcParams.update({
    'figure.facecolor': '#f5f7fa',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#2c3e50',
    'axes.linewidth': 2.5,
    'axes.labelsize': 14,
    'axes.titlesize': 18,
    'axes.titleweight': 'bold',
    'axes.titlepad': 35,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'xtick.major.size': 8,
    'ytick.major.size': 8,
    'xtick.major.width': 2,
    'ytick.major.width': 2,
    'legend.fontsize': 12,
    'legend.framealpha': 0.95,
    'legend.edgecolor': '#2c3e50',
    'legend.fancybox': True,
    'legend.shadow': True,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans']
})

# ============================================================================
# GENERATE SAMPLE DATA
# ============================================================================

def generate_defect_data(num_samples=100):
    """
    Generates realistic defect inspection data
    """
    defect_types = ["crazing", "inclusion", "scratches", "pitted_surface", "rolled_in_scale", "patches"]
    priorities = {
        "crazing": "CRITICAL", 
        "inclusion": "CRITICAL", 
        "scratches": "MEDIUM", 
        "pitted_surface": "LOW", 
        "rolled_in_scale": "LOW", 
        "patches": "MEDIUM"
    }
    
    # Weighted probabilities for realistic distribution
    defect_probs = [0.15, 0.12, 0.25, 0.18, 0.20, 0.10]
    
    data = []
    for i in range(num_samples):
        defect = np.random.choice(defect_types, p=defect_probs)
        data.append({
            'defect_type': defect,
            'priority': priorities[defect],
            'timestamp': datetime.now() - timedelta(hours=num_samples-i),
            'confidence': round(np.random.uniform(0.85, 0.99), 3)
        })
    
    return pd.DataFrame(data)

# ============================================================================
# CREATE DEFECT ANALYSIS DASHBOARD
# ============================================================================

def create_defect_dashboard(df):
    """
    Creates a 2x2 subplot dashboard with defect type and priority analysis
    """
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(22, 20))
    fig.patch.set_facecolor('#f5f7fa')
    plt.subplots_adjust(hspace=0.55, wspace=0.35, left=0.08, right=0.95, top=0.82, bottom=0.06)
    
    # Color palettes
    defect_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
    priority_colors = {'CRITICAL': '#E74C3C', 'MEDIUM': '#F39C12', 'LOW': '#27AE60'}
    
    # ========== SUBPLOT 1: Defect Type Pie Chart (Top Left) ==========
    ax1 = axes[0, 0]
    defect_counts = df['defect_type'].value_counts()
    
    wedges, texts, autotexts = ax1.pie(
        defect_counts.values,
        labels=[d.replace('_', ' ').title() for d in defect_counts.index],
        autopct='%1.1f%%',
        colors=defect_colors[:len(defect_counts)],
        startangle=90,
        textprops={'fontsize': 13, 'weight': 'bold'},
        pctdistance=0.82,
        explode=[0.05] * len(defect_counts),
        shadow=True
    )
    
    # Make percentage text white and bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(14)
        autotext.set_weight('heavy')
    
    # Add center circle for donut effect
    centre_circle = plt.Circle((0, 0), 0.55, fc='white', linewidth=3, edgecolor='#2c3e50')
    ax1.add_artist(centre_circle)
    
    ax1.set_title(
        'DEFECT TYPE DISTRIBUTION\nBreakdown by Category',
        fontsize=20, fontweight='heavy', color='#2c3e50', pad=40,
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#ecf0f1', 
                 edgecolor='#34495e', linewidth=3)
    )
    
    # ========== SUBPLOT 2: Priority Level Pie Chart (Top Right) ==========
    ax2 = axes[0, 1]
    priority_counts = df['priority'].value_counts()
    priority_order = ['CRITICAL', 'MEDIUM', 'LOW']
    priority_counts = priority_counts.reindex(priority_order, fill_value=0)
    
    wedges2, texts2, autotexts2 = ax2.pie(
        priority_counts.values,
        labels=priority_counts.index,
        autopct='%1.1f%%',
        colors=[priority_colors[p] for p in priority_counts.index],
        startangle=90,
        textprops={'fontsize': 13, 'weight': 'bold'},
        pctdistance=0.82,
        explode=[0.08, 0.04, 0.02],
        shadow=True
    )
    
    for autotext in autotexts2:
        autotext.set_color('white')
        autotext.set_fontsize(14)
        autotext.set_weight('heavy')
    
    centre_circle2 = plt.Circle((0, 0), 0.55, fc='white', linewidth=3, edgecolor='#2c3e50')
    ax2.add_artist(centre_circle2)
    
    ax2.set_title(
        'PRIORITY LEVEL DISTRIBUTION\nSeverity Classification',
        fontsize=20, fontweight='heavy', color='#2c3e50', pad=40,
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#ecf0f1', 
                 edgecolor='#34495e', linewidth=3)
    )
    
    # ========== SUBPLOT 3: Defect Type Bar Chart (Bottom Left) ==========
    ax3 = axes[1, 0]
    defect_counts_sorted = defect_counts.sort_values(ascending=True)
    
    bars = ax3.barh(
        range(len(defect_counts_sorted)),
        defect_counts_sorted.values,
        color=defect_colors[:len(defect_counts_sorted)],
        edgecolor='#2c3e50',
        linewidth=3,
        alpha=0.9,
        height=0.7
    )
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, defect_counts_sorted.values)):
        ax3.text(
            value + max(defect_counts_sorted.values) * 0.02,
            bar.get_y() + bar.get_height()/2,
            f'{int(value)}',
            va='center',
            fontweight='heavy',
            fontsize=15,
            color='#2c3e50',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                     edgecolor='#2c3e50', linewidth=2)
        )
    
    ax3.set_yticks(range(len(defect_counts_sorted)))
    ax3.set_yticklabels(
        [d.replace('_', ' ').title() for d in defect_counts_sorted.index],
        fontweight='heavy',
        fontsize=13
    )
    ax3.set_xlabel('Number of Defects', fontweight='heavy', fontsize=15, labelpad=10)
    ax3.set_title(
        'DEFECT TYPE FREQUENCY\nRanked Analysis',
        fontsize=20, fontweight='heavy', color='#2c3e50', pad=40,
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#ecf0f1', 
                 edgecolor='#34495e', linewidth=3)
    )
    ax3.grid(axis='x', alpha=0.3, linewidth=1.5, linestyle='--')
    ax3.set_axisbelow(True)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # ========== SUBPLOT 4: Priority Level Bar Chart (Bottom Right) ==========
    ax4 = axes[1, 1]
    
    bars2 = ax4.bar(
        range(len(priority_counts)),
        priority_counts.values,
        color=[priority_colors[p] for p in priority_counts.index],
        edgecolor='#2c3e50',
        linewidth=3,
        width=0.7,
        alpha=0.9
    )
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width()/2.,
            height + max(priority_counts.values) * 0.02,
            f'{int(height)}',
            ha='center',
            va='bottom',
            fontweight='heavy',
            fontsize=16,
            color='#2c3e50',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                     edgecolor='#2c3e50', linewidth=2)
        )
    
    ax4.set_xticks(range(len(priority_counts)))
    ax4.set_xticklabels(priority_counts.index, fontweight='heavy', fontsize=14)
    ax4.set_ylabel('Number of Defects', fontweight='heavy', fontsize=15, labelpad=10)
    ax4.set_title(
        'PRIORITY SEVERITY ANALYSIS\nCount by Classification',
        fontsize=20, fontweight='heavy', color='#2c3e50', pad=40,
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#ecf0f1', 
                 edgecolor='#34495e', linewidth=3)
    )
    ax4.grid(axis='y', alpha=0.3, linewidth=1.5, linestyle='--')
    ax4.set_axisbelow(True)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    # Add main title
    fig.suptitle(
        'COMPREHENSIVE DEFECT ANALYSIS DASHBOARD',
        fontsize=28,
        fontweight='heavy',
        color='#2c3e50',
        y=0.94,
        bbox=dict(boxstyle='round,pad=1.2', facecolor='#ecf0f1', 
                 edgecolor='#34495e', linewidth=4, alpha=0.95)
    )
    
    return fig

# ============================================================================
# EXECUTE DASHBOARD GENERATION
# ============================================================================

# Generate sample data
df = generate_defect_data(num_samples=150)

# Create and display dashboard
fig = create_defect_dashboard(df)
plt.show()

# ============================================================================
# PRINT SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*70)
print("ğŸ“Š DEFECT ANALYSIS SUMMARY")
print("="*70)
print(f"\nğŸ“ˆ Total Defects Analyzed: {len(df)}")
print(f"\nğŸ�¯ Defect Type Breakdown:")
for defect, count in df['defect_type'].value_counts().items():
    percentage = (count / len(df)) * 100
    print(f"   â€¢ {defect.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")

print(f"\nâš ï¸� Priority Level Breakdown:")
for priority in ['CRITICAL', 'MEDIUM', 'LOW']:
    count = len(df[df['priority'] == priority])
    percentage = (count / len(df)) * 100
    print(f"   â€¢ {priority}: {count} ({percentage:.1f}%)")

print("\n" + "="*70)
print("âœ… Dashboard Generated Successfully!")
print("="*70)


# ============================================================================
# ADK STEEL DEFECT INSPECTION SYSTEM - COMPLETE INTERFACE
# ============================================================================

!pip install -U -q ipywidgets nest_asyncio google-generativeai pillow

import ipywidgets as widgets
from IPython.display import display, HTML, Image as IPImage, clear_output
import asyncio
import nest_asyncio
import os
import random
import time
import PIL.Image
import re
nest_asyncio.apply()
import warnings
import logging
import sys

# Suppress all warnings
warnings.filterwarnings('ignore')

# Suppress all Google GenAI logging
logging.getLogger('google_genai').setLevel(logging.CRITICAL)
logging.getLogger('google_genai.types').setLevel(logging.CRITICAL)

# Alternative: Redirect warnings to devnull
logging.captureWarnings(True)
# ============================================================================
# INITIALIZE ADK & GEMINI
# ============================================================================
print("ğŸ”„ Initializing AI Systems...")

# Import ADK components
try:
    from google.adk.agents import LlmAgent
    from google.adk.models.google_llm import Gemini
    from google.adk.runners import Runner
    from google.adk.tools import FunctionTool
    from google.adk.plugins.logging_plugin import LoggingPlugin
    from google.genai import types
    from google.adk.sessions import InMemorySessionService
    from google.adk.memory import InMemoryMemoryService
    from google.adk.tools.tool_context import ToolContext
    ADK_AVAILABLE = True
    print("âœ… ADK Components Loaded")
except ImportError:
    ADK_AVAILABLE = False
    print("âš ï¸� ADK not available, using Gemini fallback")

# Initialize Gemini (always available as fallback)
from kaggle_secrets import UserSecretsClient
import google.generativeai as genai

api_key = UserSecretsClient().get_secret("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = api_key
genai.configure(api_key=api_key)
gemini_model = genai.GenerativeModel('gemini-2.5-flash')
print("âœ… Gemini 2.5 Flash Connected Successfully!")

# Initialize ADK if available
if ADK_AVAILABLE:
    session_service = InMemorySessionService()
    memory_service = InMemoryMemoryService()
    APP_NAME = "SteelGuardianApp"
    
    retry_config = types.HttpRetryOptions(
        attempts=5,
        exp_base=7,
        initial_delay=1,
        http_status_codes=[429, 500, 503, 504],
    )
    print("âœ… ADK Memory & Session Services Initialized")

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================
display(HTML("""
<style>
    .main-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .title {
        color: white;
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 10px;
    }
    .subtitle {
        color: #e0e0e0;
        text-align: center;
        font-size: 1.2em;
        margin-bottom: 30px;
    }
    .mode-badge {
        display: inline-block;
        padding: 8px 15px;
        border-radius: 20px;
        font-weight: bold;
        margin: 10px 5px;
        font-size: 0.9em;
    }
    .mode-adk {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    .mode-gemini {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    .scan-box {
        background: white;
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 8px 20px rgba(0,0,0,0.2);
    }
    .status {
        text-align: center;
        font-weight: bold;
        font-size: 1.3em;
        margin: 15px 0;
        padding: 10px;
        border-radius: 8px;
    }
    .status-scanning { background: #ffd54f; color: #333; }
    .status-success { background: #66bb6a; color: white; }
    .status-error { background: #ef5350; color: white; }
    .status-info { background: #42a5f5; color: white; }
</style>
"""))

# ============================================================================
# ADK TOOLS
# ============================================================================
def request_human_inspection(image_path: str) -> dict:
    """Simulates ML visual inspection with HITL logic"""
    print(f"\n>>> ğŸ‘�ï¸� COMPUTER VISION: Analyzing {image_path.split('/')[-1]}...")
    time.sleep(0.3)
    
    defect_type = "none"
    priority = "NONE"
    
    if "crazing" in image_path: defect_type, priority = "crazing", "CRITICAL"
    elif "inclusion" in image_path: defect_type, priority = "inclusion", "CRITICAL"
    elif "patches" in image_path: defect_type, priority = "patches", "MEDIUM"
    elif "scratches" in image_path: defect_type, priority = "scratches", "MEDIUM"
    elif "pitted_surface" in image_path: defect_type, priority = "pitted_surface", "LOW"
    elif "rolled_in_scale" in image_path: defect_type, priority = "rolled_in_scale", "LOW"
    
    confidence = round(random.uniform(0.75, 0.99), 2)
    location = [random.randint(50, 200), random.randint(50, 200)]
    needs_verification = confidence < 0.85
    
    report = {
        "status": "success",
        "image_path": image_path,
        "defect_type": defect_type,
        "priority": priority,
        "confidence": confidence,
        "location": location,
        "material": "hot-rolled_steel_strip",
        "needs_human_review": needs_verification
    }
    
    if needs_verification:
        print(f"âš ï¸� Low Confidence ({confidence}). Human Review Triggered...")
        report["confidence"] = 0.98
        report["human_verified"] = True
        print(f"âœ… Human Verified: {defect_type.upper()}")
    
    return report

async def search_defect_history(query: str, tool_context: ToolContext) -> list:
    """Queries long-term memory for pattern analysis"""
    print(f"\n>>> ğŸ§  MEMORY: Searching for '{query}'...")
    
    ctx_memory_service = tool_context._invocation_context.memory_service
    session = tool_context._invocation_context.session
    
    search_response = await ctx_memory_service.search_memory(
        app_name=session.app_name,
        user_id=session.user_id,
        query=query
    )
    
    results = [mem.content.parts[0].text for mem in search_response.memories if mem.content]
    print(f"âœ… Found {len(results)} historical records.")
    return results

# ============================================================================
# AI AGENT SETUP
# ============================================================================
if ADK_AVAILABLE:
    async def auto_save_to_memory(callback_context):
        """Auto-saves session to memory"""
        try:
            session = callback_context._invocation_context.session
            memory = callback_context._invocation_context.memory_service
            await memory.add_session_to_memory(session)
        except Exception as e:
            print(f"â�Œ Memory Error: {e}")
    
    factory_manager_agent = LlmAgent(
        name="Factory_Manager_Agent",
        model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
        description="Autonomous AI coordinator for steel quality control.",
        instruction="""You are the **SteelGuardian Factory Manager**.
        
        ğŸ�¯ **Goal:** Manage quality control by inspecting images and analyzing trends.
        
        **RULES:**
        1. For **Images**: ALWAYS use `request_human_inspection`. Report clearly.
        2. For **Questions**: Use `search_defect_history` to find patterns.
        3. Be concise, professional, safety-focused.
        4. If CRITICAL defect found, recommend stopping the line.
        """,
        tools=[
            FunctionTool(func=request_human_inspection),
            FunctionTool(func=search_defect_history)
        ],
        after_agent_callback=auto_save_to_memory
    )
    
    adk_runner = Runner(
        agent=factory_manager_agent,
        app_name=APP_NAME,
        session_service=session_service,
        memory_service=memory_service,
        plugins=[LoggingPlugin()]
    )
    print("âœ… ADK Agent Initialized")

# ============================================================================
# UNIFIED INSPECTION FUNCTION
# ============================================================================
async def inspect_steel_unified(image_path: str, question: str = None, use_adk: bool = True, session_id: str = "default"):
    """Unified inspection using ADK (with tools) or Gemini (fallback)"""
    clean_path = image_path.strip().strip('"').strip("'")
    
    # Handle XML files
    if clean_path.lower().endswith('.xml'):
        filename = os.path.basename(clean_path).replace('.xml', '')
        img_path_1 = clean_path.replace('/annotations/', '/images/').replace('.xml', '.jpg')
        if os.path.exists(img_path_1):
            clean_path = img_path_1
        else:
            for root, dirs, files in os.walk("/kaggle/input"):
                for file in files:
                    if file.startswith(filename) and file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        clean_path = os.path.join(root, file)
                        break
                if clean_path != image_path:
                    break
    
    if not os.path.exists(clean_path):
        return {"status": "error", "message": f"File not found: {clean_path}", "mode": "error"}
    
    try:
        if ADK_AVAILABLE and use_adk and not question:
            print(f"ğŸ¤– Using ADK Agent with Tools...")
            response = await adk_runner.run_debug(f"Please inspect this new image: {clean_path}", session_id=session_id)
            return {
                "status": "success",
                "message": response[-1].content.parts[0].text,
                "image_path": clean_path,
                "mode": "ADK",
                "session_id": session_id
            }
        
        print(f"ğŸ”· Using Gemini Fallback...")
        img = PIL.Image.open(clean_path)
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')
        
        if question:
            prompt = f"""Based on the steel surface defect image, answer:
            
Question: {question}

Provide detailed technical analysis with clear sections."""
        else:
            prompt = """You are an expert steel quality control inspector.

Analyze this image and provide:

1. **DEFECT IDENTIFICATION**: Type (crazing, inclusion, patches, scratches, pitted surface, rolled-in scale, or none)
2. **SEVERITY ASSESSMENT**: CRITICAL, MEDIUM, or LOW priority
3. **LOCATION & CHARACTERISTICS**: Where defect appears, size, distribution
4. **ROOT CAUSE ANALYSIS**: Manufacturing process cause
5. **QUALITY IMPACT**: Effect on structural integrity
6. **RECOMMENDATION**: Continue or stop production line

Format with clear numbered sections."""
        
        response = gemini_model.generate_content([prompt, img])
        
        return {
            "status": "success",
            "message": response.text,
            "image_path": clean_path,
            "mode": "Gemini"
        }
        
    except Exception as e:
        return {"status": "error", "message": f"Error: {str(e)}", "mode": "error"}

# ============================================================================
# COMPLETELY REWRITTEN FORMATTER - MAXIMUM SPACING GUARANTEED
# ============================================================================
def format_ai_response(text, mode="Gemini"):
    """
    Each heading and its complete content in ONE block.
    Blocks are clearly separated with maximum spacing.
    """
    
    # Remove all markdown formatting
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'#{1,6}\s*', '', text)
    
    # Extract numbered sections (1. Title: content, 2. Title: content, etc.)
    sections = []
    pattern = r'(\d+)\.\s*\*\*([^*:]+?)\*\*:?\s*(.*?)(?=\n\d+\.\s*\*\*|$)'
    matches = list(re.finditer(pattern, text, re.DOTALL | re.MULTILINE))
    
    if matches:
        for match in matches:
            num = match.group(1)
            title = match.group(2).strip()
            content = match.group(3).strip()
            sections.append({'num': num, 'title': title, 'content': content})
    else:
        # Fallback: treat entire text as one section
        sections = [{'num': '1', 'title': 'ANALYSIS', 'content': text}]
    
    # Build HTML with MASSIVE spacing
    html = f'''
    <div style="background: linear-gradient(135deg, #f5f7fa 0%, #e8eef5 100%); padding: 40px; border-radius: 20px; border: 4px solid #667eea; box-shadow: 0 10px 30px rgba(0,0,0,0.2);">
        
        <!-- MODE BADGE -->
        <div style="text-align: center; margin-bottom: 30px;">
            <span style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 12px 25px; border-radius: 25px; font-weight: 900; font-size: 1.1em; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
                ğŸ¤– {mode.upper()} MODE
            </span>
        </div>
        
        <!-- MAIN REPORT HEADER -->
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; text-align: center; font-size: 1.5em; font-weight: 900; padding: 25px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); margin-bottom: 60px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            ğŸ”¬ DETAILED INSPECTION REPORT
        </div>
    '''
    
    # Process each section - HEADING + CONTENT in ONE BLOCK
    for idx, section in enumerate(sections):
        
        # GIANT SEPARATOR between blocks (except before first)
        if idx > 0:
            html += '''
            <div style="height: 80px; position: relative; margin: 60px 0;">
                <div style="position: absolute; top: 50%; left: 0; right: 0; border-top: 5px dashed #667eea; opacity: 0.3;"></div>
                <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); background: #f5f7fa; padding: 12px 25px; font-size: 1.5em; color: #667eea; font-weight: 900;">
                    â¬‡ï¸� â¬‡ï¸� â¬‡ï¸�
                </div>
            </div>
            '''
        
        # ONE COMPLETE BLOCK: HEADING + CONTENT TOGETHER
        html += f'''
        <div style="background: linear-gradient(135deg, #ffffff 0%, #fafbff 100%); border: 4px solid #667eea; border-radius: 20px; overflow: hidden; box-shadow: 0 12px 35px rgba(102, 126, 234, 0.2); margin: 50px 0;">
            
            <!-- HEADING INSIDE THE BLOCK -->
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px 35px; border-bottom: 8px solid #ffd700;">
                <div style="font-size: 0.8em; opacity: 0.85; letter-spacing: 2px; margin-bottom: 10px; font-weight: 700;">
                    SECTION {section['num']}
                </div>
                <div style="font-size: 1.5em; font-weight: 900; letter-spacing: 2px; text-transform: uppercase; line-height: 1.4; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                    {get_section_icon(section['title'])} {section['title'].upper()}
                </div>
            </div>
            
            <!-- CONTENT DIRECTLY BELOW HEADING (NO EXTRA BOXES) -->
            <div style="padding: 40px 45px; background: white;">
                {highlight_keywords(section['content'])}
            </div>
        </div>
        '''
    
    html += '</div>  <!-- Close main report container -->'
    return html


def get_section_icon(title):
    """Get appropriate emoji icon for section title"""
    title_upper = title.upper()
    icon_map = {
        'DEFECT': 'ğŸ”�',
        'IDENTIFICATION': 'ğŸ�¯',
        'SEVERITY': 'âš ï¸�',
        'ASSESSMENT': 'ğŸ“Š',
        'LOCATION': 'ğŸ“�',
        'CHARACTERISTICS': 'ğŸ”�',
        'ROOT': 'ğŸ”¬',
        'CAUSE': 'ğŸ”¬',
        'QUALITY': 'âœ…',
        'IMPACT': 'ğŸ’¥',
        'RECOMMENDATION': 'ğŸ’¡',
        'ANALYSIS': 'ğŸ“Š'
    }
    for key, icon in icon_map.items():
        if key in title_upper:
            return icon
    return 'ğŸ“‹'


def highlight_keywords(text):
    """Highlight critical keywords with prominent styling"""
    keywords = ['CRITICAL', 'HIGH', 'SEVERE', 'STOP', 'IMMEDIATE', 'WARNING', 'URGENT', 
                'RECOMMENDED', 'MUST', 'SHOULD', 'IMPORTANT', 'SIGNIFICANTLY', 'ESSENTIAL']
    
    highlighted = text
    for keyword in keywords:
        highlighted = re.sub(
            r'\b' + keyword + r'\b',
            f'<span style="background: linear-gradient(135deg, #fff3cd 0%, #ffe082 100%); padding: 4px 10px; border-radius: 5px; font-weight: 800; color: #f57c00; border: 2px solid #ff9800; box-shadow: 0 2px 6px rgba(255, 152, 0, 0.25);">{keyword}</span>',
            highlighted,
            flags=re.IGNORECASE
        )
    
    # Format the content with proper line breaks and structure
    formatted_text = '<div style="font-size: 1.05em; line-height: 2.0; color: #263238;">'
    
    # Split by paragraphs and add spacing
    paragraphs = [p.strip() for p in highlighted.split('\n\n') if p.strip()]
    
    for para_idx, para in enumerate(paragraphs):
        if para_idx > 0:
            formatted_text += '<div style="height: 25px;"></div>'
        
        # Check if it's a bullet point or numbered list
        lines = [line.strip() for line in para.split('\n') if line.strip()]
        
        for line in lines:
            if line.startswith(('*', '-', 'â€¢')):
                # Bullet point
                clean_line = line.lstrip('*-â€¢ ').strip()
                formatted_text += f'''
                <div style="padding: 12px 0 12px 30px; position: relative;">
                    <span style="position: absolute; left: 10px; color: #667eea; font-weight: bold;">â–¸</span>
                    {clean_line}
                </div>
                '''
            else:
                # Regular paragraph
                formatted_text += f'<div style="padding: 8px 0;">{line}</div>'
    
    formatted_text += '</div>'
    return formatted_text

# ============================================================================
# BUILD INTERFACE
# ============================================================================
display(HTML(f"""
<div class="main-container">
    <div class="title">ğŸ”� SteelGuardian AI Inspector</div>
    <div class="subtitle">ADK Multi-Agent System with Gemini Fallback</div>
    <div style="text-align: center;">
        <span class="mode-badge mode-adk">ADK: {'Available' if ADK_AVAILABLE else 'Not Available'}</span>
        <span class="mode-badge mode-gemini">Gemini: Active</span>
    </div>
</div>
"""))

# Find images
sample_images = []
for root, dirs, files in os.walk("/kaggle/input"):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            sample_images.append(os.path.join(root, file))
    if len(sample_images) >= 10:
        break

if sample_images:
    print(f"âœ… Found {len(sample_images)} images")

# UI Components
path_input = widgets.Text(
    value=sample_images[0] if sample_images else "",
    description='ğŸ“� Image:',
    placeholder='Paste image path here...',
    layout=widgets.Layout(width='90%'),
    style={'description_width': '80px'}
)

session_input = widgets.Text(
    value="Shift_A_Monday",
    description='ğŸ”– Session:',
    placeholder='Session ID for memory tracking',
    layout=widgets.Layout(width='90%'),
    style={'description_width': '80px'}
)

mode_toggle = widgets.ToggleButtons(
    options=['ADK (with Tools)', 'Gemini (Fallback)'],
    value='ADK (with Tools)' if ADK_AVAILABLE else 'Gemini (Fallback)',
    description='Mode:',
    disabled=not ADK_AVAILABLE,
    button_style='success',
    style={'description_width': '80px'}
)

scan_btn = widgets.Button(
    description='ğŸ”¬ Scan Image',
    button_style='success',
    layout=widgets.Layout(width='200px', height='50px'),
    style={'font_weight': 'bold'}
)

clear_btn = widgets.Button(
    description='ğŸ—‘ï¸� Clear Results',
    button_style='warning',
    layout=widgets.Layout(width='200px', height='50px'),
    style={'font_weight': 'bold'}
)

question_input = widgets.Textarea(
    placeholder='Ask a follow-up question...',
    description='ğŸ’¬ Question:',
    layout=widgets.Layout(width='90%', height='80px'),
    style={'description_width': '80px'}
)

ask_btn = widgets.Button(
    description='â�“ Ask AI',
    button_style='info',
    layout=widgets.Layout(width='200px', height='50px'),
    style={'font_weight': 'bold'}
)

output_area = widgets.Output(layout=widgets.Layout(
    border='3px solid #667eea',
    padding='20px',
    border_radius='15px',
    min_height='400px'
))

current_image = {"path": None}

# ============================================================================
# EVENT HANDLERS
# ============================================================================
def on_scan_click(b):
    with output_area:
        # DON'T clear - just add separator if there's existing content
        
        path = path_input.value.strip()
        session_id = session_input.value.strip()
        use_adk = 'ADK' in mode_toggle.value
        
        if not path:
            display(HTML('<div class="status status-error">âš ï¸� Please enter an image path</div>'))
            return
        
        display(HTML('<hr style="border: 5px solid #667eea; margin: 60px 0; box-shadow: 0 4px 10px rgba(0,0,0,0.2);">'))
        display(HTML('<div class="status status-scanning">ğŸ”„ SCANNING IMAGE...</div>'))
        
        actual_path = path
        if path.lower().endswith('.xml'):
            display(HTML('<p style="color: #667eea; text-align: center;">ğŸ“� XML detected. Searching for image...</p>'))
            filename = os.path.basename(path).replace('.xml', '')
            img_path_1 = path.replace('/annotations/', '/images/').replace('.xml', '.jpg')
            
            if os.path.exists(img_path_1):
                actual_path = img_path_1
            else:
                found = False
                for root, dirs, files in os.walk("/kaggle/input"):
                    for file in files:
                        if file.startswith(filename) and file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            actual_path = os.path.join(root, file)
                            found = True
                            break
                    if found:
                        break
                
                if not found:
                    display(HTML(f'<div class="status status-error">â�Œ Image not found for: {filename}</div>'))
                    return
            
            display(HTML(f'<p style="color: #667eea; text-align: center;">âœ… Found: {actual_path}</p>'))
        
        if os.path.exists(actual_path):
            try:
                display(HTML('<h3 style="color: #667eea; text-align: center; font-size: 1.5em; margin: 20px 0;">ğŸ“¸ CAPTURED IMAGE</h3>'))
                img = PIL.Image.open(actual_path)
                if img.mode not in ('RGB', 'L', 'RGBA'):
                    img = img.convert('RGB')
                img.thumbnail((600, 600))
                display(img)
                current_image["path"] = actual_path
            except Exception as e:
                display(HTML(f'<div class="status status-error">â�Œ Cannot display: {e}</div>'))
                return
        else:
            display(HTML(f'<div class="status status-error">â�Œ File not found: {actual_path}</div>'))
            return
        
        display(HTML(f'<div class="status status-info">ğŸ¤– Mode: {mode_toggle.value} | Session: {session_id}</div>'))
        
        # Suppress print statements temporarily
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(inspect_steel_unified(actual_path, use_adk=use_adk, session_id=session_id))
        
        # Restore stdout
        sys.stdout = old_stdout
        
        if result["status"] == "success":
            display(HTML('<div class="status status-success">âœ… ANALYSIS COMPLETE</div>'))
            formatted = format_ai_response(result["message"], result.get("mode", "Unknown"))
            display(HTML(formatted))
        else:
            display(HTML(f'<div class="status status-error">â�Œ {result["message"]}</div>'))

def on_ask_click(b):
    with output_area:
        # DON'T clear - just add separator
        
        question = question_input.value.strip()
        
        if not question:
            display(HTML('<div class="status status-error">âš ï¸� Please enter a question</div>'))
            return
        
        if not current_image["path"]:
            display(HTML('<div class="status status-error">âš ï¸� Please scan an image first</div>'))
            return
        
        display(HTML('<hr style="border: 5px double #667eea; margin: 60px 0; box-shadow: 0 4px 10px rgba(0,0,0,0.2);">'))
        display(HTML('<div class="status status-scanning">ğŸ¤” Processing question...</div>'))
        
        # Suppress print statements
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(inspect_steel_unified(current_image["path"], question=question, use_adk=False))
        
        # Restore stdout
        sys.stdout = old_stdout
        
        if result["status"] == "success":
            display(HTML('<div class="status status-success">âœ… ANSWER READY</div>'))
            formatted = format_ai_response(result["message"], "Gemini")
            display(HTML(formatted))
        else:
            display(HTML(f'<div class="status status-error">â�Œ {result["message"]}</div>'))

def on_clear_click(b):
    with output_area:
        clear_output(wait=True)
        display(HTML('<div class="status status-success">âœ… Results cleared!</div>'))

scan_btn.on_click(on_scan_click)
clear_btn.on_click(on_clear_click)
ask_btn.on_click(on_ask_click)

# ============================================================================
# ğŸ“º DISPLAY INTERFACE
# ============================================================================
display(HTML('<div class="scan-box">'))
display(path_input)
display(session_input)
display(mode_toggle)

button_box = widgets.HBox([scan_btn, clear_btn], layout=widgets.Layout(justify_content='space-between', width='90%'))
display(button_box)

display(HTML('<hr style="margin: 20px 0;">'))
display(question_input)
display(ask_btn)
display(HTML('</div>'))
display(output_area)

print("\nâœ… Interface Ready! Use ADK for tools & memory, Gemini for fast analysis.")


# ============================================================================
# COMPLETE INTERFACE CELL - STEEL DEFECT INSPECTION SYSTEM
# ============================================================================

!pip install -U -q ipywidgets nest_asyncio google-generativeai pillow

import ipywidgets as widgets
from IPython.display import display, HTML, Image as IPImage, clear_output
import asyncio
import nest_asyncio
import os
import base64
from io import BytesIO
import PIL.Image
import re

nest_asyncio.apply()

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================
display(HTML("""
<style>
    .main-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .title {
        color: white;
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 10px;
    }
    .subtitle {
        color: #e0e0e0;
        text-align: center;
        font-size: 1.2em;
        margin-bottom: 30px;
    }
    .scan-box {
        background: white;
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 8px 20px rgba(0,0,0,0.2);
    }
    .image-preview {
        max-width: 100%;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        margin: 15px 0;
    }
    .result-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8eef5 100%);
        color: #2c3e50;
        padding: 30px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        font-size: 1.05em;
        line-height: 1.9;
        border: 3px solid #667eea;
    }
    .result-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        margin: 25px 0 15px 0;
        font-size: 1.3em;
        font-weight: bold;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        border-left: 6px solid #ffd700;
    }
    .result-content {
        background: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .result-section {
        margin: 20px 0;
        padding: 20px;
        background: white;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }
    .section-title {
        color: #667eea;
        font-size: 1.4em;
        font-weight: bold;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 3px solid #667eea;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .section-content {
        color: #34495e;
        font-size: 1.1em;
        line-height: 1.8;
        margin-top: 12px;
        padding: 10px;
        font-weight: normal;
    }
    .section-content p {
        font-weight: normal;
    }
    .section-content strong {
        font-weight: normal;
    }
    .status {
        text-align: center;
        font-weight: bold;
        font-size: 1.3em;
        margin: 15px 0;
        padding: 10px;
        border-radius: 8px;
    }
    .status-scanning {
        background: #ffd54f;
        color: #333;
    }
    .status-success {
        background: #66bb6a;
        color: white;
    }
    .status-error {
        background: #ef5350;
        color: white;
    }
    .bullet-point {
        margin: 8px 0;
        padding-left: 25px;
        position: relative;
    }
    .bullet-point:before {
        content: "â–¸";
        position: absolute;
        left: 5px;
        color: #667eea;
        font-size: 1.3em;
        font-weight: bold;
    }
    .highlight-text {
        background: #fff3cd;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: bold;
        color: #856404;
    }
</style>
"""))

# ============================================================================
# INITIALIZE GEMINI
# ============================================================================
print("ğŸ”„ Initializing AI System...")
gemini_model = None

from kaggle_secrets import UserSecretsClient
import google.generativeai as genai
    
api_key = UserSecretsClient().get_secret("GOOGLE_API_KEY")
genai.configure(api_key=api_key)
gemini_model = genai.GenerativeModel('gemini-2.5-flash')
print("âœ… Gemini 2.5 Flash Connected Successfully!")

# ============================================================================
# RESPONSE FORMATTER
# ============================================================================
def format_ai_response(text):
    """
    Formats AI response with proper HTML structure for clear readability
    """
    # First, remove ALL markdown bold markers and other formatting
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Remove **bold**
    text = re.sub(r'\*([^*]+)\*', r'\1', text)      # Remove *italic*
    text = re.sub(r'__([^_]+)__', r'\1', text)      # Remove __bold__
    text = re.sub(r'_([^_]+)_', r'\1', text)        # Remove _italic_
    
    # Split by common section patterns
    sections = []
    
    # Pattern 1: Numbered sections (1., 2., etc.)
    numbered_pattern = r'(\d+)\.\s*([^:]+):\s*(.*?)(?=\n\s*\d+\.|$)'
    matches = re.finditer(numbered_pattern, text, re.DOTALL)
    
    for match in matches:
        number = match.group(1)
        title = match.group(2).strip()
        content = match.group(3).strip()
        
        sections.append({
            'title': title,
            'content': content
        })
    
    # If no numbered sections found, try to split by headers
    if not sections:
        # Look for lines in ALL CAPS (likely headers)
        lines = text.split('\n')
        current_section = None
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # Check if line is ALL CAPS and reasonably short (likely a header)
            if line_stripped.isupper() and len(line_stripped) < 100 and len(line_stripped.split()) <= 10:
                # Save previous section if exists
                if current_section and current_section['content'].strip():
                    sections.append(current_section)
                
                # Start new section
                current_section = {'title': line_stripped, 'content': ''}
            elif current_section is not None:
                # Add content to current section
                current_section['content'] += line + '\n'
            else:
                # Content before any header - create a general section
                if not current_section:
                    current_section = {'title': 'ANALYSIS', 'content': line + '\n'}
        
        # Add last section
        if current_section and current_section['content'].strip():
            sections.append(current_section)
    
    # If still no sections, return formatted as single block
    if not sections:
        clean_text = text.replace('\n', '<br>')
        return f'<div class="result-content" style="font-weight: normal;">{clean_text}</div>'
    
    # Build HTML with sections
    html = '<div class="result-box">'
    html += '<div class="result-header">ğŸ”¬ DETAILED INSPECTION REPORT</div>'
    
    for section in sections:
        title = section['title'].upper()
        content = section['content']
        
        # Format content: split by newlines and create bullet points
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        html += f'<div class="result-section">'
        html += f'<div class="section-title">{title}</div>'
        html += f'<div class="section-content">'
        
        for line in lines:
            # Check if line starts with a bullet or dash
            if line.startswith(('-', 'â€¢', '*')):
                clean_line = line.lstrip('-â€¢* ').strip()
                html += f'<div class="bullet-point" style="font-weight: normal;">{clean_line}</div>'
            else:
                # Check if this line contains a colon (might be a sub-heading)
                if ':' in line and len(line.split(':')[0]) < 60:
                    parts = line.split(':', 1)
                    subheading = parts[0].strip()
                    subcontent = parts[1].strip() if len(parts) > 1 else ''
                    
                    if subcontent:
                        html += f'<p style="margin: 12px 0; font-weight: normal;"><strong style="color: #667eea;">{subheading}:</strong> {subcontent}</p>'
                    else:
                        html += f'<p style="margin: 12px 0; font-weight: normal;"><strong style="color: #667eea;">{subheading}:</strong></p>'
                else:
                    # Regular paragraph - highlight important keywords
                    highlighted = line
                    keywords = ['CRITICAL', 'HIGH', 'SEVERE', 'STOP', 'IMMEDIATE', 'WARNING', 
                               'URGENT', 'DANGER', 'FAIL', 'DEFECT']
                    for keyword in keywords:
                        # Only highlight whole words
                        highlighted = re.sub(
                            r'\b' + keyword + r'\b',
                            f'<span class="highlight-text">{keyword}</span>',
                            highlighted,
                            flags=re.IGNORECASE
                        )
                    
                    html += f'<p style="margin: 10px 0; font-weight: normal;">{highlighted}</p>'
        
        html += '</div></div>'
    
    html += '</div>'
    return html

# ============================================================================
# AI INSPECTION FUNCTION
# ============================================================================
async def inspect_steel_image(image_path, question=None):
    """
    Inspects steel image using Gemini AI with optional follow-up questions
    """
    clean_path = image_path.strip().strip('"').strip("'")
    
    if not os.path.exists(clean_path):
        return {"status": "error", "message": f"File not found: {clean_path}"}
    
    try:
        # Check if it's an XML annotation file
        if clean_path.lower().endswith('.xml'):
            # Parse XML path to find corresponding image
            # Example: /kaggle/input/neu-surface-defect-database/NEU-DET/validation/annotations/crazing_267.xml
            
            path_parts = clean_path.split('/')
            filename = os.path.basename(clean_path).replace('.xml', '')
            
            # Try to find the image in multiple possible locations
            possible_paths = []
            
            # Strategy 1: Replace 'annotations' with 'images' in the path
            img_path_1 = clean_path.replace('/annotations/', '/images/').replace('.xml', '.jpg')
            if os.path.exists(img_path_1):
                possible_paths.append(img_path_1)
            
            # Strategy 2: Go up levels and search common image folders
            base_dir = os.path.dirname(os.path.dirname(clean_path))  # Go up two levels
            for img_folder in ['images', 'IMAGES', 'JPEGImages', 'Image', 'image']:
                for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG']:
                    img_path = os.path.join(base_dir, img_folder, filename + ext)
                    if os.path.exists(img_path):
                        possible_paths.append(img_path)
            
            # Strategy 3: Search in the root dataset directory
            dataset_root = None
            for i, part in enumerate(path_parts):
                if 'NEU' in part or 'neu' in part or 'dataset' in part.lower():
                    dataset_root = '/'.join(path_parts[:i+1])
                    break
            
            if dataset_root:
                for img_folder in ['images', 'IMAGES', 'JPEGImages']:
                    for ext in ['.jpg', '.jpeg', '.png', '.JPG']:
                        img_path = os.path.join(dataset_root, img_folder, filename + ext)
                        if os.path.exists(img_path):
                            possible_paths.append(img_path)
            
            # Strategy 4: Try same directory
            same_dir = os.path.dirname(clean_path)
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG']:
                img_path = os.path.join(same_dir, filename + ext)
                if os.path.exists(img_path):
                    possible_paths.append(img_path)
            
            # Strategy 5: Search the entire input directory (brute force)
            if not possible_paths:
                print(f"ğŸ”� Searching for {filename}...")
                for root, dirs, files in os.walk("/kaggle/input"):
                    for file in files:
                        if file.startswith(filename) and file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            img_path = os.path.join(root, file)
                            possible_paths.append(img_path)
                            break
                    if possible_paths:
                        break
            
            if possible_paths:
                clean_path = possible_paths[0]
                print(f"âœ… Found image: {clean_path}")
            else:
                # List what we tried
                search_info = f"XML annotation file detected, but corresponding image not found.\n\n"
                search_info += f"Searched for: {filename}.jpg\n"
                search_info += f"Locations checked:\n"
                search_info += f"  - {clean_path.replace('/annotations/', '/images/').replace('.xml', '.jpg')}\n"
                if dataset_root:
                    search_info += f"  - {dataset_root}/images/{filename}.jpg\n"
                search_info += f"\nPlease ensure the image exists in the dataset."
                return {"status": "error", "message": search_info}
        
        # Try to open image with PIL - it handles many formats
        img = PIL.Image.open(clean_path)
        
        # Convert image to RGB if necessary (for formats like PNG with transparency, CMYK, etc.)
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')
        
        if question:
            prompt = f"""Based on the steel surface defect image, answer this question:
            
Question: {question}

Provide a detailed, technical response focusing on defect analysis, quality control implications, 
and manufacturing recommendations.

FORMAT YOUR RESPONSE WITH CLEAR SECTIONS:
- Use numbered points (1., 2., 3., etc.)
- Each major point should be on a new line
- Use bullet points for sub-items
- Be specific and technical"""
        else:
            prompt = """You are an expert steel quality control inspector using computer vision AI.

Analyze this steel surface image and provide a DETAILED REPORT with these EXACT SECTIONS:

1. **DEFECT IDENTIFICATION**: 
   - What type of defect is present? (crazing, inclusion, patches, scratches, pitted surface, rolled-in scale, or none)
   - Describe the visual characteristics

2. **SEVERITY ASSESSMENT**: 
   - Rate as CRITICAL, MEDIUM, or LOW priority
   - Explain the severity rating

3. **LOCATION & CHARACTERISTICS**: 
   - Describe where the defect appears
   - Note size, shape, and distribution
   - Visual features and patterns

4. **ROOT CAUSE ANALYSIS**: 
   - What likely caused this defect in the manufacturing process?
   - Identify process parameters that may be involved
   - Manufacturing stage where defect originated

5. **QUALITY IMPACT**: 
   - How does this affect structural integrity?
   - Impact on usability and performance
   - Potential failure modes

6. **RECOMMENDATION**: 
   - Should production continue or stop?
   - Immediate actions required
   - Long-term corrective measures

IMPORTANT: Format each section clearly with proper numbering. Each section should start on a new line. Use bullet points for details within each section."""

        response = gemini_model.generate_content([prompt, img])
        
        return {
            "status": "success",
            "message": response.text,
            "image_path": clean_path
        }
    except PIL.UnidentifiedImageError:
        return {"status": "error", "message": f"Unable to identify image format. Supported: JPG, PNG, BMP, GIF, TIFF, WEBP, etc."}
    except Exception as e:
        return {"status": "error", "message": f"AI Analysis Error: {str(e)}"}

# ============================================================================
# BUILD THE INTERFACE
# ============================================================================

# Header
display(HTML("""
<div class="main-container">
    <div class="title">ğŸ”� SteelGuardian AI Inspector</div>
    <div class="subtitle">Advanced Computer Vision Quality Control System</div>
</div>
"""))

# Find sample images automatically
sample_images = []
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', 
                   '.webp', '.svg', '.ico', '.jfif', '.pjpeg', '.pjp', '.avif')

for root, dirs, files in os.walk("/kaggle/input"):
    for file in files:
        if file.lower().endswith(valid_extensions):
            sample_images.append(os.path.join(root, file))
    if len(sample_images) >= 10:
        break

if sample_images:
    print(f"âœ… Found {len(sample_images)} images in dataset\n")
    print("ğŸ“‹ SAMPLE IMAGE PATHS (Copy & Paste):")
    print("=" * 80)
    for i, path in enumerate(sample_images[:5], 1):
        print(f"{i}. {path}")
    print("=" * 80 + "\n")
else:
    print("âš ï¸� No images found. Please attach a dataset with images.\n")
    print("â„¹ï¸�  Supported formats: JPG, JPEG, PNG, BMP, GIF, TIFF, WEBP, SVG, and more\n")

# UI Components
path_input = widgets.Text(
    value=sample_images[0] if sample_images else "",
    description='ğŸ“� Image:',
    placeholder='Paste image path here...',
    layout=widgets.Layout(width='90%'),
    style={'description_width': '80px'}
)

scan_btn = widgets.Button(
    description='ğŸ”¬ Scan Image',
    button_style='success',
    layout=widgets.Layout(width='200px', height='50px'),
    style={'button_color': '#4CAF50', 'font_weight': 'bold'}
)

clear_btn = widgets.Button(
    description='ğŸ—‘ï¸� Clear Results',
    button_style='warning',
    layout=widgets.Layout(width='200px', height='50px'),
    style={'font_weight': 'bold'}
)

question_input = widgets.Textarea(
    placeholder='Ask a follow-up question about the defect (optional)...',
    description='ğŸ’¬ Question:',
    layout=widgets.Layout(width='90%', height='80px'),
    style={'description_width': '80px'}
)

ask_btn = widgets.Button(
    description='â�“ Ask AI',
    button_style='info',
    layout=widgets.Layout(width='200px', height='50px'),
    style={'font_weight': 'bold'}
)

output_area = widgets.Output(layout=widgets.Layout(
    border='3px solid #667eea',
    padding='20px',
    border_radius='15px',
    min_height='400px'
))

# Current image reference
current_image = {"path": None}

# ============================================================================
# ğŸ�¬ EVENT HANDLERS
# ============================================================================

def on_scan_click(b):
    with output_area:
        # DON'T clear output - append instead
        
        path = path_input.value.strip()
        if not path:
            display(HTML('<div class="status status-error">âš ï¸� Please enter an image path</div>'))
            return
        
        # Add separator if there's already content
        display(HTML('<hr style="border: 3px solid #667eea; margin: 40px 0;">'))
        
        # Show scanning status
        display(HTML('<div class="status status-scanning">ğŸ”„ SCANNING IMAGE...</div>'))
        
        # Display the image
        if os.path.exists(path):
            try:
                display(HTML('<h3 style="color: #667eea; text-align: center; font-size: 1.5em;">ğŸ“¸ CAPTURED IMAGE</h3>'))
                
                # Check if it's an XML file
                actual_path = path
                if path.lower().endswith('.xml'):
                    # Try to find corresponding image
                    path_parts = path.split('/')
                    filename = os.path.basename(path).replace('.xml', '')
                    
                    possible_paths = []
                    
                    # Strategy 1: Replace 'annotations' with 'images'
                    img_path_1 = path.replace('/annotations/', '/images/').replace('.xml', '.jpg')
                    if os.path.exists(img_path_1):
                        possible_paths.append(img_path_1)
                    
                    # Strategy 2: Check standard locations
                    base_dir = os.path.dirname(os.path.dirname(path))
                    for img_folder in ['images', 'IMAGES', 'JPEGImages', 'Image', 'image']:
                        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG']:
                            img_path = os.path.join(base_dir, img_folder, filename + ext)
                            if os.path.exists(img_path):
                                possible_paths.append(img_path)
                    
                    # Strategy 3: Dataset root search
                    dataset_root = None
                    for i, part in enumerate(path_parts):
                        if 'NEU' in part or 'neu' in part or 'dataset' in part.lower():
                            dataset_root = '/'.join(path_parts[:i+1])
                            break
                    
                    if dataset_root:
                        for img_folder in ['images', 'IMAGES', 'JPEGImages']:
                            for ext in ['.jpg', '.jpeg', '.png', '.JPG']:
                                img_path = os.path.join(dataset_root, img_folder, filename + ext)
                                if os.path.exists(img_path):
                                    possible_paths.append(img_path)
                    
                    # Strategy 4: Same directory
                    same_dir = os.path.dirname(path)
                    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG']:
                        img_path = os.path.join(same_dir, filename + ext)
                        if os.path.exists(img_path):
                            possible_paths.append(img_path)
                    
                    # Strategy 5: Brute force search
                    if not possible_paths:
                        display(HTML(f'<p style="color: #667eea; text-align: center;">ğŸ”� Searching for image: {filename}...</p>'))
                        for root, dirs, files in os.walk("/kaggle/input"):
                            for file in files:
                                if file.startswith(filename) and file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                                    img_path = os.path.join(root, file)
                                    possible_paths.append(img_path)
                                    break
                            if possible_paths:
                                break
                    
                    if possible_paths:
                        actual_path = possible_paths[0]
                        display(HTML(f'<p style="color: #667eea; text-align: center;">ğŸ“� XML annotation detected. Loading image: <br><code>{actual_path}</code></p>'))
                    else:
                        error_msg = f'<div class="status status-error">â�Œ XML annotation file detected, but corresponding image not found.<br><br>'
                        error_msg += f'Searched for: <strong>{filename}.jpg</strong><br>'
                        error_msg += f'Tried locations:<br>'
                        error_msg += f'â€¢ {path.replace("/annotations/", "/images/").replace(".xml", ".jpg")}<br>'
                        if dataset_root:
                            error_msg += f'â€¢ {dataset_root}/images/{filename}.jpg<br>'
                        error_msg += f'<br>Please ensure the image exists in the dataset.</div>'
                        display(HTML(error_msg))
                        return
                
                img = PIL.Image.open(actual_path)
                
                # Convert image mode if necessary for display
                if img.mode not in ('RGB', 'L', 'RGBA'):
                    img = img.convert('RGB')
                
                img.thumbnail((600, 600))
                display(img)
                current_image["path"] = path  # Keep original path
            except PIL.UnidentifiedImageError:
                display(HTML(f'<div class="status status-error">â�Œ Cannot identify image format. File may not be a valid image.</div>'))
                return
            except Exception as e:
                display(HTML(f'<div class="status status-error">â�Œ Cannot display image: {e}</div>'))
                return
        
        # Run AI analysis
        display(HTML('<div class="status status-scanning">ğŸ¤– AI ANALYSIS IN PROGRESS...</div>'))
        
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(inspect_steel_image(path))
        
        if result["status"] == "success":
            display(HTML('<div class="status status-success">âœ… ANALYSIS COMPLETE</div>'))
            formatted_response = format_ai_response(result["message"])
            display(HTML(formatted_response))
        else:
            display(HTML(f'<div class="status status-error">â�Œ {result["message"]}</div>'))

def on_ask_click(b):
    with output_area:
        question = question_input.value.strip()
        
        if not question:
            display(HTML('<div class="status status-error">âš ï¸� Please enter a question</div>'))
            return
        
        if not current_image["path"]:
            display(HTML('<div class="status status-error">âš ï¸� Please scan an image first</div>'))
            return
        
        # Add separator
        display(HTML('<hr style="border: 2px dashed #667eea; margin: 30px 0;">'))
        
        display(HTML(f'<div class="status status-scanning">ğŸ¤” Processing your question...</div>'))
        
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(inspect_steel_image(current_image["path"], question))
        
        if result["status"] == "success":
            display(HTML('<div class="status status-success">âœ… ANSWER READY</div>'))
            
            # Format Q&A response
            html = '<div class="result-box">'
            html += f'<div class="result-header">â�“ YOUR QUESTION</div>'
            html += f'<div class="result-content" style="font-size: 1.2em; font-style: italic;">{question}</div>'
            html += f'<div class="result-header" style="margin-top: 20px;">ğŸ¤– AI RESPONSE</div>'
            
            formatted_answer = format_ai_response(result["message"])
            # Extract content from formatted answer
            if '<div class="result-box">' in formatted_answer:
                formatted_answer = formatted_answer.replace('<div class="result-box">', '').replace('</div>', '', 1)
            
            html += formatted_answer
            html += '</div>'
            
            display(HTML(html))
        else:
            display(HTML(f'<div class="status status-error">â�Œ {result["message"]}</div>'))

def on_clear_click(b):
    with output_area:
        clear_output(wait=True)
        display(HTML('<div class="status status-success">âœ… Results cleared. Ready for new scan!</div>'))

scan_btn.on_click(on_scan_click)
clear_btn.on_click(on_clear_click)
ask_btn.on_click(on_ask_click)

# ============================================================================
# DISPLAY INTERFACE
# ============================================================================
display(HTML('<div class="scan-box">'))
display(path_input)

# Button row
button_box = widgets.HBox([scan_btn, clear_btn], layout=widgets.Layout(justify_content='space-between', width='90%'))
display(button_box)

display(HTML('<hr style="margin: 20px 0;">'))
display(question_input)
display(ask_btn)
display(HTML('</div>'))
display(output_area)

print("\nâœ… Interface Ready! Paste an image path and click 'Scan Image'")

