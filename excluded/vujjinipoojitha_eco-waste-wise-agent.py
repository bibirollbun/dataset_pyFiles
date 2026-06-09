





# Step 1: Install necessary libraries
# We use Google Generative AI SDK to power our agents.
!pip install -q -U google-generativeai


# Cell 2: Backend Logic (Scientific System with Smart Filter)
import google.generativeai as genai
from kaggle_secrets import UserSecretsClient
import PIL.Image
import io
import base64

# ==========================================
# 1. AGENT CONFIGURATION
# ==========================================
try:
    user_secrets = UserSecretsClient()
    my_api_key = user_secrets.get_secret("GOOGLE_API_KEY")
    genai.configure(api_key=my_api_key)
    model = genai.GenerativeModel('models/gemini-2.5-flash')
    print("âœ… Eco Waste Wise: Lab Backend Ready.")
except Exception as e:
    print(f"â�Œ API Key Error: {e}")

# --- HELPER: Image Processing ---
def image_to_base64(img):
    # Fix: Convert PNG/Transparency to RGB
    if img.mode != 'RGB': img = img.convert('RGB')
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"

# ==========================================
# 2. MULTI-AGENT LOGIC
# ==========================================
def run_scientific_scan(image_bytes):
    try:
        # --- PHASE 1: PERCEPTION (Vision Agent) ---
        img = PIL.Image.open(io.BytesIO(image_bytes))
        if img.mode != 'RGB': img = img.convert('RGB')
        img_base64 = image_to_base64(img)
        
        # --- PHASE 2: COGNITIVE REASONING (Logic Agent) ---
        prompt = """
        Act as an advanced 'Eco-Scientist'. Analyze this image.
        
        Step 1: CHECK - Is this item commonly considered 'Garbage'? 
        (e.g., Phone, Keys, Person = NOT Garbage. Bottle, Wrapper = Garbage).
        
        Step 2: GENERATE HTML OUTPUT.
        
        [CASE A: NOT GARBAGE] -> Output a Warning Card:
        <div class="warning-card">
            <h3>âš ï¸� Wait! This is NOT Garbage.</h3>
            <p><b>Analysis:</b> Detected a valuable <b>ITEM_NAME</b>.<br>
            <b>Advice:</b> Please do not throw this away. Keep it safe.</p>
        </div>

        [CASE B: IS WASTE] -> Output Scientific Report with Meters:
        For EACH item, generate this HTML:
        
        <div class="report-card">
            <div class="card-header">
                <span class="item-title">ITEM_NAME</span>
                <span class="item-type">SPECIFIC_TYPE</span>
            </div>
            
            <div class="meter-section">
                <div class="meter-label">â™»ï¸� Recyclability <span style="float:right">X%</span></div>
                <div class="progress-bg"><div class="progress-fill p-green" style="width: X%;"></div></div>
                
                <div class="meter-label">ğŸ”„ Reusability <span style="float:right">Y%</span></div>
                <div class="progress-bg"><div class="progress-fill p-blue" style="width: Y%;"></div></div>
                
                <div class="meter-label">ğŸ›¡ï¸� Eco-Safety <span style="float:right">Z%</span></div>
                <div class="progress-bg"><div class="progress-fill p-safety" style="width: Z%;"></div></div>
            </div>
            
            <div class="analysis-box">
                <b>ğŸ§ª Material DNA:</b> MATERIAL_DETAILS.<br>
                <b>âš ï¸� Verdict:</b> RECYCLE or TRASH?
            </div>
            
            <div class="action-box">
                <b>ğŸ’¡ Lab Idea:</b> REUSE_IDEA.
                <br>
                <a href="https://www.youtube.com/results?search_query=DIY+reuse+ITEM_NAME" target="_blank" class="yt-link">
                    â–¶ï¸� Watch Experiment
                </a>
            </div>
        </div>
        """
        
        # --- PHASE 3: EXECUTION ---
        response = model.generate_content([prompt, img])
        final_html = response.text.replace("```html", "").replace("```", "")
        
        # Session Title Generation
        title_res = model.generate_content(["Generate a 2-word title for this image.", img])
        short_title = title_res.text.strip()
        
        return {
            "success": True,
            "item": short_title,
            "advice": final_html,
            "image_url": img_base64
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


# Cell 3: Frontend UI (Scientific Lab Dashboard with State Management)
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import functools

# ==========================================
# 1. CONFIGURATION & ASSETS
# ==========================================
PADLET_LINK = "https://padlet.com/tpass1311/eco-thoughts-vjfdpf40z9i2d69w"

# ==========================================
# 2. CSS ARCHITECTURE (High Contrast UI)
# ==========================================
style = """
<style>
    /* --- LAYOUT: FORCE LIGHT MODE FOR READABILITY --- */
    .lab-container { 
        font-family: 'Segoe UI', Helvetica, Arial, sans-serif; 
        background: #f8fafc !important; 
        padding: 25px; 
        border-radius: 15px; 
        border: 1px solid #cbd5e1; 
        max-width: 750px; 
        margin: 0 auto; 
        color: #0f172a !important; /* Dark Text for accessibility */
    }
    
    /* HEADER COMPONENT */
    .lab-header { text-align: center; border-bottom: 3px solid #0f766e; padding-bottom: 15px; margin-bottom: 25px; }
    .lab-title { color: #0f766e !important; font-size: 30px; font-weight: 900; letter-spacing: 1px; }
    
    /* REPORT CARD (Dynamic HTML from Backend) */
    .report-card { 
        background: #ffffff !important; 
        padding: 20px; 
        border-radius: 12px; 
        box-shadow: 0 4px 12px rgba(0,0,0,0.08); 
        margin-bottom: 30px; 
        border-left: 6px solid #0f766e; 
        border: 1px solid #e2e8f0;
    }
    
    .card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; border-bottom: 2px solid #f1f5f9; padding-bottom: 10px; }
    .item-title { font-size: 22px; font-weight: 800; color: #1e293b !important; }
    .item-type { background: #ccfbf1 !important; color: #0f766e !important; padding: 5px 12px; border-radius: 20px; font-size: 12px; font-weight: bold; text-transform: uppercase; }
    
    /* SCIENTIFIC METERS (Progress Bars) */
    .meter-section { background: #f8fafc !important; padding: 15px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #e2e8f0; }
    .meter-label { font-size: 12px; font-weight: bold; color: #334155 !important; margin-bottom: 5px; display: block; }
    .progress-bg { background: #cbd5e1 !important; border-radius: 10px; height: 10px; width: 100%; margin-bottom: 12px; overflow: hidden; }
    .progress-fill { height: 100%; border-radius: 10px; }
    
    /* Color Palettes for Meters */
    .p-green { background: #16a34a !important; }
    .p-blue { background: #2563eb !important; }
    .p-safety { background: #d97706 !important; }

    /* TEXT CONTENT AREAS */
    .analysis-box { font-size: 14px; color: #334155 !important; margin-bottom: 15px; line-height: 1.6; }
    .analysis-box b { color: #0f766e !important; }
    
    /* SMART FILTER WARNING (Not Garbage) */
    .warning-box {
        background: #fff1f2 !important; color: #9f1239 !important;
        padding: 15px; border-radius: 8px; border: 1px solid #fda4af;
        margin-bottom: 15px;
    }
    
    /* ACTION BOX (High Visibility for DIY Ideas) */
    .action-box { 
        background: #ffffff !important; 
        padding: 15px; 
        border-radius: 8px; 
        border: 2px solid #bbf7d0; 
        margin-top: 15px; 
        color: #14532d !important; 
    }
    .action-box b { color: #14532d !important; font-size: 15px; }
    
    /* DYNAMIC RESOURCE LINK */
    .yt-link { 
        display: inline-block; margin-top: 10px; 
        color: #ffffff !important; background: #dc2626 !important;
        text-decoration: none; font-weight: bold; font-size: 12px; 
        padding: 8px 15px; border-radius: 20px;
    }
    
    /* UI ELEMENTS (Loader & Community) */
    .loader { border: 5px solid #e2e8f0; border-top: 5px solid #0f766e; border-radius: 50%; width: 50px; height: 50px; animation: spin 1s linear infinite; margin: 0 auto; }
    @keyframes spin { 100% { transform: rotate(360deg); } }
    
    .comm-card { margin-top: 30px; background: linear-gradient(135deg, #0f766e 0%, #115e59 100%) !important; padding: 20px; border-radius: 15px; text-align: center; color: white !important; }
    .comm-btn { background: white !important; color: #0f766e !important; padding: 10px 25px; border-radius: 30px; text-decoration: none; font-weight: bold; display: inline-block; margin-top: 10px; }
    .comm-card h3, .comm-card p { color: white !important; }

</style>
"""

# ==========================================
# 3. WIDGET INITIALIZATION
# ==========================================

# App Header
header_html = widgets.HTML(f"{style}<div class='lab-container'><div class='lab-header'><div class='lab-title'>â™»ï¸� Eco Waste Wise</div><small style='color:#64748b'>Scientific Lab Analysis</small></div></div>")

# Upload Button (Acts as Camera on Mobile)
upload_btn = widgets.FileUpload(accept='image/*', multiple=False, description='ğŸ“¸ Scan / Camera')
upload_btn.style.button_color = '#0f766e'
upload_btn.layout = widgets.Layout(width='100%', height='55px', margin='0 0 20px 0')

# Output Areas
main_output = widgets.Output()
history_box = widgets.VBox([])

# Community Footer Component
comm_html = widgets.HTML(f"""
<div class="lab-container" style="border:none; background:none; padding:0;">
    <div class="comm-card">
        <h3 style="margin:0 0 10px 0;">ğŸŒ� Join the Eco-Community</h3>
        <p style="font-size:14px; opacity:0.9; margin-bottom:10px;">
            Upload your DIY projects to our public board.
        </p>
        <a href="{PADLET_LINK}" target="_blank" class="comm-btn">ğŸ“¤ Upload to Padlet Board</a>
    </div>
</div>
""")

# State Management (Session History)
session_history = []

# ==========================================
# 4. INTERACTION LOGIC
# ==========================================

def show_loader():
    """Displays animated loader during Agent Processing"""
    with main_output:
        clear_output()
        display(HTML(f"""
        <div class="lab-container" style="text-align:center; padding:50px">
            <div class="loader"></div><br><b style="color:#0f766e">Processing Sample...</b>
        </div>
        """))

def render_dashboard(data):
    """Renders the Final Report Card with Image Preview"""
    with main_output:
        clear_output()
        html = f"""
        <div class="lab-container">
            <div style="text-align:center; background:#fff; padding:10px; border-radius:10px; margin-bottom:20px; border:1px solid #eee;">
                <img src='{data['image_url']}' style='max-height:250px; max-width:100%; border-radius:8px;'>
            </div>
            {data['advice']}
        </div>
        """
        display(HTML(html))

def update_history_ui():
    """Updates the Session History list dynamically"""
    items = []
    if session_history:
        items.append(widgets.HTML(f"{style}<div class='lab-container' style='padding:10px; border:none; background:none;'><b>ğŸ“œ Recent Scans</b></div>"))
    
    for item in session_history:
        lbl = widgets.HTML(f"<b style='color:#1e293b; font-size:13px;'>{item['item']}</b>")
        btn = widgets.Button(description="View", icon='eye')
        btn.layout.width = '90px'
        btn.style.button_color = '#f1f5f9'
        # Using partial to bind specific item data to the button click
        btn.on_click(functools.partial(on_hist_click, d=item))
        
        row = widgets.HBox([lbl, btn], layout=widgets.Layout(
            margin='0 0 5px 0', width='100%', justify_content='space-between', 
            background_color='white', padding='10px', border='1px solid #e2e8f0', border_radius='8px'
        ))
        items.append(row)
        
    history_box.children = tuple(items)

def on_hist_click(b, d):
    """Restore state from history"""
    render_dashboard(d)

def on_upload(change):
    """Main Handler: Triggered when user uploads an image"""
    if not upload_btn.value: return
    show_loader()
    try:
        # Extract content (Handle Kaggle's tuple vs dict format)
        data = upload_btn.value
        if isinstance(data, tuple): c = data[0]['content']
        else: c = data[list(data.keys())[0]]['content']
        
        # Execute Backend Logic (Call Agent Chain)
        res = run_scientific_scan(c)
        
        if res['success']:
            # Update State
            session_history.insert(0, res)
            if len(session_history) > 3: session_history.pop()
            
            # Update UI
            render_dashboard(res)
            update_history_ui()
            upload_btn.value = () # Reset button
        else:
            with main_output: clear_output(); print(f"Error: {res['error']}")
    except Exception as e:
        print(f"System Error: {e}")

# Bind Event
upload_btn.observe(on_upload, names='value')

# --- 5. APP LAUNCH ---
display(widgets.VBox([header_html, upload_btn, main_output, history_box, comm_html]))

