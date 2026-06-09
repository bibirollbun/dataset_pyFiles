# Step 1: Install necessary libraries
# We use Google Generative AI SDK to power our agents.
!pip install -q -U google-generativeai


# Cell 2: Backend Logic (Core Intelligence & Robust Image Processing Pipeline)
import google.generativeai as genai
from kaggle_secrets import UserSecretsClient
import PIL.Image
from PIL import ImageFile, ImageOps # Imported ImageOps for EXIF orientation handling
import io
import base64

# ==========================================
# 1. SYSTEM CONFIGURATION & ROBUSTNESS SETUP
# ==========================================
try:
    # --- [Data Robustness: Handle Truncated Streams] ---
    # Ensures the system can process images even if the data stream is incomplete due to network instability.
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    # Securely fetch API credentials and initialize the Gemini client.
    user_secrets = UserSecretsClient()
    my_api_key = user_secrets.get_secret("GOOGLE_API_KEY")
    genai.configure(api_key=my_api_key)
    # Utilizing Gemini 2.5 Flash optimized for low-latency responses.
    model = genai.GenerativeModel('models/gemini-2.5-flash')
    print("âœ… Eco Waste Wise: Backend Initialized (Robust pre-processing enabled).")
except Exception as e:
    print(f"â�Œ Configuration Error: API Key validation failed. {e}")

# --- Helper Function: Image Standardization Pipeline ---
# Pre-processes raw input data to normalize format, orientation, and size for model inference.
def process_and_encode_image(image_bytes):
    try:
        # 1. Ingest raw image data from buffer.
        img = PIL.Image.open(io.BytesIO(image_bytes))
        
        # --- [Standardization: EXIF Orientation Fix] ---
        # Corrects image orientation based on EXIF data (common issue with mobile photos).
        img = ImageOps.exif_transpose(img)
        
        # --- [Standardization: Color Channel Normalization] ---
        # Convert varied input formats (e.g., RGBA, P, CMYK) to standard RGB.
        # This handles transparency and unusual color spaces to prevent model errors.
        if img.mode not in ('RGB', 'L'): # 'L' is grayscale, which is acceptable
            img = img.convert('RGB')
            
        # --- [Optimization: Resource & Latency Management] ---
        # Downscale high-resolution inputs (>1500px) to optimize token usage and reduce API latency
        # without compromising analytical quality. Lanczos resampling used for high-quality preservation.
        # This ensures even huge 4K/8K images are processed efficiently.
        max_dimension = 1500
        if max(img.size) > max_dimension:
            scale_factor = max_dimension / max(img.size)
            new_size = (int(img.size[0] * scale_factor), int(img.size[1] * scale_factor))
            img = img.resize(new_size, PIL.Image.LANCZOS)
        
        # 2. Compress and serialize to a standardized JPEG buffer.
        buffered = io.BytesIO()
        # Optimize=True and Quality=85 balances visual fidelity with payload size efficiency.
        # Using JPEG format ensures broad compatibility.
        img.save(buffered, format="JPEG", optimize=True, quality=85)
        
        # 3. Encode to Base64 for secure transmission to the frontend UI.
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"
        
    except Exception as e:
        # Exception handling for severely corrupted or unrecognizable input data.
        # Provides detailed error information for debugging.
        raise ValueError(f"Input data processing failed. File may be corrupted or format unsupported. Details: {e}")

# ==========================================
# 2. SEQUENTIAL COGNITIVE ARCHITECTURE (The Multi-Agent Flow)
# ==========================================
def run_scientific_scan(image_bytes):
    try:
        # --- STEP 1: PERCEPTION LAYER (Input Processing) ---
        # Execute the pre-processing pipeline to standardize input.
        # This function now handles resizing, orientation, and format conversion robustly.
        img_base64 = process_and_encode_image(image_bytes)
        
        # Re-instantiate the processed image data to ensure the generative model analyzes
        # the exact optimized data representation presented to the user.
        img_data = base64.b64decode(img_base64.split(',')[1])
        img_for_gemini = PIL.Image.open(io.BytesIO(img_data))

        
        # --- STEP 2: REASONING LAYER (Chain-of-Thought Prompt Engineering) ---
        # Define the multi-step cognitive process for the model, enforcing a "logic-filter-first" structure.
        # (Prompt remains unchanged as requested)
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
        
        # --- STEP 3: EXECUTION LAYER (Inference & Post-Processing) ---
        # Dispatch request to Gemini 2.5 Flash API with image and prompt context.
        response = model.generate_content([prompt, img_for_gemini])
        # Sanitize model output to ensure clean HTML rendering.
        final_html = response.text.replace("```html", "").replace("```", "")
        
        # Secondary inference call to generate concise metadata for session history tracking.
        title_res = model.generate_content(["Generate a 2-word title for this image.", img_for_gemini])
        short_title = title_res.text.strip()
        
        # Structure the final response payload for frontend consumption.
        return {
            "success": True,
            "item": short_title,
            "advice": final_html,
            "image_url": img_base64
        }
        
    except Exception as e:
        # Global exception handler to ensure graceful degradation and provide informative feedback.
        # This will catch issues during image processing or API calls.
        return {"success": False, "error": f"Analysis failed. System error: {str(e)}"}


# Cell 3: The "EcoScan Lab" Dashboard (Where the magic happens)
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import functools

# ==========================================
# 1. COMMUNITY CONNECTION
# ==========================================
# This link lets users share their cool upcycling projects with others.
PADLET_LINK = "https://padlet.com/tpass1311/eco-thoughts-vjfdpf40z9i2d69w"

# ==========================================
# 2. DESIGNING THE LAB (Making it look professional)
# ==========================================
# We're using a clean, high-contrast style (Teal & White) so it feels like a real scientific tool.
# It's designed to be easy to read on any device, from a laptop to a phone in bright sunlight.
style = """
<style>
    /* --- LAYOUT: A clean, modern container for our lab --- */
    .lab-container { 
        font-family: 'Segoe UI', Helvetica, Arial, sans-serif; 
        background: #f8fafc !important; 
        padding: 25px; 
        border-radius: 15px; 
        border: 1px solid #cbd5e1; 
        max-width: 750px; 
        margin: 0 auto; 
        color: #0f172a !important; /* Dark text for readability */
    }
    
    /* HEADER COMPONENT: The main title at the top */
    .lab-header { text-align: center; border-bottom: 3px solid #0f766e; padding-bottom: 15px; margin-bottom: 25px; }
    .lab-title { color: #0f766e !important; font-size: 30px; font-weight: 900; letter-spacing: 1px; }
    
    /* REPORT CARD: The main results area (generated by the AI) */
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
    
    /* SCIENTIFIC METERS: The cool progress bars for Recycle/Reuse/Safety */
    .meter-section { background: #f8fafc !important; padding: 15px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #e2e8f0; }
    .meter-label { font-size: 12px; font-weight: bold; color: #334155 !important; margin-bottom: 5px; display: block; }
    .progress-bg { background: #cbd5e1 !important; border-radius: 10px; height: 10px; width: 100%; margin-bottom: 12px; overflow: hidden; }
    .progress-fill { height: 100%; border-radius: 10px; }
    
    /* Color Palettes for Meters (Green=Good, Blue=Reuse, Orange=Safety) */
    .p-green { background: #16a34a !important; }
    .p-blue { background: #2563eb !important; }
    .p-safety { background: #d97706 !important; }

    /* TEXT CONTENT AREAS: For detailed analysis and warnings */
    .analysis-box { font-size: 14px; color: #334155 !important; margin-bottom: 15px; line-height: 1.6; }
    .analysis-box b { color: #0f766e !important; }
    
    /* SMART FILTER WARNING: The red box if you scan a valuable item */
    .warning-box {
        background: #fff1f2 !important; color: #9f1239 !important;
        padding: 15px; border-radius: 8px; border: 1px solid #fda4af;
        margin-bottom: 15px;
    }
    
    /* ACTION BOX: The green box with the DIY YouTube link */
    .action-box { 
        background: #ffffff !important; 
        padding: 15px; 
        border-radius: 8px; 
        border: 2px solid #bbf7d0; 
        margin-top: 15px; 
        color: #14532d !important; 
    }
    .action-box b { color: #14532d !important; font-size: 15px; }
    
    /* DYNAMIC RESOURCE LINK: The button that takes you to YouTube */
    .yt-link { 
        display: inline-block; margin-top: 10px; 
        color: #ffffff !important; background: #dc2626 !important;
        text-decoration: none; font-weight: bold; font-size: 12px; 
        padding: 8px 15px; border-radius: 20px;
    }
    
    /* COMMUNITY FOOTER: The bottom section to join the board */
    .comm-card { margin-top: 30px; background: linear-gradient(135deg, #0f766e 0%, #115e59 100%) !important; padding: 20px; border-radius: 15px; text-align: center; color: white !important; }
    .comm-btn { background: white !important; color: #0f766e !important; padding: 10px 25px; border-radius: 30px; text-decoration: none; font-weight: bold; display: inline-block; margin-top: 10px; }
    .comm-card h3, .comm-card p { color: white !important; }

    /* --- SKELETON ANIMATION (The cool loading effect) --- */
    /* This makes the loading state look like a shimmering app screen instead of a boring spinner */
    @keyframes shimmer { 
        0% { background-position: -1000px 0; } 
        100% { background-position: 1000px 0; } 
    }
    .skeleton { 
        animation: shimmer 2s infinite linear; 
        background: linear-gradient(to right, #eff1f3 4%, #e2e2e2 25%, #eff1f3 36%); 
        background-size: 1000px 100%; 
        border-radius: 8px; 
    }
    .sk-card { background: white; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0; margin-bottom: 20px; }
    .sk-img { height: 250px; width: 100%; border-radius: 10px; margin-bottom: 15px; }
    .sk-line { height: 15px; margin-bottom: 10px; width: 100%; }
    .sk-header { height: 30px; width: 50%; margin-bottom: 20px; }

</style>
"""

# ==========================================
# 3. BUILDING THE DASHBOARD COMPONENTS
# ==========================================

# The main title header
header_html = widgets.HTML(f"{style}<div class='lab-container'><div class='lab-header'><div class='lab-title'>â™»ï¸� Eco Waste Wise</div><small style='color:#64748b'>Scientific Lab Analysis</small></div></div>")

# The big teal button. On mobile, this opens the camera!
upload_btn = widgets.FileUpload(accept='image/*', multiple=False, description='ğŸ“¸ Upload')
upload_btn.style.button_color = '#0f766e'
upload_btn.layout = widgets.Layout(width='100%', height='55px', margin='0 0 20px 0')

# The area where the results (or the loading skeleton) will appear
main_output = widgets.Output()

# The list that holds recent scans
history_box = widgets.VBox([])

# The footer component inviting users to the community board
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

# A list to keep track of what the user has scanned in this session
session_history = []

# ==========================================
# 4. MAKING IT INTERACTIVE (The Logic)
# ==========================================

# --- FUNCTION: Show the cool Skeleton Loader ---
def show_loader():
    """Displays the shimmering skeleton animation while the AI is thinking."""
    with main_output:
        clear_output()
        # This HTML creates a fake, shimmering version of the report card
        display(HTML(f"""
        <div class="lab-container">
            <div style="text-align:center; margin-bottom:20px; color:#0f766e;">
                <b>ğŸš€ AI Agents Analyzing Scene...</b>
            </div>
            
            <div class="sk-card">
                <div class="skeleton sk-img"></div> </div>
            
            <div class="sk-card">
                <div class="skeleton sk-header"></div>
                <div class="skeleton sk-line"></div>
                <div class="skeleton sk-line"></div>
                <div class="skeleton sk-line" style="width: 70%;"></div>
                <br>
                <div class="skeleton sk-line" style="height: 40px;"></div> </div>
        </div>
        """))

# --- FUNCTION: Show the Final Results ---
def render_dashboard(data):
    """Renders the final, beautiful report card with the image and AI advice."""
    with main_output:
        clear_output()
        # We put the user's image on top, and the AI's HTML report below it.
        html = f"""
        <div class="lab-container">
            <div style="text-align:center; background:#fff; padding:10px; border-radius:10px; margin-bottom:20px; border:1px solid #eee;">
                <img src='{data['image_url']}' style='max-height:250px; max-width:100%; border-radius:8px;'>
            </div>
            {data['advice']}
        </div>
        """
        display(HTML(html))

# --- FUNCTION: Update the "Recent Scans" list ---
def update_history_ui():
    """Refresh the history section with the latest items."""
    items = []
    if session_history:
        # Add a title if there's history
        items.append(widgets.HTML(f"{style}<div class='lab-container' style='padding:10px; border:none; background:none;'><b>ğŸ“œ Recent Scans</b></div>"))
    
    # Loop through recent scans and create a button for each
    for item in session_history:
        lbl = widgets.HTML(f"<b style='color:#1e293b; font-size:13px;'>{item['item']}</b>")
        btn = widgets.Button(description="View", icon='eye')
        btn.layout.width = '90px'
        btn.style.button_color = '#f1f5f9'
        # This clever trick binds the specific item data to its button click
        btn.on_click(functools.partial(on_hist_click, d=item))
        
        # Put label and button in a nice row
        row = widgets.HBox([lbl, btn], layout=widgets.Layout(
            margin='0 0 5px 0', width='100%', justify_content='space-between', 
            background_color='white', padding='10px', border='1px solid #e2e8f0', border_radius='8px'
        ))
        items.append(row)
        
    # Update the history box widget
    history_box.children = tuple(items)

# --- FUNCTION: What happens when you click a history button ---
def on_hist_click(b, d):
    """Restore a previous analysis result instantly."""
    render_dashboard(d)

# --- MAIN FUNCTION: What happens when you upload an image ---
def on_upload(change):
    """The main event handler. Triggered when the user selects a photo."""
    if not upload_btn.value: return
    
    # 1. Immediately show the skeleton loader for good UX
    show_loader()
    
    try:
        # 2. Get the image data (handling different Kaggle upload formats)
        data = upload_btn.value
        if isinstance(data, tuple): c = data[0]['content']
        else: c = data[list(data.keys())[0]]['content']
        
        # 3. Send it to the Backend Engine (Cell 2) and wait for results
        # This is where the multi-agent magic happens!
        res = run_scientific_scan(c)
        
        if res['success']:
            # 4a. Success! Save to history and show results.
            session_history.insert(0, res)
            # Keep history short (last 3 items)
            if len(session_history) > 3: session_history.pop()
            
            render_dashboard(res)
            update_history_ui()
            # Reset the upload button for the next scan
            upload_btn.value = () 
        else:
            # 4b. Oops, something went wrong in the backend. Show the error.
            with main_output: clear_output(); print(f"Error: {res['error']}")
    except Exception as e:
        # Catch any unexpected system errors
        print(f"System Error: {e}")

# Connect the upload button to the main function
upload_btn.observe(on_upload, names='value')

# ==========================================
# 5. LAUNCH THE APP!
# ==========================================
# Stack everything together and display it in the notebook.
display(widgets.VBox([header_html, upload_btn, main_output, history_box, comm_html]))

