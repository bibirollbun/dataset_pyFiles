import os
from kaggle_secrets import UserSecretsClient

# 1. Setup Secrets (Bridge Kaggle Secrets to Environment Variables)
user_secrets = UserSecretsClient()
try:
    api_key = user_secrets.get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = api_key
    os.environ["GEMINI_API_KEY"] = api_key
    print("âœ… API Key loaded from Kaggle Secrets")
except Exception as e:
    print(f"â�Œ Error loading secrets: {e}")

# 2. Clone Repository
!rm -rf capstone-agents-mvp
!git clone https://github.com/Codedkv/capstone-agents-mvp.git

print("âœ… Repository cloned successfully")



# Suppress pip warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 3. Install Dependencies
%cd capstone-agents-mvp
!pip install -r requirements.txt -q 2>&1 | grep -v "WARNING\|ERROR: pip's dependency"
print("âœ… Dependencies installed")



# 5. Run the Agent Pipeline
os.chdir("/kaggle/working/capstone-agents-mvp")
!python main.py


# 6. Display the Generated HTML Report (with turquoise headings injected)
from IPython.display import HTML, display
import os

report_path = "/kaggle/working/capstone-agents-mvp/output/analysis_report.html"

if os.path.exists(report_path):
    print("ğŸ“Š Rendering Analysis Report:\n")
    with open(report_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    
    # Inject CSS directly into the HTML
    turquoise_css = """
    <style>
        h1, h2, h3, h4, h5, h6 {
            color: #008B8B !important;
        }
    </style>
    """
    
    # Insert CSS at the beginning of the HTML
    modified_html = turquoise_css + html_content
    html_escaped = modified_html.replace('"', '&quot;')
    
    enhanced_html = f"""
    <div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px; margin-top: 20px;">
        <iframe srcDoc="{html_escaped}" 
                style="width: 100%; height: 900px; border: none; border-radius: 5px;"></iframe>
    </div>
    """
    display(HTML(enhanced_html))
else:
    print("âš ï¸� Report not found.")


