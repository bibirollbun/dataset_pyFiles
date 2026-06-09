# Cell 1: Final Installation and Configuration
# Install the core Google GenAI SDK and necessary libraries
!pip install google-genai pillow beautifulsoup4 lxml requests pandas ipywidgets -q
print("âœ… Installation complete.")

import os
import time
from datetime import datetime
import urllib.parse
import re
import random
import json
import pandas as pd
import numpy as np
from tempfile import NamedTemporaryFile
from io import BytesIO

# --- Core Google GenAI SDK Imports ---
from google import genai
from google.genai.types import Tool 

from PIL import Image as PILImage
from IPython.display import display, Markdown, HTML, Javascript, clear_output
import ipywidgets as widgets

# Configure Gemini API by handling the key
try:
    from kaggle_secrets import UserSecretsClient
    API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
except:
    # Use environment variable 'GEMINI_API_KEY' if set, otherwise use placeholder
    API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY_HERE") 
    
# --- THE FIX: Initialize Client with the API Key ---
try:
    if API_KEY == "YOUR_API_KEY_HERE":
         print("âš ï¸� WARNING: API key is a placeholder. Please replace 'YOUR_API_KEY_HERE' or set environment variable.")
    
    # Initialize the client, passing the key directly if available
    client_for_tools = genai.Client(api_key=API_KEY) # <--- DEFINES client_for_tools
    
    # Define a global model name constant
    MODEL_NAME = 'gemini-2.5-flash' # <--- DEFINES MODEL_NAME
    
    print("âœ… SDK Client initialized successfully!")
    
except Exception as e:
    print(f"â�Œ FATAL ERROR: Could not initialize Gemini Client. Check your API key. Error: {e}")
    client_for_tools = None
    MODEL_NAME = None


# Define a base class to structure our tools and ensure compatibility
class ToolAction:
    """Base class for defining tool methods with necessary type hints."""
    pass 

print("âœ… All dependencies and configuration attempts complete.")


# We use this function to mimic the Tool.action decorator, ensuring methods are visible.
def tool_action(func):
    """Placeholder decorator for Tool action."""
    return func

class MedicineSearchTool(ToolAction):
    """
    Tool to search and estimate medicine prices from various Indian online pharmacies.
    It uses web scraping for 1mg (if possible) and Gemini for estimation.
    """
    
    def __init__(self, client_instance, model_name):
        self.client = client_instance
        self.model_name = model_name
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        }
        self.timeout = 8
    
    def clean_medicine_name(self, name: str) -> str:
        """Clean medicine names for better search"""
        cleaned = re.sub(r'\s*\d+\s*(mg|ml|tab|cap|syrup|tablet|capsule)', '',  
                         name, flags=re.IGNORECASE)
        return cleaned.strip()

    def _estimate_with_ai(self, medicine_name: str, vendor: str) -> dict:
        """Use Gemini AI to estimate prices"""
        prompt = f"""You are a pharmacy price expert in India.
Estimate the retail price for a **standard strip (10 tablets/capsules) or 100ml bottle** of: {medicine_name}

Consider typical pricing on {vendor} in India (2024-2025).

Return ONLY a JSON object:
{{"price": <number>, "confidence": "high/medium/low", "notes": "<brief note>"}}"""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            
            data = json.loads(response.text)
            base_price = float(data.get('price', 100))
            
            # Vendor-specific price adjustments (to simulate market variation)
            factor_map = {
                "1mg": random.uniform(0.90, 1.05), "PharmEasy": random.uniform(0.88, 1.02),
                "Netmeds": random.uniform(0.95, 1.05), "Apollo 24/7": random.uniform(0.98, 1.10),
                "Local Chemist": random.uniform(0.85, 1.05),
            }
            
            factor = factor_map.get(vendor, 1.0)
            adjusted_price = base_price * factor
            
            return {
                "price": round(adjusted_price, 2), "available": True,
                "delivery": "2-3 Days",
                "confidence": data.get('confidence', 'medium'), "estimated": True
            }
        except Exception as e:
            return {
                "price": 150.0, "available": True, "delivery": "2-3 Days",
                "estimated": True, "confidence": "error"
            }

    @tool_action
    def search_prices(self, medicine_name: str) -> str:
        """
        Searches the price of a single medicine across multiple online and local vendors. 
        Returns a JSON string containing the results from 1mg, PharmEasy, Netmeds, Apollo 
        24/7, and Local Chemist.
        """
        results = {}
        
        # --- 1mg Search: Prioritized Attempt (Web Scrape) ---
        is_scraped = False
        try:
            clean_name = self.clean_medicine_name(medicine_name)
            search_url = f"https://www.1mg.com/search/all?name={urllib.parse.quote(clean_name)}"
            response = requests.get(search_url, headers=self.headers, timeout=self.timeout)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                price_elem = soup.find('span', {'class': 'PriceTag_price-tag__c3z9e'}) 
                
                if price_elem:
                    price_text = price_elem.get_text()
                    numbers = re.findall(r'\d+\.?\d*', price_text.replace(',', ''))
                    if numbers:
                        results["1mg"] = {"price": float(numbers[0]), "available": True, "delivery": "2-3 Days", "estimated": False, "source": "Scrape"}
                        is_scraped = True
        except:
            pass 

        # 2. Fallback to AI estimation if scraping failed
        if not is_scraped:
             results["1mg"] = self._estimate_with_ai(medicine_name, "1mg")
             results["1mg"]["source"] = "AI Estimate"
             
        # --- Other Vendors (AI estimation for robustness) ---
        results["PharmEasy"] = self._estimate_with_ai(medicine_name, "PharmEasy")
        results["Netmeds"] = self._estimate_with_ai(medicine_name, "Netmeds")
        results["Apollo 24/7"] = self._estimate_with_ai(medicine_name, "Apollo 24/7")
        results["Local Chemist"] = {"price": self._estimate_with_ai(medicine_name, "Local Chemist")['price'], "available": True, "delivery": "1 Hour âš¡", "estimated": True}
        
        return json.dumps(results)

# Instantiate the tool using the client and model name defined in Cell 1
search_tool_instance = MedicineSearchTool(client_for_tools, MODEL_NAME)
print("âœ… MedicineSearchTool created!")


# Cell 3: Final Version: Prescription OCR Tool

# We use this function to mimic the Tool.action decorator, ensuring methods are visible.
def tool_action(func):
    """Placeholder decorator for Tool action."""
    return func

class PrescriptionOCRTool(ToolAction):
    """
    Tool to extract medicine names from a prescription image file using Gemini Vision.
    """
    
    def __init__(self, client_instance, model_name): # <-- Correct initialization
        self.client = client_instance
        self.model_name = model_name

    @tool_action
    def extract_medicines(self, image_path: str) -> str:
        """
        Extracts ALL medicine names with dosage (e.g., 'Paracetamol 650mg') from a prescription image file.
        The image_path is the local file path saved in the execution context.
        Returns a JSON string list of medicine names and confidence.
        """
        
        ocr_prompt = """You are a medical prescription analyzer. Extract ALL medicine names with dosage 
        from this prescription image. Ignore doctor/patient details.

        Return ONLY a JSON object:
        { "medicines": ["Medicine1 dosage", "Medicine2 dosage"], "confidence": "high/medium/low", "notes": "observations" }"""
        
        try:
            image_file = PILImage.open(image_path)
            
            response = self.client.models.generate_content( # <-- CORRECT API CALL
                model=self.model_name,
                contents=[image_file, ocr_prompt],
                config=genai.types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            
            data = json.loads(response.text)
            
            if not data.get('medicines'):
                 raise ValueError("No medicines detected by AI.")

            return json.dumps({
                "medicines": data.get('medicines', []),
                "confidence": data.get('confidence', 'medium'),
            })
            
        except Exception as e:
            print(f"âš ï¸� OCR Tool Error: {e}")
            return json.dumps({
                "medicines": [],
                "confidence": "error",
                "notes": f"OCR Error: {str(e)[:100]}"
            })

# Instantiate the tool using the client and model name defined in Cell 1
ocr_tool_instance = PrescriptionOCRTool(client_for_tools, MODEL_NAME)
print("âœ… PrescriptionOCRTool created!")


# Cell 4: Enhanced Agent Workflow (COMPLETELY FIXED - NO SPACES IN FORMATTING)
# Global variable to store analysis results
analysis_results = {}

def run_pharma_scout_agent_adk(image_path, output_area, progress, status_text, ocr_tool, search_tool):
    """
    Main agent workflow using the structured OCR and Search Tools with enhanced UI. 
    """
    global analysis_results
    
    def update_progress(step, message):
        progress.value = step
        status_text.value = message
        time.sleep(0.01)
        
    with output_area:
        clear_output(wait=True)
        
        # STEP 1: Load and Display Image
        update_progress(10, "ğŸ“¸ Loading prescription image...")
        try:
            image = PILImage.open(image_path)
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            image_b64 = __import__("base64").b64encode(buffered.getvalue()).decode()
            
            display(HTML(f'''
            <div style="display: flex; justify-content: space-around; align-items: center; margin: 20px 0;">
                <div style="text-align: center;">
                    <h3 style="color: #764ba2; margin-bottom: 10px;">ğŸ”� Analysis Started! </h3>
                    <img src="data:image/png;base64,{image_b64}" width="400" style="border:3px solid #764ba2; border-radius:15px; box-shadow: 0 4px 12px rgba(118,75,162,0.3);"/>
                </div>
            </div>
            '''))
        except Exception as e:
             display(Markdown(""))

        # STEP 2: OCR - Call the tool
        update_progress(30, "ğŸ”� Scanning prescription with AI...")
        ocr_json_str = ocr_tool.extract_medicines(image_path=image_path)
        ocr_result = json.loads(ocr_json_str)
        medicines = ocr_result.get('medicines', [])
        
        if not medicines:
            update_progress(100, "â�Œ Failed to detect medicines.")
            display(Markdown(f"### â�Œ ERROR: {ocr_result.get('notes', 'Could not extract any medicines.')}"))
            os.unlink(image_path)
            return
            
        update_progress(45, f"ğŸ’Š Found {len(medicines)} medicines")
        
        # Display detected medicines with enhanced styling
        medicines_html = f'''
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 10px; color: white; margin: 15px 0;">
            <h3 style="margin: 0 0 15px 0;">ğŸ’Š Medicines Detected ({ocr_result['confidence']. upper()} confidence)</h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">
        '''
        
        for i, med in enumerate(medicines, 1):
            medicines_html += f'''
                <div style="background: rgba(255,255,255,0.1); padding: 10px; border-radius: 8px; border-left: 4px solid #ffd700;">
                    <strong>{i}.</strong> {med}
                </div>
            '''
        
        medicines_html += '</div></div>'
        display(HTML(medicines_html))

        # STEP 3: Search Platforms
        update_progress(50, "ğŸŒ� Searching prices across platforms...")
        all_results = {}
        
        for i, med in enumerate(medicines):
            msg = f"ğŸ”� Searching: {med}..."
            update_progress(50 + int((i+1) * 30 / len(medicines)), msg)
            search_json_str = search_tool.search_prices(medicine_name=med)
            all_results[med] = json. loads(search_json_str)
            time.sleep(1)

        # STEP 4: Calculate Best Deals & Store Results
        update_progress(85, "ğŸ“Š Analyzing deals...")
        vendor_totals = {}
        
        if all_results and list(all_results.values()):
            first_vendor_keys = list(list(all_results. values())[0].keys())
            for vendor in first_vendor_keys:
                vendor_totals[vendor] = {"total": 0, "items": [], "delivery": "N/A", "errors": 0}
        
        for med, vendors in all_results.items():
            for vendor, data in vendors. items():
                if data.get('confidence') == 'error' or data.get('price') is None:
                    vendor_totals[vendor]['errors'] += 1
                    continue

                vendor_totals[vendor]['total'] += data['price']
                vendor_totals[vendor]['items']. append({"medicine": med, "price": data['price']})
                vendor_totals[vendor]['delivery'] = data.get('delivery', 'N/A')
        
        valid_totals = [v['total'] for v in vendor_totals.values() if v['total'] > 0]
        
        if not valid_totals:
            update_progress(100, "â�Œ No valid price data found.")
            display(Markdown("### â�Œ ERROR: Could not retrieve or estimate prices. "))
            os.unlink(image_path)
            return
            
        max_total = max(valid_totals)
        for vendor in vendor_totals:
            vendor_totals[vendor]['savings'] = max_total - vendor_totals[vendor]['total']
        
        sorted_vendors = sorted([(k, v) for k, v in vendor_totals.items() if v['total'] > 0], key=lambda x: x[1]['total'])
        
        analysis_results = {
            "medicines": medicines, 
            "all_results": all_results, 
            "vendor_totals": vendor_totals,
            "sorted_vendors": sorted_vendors, 
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # STEP 5: Display Results with Enhanced UI
        update_progress(100, "âœ… Analysis Complete!")
        
        display(HTML("<hr style='border: 1px solid #444; margin: 20px 0;'>"))
        
        # Best Deals Section
        winner_name, winner_data = sorted_vendors[0]
        
        # Format prices BEFORE using in strings
        total_price = round(winner_data['total'], 2)
        savings_price = round(winner_data['savings'], 2)
        
        best_deals_html = f'''
        <div style="margin: 20px 0;">
            <h2 style="color: #ffd700; margin: 0 0 15px 0;">ğŸ¥‡ BEST DEALS ANALYSIS</h2>
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 25px; border-radius: 12px; color: white; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
                <h3 style="margin: 0 0 10px 0; font-size: 24px;">ğŸ�¯ RECOMMENDED: {winner_name}</h3>
                <div style="font-size: 36px; font-weight: bold; margin: 15px 0; color: #ffd700;">â‚¹{total_price}</div>
                <div style="font-size: 16px; margin: 10px 0; opacity: 0.95;">
                    <strong>ğŸ’° Save â‚¹{savings_price}</strong> compared to other options
                </div>
                <div style="font-size: 14px; opacity: 0.9;">
                    â�±ï¸� Delivery: {winner_data['delivery']}
                </div>
            </div>
        </div>
        '''
        display(HTML(best_deals_html))
        
        # Order Breakdown Section
        breakdown_html = '<div style="margin: 20px 0;"><h3 style="color: #764ba2;">ğŸ“‹ Order Breakdown:</h3><ul style="background: #2a2a2a; padding: 15px 25px; border-radius: 8px; border-left: 4px solid #764ba2;">'
        
        for item in winner_data['items']:
            item_price = round(item['price'], 2)
            breakdown_html += f'<li style="margin: 8px 0; font-size: 15px;"><strong>{item["medicine"]}</strong> â†’ <span style="color: #ffd700;">â‚¹{item_price}</span></li>'
        
        breakdown_html += '</ul></div>'
        display(HTML(breakdown_html))
        
        # All Options Compared Section
        comparison_html = '<div style="margin: 20px 0;"><h3 style="color: #764ba2;">ğŸ“Š All Options Compared:</h3><div style="display: flex; flex-direction: column; gap: 10px;">'
        
        for rank, (vendor, data) in enumerate(sorted_vendors, 1):
            emoji = "ğŸ¥‡" if rank == 1 else "ğŸ¥ˆ" if rank == 2 else "ğŸ¥‰" if rank == 3 else "ğŸ’Š"
            error_note = f" (âš ï¸� {data['errors']} items failed)" if data['errors'] > 0 else ""
            bg_color = "#2a2a2a" if rank != 1 else "#1a3a1a"
            border_color = "#764ba2" if rank == 1 else "#444"
            
            vendor_total = round(data['total'], 2)
            vendor_savings = round(data['savings'], 2)
            
            comparison_html += f'''
            <div style="background: {bg_color}; padding: 15px; border-radius: 8px; border-left: 4px solid {border_color};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div><strong>{emoji} #{rank} {vendor}</strong></div>
                    <div style="color: #ffd700; font-weight: bold; font-size: 16px;">â‚¹{vendor_total}</div>
                </div>
                <div style="font-size: 12px; opacity: 0.8; margin-top: 5px;">Save: â‚¹{vendor_savings}</div>
            </div>
            '''
        
        comparison_html += '</div></div>'
        display(HTML(comparison_html))
        
        display(HTML("<hr style='border: 1px solid #444; margin: 20px 0;'>"))
        
        # Action Buttons
        url_map = {
            "1mg": "https://www.1mg.com/search/all?filter=true&name=",
            "PharmEasy": "https://pharmeasy.in/search/all?name=",
            "Netmeds": "https://www.netmeds.com/products?q=",
            "Apollo 24/7": "https://www.apollopharmacy.in/search-medicines/",
            "Local Chemist": "https://www.google.com/maps/search/nearest+pharmacy"
        }
        
        base_url = url_map.get(winner_name, "https://www.google.com/search? q=buy+")
        first_medicine = medicines[0]. split()[0] if medicines else "medicines"
        target_url = f"{base_url}{urllib.parse.quote(first_medicine)}"
        
        def on_website_click(b):
            with output_area:
                # Generate the vendor URL with the first medicine
                vendor_urls = {
                    "1mg": "https://www.1mg.com/search/all?filter=true&name=",
                    "PharmEasy": "https://pharmeasy.in/search/all?name=",
                    "Netmeds": "https://www.netmeds.com/products?q=",
                    "Apollo 24/7": "https://www.apollopharmacy.in/search-medicines/",
                    "Local Chemist": "https://www.google.com/maps/search/nearest+pharmacy"
                }
                
                # Get the base URL for the winner vendor
                base_url = vendor_urls.get(winner_name, "https://www. google.com/search? q=")
                
                # Get first medicine name for search
                first_medicine = medicines[0].split()[0] if medicines else "medicines"
                
                # Create the full URL
                vendor_url = f"{base_url}{urllib.parse.quote(first_medicine)}"
                
                display(Markdown(f"### ğŸš€ Opening {winner_name}..."))
                
                # Display the link with auto-open script
                html_code = f'''
                <div style="background: linear-gradient(135deg, #2a5a2a 0%, #1a4a1a 100%); 
                            padding: 20px; border-radius: 10px; color: white; margin: 10px 0; 
                            text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
                    <p style="margin: 0 0 15px 0; font-size: 16px;">
                        âœ… Redirecting to <strong>{winner_name}</strong>... 
                    </p>
                    <a href="{vendor_url}" target="_blank" 
                       style="display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                              color: white; padding: 15px 30px; text-decoration: none; 
                              border-radius: 10px; font-weight: bold; cursor: pointer;
                              box-shadow: 0 4px 8px rgba(0,0,0,0.2); transition: transform 0.2s;">
                        ğŸ›’ Click here to order from {winner_name}
                    </a>
                    <script>
                        window. open("{vendor_url}", "_blank");
                    </script>
                </div>
                '''
                display(HTML(html_code))

        def on_download_click(b):
            with output_area:
                if not analysis_results: 
                    display(Markdown("â�Œ No analysis results to download. "))
                    return
                    
                try:
                    display(Markdown("### ğŸ“¥ Generating CSV Report..."))
                    rows = []
                    
                    # Get the winner vendor (cheapest option)
                    winner_vendor = analysis_results['sorted_vendors'][0][0]
                    
                    # STEP 1: Add winner vendor's medicines FIRST with border
                    rows.append({
                        "Medicine": f"*** {winner_vendor. upper()} (WINNER - CHEAPEST) ***", 
                        "Vendor": "",
                        "Price": "",
                        "Delivery": "", 
                        "Available": "",
                        "Estimated": ""
                    })
                    
                    for med in analysis_results['medicines']:
                        if winner_vendor in analysis_results['all_results'][med]:
                            data = analysis_results['all_results'][med][winner_vendor]
                            price = data. get('price', 0)
                            rows.append({
                                "Medicine": med, 
                                "Vendor": winner_vendor,
                                "Price": round(price, 2),
                                "Delivery": data.get('delivery', 'N/A'), 
                                "Available": "Yes" if data.get('available') else "No",
                                "Estimated": "Yes" if data.get('estimated') else "No"
                            })
                    
                    # Add a separator row
                    rows.append({
                        "Medicine": "", 
                        "Vendor": "",
                        "Price": "",
                        "Delivery": "", 
                        "Available": "",
                        "Estimated": ""
                    })
                    
                    rows.append({
                        "Medicine": "*** OTHER VENDORS ***", 
                        "Vendor": "",
                        "Price": "",
                        "Delivery": "", 
                        "Available": "",
                        "Estimated": ""
                    })
                    
                    # STEP 2: Add all other vendors' medicines
                    for med in analysis_results['medicines']:
                        for vendor, data in analysis_results['all_results'][med].items():
                            if vendor != winner_vendor:
                                price = data.get('price', 0)
                                rows.append({
                                    "Medicine": med, 
                                    "Vendor": vendor,
                                    "Price": round(price, 2),
                                    "Delivery": data.get('delivery', 'N/A'), 
                                    "Available": "Yes" if data.get('available') else "No",
                                    "Estimated": "Yes" if data.get('estimated') else "No"
                                })
                    
                    # STEP 3: Add summary section
                    rows.append({
                        "Medicine": "", 
                        "Vendor": "",
                        "Price": "",
                        "Delivery": "", 
                        "Available": "",
                        "Estimated": ""
                    })
                    
                    rows.append({
                        "Medicine": "*** SUMMARY ***", 
                        "Vendor": "Vendor",
                        "Price": "Total Cost",
                        "Delivery": "Savings", 
                        "Available": "",
                        "Estimated": ""
                    })
                    
                    for vendor, data in analysis_results['vendor_totals'].items():
                        total_cost = round(data['total'], 2)
                        savings = round(data['savings'], 2)
                        emoji = "ğŸ¥‡ WINNER" if vendor == winner_vendor else "ğŸ’Š"
                        rows.append({
                            "Medicine": emoji, 
                            "Vendor": vendor,
                            "Price": total_cost,
                            "Delivery": savings, 
                            "Available": data.get('delivery', 'N/A'),
                            "Estimated": ""
                        })
                    
                    df = pd.DataFrame(rows)
                    filename = f"pharma_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    df.to_csv(filename, index=False)
                    display(Markdown(f"âœ… **CSV File Created:** `{filename}`"))
                    display(Markdown("ğŸ“Š **Preview of CSV:**"))
                    display(df)
                    
                except Exception as e:
                    display(Markdown(f"â�Œ **Error generating CSV:** {str(e)}"))
        
        website_btn = widgets.Button(
            description=f'ğŸ›’ Order from {winner_name}',
            button_style='success',
            layout=widgets.Layout(width='250px', height='45px'),
            style={'font_weight': 'bold'}
        )
        download_btn = widgets.Button(
            description='ğŸ“¥ Download CSV Report',
            button_style='info',
            layout=widgets.Layout(width='250px', height='45px'),
            style={'font_weight': 'bold'}
        )
        
        website_btn.on_click(on_website_click)
        download_btn.on_click(on_download_click)
        
        button_container = widgets.HBox(
            [website_btn, download_btn],
            layout=widgets.Layout(justify_content='center', gap='15px', padding='15px')
        )
        display(button_container)
        
        if os.path.exists(image_path):
             os.unlink(image_path)

print("âœ… Enhanced agent workflow created!")


# Cell 5: Enhanced UI Components and Display (FIXED BUTTON ALIGNMENT)
# --- Global UI Components with Enhanced Styling ---
current_image_data = None
progress = widgets.IntProgress(
    value=0, 
    min=0, 
    max=100, 
    description='Progress:', 
    style={'bar_color': '#764ba2'},
    layout=widgets.Layout(width='100%', height='20px')
)

status_text = widgets.Label(
    value="ğŸ‘† Upload prescription to begin", 
    style={'font_weight': 'bold', 'font_size': '14px', 'text_color': '#ffffff'}
)

output_area = widgets.Output(
    layout=widgets.Layout(
        border='1px solid #444',
        padding='15px',
        margin='10px 0',
        border_radius='10px',
        background_color='#1a1a1a'
    )
)

# Upload widget with enhanced styling - FIXED LAYOUT
uploader = widgets.FileUpload(
    accept='image/*',
    multiple=False,
    description='ğŸ“� Upload ',
    button_style='info',
    layout=widgets.Layout(
        width='auto',
        height='50px',
        padding='0px',
        min_width='180px'
    )
)

# Analyze button with enhanced styling - FIXED LAYOUT
analyze_btn = widgets.Button(
    description='ğŸ”� ANALYZE PRESCRIPTION',
    button_style='success',
    disabled=True,
    layout=widgets.Layout(
        width='auto',
        height='50px',
        padding='0px',
        min_width='250px'
    ),
    tooltip='Click to analyze the uploaded prescription'
)

# New Analysis button - FIXED LAYOUT
new_analysis_btn = widgets.Button(
    description='â�• New Analysis',
    button_style='info',
    layout=widgets.Layout(
        width='auto',
        height='50px',
        padding='0px',
        min_width='180px'
    ),
    tooltip='Start a new analysis'
)

# --- Enhanced Event Handlers ---
def on_upload(change):
    global current_image_data
    if uploader.value:
        try:
            uploaded = uploader.value
            current_image_data = list(uploaded.values())[0]['content'] if isinstance(uploaded, dict) else uploaded[0]['content']
            
            analyze_btn.disabled = False
            file_name = list(uploader.value.values())[0]['name']
            status_text.value = f"âœ… Image loaded: {file_name} - Click ANALYZE"
            progress.value = 0
            with output_area: 
                clear_output()
        except Exception as e:
            status_text.value =""

def on_analyze(b):
    global current_image_data
    if current_image_data is None:
        status_text.value = "âš ï¸� Please upload an image first."
        return
    
    analyze_btn.disabled = True
    
    temp_file = NamedTemporaryFile(suffix='.jpg', delete=False)
    image = PILImage.open(BytesIO(current_image_data))
    image.save(temp_file.name, 'JPEG', quality=95)
    temp_path = temp_file.name
    temp_file.close()
    
    try:
        run_pharma_scout_agent_adk(temp_path, output_area, progress, status_text, ocr_tool_instance, search_tool_instance)
    except Exception as e:
        with output_area:
            display(Markdown(f"### â�Œ Critical Error during execution: {str(e)}"))
        if os.path.exists(temp_path):
             os.unlink(temp_path)
    finally:
        analyze_btn.disabled = False

def on_new_analysis(b):
    global current_image_data
    current_image_data = None
    uploader.value = ()
    analyze_btn.disabled = True
    status_text.value = "ğŸ‘† Upload prescription to begin"
    progress.value = 0
    with output_area:
        clear_output()

# Attach handlers
uploader.observe(on_upload, names='value')
analyze_btn.on_click(on_analyze)
new_analysis_btn.on_click(on_new_analysis)

# --- Display Enhanced Header ---
header_html = """
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
             padding: 25px; border-radius: 15px; text-align: left; 
             color: white; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
    <h1 style="margin: 0 0 10px 0; font-size: 32px;">ğŸ�¥ Pharma-Scout Agent</h1>
    <p style="margin: 5px 0; font-size: 14px; opacity: 0.9;">AI-Powered Medicine Price Comparison & Smart Ordering</p>
    <div style="margin-top: 10px; font-size: 12px; opacity: 0.85;">
        ğŸ“¸ Scan â†’ ğŸ¤– AI Analysis â†’ ğŸ’° Compares Prices â†’ ğŸ�¯ Best Deals
    </div>
</div>
"""

# --- Display Final Interface with Enhanced Layout ---
display(HTML(header_html))

# Button section with enhanced styling - PROPERLY ALIGNED
button_box = widgets.HBox(
    [uploader, analyze_btn, new_analysis_btn],
    layout=widgets.Layout(
        justify_content='center',
        align_items='center',
        padding='15px',
        gap='20px'
    )
)

# Status and progress section
status_box = widgets.VBox(
    [status_text, progress],
    layout=widgets.Layout(width='100%', padding='10px')
)

# Main container
main_container = widgets.VBox([
    button_box,
    status_box,
    output_area
], layout=widgets.Layout(
    width='100%',
    align_items='stretch',
    padding='10px'
))

display(main_container)

print("=" * 70)
print("ğŸš€ PHARMA-SCOUT AGENT READY - Enhanced UI Loaded")
print("ğŸ“� Upload prescription â†’ Click ANALYZE â†’ Get best deals")
print("=" * 70)

