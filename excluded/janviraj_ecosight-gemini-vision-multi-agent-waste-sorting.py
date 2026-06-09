# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# EcoGuardian - Multi-Agent Waste Detection & Recycling Optimization
import os
print("ğŸš€ Setting up EcoGuardian project structure...")
# Create main project directory and all subfolders
!mkdir -p ecoguardian/tools ecoguardian/agents ecoguardian/memory ecoguardian/orchestration ecoguardian/utils ecoguardian/data
# Verify folder structure
print("ğŸ“� Project structure created:")
!find ecoguardian -type d | sort
print("âœ… EcoGuardian folder structure ready!")


# Correct TrashNet Dataset Setup for Kaggle
print("ğŸ“¥ Setting up TrashNet Dataset...")

# The correct TrashNet dataset on Kaggle is:

# Check for the dataset in common locations
dataset_paths = [
    "/kaggle/input/trashnet/data/dataset",  # Most common structure
    "/kaggle/input/trashnet/data",
    "/kaggle/input/trashnet/dataset",
    "/kaggle/input/trashnet",
    "/kaggle/input/garbage-classification/Garbage classification/Garbage classification",  # Alternative
]

found_path = None
for path in dataset_paths:
    if os.path.exists(path):
        found_path = path
        print(f"âœ… Dataset found at: {path}")
        break

if found_path:
    # Copy to our project structure
    !mkdir -p ecoguardian/data/trashnet
    !cp -r {found_path}/* ecoguardian/data/trashnet/ 2>/dev/null || echo "Using direct access"
    
    # Verify the copy worked
    if os.path.exists("ecoguardian/data/trashnet"):
        print("ğŸ“� Dataset successfully copied to project structure")
    else:
        print("ğŸ“� Using dataset from original location")
else:
    print("â�Œ TrashNet dataset not found in expected locations")
    print("""
ğŸ“� HOW TO ADD THE CORRECT TRASHNET DATASET:

1. Click '+ Add Data' button in Kaggle notebook
2. Search for: "trashnet"
3. Select the dataset: "trashnet" by yangyang111
4. Click 'Add' to attach it to your notebook
5. Wait for dataset to load (green checkmark)
6. Restart session: Session â†’ Restart Session  
7. Re-run all cells

The dataset should then be available at:
/kaggle/input/trashnet/data/dataset/

ğŸ“Š Original TrashNet Dataset Info:
- 2,527 images across 6 categories
- Categories: cardboard, glass, metal, paper, plastic, trash
- Image sizes: 512x384 pixels
- Research-quality dataset from Stanford
""")

# Verify what we can access
print("\nğŸ”� Checking dataset structure...")
if found_path:
    !find {found_path} -type d 2>/dev/null | head -15
else:
    # Check our project structure
    !find ecoguardian/data/trashnet -type d 2>/dev/null | head -10 || echo "No dataset found"

print("\nâœ… TrashNet setup complete!")


# Configure Python path to include our project
import sys
import os

# Add ecoguardian to Python path
sys.path.append('/kaggle/working/ecoguardian')

print("âœ… Python path configured for ecoguardian/ project structure")
print(f"Project root: /kaggle/working/ecoguardian")


# =============================================================================
# ğŸ”‘ SECURE GEMINI API CONFIGURATION (Using Kaggle Secrets)
# =============================================================================

import os
from kaggle_secrets import UserSecretsClient

print("ğŸ”‘ Configuring Gemini AI Integration Securely...")

try:
    # Get Gemini API key from Kaggle Secrets
    GEMINI_API_KEY = UserSecretsClient().get_secret("GEMINI_API_KEY")
    
    if GEMINI_API_KEY and GEMINI_API_KEY.startswith("AIza"):
        os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
        print("âœ… Gemini API key loaded securely from Kaggle Secrets!")
        print("ğŸš€ Real object detection will be used")
        print(f"ğŸ”� Key preview: {GEMINI_API_KEY[:10]}...")
    else:
        print("â�Œ Invalid Gemini API key format in Kaggle Secrets")
        print("ğŸ’¡ Please ensure your key starts with 'AIza'")
        os.environ["GEMINI_API_KEY"] = "MOCK_KEY_NO_VALID_KEY"
        print("ğŸ�­ Using mock mode - no valid Gemini API key detected")
        
except Exception as e:
    print(f"ğŸ”‘ Kaggle Secrets Error: {e}")
    print("""
ğŸ“� HOW TO ADD YOUR GEMINI API KEY SECURELY:

1. Click the 'Settings' tab in your Kaggle notebook
2. Scroll down to 'Secrets' section
3. Click 'Add new secret'
4. Set:
   - Name: GEMINI_API_KEY
   - Value: Your actual Gemini API key (starts with AIza...)
5. Click 'Save'
6. Restart your notebook session
7. Re-run this cell

ğŸ”’ Security Benefits:
   - API key never appears in your code
   - Key is encrypted and secure
   - Easy to rotate/update without code changes
    """)
    os.environ["GEMINI_API_KEY"] = "MOCK_KEY_SECRETS_ERROR"
    print("ğŸ�­ Using mock mode until valid API key is configured")


%%writefile ecoguardian/agents/agent_base.py
# Simplified agent foundation without complex inheritance
import logging
from typing import Any, Dict, Optional
from dataclasses import dataclass

@dataclass
class AgentContext:
    session_id: str
    user_id: Optional[str] = None
    location: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

def create_agent_logger(name: str):
    """Create a logger for any agent"""
    logger = logging.getLogger(f"agent.{name}")
    return logger

def log_metric(logger, metric_name: str, value: float, tags: Dict[str, str] = None):
    """Log observability metrics for any agent"""
    tags = tags or {}
    tags['agent'] = logger.name
    logger.info(f"METRIC:{metric_name}={value} {tags}")


%%writefile ecoguardian/tools/vision_provider.py
import random
from abc import ABC, abstractmethod
from typing import Dict, Any
import base64
import os
import asyncio

class VisionProvider(ABC):
    @abstractmethod
    async def analyze_image(self, image_b64: str) -> Dict[str, Any]:
        pass

class MockVisionProvider(VisionProvider):
    """Mock vision provider for testing - uses TrashNet categories"""
    
    async def analyze_image(self, image_b64: str) -> Dict[str, Any]:
        await asyncio.sleep(0.1)
        
        # Mock detection based on TrashNet categories
        trashnet_categories = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        detected_category = random.choice(trashnet_categories)
        
        mock_items = [
            {"name": f"{detected_category} item", "confidence": random.uniform(0.85, 0.98), "bbox": [100, 100, 50, 50]},
        ]
        
        # Sometimes detect multiple items
        if random.random() > 0.7:
            second_category = random.choice([c for c in trashnet_categories if c != detected_category])
            mock_items.append({
                "name": f"{second_category} item", 
                "confidence": random.uniform(0.75, 0.90), 
                "bbox": [200, 150, 40, 40]
            })
        
        return {
            "items": mock_items,
            "analysis_confidence": random.uniform(0.8, 0.95),
            "model_version": "mock-trashnet-v1.0",
            "provider": "mock"
        }

class GeminiVisionProvider(VisionProvider):
    """Real Google Gemini Vision implementation"""
    
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key or self.api_key == "MOCK_MODE_NO_VALID_KEY":
            raise ValueError("Gemini API key not found. Please set GEMINI_API_KEY in Kaggle Secrets")
        
        # Initialize Gemini client
        try:
            from google import genai
            from google.genai import types
            self.genai = genai
            self.types = types
            self.client = genai.Client(api_key=self.api_key)
            print("âœ… Gemini Vision client initialized successfully")
        except ImportError:
            raise ImportError("Google GenAI library not installed. Run: pip install google-genai")
    
    async def analyze_image(self, image_b64: str) -> Dict[str, Any]:
        """Analyze image using real Gemini Vision API"""
        try:
            print("ğŸ”® Calling real Gemini Vision API...")
            
            # Decode base64 image
            image_data = base64.b64decode(image_b64)
            
            # Create the prompt for waste detection
            prompt = """
            Analyze this image and identify all waste items visible. 
            For each item, provide:
            - The type of waste (e.g., plastic bottle, paper cup, glass jar, metal can, cardboard, food waste, etc.)
            - Its condition and approximate size
            
            Focus on identifying recyclable materials, compostable items, and landfill waste.
            Be specific about the material type when possible.
            Return only the list of detected items without additional commentary.
            """
            
            # Create the image part
            image_part = self.types.Part.from_bytes(
                data=image_data,
                mime_type="image/jpeg"
            )
            
            # Use correct model name
            response = self.client.models.generate_content(
                model="gemini-1.5-flash-latest",  
                contents=[image_part, prompt]
            )
            
            # Parse the response
            detected_items = self._parse_gemini_response(response.text)
            
            return {
                "items": detected_items,
                "analysis_confidence": 0.92,
                "model_version": "gemini-1.5-flash-latest", 
                "provider": "gemini",
                "raw_response": response.text[:200] + "..." if len(response.text) > 200 else response.text  # For debugging
            }
            
        except Exception as e:
            print(f"â�Œ Gemini Vision API error: {e}")
            # Fallback to mock provider
            print("ğŸ”„ Falling back to mock provider...")
            mock_provider = MockVisionProvider()
            return await mock_provider.analyze_image(image_b64)
    
    def _parse_gemini_response(self, response_text: str) -> list:
        """Parse Gemini response into structured waste items"""
        items = []
        
        # Simple parsing - look for waste-related terms
        lines = response_text.split('\n')
        
        waste_keywords = {
            'plastic': ['plastic', 'bottle', 'container', 'bag', 'wrapper', 'packaging'],
            'paper': ['paper', 'cardboard', 'newspaper', 'magazine', 'box'],
            'glass': ['glass', 'bottle', 'jar', 'container'],
            'metal': ['metal', 'can', 'aluminum', 'tin', 'container'],
            'cardboard': ['cardboard', 'box', 'packaging'],
            'trash': ['trash', 'garbage', 'waste', 'debris', 'litter']
        }
        
        for line in lines:
            line = line.strip().lower()
            if not line or len(line) < 10:  # Skip very short lines
                continue
                
            # Determine category based on keywords
            detected_category = 'trash'  # default
            confidence = 0.8
            
            for category, keywords in waste_keywords.items():
                if any(keyword in line for keyword in keywords):
                    detected_category = category
                    confidence = min(0.95, confidence + 0.1)  # Boost confidence for matches
                    break
            
            # Create item with mock bounding box
            bbox = [
                random.randint(50, 300),
                random.randint(50, 200),
                random.randint(30, 100),
                random.randint(30, 100)
            ]
            
            items.append({
                "name": f"{detected_category} object",
                "confidence": confidence,
                "bbox": bbox,
                "description": line[:100],  # First 100 chars of description
                "category_hint": detected_category
            })
        
        # If no items detected, add a default
        if not items:
            items.append({
                "name": "unidentified waste",
                "confidence": 0.5,
                "bbox": [100, 100, 50, 50],
                "description": "Waste item detected but not specifically identified",
                "category_hint": "trash"
            })
        
        return items[:5]  # Limit to 5 items max


%%writefile ecoguardian/tools/waste_db.py
from typing import Dict, Any, Optional

class WasteDB:
    """Tool for waste classification database - aligned with TrashNet categories"""
    
    def __init__(self, db_path: Optional[str] = None):
        self.classification_rules = self._load_default_rules()
    
    def _load_default_rules(self) -> Dict[str, Any]:
        # Classification rules aligned with TrashNet categories
        return {
            "cardboard": {"category": "recyclable", "material": "cardboard", "processing": "flatten"},
            "glass": {"category": "recyclable", "material": "glass", "processing": "separate_colors"},
            "metal": {"category": "recyclable", "material": "metal", "processing": "clean"},
            "paper": {"category": "recyclable", "material": "paper", "processing": "dry"},
            "plastic": {"category": "recyclable", "material": "plastic", "processing": "check_code"},
            "trash": {"category": "landfill", "material": "mixed", "notes": "non-recyclable"},
            # Additional common items
            "bottle": {"category": "recyclable", "material": "plastic", "processing": "rinse"},
            "can": {"category": "recyclable", "material": "metal", "processing": "rinse"},
            "container": {"category": "recyclable", "material": "mixed", "processing": "check_label"},
        }
    
    async def classify_item(self, item_name: str, location: Optional[str] = None) -> Dict[str, Any]:
        item_lower = item_name.lower()
        
        # Match against TrashNet categories and common patterns
        for key, rule in self.classification_rules.items():
            if key in item_lower:
                result = rule.copy()
                # Add location-specific adjustments if needed
                if location and self._has_location_override(location, key):
                    result.update(self._get_location_override(location, key))
                return result
        
        # Default classification for unknown items
        return {"category": "landfill", "material": "unknown", "notes": "unidentified"}
    
    def _has_location_override(self, location: str, item_key: str) -> bool:
        # Simplified location override check
        return False
    
    def _get_location_override(self, location: str, item_key: str) -> Dict[str, Any]:
        # Simplified location override rules
        return {}


%%writefile ecoguardian/tools/location_finder.py
from typing import Dict, Any, List, Optional

class LocationFinder:
    """Tool for location-specific waste disposal rules"""
    
    def __init__(self):
        self.location_rules = self._load_default_rules()
    
    def _load_default_rules(self) -> Dict[str, Any]:
        return {
            "NYC": {
                "recyclable": {
                    "instructions": "Place in blue recycling bin - separate paper and metal/glass/plastic",
                    "facilities": ["Curbside pickup", "DSNY recycling centers"],
                    "preparation": "Rinse containers, flatten boxes, no plastic bags"
                },
                "compost": {
                    "instructions": "Brown bin or designated compost drop-off",
                    "facilities": ["DSNY compost sites", "Farmer's markets"],
                    "preparation": "Use compostable bags, no plastic contamination"
                },
                "landfill": {
                    "instructions": "Black bin for non-recyclable waste",
                    "facilities": ["Curbside pickup"],
                    "preparation": "Bag securely to prevent litter"
                }
            },
            "SF": {
                "recyclable": {
                    "instructions": "Blue bin for mixed recyclables",
                    "facilities": ["Curbside pickup", "Recology centers"],
                    "preparation": "No plastic bags, rinse containers"
                },
                "compost": {
                    "instructions": "Green bin for compostables",
                    "facilities": ["Curbside pickup", "Community compost"],
                    "preparation": "No plastic contamination"
                }
            },
            "default": {
                "recyclable": {
                    "instructions": "Check local recycling guidelines",
                    "facilities": ["Local recycling center"],
                    "preparation": "Rinse and sort by material"
                },
                "compost": {
                    "instructions": "Compost bin or municipal collection",
                    "facilities": ["Local compost facility"],
                    "preparation": "No plastic contamination"
                },
                "landfill": {
                    "instructions": "Regular trash bin",
                    "facilities": ["Curbside pickup", "Landfill site"],
                    "preparation": "Bag securely"
                }
            }
        }
    
    async def get_disposal_instructions(self, category: str, location: Optional[str] = None) -> Dict[str, Any]:
        location_key = location if location and location in self.location_rules else "default"
        
        # Get category rules with fallback
        category_rules = self.location_rules[location_key].get(
            category, 
            self.location_rules["default"].get(category, {})
        )
        
        return {
            "instructions": category_rules.get("instructions", "Dispose according to local regulations"),
            "facilities": category_rules.get("facilities", []),
            "preparation": category_rules.get("preparation", "Follow general guidelines"),
            "location": location_key
        }


%%writefile ecoguardian/tools/pdf_generator.py

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing, Rect
from reportlab.graphics.charts.barcharts import VerticalBarChart
import io
from datetime import datetime
from typing import Dict, Any
import os

class PDFGenerator:
    """Enhanced PDF generator with professional styling and branding"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup enhanced custom styles for professional reports"""
        # EcoGuardian color scheme
        self.primary_color = colors.HexColor('#2E7D32')  # Eco green
        self.secondary_color = colors.HexColor('#4CAF50')  # Light green
        self.accent_color = colors.HexColor('#FF9800')    # Orange
        self.dark_color = colors.HexColor('#1B5E20')      # Dark green
        self.light_color = colors.HexColor('#E8F5E8')     # Light background
        
        # Enhanced title style
        self.styles.add(ParagraphStyle(
            name='EcoTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=20,
            textColor=self.primary_color,
            alignment=1,  # Center
            fontName='Helvetica-Bold'
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='EcoHeading1',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
            textColor=self.dark_color,
            fontName='Helvetica-Bold',
            leftIndent=10
        ))
        
        # Subsection style
        self.styles.add(ParagraphStyle(
            name='EcoHeading2',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=8,
            textColor=self.primary_color,
            fontName='Helvetica-Bold'
        ))
        
        # Body text with better spacing
        self.styles.add(ParagraphStyle(
            name='EcoBody',
            parent=self.styles['BodyText'],
            fontSize=10,
            spaceAfter=6,
            textColor=colors.black,
            fontName='Helvetica'
        ))
        
        # Highlight style for important numbers
        self.styles.add(ParagraphStyle(
            name='EcoHighlight',
            parent=self.styles['BodyText'],
            fontSize=11,
            textColor=self.primary_color,
            fontName='Helvetica-Bold',
            backColor=self.light_color,
            borderPadding=5,
            spaceAfter=8
        ))
        
        # Footer style
        self.styles.add(ParagraphStyle(
            name='EcoFooter',
            parent=self.styles['Italic'],
            fontSize=8,
            textColor=colors.gray,
            alignment=1  # Center
        ))
    
    def _create_header(self, story):
        """Create professional header with branding"""
        # Header with logo placeholder and title
        header_table_data = [
            ['ECO GUARDIAN', 'AI-Powered Waste Analysis']
        ]
        
        header_table = Table(header_table_data, colWidths=[3*inch, 3*inch])
        header_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, 0), self.primary_color),
            ('BACKGROUND', (1, 0), (1, 0), self.secondary_color),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (0, 0), 16),
            ('FONTSIZE', (1, 0), (1, 0), 12),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOX', (0, 0), (-1, -1), 1, colors.white),
            ('LINEBELOW', (0, 0), (-1, -1), 1, colors.white),
        ]))
        
        story.append(header_table)
        story.append(Spacer(1, 0.1*inch))
    
    def _create_cover_page(self, story, session_id):
        """Create a professional cover page"""
        # Title
        story.append(Paragraph("WASTE ANALYSIS REPORT", self.styles['EcoTitle']))
        story.append(Spacer(1, 0.3*inch))
        
        # Session info in a styled box
        info_data = [
            ['Session ID:', session_id],
            ['Generated:', datetime.now().strftime('%Y-%m-%d %H:%M')],
            ['Report Type:', 'Comprehensive Waste Analysis'],
            ['AI Model:', 'Gemini Vision + Multi-Agent System']
        ]
        
        info_table = Table(info_data, colWidths=[1.5*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), self.light_color),
            ('BACKGROUND', (1, 0), (1, -1), colors.white),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, self.primary_color),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        
        story.append(info_table)
        story.append(Spacer(1, 0.5*inch))
        
        # Mission statement
        mission_text = """
        <b>Our Mission:</b> To leverage artificial intelligence for smarter waste management, 
        promoting recycling efficiency and environmental sustainability through advanced 
        multi-agent systems and computer vision technology.
        """
        story.append(Paragraph(mission_text, self.styles['EcoBody']))
        
        # Page break for next section
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph("--- Report Contents ---", self.styles['EcoFooter']))
    
    def _create_executive_summary(self, story, report_data):
        """Create enhanced executive summary with visual elements"""
        story.append(Paragraph("Executive Summary", self.styles['EcoHeading1']))
        story.append(Spacer(1, 0.1*inch))
        
        summary = report_data.get('summary', {})
        environmental = report_data.get('environmental_impact', {})
        
        # Key metrics in a professional table
        metrics_data = [
            ['METRIC', 'VALUE', 'IMPACT'],
            ['Total Items Analyzed', str(summary.get('total_items', 0)), 'Analysis Scope'],
            ['Recycling Rate', f"{summary.get('recyclable_percent', 0):.1f}%", 'Efficiency Score'],
            ['COâ‚‚ Reduction', f"{environmental.get('co2_saved_kg', 0):.1f} kg", 'Environmental Impact'],
            ['Water Saved', f"{environmental.get('water_saved_liters', 0):.0f} L", 'Resource Conservation'],
            ['Energy Saved', f"{environmental.get('energy_saved_kwh', 0):.1f} kWh", 'Energy Efficiency']
        ]
        
        metrics_table = Table(metrics_data, colWidths=[1.8*inch, 1.5*inch, 2.2*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.primary_color),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, self.light_color]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        
        story.append(metrics_table)
        story.append(Spacer(1, 0.2*inch))
        
        # Performance indicator
        recyclable_rate = summary.get('recyclable_percent', 0)
        if recyclable_rate >= 70:
            performance = "Excellent"
            color = self.primary_color
        elif recyclable_rate >= 50:
            performance = "Good"
            color = self.accent_color
        else:
            performance = "Needs Improvement"
            color = colors.red
        
        performance_text = f"""
        <b>Performance Rating:</b> <font color="{color.toHex()}">{performance}</font><br/>
        <b>Overall Assessment:</b> Your waste stream shows a recycling efficiency of {recyclable_rate:.1f}%. 
        This analysis provides actionable insights to optimize your waste management practices.
        """
        story.append(Paragraph(performance_text, self.styles['EcoBody']))
        story.append(Spacer(1, 0.2*inch))
    
    def _create_waste_breakdown(self, story, report_data):
        """Create visual waste breakdown section"""
        story.append(Paragraph("Waste Composition Analysis", self.styles['EcoHeading1']))
        story.append(Spacer(1, 0.1*inch))
        
        summary = report_data.get('summary', {})
        
        # Waste distribution table
        distribution_data = [
            ['CATEGORY', 'PERCENTAGE', 'ITEMS'],
            ['Recyclable', f"{summary.get('recyclable_percent', 0):.1f}%", 'Plastic, Metal, Glass, Paper'],
            ['Compostable', f"{summary.get('compost_percent', 0):.1f}%", 'Food Waste, Organic Materials'],
            ['Landfill', f"{summary.get('landfill_percent', 0):.1f}%", 'Non-Recyclable Items'],
            ['Hazardous', f"{summary.get('hazardous_percent', 0):.1f}%", 'Special Handling Required']
        ]
        
        dist_table = Table(distribution_data, colWidths=[1.5*inch, 1.2*inch, 3*inch])
        dist_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.dark_color),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
            ('BACKGROUND', (0, 1), (0, 1), self.secondary_color),
            ('BACKGROUND', (0, 2), (0, 2), colors.orange),
            ('BACKGROUND', (0, 3), (0, 3), colors.lightgrey),
            ('BACKGROUND', (0, 4), (0, 4), colors.red),
            ('TEXTCOLOR', (0, 1), (0, 4), colors.white),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        
        story.append(dist_table)
        story.append(Spacer(1, 0.2*inch))
    
    def _create_detailed_analysis(self, story, report_data):
        """Create enhanced detailed analysis section"""
        story.append(Paragraph("Detailed Item Analysis", self.styles['EcoHeading1']))
        story.append(Spacer(1, 0.1*inch))
        
        breakdown = report_data.get('detailed_breakdown', [])
        
        if breakdown:
            # Create professional table for item analysis
            table_data = [['ITEM', 'CATEGORY', 'DISPOSAL GUIDANCE', 'CONFIDENCE']]
            
            for item in breakdown:
                table_data.append([
                    item.get('item', 'Unknown'),
                    item.get('category', 'Unknown').title(),
                    item.get('disposal_instructions', {}).get('instructions', 'N/A'),
                    f"{item.get('confidence', 0)*100:.1f}%"
                ])
            
            analysis_table = Table(table_data, colWidths=[1.2*inch, 1*inch, 2.2*inch, 0.8*inch])
            analysis_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), self.primary_color),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('ALIGN', (3, 1), (3, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, self.light_color]),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('PADDING', (0, 0), (-1, -1), 4),
            ]))
            
            story.append(analysis_table)
        else:
            story.append(Paragraph("No detailed item data available for this analysis.", self.styles['EcoBody']))
        
        story.append(Spacer(1, 0.2*inch))
    
    def _create_environmental_impact(self, story, report_data):
        """Create enhanced environmental impact section"""
        story.append(Paragraph("Environmental Impact Assessment", self.styles['EcoHeading1']))
        story.append(Spacer(1, 0.1*inch))
        
        impact = report_data.get('environmental_impact', {})
        
        impact_data = [
            ['ENVIRONMENTAL METRIC', 'AMOUNT SAVED', 'EQUIVALENT TO'],
            ['COâ‚‚ Emissions', f"{impact.get('co2_saved_kg', 0):.1f} kg", 'Driving 5 miles in a car'],
            ['Water Usage', f"{impact.get('water_saved_liters', 0):.0f} liters", '100 showers'],
            ['Energy Consumption', f"{impact.get('energy_saved_kwh', 0):.1f} kWh", 'Powering a home for 1 day']
        ]
        
        impact_table = Table(impact_data, colWidths=[2*inch, 1.5*inch, 2*inch])
        impact_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.accent_color),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        
        story.append(impact_table)
        story.append(Spacer(1, 0.2*inch))
        
        # Impact summary
        total_impact = sum([impact.get('co2_saved_kg', 0), impact.get('water_saved_liters', 0)/100, impact.get('energy_saved_kwh', 0)])
        impact_text = f"""
        <b>Overall Environmental Impact Score:</b> {total_impact:.1f} points<br/>
        Your recycling efforts are making a measurable difference in reducing environmental footprint 
        and promoting sustainable waste management practices.
        """
        story.append(Paragraph(impact_text, self.styles['EcoHighlight']))
        story.append(Spacer(1, 0.2*inch))
    
    def _create_recommendations(self, story, report_data):
        """Create enhanced recommendations section"""
        story.append(Paragraph("AI-Powered Recommendations", self.styles['EcoHeading1']))
        story.append(Spacer(1, 0.1*inch))
        
        tips = report_data.get('personalized_tips', [])
        
        if tips:
            for i, tip in enumerate(tips, 1):
                tip_text = f"<b>Recommendation {i}:</b> {tip}"
                story.append(Paragraph(tip_text, self.styles['EcoBody']))
                story.append(Spacer(1, 0.1*inch))
        else:
            story.append(Paragraph("No specific recommendations available for this analysis.", self.styles['EcoBody']))
        
        story.append(Spacer(1, 0.2*inch))
        
        # General best practices
        best_practices = [
            "âœ… Always rinse recyclable containers before disposal",
            "âœ… Separate materials according to local guidelines", 
            "âœ… Reduce single-use plastics when possible",
            "âœ… Compost food waste to minimize landfill impact",
            "âœ… Stay informed about local recycling program updates"
        ]
        
        story.append(Paragraph("<b>General Best Practices:</b>", self.styles['EcoHeading2']))
        for practice in best_practices:
            story.append(Paragraph(practice, self.styles['EcoBody']))
            story.append(Spacer(1, 0.05*inch))
    
    def _create_footer(self, story):
        """Create professional footer"""
        story.append(Spacer(1, 0.3*inch))
        
        footer_text = """
        <b>EcoGuardian AI System</b><br/>
        Multi-Agent Waste Detection & Recycling Optimization Platform<br/>
        Generated with Advanced Computer Vision and AI Analysis<br/>
        <i>Driving sustainability through intelligent waste management</i>
        """
        story.append(Paragraph(footer_text, self.styles['EcoFooter']))
    
    async def generate_ecoguardian_report(self, report_data: Dict[str, Any], session_id: str) -> bytes:
        """Generate enhanced professional PDF report"""
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
        story = []
        
        # Build enhanced PDF content
        self._create_header(story)
        self._create_cover_page(story, session_id)
        self._create_executive_summary(story, report_data)
        self._create_waste_breakdown(story, report_data)
        self._create_detailed_analysis(story, report_data) 
        self._create_environmental_impact(story, report_data)
        self._create_recommendations(story, report_data)
        self._create_footer(story)
        
        # Build PDF
        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        return pdf_bytes


%%writefile ecoguardian/utils/pdf_downloader.py
import base64
from IPython.display import HTML, display, IFrame
from typing import Optional
import tempfile
import os

class PDFDownloader:
    """Enhanced utility for PDF handling with preview and download functionality"""
    
    @staticmethod
    def create_download_link(pdf_bytes: bytes, filename: str, link_text: str = "Download PDF Report") -> HTML:
        """Create a downloadable link for PDF in Kaggle notebook"""
        try:
            b64 = base64.b64encode(pdf_bytes).decode()
            download_html = f'''
            <div style="padding: 15px; border: 2px solid #4CAF50; border-radius: 8px; background-color: #f9f9f9; margin: 10px 0;">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div>
                        <strong style="color: #2E7D32;">ğŸ“„ {filename}</strong><br>
                        <span style="color: #666; font-size: 12px;">Size: {len(pdf_bytes)/1024:.1f} KB</span>
                    </div>
                    <a href="data:application/pdf;base64,{b64}" download="{filename}" 
                       style="background-color: #4CAF50; color: white; padding: 10px 20px; 
                              text-decoration: none; border-radius: 5px; font-weight: bold;
                              transition: background-color 0.3s;">
                       {link_text}
                    </a>
                </div>
            </div>
            '''
            return HTML(download_html)
        except Exception as e:
            error_html = f'''
            <div style="color: red; padding: 10px; border: 1px solid red; border-radius: 5px;">
                â�Œ Error creating download link: {str(e)}
            </div>
            '''
            return HTML(error_html)
    
    @staticmethod
    def create_pdf_preview(pdf_bytes: bytes, filename: str, width: str = "100%", height: str = "600px") -> HTML:
        """Create an embedded PDF preview with download options"""
        try:
            b64 = base64.b64encode(pdf_bytes).decode()
            
            preview_html = f'''
            <div style="border: 2px solid #4CAF50; border-radius: 10px; padding: 15px; margin: 15px 0; background: white;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 1px solid #eee;">
                    <h3 style="color: #2E7D32; margin: 0;">ğŸ”� PDF Preview: {filename}</h3>
                    <div style="display: flex; gap: 10px;">
                        <a href="data:application/pdf;base64,{b64}" download="{filename}" 
                           style="background-color: #4CAF50; color: white; padding: 8px 16px; 
                                  text-decoration: none; border-radius: 4px; font-size: 14px;">
                           ğŸ“¥ Download
                        </a>
                    </div>
                </div>
                
                <div style="background: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                    <strong>File Info:</strong> {len(pdf_bytes)/1024:.1f} KB â€¢ Generated by EcoGuardian AI
                </div>
                
                <iframe src="data:application/pdf;base64,{b64}" 
                        width="{width}" 
                        height="{height}" 
                        style="border: 1px solid #ddd; border-radius: 5px;">
                </iframe>
                
                <div style="margin-top: 10px; text-align: center; color: #666; font-size: 12px;">
                    ğŸ’¡ Scroll to navigate â€¢ Use download button above to save
                </div>
            </div>
            '''
            return HTML(preview_html)
        except Exception as e:
            error_html = f'''
            <div style="color: red; padding: 15px; border: 1px solid red; border-radius: 5px; margin: 10px 0;">
                â�Œ Error creating PDF preview: {str(e)}
            </div>
            '''
            return HTML(error_html)
    
    @staticmethod
    def create_interactive_pdf_dashboard(pdf_bytes: bytes, filename: str) -> HTML:
        """Create an interactive PDF dashboard with preview and multiple download options"""
        try:
            b64 = base64.b64encode(pdf_bytes).decode()
            file_size_kb = len(pdf_bytes) / 1024
            
            dashboard_html = f'''
            <div style="border: 2px solid #2E7D32; border-radius: 12px; padding: 20px; margin: 20px 0; background: linear-gradient(135deg, #f8fff8, #e8f5e8);">
                <!-- Header -->
                <div style="text-align: center; margin-bottom: 20px;">
                    <h2 style="color: #2E7D32; margin: 0 0 5px 0;">ğŸŒ¿ EcoGuardian Report Ready</h2>
                    <p style="color: #666; margin: 0;">AI-Powered Waste Analysis Complete</p>
                </div>
                
                <!-- File Info Card -->
                <div style="background: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #4CAF50;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong style="color: #2E7D32;">ğŸ“Š Report Details</strong>
                            <div style="color: #666; font-size: 14px; margin-top: 5px;">
                                ğŸ“„ <strong>{filename}</strong><br>
                                ğŸ’¾ Size: {file_size_kb:.1f} KB<br>
                                ğŸ�·ï¸� Type: Professional Waste Analysis
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="background: #4CAF50; color: white; padding: 5px 10px; border-radius: 15px; font-size: 12px;">
                                âœ… READY
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Action Buttons -->
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 20px;">
                    <a href="data:application/pdf;base64,{b64}" download="{filename}" 
                       style="background: linear-gradient(135deg, #4CAF50, #2E7D32); color: white; padding: 12px; 
                              text-decoration: none; border-radius: 6px; text-align: center; font-weight: bold;
                              transition: transform 0.2s; display: block;">
                       ğŸ“¥ Download PDF
                    </a>
                    <button onclick="togglePreview()" 
                            style="background: linear-gradient(135deg, #FF9800, #F57C00); color: white; padding: 12px; 
                                   border: none; border-radius: 6px; text-align: center; font-weight: bold;
                                   cursor: pointer; transition: transform 0.2s;">
                       ğŸ‘�ï¸� Toggle Preview
                    </button>
                </div>
                
                <!-- Preview Section -->
                <div id="previewSection" style="display: none;">
                    <div style="background: white; padding: 15px; border-radius: 8px; border: 1px solid #ddd;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                            <strong style="color: #2E7D32;">ğŸ”� Live Preview</strong>
                            <span style="color: #666; font-size: 12px;">Scroll to navigate â€¢ Click to interact</span>
                        </div>
                        <iframe src="data:application/pdf;base64,{b64}#toolbar=1&navpanes=1" 
                                width="100%" 
                                height="500px" 
                                style="border: 1px solid #ddd; border-radius: 5px;">
                        </iframe>
                    </div>
                </div>
                
                <!-- Quick Actions -->
                <div style="background: #e8f5e8; padding: 15px; border-radius: 8px; margin-top: 15px;">
                    <strong style="color: #2E7D32;">ğŸš€ Quick Actions</strong>
                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; margin-top: 10px;">
                        <a href="data:application/pdf;base64,{b64}" download="{filename.replace('.pdf', '_compact.pdf')}" 
                           style="background: white; color: #2E7D32; padding: 8px; text-decoration: none; 
                                  border-radius: 4px; text-align: center; font-size: 12px; border: 1px solid #4CAF50;">
                           ğŸ’¾ Save Copy
                        </a>
                        <button onclick="window.print()" 
                                style="background: white; color: #2E7D32; padding: 8px; border: 1px solid #4CAF50; 
                                       border-radius: 4px; font-size: 12px; cursor: pointer;">
                           ğŸ–¨ï¸� Print
                        </button>
                        <button onclick="shareReport()" 
                                style="background: white; color: #2E7D32; padding: 8px; border: 1px solid #4CAF50; 
                                       border-radius: 4px; font-size: 12px; cursor: pointer;">
                           ğŸ“¤ Share
                        </button>
                    </div>
                </div>
            </div>
            
            <script>
            function togglePreview() {{
                var preview = document.getElementById('previewSection');
                if (preview.style.display === 'none') {{
                    preview.style.display = 'block';
                }} else {{
                    preview.style.display = 'none';
                }}
            }}
            
            function shareReport() {{
                alert('Share functionality: Download the PDF and share the file directly. For web apps, this would integrate with sharing APIs.');
            }}
            </script>
            
            <style>
            a:hover, button:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }}
            </style>
            '''
            return HTML(dashboard_html)
        except Exception as e:
            error_html = f'''
            <div style="color: red; padding: 20px; border: 1px solid red; border-radius: 8px; margin: 20px 0; text-align: center;">
                â�Œ Error creating PDF dashboard: {str(e)}
            </div>
            '''
            return HTML(error_html)
    
    @staticmethod
    def display_pdf_info(pdf_bytes: bytes, filename: str):
        """Display enhanced PDF file information"""
        size_kb = len(pdf_bytes) / 1024
        print(f"ğŸ“Š PDF Report Generated:")
        print(f"   ğŸ“„ Filename: {filename}")
        print(f"   ğŸ“¦ File size: {size_kb:.1f} KB")
        print(f"   ğŸ�·ï¸�  Pages: 1 (Professional Report)")
        print(f"   ğŸ”§ Features: Enhanced Styling, Branding, Analytics")
        print(f"   ğŸ’¾ Download methods:")
        print(f"      â€¢ Click download link below")
        print(f"      â€¢ Use preview interface")
        print(f"      â€¢ Access from Kaggle output tab")
    
    @staticmethod
    def save_pdf_to_output(pdf_bytes: bytes, filename: str) -> str:
        """Save PDF to Kaggle working directory for manual download"""
        try:
            filepath = f"/kaggle/working/{filename}"
            with open(filepath, 'wb') as f:
                f.write(pdf_bytes)
            print(f"ğŸ’¾ PDF saved to output directory: {filepath}")
            return filepath
        except Exception as e:
            raise Exception(f"Failed to save PDF: {str(e)}")
    
    @staticmethod
    def create_simple_preview(pdf_bytes: bytes, filename: str) -> HTML:
        """Create a simple PDF preview for quick viewing"""
        try:
            b64 = base64.b64encode(pdf_bytes).decode()
            
            simple_html = f'''
            <div style="border: 1px solid #ccc; border-radius: 5px; padding: 10px; margin: 10px 0;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <strong>Preview: {filename}</strong>
                    <a href="data:application/pdf;base64,{b64}" download="{filename}" 
                       style="background: #4CAF50; color: white; padding: 5px 10px; text-decoration: none; border-radius: 3px; font-size: 12px;">
                       Download
                    </a>
                </div>
                <iframe src="data:application/pdf;base64,{b64}" 
                        width="100%" 
                        height="400px" 
                        style="border: 1px solid #ddd;">
                </iframe>
            </div>
            '''
            return HTML(simple_html)
        except Exception as e:
            return HTML(f'<div style="color: red;">Preview error: {str(e)}</div>')


%%writefile ecoguardian/memory/memory_bank.py
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

class MemoryBank:
    """Long-term memory storage for user profiles and analysis history"""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path
        self.user_profiles = {}
        self.analysis_history = {}
        
    async def store_analysis(self, user_id: str, session_id: str, results: Dict[str, Any]):
        if user_id not in self.analysis_history:
            self.analysis_history[user_id] = []
        
        record = {
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "results": results,
            "summary": self._extract_summary(results)
        }
        
        self.analysis_history[user_id].append(record)
        await self._update_user_profile(user_id, record)
        
    async def get_user_history(self, user_id: str, days: int = 30) -> List[Dict[str, Any]]:
        if user_id not in self.analysis_history:
            return []
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        history = self.analysis_history[user_id]
        
        return [
            record for record in history
            if datetime.fromisoformat(record['timestamp'].replace('Z', '+00:00')) > cutoff_date
        ]
    
    async def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        if user_id not in self.user_profiles:
            return self._create_default_profile(user_id)
        return self.user_profiles[user_id]
    
    async def _update_user_profile(self, user_id: str, new_record: Dict[str, Any]):
        profile = await self.get_user_profile(user_id)
        summary = new_record['summary']
        
        profile['total_analyses'] += 1
        profile['total_items_analyzed'] += summary['total_items']
        
        for category in ['recyclable', 'compost', 'landfill', 'hazardous']:
            current_avg = profile['category_percentages'].get(category, 0)
            new_value = summary.get(f'{category}_percent', 0)
            profile['category_percentages'][category] = (
                (current_avg * (profile['total_analyses'] - 1) + new_value) / 
                profile['total_analyses']
            )
        
        profile['last_analysis'] = new_record['timestamp']
        self.user_profiles[user_id] = profile
    
    def _extract_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        classified_items = results.get('classified_items', [])
        stats = results.get('summary_stats', {})
        
        return {
            'total_items': len(classified_items),
            'recyclable_percent': stats.get('recyclable_percent', 0),
            'compost_percent': stats.get('compost_percent', 0),
            'landfill_percent': stats.get('landfill_percent', 0),
            'hazardous_percent': stats.get('hazardous_percent', 0)
        }
    
    def _create_default_profile(self, user_id: str) -> Dict[str, Any]:
        profile = {
            'user_id': user_id,
            'total_analyses': 0,
            'total_items_analyzed': 0,
            'category_percentages': {
                'recyclable': 0,
                'compost': 0,
                'landfill': 0,
                'hazardous': 0
            },
            'created_date': datetime.utcnow().isoformat(),
            'last_analysis': None,
        }
        
        self.user_profiles[user_id] = profile
        return profile


%%writefile ecoguardian/agents/vision_agent.py
import base64
import json
from typing import List, Dict, Any
from ecoguardian.agents.agent_base import AgentContext, create_agent_logger, log_metric

class VisionAnalysisAgent:
    """Agent for analyzing waste items in images using vision AI"""
    
    def __init__(self, vision_provider: Any, use_gemini: bool = False):
        self.name = "vision_analysis"
        self.logger = create_agent_logger(self.name)
        self.vision_provider = vision_provider
        self.use_gemini = use_gemini
        
        if use_gemini:
            print("ğŸ”® Vision Agent configured to use Gemini (if API key available)")
    
    async def execute(self, context: AgentContext, image_data: bytes) -> Dict[str, Any]:
        self.logger.info(f"Analyzing image for session {context.session_id}")
        
        try:
            # Convert image to base64 for API consumption
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            if self.use_gemini:
                print("ğŸ�¯ Attempting Gemini Vision analysis...")
            
            # Call vision provider (Gemini or Mock)
            detections = await self.vision_provider.analyze_image(image_b64)
            
            log_metric(self.logger, "images_processed", 1)
            log_metric(self.logger, "items_detected", len(detections.get('items', [])))
            
            # Enhanced logging
            provider = detections.get('provider', 'mock')
            model_version = detections.get('model_version', 'unknown')
            print(f"âœ… Vision analysis complete - Provider: {provider}, Model: {model_version}")
            
            return {
                "session_id": context.session_id,
                "detections": detections,
                "agent": self.name,
                "provider": provider
            }
            
        except Exception as e:
            self.logger.error(f"Vision analysis failed: {str(e)}")
            raise


%%writefile ecoguardian/agents/classification_agent.py
from typing import Dict, Any, List
from ecoguardian.agents.agent_base import AgentContext, create_agent_logger, log_metric

class WasteClassificationAgent:
    """Agent for classifying waste items and determining disposal methods"""
    
    def __init__(self, waste_db_tool: Any, location_tool: Any):
        self.name = "waste_classification"
        self.logger = create_agent_logger(self.name)
        self.waste_db = waste_db_tool
        self.location_tool = location_tool
    
    async def execute(self, context: AgentContext, vision_results: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info(f"Classifying items for session {context.session_id}")
        
        classified_items = []
        
        # Process each detected item
        for item in vision_results['detections'].get('items', []):
            # Classify using waste database
            classification = await self.waste_db.classify_item(
                item['name'], 
                context.location
            )
            
            # Get location-specific disposal instructions
            disposal_info = await self.location_tool.get_disposal_instructions(
                classification['category'],
                context.location
            )
            
            classified_item = {
                **item,
                "classification": classification,
                "disposal_instructions": disposal_info,
                "recycling_label": self._generate_recycling_label(classification)
            }
            classified_items.append(classified_item)
        
        # Calculate summary statistics
        stats = self._calculate_classification_stats(classified_items)
        
        log_metric(self.logger, "items_classified", len(classified_items))
        log_metric(self.logger, "recyclable_percent", stats['recyclable_percent'])
        
        return {
            "session_id": context.session_id,
            "classified_items": classified_items,
            "summary_stats": stats,
            "agent": self.name
        }
    
    def _generate_recycling_label(self, classification: Dict[str, Any]) -> str:
        category = classification['category']
        if category == 'recyclable':
            return f"RECYCLE_{classification.get('material', 'MIXED')}"
        elif category == 'compost':
            return "COMPOST"
        elif category == 'hazardous':
            return "HAZARDOUS"
        else:
            return "LANDFILL"
    
    def _calculate_classification_stats(self, items: List[Dict]) -> Dict[str, float]:
        categories = [item['classification']['category'] for item in items]
        total = len(categories)
        
        if total == 0:
            return {'recyclable_percent': 0, 'compost_percent': 0, 
                   'landfill_percent': 0, 'hazardous_percent': 0}
        
        return {
            'recyclable_percent': (categories.count('recyclable') / total) * 100,
            'compost_percent': (categories.count('compost') / total) * 100,
            'landfill_percent': (categories.count('landfill') / total) * 100,
            'hazardous_percent': (categories.count('hazardous') / total) * 100
        }


%%writefile ecoguardian/agents/reporting_agent.py
from typing import Dict, Any, List
from datetime import datetime
from ecoguardian.agents.agent_base import AgentContext, create_agent_logger, log_metric
from ecoguardian.tools.pdf_generator import PDFGenerator
from ecoguardian.utils.pdf_downloader import PDFDownloader

class ReportingEducationAgent:
    """Agent for generating reports and educational content with PDF support"""
    
    def __init__(self, memory_bank: Any):
        self.name = "reporting_education"
        self.logger = create_agent_logger(self.name)
        self.memory_bank = memory_bank
        self.pdf_generator = PDFGenerator()
    
    async def execute(self, context: AgentContext, classification_results: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info(f"Generating report for session {context.session_id}")
        
        # Store results in memory for historical tracking
        await self.memory_bank.store_analysis(
            context.user_id,
            context.session_id,
            classification_results
        )
        
        # Get user history for personalized insights
        user_history = await self.memory_bank.get_user_history(context.user_id)
        
        # Generate standard report
        report = await self._generate_comprehensive_report(
            classification_results,
            user_history,
            context
        )
        
        # Generate PDF report
        pdf_bytes = await self.generate_pdf_report(report, context.session_id)
        
        # Save PDF to file system
        filename = f"ecoguardian_report_{context.session_id}.pdf"
        saved_path = PDFDownloader.save_pdf_to_output(pdf_bytes, filename)
        print(f"ğŸ’¾ PDF saved to: {saved_path}")
        
        # Add PDF data to report
        report['pdf_bytes'] = pdf_bytes
        report['pdf_filename'] = filename
        report['saved_path'] = saved_path
        
        log_metric(self.logger, "reports_generated", 1)
        log_metric(self.logger, "environmental_savings_kg", report['environmental_impact']['co2_saved_kg'])
        
        return report
    
    async def generate_pdf_report(self, report_data: Dict[str, Any], session_id: str) -> bytes:
        """Generate PDF version of the report"""
        self.logger.info(f"Generating PDF report for session {session_id}")
        
        try:
            pdf_bytes = await self.pdf_generator.generate_ecoguardian_report(report_data, session_id)
            log_metric(self.logger, "pdf_reports_generated", 1)
            return pdf_bytes
        except Exception as e:
            self.logger.error(f"PDF generation failed: {str(e)}")
            raise
    
    async def _generate_comprehensive_report(self, classification_results: Dict, user_history: List[Dict], context: AgentContext) -> Dict[str, Any]:
        stats = classification_results['summary_stats']
        items = classification_results['classified_items']
        
        # Calculate environmental impact
        environmental_impact = self._calculate_environmental_impact(items)
        
        # Generate personalized tips based on current stats and history
        personalized_tips = self._generate_personalized_tips(stats, user_history)
        
        return {
            "session_id": context.session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "summary": {
                "total_items": len(items),
                "recyclable_percent": stats['recyclable_percent'],
                "compost_percent": stats['compost_percent'],
                "landfill_percent": stats['landfill_percent']
            },
            "environmental_impact": environmental_impact,
            "personalized_tips": personalized_tips,
            "detailed_breakdown": [
                {
                    "item": item['name'],
                    "category": item['classification']['category'],
                    "disposal_instructions": item['disposal_instructions'],
                    "recycling_label": item.get('recycling_label')
                }
                for item in items
            ],
            "agent": self.name
        }
    
    def _calculate_environmental_impact(self, items: List[Dict]) -> Dict[str, float]:
        recyclable_count = sum(1 for item in items if item['classification']['category'] == 'recyclable')
        
        return {
            'co2_saved_kg': recyclable_count * 0.5,
            'water_saved_liters': recyclable_count * 10,
            'energy_saved_kwh': recyclable_count * 0.3
        }
    
    def _generate_personalized_tips(self, current_stats: Dict, user_history: List[Dict]) -> List[str]:
        tips = []
        
        if current_stats['recyclable_percent'] < 50:
            tips.append("Try to separate more recyclable materials like plastic, paper, and metal.")
        
        if current_stats['compost_percent'] < 20:
            tips.append("Consider composting food waste to reduce landfill usage.")
        
        # Add general educational tips
        tips.extend([
            "Rinse recyclable containers to reduce contamination.",
            "Check local guidelines for specific recycling rules.",
            "Reduce single-use plastics by choosing reusable alternatives."
        ])
        
        return tips


%%writefile ecoguardian/orchestration/orchestrator.py
import asyncio
from typing import Dict, Any, List
from datetime import datetime
import uuid

from ecoguardian.agents.vision_agent import VisionAnalysisAgent
from ecoguardian.agents.classification_agent import WasteClassificationAgent
from ecoguardian.agents.reporting_agent import ReportingEducationAgent
from ecoguardian.agents.agent_base import AgentContext

class EcoGuardianOrchestrator:
    """Orchestrates the multi-agent workflow for waste analysis"""
    
    def __init__(self, vision_provider, waste_db, location_finder, memory_bank, use_gemini_vision: bool = False):
        self.vision_agent = VisionAnalysisAgent(vision_provider, use_gemini=use_gemini_vision)
        self.classification_agent = WasteClassificationAgent(waste_db, location_finder)
        self.reporting_agent = ReportingEducationAgent(memory_bank)
        
    async def process_single_image(self, image_data: bytes, user_id: str = None, 
                                 location: str = None) -> Dict[str, Any]:
        """Sequential workflow: Vision â†’ Classification â†’ Reporting"""
        
        session_id = str(uuid.uuid4())
        context = AgentContext(
            session_id=session_id,
            user_id=user_id or "anonymous",
            location=location,
            metadata={"timestamp": datetime.utcnow().isoformat()}
        )
        
        try:
            print("ğŸ”� Starting Vision Analysis...")
            vision_results = await self.vision_agent.execute(context, image_data)
            print(f"   Detected {len(vision_results['detections'].get('items', []))} items")
            
            print("ğŸ�·ï¸� Starting Waste Classification...")
            classification_results = await self.classification_agent.execute(context, vision_results)
            print(f"   Classified {len(classification_results['classified_items'])} items")
            
            print("ğŸ“Š Generating Report & PDF...")
            final_report = await self.reporting_agent.execute(context, classification_results)
            print("   PDF report generated successfully")
            
            return {
                "success": True,
                "session_id": session_id,
                "report": final_report,
            }
            
        except Exception as e:
            print(f"â�Œ Pipeline error: {str(e)}")
            return {
                "success": False,
                "session_id": session_id,
                "error": str(e),
            }
    
    async def process_multiple_images(self, images_data: List[bytes], user_id: str = None,
                                    location: str = None) -> Dict[str, Any]:
        """Parallel image processing workflow"""
        
        session_id = str(uuid.uuid4())
        tasks = []
        
        print(f"ğŸ”„ Processing {len(images_data)} images in parallel...")
        
        for i, image_data in enumerate(images_data):
            task_context = AgentContext(
                session_id=f"{session_id}_{i}",
                user_id=user_id,
                location=location,
                metadata={"image_index": i}
            )
            
            task = asyncio.create_task(
                self._process_single_image_parallel(image_data, task_context)
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        successful_results = [r for r in results if not isinstance(r, Exception)]
        
        return {
            "session_id": session_id,
            "total_images": len(images_data),
            "successful_processing": len(successful_results),
            "results": successful_results
        }
    
    async def _process_single_image_parallel(self, image_data: bytes, context: AgentContext) -> Dict[str, Any]:
        """Process single image in parallel workflow"""
        vision_results = await self.vision_agent.execute(context, image_data)
        classification_results = await self.classification_agent.execute(context, vision_results)
        return classification_results


# Install required packages for EcoGuardian
print("ğŸ“¦ Installing EcoGuardian dependencies...")

!pip install reportlab
!pip install google-genai

print("âœ… Dependencies installed successfully!")
print("ğŸ“š Required packages: reportlab, asyncio, base64, typing")


# Robust TrashNet Image Loader for Correct Dataset Structure
import os
from PIL import Image
import random

def load_trashnet_sample(category=None):
    """Load a random sample image from TrashNet dataset"""
    
    # Try multiple possible dataset locations (correct TrashNet structure)
    possible_paths = [
        "ecoguardian/data/trashnet/data/dataset",  # Copied structure
        "ecoguardian/data/trashnet/dataset",       # Alternative copy
        "ecoguardian/data/trashnet",               # Direct copy
        "/kaggle/input/trashnet/data/dataset",     # Original location
        "/kaggle/input/trashnet/data",             # Alternative original
        "/kaggle/input/trashnet/dataset",          # Another alternative
        "/kaggle/input/trashnet",                  # Root location
        "/kaggle/input/garbage-classification/Garbage classification/Garbage classification",  # Fallback
    ]
    
    base_path = None
    for path in possible_paths:
        if os.path.exists(path):
            base_path = path
            print(f"ğŸ“� Using dataset from: {path}")
            break
    
    if not base_path:
        print("â�Œ TrashNet dataset not found in any location")
        print("ğŸ’¡ Using mock data for demonstration")
        return None
    
    # TrashNet categories (correct order from original dataset)
    categories = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    
    # If no category specified, pick random one
    if category is None:
        category = random.choice(categories)
    elif category not in categories:
        print(f"âš ï¸� Category '{category}' not found. Available: {categories}")
        category = random.choice(categories)
    
    # Try to find the category directory in various possible structures
    category_paths_to_try = [
        os.path.join(base_path, category),
        os.path.join(base_path, 'data', 'dataset', category),
        os.path.join(base_path, 'dataset', category),
    ]
    
    category_path = None
    for test_path in category_paths_to_try:
        if os.path.exists(test_path):
            category_path = test_path
            break
    
    if not category_path:
        print(f"â�Œ Could not find category '{category}' in dataset")
        print(f"ğŸ’¡ Available directories in {base_path}:")
        try:
            items = os.listdir(base_path)
            dirs = [item for item in items if os.path.isdir(os.path.join(base_path, item))]
            print(f"   {dirs}")
        except:
            print("   Could not list directory contents")
        return None
    
    # Get all image files in category
    try:
        image_files = [f for f in os.listdir(category_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    except FileNotFoundError:
        print(f"â�Œ Category directory not found: {category_path}")
        return None
    
    if not image_files:
        print(f"â�Œ No images found in {category} category at {category_path}")
        return None
    
    # Select random image
    selected_image = random.choice(image_files)
    image_path = os.path.join(category_path, selected_image)
    
    try:
        # Load and convert to bytes (matching VisionAgent input format)
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize if too large (for efficiency)
            if img.size[0] > 800 or img.size[1] > 800:
                img.thumbnail((800, 800))
            
            # Convert to bytes
            import io
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG', quality=85)
            image_data = img_bytes.getvalue()
        
        print(f"ğŸ“¸ Loaded TrashNet image: {category}/{selected_image}")
        print(f"ğŸ“� Image size: {img.size}, Format: JPEG")
        
        return image_data, category, selected_image
        
    except Exception as e:
        print(f"â�Œ Error loading image {image_path}: {e}")
        return None

def list_trashnet_categories():
    """List all available categories in TrashNet dataset"""
    possible_paths = [
        "ecoguardian/data/trashnet/data/dataset",
        "ecoguardian/data/trashnet/dataset", 
        "ecoguardian/data/trashnet",
        "/kaggle/input/trashnet/data/dataset",
        "/kaggle/input/trashnet/data",
        "/kaggle/input/trashnet/dataset",
        "/kaggle/input/trashnet",
    ]
    
    categories = []
    base_path = None
    
    for path in possible_paths:
        if os.path.exists(path):
            base_path = path
            break
    
    if base_path:
        print(f"ğŸ“� Dataset base path: {base_path}")
        try:
            # Try to list directories at base path
            items = os.listdir(base_path)
            potential_categories = [item for item in items 
                                  if os.path.isdir(os.path.join(base_path, item))]
            
            # Filter for actual TrashNet categories
            trashnet_categories = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
            categories = [cat for cat in potential_categories if cat in trashnet_categories]
            
            if not categories:
                # Maybe we're at a higher level, check subdirectories
                for item in items:
                    sub_path = os.path.join(base_path, item)
                    if os.path.isdir(sub_path):
                        sub_items = os.listdir(sub_path)
                        for sub_item in sub_items:
                            if sub_item in trashnet_categories and sub_item not in categories:
                                categories.append(sub_item)
            
            print("ğŸ“‹ Available TrashNet categories:")
            for cat in sorted(categories):
                # Find the actual path to count images
                for test_base in possible_paths:
                    test_paths = [
                        os.path.join(test_base, cat),
                        os.path.join(test_base, 'data', 'dataset', cat),
                        os.path.join(test_base, 'dataset', cat),
                    ]
                    for test_path in test_paths:
                        if os.path.exists(test_path):
                            try:
                                image_count = len([f for f in os.listdir(test_path) 
                                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                                print(f"   â€¢ {cat}: {image_count} images")
                                break
                            except:
                                print(f"   â€¢ {cat}: found but cannot access")
                            break
                    else:
                        continue
                    break
                else:
                    print(f"   â€¢ {cat}: directory not found")
                    
        except Exception as e:
            print(f"â�Œ Error reading dataset structure: {e}")
    else:
        print("â�Œ TrashNet dataset not found")
        print("ğŸ’¡ Please add: 'trashnet' by yangyang111 via '+ Add Data'")
        categories = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']  # Default list
    
    return categories

print("ğŸ”„ TrashNet utilities loaded with correct dataset structure!")


# EcoGuardian System Initialization
print("ğŸš€ Initializing EcoGuardian Multi-Agent System...")

import sys
import os

# Ensure ecoguardian is in path
sys.path.append('/kaggle/working/ecoguardian')

# Import all components
from ecoguardian.tools.vision_provider import MockVisionProvider, GeminiVisionProvider
from ecoguardian.tools.waste_db import WasteDB
from ecoguardian.tools.location_finder import LocationFinder
from ecoguardian.tools.pdf_generator import PDFGenerator
from ecoguardian.memory.memory_bank import MemoryBank
from ecoguardian.orchestration.orchestrator import EcoGuardianOrchestrator
from ecoguardian.utils.pdf_downloader import PDFDownloader

# Initialize components with error handling
try:
    # Try Gemini first if API key is available
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if gemini_api_key and gemini_api_key != "MOCK_MODE_NO_VALID_KEY" and gemini_api_key.startswith("AIza"):
        print("ğŸ”® Using REAL Gemini Vision Provider")
        vision_provider = GeminiVisionProvider()
        use_gemini = True
    else:
        print("ğŸ�­ Using Mock Vision Provider (no valid Gemini API key)")
        vision_provider = MockVisionProvider()
        use_gemini = False
except Exception as e:
    print(f"ğŸ�­ Using Mock Vision Provider (Gemini initialization failed: {e})")
    vision_provider = MockVisionProvider()
    use_gemini = False

# Initialize other components
waste_db = WasteDB()
location_finder = LocationFinder()
memory_bank = MemoryBank()

# Create orchestrator
orchestrator = EcoGuardianOrchestrator(
    vision_provider=vision_provider,
    waste_db=waste_db,
    location_finder=location_finder,
    memory_bank=memory_bank,
    use_gemini_vision=use_gemini
)

print("âœ… EcoGuardian system initialized successfully!")
print("ğŸ“‹ Components ready:")
print("   - Vision Analysis Agent")
print("   - Waste Classification Agent") 
print("   - Reporting & Education Agent (with PDF)")
print("   - Memory Bank")
print("   - PDF Generator")
print(f"   - Vision Provider: {'Gemini' if use_gemini else 'Mock'}")
print("   - TrashNet dataset integrated")

# Display Gemini status
if use_gemini:
    print("ğŸ�‰ Real AI detection: ENABLED")
else:
    print("ğŸ’¡ Real AI detection: DISABLED - Add Gemini API key to Kaggle Secrets")


# =============================================================================
# ğŸ§ª TEST CELL: Dataset Verification (FINAL FIX)
# =============================================================================

print("ğŸ§ª Verifying TrashNet Dataset Access...")
print("1. Checking available categories...")

import os
import glob

def find_actual_categories():
    """Find the actual waste categories with images"""
    
    # The dataset path
    base_path = '/kaggle/input/trashnet/trashnet'
    
    if not os.path.exists(base_path):
        print(f"â�Œ Dataset path not found: {base_path}")
        return None, []
    
    print(f"ğŸ“� Dataset path: {base_path}")
    
    # List ALL items in the base path
    all_items = os.listdir(base_path)
    print(f"ğŸ“¦ Main folders: {all_items}")
    
    # Check each subfolder (train, val, test) for categories
    found_categories = {}
    
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(base_path, split)
        if os.path.exists(split_path):
            print(f"\nğŸ”� Checking {split} folder:")
            
            # List categories in this split
            categories_in_split = os.listdir(split_path)
            print(f"   Categories in {split}: {categories_in_split}")
            
            for category in categories_in_split:
                category_path = os.path.join(split_path, category)
                if os.path.isdir(category_path):
                    # Count images in this category
                    image_files = glob.glob(os.path.join(category_path, "*.jpg")) + \
                                 glob.glob(os.path.join(category_path, "*.jpeg")) + \
                                 glob.glob(os.path.join(category_path, "*.png"))
                    
                    if image_files:
                        if category not in found_categories:
                            found_categories[category] = {'total': 0, 'train': 0, 'val': 0, 'test': 0}
                        found_categories[category][split] = len(image_files)
                        found_categories[category]['total'] += len(image_files)
                        print(f"   âœ… {category}: {len(image_files)} images")
    
    return base_path, found_categories

# Find the actual categories
dataset_path, categories_dict = find_actual_categories()

if categories_dict:
    print(f"\nğŸ�‰ SUCCESS: Found TrashNet dataset with categories!")
    print("ğŸ“‹ Available TrashNet categories (with split counts):")
    
    for category in sorted(categories_dict.keys()):
        counts = categories_dict[category]
        print(f"   âœ… {category}:")
        print(f"      ğŸ“Š Total: {counts['total']} images")
        if counts['train'] > 0:
            print(f"      ğŸ�‹ï¸�â€�â™‚ï¸� Train: {counts['train']} images")
        if counts['val'] > 0:
            print(f"      ğŸ“ˆ Val: {counts['val']} images")
        if counts['test'] > 0:
            print(f"      ğŸ§ª Test: {counts['test']} images")
    
    # Calculate totals
    total_images = sum(cat_info['total'] for cat_info in categories_dict.values())
    total_train = sum(cat_info['train'] for cat_info in categories_dict.values())
    total_val = sum(cat_info['val'] for cat_info in categories_dict.values())
    total_test = sum(cat_info['test'] for cat_info in categories_dict.values())
    
    print(f"\nğŸ“Š DATASET SUMMARY:")
    print(f"   ğŸ�¯ Categories: {len(categories_dict)}")
    print(f"   ğŸ–¼ï¸�  Total images: {total_images}")
    print(f"   ğŸ�‹ï¸�â€�â™‚ï¸� Training images: {total_train}")
    print(f"   ğŸ“ˆ Validation images: {total_val}")
    print(f"   ğŸ§ª Test images: {total_test}")
    
else:
    print("\nâ�Œ Could not find any categories!")
    if dataset_path:
        print(f"ğŸ”� Debugging structure of: {dataset_path}")
        print(f"\nğŸ“‚ Complete contents:")
        for item in os.listdir(dataset_path):
            item_path = os.path.join(dataset_path, item)
            if os.path.isdir(item_path):
                print(f"\n   ğŸ“� {item}/:")
                subitems = os.listdir(item_path)
                for subitem in subitems:
                    subitem_path = os.path.join(item_path, subitem)
                    if os.path.isdir(subitem_path):
                        image_files = glob.glob(os.path.join(subitem_path, "*.jpg")) + \
                                     glob.glob(os.path.join(subitem_path, "*.jpeg")) + \
                                     glob.glob(os.path.join(subitem_path, "*.png"))
                        print(f"      ğŸ“� {subitem}/: {len(image_files)} images")
                        if len(image_files) > 0:
                            for img in image_files[:2]:  # Show first 2 images
                                print(f"         ğŸ–¼ï¸�  {os.path.basename(img)}")

print("\n" + "="*60)
print("âœ… Dataset verification completed!")


# =============================================================================
# ğŸ§ª SINGLE TRASHNET IMAGE TEST 
# =============================================================================

print("ğŸ§ª Testing EcoGuardian with Real TrashNet Waste Images...")

import os
import random
import glob
from PIL import Image
import io
import reportlab.lib.colors as colors

# COMPREHENSIVE FIX FOR COLOR ISSUE
def fix_color_issues():
    """Fix all color-related issues in PDF generation"""
    try:
        # Fix 1: Add proper toHex method that handles float values
        if not hasattr(colors.Color, 'toHex'):
            def color_toHex(self):
                # Convert float RGB values (0-1) to integer (0-255)
                r = int(self.red * 255)
                g = int(self.green * 255) 
                b = int(self.blue * 255)
                return "#%02x%02x%02x" % (r, g, b)
            colors.Color.toHex = color_toHex
            print("âœ… Fixed Color.toHex with float-to-int conversion")
        
        # Fix 2: Also patch any other color conversion issues
        import reportlab.lib.utils as utils
        original_getBytes = getattr(utils, 'getBytes', None)
        
        if original_getBytes:
            def safe_getBytes(s):
                try:
                    return original_getBytes(s)
                except:
                    return str(s).encode('utf-8')
            utils.getBytes = safe_getBytes
            print("âœ… Patched getBytes for safe string conversion")
            
    except Exception as e:
        print(f"âš ï¸�  Color fix warning: {e}")

# Apply the color fixes
fix_color_issues()

def get_trashnet_categories():
    """Get available waste categories with their image counts"""
    base_path = '/kaggle/input/trashnet/trashnet'
    
    if not os.path.exists(base_path):
        print("â�Œ Could not find TrashNet dataset")
        return None, {}
    
    categories = set()
    
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(base_path, split)
        if os.path.exists(split_path):
            categories.update([d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))])
    
    category_counts = {}
    for category in categories:
        total_images = 0
        for split in ['train', 'val', 'test']:
            split_path = os.path.join(base_path, split, category)
            if os.path.exists(split_path):
                image_files = glob.glob(os.path.join(split_path, "*.jpg")) + \
                             glob.glob(os.path.join(split_path, "*.jpeg")) + \
                             glob.glob(os.path.join(split_path, "*.png"))
                total_images += len(image_files)
        category_counts[category] = total_images
    
    return base_path, category_counts

def get_random_image(category=None):
    """Get a random image from the dataset and convert to bytes"""
    base_path = '/kaggle/input/trashnet/trashnet'
    
    _, categories_dict = get_trashnet_categories()
    
    if not categories_dict:
        return None, None, None
    
    if category and category in categories_dict:
        selected_category = category
    else:
        selected_category = random.choice(list(categories_dict.keys()))
    
    for split in ['train', 'val', 'test']:
        category_path = os.path.join(base_path, split, selected_category)
        if os.path.exists(category_path):
            image_files = glob.glob(os.path.join(category_path, "*.jpg")) + \
                         glob.glob(os.path.join(category_path, "*.jpeg")) + \
                         glob.glob(os.path.join(category_path, "*.png"))
            if image_files:
                selected_image_path = random.choice(image_files)
                
                with Image.open(selected_image_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    img_bytes = io.BytesIO()
                    img.save(img_bytes, format='JPEG')
                    image_data = img_bytes.getvalue()
                
                return image_data, selected_category, selected_image_path
    
    return None, selected_category, None

# Get available categories
dataset_path, categories_dict = get_trashnet_categories()

if categories_dict:
    print("ğŸ�‰ TrashNet Dataset Found!")
    print("ğŸ“‹ Available TrashNet categories:")
    for category in sorted(categories_dict.keys()):
        print(f"   âœ… {category}: {categories_dict[category]} images")
    
    total_images = sum(categories_dict.values())
    print(f"ğŸ“Š TOTAL: {total_images} images")
else:
    print("â�Œ No waste categories found!")

async def test_trashnet_pipeline(category=None, use_mock=True, skip_pdf=False):
    """Test the pipeline with a random image from TrashNet"""
    print(f"\nğŸš€ Starting pipeline test with TrashNet image...")
    
    image_data, selected_category, image_path = get_random_image(category)
    
    if not image_data:
        print("â�Œ Could not load any images!")
        return
    
    print(f"ğŸ�² Selected category: {selected_category}")
    print(f"ğŸ–¼ï¸�  Selected image: {os.path.basename(image_path)}")
    
    try:
        # Force mock mode to avoid API issues
        original_use_gemini = getattr(orchestrator, 'use_gemini', True)
        
        if use_mock:
            orchestrator.use_gemini = False
            print("ğŸ”§ Using mock mode to avoid API issues")
        
        # Skip PDF generation if requested
        original_generate_pdf = getattr(orchestrator, 'generate_pdf', True)
        if skip_pdf:
            orchestrator.generate_pdf = False
            print("ğŸ”§ Skipping PDF generation to avoid color issues")
        
        print("ğŸ”� Processing image through EcoGuardian pipeline...")
        
        # Call the orchestrator's process_single_image method
        result = await orchestrator.process_single_image(
            image_data=image_data,
            user_id=f"trashnet_test_{selected_category}",
            location="NYC"
        )
        
        # Restore original settings
        orchestrator.use_gemini = original_use_gemini
        if skip_pdf:
            orchestrator.generate_pdf = original_generate_pdf
        
        if result["success"]:
            print(f"âœ… Pipeline test successful!")
            report = result["report"]
            
            # Display results
            print(f"\nğŸ“Š ANALYSIS RESULTS:")
            print(f"   Items detected: {report['summary']['total_items']}")
            print(f"   Recyclable: {report['summary']['recyclable_percent']:.1f}%")
            print(f"   Compost: {report['summary']['compost_percent']:.1f}%")
            print(f"   Landfill: {report['summary']['landfill_percent']:.1f}%")
            
            if report['detailed_breakdown']:
                print(f"\nğŸ”� DETECTED ITEMS:")
                for item in report['detailed_breakdown']:
                    print(f"   â€¢ {item['item']} â†’ {item['category']}")
            
            print(f"\nğŸ’¡ PERSONALIZED TIPS:")
            for tip in report['personalized_tips'][:3]:
                print(f"   â€¢ {tip}")
                
            # Show PDF info if available
            if 'pdf_bytes' in report:
                pdf_size = len(report['pdf_bytes']) / 1024
                print(f"\nğŸ“„ PDF Report: {report['pdf_filename']} ({pdf_size:.1f} KB)")
            else:
                print(f"\nğŸ“„ PDF: Not generated (skipped)")
                
            return result
        else:
            print(f"â�Œ Pipeline failed: {result.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"â�Œ Pipeline test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Run tests with different options
print("\n" + "="*60)
if categories_dict:
    print("ğŸ’¡ Available test options:")
    print("   1. await test_trashnet_pipeline()                    # Basic test")
    print("   2. await test_trashnet_pipeline('plastic')           # Specific category") 
    print("   3. await test_trashnet_pipeline(skip_pdf=True)       # Skip PDF generation")
    
    # Test 1: Basic test with PDF
    print(f"\nğŸ�¯ Test 1: Basic test with PDF generation...")
    result1 = await test_trashnet_pipeline(use_mock=True, skip_pdf=False)
    
    if result1 and not result1["success"]:
        # Test 2: Skip PDF generation
        print(f"\nğŸ”„ Test 2: Trying without PDF generation...")
        result2 = await test_trashnet_pipeline(use_mock=True, skip_pdf=True)
        
        if result2 and result2["success"]:
            print(f"\nğŸ�‰ SUCCESS! Pipeline works without PDF generation.")
            print("ğŸ’¡ The issue is specifically in PDF color conversion.")
        else:
            print(f"\nğŸ”§ Need deeper troubleshooting...")
    
else:
    print("â�Œ Cannot run tests - no categories available")

# Additional debugging
print("\n" + "="*60)
print("ğŸ”§ DEBUGGING INFO:")
print("âœ… Color.toHex method fixed with float-to-int conversion")
print("âœ… Mock mode enabled to avoid Gemini API issues")
print("ğŸ’¡ If PDF still fails, the issue is in specific color usage in your PDF template")


# =============================================================================
# ğŸ§ª GEMINI AI VERIFICATION TEST
# =============================================================================

print("ğŸ§ª Testing Gemini AI Integration...")

import base64  

async def test_gemini_detection():
    """Test if real Gemini detection is working"""
    if use_gemini:
        print("ğŸš€ Testing REAL Gemini Vision detection...")
        # Load a test image
        image_data, category, image_path = get_random_image()
        if image_data:
            try:
                # Test direct Gemini call
                image_b64 = base64.b64encode(image_data).decode('utf-8')
                result = await vision_provider.analyze_image(image_b64)
                
                print(f"âœ… Real Gemini detection successful!")
                print(f"   Provider: {result['provider']}")
                print(f"   Model: {result['model_version']}")
                print(f"   Items detected: {len(result['items'])}")
                
                for i, item in enumerate(result['items'][:3], 1):  # Show first 3 items
                    print(f"   {i}. {item['name']} (confidence: {item['confidence']:.2f})")
                    if 'description' in item:
                        print(f"      Description: {item['description'][:60]}...")
                
                if 'raw_response' in result:
                    print(f"   Raw response preview: {result['raw_response'][:100]}...")
                    
            except Exception as e:
                print(f"â�Œ Real Gemini test failed: {e}")
                print("ğŸ’¡ Falling back to mock mode for this session")
    else:
        print("ğŸ�­ Using mock detection - no real Gemini API key configured")
        print("ğŸ’¡ Add your Gemini API key to Kaggle Secrets for real AI detection")

await test_gemini_detection()


# =============================================================================
# ğŸ”„ MULTIPLE TRASHNET IMAGES TEST (CORRECTED)
# =============================================================================

print("ğŸ”„ Testing Bulk Processing with Multiple TrashNet Images...")

async def test_multiple_trashnet_images():
    # Load multiple images from different categories using the CORRECT path
    categories = ['plastic', 'paper', 'glass', 'metal']
    test_images = []
    loaded_categories = []
    
    print("ğŸ“¥ Loading multiple TrashNet images...")
    
    for category in categories:
        # Use the get_random_image function from the previous cell which works correctly
        image_data, actual_category, image_path = get_random_image(category)
        
        if image_data:
            test_images.append(image_data)
            loaded_categories.append(actual_category)
            print(f"   âœ… Loaded {actual_category} image: {os.path.basename(image_path)}")
        else:
            print(f"   â�Œ Failed to load {category} image")
    
    if not test_images:
        print("â�Œ No images loaded for bulk testing")
        print("ğŸ’¡ Check if TrashNet dataset is properly loaded at /kaggle/input/trashnet/trashnet/")
        return
    
    print(f"ğŸ“¸ Processing {len(test_images)} TrashNet images in parallel...")
    print(f"ğŸ�¯ Categories: {', '.join(loaded_categories)}")
    
    try:
        result = await orchestrator.process_multiple_images(
            images_data=test_images,
            user_id="bulk_trashnet_user",
            location="SF"
        )
        
        print(f"âœ… Bulk processing completed!")
        print(f"   Total images processed: {result['total_images']}")
        print(f"   Successful analyses: {result['successful_processing']}")
        
        if result['successful_processing'] > 0:
            # Calculate total statistics
            total_items = 0
            total_recyclable = 0
            valid_results = 0
            
            for batch_result in result['results']:
                if batch_result and 'classified_items' in batch_result:
                    total_items += len(batch_result.get('classified_items', []))
                    stats = batch_result.get('summary_stats', {})
                    total_recyclable += stats.get('recyclable_percent', 0)
                    valid_results += 1
            
            if valid_results > 0:
                avg_recyclable = total_recyclable / valid_results
                print(f"   Total items analyzed: {total_items}")
                print(f"   Average recyclable rate: {avg_recyclable:.1f}%")
                
                # Show individual results
                print(f"\nğŸ“Š INDIVIDUAL IMAGE RESULTS:")
                for i, batch_result in enumerate(result['results']):
                    if batch_result and 'summary_stats' in batch_result:
                        stats = batch_result['summary_stats']
                        print(f"   Image {i+1}: {stats.get('recyclable_percent', 0):.1f}% recyclable")
            else:
                print("   âš ï¸� No valid results to analyze")
        else:
            print("   âš ï¸� No successful analyses completed")
        
        return result
        
    except Exception as e:
        print(f"â�Œ Bulk processing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Alternative function using direct TrashNet access
async def test_multiple_trashnet_direct():
    """Alternative approach using direct TrashNet dataset access"""
    print("ğŸ”„ Testing with direct TrashNet access...")
    
    test_images = []
    loaded_info = []
    
    # Use the working get_random_image function
    for category in ['plastic', 'paper', 'glass', 'metal']:
        image_data, actual_category, image_path = get_random_image(category)
        if image_data:
            test_images.append(image_data)
            loaded_info.append(f"{actual_category}: {os.path.basename(image_path)}")
    
    if test_images:
        print(f"âœ… Loaded {len(test_images)} images:")
        for info in loaded_info:
            print(f"   ğŸ“¸ {info}")
        
        result = await orchestrator.process_multiple_images(
            images_data=test_images,
            user_id="direct_trashnet_user", 
            location="NYC"
        )
        return result
    else:
        print("â�Œ Could not load any images")
        return None

# Run bulk test
print("ğŸ�¯ Attempting bulk processing test...")
result = await test_multiple_trashnet_images()

if not result or result['successful_processing'] == 0:
    print("\nğŸ”„ Trying alternative approach...")
    await test_multiple_trashnet_direct()

print("\nğŸ’¡ Troubleshooting Tips:")
print("1. Ensure TrashNet dataset is added via '+ Add Data'")
print("2. Check dataset path: /kaggle/input/trashnet/trashnet/")
print("3. Verify categories exist in train/val/test subdirectories")
print("4. Try single image test first to verify pipeline works")


# =============================================================================
# ğŸ“„ PDF GENERATION TEST (CLEAN VERSION)
# =============================================================================

print("ğŸ”„ Loading PDF downloader...")

# Import the correct PDFDownloader
from ecoguardian.utils.pdf_downloader import PDFDownloader

# Verify the import worked
print("âœ… PDFDownloader methods available:")
print(f"   - create_interactive_pdf_dashboard: {hasattr(PDFDownloader, 'create_interactive_pdf_dashboard')}")
print(f"   - create_simple_preview: {hasattr(PDFDownloader, 'create_simple_preview')}")
print(f"   - create_download_link: {hasattr(PDFDownloader, 'create_download_link')}")
print(f"   - display_pdf_info: {hasattr(PDFDownloader, 'display_pdf_info')}")
print(f"   - save_pdf_to_output: {hasattr(PDFDownloader, 'save_pdf_to_output')}")

async def test_enhanced_pdf_generation():
    """Test PDF generation with enhanced preview functionality"""
    # Load a TrashNet image for testing
    print("ğŸ”„ Loading waste image for enhanced PDF test...")
    image_data, category, image_path = get_random_image('plastic')
    
    if image_data is None:
        print("â�Œ Failed to load image, trying random category...")
        image_data, category, image_path = get_random_image()
    
    if image_data is None:
        print("â�Œ Could not load any images for PDF test")
        return
    
    print(f"ğŸ�¯ Generating enhanced PDF report for: {category} waste")
    print(f"ğŸ–¼ï¸� Source image: {os.path.basename(image_path)}")
    
    # Process through pipeline
    print("ğŸ”� Processing through EcoGuardian pipeline...")
    pipeline_result = await orchestrator.process_single_image(
        image_data=image_data,
        user_id=f"enhanced_pdf_test_{category}",
        location="NYC"
    )
    
    if pipeline_result["success"]:
        if 'pdf_bytes' in pipeline_result["report"]:
            report = pipeline_result["report"]
            pdf_bytes = report['pdf_bytes']
            pdf_filename = report['pdf_filename']
            
            # Display enhanced PDF information
            print("âœ… Enhanced PDF Generation Successful!")
            PDFDownloader.display_pdf_info(pdf_bytes, pdf_filename)
            
            # Save to output directory
            saved_path = PDFDownloader.save_pdf_to_output(pdf_bytes, pdf_filename)
            
            print("\n" + "="*60)
            print("ğŸ�¨ ENHANCED PDF PREVIEW OPTIONS:")
            print("="*60)
            
            # Option 1: Interactive Dashboard (Recommended)
            print("\n1. ğŸš€ INTERACTIVE DASHBOARD (Recommended):")
            dashboard = PDFDownloader.create_interactive_pdf_dashboard(pdf_bytes, pdf_filename)
            display(dashboard)
            
            # Option 2: Simple Preview
            print("\n2. ğŸ‘€ SIMPLE PREVIEW:")
            simple_preview = PDFDownloader.create_simple_preview(pdf_bytes, pdf_filename)
            display(simple_preview)
            
            # Option 3: Standard Download Link
            print("\n3. ğŸ“¥ STANDARD DOWNLOAD:")
            download_link = PDFDownloader.create_download_link(pdf_bytes, pdf_filename, "ğŸ“¥ Download Enhanced Report")
            display(download_link)
            
        else:
            print("â�Œ PDF generation failed - no PDF data in report")
    else:
        print("â�Œ Pipeline processing failed")
        print(f"   Error: {pipeline_result.get('error', 'Unknown error')}")

# Run enhanced PDF generation test
print("ğŸš€ Starting enhanced PDF generation test...")
await test_enhanced_pdf_generation()


# =============================================================================
# ğŸ“„ ENHANCED PDF REPORT GENERATION TEST (WITH PREVIEW)
# =============================================================================
from ecoguardian.utils.pdf_downloader import PDFDownloader

print("ğŸ“„ Testing Enhanced PDF Report Generation with Preview...")

async def test_enhanced_pdf_generation():
    """Test PDF generation with enhanced preview functionality"""
    # Load a TrashNet image for testing
    print("ğŸ”„ Loading waste image for enhanced PDF test...")
    image_data, category, image_path = get_random_image('plastic')
    
    if image_data is None:
        print("â�Œ Failed to load image, trying random category...")
        image_data, category, image_path = get_random_image()
    
    if image_data is None:
        print("â�Œ Could not load any images for PDF test")
        return
    
    print(f"ğŸ�¯ Generating enhanced PDF report for: {category} waste")
    print(f"ğŸ–¼ï¸� Source image: {os.path.basename(image_path)}")
    
    # Process through pipeline
    print("ğŸ”� Processing through EcoGuardian pipeline...")
    pipeline_result = await orchestrator.process_single_image(
        image_data=image_data,
        user_id=f"enhanced_pdf_test_{category}",
        location="NYC"
    )
    
    if pipeline_result["success"]:
        if 'pdf_bytes' in pipeline_result["report"]:
            report = pipeline_result["report"]
            pdf_bytes = report['pdf_bytes']
            pdf_filename = report['pdf_filename']
            
            # Display enhanced PDF information
            print("âœ… Enhanced PDF Generation Successful!")
            PDFDownloader.display_pdf_info(pdf_bytes, pdf_filename)
            
            # Save to output directory
            saved_path = PDFDownloader.save_pdf_to_output(pdf_bytes, pdf_filename)
            
            print("\n" + "="*60)
            print("ğŸ�¨ ENHANCED PDF PREVIEW OPTIONS:")
            print("="*60)
            
            # Option 1: Interactive Dashboard (Recommended)
            print("\n1. ğŸš€ INTERACTIVE DASHBOARD (Recommended):")
            dashboard = PDFDownloader.create_interactive_pdf_dashboard(pdf_bytes, pdf_filename)
            display(dashboard)
            
            # Option 2: Simple Preview
            print("\n2. ğŸ‘€ SIMPLE PREVIEW:")
            simple_preview = PDFDownloader.create_simple_preview(pdf_bytes, pdf_filename)
            display(simple_preview)
            
            # Option 3: Standard Download Link
            print("\n3. ğŸ“¥ STANDARD DOWNLOAD:")
            download_link = PDFDownloader.create_download_link(pdf_bytes, pdf_filename, "ğŸ“¥ Download Enhanced Report")
            display(download_link)
            
            # Report summary
            print("\nğŸ“‹ ENHANCED REPORT SUMMARY:")
            print(f"   ğŸ“Š Items Analyzed: {report['summary']['total_items']}")
            print(f"   â™»ï¸�  Recyclable: {report['summary']['recyclable_percent']:.1f}%")
            print(f"   ğŸŒ± Compost: {report['summary']['compost_percent']:.1f}%")
            print(f"   ğŸ—‘ï¸�  Landfill: {report['summary']['landfill_percent']:.1f}%")
            print(f"   ğŸŒ� CO2 Saved: {report['environmental_impact']['co2_saved_kg']:.1f} kg")
            
            # Show preview tips
            print("\nğŸ’¡ PREVIEW TIPS:")
            print("   â€¢ Use the Interactive Dashboard for best experience")
            print("   â€¢ Toggle preview to view PDF without downloading")
            print("   â€¢ Download multiple copies if needed")
            print("   â€¢ Check Kaggle output tab for saved file")
            
        else:
            print("â�Œ PDF generation failed - no PDF data in report")
    else:
        print("â�Œ Pipeline processing failed")
        print(f"   Error: {pipeline_result.get('error', 'Unknown error')}")

# Run enhanced PDF generation test
print("ğŸš€ Starting enhanced PDF generation test...")
await test_enhanced_pdf_generation()

print("\n" + "="*60)
print("ğŸ�¯ ADDITIONAL PREVIEW OPTIONS:")
print("="*60)

# Test different preview styles
async def test_multiple_preview_styles():
    """Test different PDF preview styles"""
    categories = ['paper', 'glass', 'metal']
    
    for category in categories:
        print(f"\nğŸ”� Testing {category} waste with different previews...")
        image_data, actual_category, image_path = get_random_image(category)
        
        if image_data:
            result = await orchestrator.process_single_image(
                image_data=image_data,
                user_id=f"preview_test_{actual_category}",
                location="SF"
            )
            
            if result["success"] and 'pdf_bytes' in result["report"]:
                pdf_bytes = result["report"]['pdf_bytes']
                filename = result["report"]['pdf_filename']
                
                print(f"âœ… {actual_category} PDF ready - testing preview styles...")
                
                # Test simple preview
                preview = PDFDownloader.create_simple_preview(pdf_bytes, filename)
                display(preview)
                
                # Also save to output
                PDFDownloader.save_pdf_to_output(pdf_bytes, filename)

print("\nğŸ§ª Testing multiple preview styles...")
await test_multiple_preview_styles()

print("\nğŸ�‰ ENHANCED PDF PREVIEW SYSTEM READY!")
print("ğŸ’¡ Features available:")
print("   â€¢ Interactive PDF dashboards")
print("   â€¢ Embedded PDF previews") 
print("   â€¢ Multiple download options")
print("   â€¢ Professional styling")
print("   â€¢ File management tools")

