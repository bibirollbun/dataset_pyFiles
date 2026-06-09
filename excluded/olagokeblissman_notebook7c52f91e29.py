import re
import xml.etree.ElementTree as ET
from openai import OpenAI

class Model:
    def __init__(self):
        # Load the LLM (assuming API access to GPT-based models)
        self.client = OpenAI(api_key="your_api_key")  # Use Kaggle Secrets for API keys
    
    def generate_svg(self, prompt: str) -> str:
        """Generate SVG code from text prompt using an LLM."""
        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": "You are an SVG code generator."},
                      {"role": "user", "content": f"Generate SVG code for: {prompt}"}],
            temperature=0.7
        )
        svg_code = response.choices[0].message.content

        # Post-process output: Extract valid SVG content
        svg_code = self.clean_svg(svg_code)
        
        return svg_code
    
    def clean_svg(self, svg_code: str) -> str:
        """Ensure valid SVG format and remove unwanted text."""
        match = re.search(r"<svg[\s\S]*?</svg>", svg_code, re.IGNORECASE)
        return match.group(0) if match else "<svg></svg>"

    def validate_svg(self, svg_code: str) -> bool:
        """Check if the generated SVG code is valid."""
        try:
            ET.fromstring(svg_code)  # Parse as XML
            return True
        except ET.ParseError:
            return False

    def predict(self, prompt: str) -> str:
        """Main function for Kaggle submission, ensuring valid SVG output."""
        svg_code = self.generate_svg(prompt)
        return svg_code if self.validate_svg(svg_code) else "<svg></svg>"


