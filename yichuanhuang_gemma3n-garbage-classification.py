!pip install timm --upgrade -q
!pip install accelerate -q
!pip install transformers==4.53 -q


import kagglehub

GEMMA_PATH = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e4b-it")


# Import required libraries
import torch
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from io import BytesIO
import logging
from typing import Union, Tuple
from dataclasses import dataclass
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("Libraries imported successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")


# Configuration class
@dataclass
class Config:
    # Gemma3n model configuration
    MODEL_NAME: str = GEMMA_PATH
    
    # Generation parameters
    MAX_NEW_TOKENS: int = 512
    
    # Device configuration
    TORCH_DTYPE: str = torch.bfloat16
    DEVICE_MAP: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Image preprocessing
    IMAGE_SIZE: int = 512
    
    # Hugging Face token (if needed)
    HF_TOKEN: str = ""

# Initialize config
config = Config()
print(f"Model: {config.MODEL_NAME}")
print(f"Device: {config.DEVICE_MAP}")
print(f"Data type: {config.TORCH_DTYPE}")


class GarbageClassificationKnowledge:
    @staticmethod
    def get_system_prompt():
        return """You are a professional garbage classification expert. You need to carefully observe the items in the picture, analyze their materials, properties and uses, and then make accurate judgments according to garbage classification standards.

IMPORTANT: You should ONLY classify items that are actually garbage/waste. If the image contains people, living things, furniture, electronics in use, or other non-waste items, you should classify it as "Unable to classify" and explain that it's not garbage.

**MIXED GARBAGE HANDLING RULES - ABSOLUTE COMPLIANCE REQUIRED:**

1. **Food with Recyclable Containers (EXACT TIP MANDATORY)**:
   - OBVIOUSLY VISIBLE FOOD (chunks, liquids, substantial residue) in recyclable containers: Container goes to "Food/Kitchen Waste" 
   - **COMPULSORY OUTPUT**: "âš ï¸� Tip: Empty and rinse this container first, then it can be recycled!"
   - **ZERO TOLERANCE**: This exact tip MUST appear in your response - failure to include = incorrect classification
   - MINOR RESIDUE (grease stains, light film, pizza box grease spots) in recyclable containers: Container remains "Recyclable Waste"
   - This is the ONLY allowed mixed-type situation (recyclable container + food content)
   
2. **Different Type Mixed (EXACT WARNING MANDATORY)**:
   - If image shows clearly different waste categories mixed together: classify as "Unable to classify" 
   - YOU MUST INCLUDE THIS EXACT TEXT: "âš ï¸� Warning: Multiple garbage types detected. Please separate items for proper classification."
   - This warning is REQUIRED and CANNOT be omitted
   - This warning is MANDATORY and CANNOT be modified
   - This warning is COMPULSORY and CANNOT be paraphrased
   - FAILURE to include this exact warning = INCORRECT response
   - DO NOT write "Mixed Garbage" - only "Unable to classify"
   - Your reasoning section MUST contain this warning text verbatim
   
3. **Same Type Mixed (CLASSIFICATION MANDATORY)**:
   - Multiple electronics = "Hazardous Waste" - **NEVER "Unable to classify"**
   - Multiple recyclables = "Recyclable Waste" - **NEVER "Unable to classify"**
   - Multiple food items = "Food/Kitchen Waste" - **NEVER "Unable to classify"**
   - **ZERO TOLERANCE**: Same-category items MUST be classified to their category - calling them "mixed" = wrong

**ABSOLUTE ENFORCEMENT**: Any deviation from these exact outputs constitutes classification failure. Follow these rules with 100% precision or your response is incorrect.

**Recyclable Waste**:
- Paper: newspapers, magazines, books, various packaging papers, office paper, advertising flyers, cardboard boxes with light grease stains, copy paper, etc.
- Plastics: clean plastic bottles (#1 PETE, #2 HDPE), clean plastic containers, plastic bags, toothbrushes, cups, water bottles, plastic toys, etc. (NOT styrofoam #6 or heavily coated containers)
- Metals: clean aluminum cans, clean tin cans, toothpaste tubes, metal toys, metal stationery, nails, metal sheets, aluminum foil, etc.
- Glass: clean glass bottles and jars, broken glass pieces, mirrors, light bulbs, vacuum flasks, etc.
- Textiles: old clothing, textile products, shoes, curtains, towels, bags, etc.
- NOTE: Light grease stains or minor residue are acceptable for recycling. Only obvious food content requires cleaning first.

**Food/Kitchen Waste**:
- Food scraps: rice, noodles, bread, meat, fish, shrimp shells, crab shells, bones, etc.
- Fruit peels and cores: watermelon rinds, apple cores, orange peels, banana peels, nut shells, etc.
- Plants: withered branches and leaves, flowers, traditional Chinese medicine residue, etc.
- Expired food: expired canned food, cookies, candy, etc.
- Containers with obvious food content (chunks, liquids, substantial residue)

**Hazardous Waste**:
- Batteries: dry batteries, rechargeable batteries, button batteries, and all types of batteries
- Light tubes: energy-saving lamps, fluorescent tubes, incandescent bulbs, LED lights, etc.
- Pharmaceuticals: expired medicines, medicine packaging, thermometers, blood pressure monitors, etc.
- Paints: paint, coatings, glue, nail polish, cosmetics, etc.
- Others: pesticides, cleaning agents, agricultural chemicals, X-ray films, etc.

**Other Waste**:
- Contaminated non-recyclable paper: toilet paper, diapers, wet wipes, napkins, etc.
- Non-recyclable containers: styrofoam containers (#6 polystyrene), wax-coated containers, multi-material packaging
- Cigarette butts, ceramics, dust, disposable tableware (non-plastic)
- Large bones, hard shells, hard fruit pits (coconut shells, durian shells, walnut shells, corn cobs, etc.)
- Hair, pet waste, cat litter, etc.

**Unable to classify**:
- People, human faces, human body parts
- Living animals, pets
- Furniture, appliances, electronics in normal use
- Buildings, landscapes, vehicles
- Any item that is not intended to be discarded as waste
- Multiple different garbage types mixed together

Please observe the items in the image carefully according to the above classification standards and provide accurate classification results.

Format your response EXACTLY as follows:

**Classification**: [Category Name or "Unable to classify"]
**Reasoning**: [Brief explanation of why this item belongs to this category, or why it cannot be classified as garbage]
**Confidence Score**: [Number from 1-10]"""

    @staticmethod
    def get_categories():
        return [
            "Recyclable Waste",
            "Food/Kitchen Waste",
            "Hazardous Waste",
            "Other Waste",
            "Unable to classify",
        ]

    @staticmethod
    def get_category_descriptions():
        return {
            "Recyclable Waste": "Items that can be processed and reused, including paper, plastic, metal, glass, and textiles (light grease stains acceptable)",
            "Food/Kitchen Waste": "Organic waste from food preparation and consumption, including containers with obvious food content",
            "Hazardous Waste": "Items containing harmful substances that require special disposal",
            "Other Waste": "Items that don't fit into other categories and go to general waste",
            "Unable to classify": "Items that are not garbage/waste, such as people, living things, functioning objects, or multiple different garbage types mixed together",
        }


from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch
import logging
from typing import Union, Tuple
import re


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Preprocess image to meet Gemma3n requirements (512x512)
    """
    # Convert to RGB if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize to 512x512 as required by Gemma3n
    target_size = (512, 512)

    # Calculate aspect ratio preserving resize
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height

    if aspect_ratio > 1:
        # Width is larger
        new_width = target_size[0]
        new_height = int(target_size[0] / aspect_ratio)
    else:
        # Height is larger or equal
        new_height = target_size[1]
        new_width = int(target_size[1] * aspect_ratio)

    # Resize image maintaining aspect ratio
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Create a new image with target size and paste the resized image
    processed_image = Image.new(
        "RGB", target_size, (255, 255, 255)
    )  # White background

    # Calculate position to center the image
    x_offset = (target_size[0] - new_width) // 2
    y_offset = (target_size[1] - new_height) // 2

    processed_image.paste(image, (x_offset, y_offset))

    return processed_image


class GarbageClassifier:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.knowledge = GarbageClassificationKnowledge()
        self.processor = None
        self.model = None
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_model(self):
        """Load the model and processor"""
        try:
            self.logger.info(f"Loading model: {self.config.MODEL_NAME}")

            # Load processor
            kwargs = {}
            if self.config.HF_TOKEN:
                kwargs["token"] = self.config.HF_TOKEN

            self.processor = AutoProcessor.from_pretrained(
                self.config.MODEL_NAME, **kwargs
            )

            # Load model
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.config.MODEL_NAME,
                torch_dtype=self.config.TORCH_DTYPE,
                device_map=self.config.DEVICE_MAP,
            )

            self.logger.info("Model loaded successfully")

        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def classify_image(self, image: Union[str, Image.Image]) -> Tuple[str, str, int]:
        """
        Classify garbage in the image

        Args:
            image: PIL Image or path to image file

        Returns:
            Tuple of (classification_result, detailed_analysis, confidence_score)
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Load and process image
            if isinstance(image, str):
                image = Image.open(image)
            elif not isinstance(image, Image.Image):
                raise ValueError("Image must be a PIL Image or file path")

            # Preprocess image to meet Gemma3n requirements
            processed_image = preprocess_image(image)

            # Prepare messages with system prompt and user query
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": self.knowledge.get_system_prompt(),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": processed_image},
                        {
                            "type": "text",
                            "text": "Please classify what you see in this image. If it shows garbage/waste items, classify them according to the garbage classification standards. If it shows people, living things, or other non-waste items, classify it as 'Unable to classify' and explain why it's not garbage. Also provide a confidence score from 1-10 indicating how certain you are about your classification.",
                        },
                    ],
                },
            ]

            # Apply chat template and tokenize
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device, dtype=self.model.dtype)
            input_len = inputs["input_ids"].shape[-1]

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.MAX_NEW_TOKENS,
                disable_compile=True,
            )
            response = self.processor.batch_decode(
                outputs[:, input_len:],
                skip_special_tokens=True,
            )[0]

            # Extract classification from response
            classification = self._extract_classification(response)

            # Extract reasoning from response
            reasoning = self._extract_reasoning(response)

            # Extract confidence score from response
            confidence_score = self._extract_confidence_score(response, classification)

            return classification, reasoning, confidence_score

        except Exception as e:
            self.logger.error(f"Error during classification: {str(e)}")
            import traceback

            traceback.print_exc()
            return "Error", f"Classification failed: {str(e)}", 0


    def _calculate_confidence_heuristic(self, response_lower: str, classification: str) -> int:
        """Calculate confidence based on response content and classification type"""
        base_confidence = 5

        # Confidence indicators (increase confidence)
        high_confidence_words = ["clearly", "obviously", "definitely", "certainly", "exactly"]
        medium_confidence_words = ["appears", "seems", "likely", "probably"]

        # Uncertainty indicators (decrease confidence)
        uncertainty_words = ["might", "could", "possibly", "maybe", "unclear", "difficult"]

        # Adjust based on confidence words
        for word in high_confidence_words:
            if word in response_lower:
                base_confidence += 2
                break

        for word in medium_confidence_words:
            if word in response_lower:
                base_confidence += 1
                break

        for word in uncertainty_words:
            if word in response_lower:
                base_confidence -= 2
                break

        # Classification-specific adjustments
        if classification == "Unable to classify":
            if any(indicator in response_lower for indicator in ["person", "people", "human", "living"]):
                base_confidence += 1  # High confidence when clearly not waste
            else:
                base_confidence -= 1  # Lower confidence for unclear items

        elif classification == "Error":
            base_confidence = 1

        else:
            # Check for specific material mentions (increases confidence)
            specific_materials = ["aluminum", "plastic", "glass", "metal", "cardboard", "paper"]
            if any(material in response_lower for material in specific_materials):
                base_confidence += 1

        return min(max(base_confidence, 1), 10)

    def _extract_confidence_score(self, response: str, classification: str) -> int:
        """Extract confidence score from response or calculate based on classification"""
        response_lower = response.lower()

        # Look for explicit confidence scores in the response
        confidence_patterns = [
            r'\*\*confidence score\*\*[:\s]*(\d+)',  # For **Confidence Score**: format
            r'confidence[:\s]*(\d+)',
            r'confident[:\s]*(\d+)',
            r'certainty[:\s]*(\d+)',
            r'score[:\s]*(\d+)',
            r'(\d+)/10',
            r'(\d+)\s*out\s*of\s*10'
        ]

        for pattern in confidence_patterns:
            match = re.search(pattern, response_lower)
            if match:
                score = int(match.group(1))
                return min(max(score, 1), 10)  # Clamp between 1-10

        # If no explicit score found, calculate based on classification indicators
        return self._calculate_confidence_heuristic(response_lower, classification)

    def _extract_classification(self, response: str) -> str:
        """Extract the main classification from the response - trust Gemma 3n intelligence more"""
        response_lower = response.lower()

        # Primary: Trust explicit category mentions from Gemma 3n
        categories = self.knowledge.get_categories()

        for category in categories:
            if category.lower() in response_lower:
                # Simple negation check
                category_index = response_lower.find(category.lower())
                context_before = response_lower[max(0, category_index - 20):category_index]

                if not any(neg in context_before[-10:] for neg in ["not", "cannot", "isn't"]):
                    return category

        # Secondary: Look for explicit mixed garbage warnings from model
        mixed_warnings = [
            "multiple garbage types detected",
            "separate items",
            "different garbage types",
            "mixed together"
        ]

        if any(warning in response_lower for warning in mixed_warnings):
            return "Unable to classify"

        # Tertiary: Basic material detection (simplified)
        if any(material in response_lower for material in
               ["recyclable", "aluminum", "plastic", "glass", "metal", "cardboard"]):
            # Check for contamination
            if any(cont in response_lower for cont in ["obvious food", "substantial residue", "chunks", "liquids"]):
                return "Food/Kitchen Waste"
            return "Recyclable Waste"

        if any(food in response_lower for food in ["food", "organic", "kitchen", "fruit", "vegetable"]):
            return "Food/Kitchen Waste"

        if any(hazard in response_lower for hazard in ["battery", "hazardous", "chemical", "toxic"]):
            return "Hazardous Waste"

        if any(other in response_lower for other in ["cigarette", "ceramic", "styrofoam"]):
            return "Other Waste"

        # Non-garbage detection
        if any(non_garbage in response_lower for non_garbage in ["person", "people", "human", "living", "animal"]):
            return "Unable to classify"

        # Final fallback - let Gemma 3n's reasoning guide us
        if any(unable in response_lower for unable in ["unable to classify", "cannot classify", "not garbage"]):
            return "Unable to classify"

        # Default to Unable to classify if unclear
        return "Unable to classify"

    def _extract_reasoning(self, response: str) -> str:
        """Extract only the reasoning content, removing all formatting markers and classification info"""
        import re

        # Remove all formatting markers
        cleaned_response = response.replace("**Classification**:", "")
        cleaned_response = cleaned_response.replace("**Reasoning**:", "")
        cleaned_response = re.sub(r'\*\*.*?\*\*:', '', cleaned_response)  # Remove any **text**: patterns
        cleaned_response = cleaned_response.replace("**", "")  # Remove remaining ** markers

        # Remove category names that might appear at the beginning
        categories = self.knowledge.get_categories()
        for category in categories:
            if cleaned_response.strip().startswith(category):
                cleaned_response = cleaned_response.replace(category, "", 1)
                break

        # Remove common material names that might appear at the beginning
        material_names = [
            "Glass", "Plastic", "Metal", "Paper", "Cardboard", "Aluminum",
            "Steel", "Iron", "Tin", "Foil", "Wood", "Ceramic", "Fabric",
            "Recyclable Waste", "Food/Kitchen Waste", "Hazardous Waste", "Other Waste"
        ]

        # Clean the response
        cleaned_response = cleaned_response.strip()

        # Remove material names at the beginning
        for material in material_names:
            if cleaned_response.startswith(material):
                # Remove the material name and any following punctuation/whitespace
                cleaned_response = cleaned_response[len(material):].lstrip(" .,;:")
                break

        # Split into sentences and clean up
        sentences = []

        # Split by common sentence endings, but keep the endings
        parts = re.split(r'([.!?])\s+', cleaned_response)

        # Rejoin parts to maintain sentence structure
        reconstructed_parts = []
        for i in range(0, len(parts), 2):
            if i < len(parts):
                sentence = parts[i]
                if i + 1 < len(parts):
                    sentence += parts[i + 1]  # Add the punctuation back
                reconstructed_parts.append(sentence)

        for part in reconstructed_parts:
            part = part.strip()
            if not part:
                continue

            # Skip parts that are just category names or material names
            if part in categories or part.rstrip(".,;:") in material_names:
                continue

            # Skip parts that start with category names or material names
            is_category_line = False
            for item in categories + material_names:
                if part.startswith(item):
                    is_category_line = True
                    break

            if is_category_line:
                continue

            # Clean up the sentence
            part = re.sub(r'^[A-Za-z\s]+:', '', part).strip()  # Remove "Category:" type prefixes

            if part and len(part) > 3:  # Only keep meaningful content
                sentences.append(part)

        # Join sentences
        reasoning = ' '.join(sentences)

        # Final cleanup - remove any remaining standalone material words at the beginning
        reasoning_words = reasoning.split()
        if reasoning_words and reasoning_words[0] in [m.lower() for m in material_names]:
            reasoning_words = reasoning_words[1:]
            reasoning = ' '.join(reasoning_words)

        # Ensure proper capitalization
        if reasoning:
            reasoning = reasoning[0].upper() + reasoning[1:] if len(reasoning) > 1 else reasoning.upper()

            # Ensure proper punctuation
            if not reasoning.endswith(('.', '!', '?')):
                reasoning += '.'

        return reasoning if reasoning else "Analysis not available"

    def get_categories_info(self):
        """Get information about all categories"""
        return self.knowledge.get_category_descriptions()


# Utility function to load image from URL
def load_image_from_url(url: str) -> Image.Image:
    """Load image from URL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return image
    except Exception as e:
        logger.error(f"Error loading image from URL: {str(e)}")
        raise

# Function to display image
def display_image_with_info(image: Image.Image, title: str = "Input Image"):
    """Display image with information"""
    print(f"\n{title}")
    print(f"Image size: {image.size}")
    print(f"Image mode: {image.mode}")
    display(image)

def get_confidence_emoji(confidence_score: int) -> str:
    """Get emoji representation of confidence score"""
    if confidence_score >= 8:
        return "ğŸŸ¢"  # High confidence
    elif confidence_score >= 6:
        return "ğŸŸ¡"  # Medium confidence
    else:
        return "ğŸ”´"  # Low confidence

def format_classification_output(classification: str, reasoning: str, confidence_score: int):
    """Format the classification output in a user-friendly way"""
    confidence_emoji = get_confidence_emoji(confidence_score)
    
    print(f"\n{'='*50}")
    print(f"ğŸ“Š CLASSIFICATION RESULT")
    print(f"{'='*50}")
    print(f"ğŸ�·ï¸�  Category: {classification}")
    print(f"ğŸ�¯ Confidence: {confidence_emoji} {confidence_score}/10")
    print(f"ğŸ’¡ Analysis: {reasoning}")
    print(f"{'='*50}")

def classify_and_display(classifier, image_path_or_url: str, title: str = ""):
    """Helper function to classify an image and display results"""
    try:
        # Determine if it's a URL or local path
        if image_path_or_url.startswith(('http://', 'https://')):
            image = load_image_from_url(image_path_or_url)
            source_type = "URL"
        else:
            image = Image.open(image_path_or_url)
            source_type = "Local file"
        
        # Display image
        if title:
            print(f"\nğŸ–¼ï¸� {title}")
        print(f"ğŸ“� Source: {source_type}")
        display(image)
        
        # Classify
        classification, reasoning, confidence_score = classifier.classify_image(image)
        
        # Display results
        format_classification_output(classification, reasoning, confidence_score)
        
        return classification, reasoning, confidence_score
        
    except Exception as e:
        print(f"â�Œ Error processing image: {str(e)}")
        return "Error", str(e), 0

print("Enhanced utility functions defined!")


# Initialize and load the model
print("Initializing classifier...")
classifier = GarbageClassifier(config)

print("Loading model... This may take a few minutes on first run.")
classifier.load_model()
print("âœ… Model loaded successfully!")


# Test with URL images

test_urls = {
    "example1": "https://respic.3d66.com/coverimg/cache/50a7/b9e60765e8f4bbaf0c7fbf6296886e2a.jpg!medium-size-2?v=2152416&k=D41D8CD98F00B204E9800998ECF8427E",
    "example2": "https://www.arbin.com/wp-content/uploads/2020/03/Batteries-298x143-1.jpg"
}

for material, url in test_urls.items():
    print(f"\n{'='*60}")
    print(f"ğŸ”� TESTING {material.upper()} FROM URL")
    print(f"URL: {url}")
    print('='*60)
    
    try:
        # Load image from URL
        image = load_image_from_url(url)
        display(image)
        
        # Classify the image
        classification, reasoning, confidence_score = classifier.classify_image(image)
        
        # Output results with confidence
        confidence_emoji = get_confidence_emoji(confidence_score)
        print(f"\nğŸ�¯ Classification Results:")
        print(f"ğŸ“‚ Category: {classification}")
        print(f"ğŸ�¯ Confidence: {confidence_emoji} {confidence_score}/10")
        print(f"ğŸ’­ Reasoning: {reasoning}")
        
    except Exception as e:
        print(f"â�Œ Error: {str(e)}")


# Display category information
knowledge = GarbageClassificationKnowledge()
categories_info = knowledge.get_category_descriptions()

print("ğŸ—‚ï¸� GARBAGE CLASSIFICATION CATEGORIES")
print("="*50)

for category, description in categories_info.items():
    print(f"\nğŸ“‚ {category}")
    print(f"   {description}")

print("\n" + "="*50)
print("\nğŸ’¡ Tips for best results:")
print("â€¢ Use clear, well-lit images")
print("â€¢ Ensure the garbage item is clearly visible")
print("â€¢ Single items work better than cluttered scenes")
print("â€¢ The system works best with common garbage types")

