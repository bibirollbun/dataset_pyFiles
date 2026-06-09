# !pip install google-generativeai pillow

# Imports and Setup
import google.generativeai as genai
import hashlib
import json
from PIL import Image
import io
import base64


# Configure Gemini API
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
GEMINI_API_KEY = secrets.get_secret("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)


def generate_biometric_hash(image_path: str) -> dict:
    """
    Analyzes face in image using Gemini Vision and generates SHA-256 hash
    from extracted biometric features.
    """
    # Load image
    img = Image.open(image_path)
    
    # Initialize Gemini Vision model
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # Prompt to extract consistent facial features
    extraction_prompt = """
    Analyze this face image and extract ONLY these biometric descriptors.
    Be consistent - same person should get same values across different photos.
    
    Return ONLY valid JSON with these exact keys (no markdown, no explanation):
    {
        "face_shape": "oval|round|square|heart|oblong",
        "eye_spacing": "close|average|wide",
        "nose_type": "narrow|medium|wide",
        "lip_fullness": "thin|medium|full",
        "jaw_definition": "soft|medium|strong",
        "forehead_height": "low|medium|high",
        "cheekbone_prominence": "flat|medium|prominent",
        "eyebrow_arch": "flat|soft|high",
        "face_symmetry_score": <integer 1-10>,
        "unique_markers": "<describe 2-3 distinctive features>"
    }
    
    Focus on STRUCTURAL features that don't change with expression/lighting.
    """
    
    # Call Gemini Vision API
    response = model.generate_content([extraction_prompt, img])
    
    # Parse the response
    raw_text = response.text.strip()
    # Clean markdown if present
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
    raw_text = raw_text.strip()
    
    try:
        bio_vector = json.loads(raw_text)
    except json.JSONDecodeError:
        return {"error": "Failed to parse response", "raw": raw_text}
    
    # Create deterministic string from features (sorted for consistency)
    feature_string = json.dumps(bio_vector, sort_keys=True)
    
    # Generate SHA-256 hash
    bio_hash = hashlib.sha256(feature_string.encode()).hexdigest()
    
    return {
        "biometric_vector": bio_vector,
        "feature_string": feature_string,
        "sha256_hash": bio_hash,
        "hash_short": bio_hash[:16] + "..." + bio_hash[-8:]
    }



print("=" * 60)
print("VERIFAI // BIOMETRIC HASH GENERATOR v0.9")
print("=" * 60)

# Upload your first image
IMAGE_1_PATH = "/kaggle/input/input-image/image1.png"  # Change this path

result_1 = generate_biometric_hash(IMAGE_1_PATH)

print("\n[IMAGE 1 ANALYSIS]")
print(f"Hash: {result_1['hash_short']}")
print(f"\nExtracted Features:")
for k, v in result_1['biometric_vector'].items():
    print(f"  {k}: {v}")

#Test with Image 2 (Different photo of same person)
IMAGE_2_PATH = "/kaggle/input/input-image/test_img1.png"  # Change this path

result_2 = generate_biometric_hash(IMAGE_2_PATH)

print("\n[IMAGE 2 ANALYSIS]")
print(f"Hash: {result_2['hash_short']}")
print(f"\nExtracted Features:")
for k, v in result_2['biometric_vector'].items():
    print(f"  {k}: {v}")


print("\n" + "=" * 60)
print("VERIFICATION RESULT")
print("=" * 60)

hash_match = result_1['sha256_hash'] == result_2['sha256_hash']

if hash_match:
    print("âœ… IDENTITY VERIFIED - Hashes Match!")
    print(f"   Common Hash: {result_1['hash_short']}")
else:
    print("âš ï¸�  HASH MISMATCH - Investigating differences...")
    print(f"\n   Hash 1: {result_1['sha256_hash']}")
    print(f"   Hash 2: {result_2['sha256_hash']}")
    
    # Show which features differed
    print("\n   Feature Comparison:")
    v1, v2 = result_1['biometric_vector'], result_2['biometric_vector']
    for key in v1:
        if key in v2:
            match = "âœ“" if v1[key] == v2[key] else "âœ—"
            print(f"   {match} {key}: '{v1[key]}' vs '{v2[key]}'")



# Cell 7: Fuzzy Matching (More realistic approach)
def calculate_similarity(vec1: dict, vec2: dict) -> float:
    """Calculate % similarity between two biometric vectors"""
    if not vec1 or not vec2:
        return 0.0
    
    matches = 0
    total = 0
    
    for key in vec1:
        if key in vec2 and key != "unique_markers":
            total += 1
            if vec1[key] == vec2[key]:
                matches += 1
    
    return (matches / total * 100) if total > 0 else 0.0

similarity = calculate_similarity(
    result_1.get('biometric_vector', {}),
    result_2.get('biometric_vector', {})
)

print(f"\nğŸ“Š SIMILARITY SCORE: {similarity:.1f}%")
print(f"   Threshold for match: 80%")
print(f"   Result: {'PASS âœ…' if similarity >= 80 else 'FAIL â�Œ'}")

