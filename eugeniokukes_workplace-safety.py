%%capture
import os
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth
else:
    # Do this only in Colab notebooks! Otherwise use pip install unsloth
    !pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl triton cut_cross_entropy unsloth_zoo
    !pip install sentencepiece protobuf "datasets>=3.4.1,<4.0.0" huggingface_hub hf_transfer
    !pip install --no-deps unsloth


%%capture
!pip install --no-deps git+https://github.com/huggingface/transformers.git
!pip install --no-deps --upgrade timm # Only for Gemma 3N


from unsloth import FastModel
from transformers import TextStreamer
from unsloth import is_bfloat16_supported
from datasets import load_dataset

import torch
import os

fourbit_models = [
    # 4bit dynamic quants for superior accuracy and low memory use
    "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit",
    "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit",
    # Pretrained models
    "unsloth/gemma-3n-E4B-unsloth-bnb-4bit",
    "unsloth/gemma-3n-E2B-unsloth-bnb-4bit",

    # Other Gemma 3 quants
    "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",
] # More models at https://huggingface.co/unsloth

# Load Gemma 3n E4B model
model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3n-E4B-it",
    dtype = None, # None for auto detection
    max_seq_length = 1024, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)


from transformers import TextStreamer
# Helper function for inference
def do_gemma_3n_inference(messages, max_new_tokens = 128):
    msg = [{"role" : "user","content": [{ "type": "text",  "text" : messages }]}]
    _ = model.generate(
        **tokenizer.apply_chat_template(
            msg,
            add_generation_prompt = True, # Must add for generation
            tokenize = True,
            return_dict = True,
            return_tensors = "pt",
        ).to("cuda"),
        max_new_tokens = max_new_tokens,
        temperature = 1.0, top_p = 0.95, top_k = 128,
        streamer = TextStreamer(tokenizer, skip_prompt = True),
    )


do_gemma_3n_inference("Hi Gemma, how are you? What do you know about workplace safety?")


from PIL import Image
import numpy as np

image_test1 = Image.open("/kaggle/input/sh17-dataset-for-ppe-detection/images/pexels-photo-1018565.jpeg")
image_test1.resize((480,600))



from transformers import TextStreamer
# Helper function for inference
def do_gemma_3n_inference_mm(image, messages, max_new_tokens = 128):
    msg = [{"role" : "user","content": [
            { "type": "image", "image" : image },
            { "type": "text",  "text" : messages }
        ]}]
    _ = model.generate(
        **tokenizer.apply_chat_template(
            msg,
            add_generation_prompt = True, # Must add for generation
            tokenize = True,
            return_dict = True,
            return_tensors = "pt",
        ).to("cuda"),
        max_new_tokens = max_new_tokens,
        temperature = 0.3, top_p = 0.95, top_k = 64,
        streamer = TextStreamer(tokenizer, skip_prompt = True),
    )



from PIL import Image
import re

def detect_work_context(image, model, tokenizer):
    """
    Detecta contexto de trabajo
    """
    context_prompt = """Look at this image carefully and respond ONLY with one of these options:

INDUSTRIAL - If there's construction, factory, worksite, welding, machinery, heavy tools, building work
LABORATORY - If there are lab coats, medical/scientific equipment, sterile environment, chemical handling
OFFICE - If it's an office, home, casual indoor environment without work risks
OUTDOOR - If it's street, park, outdoor environment (could include outdoor work)

Respond with ONLY one word: INDUSTRIAL, LABORATORY, OFFICE, or OUTDOOR"""

    msg = [{"role" : "user","content": [
            { "type": "image", "image" : image },
            { "type": "text",  "text" : context_prompt }
        ]}]
    
    inputs = tokenizer.apply_chat_template(
        msg,
        add_generation_prompt = True,
        tokenize = True,
        return_dict = True,
        return_tensors = "pt",
    ).to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens = 20,
        temperature = 0.1,
        do_sample = False
    )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    response_clean = response.strip().upper()
    
    for category in ['INDUSTRIAL', 'LABORATORY', 'OFFICE', 'OUTDOOR']:
        if category in response_clean:
            return category
    
    return 'UNKNOWN'

def analyze_laboratory_safety(image, model, tokenizer):
    """
    Analiza seguridad especÃ­fica de laboratorio
    """
    prompt = """You are a laboratory safety expert. Analyze this laboratory image for safety violations.

CRITICAL SAFETY VIOLATIONS (These should result in VERY LOW scores 0-30):

1. CHEMICAL HANDLING WITHOUT GLOVES: Are people touching bottles, containers, or liquids with bare hands? This is EXTREMELY DANGEROUS - chemical burns, absorption through skin.

2. NO EYE PROTECTION: Are people handling chemicals, pouring liquids, or working with substances without safety glasses? This is EXTREMELY DANGEROUS - chemical splashes can cause blindness.

3. NO FACE PROTECTION: Are people pouring chemicals, especially from height or between containers, without face shields or masks? This is VERY DANGEROUS - inhalation of vapors, splashes to face.

4. IMPROPER CHEMICAL TRANSFER: Are people pouring liquids in unsafe ways, without proper containment, or in ways that could cause spills/splashes?

5. CONTAMINATION RISKS: Are people working without proper sterile technique in clean environments?

SCORING GUIDELINES:
- If you see people handling chemicals/liquids WITHOUT GLOVES = Score must be 0-20 (CRITICAL DANGER)
- If you see chemical work WITHOUT EYE PROTECTION = Score must be 0-30 (CRITICAL DANGER)  
- If multiple violations = Score should be 0-15 (EXTREME DANGER)
- Minor violations only = Score 40-60 (CAUTION)
- Proper protection visible = Score 80-100 (SAFE)

Look carefully at hands, eyes, face protection. Be STRICT about safety.

Respond in this exact format:
VIOLATIONS: [List specific violations you see, or "NONE" if safe]
SCORE: [Number from 0-100]
LEVEL: [DANGER/CAUTION/SAFE]"""

    msg = [{"role" : "user","content": [
            { "type": "image", "image" : image },
            { "type": "text",  "text" : prompt }
        ]}]
    
    inputs = tokenizer.apply_chat_template(
        msg,
        add_generation_prompt = True,
        tokenize = True,
        return_dict = True,
        return_tensors = "pt",
    ).to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens = 300,
        temperature = 0.2
    )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response

def analyze_industrial_safety(image, model, tokenizer):
    """
    Analiza seguridad industrial especÃ­fica
    """
    prompt = """You are an industrial safety expert. Analyze this construction/industrial image for safety violations.

CRITICAL SAFETY VIOLATIONS (These should result in VERY LOW scores 0-30):

1. NO HARD HAT/HELMET: Are people in construction areas, near machinery, or industrial sites without head protection? This is EXTREMELY DANGEROUS - falling objects can cause death.

2. WORK AT HEIGHT WITHOUT HARNESS: Are people on roofs, scaffolding, ladders, or elevated areas without safety harnesses or fall protection? This is EXTREMELY DANGEROUS - falls are a leading cause of workplace death.

3. NO SAFETY GLASSES: Are people using tools, near machinery, or in areas with flying debris without eye protection? This is VERY DANGEROUS - eye injuries are permanent.

4. UNSAFE POSITIONING: Are people working directly under heavy loads, in line with moving machinery, or in other dangerous positions?

5. NO HIGH-VISIBILITY CLOTHING: Are people in construction zones without bright/reflective clothing where machinery operates?

SCORING GUIDELINES:
- If you see people at height WITHOUT HARNESS = Score must be 0-15 (EXTREME DANGER)
- If you see people in hard hat zones WITHOUT HELMETS = Score must be 0-20 (CRITICAL DANGER)
- If multiple violations = Score should be 0-15 (EXTREME DANGER)
- Minor violations only = Score 40-60 (CAUTION)
- Proper protection visible = Score 80-100 (SAFE)

Look carefully at heads, harnesses, positioning. Be STRICT about safety.

Respond in this exact format:
VIOLATIONS: [List specific violations you see, or "NONE" if safe]
SCORE: [Number from 0-100]
LEVEL: [DANGER/CAUTION/SAFE]"""

    msg = [{"role" : "user","content": [
            { "type": "image", "image" : image },
            { "type": "text",  "text" : prompt }
        ]}]
    
    inputs = tokenizer.apply_chat_template(
        msg,
        add_generation_prompt = True,
        tokenize = True,
        return_dict = True,
        return_tensors = "pt",
    ).to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens = 300,
        temperature = 0.2
    )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response

def analyze_outdoor_safety(image, model, tokenizer):
    """
    Analiza seguridad en ambientes exteriores
    """
    prompt = """You are a workplace safety expert. Analyze this outdoor image to determine if there are work-related safety risks.

FIRST: Determine if this is WORK activity or CASUAL activity:
- WORK: Construction, tree cutting, roofing, maintenance, industrial outdoor work
- CASUAL: Walking, tourism, sports, leisure activities, photography

IF THIS IS CASUAL OUTDOOR ACTIVITY (no work being performed):
- Score should be 90-100 (SAFE)
- Respond with "CASUAL OUTDOOR - NO WORK RISKS"

IF THIS IS OUTDOOR WORK ACTIVITY, check for violations:

CRITICAL VIOLATIONS for outdoor work:
1. CONSTRUCTION AT HEIGHT WITHOUT HELMET: Building, roofing, scaffolding work without hard hats
2. TREE WORK WITHOUT PROTECTION: Chainsaw use, tree cutting without helmets, eye protection
3. ROOFING WITHOUT HARNESS: People on roofs without fall protection
4. HEAVY MACHINERY WITHOUT PROTECTION: Operating equipment without proper safety gear

SCORING GUIDELINES for work activities:
- Height work WITHOUT proper protection = Score 0-20 (CRITICAL DANGER)
- Tool use WITHOUT protection = Score 0-30 (CRITICAL DANGER)
- Proper protection visible = Score 80-100 (SAFE)

Respond in this exact format:
VIOLATIONS: [List specific violations you see, or "CASUAL OUTDOOR - NO WORK RISKS", or "NONE" if work but safe]
SCORE: [Number from 0-100]
LEVEL: [DANGER/CAUTION/SAFE]"""

    msg = [{"role" : "user","content": [
            { "type": "image", "image" : image },
            { "type": "text",  "text" : prompt }
        ]}]
    
    inputs = tokenizer.apply_chat_template(
        msg,
        add_generation_prompt = True,
        tokenize = True,
        return_dict = True,
        return_tensors = "pt",
    ).to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens = 300,
        temperature = 0.2
    )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response

def parse_safety_response(response):
    """
    Parsea la respuesta de Gemma y extrae datos
    """
    # Parse violations
    violations_match = re.search(r'VIOLATIONS:\s*(.+?)(?=SCORE|$)', response, re.DOTALL)
    violations = violations_match.group(1).strip() if violations_match else "Unknown"
    
    # Parse score
    score_match = re.search(r'SCORE:\s*(\d+)', response)
    score = int(score_match.group(1)) if score_match else 50
    
    # Parse level
    level_match = re.search(r'LEVEL:\s*(DANGER|CAUTION|SAFE)', response)
    level = level_match.group(1) if level_match else 'CAUTION'
    
    # Force score consistency with level
    if "CRITICAL DANGER" in violations or "EXTREME DANGER" in violations:
        score = min(score, 20)
        level = "DANGER"
    elif "DANGER" in level and score > 40:
        score = min(score, 30)
    elif "CAUTION" in level and score > 79:
        score = min(score, 70)
    
    # Determine emoji and final assessment
    if score >= 80:
        emoji = 'âœ…'
        final_level = 'SAFE'
        is_dangerous = False
    elif score >= 40:
        emoji = 'âš ï¸�'
        final_level = 'CAUTION'
        is_dangerous = True
    else:
        emoji = 'ğŸ”¥'
        final_level = 'DANGER'
        is_dangerous = True
    
    return {
        'violations': violations,
        'score': score,
        'level': final_level,
        'emoji': emoji,
        'is_dangerous': is_dangerous,
        'raw_response': response
    }

def analyze_image_safety(image_path, model, tokenizer):
    """
    Analiza una imagen completa
    """
    image = Image.open(image_path)
    
    # Step 1: Detect context
    context = detect_work_context(image, model, tokenizer)
    
    # Step 2: Analyze safety based on context
    if context == 'OFFICE':
        return {
            'context': context,
            'violations': 'Office environment - No PPE required',
            'score': 100,
            'level': 'SAFE',
            'emoji': 'âœ…',
            'is_dangerous': False,
            'raw_response': 'Office environment analysis'
        }
    elif context == 'LABORATORY':
        response = analyze_laboratory_safety(image, model, tokenizer)
        return {**parse_safety_response(response), 'context': context}
    elif context == 'INDUSTRIAL':
        response = analyze_industrial_safety(image, model, tokenizer)
        return {**parse_safety_response(response), 'context': context}
    elif context == 'OUTDOOR':
        response = analyze_outdoor_safety(image, model, tokenizer)
        return {**parse_safety_response(response), 'context': context}
    else:
        return {
            'context': 'UNKNOWN',
            'violations': 'Could not determine context',
            'score': 50,
            'level': 'CAUTION',
            'emoji': 'âš ï¸�',
            'is_dangerous': True,
            'raw_response': 'Unknown context'
        }

def analyze_five_images_clean(model, tokenizer):
    """
    Analiza las 5 imÃ¡genes especÃ­ficas
    """
    image_paths = [
        "/kaggle/input/sh17-dataset-for-ppe-detection/images/pexels-photo-1018565.jpeg",
        "/kaggle/input/sh17-dataset-for-ppe-detection/images/pexels-photo-11293626.jpeg",
        # "/kaggle/input/sh17-dataset-for-ppe-detection/images/pexels-photo-11977314.jpeg",
        # "/kaggle/input/sh17-dataset-for-ppe-detection/images/pexels-photo-11930042.jpeg",
        # "/kaggle/input/sh17-dataset-for-ppe-detection/images/pexels-photo-11784496.jpeg"
    ]
    
    print("ğŸ”� SAFETY ANALYSIS - 5 IMAGES")
    print("="*60)
    
    results = []
    dangerous_count = 0
    
    for i, image_path in enumerate(image_paths):
        print(f"\n--- IMAGE {i+1}/5 ---")
        print(f"ğŸ“„ File: {image_path.split('/')[-1]}")
        
        try:
            result = analyze_image_safety(image_path, model, tokenizer)
            results.append(result)
            
            print(f"ğŸ“� Context: {result['context']}")
            print(f"ğŸ�¯ Level: {result['level']} {result['emoji']}")
            print(f"ğŸ“Š Score: {result['score']}/100")
            print(f"ğŸ“� Violations: {result['violations']}")
            
            if result['is_dangerous']:
                dangerous_count += 1
                print(f"ğŸš¨ FLAGGED AS DANGEROUS")
            
            print(f"\nğŸ¤– Full AI Response:")
            print(result['raw_response'])
            
        except Exception as e:
            print(f"â�Œ Error: {e}")
    
    print(f"\nğŸ“Š FINAL SUMMARY:")
    print(f"   Total analyzed: {len(results)}")
    print(f"   Dangerous images: {dangerous_count}")
    
    if dangerous_count > 0:
        print(f"\nğŸš¨ DANGEROUS IMAGES:")
        for i, result in enumerate(results):
            if result.get('is_dangerous', False):
                filename = image_paths[i].split('/')[-1]
                print(f"   â€¢ {filename}: {result['level']} {result['emoji']} ({result['score']}/100)")
    
    return results

def run_clean_analysis(model, tokenizer):
    """
    Ejecuta el anÃ¡lisis limpio
    """
    return analyze_five_images_clean(model, tokenizer)


# Ejecutar anÃ¡lisis de las 5 imÃ¡genes
results = run_clean_analysis(model, tokenizer)


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# Image paths
image_paths = [
    "/kaggle/input/sh17-dataset-for-ppe-detection/images/pexels-photo-1018565.jpeg",
    "/kaggle/input/sh17-dataset-for-ppe-detection/images/pexels-photo-11293626.jpeg",
    "/kaggle/input/sh17-dataset-for-ppe-detection/images/pexels-photo-11977314.jpeg",
    "/kaggle/input/sh17-dataset-for-ppe-detection/images/pexels-photo-11930042.jpeg",
    "/kaggle/input/sh17-dataset-for-ppe-detection/images/pexels-photo-11784496.jpeg"
]

def load_images(paths, target_size=(480, 600)):
    """
    Load images with PIL and resize them
    """
    images = {}
    
    for i, path in enumerate(paths):
        try:
            # Load image
            img = Image.open(path)
            
            # Resize image
            img_resized = img.resize(target_size)
            
            # Store with simple name
            img_name = f"image_{i+1}"
            images[img_name] = {
                'original': img,
                'resized': img_resized,
                'path': path,
                'filename': os.path.basename(path)
            }
            
            print(f"âœ… Loaded {img_name}: {images[img_name]['filename']}")
            print(f"   Original size: {img.size}")
            print(f"   Resized to: {img_resized.size}")
            
        except Exception as e:
            print(f"â�Œ Error loading {path}: {e}")
    
    return images

def display_images(images, cols=3):
    """
    Display all loaded images in a grid
    """
    num_images = len(images)
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    
    # Handle single row case
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, (name, img_data) in enumerate(images.items()):
        if i < len(axes):
            axes[i].imshow(img_data['resized'])
            axes[i].set_title(f"{name}\n{img_data['filename']}")
            axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Load all images
print("ğŸ–¼ï¸�  LOADING TEST IMAGES")
print("="*50)

loaded_images = load_images(image_paths)

print(f"\nğŸ“Š SUMMARY:")
print(f"   Total images loaded: {len(loaded_images)}")
print(f"   Available as: {list(loaded_images.keys())}")

# Display images
print(f"\nğŸ�¨ DISPLAYING IMAGES:")
display_images(loaded_images)

# Individual image variables for easy access
image_1 = loaded_images['image_1']['resized']
image_2 = loaded_images['image_2']['resized'] 
image_3 = loaded_images['image_3']['resized']
image_4 = loaded_images['image_4']['resized']
image_5 = loaded_images['image_5']['resized']

print(f"\nâœ… Individual images available as:")
print(f"   image_1, image_2, image_3, image_4, image_5")

# For compatibility with your existing code
image_test1 = image_1

print(f"\nğŸ”„ COMPATIBILITY:")
print(f"   image_test1 = image_1 (for existing code)")


import json
import os
import random
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any

class PPEDatasetGenerator:
    """
    Generador de ejemplos de entrenamiento para Gemma 3n usando dataset SH17
    """
    
    def __init__(self, base_path: str = "/kaggle/input/sh17-dataset-for-ppe-detection"):
        self.base_path = Path(base_path)
        self.images_path = self.base_path / "images"
        self.labels_path = self.base_path / "labels" 
        self.metadata_path = self.base_path / "meta-data"
        
        # Mapeo de clases
        self.classes = {
            0: "person", 1: "head", 2: "face", 3: "glasses", 
            4: "face-mask-medical", 5: "face-guard", 6: "ear", 
            7: "earmuffs", 8: "hands", 9: "gloves", 10: "foot", 
            11: "shoes", 12: "safety-vest", 13: "tools", 
            14: "helmet", 15: "medical-suit", 16: "safety-suit"
        }
        
        # Cargar splits
        self.train_files = self._load_split_files("train_files.txt")
        self.val_files = self._load_split_files("val_files.txt")
    
    def _load_split_files(self, filename: str) -> List[str]:
        """Carga los archivos de split"""
        split_file = self.base_path / filename
        if split_file.exists():
            with open(split_file, 'r') as f:
                return [line.strip() for line in f.readlines()]
        return []
    
    def convert_yolo_to_spatial(self, x_center: float, y_center: float, width: float, height: float) -> str:
        """Convierte coordenadas YOLO a descripciÃ³n espacial"""
        if x_center < 0.33:
            h_pos = "izquierda"
        elif x_center > 0.66:
            h_pos = "derecha"
        else:
            h_pos = "centro"
        
        if y_center < 0.33:
            v_pos = "arriba"
        elif y_center > 0.66:
            v_pos = "abajo"
        else:
            v_pos = "centro"
        
        if h_pos == "centro" and v_pos == "centro":
            return "en el centro"
        elif h_pos == "centro":
            return f"en la parte {v_pos}"
        elif v_pos == "centro":
            return f"a la {h_pos}"
        else:
            return f"en la parte {v_pos} {h_pos}"
    
    def parse_annotation_file(self, image_name: str) -> List[Dict]:
        """Parsea el archivo de anotaciones YOLO"""
        annotation_file = self.labels_path / f"{Path(image_name).stem}.txt"
        objects = []
        
        if annotation_file.exists():
            with open(annotation_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])
                        
                        objects.append({
                            "class_id": class_id,
                            "class_name": self.classes.get(class_id, f"unknown_{class_id}"),
                            "position": self.convert_yolo_to_spatial(x_center, y_center, width, height),
                            "coordinates": [x_center, y_center, width, height]
                        })
        
        return objects
    
    def load_metadata(self, image_name: str) -> Dict:
        """Carga metadata de la imagen"""
        metadata_file = self.metadata_path / f"{Path(image_name).stem}.json"
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def generate_prompt_for_llm(self, image_name: str, question_type: str = "safety_check") -> str:
        """
        Genera el prompt para enviar al LLM externo (Gemini/Claude)
        """
        objects = self.parse_annotation_file(image_name)
        metadata = self.load_metadata(image_name)
        
        # Extraer informaciÃ³n relevante
        alt_text = metadata.get('alt', 'Imagen industrial')
        photographer = metadata.get('photographer', 'Desconocido')
        avg_color = metadata.get('avg_color', '#000000')
        
        # Construir descripciÃ³n de objetos detectados
        objects_description = []
        for obj in objects:
            objects_description.append(f"- {obj['class_name']} {obj['position']}")
        
        # Templates de preguntas segÃºn el tipo
        question_templates = {
            "safety_check": "Â¿Es segura esta escena industrial?",
            "compliance": "Â¿Los trabajadores cumplen con las normas de seguridad?",
            "ppe_detection": "Describe todos los elementos de protecciÃ³n personal presentes",
            "risk_assessment": "Â¿QuÃ© riesgos de seguridad identificas en esta imagen?",
            "equipment_count": "Cuenta y describe los equipos de seguridad visibles"
        }
        
        prompt = f"""
Eres un experto en seguridad industrial. BasÃ¡ndote en la siguiente informaciÃ³n, genera un ejemplo de entrenamiento para un modelo de IA que monitorearÃ¡ seguridad en tiempo real.

INFORMACIÃ“N DE LA IMAGEN:
- DescripciÃ³n: {alt_text}
- Objetos detectados:
{chr(10).join(objects_description)}
- FotÃ³grafo: {photographer}
- Tono de color: {avg_color}

PREGUNTA OBJETIVO: {question_templates.get(question_type, question_templates["safety_check"])}

GENERA UN EJEMPLO EN ESTE FORMATO JSON:
{{
    "question": "pregunta especÃ­fica sobre seguridad",
    "answer": "respuesta concisa y precisa (mÃ¡ximo 2-3 oraciones)"
}}

La respuesta debe ser:
- Directa y prÃ¡ctica para monitoreo en tiempo real
- Enfocada en compliance de PPE
- Ãštil para alertas automÃ¡ticas
- Sin jerga tÃ©cnica innecesaria

EJEMPLO JSON:
"""
        
        return prompt
    
    def process_sample_images(self, num_samples: int = 5) -> List[Dict]:
        """
        Procesa algunas imÃ¡genes de muestra para testing
        """
        sample_files = self.train_files[:num_samples] if self.train_files else []
        results = []
        
        question_types = ["safety_check", "compliance", "ppe_detection", "risk_assessment"]
        
        for i, image_file in enumerate(sample_files):
            question_type = question_types[i % len(question_types)]
            
            prompt = self.generate_prompt_for_llm(image_file, question_type)
            
            # Info para debug
            objects = self.parse_annotation_file(image_file)
            metadata = self.load_metadata(image_file)
            
            results.append({
                "image_path": str(self.images_path / image_file),
                "image_name": image_file,
                "prompt_for_llm": prompt,
                "objects_detected": objects,
                "metadata": metadata,
                "question_type": question_type
            })
        
        return results
    
    def save_prompts_for_llm(self, output_file: str = "prompts_for_llm.json"):
        """
        Genera prompts para todas las imÃ¡genes y los guarda para procesamiento con LLM externo
        """
        all_prompts = []
        
        print(f"Procesando {len(self.train_files)} imÃ¡genes de entrenamiento...")
        
        question_types = ["safety_check", "compliance", "ppe_detection", "risk_assessment", "equipment_count"]
        
        for i, image_file in enumerate(self.train_files):
            if i % 1000 == 0:
                print(f"Procesado {i}/{len(self.train_files)} imÃ¡genes...")
            
            # Rotar tipos de preguntas
            question_type = question_types[i % len(question_types)]
            
            prompt = self.generate_prompt_for_llm(image_file, question_type)
            
            all_prompts.append({
                "id": i,
                "image_path": f"/kaggle/input/sh17-dataset-for-ppe-detection/images/{image_file}",
                "image_name": image_file,
                "prompt": prompt,
                "question_type": question_type
            })
        
        # Guardar prompts
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_prompts, f, indent=2, ensure_ascii=False)
        
        print(f"âœ… Prompts guardados en {output_file}")
        print(f"ğŸ“Š Total de ejemplos: {len(all_prompts)}")
        
        return all_prompts






import google.generativeai as genai
from kaggle_secrets import UserSecretsClient
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import asyncio
import json
from pathlib import Path
from PIL import Image
import logging
from tqdm.notebook import tqdm

# --- 1. ESTRUCTURA DE DATOS ACTUALIZADA ---
@dataclass
class GeneratedPair:
    """Representa un conjunto de preguntas y respuestas generadas para una imagen."""
    image_name: str
    description_question: str
    description_answer: str
    ppe_question: str
    ppe_answer: str

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GeminiDatasetCreator:
    MODEL_NAME = "gemini-2.5-flash"
    MAX_RETRIES = 3
    RETRY_DELAY_SECONDS = 5

    def __init__(self, concurrency: int = 4):
        self.api_key = self._load_api_key_from_kaggle()
        if not self.api_key:
            raise ValueError("No se encontrÃ³ la API key 'GEMINI_API_KEY' en Kaggle Secrets.")
        genai.configure(api_key=self.api_key)
        self.semaphore = asyncio.Semaphore(concurrency)
        logging.info(f"Gemini Creator inicializado. Concurrencia limitada a {concurrency}.")

    def _load_api_key_from_kaggle(self) -> Optional[str]:
        try:
            user_secrets = UserSecretsClient()
            return user_secrets.get_secret("GEMINI_API_KEY_1")
        except Exception as e:
            logging.error(f"No se pudo cargar la API key 'GEMINI_API_KEY': {e}")
            return None

    async def _process_single_item_async(
        self, 
        item_data: Dict[str, Any], 
        prompt_template: str
    ) -> Optional[GeneratedPair]:
        image_path = Path(item_data['image_path'])
        if not image_path.exists():
            logging.warning(f"La imagen no existe: {image_path}")
            return None

        async with self.semaphore:
            formatted_prompt = prompt_template.format(
                alt_text=item_data['metadata'].get('alt', 'a work scene'),
                objects_detected=", ".join([obj['class_name'] for obj in item_data['objects_detected']])
            )
            for attempt in range(self.MAX_RETRIES):
                try:
                    model = genai.GenerativeModel(self.MODEL_NAME)
                    image = Image.open(image_path)
                    response = await model.generate_content_async([image, formatted_prompt])
                    cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
                    data = json.loads(cleaned_response)
                    
                    # --- 2. MAPEO A LA NUEVA ESTRUCTURA DE DATOS ---
                    return GeneratedPair(
                        image_name=image_path.name,
                        description_question=data['description_question'],
                        description_answer=data['description_answer'],
                        ppe_question=data['ppe_question'],
                        ppe_answer=data['ppe_answer']
                    )
                except (json.JSONDecodeError, KeyError) as e:
                     logging.error(f"Error de formato JSON o clave faltante en la respuesta para {image_path.name}: {e}")
                     # No reintentar por errores de formato, pasar al siguiente
                     break
                except Exception as e:
                    logging.error(f"Error con {image_path.name} (intento {attempt + 1}): {e}")
                    if attempt < self.MAX_RETRIES - 1:
                        await asyncio.sleep(self.RETRY_DELAY_SECONDS)
            return None

    async def create_dataset_from_data(
        self, 
        data_items: List[Dict[str, Any]], 
        prompt_template: str,
        output_filename: str
    ):
        logging.info(f"Iniciando la generaciÃ³n de dataset para {len(data_items)} items.")
        tasks = [self._process_single_item_async(item, prompt_template) for item in data_items]
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write('[\n')
            first_item = True
            for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generando Dataset"):
                result = await task
                if result:
                    if not first_item:
                        f.write(',\n')
                    json.dump(asdict(result), f, indent=4, ensure_ascii=False)
                    f.flush()
                    first_item = False
            f.write('\n]')
        logging.info(f"Proceso completado. Dataset guardado en '{output_filename}'.")

# --- 3. EJECUCIÃ“N ---

# AsegÃºrate de que la clase PPEDatasetGenerator estÃ© definida.
generator = PPEDatasetGenerator()
sample_data_items = generator.process_sample_images(num_samples=50) 

# --- Usamos el nuevo prompt en inglÃ©s ---
prompt_template_for_generation = """
You are an expert in industrial safety creating a training dataset for an AI.
Analyze the provided image and information to generate two distinct question-and-answer pairs.

IMAGE INFORMATION:
- General description from metadata: {alt_text}
- Detected objects: {objects_detected}

YOUR TASK:
Generate a single, valid JSON object containing two pairs of questions and answers based on the image:

1.  **Full Scene Description:**
    - `description_question`: A question asking for a detailed description of the scene, the people, and their actions.
    - `description_answer`: A comprehensive, detailed answer describing the entire scene.

2.  **Safety and Risk Analysis:**
    - `ppe_question`: A question asking for a full safety analysis, including PPE compliance and any environmental or behavioral risks.
    - `ppe_answer`: A detailed answer that first addresses PPE use, and then identifies any other hazards. For example: "No, the worker is not wearing a hard hat. Additionally, the environment is hazardous due to unstable footing on loose gravel and the proximity to heavy machinery." If the scene is safe, state that.

**IMPORTANT**: Respond ONLY with a single, valid JSON object. Do not include any other text, markdown, or explanations.

EXAMPLE OUTPUT FORMAT:
{{
    "description_question": "Can you describe the scene in this image in detail?",
    "description_answer": "The image shows a construction worker on a roof. The worker is using a nail gun to install shingles. The weather appears sunny, and there are other construction materials visible in the background.",
    "ppe_question": "Provide a full safety analysis of this scene, including PPE compliance and any other risks.",
    "ppe_answer": "The worker is not wearing appropriate fall protection like a harness, which is a critical risk. The sloped surface of the roof also presents a significant hazard for slips and falls."
}}
"""

async def build_rich_dataset():
    output_filename = "ppe_rich_training_dataset.json"
    creator = GeminiDatasetCreator(concurrency=4)
    await creator.create_dataset_from_data(
        data_items=sample_data_items,
        prompt_template=prompt_template_for_generation,
        output_filename=output_filename
    )
    print(f"\nâœ… Proceso finalizado. Revisa el archivo '{output_filename}'.")

# Ejecutar en el notebook
await build_rich_dataset()


import json
import random
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import textwrap

def display_qa_with_images(
    json_filename: str, 
    images_base_path: str, 
    num_examples: int = 2
):
    """
    Carga un dataset JSON y muestra ejemplos aleatorios con sus imÃ¡genes,
    asegurando que el texto no se superponga con la imagen.
    """
    try:
        with open(json_filename, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"Error: No se encontrÃ³ el archivo '{json_filename}'.")
        return
    except json.JSONDecodeError:
        print(f"Error: El archivo '{json_filename}' no es un JSON vÃ¡lido.")
        return

    if not dataset:
        print("El dataset estÃ¡ vacÃ­o.")
        return

    images_path = Path(images_base_path)
    if not images_path.exists():
        print(f"Error: La carpeta de imÃ¡genes '{images_base_path}' no existe.")
        return

    selected_examples = random.sample(dataset, min(num_examples, len(dataset)))

    for example in selected_examples:
        image_name = example.get('image_name')
        if not image_name:
            continue

        image_file_path = images_path / image_name
        if not image_file_path.exists():
            print(f"Imagen no encontrada: {image_file_path}")
            continue

        img = Image.open(image_file_path)
        fig, ax = plt.subplots(figsize=(10, 12))
        ax.imshow(img)
        ax.axis('off')
        
        # Usamos los nombres de clave correctos del Ãºltimo prompt que creamos
        desc_q = textwrap.fill(f"Q (Desc): {example.get('description_question', 'N/A')}", width=80)
        desc_a = textwrap.fill(f"A (Desc): {example.get('description_answer', 'N/A')}", width=80)
        risk_q = textwrap.fill(f"Q (Risk): {example.get('ppe_question', 'N/A')}", width=80)
        risk_a = textwrap.fill(f"A (Risk): {example.get('ppe_answer', 'N/A')}", width=80)
        
        full_text = (
            f"--- 1. Scene Description ---\n{desc_q}\n{desc_a}\n\n"
            f"--- 2. Risk Analysis ---\n{risk_q}\n{risk_a}"
        )
        
        fig.suptitle(f"Ejemplo: {image_name}", fontsize=16, weight='bold')
        
        # --- CAMBIOS CLAVE AQUÃ� ---
        # 1. Ajustamos el espacio inferior para dejar sitio al texto. 
        #    Esto reserva el 30% inferior de la figura para el texto.
        plt.subplots_adjust(bottom=0.3)

        # 2. Colocamos el texto en el espacio que acabamos de reservar.
        #    Lo centramos verticalmente en ese 30% (en y=0.15)
        plt.figtext(0.5, 0.15, full_text, ha="center", va="top", fontsize=12, wrap=True,
                    bbox={"facecolor":"lightgrey", "alpha":0.5, "pad":5})
        
        # 3. Eliminamos tight_layout, ya que ahora controlamos el layout manualmente.
        plt.show()



# El archivo JSON que creaste con el prompt de anÃ¡lisis de riesgo
dataset_filename = "/kaggle/working/ppe_rich_training_dataset.json" 

# La carpeta de imÃ¡genes original
images_folder_path = "/kaggle/input/sh17-dataset-for-ppe-detection/images"

# Llamar a la funciÃ³n corregida
display_qa_with_images(
    json_filename=dataset_filename,
    images_base_path=images_folder_path,
    num_examples=4
)


# Configure LoRA for vision fine-tuning
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=True,    # Enable vision layer fine-tuning
    finetune_language_layers=True,  # Enable language layer fine-tuning
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=2,                          # LoRA rank
    lora_alpha=4,                 # LoRA alpha
    lora_dropout=0,
    bias="none",
    random_state=3407,
    target_modules="all-linear",
    modules_to_save=["lm_head", "embed_tokens"],
)


import json
from pathlib import Path
from datasets import Dataset, DatasetDict

# --- 1. Definir las rutas ---
# La ruta a tu archivo JSON generado
json_filepath = "/kaggle/working/ppe_rich_training_dataset.json"

# La ruta a la carpeta donde estÃ¡n las imÃ¡genes originales
images_base_path = Path("/kaggle/input/sh17-dataset-for-ppe-detection/images")


# --- 2. Cargar y procesar el JSON en una lista ---
print(f"Cargando y procesando el archivo: {json_filepath}")

processed_data = []
try:
    with open(json_filepath, 'r', encoding='utf-8') as f:
        source_data = json.load(f)
except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"â�Œ Error al cargar el archivo JSON: {e}")
    # Detener la ejecuciÃ³n si el archivo no se puede cargar
    source_data = []

for item in source_data:
    image_path = images_base_path / item.get('image_name', '')
    
    # Solo procesar si la imagen realmente existe
    if image_path.exists():
        # Ejemplo 1: Pregunta y respuesta de DESCRIPCIÃ“N
        processed_data.append({
            "image": str(image_path),
            "question": item.get('description_question'),
            "answer": item.get('description_answer')
        })
        
        # Ejemplo 2: Pregunta y respuesta de RIESGO
        processed_data.append({
            "image": str(image_path),
            "question": item.get('risk_question'),
            "answer": item.get('risk_answer')
        })

print(f"Procesamiento completado. Se generaron {len(processed_data)} ejemplos de entrenamiento a partir de {len(source_data)} imÃ¡genes.")


# --- 3. Crear un Dataset de Hugging Face ---
# Convertimos nuestra lista de diccionarios en un objeto Dataset
hf_dataset = Dataset.from_list(processed_data)

print("\nEstructura del dataset inicial:")
print(hf_dataset)


# --- 4. Definir la funciÃ³n de formateo (la que proporcionaste) ---
def format_for_finetuning(examples):
    texts = []
    # La columna de imagen en el dataset de HF se llama "image"
    # Asegurarse de que la ruta de la imagen se maneje correctamente si es necesario,
    # pero para el formato de texto, solo necesitamos las preguntas y respuestas.
    for i in range(len(examples["question"])):
        # El placeholder <image> le dice al modelo que mire la imagen asociada a este texto.
        text = f"""<|user|>
<image>
{examples["question"][i]}
<|assistant|>
{examples["answer"][i]}<|end_of_text|>"""
        texts.append(text)
    return {"text": texts}


# --- 5. Aplicar el formateo al dataset ---
# Usamos .map() para aplicar la funciÃ³n a todos los ejemplos.
# `batched=True` lo hace mucho mÃ¡s rÃ¡pido.
# `remove_columns` limpia el dataset, dejando solo la columna "text" y la de "image" que el trainer usarÃ¡.
formatted_dataset = hf_dataset.map(
    format_for_finetuning, 
    batched=True, 
    remove_columns=["question", "answer"]
)

print("\nEstructura del dataset formateado:")
print(formatted_dataset)


# --- 6. (Recomendado) Dividir en entrenamiento y prueba ---
# Es una buena prÃ¡ctica tener un conjunto de prueba para evaluar el modelo.
# test_size=0.1 significa que el 10% de los datos se usarÃ¡ para pruebas.
final_dataset_dict = formatted_dataset.train_test_split(test_size=0.1, seed=42)

print("\nDataset final dividido en entrenamiento y prueba:")
print(final_dataset_dict)

# Ahora puedes acceder a tus datasets asÃ­:
train_dataset = final_dataset_dict["train"]
test_dataset = final_dataset_dict["test"]


# --- 7. Verificar el resultado final ---
print("\n--- Ejemplo de un dato de entrenamiento formateado ---")
# Mostramos el primer ejemplo del conjunto de entrenamiento
# La columna 'image' sigue presente con la ruta al archivo de imagen
print("Ruta de la imagen:", train_dataset[0]['image'])
# La columna 'text' tiene el formato que necesitas
print("Texto formateado:\n" + train_dataset[0]['text'])


from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=final_dataset_dict["train"],
    dataset_text_field="text",
    max_seq_length=2048,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=3,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        save_strategy="steps",
        save_steps=30,
    ),
)

# Start training
trainer_stats = trainer.train()




