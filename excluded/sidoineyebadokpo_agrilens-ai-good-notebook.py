!pip install bitsandbytes -q



# ===================================================================
# SIDOINE - GEMMA 3N GENERIC TOKENIZER APPROACH
# Utilise un tokenizer générique si GemmaTokenizer n'est pas disponible
# ===================================================================

# Step 1: Install necessary libraries (including missing dependencies)
print("\nStep 1: Installing libraries...\n")
!pip install --upgrade -q \
  transformers \
  timm accelerate \
  torchao triton \
  torchaudio # Ajout de torchaudio pour la Cellule 3
print("\n--- Libraries are ready.")

# Step 2: Import all necessary libraries
print("\nStep 2: Importing libraries...")
import torch
import os  # Import the os module
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, GenerationConfig
print("✅ Libraries imported.")

# Step 3: Check CUDA availability
print("\nStep 3: Checking CUDA availability...")

# --- FORÇAGE CPU OBLIGATOIRE ---
# En raison des erreurs CUDA et des redémarrages de kernel, on force l'utilisation du CPU.
device = torch.device('cpu')
print("❌ GPU operations disabled, forcing CPU usage.")
# -------------------------------

# Step 4: Define model path and verify
GEMMA_PATH = '/kaggle/input/gemma-3n/transformers/gemma-3n-e2b-it/1'
if not os.path.exists(GEMMA_PATH):
  raise FileNotFoundError(f"Model path not found: {GEMMA_PATH}") # Added GEMMA_PATH to the error message for clarity
print(f"✅ Model path: {GEMMA_PATH}") # Added GEMMA_PATH to the print statement for clarity

# Step 5: Attempt loading model and processor with fallback
print("\nStep 5: Loading tokenizer, processor & model...\n")
success = False
tokenizer = None
processor = None
model = None

def load_model_and_processor(token_args, model_args, proc_args):
  global tokenizer, model, processor

  print("🔄 Loading tokenizer...")
  tokenizer = AutoTokenizer.from_pretrained(
    GEMMA_PATH, local_files_only=True, trust_remote_code=True, **token_args
  )
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
  print("✅ Tokenizer loaded.")

  print("🔄 Loading processor...")
  processor = AutoProcessor.from_pretrained(
    GEMMA_PATH, local_files_only=True, trust_remote_code=True, **proc_args
  )
  print("✅ Processor loaded.")

  print("🔄 Loading model...")
  model = AutoModelForCausalLM.from_pretrained(
    GEMMA_PATH,
    local_files_only=True,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    # REMOVED: low_cpu_mem_usage=True, # This was causing the "multiple values" error
    **model_args # 'low_cpu_mem_usage' will now be passed from here if present in model_args
  )
  model.to(device) # Déplacement explicite sur CPU
  print(f"✅ Model loaded and moved to {device}.") # Added device to the print statement

# --- Tentatives de chargement simplifiées ---

# 5.1 Primary attempt (simplified)
try:
  print("\n--- Attempting Primary Load (Simplified) ---")
  print("Config: float16, no device_map, left padding")
  load_model_and_processor(
    token_args={'use_fast':False,'padding_side':'left'},
    model_args={'low_cpu_mem_usage':True}, # This is where low_cpu_mem_usage is now exclusively passed
    proc_args={}
  )
  success=True
  print("✅ Primary load succeeded.")
except Exception as e:
  print(f"❌ Primary load failed: {e}") # Added exception details to the print statement
  # 5.2 Fallback attempt (simplified)
  try:
    print("\n--- Attempting Fallback Load (Simplified) ---")
    print("Config: float16, no device_map, right padding, ignore sizes")
    load_model_and_processor(
      token_args={'use_fast':False,'padding_side':'right','truncation_side':'right'},
      model_args={'low_cpu_mem_usage':True, 'ignore_mismatched_sizes':True}, # low_cpu_mem_usage and ignore_mismatched_sizes are passed here
      proc_args={}
    )
    success=True
    print("✅ Fallback load succeeded.")
  except Exception as e2:
    print(f"❌ All attempts failed: {e2}") # Added exception details to the print statement

# Step 6: Setup generation if loaded
if success:
  if tokenizer.pad_token_id is None:
    if tokenizer.eos_token_id is not None:
      tokenizer.pad_token_id = tokenizer.eos_token_id
      print(f"ℹ️ Tokenizer's pad_token_id set to eos_token_id ({tokenizer.eos_token_id})")
    else:
      print("❌ WARNING: Neither pad_token nor eos_token is available. Generation might fail.")

  # Generation config: On maintient des max_new_tokens plus bas pour les tests
  generation_config = GenerationConfig(
    max_new_tokens=128, # Gardé bas pour tester la charge mémoire
    do_sample=True,
    temperature=0.7,
    pad_token_id=tokenizer.pad_token_id
  )
  print("\nStep 6: GenerationConfig ready.")
  print("🎉 Model and Processor loaded successfully!")
else:
  print("\n❌ Could not load model and processor after all attempts.")

# Optional test function (pour vérifier si tout fonctionne bien)
if success:
  def test_model_and_processor():
    print("\n--- Performing a quick test generation ---")
    text = "Hello, world!"
    try:
      inputs = tokenizer(text, return_tensors='pt').to(device) # Utilise le device CPU
      print(f"Input tokens shape: {inputs['input_ids'].shape}")

      with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20, generation_config=generation_config) # Exécution sur CPU

      decoded_output = tokenizer.decode(out[0], skip_special_tokens=True)
      print(f"Test Output: {decoded_output}") # Added decoded_output to the print statement
      print("✅ Test generation successful.")
    except Exception as e:
      print(f"❌ Test generation failed: {e}") # Added exception details to the print statement

  # Call the test function
  test_model_and_processor()


# Cellule 2 : Test de génération de texte du modèle Gemma

import torch
from PIL import Image
import os # Assurer que os est importé si utilisé ici

# --- Vérification de la disponibilité des objets globaux ---
# Assurez-vous que les variables globales model et tokenizer ont été définies dans la Cellule 1
if 'model' not in globals() or 'tokenizer' not in globals() or model is None or tokenizer is None:
    raise RuntimeError("Model or tokenizer not available. Please run Cell 1 successfully first.")

print("\n--- Starting AgriLens AI Text Generation Test (using globally loaded model) ---")

# --- Utilisation de la mémoire ---
def print_memory(stage=""):
  if torch.cuda.is_available() and device.type == 'cuda': # Vérifie si le device est CUDA
    allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
    print(f"Memory ({stage}): Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
  else:
    # Pour CPU, on peut afficher l'utilisation de la RAM système, mais c'est plus complexe.
    # Pour simplifier, on affiche juste qu'on est sur CPU.
    print(f"Memory ({stage}): Using CPU.")

# Clear cache to free up memory (optional, but good practice if something went wrong before)
if torch.cuda.is_available() and device.type == 'cuda':
  torch.cuda.empty_cache()
  print("✅ CUDA cache cleared.")

# Print the start of the text generation test
print_memory("Before Text Gen (Initial State)")

# Define the path to the pre-trained model (just for context, not for loading again)
model_path = "/kaggle/input/gemma-3n/transformers/gemma-3n-e2b-it/1"

try:
  # --- Utilisation du modèle et tokenizer globaux ---
  # On récupère le device sur lequel le modèle global est déjà chargé.
  current_device = next(model.parameters()).device
  print(f"Using model on device: {current_device}")

  # Définir le prompt agricole
  prompt_agri = """
  Provide a detailed description of the common symptoms of late blight disease in tomato plants.
  Include information on the appearance, progression, and specific parts of the plant affected.
  Also, describe any environmental conditions that favor the development of this disease.
  """

  # Tokenize the input using the global tokenizer
  # On s'assure que les inputs sont sur le même device que le modèle
  inputs_agri = tokenizer(prompt_agri, return_tensors="pt").to(current_device)

  # Print memory usage after moving inputs to device
  print_memory("After Inputs To Device")

  # Générer la réponse en utilisant le modèle global
  print("Generating text...")
  with torch.no_grad():
    outputs_agri = model.generate(
        **inputs_agri,
        max_new_tokens=300, # Ajusté pour un test plus court
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
        )

  # Print memory usage after text generation
  print_memory("After Text Gen")

  # Decode and print the result using the global tokenizer
  result_agri = tokenizer.decode(outputs_agri[0], skip_special_tokens=True)
  print("\n--- Model Output for Agricultural Query ---")
  print(result_agri)

except Exception as e:
  print(f"An error occurred during text generation: {type(e).__name__} -> {e}")


# Cell 3: Launching AgriLens AI Image+Text Diagnostic (using pipeline)

import torch
from PIL import Image
from transformers import pipeline
from IPython.display import display, Markdown
import os

print("\n--- PHASE 2: Launching AgriLens AI Image+Text Diagnostic (using pipeline) ---")

try:
    # --- Check if model and processor were loaded successfully in Cell 1 ---
    if 'model' not in globals() or 'processor' not in globals() or model is None or processor is None:
        raise NameError("Model or processor not found. Please ensure Cell 1 ran successfully and loaded 'model' and 'processor'.")
    print("✅ Model and Processor objects found from previous cell.")

    # --- Device Setup ---
    pipeline_device = "cpu" # Default to CPU
    try:
        # Attempt to import torch_xla and detect TPU
        import torch_xla
        import torch_xla.core.xla_model as xm
        tpu_device = xm.xla_device()
        if tpu_device.type == 'xla':
            pipeline_device = "xla"
            print(f"🧠 TPU detected and configured for pipeline: {pipeline_device}")
        else:
            print(f"⚠️ XLA device detected but not TPU: {tpu_device}. Trying GPU next.")
            if torch.cuda.is_available():
                pipeline_device = "cuda"
                print(f"✅ GPU detected for pipeline: {pipeline_device}")
            else:
                print("⚠️ GPU not available. Falling back to CPU.")
                pipeline_device = "cpu"
    except ImportError:
        print("`torch_xla` not found. Trying GPU detection...")
        if torch.cuda.is_available():
            pipeline_device = "cuda"
            print(f"✅ GPU detected for pipeline: {pipeline_device}")
        else:
            print("⚠️ GPU not available. Using CPU for pipeline.")
            pipeline_device = "cpu"
    except Exception as e:
        print(f"❌ Error during device detection: {e}. Falling back to CPU.")
        pipeline_device = "cpu"

    print(f"🧠 Diagnostic pipeline configured to run on: {pipeline_device}")

    # 📸 Load the image to be analyzed
    # Make sure this path is correct in your Kaggle environment.
    image_path = "/kaggle/input/tomato/tomato_early_blight.jpg" # <<<------- Make sure this path is correct!
    try:
        image = Image.open(image_path).convert("RGB")
        print(f"✅ Image loaded successfully: {image_path}")
        print("Displaying loaded image:")
        display(image)
    except FileNotFoundError:
        print(f"❌ ERROR: Image file not found at '{image_path}'. Please verify the path.")
        raise
    except Exception as e:
        print(f"❌ Error loading image from {image_path}: {e}")
        raise

    # 💬 Structured prompt for a comprehensive diagnosis
    prompt_text_multimodal = (
        "Analyse cette feuille de tomate. Décris les symptômes visibles : taille, forme, couleur et répartition des lésions. "
        + "Donne ensuite un diagnostic structuré en 5 parties :\n"
        + "1. Nom de la maladie probable\n"
        + "2. Agent pathogène suspecté\n"
        + "3. Mode d'infection et de transmission\n"
        + "4. Conditions climatiques favorables à la maladie\n"
        + "5. Méthodes de lutte (préventives et curatives)"
    )
    print("✅ Diagnostic prompt prepared.")

    # --- Using the pipeline ---
    print("Initializing image-text-to-text pipeline...")

    try:
        pipe = pipeline(
            "image-text-to-text",
            model=model,
            tokenizer=processor.tokenizer, # Assuming processor has a tokenizer attribute
            image_processor=processor.image_processor, # Assuming processor has an image_processor attribute
            processor=processor, # Pass the entire processor if it's needed directly
            device=pipeline_device,
            # torch_dtype can be specified here if needed, but often better handled during model loading.
            # If on CPU, ensure model was loaded with float32. If on GPU/TPU, bfloat16 is common.
            # For robustness, if you are SURE about the CPU case:
            # torch_dtype=torch.float32 if pipeline_device == "cpu" else torch.bfloat16,
        )
        print("Pipeline initialized successfully with pre-loaded model and processor.")

    except Exception as e:
        print(f"❌ Error initializing pipeline: {e}")
        raise

    # Prepare messages for the pipeline in the expected format for multimodal models
    messages_for_pipeline = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text_multimodal}
            ]
        }
    ]
    print("✅ Input messages for pipeline prepared.")

    # Perform inference using the pipeline
    print("Generating response using the pipeline...")
    try:
        generation_params = {
            "max_new_tokens": 500,
            "temperature": 0.7,
            "do_sample": True,
        }

        if pipeline_device == "cpu":
            print("Applying conservative generation settings for CPU...")
            generation_params["temperature"] = 0.1
            generation_params["do_sample"] = False
            # You might also try reducing top_k and top_p, or use do_sample=False

        output = pipe(
            text=messages_for_pipeline,
            **generation_params
        )
        print("Pipeline generation completed.")

        # --- SIMPLIFIED AND MORE ROBUST OUTPUT EXTRACTION ---
        print("\n--- Raw Pipeline Output ---")
        print(output) # <<<----- ADDED: Print the raw output to inspect its structure
        print("---------------------------\n")

        result_text = None
        if output and isinstance(output, list) and len(output) > 0:
            output_item = output[0] # Get the first item

            # Check for common formats of generated text
            if 'generated_text' in output_item:
                generated_content = output_item['generated_text']

                # Case 1: 'generated_text' is a list of dictionaries (chat format)
                if isinstance(generated_content, list):
                    # Find the last message with role 'assistant' and extract text content
                    for msg in reversed(generated_content):
                        if msg.get("role") == "assistant" and msg.get("content"):
                            content_parts = msg["content"]
                            if isinstance(content_parts, list):
                                # Find the first 'text' part in the content list
                                for part in content_parts:
                                    if isinstance(part, dict) and part.get("type") == "text":
                                        result_text = part.get("content")
                                        break # Found text part
                            elif isinstance(content_parts, str): # If content is directly a string
                                result_text = content_parts
                            if result_text: break # Found assistant message with content

                # Case 2: 'generated_text' is a simple string
                elif isinstance(generated_content, str):
                    result_text = generated_content

        if result_text:
            print("\n" + "="*80)
            print("📋 AgriLens AI Diagnostic Result (Pipeline) 📋")
            print("="*80 + "\n")
            display(Markdown(result_text))
            print("\n\n🎉🎉🎉 IMAGE+TEXT DIAGNOSTIC COMPLETED SUCCESSFULLY (PIPELINE)! 🎉🎉🎉")
        else:
            # This error now will only trigger if the structured extraction failed,
            # but the raw output is still present and printed above for debugging.
            print("❌ ERROR: Could not extract meaningful text from pipeline output.")
            # If you reach here, it means the 'output' was not empty, but parsing failed.
            # Inspect the "Raw Pipeline Output" printed above to understand the structure.

    except RuntimeError as re:
        if "inf" in str(re) or "nan" in str(re) or "<0" in str(re):
            print(f"\n❌ GENERAL ERROR in PHASE 2 (Pipeline): RuntimeError -> probability tensor contains either `inf`, `nan` or element < 0")
            print("This indicates a numerical instability. The model might be too complex for the current device/dtype configuration.")
            print("Consider:")
            print(f" 1. If on CPU: Ensure model was loaded with `torch_dtype=torch.float32` in Cell 1.")
            print(f" 2. If on TPU: Ensure model was loaded with `torch_dtype=torch.bfloat16` in Cell 1.")
            print(" 3. Further reducing generation parameters (temperature, sampling).")
            print(" 4. Using a dedicated GPU or TPU environment.")
        else:
            print(f"\n❌ GENERAL ERROR in PHASE 2 (Pipeline): RuntimeError -> {re}")
        print("Please check previous error messages for specific issues.")

    except Exception as e:
        print(f"\n❌ GENERAL ERROR in PHASE 2 (Pipeline): {type(e).__name__} -> {e}")
        print("Please check previous error messages for specific issues.")

except NameError as ne:
    print(f"\n❌ SETUP ERROR in PHASE 2: {ne}")
    print("Ensure that 'model' and 'processor' were successfully loaded in a preceding cell.")
except Exception as e:
    print(f"\n❌ UNEXPECTED ERROR in PHASE 2: {type(e).__name__} -> {e}")




