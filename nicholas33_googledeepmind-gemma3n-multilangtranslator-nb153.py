# 1. SETUP AND INSTALLATION
# -------------------------
print("Installing required packages (transformers from source)...")
!pip install -q timm --upgrade
!pip install -q accelerate
!pip install -q git+https://github.com/huggingface/transformers.git

import os
import time
import torch
import kagglehub
from transformers import AutoTokenizer, AutoModelForCausalLM 
from typing import List

print("ğŸ”„ Configuring PyTorch backend for compatibility...")
try:
    import torch._dynamo
    torch._dynamo.reset() 
    os.environ["TORCH_COMPILE_BACKEND"] = "eager"
    torch._dynamo.config.suppress_errors = True
except ImportError:
    print("âš ï¸� Could not import torch._dynamo. Skipping dynamo configuration.")

print("âœ… Packages installed successfully!")


# 2. MODEL CONFIGURATION
# ----------------------
print("Downloading model files with kagglehub...")
MODEL_PATH = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b-it")
print(f"âœ… Model path set to: {MODEL_PATH}")


# 3. GENERAL-PURPOSE TRANSLATOR CLASS
# -----------------------------------
class GemmaTranslator: 
    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialize_model()

    def _initialize_model(self):
        """Loads the Gemma model using AutoModelForCausalLM."""
        if not os.path.exists(self.model_path):
            print("â�Œ Cannot initialize model: Path does not exist.")
            return

        try:
            print("ğŸ”„ Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            print("ğŸ”„ Loading model... (This may take a moment)")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
            ).to(self.device)

            print("âœ… Translator initialized successfully!")
            if self.model:
                  print(f"ğŸ“Š Model configured for device: {self.device}")

        except Exception as e:
            print(f"â�Œ Error initializing model: {e}")
            self.model = None
            self.tokenizer = None
            
    # CHANGED: This method now accepts a `target_language` argument.
    def translate(self, english_text: str, target_language: str) -> str:
        """
        Translates English text into the specified target language.
        """
        if not self.model or not self.tokenizer:
            return "â�Œ Model not initialized. Please check for errors above."

        prompt = f"""Translate the following English text exactly into {target_language}. Provide only the {target_language} translation.

English: "{english_text}"
{target_language}:"""

        try:
            start_time = time.time()
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # This is the key fix for GPU compatibility.
            outputs = self.model.generate(**inputs, max_new_tokens=100, disable_compile=True)
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            end_time = time.time()
            
            # The response parsing is now also dynamic.
            if f"{target_language}:" in response:
                translation = response.split(f"{target_language}:")[-1].strip()
            else:
                translation = response[len(prompt) - len(f'\n{target_language}:'):].strip()
            
            translation = translation.replace('"', '').strip()
            
            print(f"â�±ï¸�  '{english_text[:30]}...' -> {target_language} in {end_time - start_time:.2f}s")
            return translation

        except Exception as e:
            return f"â�Œ Translation error: {e}"


# 4. INITIALIZE AND TEST THE MULTILINGUAL TRANSLATOR
# ----------------------------------------------------
print("\n--- Initializing Translator ---")
translator = GemmaTranslator(MODEL_PATH)
print("-----------------------------\n")

if translator.model:
    # A dictionary of sentences to test the new multilingual capability.
    test_cases = {
        "Where is the nearest library?": "Spanish",
        "How much does this cost?": "German",
        "I would like to order a coffee.": "French",
        "This is a very powerful language model.": "Italian",
    }

    print("--- Running Multilingual Translation Examples ---")
    for sentence, language in test_cases.items():
        translation = translator.translate(sentence, language)
        print(f"ğŸ‡ºğŸ‡¸ English: {sentence}")
        print(f"ğŸŒ� {language}: {translation}")
        print("-" * 40)
        
    print("\nğŸ�‰ All tasks complete!")
else:
    print("Could not run translation examples because the model failed to initialize.")


print("ğŸŒ� Gemma Interactive Translator")
print("---------------------------------")
print("Enter the English text you want to translate.")
print("Then, enter the target language (e.g., Spanish, German, Japanese).")
print("Type 'quit' at any time to exit the translator.")
print("---------------------------------")

# Start an infinite loop to keep the translator running
while True:
    # Get input text from the user
    english_input = input("\nğŸ‡¬ğŸ‡§ English text (or 'quit' to exit): ")
    
    # Check if the user wants to exit
    if english_input.lower() == 'quit':
        print("\nExiting translator. Goodbye!")
        break
        
    # Get the target language from the user
    language_input = input(f"ğŸŒ� Target language for '{english_input[:50]}...': ")

    # Check for exit condition again
    if language_input.lower() == 'quit':
        print("\nExiting translator. Goodbye!")
        break
        
    print("\nğŸ”„ Translating...")
    
    # Use the existing 'translator' object to perform the translation
    translated_text = translator.translate(english_input, language_input)
    
    # Print the final translation
    print(f"\nâœ… Result in {language_input}:")
    print(translated_text)
    print("---------------------------------")

