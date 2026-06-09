import torch 
print("Installing llama-cpp-python...")
try:
    # Use !pip for shell commands in notebooks
    # Check for CUDA availability to install the correct version
    if torch.cuda.is_available():
        !pip install llama-cpp-python[cuda] --force-reinstall --no-cache-dir
        print("Installed llama-cpp-python with CUDA support.")
    else:
        !pip install llama-cpp-python --force-reinstall --no-cache-dir
        print("Installed llama-cpp-python (CPU-only).")
except Exception as e:
    print(f"Error installing llama-cpp-python: {e}")
    print("Please ensure your environment is set up correctly for pip installations.")
    exit()




from llama_cpp import Llama
import os

GGUF_MODEL_PATH = "/kaggle/input/gemma-2b-finetuned-gguf/gguf/default/1/merged_2B_model.Q8_0.gguf" # Example path, adjust as needed!

print(f"\nLoading GGUF model from: {GGUF_MODEL_PATH}...")
try:
    if not os.path.exists(GGUF_MODEL_PATH):
        raise FileNotFoundError(f"GGUF model not found at: {GGUF_MODEL_PATH}")

    llm = Llama(
        model_path=GGUF_MODEL_PATH,
        n_gpu_layers=-1 if torch.cuda.is_available() else 0, 
        n_ctx=2048, 
        verbose=False, # Show verbose loading output
    )
    print("GGUF model loaded successfully!")

except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()
except Exception as e:
    print(f"Error loading GGUF model: {e}")
    exit()




print("\nPerforming inference...")
try:
    system_prompt = ("You are a cheerful and friendly tutor for children aged 5 to 7."
    " Use simple words and fun metaphors to explain things clearly. Be playful and keep "
    "answers short and exciting. You can use characters like 'sugar bugs' or 'energy monsters' "
    "to make it fun.")
    user_prompt = "Why do we brush our teeth?"
    
    # Format to the template of Gemma models
    formatted_prompt = (
        f"<start_of_turn>system\n{system_prompt}<end_of_turn>\n"
        f"<start_of_turn>user\n{user_prompt}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

    print(f"\nPrompt for the Model: {user_prompt}")

    output = llm.create_completion(
        formatted_prompt,
        max_tokens=512,
        temperature=0.7,
        top_p=0.9,
        stop=["<end_of_turn>"],
        echo=True,
    )
    generated_text = output["choices"][0]["text"]
    # To get only the model's response, strip the prompt part
    response_only = generated_text.replace(formatted_prompt, "").strip()
    print("\nModel's Response Only:")
    print(response_only)
except Exception as e:
    print(f"Error during inference: {e}")

