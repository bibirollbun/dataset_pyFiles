# Function to safely install packages
def install_package(package, version=None):
    """Install a package safely, checking if it's already installed."""
    try:
        if version:
            # Use >= for minimum version specification
            !pip install -q {package}{version} 
        else:
            !pip install -q {package} 
        print(f"Successfully installed {package}")
        return True
    except Exception as e:
        print(f"Failed to install {package}: {e}")
        return False

# List of required packages with versions
required_packages = {
    "gradio": "latest",
    "ollama": "latest",
    "markdown2": "latest",
    # "torch": ">=2.4.0",
    # "transformers": ">=4.53.0"
    }
print("!!! Installation concluded !!!")

# Install packages
for package, version in required_packages.items():
    if version == "latest":
        install_package(package)
    else:
        install_package(package, version)


%%time
import os
import sys
import psutil
import subprocess
import logging
import warnings
import gradio as gr
import ollama
from ollama import chat
from PIL import Image
import io
import base64
import markdown2
# import torch
# from transformers import AutoProcessor, AutoModelForImageTextToText
# !pip list | grep -E 'torch|transformers'


%%time
# Cell: Logger Configuration
# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create file handler and set level to debug
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.INFO)

# Create console handler and set level to error
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)

# Create formatters
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add formatters to handlers
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


# ==================================================
# _/$\_\%/_/&\_\@/_/$\_\%/_/&\_\@/_/$\_\%/_/&\_\@/_
# ==================================================
#  **********    System Information   *************
# ==================================================
# _/$\_\%/_/&\_\@/_/$\_\%/_/&\_\@/_/$\_\%/_/&\_\@/_
# ==================================================
def print_system_info(use_torch=False):
    print("System Information:")
    print(f"â€¢ Python version: {sys.version}")
    print(f"â€¢ Current working directory: {os.getcwd()}")
    
    if use_torch:
        # No need to import torch. Just use it.
        print(f"â€¢ PyTorch version: {torch.__version__}")
        # Check GPU availability and details
        if torch.cuda.is_available():
            gpu_info = {
                "CUDA Available": torch.cuda.is_available(),
                "CUDA Device Count": torch.cuda.device_count(),
                "Current CUDA Device": torch.cuda.current_device(),
                "Device Name": torch.cuda.get_device_name(torch.cuda.current_device()),
                "Memory Allocated (MB)": round(torch.cuda.memory_allocated(0) / 1024**2, 2),
                "Memory Reserved (MB)": round(torch.cuda.memory_reserved(0) / 1024**2, 2),
            }
            
            print("\nâš¡ GPU Detected:")
            for key, value in gpu_info.items():
                print(f"  â€¢ {key}: {value}")
        else:
            print("\nğŸ˜­ No GPU detected. Running on CPU only.")

    # Memory information
    ram = psutil.virtual_memory()
    print("\nğŸ�˜ System Memory:")
    print(f"  â€¢ Total RAM: {round(ram.total / 1024**2, 2)} MB")
    print(f"  â€¢ Available RAM: {round(ram.available / 1024**2, 2)} MB")
    print(f"  â€¢ Used RAM: {round(ram.used / 1024**2, 2)} MB")
    print(f"  â€¢ RAM Percentage: {ram.percent}% used")

# Check if torch is imported
try:
    import torch
    torch_imported = True  # Indicate that torch is available
except ImportError:
    torch_imported = False  # Indicate that torch is not available

# Call the function based on whether torch is imported
if torch_imported:
    print_system_info(use_torch=True)  # Call the function with use_torch=True
else:
    print_system_info(use_torch=False)  # Call the function with use_torch=False


%%time
!curl -fsSL https://ollama.com/install.sh | sh


%%time
process = subprocess.Popen("ollama serve", shell=True)


%%time
# !ollama pull gemma3n:e4b 
!ollama pull gemma3n:e2b


# Unified function for chat with Gemma
def gemma_chat(history, url_input, question):
    try:
        # Validate the question input
        question = question.strip()
        if not question:
            return history, "Please enter a valid question."
        if url_input:  # If a URL has been provided
            message_content = f"You have received a URL for a file: {url_input}. Question: {question}"
        else:  # No URL provided, just process the question
            message_content = f"Question: {question}"
        # Send the prompt to the Ollama model
        response = chat(model='gemma3n:e2b', messages=[
            {
                "role": "user",
                "content": message_content
            },
        ])
        answer = response['message']['content']
        # Convert the Markdown answer to HTML
        answer_html = markdown2.markdown(answer)
        # Update history with the new interaction
        history.append(f"<div style='color: blue;'>You: {question}</div>")
        history.append(f"<div style='color: green;'>Gemma 3n: {answer_html}</div>")
        history_text = "<br>".join(history)
        
        return history_text, answer
    except Exception as e:
        return history, f"Error occurred: {str(e)}"

# Create Gradio interface
with gr.Blocks() as demo:
    history = gr.State([])
    with gr.Column():
        gr.Markdown("# Welcome to the Cruzeta Analysis Portal")
        chat_output = gr.HTML(label="Chat History")
        response_output = gr.Textbox(label="Response", placeholder="Model response will appear here...", interactive=False)
        url_input = gr.Textbox(lines=1, label="Enter File URL (audio/image) or leave it blank")
        question_input = gr.Textbox(lines=2, label="Ask Gemma")
        
        submit_button = gr.Button("Submit")
    
    # Connect inputs and outputs
    submit_button.click(
        gemma_chat,
        inputs=[history, url_input, question_input],
        outputs=[chat_output, response_output]
    )

# Launch the Gradio interface
demo.launch()

