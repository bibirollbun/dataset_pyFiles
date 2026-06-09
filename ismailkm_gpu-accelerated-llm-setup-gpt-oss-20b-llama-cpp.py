!pip install huggingface-hub llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122 --no-cache-dir --upgrade --force-reinstall --ignore-installed



from huggingface_hub import hf_hub_download

# Define the model repository and filename
# We're using a 4-bit quantized model (mxfp4) for efficient GPU usage
repo_id = "ggml-org/gpt-oss-20b-GGUF"
filename = "gpt-oss-20b-mxfp4.gguf"

# Download the model file from the Hugging Face Hub
model_path = hf_hub_download(repo_id=repo_id, filename=filename)
print(f"Model downloaded to: {model_path}")


from llama_cpp import Llama

# Initialize the model with optimized settings
# 'n_ctx=2048' sets the maximum context length (how much the model 'remembers')
# 'n_gpu_layers=-1' offloads all possible layers to the GPU
llm = Llama(model_path=model_path, n_ctx=2048, n_gpu_layers=-1)



# Define a simple prompt
prompt = """
You are a helpful assistant. Please respond to the user's question directly.

User: Hello, how are you?
"""

# Run the model to generate a response
# 'max_tokens' controls the length of the generated output
response = llm(prompt, max_tokens=500)

# Print the generated text
print(response['choices'][0]['text'])

