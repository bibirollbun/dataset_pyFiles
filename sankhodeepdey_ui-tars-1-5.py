!pip install -U transformers


!CUDA_VERSION=cu126


!pip uninstall -y vllm
!pip install --no-cache-dir vllm --extra-index-url https://download.pytorch.org/whl/${CUDA_VERSION}


!pip install pyngrok

from pyngrok import ngrok

# **Authenticate with your authtoken**
ngrok.set_auth_token("2u8z4RHM4iISFsjUerWhT8pcdQV_6YUwWZiZ4wdBgxbAcKyyn")  # Replace YOUR_AUTHTOKEN with your actual authtoken

# 8000 port e je server cholche, ota expose korbo
public_url = ngrok.connect(8000)
print(f"Public URL: {public_url}")


!python -m vllm.entrypoints.openai.api_server \
    --served-model-name ui-tars \
    --model "ByteDance-Seed/UI-TARS-1.5-7B" \
    --limit-mm-per-prompt image=5 -tp 4 \
    --dtype=half

