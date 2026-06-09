%%writefile requirements.txt

# [TPU] [Perf] Improve Memory Usage Estimation (#15671)
vllm @ https://wheels.vllm.ai/038bededbac67e873d3aa6155c7c05674b98db8c/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl

# faster safetensors loading
runai-model-streamer==0.11.2

# numpy==2.2.3
# transformers==4.49.0

# don't go beyond 0.2.2.post1 as "success" is removed from 0.2.3 in sampling API and causes error
flashinfer-python @ https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.2.post1/flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl


!pip install --target=/kaggle/working -r requirements.txt -U
!rm -rf /kaggle/working/ray*


# delay importing flashinfer as it causes early cuda initialization
# and that will force spawn for multiprocessing even if fork is set


import os
import sys
import shutil
from importlib.util import find_spec

# Find the vLLM package location
vllm_path = None
for path in sys.path:
    if os.path.exists(os.path.join(path, 'vllm')):
        vllm_path = os.path.join(path, 'vllm')
        break

if not vllm_path:
    raise ImportError("vllm package not found in sys.path")

sampler_path = os.path.join(vllm_path, 'model_executor', 'layers', 'sampler.py')
print(f"Found sampler.py at: {sampler_path}")

# Read the file content
with open(sampler_path, 'r') as f:
    content = f.read()

# Replace the direct import with lazy loading
modified_content = content.replace(
    '''if envs.VLLM_USE_FLASHINFER_SAMPLER and find_spec("flashinfer"):
    import flashinfer.sampling
    # yapf: disable
    from flashinfer.sampling import (
        top_k_top_p_sampling_from_probs as flashinfer_top_k_top_p_sampling)

    # yapf: enable
else:
    flashinfer_top_k_top_p_sampling = None''',
    
    '''# Lazy loading for flashinfer
_flashinfer_loaded = False
flashinfer_top_k_top_p_sampling = None

def _load_flashinfer():
    global _flashinfer_loaded, flashinfer_top_k_top_p_sampling
    if not _flashinfer_loaded:
        try:
            if envs.VLLM_USE_FLASHINFER_SAMPLER and find_spec("flashinfer"):
                import flashinfer.sampling
                from flashinfer.sampling import top_k_top_p_sampling_from_probs
                flashinfer_top_k_top_p_sampling = top_k_top_p_sampling_from_probs
                _flashinfer_loaded = True
                return True
        except Exception as e:
            print(f"Warning: Failed to load flashinfer: {e}")
    return flashinfer_top_k_top_p_sampling is not None'''
)

# Now modify the _top_k_top_p_multinomial_with_flashinfer function
modified_content = modified_content.replace(
    '''def _top_k_top_p_multinomial_with_flashinfer(
        probs: torch.Tensor, top_ks: torch.Tensor, top_ps: torch.Tensor,
        num_samples: int, seq_groups: Optional[List[SequenceGroupToSample]]):''',
    
    '''def _top_k_top_p_multinomial_with_flashinfer(
        probs: torch.Tensor, top_ks: torch.Tensor, top_ps: torch.Tensor,
        num_samples: int, seq_groups: Optional[List[SequenceGroupToSample]]):
    # Try to load flashinfer on demand
    if not _load_flashinfer():
        # Fallback to normal top-k, top-p if flashinfer is not available
        probs_modified = probs.clone()
        for i in range(probs.shape[0]):
            filtered_logits = torch.log(probs_modified[i:i+1])
            filtered_logits = _apply_top_k_top_p(filtered_logits, top_ps[i:i+1], top_ks[i:i+1])
            probs_modified[i] = torch.softmax(filtered_logits, dim=-1)
        return _multinomial(probs_modified, num_samples, seq_groups=seq_groups)'''
)

# Write the modified content back
with open(sampler_path, 'w') as f:
    f.write(modified_content)
print("Successfully modified sampler.py to use lazy loading for flashinfer")




