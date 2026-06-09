# %pip download \
#     "sglang[all]" "sglang[runtime_common]" "aiohttp" "decord" "fastapi" \
#     "hf_transfer" "huggingface_hub" "interegular" "modelscope" \
#     "orjson" "outlines>=0.0.44,<0.1.0" \
#     "packaging" "pillow" "prometheus-client>=0.20.0" \
#     "psutil" "pydantic" "python-multipart" \
#     "pyzmq>=25.1.2" "torchao>=0.7.0" "uvicorn" "uvloop" \
#     "xgrammar>=0.1.6" "cuda-python" \
#     "sgl-kernel>=0.0.2.post11" "torch" "vllm>=0.6.3.post1,<=0.6.4.post1" \
#     "setproctitle" \
#     "flashinfer==0.1.6" --find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/ \
#     --no-deps


# !pip uninstall IPython -y
!pip install  traitlets>=5.0 -U





!pip install --target=/kaggle/working -U "sglang[srt]" 
# Install FlashInfer accelerated kernels
!pip install  --target=/kaggle/working   flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/


!nvidia-smi


# import sglang  as sgl
# import asyncio,os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# llm_model_pth = 'PowerInfer/SmallThinker-3B-Preview'
# llm = sgl.Engine(model_path=llm_model_pth,tp_size=2,enable_p2p_check=True)






