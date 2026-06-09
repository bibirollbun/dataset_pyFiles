!pip install timm --upgrade
!pip install accelerate
!pip install git+https://github.com/huggingface/transformers.git


import kagglehub

GEMMA_PATH = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b-it")


import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

tokenizer = AutoTokenizer.from_pretrained(GEMMA_PATH, trust_remote_code=True)

# ⚠️ 모델을 CPU로 로드하고 데이터 타입을 f32로 지정합니다.
model = AutoModelForCausalLM.from_pretrained(
    GEMMA_PATH,
    trust_remote_code=True,
    torch_dtype=torch.float32
).to("cpu")

prompt = "케글에는 오리가 왜 이렇게 많을까?"

# ⚠️ 입력 데이터도 CPU로 보냅니다.
inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

generation_config = GenerationConfig(max_new_tokens=150, do_sample=True, temperature=0.7)
outputs = model.generate(**inputs, generation_config=generation_config)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)


print(result)


import gc
import torch

# 이전에 사용한 모델과 토크나이저 변수 삭제
try:
    del model
    del tokenizer
    del inputs
    del outputs
    del result
    print("이전 모델과 변수들을 메모리에서 삭제했습니다.")
except NameError:
    print("삭제할 변수가 없습니다. 계속 진행합니다.")

# 가비지 컬렉션 실행
gc.collect()

# (GPU 사용 시) 캐시 비우기 - CPU 환경에서는 큰 의미 없지만 좋은 습관입니다.
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("GPU 캐시를 비웠습니다.")


from IPython.display import Image
IMAGE_URL="https://storage.googleapis.com/kaggle-media/competitions/question_goose.png"
Image(url=IMAGE_URL,height=250,width=250)


from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained(GEMMA_PATH)
# ⚠️ 수정된 부분: 모델을 CPU로 로드합니다.
model = AutoModelForImageTextToText.from_pretrained(
    GEMMA_PATH,
    torch_dtype=torch.float32, # CPU 연산을 위해 float32로 변경
    # device_map="auto" 삭제
).to("cpu") # CPU로 모델을 보냄

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": IMAGE_URL},
            {"type": "text", "text": "Describe this image in detail."}
        ]
    }
]

# ⚠️ 수정된 부분: 입력 데이터(inputs)를 CPU로 보냅니다.
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to("cpu") # to(model.device) 대신 명시적으로 "cpu" 사용

input_len = inputs["input_ids"].shape[-1]
outputs = model.generate(**inputs, max_new_tokens=512, disable_compile=True)

text = processor.batch_decode(
    outputs[:, input_len:],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True
)


print(text[0])




