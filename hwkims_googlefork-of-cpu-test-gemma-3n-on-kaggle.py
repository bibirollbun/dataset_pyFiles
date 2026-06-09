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

prompt = """
당신은 월스트리트의 유능한 금융 데이터 분석가입니다. 아래 제공된 Alphabet(GOOG) 주식 데이터를 기반으로, 현재 상황을 분석하고 단기적인 전망에 대한 리포트를 작성해 주세요. 리포트에는 긍정적 요인과 부정적 요인을 모두 포함해야 합니다. 구체적인 목표 주가를 제시하지 말고, 시장의 전반적인 센티먼트와 잠재적 추세에 대해 논리적으로 서술하세요.

[데이터 요약]
- 종목명: Alphabet Inc. Class C (GOOG)
- 기준일: 7월 3일 (장 마감)
- 현재가: 180.55 USD
- 52주 변동폭: 142.66 USD ~ 208.70 USD
- 시가총액: 2.18조 USD
- 주가수익률(P/E): 20.45
- 최근 실적: 2024년 1분기 매출 및 주당순이익(EPS)이 시장 예상치를 크게 상회하며 강력한 성장세를 보임.
- 시장 분석:
  1) 긍정적 요인: 압도적인 1분기 실적 발표.
  2) 부정적/중립적 요인: 주가에 대한 '3가지 위협 요인'이 거론되며, 주가가 과도하게 반응하지 않을 것이라는 보수적인 시각 존재.

[리포트 작성 시작]
"""

# ⚠️ 입력 데이터도 CPU로 보냅니다.
inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

generation_config = GenerationConfig(max_new_tokens=5000, do_sample=True, temperature=0.7)
outputs = model.generate(**inputs, generation_config=generation_config)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)


print(result)

