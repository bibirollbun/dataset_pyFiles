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
    torch_dtype=torch.float32,
    # device_map="auto" # GPU 사용 시 이 옵션을 고려해볼 수 있습니다.
).to("cpu")

# --- 개선된 영문 프롬프트 ---
prompt = """
**Role:** You are a seasoned Senior Financial Analyst at a top Wall Street firm, known for your insightful and balanced reports.

**Task:** Based on the provided summary of Alphabet Inc. (GOOG) stock data, write a short-term outlook report for institutional investors. Your analysis must be logical, data-driven, and clearly structured.

**Instructions:**
1.  Start with a compelling and professional title for the report.
2.  Provide a brief Executive Summary.
3.  Analyze the positive factors (Bull Case / Catalysts).
4.  Analyze the negative/neutral factors (Bear Case / Headwinds). Elaborate on what the "3 threat factors" might hypothetically be, given the market context (e.g., regulatory scrutiny, AI competition, macroeconomic pressures).
5.  Conclude with an overall sentiment and outlook on the potential short-term trend.
6.  **Crucially, do not provide a specific price target.** Focus on the narrative and strategic analysis.

**[Data Summary]**
- **Ticker:** Alphabet Inc. Class C (GOOG)
- **As of:** July 3rd (End of Day)
- **Current Price:** 180.55 USD
- **52-Week Range:** 142.66 USD - 208.70 USD
- **Market Cap:** 2.18T USD
- **P/E Ratio:** 20.45
- **Recent Earnings:** Q1 2024 revenue and EPS significantly beat market expectations, showing strong growth.
- **Market Analysis:**
  1) **Positive Factor:** Overwhelmingly strong Q1 earnings announcement.
  2) **Negative/Neutral Factor:** Mentions of "3 threat factors" are causing some conservative sentiment, suggesting the stock may not rally excessively despite strong results.

**[Begin Report]**
"""

# ⚠️ 입력 데이터도 CPU로 보냅니다.
inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

# 생성 옵션: 온도를 0.7로 설정하여 약간의 창의성을 부여하면서도 일관성을 유지
generation_config = GenerationConfig(max_new_tokens=1024, do_sample=True, temperature=0.7, top_p=0.95)

print("--- Generating Report ---")
outputs = model.generate(**inputs, generation_config=generation_config)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 프롬프트 부분을 제외하고 결과만 출력
result_only = result[len(prompt):]
print(result_only)


print(result)

