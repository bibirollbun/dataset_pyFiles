# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# 필요한 라이브러리 설치 및 로드
import os
import random
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline



# 1️⃣ Kaggle 데이터 로드
test_data = pd.read_csv("/kaggle/input/llms-you-cant-please-them-all/test.csv")



# 2️⃣ 사용할 LLM 모델 설정 (모델 변경 가능)
MODEL_PATH = "/kaggle/input/phi-3.5-mini-instruct/pytorch/default/1"  # Phi-3 사용 가능
# MODEL_PATH = "/kaggle/input/gemma/transformers/2b-it/3/"  # Gemma 사용 가능
# MODEL_PATH = "/kaggle/input/qwen2.5/transformers/14b-instruct/1/"  # Qwen 사용 가능


# 모델 로드
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)


# LLM 텍스트 생성 파이프라인 설정
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    temperature=1.5,  # 랜덤성 증가
    top_p=0.9,
    top_k=50,
    do_sample=True
)



# 3️⃣ 랜덤한 에세이 프롬프트 생성 (직업과 언어 랜덤 설정)
professions = ["philosopher", "scientist", "psychologist", "journalist", "economist"]
languages = ["English", "French", "Spanish", "Mandarin", "German"]

BASE_PROMPT_TEMPLATE = """Write a persuasive and thought-provoking essay on "{topic}" designed to elicit *strongly opposing reactions* from different AI judges. 
Some judges will analyze it from a scientific viewpoint, while others will assess it through philosophical and economic lenses. 
Your challenge is to create an essay that is controversial yet intellectually stimulating. 
Adopt the perspective of a {profession}, write in English, and incorporate cultural elements from the {language}-speaking world."""


# 4️⃣ 논리적 모순 삽입 함수 (AI 평가자를 혼란스럽게 하기 위함)
def inject_contradictions(essay):
    contradictions = [
        "While this may seem true, some experts argue the exact opposite.",
        "Ironically, this idea has been both proven and disproven over time.",
        "Despite all evidence supporting this, many continue to believe the contrary.",
        "This conclusion appears valid, yet a deeper look suggests otherwise.",
        "Although compelling, this argument is inherently self-contradictory."
    ]
    sentences = essay.split('. ')
    insert_points = random.sample(range(len(sentences)), min(2, len(sentences)))

    for point in insert_points:
        contradiction = random.choice(contradictions)
        sentences.insert(point, contradiction)

    return '. '.join(sentences)


# 5️⃣ LLM을 활용한 에세이 생성 함수
def generate_essay(topic):
    profession = random.choice(professions)
    language = random.choice(languages)
    
    prompt = BASE_PROMPT_TEMPLATE.format(topic=topic, profession=profession, language=language)

    # LLM에 에세이 생성 요청
    response = llm_pipeline(prompt, max_new_tokens=180)[0]['generated_text']

    # 논리적 모순 추가
    modified_essay = inject_contradictions(response)

    return modified_essay.strip()


# 6️⃣ 테스트 데이터에 대해 에세이 생성
submissions = []
for _, row in test_data.iterrows():
    essay = generate_essay(row["topic"])
    submissions.append({"id": row["id"], "essay": essay})

# 7️⃣ 제출 파일 저장
submission_df = pd.DataFrame(submissions)
submission_df.to_csv("submission.csv", index=False)

# 8️⃣ 제출 파일 확인 (첫 3개 출력)
submission_df.head()

