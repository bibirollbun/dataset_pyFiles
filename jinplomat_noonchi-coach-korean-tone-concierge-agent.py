# ===== 1. 환경 설정 =====
import os
from kaggle_secrets import UserSecretsClient

# Kaggle Secrets 에서 Gemini API 키 가져오기
user_secrets = UserSecretsClient()
GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in Kaggle Secrets.")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

print("✅ GOOGLE_API_KEY loaded and environment configured.")



# ===== 2. ADK 기본 세팅 =====
from google.genai import types
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner

MODEL_NAME = "gemini-2.5-flash-lite"

retry_config = types.HttpRetryOptions(
    attempts=3,
    exp_base=2,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)



# ===== 3. Custom Tool: Korean Tone Heuristic Analyzer =====
from typing import Dict, List

def analyze_korean_tone(utterance: str) -> Dict:
    """
    Roughly classify the politeness level of a single Korean utterance.
    """
    text = utterance.strip()
    features: List[str] = []

    ending = text[-5:] if len(text) >= 5 else text
    level = "unknown"

    # 아주 러프한 규칙 기반 톤 탐지 (Demo용)
    if any(text.endswith(suffix) for suffix in ["야", "라", "해", "했어", "하냐", "하니"]):
        level = "banmal"
        features.append("casual_ending")
    if any(suffix in text for suffix in ["요", "어요", "아요", "해요", "했어요"]):
        level = "polite-casual"
        features.append("haeyo_style")
    if any(suffix in text for suffix in ["습니다", "입니까", "합니까", "십시오", "하겠습니다"]):
        level = "formal"
        features.append("hapsyo_style")

    # honorific-ish keywords (아주 단순)
    if any(word in text for word in ["께서", "드셨", "진지", "모시고"]):
        features.append("honorific_keyword")

    return {
        "status": "success",
        "level": level,
        "features": features,
        "ending": ending,
    }

print(analyze_korean_tone("오늘 처음 뵙겠습니다. 잘 부탁드립니다."))



# ===== 4. Noonchi Tone Coach Agent 정의 =====

noonchi_agent = Agent(
    name="noonchi_tone_coach",
    model=Gemini(
        model=MODEL_NAME,
        retry_options=retry_config,
    ),
    description=(
        "A concierge-style agent that helps Korean learners "
        "check politeness level and improve their tone before "
        "they arrive in Korea."
    ),
    instruction="""
You are 'Noonchi', a Korean tone and politeness coach.
Your main users are foreigners who will study, work, or marry into families in Korea.

When the user gives you:
- A short situation description (in Korean or English), and
- Their Korean sentence (or draft reply),

You MUST:

1) Call the `analyze_korean_tone` tool first with the learner's Korean sentence.
2) Use that result + your own understanding of Korean to decide if the tone is:
   - too casual / risky,
   - natural / appropriate,
   - overly formal / distant,
   - or awkward / unnatural.
3) Respond in Korean first, then add one short English hint.

Your response format:

[톤 요약]
[피드백]
[추천 표현]
[English hint]

Keep total length under 10 lines.
""",
    tools=[analyze_korean_tone],
)

runner = InMemoryRunner(agent=noonchi_agent)
print("✅ Noonchi Tone Coach agent initialized.")



# ===== 5. Demo: 몇 개 예시 질의 돌려보기 =====

sample_queries = [
    """상황: 첫 출근 날 팀장님께 인사할 때.
문장: 팀장님, 오늘 처음 뵙네요. 잘 부탁해요!""",
    """상황: 부장님께 점심 식사 여부를 여쭤볼 때.
문장: 부장님, 진지 잡수셨습니까?""",
    """상황: 사무실에서 동료에게 회의 시간을 다시 확인.
문장: 혹시 내일 회의가 오전 열 시 맞죠?""",
]

async def run_demos():
    for q in sample_queries:
        print("=" * 80)
        print("USER INPUT:\n", q)
        print("-" * 80)
        await runner.run_debug(q, verbose=True)

await run_demos()


