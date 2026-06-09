# StyleSense Concierge — Full Multi-Agent Codebase (Repo)

## Repo structure

```
stylesense-concierge/
├─ README.md
├─ LICENSE
├─ requirements.txt
├─ .env.example
├─ docker-compose.yml
├─ Dockerfile
├─ backend/
│  ├─ app/
│  │  ├─ main.py
│  │  ├─ orchestrator.py
│  │  ├─ llm_client.py
│  │  ├─ agents/
│  │  │  ├─ __init__.py
│  │  │  ├─ coordinator_agent.py
│  │  │  ├─ skin_tone_agent.py
│  │  │  ├─ body_type_agent.py
│  │  │  ├─ occasion_agent.py
│  │  │  ├─ color_harmony_agent.py
│  │  │  ├─ outfit_generator_agent.py
│  │  │  └─ evaluator_agent.py
│  │  ├─ tools/
│  │  │  ├─ __init__.py
│  │  │  ├─ palette_generator.py
│  │  │  └─ shop_connector.py
│  │  ├─ memory/
│  │  │  ├─ __init__.py
│  │  │  ├─ session_service.py
│  │  │  └─ memory_bank.py
│  │  └─ utils/
│  │     ├─ logging_config.py
│  │     └─ schemas.py
├─ ui/
│  ├─ streamlit_app.py
│  └─ static/
│     └─ thumbnail.png
└─ .github/
   └─ workflows/ci.yml
```

---

## README.md (top of repo)

```md
# StyleSense Concierge

AI Fashion Concierge — a multi-agent personal stylist that recommends outfit color combinations using body type and skin tone.

This repo contains a minimal, extensible multi-agent system built with FastAPI. Replace LLM client with your provider of choice.

## Quickstart (local)

1. Copy `.env.example` to `.env` and fill keys.
2. `python -m venv .venv && source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. `uvicorn backend.app.main:app --reload --port 8000`
5. Open `http://localhost:8000/docs` to try the API.

Streamlit UI: `streamlit run ui/streamlit_app.py` (port 8501)

## Structure
- `backend/app/` — FastAPI app, agents, tools, and memory
- `ui/` — simple front-end to demo

## Notes
- This project uses a placeholder `LLMClient`. Plug in your OpenAI/Anthropic/other SDK and keys.
```

```

---

## requirements.txt

```

fastapi==0.95.2
uvicorn[standard]==0.22.0
pydantic==1.10.11
requests==2.31.0
python-dotenv==1.0.0
streamlit==1.25.0
sqlalchemy==2.0.19
httpx==0.24.1
prometheus-client==0.16.0
opentelemetry-api==1.20.0
opentelemetry-sdk==1.20.0

```

---

## .env.example

```

LLM_API_KEY=
SHOP_API_KEY=
DATABASE_URL=sqlite:///./stylesense.db

````

---

## Dockerfile (simple)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
````

---

## backend/app/llm_client.py

```python
# llm_client.py
import os

class LLMClient:
    """Minimal provider-agnostic LLM client interface.
    Replace `generate` with actual SDK calls (OpenAI/Anthropic/etc.).
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("LLM_API_KEY")

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        # Placeholder: replace with real API call
        # e.g., openai.ChatCompletion.create(...)
        # For now we echo a simple canned response for testing.
        return f"[LLM_RESPONSE] For prompt: {prompt[:120]}..."
```

---

## backend/app/utils/schemas.py

```python
from pydantic import BaseModel
from typing import Optional, List, Dict

class UserInput(BaseModel):
    user_id: str
    occasion: str
    body_description: Optional[str] = None
    skin_description: Optional[str] = None
    wardrobe_items: Optional[List[Dict]] = []

class AgentResult(BaseModel):
    agent: str
    output: Dict
    score: Optional[float] = None
```

---

## backend/app/utils/logging_config.py

```python
import logging

LOG = logging.getLogger("stylesense")
LOG.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
LOG.addHandler(handler)
```

---

## backend/app/memory/session_service.py

```python
# Simple in-memory session store for short-term state
from typing import Dict
import time

class InMemorySessionService:
    def __init__(self):
        self._store: Dict[str, Dict] = {}

    def create(self, session_id: str, data: Dict):
        data["created_at"] = time.time()
        self._store[session_id] = data

    def get(self, session_id: str):
        return self._store.get(session_id)

    def update(self, session_id: str, data: Dict):
        if session_id in self._store:
            self._store[session_id].update(data)

    def delete(self, session_id: str):
        if session_id in self._store:
            del self._store[session_id]
```

---

## backend/app/memory/memory_bank.py

```python
# Minimal persistent memory using SQLite via SQLAlchemy
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./stylesense.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class UserMemory(Base):
    __tablename__ = "user_memory"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    key = Column(String, index=True)
    value = Column(Text)

Base.metadata.create_all(bind=engine)

class MemoryBank:
    def __init__(self):
        self.db = SessionLocal()

    def upsert(self, user_id: str, key: str, value: str):
        existing = self.db.query(UserMemory).filter_by(user_id=user_id, key=key).first()
        if existing:
            existing.value = value
        else:
            m = UserMemory(user_id=user_id, key=key, value=value)
            self.db.add(m)
        self.db.commit()

    def get(self, user_id: str, key: str):
        found = self.db.query(UserMemory).filter_by(user_id=user_id, key=key).first()
        return found.value if found else None
```

---

## backend/app/tools/palette_generator.py

```python
# palette_generator.py
from typing import List, Dict

def seed_by_undertone(undertone: str) -> str:
    mapping = {
        "warm": "#C46210",
        "cool": "#2E86AB",
        "neutral": "#7F8C8D",
    }
    return mapping.get(undertone, "#7F8C8D")


def generate_palette(undertone: str, strategy: str = "complementary") -> Dict:
    seed = seed_by_undertone(undertone)
    # Lightweight example palette construction
    if strategy == "complementary":
        return {"seed": seed, "palette": [seed, "#FFD700", "#8A0303"], "score": 0.85}
    if strategy == "triadic":
        return {"seed": seed, "palette": [seed, "#0E9AA7", "#F6CD61"], "score": 0.82}
    return {"seed": seed, "palette": [seed, "#FFFFFF"], "score": 0.7}
```

---

## backend/app/tools/shop_connector.py (mock)

```python
# shop_connector.py

def search_items(query: str, api_key: str | None = None):
    # Mocked response — replace with real OpenAPI calls
    return [
        {"id": "s1", "title": "Navy Blazer", "color": "navy", "price": 79.0},
        {"id": "s2", "title": "Gold clutch", "color": "gold", "price": 49.0},
    ]


def add_to_cart(item_id: str, user_id: str, api_key: str | None = None):
    return {"status": "ok", "cart_id": f"cart_{user_id}", "item_id": item_id}
```

---

## backend/app/agents/skin_tone_agent.py

```python
# skin_tone_agent.py
from ..llm_client import LLMClient
from ..utils.logging_config import LOG

class SkinToneAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def classify(self, description: str | None) -> dict:
        # If description provided, use LLM to classify. Fallback to default.
        prompt = f"Classify this skin description into undertone categories: {description}"
        resp = self.llm.generate(prompt)
        LOG.debug("SkinToneAgent LLM response: %s", resp)
        # VERY simple heuristics for demo — replace with model parsing
        if description:
            d = description.lower()
            if "warm" in d or "wheat" in d:
                return {"undertone": "warm", "category": "medium-warm"}
            if "cool" in d or "fair" in d:
                return {"undertone": "cool", "category": "fair-cool"}
        return {"undertone": "neutral", "category": "unknown"}
```

---

## backend/app/agents/body_type_agent.py

```python
# body_type_agent.py
from ..llm_client import LLMClient
from ..utils.logging_config import LOG

class BodyTypeAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def analyze(self, description: str | None) -> dict:
        prompt = f"Identify body type from: {description}"
        resp = self.llm.generate(prompt)
        LOG.debug("BodyTypeAgent LLM response: %s", resp)
        if description:
            d = description.lower()
            if "hourglass" in d:
                return {"body_type": "hourglass"}
            if "rectangle" in d:
                return {"body_type": "rectangle"}
            if "triangle" in d or "pear" in d:
                return {"body_type": "triangle"}
        return {"body_type": "unknown"}
```

---

## backend/app/agents/occasion_agent.py

```python
# occasion_agent.py
from ..llm_client import LLMClient
from ..utils.logging_config import LOG

class OccasionAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def parse(self, text: str) -> dict:
        prompt = f"Convert this occasion text to structured attributes: {text}"
        resp = self.llm.generate(prompt)
        LOG.debug("OccasionAgent LLM response: %s", resp)
        # Basic heuristics for demo
        text = text.lower()
        if any(x in text for x in ["wedding", "reception", "ceremony"]):
            return {"type": "wedding", "formality": "high", "season": "any"}
        if any(x in text for x in ["office", "interview"]):
            return {"type": "work", "formality": "high", "season": "any"}
        return {"type": "casual", "formality": "low", "season": "any"}
```

---

## backend/app/agents/color_harmony_agent.py

```python
# color_harmony_agent.py
from ..tools.palette_generator import generate_palette

class ColorHarmonyAgent:
    def __init__(self):
        pass

    def suggest(self, undertone: str, constraints: dict | None = None) -> dict:
        strategy = (constraints or {}).get("strategy", "complementary")
        palette = generate_palette(undertone, strategy=strategy)
        return palette
```

---

## backend/app/agents/outfit_generator_agent.py

```python
# outfit_generator_agent.py
from ..tools.shop_connector import search_items

class OutfitGeneratorAgent:
    def __init__(self):
        pass

    def generate(self, profile: dict, palette: dict, wardrobe_items: list | None = None) -> dict:
        # For demo: try to match wardrobe items by color or suggest shopping
        items = wardrobe_items or []
        suggestions = []
        if items:
            # filter by color word appearing in item color
            for it in items:
                if any(p[1:].lower() in it.get("color","") for p in palette.get("palette", [])):
                    suggestions.append(it)
        if not suggestions:
            # fallback to shop search
            suggestions = search_items("formal outfit")
        return {"outfits": suggestions[:3], "palette": palette}
```

---

## backend/app/agents/evaluator_agent.py

```python
# evaluator_agent.py
class EvaluatorAgent:
    def __init__(self):
        pass

    def score(self, outfit: dict) -> float:
        # Minimal rule-based scoring for demo
        score = 0.0
        if outfit.get("palette"):
            score += outfit["palette"].get("score", 0)
        if outfit.get("outfits"):
            score += 0.1
        return min(score, 1.0)
```

---

## backend/app/agents/coordinator_agent.py

```python
# coordinator_agent.py
from ..llm_client import LLMClient
from .skin_tone_agent import SkinToneAgent
from .body_type_agent import BodyTypeAgent
from .occasion_agent import OccasionAgent
from .color_harmony_agent import ColorHarmonyAgent
from .outfit_generator_agent import OutfitGeneratorAgent
from .evaluator_agent import EvaluatorAgent
from ..memory.memory_bank import MemoryBank
from ..memory.session_service import InMemorySessionService
from ..utils.logging_config import LOG
import uuid

class CoordinatorAgent:
    def __init__(self):
        llm = LLMClient()
        self.skin = SkinToneAgent(llm)
        self.body = BodyTypeAgent(llm)
        self.occasion = OccasionAgent(llm)
        self.color = ColorHarmonyAgent()
        self.outfit = OutfitGeneratorAgent()
        self.evaluator = EvaluatorAgent()
        self.mem = MemoryBank()
        self.sessi
```


