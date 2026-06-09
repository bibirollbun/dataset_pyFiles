# InstalaciÃ³n de dependencias necesarias
!pip install -q -U immutabledict sentencepiece

import sys
import os
import torch
import contextlib



# Clonar gemma_pytorch si no existe
if not os.path.exists("/kaggle/working/gemma_pytorch"):
    !git clone https://github.com/google/gemma_pytorch.git /kaggle/working/gemma_pytorch

sys.path.append("/kaggle/working/gemma_pytorch/")
from gemma.config import get_config_for_2b
from gemma.model import GemmaForCausalLM
from gemma.tokenizer import Tokenizer

VARIANT = "2b-it"
MACHINE_TYPE = "cpu"
weights_dir = "/kaggle/input/gemma/pytorch/2b-it/2"

@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(torch.float)

model_config = get_config_for_2b()
model_config.tokenizer = os.path.join(weights_dir, "tokenizer.model")
model_config.quant = "quant" in VARIANT

device = torch.device(MACHINE_TYPE)
with _set_default_tensor_type(model_config.get_dtype()):
    model = GemmaForCausalLM(model_config)
    ckpt_path = os.path.join(weights_dir, f'gemma-{VARIANT}.ckpt')
    model.load_weights(ckpt_path)
    model = model.to(device).eval()



import re
import json
import itertools

# Rutas de archivos
KAGGLE_AGENT_PATH = "/kaggle_simulations/agent/"
if os.path.exists(KAGGLE_AGENT_PATH):
    WEIGHTS_PATH = os.path.join(KAGGLE_AGENT_PATH, "gemma/pytorch/2b-it/2")
    KEYWORDS_PATH = os.path.join(KAGGLE_AGENT_PATH, "llm_20_questions/keywords.py")
else:
    WEIGHTS_PATH = "/kaggle/input/gemma/pytorch/2b-it/2"
    KEYWORDS_PATH = "/kaggle/input/llm-20-questions/llm_20_questions/keywords.py"

# Parseo robusto de KEYWORDS_JSON
with open(KEYWORDS_PATH, "r", encoding="utf-8") as f:
    content = f.read()
match = re.search(r"KEYWORDS_JSON\s*=\s*[ru]?([\"\']{3})(.+?)\1", content, re.DOTALL)
if not match:
    raise ValueError("No se encontrÃ³ el bloque KEYWORDS_JSON en keywords.py")
KEYWORDS = json.loads(match.group(2).strip())
CATEGORY_WORDS = {k["category"]: [w["keyword"] for w in k["words"]] for k in KEYWORDS}

def interleave(a, b):
    """Intercala dos listas manteniendo orden."""
    return [e for p in itertools.zip_longest(a, b) for e in p if e]

@contextlib.contextmanager
def _default_dtype(dtype):
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(torch.float)



class Formatter:
    """Convierte turnos en prompt estilo Gemma."""
    ST, ET = "<start_of_turn>", "<end_of_turn>"

    def __init__(self, system_prompt=None, few_shot=None):
        self.system_prompt = system_prompt
        self.few_shot = few_shot or []
        self.reset()

    def reset(self):
        self.buf = ""
        if self.system_prompt:
            self.user(self.system_prompt)
        if self.few_shot:
            self.apply(self.few_shot, start="user")

    def user(self, text):
        self.buf += f"{self.ST}user\n{text}{self.ET}\n"
        return self

    def model(self, text):
        self.buf += f"{self.ST}model\n{text}{self.ET}\n"
        return self

    def start_model(self):
        self.buf += f"{self.ST}model\n"
        return self

    def apply(self, turns, start="user"):
        cycle = itertools.cycle([self.user, self.model]) if start == "user" else itertools.cycle([self.model, self.user])
        for fn, t in zip(cycle, turns):
            fn(t)



class GemmaAgent:
    def __init__(self, device="cpu", variant="2b-it", system_prompt=None, few_shot=None):
        self.device = torch.device(device)
        self.fmt = Formatter(system_prompt, few_shot)
        print("Initializing model")
        cfg = get_config_for_2b()
        cfg.tokenizer = os.path.join(WEIGHTS_PATH, "tokenizer.model")
        cfg.quant = "quant" in variant
        with _default_dtype(cfg.get_dtype()):
            self.model = GemmaForCausalLM(cfg)
            self.model.load_weights(os.path.join(WEIGHTS_PATH, f"gemma-{variant}.ckpt"))
            self.model = self.model.to(self.device).eval()

    def _llm(self, prompt, max_new=64):
        return self.model.generate(prompt, device=self.device, output_len=max_new, temperature=0.7)

    @staticmethod
    def _extract_kw(text: str) -> str:
        m = re.search(r"\*\*([\w\s\-]+?)\*\*", text)
        return m.group(1).strip().lower() if m else ""

class Questioner(GemmaAgent):
    def __call__(self, obs):
        self.fmt.reset()
        self.fmt.user(
            f"You are the Questioner in 20 Questions. The keyword belongs to the category **{obs.category}**."
        )
        self.fmt.apply(interleave(obs.questions, obs.answers), start="model")
        if obs.turnType == "ask":
            asked = "\n- " + "\n- ".join(obs.questions) if obs.questions else " none"
            self.fmt.user(
                f"You have already asked:{asked}\n"
                "Ask ONE NEW yesâ€‘orâ€‘no question, different in topic and wording, that helps narrow the possibilities logically."
            )
        else:
            self.fmt.user(
                "Based on all previous questions and answers, guess the keyword now. "
                "Reply ONLY with the keyword wrapped in **double asterisks** (e.g. **laptop**). "
                "Avoid repeating previous guesses or answering randomly. Think carefully."
            )
        self.fmt.start_model()
        response = self._llm(self.fmt.buf)
        return self._parse(response, obs.turnType)

    def _parse(self, text, ttype):
        if ttype == "ask":
            m = re.search(r"([A-Z][^?]*\?)", text, re.S)
            return m.group(1).strip() if m else "Is it an object?"
        return self._extract_kw(text)

class Answerer(GemmaAgent):
    def __call__(self, obs):
        self.fmt.reset()
        kw = obs.keyword.lower(); cat = obs.category.lower()
        self.fmt.user(
            f"You are the Answerer. The keyword is **{kw}** (category: {cat}). Respond truthfully."
        )
        self.fmt.apply(interleave(obs.questions, obs.answers), start="user")
        self.fmt.user("Answer only with **yes** or **no**.")
        self.fmt.start_model()
        ans = self._extract_kw(self._llm(self.fmt.buf))
        return "yes" if ans == "yes" else "no"



FEW_EXAMPLES = [
    "Is it a living thing?", "**yes**",
    "Is it a mammal?", "**yes**",
    "Is it commonly kept as a pet?", "**yes**",
    "Now guess the keyword.", "**dog**", "Correct!",
    "Is it edible?", "**yes**",
    "Is it a fruit?", "**yes**",
    "Is it yellow?", "**yes**",
    "Now guess the keyword.", "**banana**", "Correct!",
    "Is it a place?", "**yes**",
    "Is it in South America?", "**yes**",
    "Is Spanish an official language there?", "**yes**",
    "Now guess the keyword.", "**argentina**", "Correct!",
    "Is it manâ€‘made?", "**yes**",
    "Is it electronic?", "**yes**",
    "Is it primarily used for computing?", "**yes**",
    "Now guess the keyword.", "**laptop**", "Correct!"
]

_agents = {}

def get_agent(role):
    if role not in _agents:
        if role == "questioner":
            _agents[role] = Questioner(
                system_prompt="You're playing 20 Questions. Ask strategic questions to identify the keyword efficiently.",
                few_shot=FEW_EXAMPLES,
            )
        elif role == "answerer":
            _agents[role] = Answerer(system_prompt="You are the Answerer. Respond accurately and concisely.")
        else:
            raise ValueError(role)
    return _agents[role]

def agent_fn(obs, cfg):
    role = "questioner" if obs.turnType in ("ask", "guess") else "answerer"
    return get_agent(role)(obs)



from types import SimpleNamespace

def simulate_game(keyword="france", category="country", max_turns=10):
    questions, answers = [], []
    for turn in range(max_turns):
        obs_q = SimpleNamespace(turnType="ask", questions=questions, answers=answers, keyword=keyword, category=category)
        question = agent_fn(obs_q, {})
        print(f"\nğŸ§  Pregunta {turn+1}: {question}")
        questions.append(question)
        obs_a = SimpleNamespace(turnType="answer", questions=questions, answers=answers, keyword=keyword, category=category)
        answer = agent_fn(obs_a, {})
        print(f"ğŸ¤– Respuesta: {answer}")
        answers.append(answer)
        obs_g = SimpleNamespace(turnType="guess", questions=questions, answers=answers, keyword=keyword, category=category)
        guess = agent_fn(obs_g, {})
        print(f"ğŸ�¯ Adivinanza: {guess}")
        if guess.lower() == keyword.lower():
            print("âœ… Â¡AdivinÃ³ correctamente!")
            return
    print(f"â�Œ No adivinÃ³. La palabra era: {keyword}")

# Ejemplo de uso:
simulate_game("andorra", "country")


