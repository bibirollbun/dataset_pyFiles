# Narrative Agency Agent - Kaggle Notebook (single-file)
# Author: Shweta
# Purpose: Lightweight, runnable implementation of the Narrative Agency Agent
# Instructions:
#  - On Kaggle: create a new notebook, upload this file, then use `%run Narrative_Agency_Agent_Kaggle_Notebook.py`
#  - On GitHub: place in a repo as `notebooks/` or root; use as a script or convert to .ipynb

# %%
"""Setup & Imports"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Tuple
import json
import math
import random
import pprint

# Optional: visualization
import matplotlib.pyplot as plt

# %%
"""Core data structures"""
@dataclass
class SceneInput:
    scene_text: str
    decision_text: str

@dataclass
class PathOutcome:
    title: str
    text: str
    agency_score: float
    ethical_tension: float
    coherence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

# %%
"""Simple Memory Store to preserve character arcs and thematic threads"""
class MemoryStore:
    def __init__(self):
        # store keyed by character or theme name
        self.characters: Dict[str, Dict[str, Any]] = {}
        self.themes: Dict[str, Dict[str, Any]] = {}
        self.history: List[Dict[str,Any]] = []

    def add_character_state(self, name: str, state: Dict[str,Any]):
        self.characters.setdefault(name, {}).update(state)

    def add_theme(self, theme: str, info: Dict[str,Any]):
        self.themes.setdefault(theme, {}).update(info)

    def log(self, entry: Dict[str,Any]):
        self.history.append(entry)

    def retrieve_context(self) -> Dict[str,Any]:
        return {
            'characters': self.characters,
            'themes': self.themes,
            'history': self.history[-10:],
        }

# %%
"""Outcome Generator (mocked, deterministic-ish)"""
class OutcomeGenerator:
    def __init__(self, rng_seed: int = 42):
        random.seed(rng_seed)

    def generate_alternatives(self, scene: SceneInput, memory: MemoryStore) -> List[PathOutcome]:
        # For a production system, replace this with a language model call.
        # Here we craft three archetypal paths based on the decision text.
        base = scene.scene_text
        decision = scene.decision_text.lower()

        # Simple heuristics to create titles and text
        paths = []
        if 'expose' in decision or 'exposure' in decision or 'truth' in decision:
            paths.append(('Exposure Breeds Betrayal', f"{base} Riya exposes the dossier; repercussions spiral."))
            paths.append(('Protection at a Cost', f"{base} Riya shields Maya; loyalties are tested."))
            paths.append(('Truth Delayed', f"{base} Riya hesitates; the truth waits like a wound."))
        else:
            # generic three-path template
            paths.append(('Decisive Action', f"{base} A decisive action reshapes the scene."))
            paths.append(('Compromise', f"{base} A compromise creates uneasy peace."))
            paths.append(('Avoidance', f"{base} Delay and avoidance introduce tension."))

        outcomes = []
        for i,(title,text) in enumerate(paths):
            # simple scoring heuristics
            agency = round(1.0 - (i * 0.25) + random.uniform(-0.05, 0.05), 3)
            ethical = round(0.5 + (i * 0.2) + random.uniform(-0.05, 0.05),3)
            coherence = round(0.8 - (i * 0.1) + random.uniform(-0.05, 0.05),3)
            outcomes.append(PathOutcome(title=title, text=text, agency_score=agency, ethical_tension=ethical, coherence=coherence))
        return outcomes

# %%
"""Ethics Evaluator: scores outcomes for agency, ethical tension, coherence."""
class EthicsEvaluator:
    def __init__(self):
        pass

    def evaluate(self, outcome: PathOutcome, context: Dict[str,Any]) -> PathOutcome:
        # In a real implementation this would call a reasoning model to score.
        # Here we slightly adjust scores based on context (mock logic).
        char_count = len(context.get('characters',{}))
        adj = min(0.1, 0.02 * char_count)
        outcome.agency_score = max(0.0, min(1.0, outcome.agency_score + adj))
        outcome.coherence = max(0.0, min(1.0, outcome.coherence + adj/2))
        # ethical tension might increase if many themes are present
        theme_count = len(context.get('themes',{}))
        outcome.ethical_tension = max(0.0, min(1.0, outcome.ethical_tension + 0.01*theme_count))
        return outcome

# %%
"""Orchestrator: wires agents together, logs, and returns output"""
class Orchestrator:
    def __init__(self, memory: MemoryStore, generator: OutcomeGenerator, evaluator: EthicsEvaluator):
        self.memory = memory
        self.generator = generator
        self.evaluator = evaluator
        self.logs: List[Dict[str,Any]] = []

    def run(self, scene: SceneInput) -> Dict[str,Any]:
        context = self.memory.retrieve_context()
        self._log_event('input_received', {'scene': scene.scene_text, 'decision': scene.decision_text})

        raw_paths = self.generator.generate_alternatives(scene, self.memory)
        scored = []
        for p in raw_paths:
            evaluated = self.evaluator.evaluate(p, context)
            scored.append(evaluated)
            # log each path
            self._log_event('path_generated', {
                'title': evaluated.title,
                'agency': evaluated.agency_score,
                'ethical_tension': evaluated.ethical_tension,
                'coherence': evaluated.coherence
            })

        # recommendation: highest combined transform score
        best = max(scored, key=lambda x: (x.agency_score * 0.5 + x.ethical_tension * 0.3 + x.coherence * 0.2))
        self._log_event('recommendation', {'recommended': best.title})

        result = {
            'input': {'scene': scene.scene_text, 'decision': scene.decision_text},
            'paths': [p.__dict__ for p in scored],
            'recommendation': best.__dict__,
            'logs': self.logs
        }
        return result

    def _log_event(self, kind: str, data: Dict[str,Any]):
        entry = {'kind': kind, 'data': data}
        self.logs.append(entry)
        self.memory.log(entry)

# %%
"""Demo: run the system with the example provided in the project description."""
def demo_example():
    memory = MemoryStore()
    # seed some memory (characters/themes) to show how it affects scores
    memory.add_character_state('Riya', {'role': 'protagonist', 'arc': 'seeker'})
    memory.add_character_state('Maya', {'role': 'mentor', 'arc': 'keeper_of_secrets'})
    memory.add_theme('truth_vs_loyalty', {'importance': 0.9})

    generator = OutcomeGenerator(rng_seed=2025)
    evaluator = EthicsEvaluator()
    orchestrator = Orchestrator(memory=memory, generator=generator, evaluator=evaluator)

    scene = SceneInput(
        scene_text="Riya confronts her mentor about a sealed dossier.",
        decision_text="Should she expose the truth or protect Maya?"
    )

    result = orchestrator.run(scene)
    print('\n=== Recommendation ===')
    pprint.pprint(result['recommendation'])
    print('\n=== All Paths ===')
    for p in result['paths']:
        pprint.pprint(p)

    # simple bar chart of scores
    titles = [p['title'] for p in result['paths']]
    agency = [p['agency_score'] for p in result['paths']]
    ethics = [p['ethical_tension'] for p in result['paths']]
    coh = [p['coherence'] for p in result['paths']]

    x = range(len(titles))
    plt.figure(figsize=(8,4))
    plt.plot(x, agency, marker='o', label='Agency')
    plt.plot(x, ethics, marker='o', label='Ethical Tension')
    plt.plot(x, coh, marker='o', label='Coherence')
    plt.xticks(x, titles, rotation=20)
    plt.title('Path Scores')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Save result to JSON for reproducibility
    with open('nai_agent_result.json', 'w') as f:
        json.dump(result, f, indent=2)
    print('\nSaved results to nai_agent_result.json')

# %%
if __name__ == '__main__':
    demo_example()


