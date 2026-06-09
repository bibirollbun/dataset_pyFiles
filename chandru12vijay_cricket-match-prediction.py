# Match Prediction Agent

import random
from dataclasses import dataclass


# Simple Memory Storage

class MemoryStore:
    def __init__(self):
        self.data = {}

    def save(self, key, value):
        self.data[key] = value

    def dump(self):
        return self.data


# Agents

@dataclass
class StatsAgent:
    name: str = "StatsAgent"

    def run(self, match_info):
        random.seed(hash(str(match_info)) & 0xffffffff)
        return {"agent": self.name, "signal": round(random.uniform(0.15, 0.5), 3)}


@dataclass
class FormAgent:
    name: str = "FormAgent"

    def run(self, match_info):
        random.seed(hash(str(match_info) + "form") & 0xffffffff)
        return {"agent": self.name, "signal": round(random.uniform(-0.3, 0.4), 3)}


@dataclass
class WeatherAgent:
    name: str = "WeatherAgent"

    def run(self, match_info):
        random.seed(hash(str(match_info) + "weather") & 0xffffffff)
        return {"agent": self.name, "signal": round(random.uniform(-0.15, 0.2), 3)}


@dataclass
class TossAgent:
    name: str = "TossAgent"

    def run(self, match_info):
        random.seed(hash(str(match_info) + "toss") & 0xffffffff)
        return {"agent": self.name, "signal": round(random.uniform(-0.1, 0.15), 3)}


# Simple "LLM" 

class Reasoner:
    def synthesize(self, match_info, signals):
        total_score = sum(agent["signal"] for agent in signals)

        # Convert score to probability using sigmoid
        probability = 1 / (1 + (2.71828 ** (-2 * total_score)))

        winner = match_info["team_a"] if probability > 0.5 else match_info["team_b"]

        return {
            "prediction": winner,
            "confidence": round(probability * 100, 2),
            "signals": signals
        }



# Coordinator (MCP Logic)

class CricketAIAgent:
    def __init__(self):
        self.memory = MemoryStore()
        self.agents = [
            StatsAgent(),
            FormAgent(),
            WeatherAgent(),
            TossAgent()
        ]
        self.reasoner = Reasoner()

    def predict(self, match_info):
        signals = [agent.run(match_info) for agent in self.agents]
        result = self.reasoner.synthesize(match_info, signals)
        self.memory.save(match_info["id"], result)
        return result



# Test Run

match = {
    "id": "MATCH_001",
    "team_a": "India",
    "team_b": "Australia",
    "venue": "Wankhede Stadium"
}

agent_system = CricketAIAgent()
result = agent_system.predict(match)


# Display Output

print("ğŸ�� CRICKET MATCH PREDICTION\n")
print(f"Match: {match['team_a']} vs {match['team_b']} â€” {match['venue']}\n")
print(f"ğŸ”® Predicted Winner: **{result['prediction']}**")
print(f"ğŸ“Š Confidence Level: {result['confidence']}%\n")

print("ğŸ“Œ Agent Model Signals:")
for s in result["signals"]:
    print(f"   â†’ {s['agent']}: {s['signal']}")

print("\nğŸ§  Memory Store:", agent_system.memory.dump())

