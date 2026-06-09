# -------------------------------------------
# Memory Class
# -------------------------------------------
class SessionMemory:
    def __init__(self):
        self.history = []

    def add(self, role, message):
        self.history.append({"role": role, "message": message})

    def get(self):
        return self.history


# -------------------------------------------
# Agent 1: Information Gathering Agent
# -------------------------------------------
class InfoAgent:
    def process(self, user_input):
        return f"Information received: '{user_input}'. I will gather disaster-related details."


# -------------------------------------------
# Agent 2: Disaster Analysis Agent
# -------------------------------------------
class AnalysisAgent:
    def process(self, user_input):
        return f"Analyzing situation: Based on input '{user_input}', preliminary assessment prepared."


# -------------------------------------------
# Agent 3: Map Agent (Dummy map generator)
# -------------------------------------------
class MapAgent:
    def process(self, user_input):
        return f"Map generated (simulation) for location in: '{user_input}'. (Note: Kaggle doesn't support real maps unless API keys are added.)"


# -------------------------------------------
# Coordinator Agent – decides who responds
# -------------------------------------------
class CoordinatorAgent:
    def __init__(self):
        self.info = InfoAgent()
        self.analysis = AnalysisAgent()
        self.map_agent = MapAgent()

    def route(self, user_input):
        text = user_input.lower()
        if "map" in text or "location" in text:
            return self.map_agent.process(user_input)
        elif "analyze" in text or "analysis" in text:
            return self.analysis.process(user_input)
        else:
            return self.info.process(user_input)


# -------------------------------------------
# MAIN APP
# -------------------------------------------
def main():
    memory = SessionMemory()
    coordinator = CoordinatorAgent()

    print("=== Disaster Response Assistant (Kaggle Version) ===\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Assistant: Goodbye! Stay safe!")
            break

        memory.add("user", user_input)

        output = coordinator.route(user_input)
        memory.add("assistant", output)

        print("Assistant:", output)


# Run the assistant
main()


