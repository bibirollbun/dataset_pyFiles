# Cell 1 â€” Setup & imports
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from pprint import pprint

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("Setup complete. Ready to train the toy agent.")


# Cell 2 â€” Toy dataset (intents + examples)
data = [
    ("create task buy groceries", "create_task"),
    ("add a new task for laundry", "create_task"),
    ("remind me to call mom tomorrow", "create_task"),
    ("delete task buy groceries", "delete_task"),
    ("remove the laundry task", "delete_task"),
    ("mark laundry as done", "complete_task"),
    ("i finished the report", "complete_task"),
    ("what tasks do I have?", "list_tasks"),
    ("show my to-dos", "list_tasks"),
    ("hello", "smalltalk"),
    ("hey there", "smalltalk"),
    ("what's your name", "smalltalk"),
    ("how many tasks are pending", "list_tasks"),
    ("help me plan a trip", "help_request"),
    ("i need assistance with booking", "help_request"),
    ("thanks a lot", "gratitude"),
    ("thank you!", "gratitude"),
    ("cancel my trip reminder", "delete_task"),
    ("schedule meeting with team", "create_task"),
    ("reschedule the meeting", "update_task"),
    ("update task meeting time", "update_task"),
    ("what can you do?", "capabilities"),
    ("explain your features", "capabilities"),
]

df = pd.DataFrame(data, columns=["text", "intent"])
df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
df.head(10)


# Cell 3 â€” Split and build pipeline
X = df["text"].values
y = df["intent"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y
)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=1)),
    ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED))
])

pipe.fit(X_train, y_train)
print("Model trained on toy dataset.")


# Cell 4 â€” Evaluation metrics
y_pred = pipe.predict(X_test)
print("Classification report on test set:\n")
print(classification_report(y_test, y_pred, zero_division=0))

print("Confusion matrix (rows=true, cols=pred):")
print(confusion_matrix(y_test, y_pred))


# Cell 5 â€” Define agent + handlers (simple multi-agent router)
class SmartTaskAgent:
    def __init__(self, model_pipeline):
        self.model = model_pipeline
        # In-memory task store for demo purposes
        self.tasks = []
    
    def predict_intent(self, text):
        intent = self.model.predict([text])[0]
        proba = max(self.model.predict_proba([text])[0])
        return intent, proba
    
    # Handlers simulate separate sub-agents
    def handle_create_task(self, text):
        # Very simple extraction: take last 4 words as task title if short
        parts = text.split()
        title = " ".join(parts[-4:]) if len(parts) >= 3 else text
        self.tasks.append({"title": title, "status": "pending"})
        return f"Task created: '{title}'. You now have {len(self.tasks)} task(s)."
    
    def handle_delete_task(self, text):
        if not self.tasks:
            return "No tasks to delete."
        # simple heuristics: delete last task
        removed = self.tasks.pop()
        return f"Removed task: '{removed['title']}'."
    
    def handle_complete_task(self, text):
        if not self.tasks:
            return "No tasks to complete."
        # mark last as complete
        self.tasks[-1]["status"] = "done"
        return f"Marked '{self.tasks[-1]['title']}' as done."
    
    def handle_list_tasks(self, text):
        if not self.tasks:
            return "You have no tasks."
        lines = [f"- [{t['status']}] {t['title']}" for t in self.tasks]
        return "Your tasks:\n" + "\n".join(lines)
    
    def handle_help_request(self, text):
        return ("I can create, list, update, complete, and delete tasks. "
                "Try: 'create task buy milk' or 'what tasks do I have?'.")
    
    def handle_update_task(self, text):
        if not self.tasks:
            return "No tasks to update."
        # naive update: append note
        self.tasks[-1]["title"] += " (updated)"
        return f"Updated last task: '{self.tasks[-1]['title']}'"
    
    def handle_smalltalk(self, text):
        return "Hi! I'm SmartTask Agent â€” I help with simple task automation."
    
    def handle_gratitude(self, text):
        return "You're welcome! ðŸ˜Š"
    
    def handle_capabilities(self, text):
        return ("I demonstrate intent classification + handler routing. "
                "Extend me with real NLU, memory, and tool calls.")
    
    def default_handler(self, text):
        return ("Sorry, I didn't understand that. Try 'create task ...' or 'what tasks do I have?'.")
    
    def route(self, text):
        intent, confidence = self.predict_intent(text)
        # Handler routing map
        handlers = {
            "create_task": self.handle_create_task,
            "delete_task": self.handle_delete_task,
            "complete_task": self.handle_complete_task,
            "list_tasks": self.handle_list_tasks,
            "help_request": self.handle_help_request,
            "update_task": self.handle_update_task,
            "smalltalk": self.handle_smalltalk,
            "gratitude": self.handle_gratitude,
            "capabilities": self.handle_capabilities,
        }
        handler = handlers.get(intent, self.default_handler)
        response = handler(text)
        metadata = {"intent": intent, "confidence": float(confidence)}
        return response, metadata

# Create agent instance
agent = SmartTaskAgent(pipe)
print("Agent initialized. Ready to route requests.")


# Cell 6 â€” Demo interactions (batch examples)
examples = [
    "create task buy groceries tomorrow",
    "what tasks do I have?",
    "mark laundry as done",
    "add a new task for project report",
    "show my to-dos",
    "delete task buy groceries",
    "thanks"
]

for ex in examples:
    resp, meta = agent.route(ex)
    print(f">>> {ex}")
    print(resp)
    print(f"  -> intent: {meta['intent']}, confidence: {meta['confidence']:.2f}")
    print()


# Cell 7 â€” Interactive demo: run a simple REPL loop (press stop to end in Kaggle)
def interactive_demo(agent, n_steps=6):
    print("Interactive demo â€” try typing short commands (e.g., 'create task read book').")
    for i in range(n_steps):
        text = input(f"[You {i+1}/{n_steps}] ")
        resp, meta = agent.route(text)
        print(f"[Agent] {resp}  (intent: {meta['intent']}, conf: {meta['confidence']:.2f})")
    print("Demo session ended. You can run again or extend the agent.")

# Run interactive_demo(agent) if you want interactive input in the notebook environment.
# To execute in Kaggle's run environment, uncomment the next line:
# interactive_demo(agent, n_steps=6)


## Next steps (how to make this production-grade)
- Replace the toy classifier with an LLM or embedding+semantic search for richer NLU.
- Add persistent storage (DB) for tasks and user sessions.
- Implement authentication / multi-user support.
- Create a more robust NER-based task-extraction or use rule-based parsers.
- Add observability: logs, traces, evaluation dashboards.
- Convert handlers into separate microservices (true multi-agent deployment).

