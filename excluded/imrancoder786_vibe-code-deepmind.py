
import json
from pathlib import Path

output = {
    "project": "Global Life-Saver: AI Disaster Assistant",
    "model": "Gemini 3 Pro (Google AI Studio)",
    "description": "Notebook documenting the architecture, reasoning pipeline, and demo links for the AI-powered disaster assistant.",
    "app link" :"https://ai.studio/apps/drive/1k8eKqPyARfQXC9s-Rk-v1pNUiws2iYY8?fullscreenApplet=true",
    "github link" : "https://github.com/imrancoder786/Global-Life-Saver.git",
    "status": "success"
}

path = Path("/kaggle/working/submission.json")
with open(path, "w") as f:
    json.dump(output, f, indent=2)

print("Submission file created at:", path)

