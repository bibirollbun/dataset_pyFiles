import json
from pathlib import Path

# ---------------------------------------------------------
# Create a minimal Kaggle submission artifact
# ---------------------------------------------------------

output = {
    "project": "AI Product Manager Copilot",
    "platform": "Google AI Studio",
    "model": "Gemini 3 Pro",
    "description": (
        "This notebook documents the concept, architecture, and demo links "
        "for an AI-powered Product Manager assistant built using Vibe Coding "
        "with Gemini 3 Pro during the Google DeepMind x Kaggle sprint."
    ),
    "features": [
        "PRD and roadmap generation",
        "User stories and sprint planning",
        "Competitor and market analysis",
        "Beginner-friendly PM guidance",
        "Multimodal reasoning (text + voice)"
    ],
    "demo_links": {
        "video_demo": "https://youtu.be/u5zqUnTc_QQ",
        "ai_studio_app": "https://ai.studio/apps/drive/18f8YBrBM6-o4mt9Id_hEFbcvE7SNiIP-?fullscreenApplet=true"
    },
    "status": "success"
}

# Save artifact
output_path = Path("/kaggle/working/submission.json")
with open(output_path, "w") as f:
    json.dump(output, f, indent=2)

print("Submission artifact created at:", output_path)


