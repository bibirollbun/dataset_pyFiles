from dataclasses import dataclass

@dataclass
class UserState:
    """Very small state to keep track of the user."""
    energy: int = 5          # 0â€“10
    overwhelm: int = 0       # 0â€“10
    hunger: int = 0          # 0â€“10
    last_tool: str | None = None


class SavePointAgent:
    """
    Minimal, non-intrusive agent.
    In a real ADK setup, each 'action_*' would be exposed as a Tool.
    """
    def __init__(self):
        self.state = UserState()

    def observe_text(self, text: str) -> None:
        """Update internal state from a short user message."""
        t = text.lower()

        # Energy & overwhelm
        if "tired" in t or "ç–²ã‚Œã�Ÿ" in t:
            self.state.energy = max(self.state.energy - 2, 0)
            self.state.overwhelm = min(self.state.overwhelm + 1, 10)

        if "ã‚‚ã�†ã‚„ã� " in t or "ã‚‚ã�†ã�„ã‚„" in t or "ã‚‚ã�†ç„¡ç�†" in t or "done" in t:
            self.state.overwhelm = min(self.state.overwhelm + 3, 10)

        # Hunger
        if ("hungry" in t or "ã�Šã�ªã�‹ã�™ã�„ã�Ÿ" in t or "ã�Šè…¹ã�™ã�„ã�Ÿ" in t
                or "ã�”ã�¯ã‚“" in t or "ã�”é£¯" in t):
            self.state.hunger = min(self.state.hunger + 4, 10)

    # --- Tools (would map to ADK tools) ---------------------------------
    def action_play_calm_audio(self):
        self.state.last_tool = "calm_audio"
        return "ğŸ�§ Calm audio: play healing music playlist."

    def action_show_soft_visuals(self):
        self.state.last_tool = "soft_visuals"
        return "ğŸ�¬ Soft visuals: show slow light & color animation."

    def action_open_simple_puzzle(self):
        self.state.last_tool = "simple_puzzle"
        return "ğŸ§© Open a very simple puzzle game (solitaire style)."

    def action_open_rage_bin(self):
        self.state.last_tool = "rage_bin"
        return "ğŸ—‘ï¸� Open rage bin: type anything, then it will be burned."

    def action_open_food_helper(self):
        self.state.last_tool = "food_helper"
        self.state.hunger = max(self.state.hunger - 3, 0)
        return "ğŸ�± Food helper: open your comfort food delivery app with a few favorite options."

    def action_do_nothing(self):
        self.state.last_tool = "idle"
        return "â€¦Stay quiet. Just keep the save point softly glowing."

    # --------------------------------------------------------------------
    def decide_next_action(self, text: str) -> str:
        """
        Routing logic.
        In ADK terms: policy + tool selection.
        """
        self.observe_text(text)

        # 1) Very hungry + low-ish energy â†’ suggest food helper
        if self.state.hunger >= 4:
            return self.action_open_food_helper()

        # 2) High overwhelm â†’ rage bin
        if self.state.overwhelm >= 7:
            return self.action_open_rage_bin()

        # 3) Very low energy â†’ calm audio
        if self.state.energy <= 2:
            return self.action_play_calm_audio()

        # 4) Explicit wishes
        if "ã‚²ãƒ¼ãƒ " in text or "game" in text:
            return self.action_open_simple_puzzle()
        if "è¦‹ã�Ÿã�„" in text or "video" in text:
            return self.action_show_soft_visuals()

        # 5) Default: stay quietly with the user
        return self.action_do_nothing()


# --- demo ---------------------------------------------------------------

agent = SavePointAgent()

samples = [
    "ã�Šã�ªã�‹ã�™ã�„ã�Ÿâ€¦ã‚‚ã�†ç„¡ç�†â€¦",
    "ç–²ã‚Œã�Ÿã€‚",
    "ã�ªã‚“ã�‹ã‚²ãƒ¼ãƒ ã�—ã�Ÿã�„",
    "ã�¼ãƒ¼ã�£ã�¨å‹•ç”»è¦‹ã�Ÿã�„",
    "â€¦â€¦ï¼ˆä½•ã‚‚è¨€ã‚�ã�ªã�„ï¼‰",
]

for s in samples:
    action = agent.decide_next_action(s)
    print(f"User: {s}\nAgent: {action}\nState: {agent.state}\n---")


import pandas as pd

# Simple dummy submission file just to satisfy competition requirement
df = pd.DataFrame(
    [
        {
            "track": "Agents for Good",
            "notebook": "SavePoint: A Non-Intrusive Emotional Checkpoint Agent",
            "author": "Manami M",
        }
    ]
)

df.to_csv("/kaggle/working/submission.csv", index=False)
print("Created submission.csv")

