import google.generativeai as genai

# ============================
# 1. GOOGLE API CONFIG
# ============================

API_KEY = "YOUR_API_KEY_HERE"   # <<< REPLACE THIS

genai.configure(api_key=API_KEY)

model = genai.GenerativeModel("gemini-1.5-flash")

# ============================
# 2. Scene Parser (LLM-powered)
# ============================

class SceneParser:

    def parse_scene(self, text):

        prompt = f"""
        You are a film scene analysis expert.

        Extract structured information from the following movie scene text:

        Scene:
        {text}

        Return JSON ONLY in this exact format:

        {{
            "characters": ["name1", "name2"],
            "actions": ["verb1", "verb2"],
            "emotions": ["emotion1", "emotion2"],
            "setting": "location"
        }}
        """

        response = model.generate_content(prompt)

        # Convert JSON text to Python dict safely
        try:
            return eval(response.text)
        except:
            return {"characters": [], "actions": [], "emotions": [], "setting": ""}


# ==================================
# 3. Camera Angle Generator (LLM)
# ==================================

class CameraEngine:

    def generate_shot_list(self, scene):

        prompt = f"""
        You are a Hollywood cinematographer.
        Based on the following extracted scene data, generate a professional camera shot list.

        Scene Data:
        {scene}

        Rules:
        - Include establishing shot (if setting exists)
        - Use emotion-based shots (close-ups, push-ins, shaky cam)
        - Action-based shots (tracking, wide, dynamic)
        - Dialogue shots if characters > 1
        - Give 6â€“10 shots total
        - Number them (1, 2, 3...)

        Output ONLY the list of shots.
        """

        response = model.generate_content(prompt)
        return response.text.split("\n")


# =======================
# 4. Main CineAngle Agent
# =======================

class CineAngleAI:

    def __init__(self):
        self.parser = SceneParser()
        self.engine = CameraEngine()

    def generate_camera_plan(self, text):
        scene = self.parser.parse_scene(text)
        shots = self.engine.generate_shot_list(scene)
        return scene, shots


# ===============================
# 5. MAIN PROGRAM (Demo Example)
# ===============================

if __name__ == "__main__":

    ai = CineAngleAI()

    print("\nðŸŽ¥ CINEANGLEAI â€” GOOGLE AI VERSION\n")

    scene_description = """
    John enters the abandoned warehouse slowly.
    The atmosphere is tense. He looks terrified.
    Sarah follows behind with a flashlight.
    Suddenly, a loud crash echoes and both start running.
    """

    print("Input Scene:")
    print(scene_description)

    scene, shots = ai.generate_camera_plan(scene_description)

    print("\n=====================")
    print(" SCENE ANALYSIS")
    print("=====================")
    print(scene)

    print("\n=====================")
    print(" CAMERA SHOT LIST")
    print("=====================")
    for s in shots:
        if s.strip():
            print(s)


