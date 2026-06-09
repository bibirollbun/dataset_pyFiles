# ah-boo: The Multimodal Literacy Lens

## ðŸ”— Links
Demo Video: [Watch on YouTube](https://youtube.com/shorts/EQOS0D2I4io)
Try the App:[Launch ah-boo (Google AI Studio)](https://ai.studio/apps/drive/1e6-dCmc4g3W5-rriCXCMeWzZhGhpXji5)

Kaggle Competition: Google DeepMind - Vibe Code with Gemini 3 Pro

---

The Challenge
Illiteracy affects 763 million adults worldwide. Traditional learning tools are expensive, text-heavy, and often disconnected from the learner's physical reality.

The Solution: ah-boo
ah-boo is a "Multimodal Literacy Lens" that turns the entire world into a personalized sticker book. By combining Gemini 3 Pro's advanced vision capabilities with Gemini 2.0 Flash's low-latency audio, the app allows users to learn purely through interaction with their environmentâ€”no reading required.

How it works (The Tech Stack)
Vision:I use Gemini 3 Pro (Preview) to analyze objects with extreme cultural nuance.
Example: Scanning a Dr. Peppers can in Swahili mode doesn't just return "Soda", but the local term "Ka-leng" (Can), teaching relevant vocabulary.
Audio: Gemini 2.0 Flash Exp powers the language detection, identifying the learner's native tongue from a simple spoken greeting.
Hybrid TTS: A custom text-to-speech engine reads both the word and a generated "Fun Fact" to reinforce learning through storytelling.
Architecture: The entire app runs as a single-file executable in the browser, demonstrating the power of lightweight, AI-native development.

Key Features
1. Zero-UI Setup: No complex menus. Just say "Hello" in your own language, and the app adapts to that specific language.
2. Cultural Context: Recognizes objects in their local context, not just generic translations.
3. The Sticker Book: Gamifies the experience by collecting scanned items in a visual library.

**ah-boo** proves that the future of education isn't in textbooks, but in the palm of your hand, powered by Gemini.

import pandas as pd
# Lager en "falsk" resultatfil for Ã¥ gjÃ¸re Kaggle fornÃ¸yd
df = pd.DataFrame({'id': [1], 'value': [0]})
df.to_csv('submission.csv', index=False)
print("Dummy submission file created!")


import pandas as pd
# Lager en "falsk" resultatfil for Ã¥ gjÃ¸re Kaggle fornÃ¸yd
df = pd.DataFrame({'id': [1], 'value': [0]})
df.to_csv('submission.csv', index=False)
print("Dummy submission file created!")

