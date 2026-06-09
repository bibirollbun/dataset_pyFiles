# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# DreamCatcher AI - Fully Runnable Mock Version
# No API key needed; ready for VS Code / Colab

def interpret_dream(dream):
    # Mocked interpretation
    return f"ğŸŒ³ Interpretation: Seeing '{dream}' symbolizes growth, reflection, and your subconscious mind exploring new ideas."

def generate_story(dream):
    # Mocked story
    return f"ğŸ“– Story: In a land where dreams come alive, '{dream}' unfolded into a magical adventure full of wonder and self-discovery."

def generate_insights(dream):
    # Mocked insights
    return f"ğŸ’¡ Insights / Ideas: Consider how '{dream}' reflects your current challenges and aspirations. Journaling or reflection might bring clarity."

def dreamcatcher_ai():
    print("ğŸŒ™ Welcome to DreamCatcher AI ğŸŒ™")
    print("Enter your dream, or type 'exit' to quit.\n")
    
    while True:
        dream = input("> ").strip()
        if dream.lower() == "exit":
            print("Goodbye! Keep dreaming âœ¨")
            break
        if not dream:
            print("âš ï¸� Please enter a valid dream.")
            continue

        # Display interpretation
        print("\nâœ¨ Interpreting your dream...")
        print("\nğŸ“� Interpretation:\n", interpret_dream(dream))

        # Display story
        print("\nâœ¨ Generating story...")
        print("\nğŸ“– Story:\n", generate_story(dream))

        # Display insights
        print("\nâœ¨ Extracting insights...")
        print("\nğŸ’¡ Insights / Ideas:\n", generate_insights(dream))

        # Optionally save output
        save = input("\nDo you want to save this output to a file? (yes/no): ").strip().lower()
        if save == "yes":
            with open("dream_output.txt", "a", encoding="utf-8") as f:
                f.write(f"Dream: {dream}\n")
                f.write(f"Interpretation: {interpret_dream(dream)}\n")
                f.write(f"Story: {generate_story(dream)}\n")
                f.write(f"Insights: {generate_insights(dream)}\n")
                f.write("="*50 + "\n")
            print("âœ… Saved successfully!\n")
        else:
            print("\nNext dream!\n")

if __name__ == "__main__":
    dreamcatcher_ai()


