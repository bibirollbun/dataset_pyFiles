import pandas as pd
import random

def count_words(text):
    """Counts the number of words in a text."""
    return len(text.split())

def generate_controversial_essay(topic, intensity="medium"):
    """Generates a controversial essay with random phrasing based on intensity."""
    
    openings = {
        "mild": [
            f"Some argue that {topic} is a balanced issue, but the reality is more complex.",
            f"The debate about {topic} has been ongoing, yet a clear answer remains elusive."
        ],
        "medium": [
            f"Anyone who believes {topic} is a balanced issue is clearly misinformed.",
            f"It's time to recognize that {topic} is not as simple as people think."
        ],
        "extreme": [
            f"The idea that {topic} is debatable is absurd.",
            f"History has proven that {topic} only has one correct answer."
        ]
    }

    arguments = {
        "mild": [
            f"While teamwork has its advantages, self-reliance fosters true independence and personal growth.",
            f"Technology in education should be regulated to avoid potential negative consequences."
        ],
        "medium": [
            f"Self-reliance is the foundation of success, while teamwork often dilutes individual brilliance.",
            f"The rapid adoption of AI is inevitable, and resisting it is futile."
        ],
        "extreme": [
            f"Teamwork is nothing more than a safety net for those who cannot think for themselves.",
            f"AI will replace all human jobs, and those who deny it are ignorant of technological progress."
        ]
    }

    conclusions = {
        "mild": [
            f"This issue requires careful consideration and an open mind.",
            f"The discussion is far from over, but we must approach it with rationality."
        ],
        "medium": [
            f"We must acknowledge the facts and move toward a more informed future.",
            f"The evidence speaks for itself, and those who refuse to accept it are naive."
        ],
        "extreme": [
            f"Those who disagree are simply blind to reality.",
            f"The truth is undeniable, and history will prove it right."
        ]
    }

    essay = (
        f"{random.choice(openings[intensity])} "
        f"{random.choice(arguments[intensity])} "
        f"{random.choice(conclusions[intensity])}"
    )

    # Ensure the essay contains at least 100 words
    while count_words(essay) < 100:
        essay += " " + random.choice(arguments[intensity])

    return essay

def create_submission():
    """Generates the submission.csv file with essays of approximately 100 words."""
    
    # Predefined IDs and topics
    ids = [1, 2, 3]
    topics = [
        "Self-reliance vs Teamwork",
        "The role of technology in education",
        "Should AI replace human jobs?"
    ]
    
    # Assign intensity levels for each essay
    intensities = ["medium", "medium", "extreme"]  # The third essay is more extreme

    submissions = []
    for id_, topic, intensity in zip(ids, topics, intensities):
        essay = generate_controversial_essay(topic, intensity)
        word_count = count_words(essay)
        print(f"Essay for ID {id_}: {word_count} words")  # Display word count

        submissions.append({"id": id_, "essay": essay})

    # Save the essays to submission.csv
    submission_df = pd.DataFrame(submissions)
    submission_df.to_csv("submission.csv", index=False)
    
    print("Submission file saved as submission.csv")

if __name__ == "__main__":
    create_submission()


