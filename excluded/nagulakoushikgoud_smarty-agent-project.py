from datetime import datetime

# 1. Initialize Services
session_service = InMemorySessionService()

# 2. Create Resources and Flashcards
resource_ml = Resource(
    title="Introduction to Machine Learning",
    url="http://example.com/ml-intro",
    description="Foundational concepts of ML.",
    tags=["ML", "Python", "Beginner"]
)

flashcard_1 = Flashcard(
    resource_id=resource_ml.id,
    question="What is a 'feature' in machine learning?",
    answer="An individual measurable property or characteristic of a phenomenon being observed."
)

flashcard_2 = Flashcard(
    resource_id=resource_ml.id,
    question="Name the two main types of supervised learning.",
    answer="Classification and Regression."
)

print("\n--- Resources and Flashcards Created ---")
print(f"Resource: {resource_ml.title}")
print(f"Flashcard 1 Q: {flashcard_1.question[:40]}...")


# 3. Simulate a User Session
USER_ID = "kag_user_123"
session_service.start_session(USER_ID)

# Store the last viewed resource in the session
session_service.update_session_state(
    user_id=USER_ID, 
    key="last_viewed_resource_id", 
    value=resource_ml.id
)

# Simulate a review and update flashcard state
flashcard_1.correct_streak = 1
flashcard_1.last_reviewed = datetime.now().isoformat()

session_service.update_session_state(
    user_id=USER_ID,
    key="flashcard_1_streak",
    value=flashcard_1.correct_streak
)

# 4. Retrieve and Print Session Data
print("\n--- Current Session State ---")
current_state = session_service.get_session_state(USER_ID)
print(f"Current State: {current_state}")
print(f"Flashcard 1 Streak: {flashcard_1.correct_streak}")

# 5. End Session
print()
session_service.end_session(USER_ID)

