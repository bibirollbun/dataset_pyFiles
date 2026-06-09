import json
import re



def data_collector_agent(user_text: str):
    """
    Rich extraction of structured info from the user's description.
    Handles followers, story views, likes, timing, sudden change words,
    busy periods (exams, projects), and special events like blocks/unfollows.
    """
    text = user_text.lower()

    data = {
        "follower_before": None,
        "follower_after": None,
        "follower_change": None,
        "story_views_before": None,
        "story_views_after": None,
        "post_likes_before": None,
        "post_likes_after": None,
        "posting_time": None,           # morning / afternoon / evening / night
        "posting_frequency": None,      # daily / weekly / rarely / spammy
        "sudden_change": False,
        "busy_period": False,
        "specific_people": [],
        "major_events": [],             # ["unfollow", "block", "close_friends_removed", ...]
        "raw_text": user_text
    }

    # Followers: e.g. "had 320 followers, now 305"
    follower_match = re.search(r'(\d+)\s+followers?[^0-9]+(\d+)', text)
    if follower_match:
        before = int(follower_match.group(1))
        after = int(follower_match.group(2))
        data["follower_before"] = before
        data["follower_after"] = after
        data["follower_change"] = after - before

    # Story views: e.g. "views dropped from 180 to 95"
    story_match = re.search(r'story.*?(\d+)[^\d]+(\d+)', text)
    if story_match:
        data["story_views_before"] = int(story_match.group(1))
        data["story_views_after"] = int(story_match.group(2))

    # Post likes: e.g. "likes went from 120 to 60"
    like_match = re.search(r'likes?.*?(\d+)[^\d]+(\d+)', text)
    if like_match:
        data["post_likes_before"] = int(like_match.group(1))
        data["post_likes_after"] = int(like_match.group(2))

    # Posting time
    if "late night" in text or "night" in text:
        data["posting_time"] = "night"
    elif "afternoon" in text:
        data["posting_time"] = "afternoon"
    elif "evening" in text:
        data["posting_time"] = "evening"
    elif "morning" in text:
        data["posting_time"] = "morning"

    # Posting frequency
    if "daily" in text or "every day" in text:
        data["posting_frequency"] = "daily"
    elif "once a week" in text or "weekly" in text:
        data["posting_frequency"] = "weekly"
    elif "rarely" in text or "hardly post" in text:
        data["posting_frequency"] = "rarely"
    elif "too much" in text or "spam" in text or "many posts" in text:
        data["posting_frequency"] = "spammy"

    # Sudden change words
    sudden_keywords = ["suddenly", "all of a sudden", "overnight", "out of nowhere"]
    if any(k in text for k in sudden_keywords):
        data["sudden_change"] = True

    # Busy period keywords (exams, projects, deadlines etc.)
    busy_keywords = ["exam", "exams", "internals", "project", "deadlines", "semester", "festival", "vacation"]
    if any(k in text for k in busy_keywords):
        data["busy_period"] = True

    # Major events
    if "unfollowed me" in text or "unfollow" in text:
        data["major_events"].append("unfollow")
    if "blocked me" in text or "blocked" in text:
        data["major_events"].append("block")
    if "removed me from close friends" in text or "close friends list" in text:
        data["major_events"].append("close_friends_removed")
    if "muted" in text:
        data["major_events"].append("possibly_muted")

    # Specific people
    if "friend" in text:
        data["specific_people"].append("friend")
    if "bestie" in text:
        data["specific_people"].append("bestie")
    if "crush" in text:
        data["specific_people"].append("crush")

    return data



def pattern_analyzer_agent(structured_data: dict):
    """
    Detects richer engagement + behavior patterns.
    """
    patterns = []

    before = structured_data.get("follower_before")
    after = structured_data.get("follower_after")
    change = structured_data.get("follower_change")

    if before is not None and after is not None:
        if change <= -20:
            patterns.append("Large follower drop in short time.")
        elif change < 0:
            patterns.append("Gradual follower decrease.")
        elif change > 0:
            patterns.append("Follower count increasing.")

    sv_before = structured_data.get("story_views_before")
    sv_after = structured_data.get("story_views_after")
    if sv_before and sv_after:
        ratio = sv_after / sv_before if sv_before > 0 else 1
        if ratio <= 0.5:
            patterns.append("Story views dropped heavily.")
        elif ratio < 1:
            patterns.append("Story views slightly decreased.")
        elif ratio > 1.2:
            patterns.append("Story views increased noticeably.")

    likes_before = structured_data.get("post_likes_before")
    likes_after = structured_data.get("post_likes_after")
    if likes_before and likes_after:
        like_ratio = likes_after / likes_before if likes_before > 0 else 1
        if like_ratio <= 0.5:
            patterns.append("Post likes dropped significantly.")
        elif like_ratio < 1:
            patterns.append("Post likes slightly decreased.")
        elif like_ratio > 1.2:
            patterns.append("Post likes increased.")

    # Posting time + frequency patterns
    if structured_data.get("posting_time") == "night":
        patterns.append("Posting mostly at night, which can limit reach.")
    if structured_data.get("posting_frequency") == "rarely":
        patterns.append("User posts rarely; algorithm may treat account as less active.")
    if structured_data.get("posting_frequency") == "spammy":
        patterns.append("Very high posting frequency may cause follower fatigue.")

    # Sudden change pattern
    if structured_data.get("sudden_change"):
        patterns.append("User feels the change was sudden.")

    # Busy periods
    if structured_data.get("busy_period"):
        patterns.append("User or audience might be in a busy period (exams, projects, etc.).")

    # Specific people + events
    if structured_data.get("specific_people"):
        patterns.append("There is a concern about a specific person's engagement.")
    if structured_data.get("major_events"):
        patterns.append("A strong event was mentioned (unfollow, block, mute, etc.).")

    return patterns




def behavior_insight_agent(patterns):
    """
    Converts patterns into deeper, human-style interpretations.
    """
    insights = []

    for p in patterns:
        lower = p.lower()

        if "large follower drop" in lower:
            insights.append(
                "Large drops usually come from algorithm cleanups, inactive accounts leaving, or strong changes in content style."
            )
        if "gradual follower decrease" in lower:
            insights.append(
                "Gradual follower decrease is common and rarely personal; people cycle through interests over time."
            )
        if "follower count increasing" in lower:
            insights.append(
                "Follower growth indicates that your overall content direction is working."
            )
        if "story views dropped heavily" in lower:
            insights.append(
                "Heavy story-view drops can happen if fewer people are active, your posting time changed, or some followers muted stories."
            )
        if "story views slightly decreased" in lower:
            insights.append(
                "Small drops in story views are normal and usually not a sign of any problem."
            )
        if "post likes dropped significantly" in lower:
            insights.append(
                "A big drop in likes can indicate algorithm changes, less reach, or content that feels less engaging to your usual audience."
            )
        if "post likes slightly decreased" in lower:
            insights.append(
                "Minor dips in likes are normal and often random."
            )
        if "posting mostly at night" in lower:
            insights.append(
                "Posting mostly at night reduces the chance of your posts being seen when most followers are active."
            )
        if "user posts rarely" in lower:
            insights.append(
                "Rare posting can cause the algorithm to show your content less often; people also forget to engage with less active accounts."
            )
        if "follower fatigue" in lower:
            insights.append(
                "Posting too often may overwhelm some followers, causing them to skip or unfollow."
            )
        if "sudden" in lower:
            insights.append(
                "When a change feels sudden, it often combines several factors: posting gap, algorithm reset, busy periods, and shifting attention."
            )
        if "busy period" in lower:
            insights.append(
                "Busy times like exams or project season reduce everyone's online activity; lower engagement during these times is expected."
            )
        if "specific person's engagement" in lower:
            insights.append(
                "Individual engagement is very unpredictable. One person's reduced activity does not necessarily mean a negative feeling towards you."
            )
        if "strong event" in lower:
            insights.append(
                "Events like unfollows or being removed from close friends can feel personal, but they may also reflect the other person's boundaries, mood, or cleanup of their list."
            )

    if not insights:
        insights.append("No strong warning signals detected; overall behavior appears within normal social-media variation.")

    return insights



def strategy_agent(insights):
    """
    Generates practical strategy + emotional well-being suggestions
    based on the type of insights produced.
    """
    advice = []

    # Base suggestions
    advice.append("Review your posting times and try sharing when your audience is usually active (often evening or early night).")
    advice.append("Experiment with different formats: reels, carousels, polls, and question stickers.")
    advice.append("Check your metrics weekly instead of multiple times per day to avoid stress.")

    joined_insights = " ".join(insights).lower()

    if "algorithm" in joined_insights or "reach" in joined_insights:
        advice.append("Use interactive stories (polls, sliders, Q&A) for a few days to rebuild algorithm visibility.")
    if "rare posting" in joined_insights or "less often" in joined_insights:
        advice.append("Try a simple, consistent schedule like 2–3 posts per week and 2–4 stories on active days.")
    if "follower fatigue" in joined_insights or "overwhelm" in joined_insights:
        advice.append("If you feel you post too often, slow down slightly and focus on quality rather than quantity.")
    if "specific person" in joined_insights or "individual engagement" in joined_insights:
        advice.append("Avoid judging your relationship with someone purely by likes or views; consider offline or direct communication if needed.")
    if "busy times like exams" in joined_insights:
        advice.append("During exam or project periods, expect less engagement in general. Focus on your priorities first.")
    if "no strong warning signals" in joined_insights:
        advice.append("Your patterns look mostly normal; you can relax and treat small changes as part of regular social media fluctuation.")

    # Emotional hygiene
    advice.append("Remind yourself that social media interactions are not a full reflection of your worth or real relationships.")
    advice.append("If you notice that checking stats makes you anxious, take short breaks or set specific times to check them.")

    # Remove duplicates while keeping order
    seen = set()
    unique_advice = []
    for a in advice:
        if a not in seen:
            unique_advice.append(a)
            seen.add(a)

    return unique_advice



def run_pipeline(user_input: str):
    data = data_collector_agent(user_input)
    patterns = pattern_analyzer_agent(data)
    insights = behavior_insight_agent(patterns)
    advice = strategy_agent(insights)

    return {
        "Structured Data": data,
        "Detected Patterns": patterns,
        "Behavior Insights": insights,
        "Advice": advice
    }


def print_output(result):
    sd = result["Structured Data"]

    print("=== Structured Data ===")
    print(f"Follower Before: {sd.get('follower_before')}")
    print(f"Follower After : {sd.get('follower_after')}")
    print(f"Follower Change: {sd.get('follower_change')}")
    print(f"Story Views    : {sd.get('story_views_before')} → {sd.get('story_views_after')}")
    print(f"Post Likes     : {sd.get('post_likes_before')} → {sd.get('post_likes_after')}")
    print(f"Posting Time   : {sd.get('posting_time')}")
    print(f"Posting Freq   : {sd.get('posting_frequency')}")
    print(f"Sudden Change? : {sd.get('sudden_change')}")
    print(f"Busy Period?   : {sd.get('busy_period')}")
    print(f"Specific People: {sd.get('specific_people')}")
    print(f"Events Flags   : {sd.get('major_events')}")
    print()

    print("=== Detected Patterns ===")
    for p in result["Detected Patterns"]:
        print(f"• {p}")
    if not result["Detected Patterns"]:
        print("• No strong patterns detected.")
    print()

    print("=== Behavior Insights ===")
    for i in result["Behavior Insights"]:
        print(f"• {i}")
    if not result["Behavior Insights"]:
        print("• Overall behavior looks normal.")
    print()

    print("=== Advice ===")
    for a in result["Advice"]:
        print(f"• {a}")
    if not result["Advice"]:
        print("• No specific advice needed; keep doing what works.")



input_text = """Last month I had 320 followers, now I have 305.
My story views dropped from 180 to 95.
A friend who used to like every post suddenly stopped engaging.
I mostly post late at night and I was busy with exams."""
result = run_pipeline(input_text)
print_output(result)



input_text2 = """This month my followers increased from 400 to 450.
My story views went from 120 to 180 and likes improved as well.
I post almost daily in the evening and try different formats."""
print_output(run_pipeline(input_text2))


output_text = "SociaLens Kaggle Submission Completed Successfully."

with open("/kaggle/working/socailens_output.txt", "w") as f:
    f.write(output_text)

print("Output file created at /kaggle/working/socailens_output.txt")


