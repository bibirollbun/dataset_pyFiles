# smart_peer_movie_recommender.py
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import random
import datetime

# -----------------------------
# Data classes (simple structs)
# -----------------------------
@dataclass
class MovieOption:
    title: str
    genres: List[str]
    duration_min: int
    rating: float
    description: str


@dataclass
class PeerProfile:
    peer_id: str
    name: str
    preferred_genres: List[str]
    available_weekdays: List[str]  # e.g., ['Fri', 'Sat']
    bio: Optional[str] = None


# -----------------------------
# Tools (mocked for now)
# -----------------------------
def mock_movie_search(preferred_genres: List[str], mood: str, max_results: int = 6) -> List[MovieOption]:
    """
    Fake movie search. In practice this would query an API or a database.
    We produce a small catalog and filter by genre/mood heuristics.
    """
    catalog = [
        MovieOption("Eternal Autumn", ["drama", "romance"], 110, 8.1, "A reflective love story in the fall."),
        MovieOption("Quantum Run", ["sci-fi", "thriller"], 125, 7.8, "A high-speed chase across parallel worlds."),
        MovieOption("Laugh Riot", ["comedy"], 95, 7.0, "A slapstick comedy about a chaotic family reunion."),
        MovieOption("Hidden Trails", ["adventure", "family"], 105, 7.4, "A family hikes and learns about nature."),
        MovieOption("Mind Palace", ["mystery", "psychological"], 130, 8.3, "A detective explores memory and identity."),
        MovieOption("Space Bakers", ["animation", "family", "comedy"], 88, 6.9, "Cheerful animated cooking on Mars."),
        MovieOption("Neon Nights", ["crime", "drama"], 118, 7.6, "Stylish city noir with a heart."),
        MovieOption("Serene Rivers", ["documentary"], 72, 8.7, "A calming journey through great rivers."),
    ]

    # Simple scoring: +1 if genres overlap with preferred_genres, +1 if mood matches some keyword
    mood_tags = {
        "chill": {"drama", "documentary", "romance", "family"},
        "thrill": {"thriller", "sci-fi", "mystery", "crime"},
        "fun": {"comedy", "animation"},
        "adventure": {"adventure", "family", "sci-fi"},
    }
    desired = set(g.lower() for g in preferred_genres)
    mood_set = mood_tags.get(mood.lower(), set())

    scored = []
    for m in catalog:
        score = 0
        m_genres = set(g.lower() for g in m.genres)
        score += len(desired & m_genres)
        if m_genres & mood_set:
            score += 1
        # small random tie-breaker
        score += random.uniform(0, 0.5)
        scored.append((score, m))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored[:max_results]]


def similarity_score(profile_a: Dict[str, Any], profile_b: Dict[str, Any]) -> float:
    """Simple Jaccard-like score on preferred genres."""
    a = set(g.lower() for g in profile_a.get("preferred_genres", []))
    b = set(g.lower() for g in profile_b.get("preferred_genres", []))
    if not a and not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b) if len(a | b) > 0 else 1
    return intersection / union


# -----------------------------
# Memory (very simple)
# -----------------------------
class SimpleMemory:
    """Tiny in-memory store for user preferences and past picks."""
    def __init__(self):
        self.store: Dict[str, Any] = {}

    def set(self, key: str, value: Any):
        self.store[key] = value

    def get(self, key: str, default=None):
        return self.store.get(key, default)


# -----------------------------
# Agents
# -----------------------------
class MovieSearchAgent:
    def search(self, genres: List[str], mood: str, max_results: int = 6) -> List[MovieOption]:
        return mock_movie_search(genres, mood, max_results)


class PeerPoolAgent:
    def __init__(self, peers: List[PeerProfile]):
        self.peers = peers

    def find_available_peers(self, preferred_weekday: str) -> List[PeerProfile]:
        return [p for p in self.peers if preferred_weekday in p.available_weekdays]


class PeerMatchAgent:
    def __init__(self, peer_pool_agent: PeerPoolAgent):
        self.peer_pool = peer_pool_agent

    def rank_peers_for_user(self, user_profile: Dict[str, Any], candidates: List[PeerProfile]) -> List[Tuple[PeerProfile, float]]:
        scored = []
        for c in candidates:
            score = similarity_score(user_profile, {'preferred_genres': c.preferred_genres})
            # boost if bios mention common interests (demo)
            if c.bio and any(tok.lower() in user_profile.get("bio", "").lower() for tok in ["comedy", "sci-fi", "drama", "thriller"]):
                score += 0.05
            scored.append((c, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


class RecommendationAgent:
    """
    Chooses a movie + peer(s) suggestion aiming to maximize genre overlap
    and availability while respecting simple constraints (duration tolerance).
    """
    def __init__(self, movie_search_agent: MovieSearchAgent, peer_match_agent: PeerMatchAgent, memory: SimpleMemory):
        self.movie_search = movie_search_agent
        self.peer_match = peer_match_agent
        self.memory = memory

    def recommend(self, user_profile: Dict[str, Any], mood: str, preferred_weekday: str, max_duration_min: int = 140):
        # 1. Search movies
        movies = self.movie_search.search(user_profile.get("preferred_genres", []), mood)

        # 2. Filter by duration preference
        movies = [m for m in movies if m.duration_min <= max_duration_min]
        if not movies:
            # fallback: relax duration
            movies = self.movie_search.search(user_profile.get("preferred_genres", []), mood, max_results=8)

        # 3. Find available peers and rank
        peer_pool = self.peer_match.peer_pool.peers
        candidates = self.peer_match.peer_pool.find_available_peers(preferred_weekday)
        ranked_peers = self.peer_match.rank_peers_for_user(user_profile, candidates)

        # 4. Heuristic choose best movie-peer combo:
        # Score = movie_genre_overlap + peer_similarity + normalized rating
        best_combo = None
        best_score = -1.0
        for movie in movies:
            movie_genres = set(g.lower() for g in movie.genres)
            overlap = len(set(g.lower() for g in user_profile.get("preferred_genres", [])) & movie_genres)
            for peer, peer_score in ranked_peers[:5]:  # top 5 peers
                # small bonus if peer shares specific genre
                combo_score = overlap + peer_score + (movie.rating / 10.0)
                # slight preference for shorter movies if user has limited time
                if user_profile.get("time_limit_min") and movie.duration_min > user_profile["time_limit_min"]:
                    combo_score -= 0.5
                if combo_score > best_score:
                    best_score = combo_score
                    best_combo = (movie, peer, combo_score)

        # 5. Construct watch-plan suggestion
        if best_combo is None:
            return {"message": "No suitable movie-peer combo found. Try changing preferences or day."}

        movie, peer, score = best_combo
        plan = {
            "movie": movie,
            "peer": peer,
            "scheduled_day": preferred_weekday,
            "scheduled_time": "20:00",
            "estimated_duration_min": movie.duration_min,
            "explanation": f"Selected because of genre overlap and peer match (score={score:.2f}).",
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }

        # Save last recommendation in memory
        self.memory.set("last_recommendation", {
            "user_profile": user_profile,
            "plan": {
                "movie_title": movie.title,
                "peer_id": peer.peer_id,
                "day": preferred_weekday,
                "time": "20:00",
            },
            "score": score,
            "ts": plan["timestamp"]
        })

        return plan


# -----------------------------
# Demo / Example usage
# -----------------------------
def demo():
    # Setup demo peers
    demo_peers = [
        PeerProfile("p1", "Aisha", ["drama", "romance", "documentary"], ["Fri", "Sat"], bio="Loves cozy dramas and documentaries."),
        PeerProfile("p2", "Ravi", ["sci-fi", "mystery", "thriller"], ["Thu", "Sat"], bio="Sci-fi nerd, enjoys twists."),
        PeerProfile("p3", "Sana", ["comedy", "animation", "family"], ["Sat", "Sun"], bio="Comedy fan and animation enthusiast."),
        PeerProfile("p4", "Karan", ["crime", "drama", "thriller"], ["Fri", "Sun"], bio="Enjoys noir and city stories."),
    ]

    # Example user profile
    user_profile = {
        "user_id": "u123",
        "name": "You",
        "preferred_genres": ["sci-fi", "mystery"],
        "bio": "I love sci-fi and clever mysteries. Prefer evening watches.",
        "time_limit_min": 140,  # maximum tolerable duration
    }

    # Parameters from "UI"
    mood = "thrill"             # user mood: chill / thrill / fun / adventure
    preferred_weekday = "Sat"   # when user wants to watch

    # Agents and memory
    memory = SimpleMemory()
    movie_agent = MovieSearchAgent()
    peer_pool_agent = PeerPoolAgent(demo_peers)
    peer_match_agent = PeerMatchAgent(peer_pool_agent)
    recommender = RecommendationAgent(movie_agent, peer_match_agent, memory)

    # Run recommendation
    plan = recommender.recommend(user_profile, mood, preferred_weekday)

    # Print plan
    print("=== Peer Movie Recommendation ===")
    if "message" in plan:
        print(plan["message"])
        return

    m: MovieOption = plan["movie"]
    p: PeerProfile = plan["peer"]
    print(f"Movie: {m.title} ({', '.join(m.genres)}) - {m.duration_min} min - rating {m.rating}")
    print(f"Description: {m.description}")
    print(f"Suggested peer: {p.name} (id={p.peer_id}) - prefers {', '.join(p.preferred_genres)} - available on {', '.join(p.available_weekdays)}")
    print(f"Scheduled: {plan['scheduled_day']} at {plan['scheduled_time']}")
    print(f"Reason: {plan['explanation']}")
    print(f"Saved recommendation snapshot (memory): {memory.get('last_recommendation')}")


if __name__ == "__main__":
    demo()


