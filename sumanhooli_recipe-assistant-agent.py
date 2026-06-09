from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import time
import uuid

# ============================
# 1. Data Models
# ============================

@dataclass
class Ingredient:
    name: str
    quantity: str  # e.g. "2 cups", "1 tsp"


@dataclass
class Recipe:
    id: str
    name: str
    difficulty: str         # "easy", "medium", "hard"
    duration_minutes: int
    ingredients: List[Ingredient]
    steps: List[str]
    tags: List[str]         # e.g. ["vegan", "quick", "dinner"]


# ============================
# 2. Simple Recipe "Database"
# ============================

def build_recipe_db() -> Dict[str, Recipe]:
    """Create a small in-memory recipe database."""
    db: Dict[str, Recipe] = {}

    r1 = Recipe(
        id=str(uuid.uuid4()),
        name="Veggie Fried Rice",
        difficulty="easy",
        duration_minutes=20,
        ingredients=[
            Ingredient("Cooked rice", "2 cups"),
            Ingredient("Mixed vegetables", "1 cup"),
            Ingredient("Soy sauce", "2 tbsp"),
            Ingredient("Oil", "1 tbsp"),
            Ingredient("Garlic", "2 cloves"),
        ],
        steps=[
            "Heat oil in a pan.",
            "Add chopped garlic and sautÃ© until fragrant.",
            "Add mixed vegetables and stir fry for 3â€“4 minutes.",
            "Add cooked rice and soy sauce.",
            "Mix everything well and cook for 2â€“3 more minutes.",
        ],
        tags=["quick", "dinner", "vegetarian"]
    )

    r2 = Recipe(
        id=str(uuid.uuid4()),
        name="Simple Banana Smoothie",
        difficulty="easy",
        duration_minutes=5,
        ingredients=[
            Ingredient("Banana", "1 large"),
            Ingredient("Milk or plant milk", "1 cup"),
            Ingredient("Honey or sugar", "1 tbsp"),
            Ingredient("Ice cubes", "3â€“4"),
        ],
        steps=[
            "Peel and slice the banana.",
            "Add banana, milk, honey, and ice cubes to a blender.",
            "Blend until smooth and creamy.",
            "Pour into a glass and serve immediately.",
        ],
        tags=["breakfast", "drink", "quick"]
    )

    r3 = Recipe(
        id=str(uuid.uuid4()),
        name="Chickpea Salad Bowl",
        difficulty="medium",
        duration_minutes=15,
        ingredients=[
            Ingredient("Boiled chickpeas", "1 cup"),
            Ingredient("Cucumber", "1/2, chopped"),
            Ingredient("Tomato", "1, chopped"),
            Ingredient("Onion", "1/4, finely chopped"),
            Ingredient("Lemon juice", "2 tbsp"),
            Ingredient("Olive oil", "1 tbsp"),
            Ingredient("Salt & pepper", "to taste"),
        ],
        steps=[
            "Add chickpeas, cucumber, tomato, and onion to a large bowl.",
            "In a small bowl, whisk together lemon juice, olive oil, salt, and pepper.",
            "Pour the dressing over the salad.",
            "Mix gently until everything is well coated.",
            "Serve immediately or chill for 10 minutes.",
        ],
        tags=["lunch", "healthy", "vegan"]
    )

    for r in [r1, r2, r3]:
        db[r.id] = r

    return db


RECIPE_DB: Dict[str, Recipe] = build_recipe_db()


# ============================
# 3. In-Memory Session Service
# ============================

class InMemorySessionService:
    """Simple session store keyed by session_id."""

    def __init__(self):
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def create_session(self, user_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        sid = str(uuid.uuid4())
        self._sessions[sid] = {
            "user_id": user_id,
            "created_at": time.time(),
            "metadata": metadata or {},
            "state": {},           # you can store arbitrary state here
        }
        return sid

    def get(self, session_id: str) -> Dict[str, Any]:
        return self._sessions.get(session_id, {})

    def set_state(self, session_id: str, key: str, value: Any):
        if session_id not in self._sessions:
            raise KeyError("session not found")
        self._sessions[session_id]["state"][key] = value

    def get_state(self, session_id: str, key: str, default=None):
        return self._sessions.get(session_id, {}).get("state", {}).get(key, default)


# ============================
# 4. Mock Recipe Agent (fake LLM)
# ============================

class RecipeAgent:
    """
    A mock 'AI agent' that works like an LLM-based assistant,
    but uses simple Python logic so it runs anywhere.
    """

    def __init__(self, recipes: Dict[str, Recipe]):
        self.recipes = recipes

    # ---- Helper methods ----

    def _filter_by_ingredients(self, available: List[str]) -> List[Recipe]:
        available_lower = {a.strip().lower() for a in available}
        result = []
        for recipe in self.recipes.values():
            needed = {ing.name.lower() for ing in recipe.ingredients}
            if needed.issubset(available_lower) or needed.intersection(available_lower):
                result.append(recipe)
        return result

    def _filter_by_time(self, recipes: List[Recipe], max_time: int) -> List[Recipe]:
        return [r for r in recipes if r.duration_minutes <= max_time]

    def _filter_by_tags(self, recipes: List[Recipe], preferred_tags: List[str]) -> List[Recipe]:
        if not preferred_tags:
            return recipes
        tags_lower = {t.lower() for t in preferred_tags}
        return [r for r in recipes if tags_lower.intersection({t.lower() for t in r.tags})]

    # ---- Public agent-like methods ----

    def suggest_recipes(
        self,
        ingredients: List[str],
        max_time: int = 30,
        preferred_tags: Optional[List[str]] = None,
    ) -> List[Recipe]:
        """
        Given ingredients + time + optional tags, returns a list of recipes.
        """
        preferred_tags = preferred_tags or []
        candidates = self._filter_by_ingredients(ingredients)
        candidates = self._filter_by_time(candidates, max_time)
        candidates = self._filter_by_tags(candidates, preferred_tags)
        return candidates

    def explain_recipe(self, recipe_id: str) -> str:
        """
        Returns a friendly explanation of the recipe.
        """
        recipe = self.recipes.get(recipe_id)
        if not recipe:
            return "Sorry, I couldn't find that recipe."

        lines = []
        lines.append(f"ğŸ‘©â€�ğŸ�³ *{recipe.name}*")
        lines.append(f"- Difficulty: {recipe.difficulty}")
        lines.append(f"- Time: {recipe.duration_minutes} minutes")
        lines.append("\nIngredients:")
        for ing in recipe.ingredients:
            lines.append(f"  â€¢ {ing.quantity} {ing.name}")
        lines.append("\nSteps:")
        for idx, step in enumerate(recipe.steps, start=1):
            lines.append(f"  {idx}. {step}")
        return "\n".join(lines)

    def generate_shopping_list(self, recipe_ids: List[str]) -> List[Ingredient]:
        """
        Merge ingredients for multiple recipes into a single shopping list.
        (Simple version: just concatenates all ingredients.)
        """
        items: List[Ingredient] = []
        for rid in recipe_ids:
            recipe = self.recipes.get(rid)
            if recipe:
                items.extend(recipe.ingredients)
        return items

    def answer_question(self, question: str) -> str:
        """
        Very simple FAQ-style answerer, just to mimic an AI.
        """
        q = question.lower()
        if "healthy" in q or "diet" in q:
            return "In general, recipes with more vegetables, beans, and less frying are healthier. Our Chickpea Salad Bowl is a good option."
        if "quick" in q or "fast" in q:
            return "For something quick, try the Simple Banana Smoothie or Veggie Fried Rice."
        if "vegan" in q:
            return "The Chickpea Salad Bowl is fully vegan in this mini recipe set."
        return "I'm a simple recipe agent. I can help suggest recipes, explain them, or build a shopping list from the recipes I know."


# ============================
# 5. Orchestrator (like your smart AI agent)
# ============================

class CookingCompanion:
    """
    High-level orchestrator that combines:
    - session state
    - the RecipeAgent "intelligence"
    """

    def __init__(self, session_service: InMemorySessionService, recipe_agent: RecipeAgent):
        self.sessions = session_service
        self.agent = recipe_agent

    def start_session(self, user_id: str) -> str:
        return self.sessions.create_session(user_id)

    def plan_meal(
        self,
        session_id: str,
        ingredients: List[str],
        max_time: int,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Main entry: suggest recipes based on user request,
        and store the selected recipes in session state.
        """
        recipes = self.agent.suggest_recipes(ingredients, max_time, tags)
        self.sessions.set_state(session_id, "last_suggestions", [r.id for r in recipes])

        result = {
            "count": len(recipes),
            "recipes": [
                {
                    "id": r.id,
                    "name": r.name,
                    "time": r.duration_minutes,
                    "tags": r.tags,
                    "difficulty": r.difficulty,
                }
                for r in recipes
            ],
        }
        return result

    def get_recipe_details(self, session_id: str, recipe_id: str) -> str:
        """
        Return a detailed explanation and also remember the last viewed recipe.
        """
        self.sessions.set_state(session_id, "last_recipe", recipe_id)
        return self.agent.explain_recipe(recipe_id)

    def build_shopping_list_from_last(self, session_id: str) -> List[str]:
        """
        Generate a shopping list from the last suggestions in this session.
        """
        last_ids = self.sessions.get_state(session_id, "last_suggestions", [])
        ingredients = self.agent.generate_shopping_list(last_ids)
        # Convert to friendly strings
        return [f"{ing.quantity} {ing.name}" for ing in ingredients]

    def ask(self, session_id: str, question: str) -> str:
        """
        Generic Q&A about recipes / cooking choices.
        """
        # (We don't use session state here, but you could log questions)
        return self.agent.answer_question(question)


# ============================
# 6. Demo / Example Usage
# ============================

if __name__ == "__main__":
    # Create core objects
    session_service = InMemorySessionService()
    recipe_agent = RecipeAgent(RECIPE_DB)
    app = CookingCompanion(session_service, recipe_agent)

    # Simulate a user session
    user_id = "user123"
    session_id = app.start_session(user_id)
    print(f"ğŸ”� New session created: {session_id}\n")

    # 1) Plan a meal with some ingredients
    available_ingredients = ["Cooked rice", "Mixed vegetables", "Soy sauce", "Garlic"]
    plan = app.plan_meal(
        session_id=session_id,
        ingredients=available_ingredients,
        max_time=25,
        tags=["dinner"]
    )

    print("ğŸ�½ Suggested recipes based on your ingredients:\n")
    for r in plan["recipes"]:
        print(f"- {r['name']} ({r['time']} min, difficulty: {r['difficulty']}, tags: {r['tags']})")

    # 2) Show details of the first recipe
    if plan["recipes"]:
        first_id = plan["recipes"][0]["id"]
        print("\nğŸ“– Recipe details:\n")
        print(app.get_recipe_details(session_id, first_id))

    # 3) Build a shopping list from all suggested recipes
    print("\nğŸ›’ Combined shopping list from suggestions:\n")
    for item in app.build_shopping_list_from_last(session_id):
        print(f"- {item}")

    # 4) Ask a generic question
    print("\nâ�“ Q&A example:")
    answer = app.ask(session_id, "Which recipe is healthy and vegan?")
    print("Agent:", answer)


