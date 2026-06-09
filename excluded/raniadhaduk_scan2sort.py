# ============================================
# 0. Setup & Dependencies
# ============================================

!pip install -q pillow
print("Pillow installed")

import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

import google.generativeai as genai
from kaggle_secrets import UserSecretsClient
from PIL import Image

# --- Configure Gemini using Kaggle Secrets ---
GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)

MODEL_NAME = "gemini-2.5-flash-lite"
model = genai.GenerativeModel(MODEL_NAME)

print("Gemini configured with model:", MODEL_NAME)


# ============================================
# 1. Sessions & Memory (per-user, with persistence)
# ============================================

class SessionStore:
    """
    Simple in-memory session + memory store with disk persistence.
    - Per-user streak
    - Total scans
    - Total points
    - Mistakes by item
    """

    def __init__(self, path: str = "session_store.json"):
        self.path = path
        self.sessions: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "total_scans": 0,
                "streak": 0,
                "total_points": 0,
                "mistakes": defaultdict(int),
            }
        )
        self.load()

    def get_user_state(self, user_id: str) -> Dict[str, Any]:
        return self.sessions[user_id]

    def update_after_scan(
        self,
        user_id: str,
        points: int,
        all_confident: bool,
        mistaken_items: List[str],
    ):
        state = self.sessions[user_id]
        state["total_scans"] += 1
        state["total_points"] += points

        if all_confident:
            state["streak"] += 1
        else:
            state["streak"] = 0

        for item in mistaken_items:
            state["mistakes"][item] += 1

        # persist after each update
        self.save()
        return state

    # ---------- Persistence helpers ----------

    def _to_serializable(self) -> Dict[str, Any]:
        """Convert defaultdicts to normal dicts for JSON."""
        data = {}
        for user_id, state in self.sessions.items():
            data[user_id] = {
                "total_scans": state.get("total_scans", 0),
                "streak": state.get("streak", 0),
                "total_points": state.get("total_points", 0),
                "mistakes": dict(state.get("mistakes", {})),  # defaultdict -> dict
            }
        return data

    def save(self):
        try:
            data = self._to_serializable()
            with open(self.path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            print("[WARN] Failed to save SessionStore:", e)

    def load(self):
        try:
            if not os.path.exists(self.path):
                return
            with open(self.path, "r") as f:
                data = json.load(f)

            for user_id, state in data.items():
                self.sessions[user_id] = {
                    "total_scans": state.get("total_scans", 0),
                    "streak": state.get("streak", 0),
                    "total_points": state.get("total_points", 0),
                    "mistakes": defaultdict(int, state.get("mistakes", {})),
                }
        except Exception as e:
            print("[WARN] Failed to load SessionStore:", e)


SESSION_STORE = SessionStore()


# ============================================
# 2. Observability: Logging & TWO-LEVEL Metrics
# ============================================

class Metrics:
    """
    Two-level metrics:
    - session_counters: reset when kernel restarts
    - global_counters: persisted in metrics_store.json
    """

    def __init__(self, path: str = "metrics_store.json"):
        self.path = path
        self.session_counters = defaultdict(int)
        self.global_counters = defaultdict(int)
        self.load()

    def inc(self, name: str, amount: int = 1):
        # update both session + global
        self.session_counters[name] += amount
        self.global_counters[name] += amount
        self.save()

    def snapshot_session(self) -> Dict[str, int]:
        return dict(self.session_counters)

    def snapshot_global(self) -> Dict[str, int]:
        return dict(self.global_counters)

    # ---------- Persistence helpers ----------

    def save(self):
        try:
            data = dict(self.global_counters)
            with open(self.path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            print("[WARN] Failed to save Metrics:", e)

    def load(self):
        try:
            if not os.path.exists(self.path):
                return
            with open(self.path, "r") as f:
                data = json.load(f)
            self.global_counters = defaultdict(int, data)
        except Exception as e:
            print("[WARN] Failed to load Metrics:", e)


METRICS = Metrics()


def log_event(event: str, **kwargs):
    # Developer logs (not shown to end-user in user_view)
    print(f"[LOG] {event} | " + ", ".join(f"{k}={v}" for k, v in kwargs.items()))


# ============================================
# 3. Data Structures (Agents I/O)
# ============================================

@dataclass
class VisionResult:
    description: str
    confidence: float = 0.7
    notes: str = ""


@dataclass
class PolicyDecision:
    item: str
    bin_type: str
    explanation: str
    tips: str
    source: str  # "material_rules" | "local_db" | "backup_ai"


@dataclass
class CoachResponse:
    message: str
    total_points: int
    new_streak: int
    badge_unlocked: Optional[str] = None
    user_state: Optional[Dict[str, Any]] = None


# ============================================
# 4. Waste Rules DB (Ottawa)
# ============================================

RAW_RULE_CATEGORIES = [
    # --- GREEN BIN (ORGANICS) ---
    (
        "Green Bin (Organics)",
        "Food scraps and compostable kitchen waste go in the green bin.",
        [
            "apple core", "banana peel", "orange peel", "lemon peel", "lime peel",
            "grape stems", "mango peel", "melon rind", "fruit scraps",
            "vegetable peel", "potato peel", "carrot peel", "onion skins",
            "garlic skins", "broccoli stalk", "lettuce core", "cabbage leaves",
            "coffee grounds", "coffee filter", "tea bag", "loose tea leaves",
            "bread crust", "stale bread", "pasta leftovers", "rice leftovers",
            "leftover food", "plate scrapings", "pizza crust",
            "egg shell", "eggshells",
            "chicken bone", "meat bone", "fish bone",
            "paper towel", "paper napkin", "tissue (used)",
            "greasy pizza box", "soiled paper plate", "compostable paper cup",
            "small plant trimmings", "wilted flowers", "dead flowers",
            "pumpkin guts", "pumpkin seeds (cooked)", "corn cob", "corn husk",
        ],
    ),

    # --- GARBAGE ---
    (
        "Garbage",
        "Garbage items go in the regular garbage bin, not in the green bin or recycling.",
        [
            "diaper", "baby diaper", "adult diaper",
            "menstrual pad", "tampon", "panty liner",
            "plastic cutlery", "plastic fork", "plastic spoon", "plastic knife",
            "plastic straw", "drinking straw",
            "chip bag", "chips bag", "crisp packet",
            "candy wrapper", "chocolate wrapper", "snack wrapper",
            "styrofoam", "foam cup", "foam takeout container", "foam tray",
            "toothbrush", "toothpaste tube", "floss", "dental floss",
            "cotton swab", "q-tip", "cotton ball", "makeup wipe",
            "rubber band", "balloon", "latex balloon", "latex glove",
            "broken ceramic mug", "broken mug", "broken plate", "ceramic plate",
            "mirror shard", "small broken mirror",
            "plastic wrap", "plastic film", "cling film", "saran wrap",
            "plastic bag", "grocery bag", "shopping bag",
            "vacuum bag", "vacuum dust", "swept dust",
            "dryer lint", "lint from dryer filter",
            "cigarette butt", "ashtray contents (cold)",
            "disposable razor", "disposable face mask", "nitrile glove",
            "pet waste bag", "dog poop bag", "cat litter clumps (in bag)",
            "old sponge", "kitchen sponge", "scrub sponge",
            "broken toy", "small plastic toy", "rubber toy",
            "pen", "marker", "mechanical pencil",
            "hair tie", "elastic hair band", "broken hair clip",
            "old makeup brush", "mascara tube", "lipstick tube",
        ],
    ),

    # --- BLUE BIN â€“ PAPER / CARDBOARD ---
    (
        "Recycling (Blue Bin)",
        "Clean paper and cardboard go in the blue recycling bin.",
        [
            "newspaper", "magazine", "flyer", "brochure",
            "office paper", "printer paper", "notebook paper",
            "paper envelope", "window envelope",
            "paper bag", "shopping paper bag", "brown paper bag",
            "cardboard box", "shipping box", "corrugated box",
            "cereal box", "pasta box", "snack box", "shoe box",
            "paper egg carton", "cardboard egg carton",
            "toilet paper roll", "paper towel roll", "cardboard tube",
            "paper insert", "cardboard sleeve", "paper packaging",
            "paper file folder", "manila folder",
        ],
    ),

    # --- BLUE BIN â€“ CONTAINERS ---
    (
        "Recycling (Blue Bin)",
        "Clean, empty containers made of glass, metal, or accepted plastics go in the blue bin.",
        [
            "plastic water bottle", "water bottle", "pop bottle", "soda bottle",
            "juice bottle", "sports drink bottle",
            "milk jug", "juice jug", "detergent jug",
            "yogurt tub", "yogurt container", "margarine tub",
            "plastic clamshell", "berry container", "salad container",
            "tin can", "metal can", "soup can", "bean can",
            "aluminum can", "pop can", "beer can",
            "glass jar", "jam jar", "pasta sauce jar",
            "glass bottle", "olive oil bottle", "vinegar bottle",
            "metal jar lid", "metal lid", "metal bottle cap",
            "aluminum tray", "foil tray", "aluminum pie plate",
            "clean aluminum foil", "clean tinfoil",
        ],
    ),

    # --- REFUND / RETURN ---
    (
        "Refund/Return",
        "Many alcoholic beverage containers can be returned for a refund where programs exist.",
        [
            "beer bottle", "beer can", "wine bottle", "spirit bottle",
            "cooler bottle", "cooler can", "lcbo bottle",
        ],
    ),

    # --- HAZARDOUS WASTE ---
    (
        "Hazardous Waste",
        "Hazardous waste must go to a depot or collection program, not in any household bin.",
        [
            "aa battery", "aaa battery", "lithium battery", "button cell battery",
            "rechargeable battery", "power tool battery",
            "syringe", "needle", "epipen", "insulin pen",
            "paint can with paint", "leftover paint",
            "nail polish", "nail polish remover", "acetone",
            "bleach", "drain cleaner", "oven cleaner", "strong cleaner",
            "motor oil", "used motor oil", "antifreeze",
            "pesticide", "herbicide", "weed killer", "bug spray (full)",
            "propane tank", "propane cylinder", "camping fuel canister",
            "pool chemical", "chlorine tablets",
            "gasoline container with fuel", "fuel can with fuel",
            "aerosol can (full)", "spray paint can (full)",
            "medicine", "pill", "tablet", "liquid medicine", "syrup medicine",
        ],
    ),

    # --- ELECTRONIC WASTE ---
    (
        "Electronic Waste",
        "Electronics go to e-waste collection, not household garbage or recycling bins.",
        [
            "old tv", "television", "monitor",
            "laptop", "notebook computer", "desktop computer",
            "tablet", "ipad", "android tablet",
            "smartphone", "cellphone", "mobile phone",
            "printer", "scanner", "fax machine",
            "keyboard", "computer mouse",
            "router", "modem", "wifi router",
            "game console", "xbox", "playstation", "nintendo switch",
            "dvd player", "blu-ray player", "stereo receiver",
            "cable box", "satellite receiver",
            "digital camera", "camcorder",
        ],
    ),

    # --- BULKY / SPECIAL COLLECTION ---
    (
        "Special Collection / Call 3-1-1",
        "Large or bulky items usually require special collection â€“ contact the city or property manager.",
        [
            "sofa", "couch", "loveseat", "armchair",
            "mattress", "box spring", "bed frame",
            "wardrobe", "dresser", "bookshelf",
            "desk", "office chair",
            "fridge", "refrigerator", "freezer",
            "stove", "oven", "range", "cooktop",
            "dishwasher", "washing machine", "dryer",
            "water heater", "hot water tank",
            "air conditioner", "window ac unit",
            "large rug", "carpet roll",
            "door", "interior door", "cabinet door",
            "bathtub (old)", "toilet (old)", "sink (old)",
        ],
    ),

    # --- VARIES (EMPTY/RINSED RULE) ---
    (
        "Varies by material",
        "All items must be empty/rinsed before being placed in recycling or organics.",
        [
            "empty container", "rinsed container", "clean jar", "clean can",
        ],
    ),
]

WASTE_RULES_DB_OTTAWA: List[Dict[str, Any]] = []
for bin_type, notes, items in RAW_RULE_CATEGORIES:
    WASTE_RULES_DB_OTTAWA.append({
        "keywords": items,
        "bin": bin_type,
        "notes": notes,
    })

print("Rules groups:", len(WASTE_RULES_DB_OTTAWA))
print("Total item keywords:", sum(len(r["keywords"]) for r in WASTE_RULES_DB_OTTAWA))


# ============================================
# 5. Two MCP-style Tools
#    1) LocalWasteRulesTool
#    2) BackupWasteAIAssistantTool
# ============================================

class LocalWasteRulesTool:
    """
    MCP Tool #1:
      - Looks up Ottawa waste rules from local DB using simple keyword matching.
    """

    def __init__(self, rules_db: List[Dict[str, Any]]):
        self.rules_db = rules_db

    def classify(self, description: str) -> Optional[Dict[str, Any]]:
        METRICS.inc("local_tool_calls")
        text = description.lower()
        best_match = None
        best_score = 0
        for rule in self.rules_db:
            score = sum(1 for kw in rule["keywords"] if kw in text)
            if score > best_score:
                best_score = score
                best_match = rule
        if best_score == 0:
            return None
        return {
            "bin": best_match["bin"],
            "notes": best_match["notes"],
            "source": "local_db",
        }


class BackupWasteAIAssistantTool:
    """
    MCP Tool #2 (backup):
      - Asks Gemini directly for the best guess bin & explanation.
    """

    def classify(self, description: str) -> Dict[str, Any]:
        METRICS.inc("backup_tool_calls")
        prompt = f"""
        You are a waste-sorting assistant for the City of Ottawa, Canada.

        Based on your knowledge of waste sorting rules in Ottawa, determine:
        1. Which bin this item MOST LIKELY belongs in:
           (Garbage, Recycling (Blue Bin), Green Bin (Organics), Hazardous Waste, Special Drop-Off)
        2. Write a short 2â€“3 sentence explanation.

        If you are unsure, say the bin is "Unknown" and recommend checking the
        official City of Ottawa Waste Explorer or calling 3-1-1.

        Item: "{description}"
        """
        resp = model.generate_content(prompt)
        explanation = resp.text.strip()

        return {
            "bin": "Unknown (AI backup)",
            "notes": explanation,
            "source": "backup_ai",
        }


LOCAL_WASTE_TOOL = LocalWasteRulesTool(WASTE_RULES_DB_OTTAWA)
BACKUP_AI_TOOL = BackupWasteAIAssistantTool()


# ============================================
# 6. Extra Material Rules (multi-part items)
# ============================================

MATERIAL_RULES: Dict[str, Dict[str, str]] = {
    "paper cup": {
        "bin": "Green Bin (Organics)",
        "notes": "Paper coffee cups go in the green bin in Ottawa."
    },
    "coffee cup": {
        "bin": "Green Bin (Organics)",
        "notes": "Paper coffee cups go in the green bin in Ottawa."
    },
    "plastic lid": {
        "bin": "Recycling (Blue Bin)",
        "notes": "Clean plastic drink lids go in the blue bin."
    },
    "yogurt container": {
        "bin": "Recycling (Blue Bin)",
        "notes": "Clean plastic yogurt containers go in the blue bin."
    },
    "foil lid": {
        "bin": "Recycling (Blue Bin)",
        "notes": "Clean foil lids go in the blue bin (scrunched into a ball if small)."
    },
    "glass jar": {
        "bin": "Recycling (Blue Bin)",
        "notes": "Clean glass jars go in the blue bin."
    },
    "metal lid": {
        "bin": "Recycling (Blue Bin)",
        "notes": "Metal lids go in the blue bin."
    },
    "takeout bowl": {
        "bin": "Green Bin (Organics)",
        "notes": "Many fiber/cardboard takeout bowls go in the green bin if food-soiled."
    },
    "takeout container": {
        "bin": "Recycling (Blue Bin)",
        "notes": "Clean plastic takeout containers go in the blue bin."
    },
}
print("Material rule entries:", len(MATERIAL_RULES))


# ============================================
# 7. Multi-material handling helpers
# ============================================

def split_multimaterial_item(description: str) -> List[str]:
    """
    Detect known multi-material patterns like:
      - 'paper cup with plastic lid'
      - 'yogurt container with foil lid'
      - 'glass jar with metal lid'
    """
    desc = description.lower()

    patterns = [
        ("paper cup", "plastic lid"),
        ("coffee cup", "plastic lid"),
        ("cardboard cup", "plastic lid"),
        ("yogurt container", "foil lid"),
        ("glass jar", "metal lid"),
        ("takeout bowl", "plastic lid"),
        ("takeout container", "plastic lid"),
    ]

    for a, b in patterns:
        if a in desc and b in desc:
            return [a, b]

    return [desc]  # fallback: single-component


def classify_single_component(component_desc: str) -> PolicyDecision:
    """
    Policy-time tool orchestration:
      1) Check MATERIAL_RULES (hard-coded, high-confidence)
      2) Try LocalWasteRulesTool (MCP tool #1)
      3) If still unknown, call BackupWasteAIAssistantTool (MCP tool #2)
    """
    comp = component_desc.lower().strip()

    # Material rules
    if comp in MATERIAL_RULES:
        r = MATERIAL_RULES[comp]
        return PolicyDecision(
            item=component_desc,
            bin_type=r["bin"],
            explanation=r["notes"],
            tips=r["notes"],
            source="material_rules",
        )

    # Local tool
    local_res = LOCAL_WASTE_TOOL.classify(component_desc)
    if local_res is not None:
        return PolicyDecision(
            item=component_desc,
            bin_type=local_res["bin"],
            explanation=local_res["notes"],
            tips=local_res["notes"],
            source=local_res["source"],
        )

    log_event("LocalTool.miss", item=component_desc)

    # Backup MCP tool
    backup_res = BACKUP_AI_TOOL.classify(component_desc)
    if "Unknown" in backup_res["bin"]:
        METRICS.inc("unknown_bin")

    return PolicyDecision(
        item=component_desc,
        bin_type=backup_res["bin"],
        explanation=backup_res["notes"],
        tips="This suggestion is based on AI backup reasoning. Check Ottawa's tools if unsure.",
        source=backup_res["source"],
    )


def classify_item(description: str) -> List[PolicyDecision]:
    components = split_multimaterial_item(description)
    return [classify_single_component(c) for c in components]


# ============================================
# 8. Vision Agent
# ============================================

VISION_SYSTEM_PROMPT = (
    "You are an assistant that identifies household waste items from images. "
    "Describe the item in ONE short sentence, including materials if possible, "
    "for example: 'paper cup with plastic lid', 'plastic yogurt container with foil lid'. "
    "Do NOT give disposal instructions. Only describe what you see."
)

def vision_agent(image_path: str) -> VisionResult:
    """
    Agent 1: Vision Agent
    """
    img = Image.open(image_path)
    METRICS.inc("images_scanned")

    resp = model.generate_content(
        [
            VISION_SYSTEM_PROMPT,
            img,
        ]
    )
    desc = resp.text.strip()
    log_event("VisionAgent.output", description=desc)

    return VisionResult(description=desc, confidence=0.7, notes="")


# ============================================
# 9. Policy Agent
# ============================================

def policy_agent(vision: VisionResult) -> List[PolicyDecision]:
    """
    Agent 2: Policy Agent
    - Calls the MCP tools orchestrated via classify_item()
    """
    decisions = classify_item(vision.description)
    log_event(
        "PolicyAgent.decisions",
        description=vision.description,
        decisions=str([asdict(d) for d in decisions]),
    )
    return decisions


# ============================================
# 10. Coach Agent (Sessions & Memory)
# ============================================

def coach_agent(
    user_id: str,
    decisions: List[PolicyDecision],
) -> CoachResponse:
    """
    Agent 3: Coach Agent
    - Uses SESSION_STORE as memory
    - Awards points & updates streak
    """
    total_points = 0
    mistaken_items: List[str] = []

    for d in decisions:
        if d.source in ("material_rules", "local_db"):
            total_points += 10
        elif d.source == "backup_ai":
            total_points += 5
            mistaken_items.append(d.item)
        else:
            total_points += 2
            mistaken_items.append(d.item)

    all_confident = all(d.source in ("material_rules", "local_db") for d in decisions)

    state = SESSION_STORE.update_after_scan(
        user_id=user_id,
        points=total_points,
        all_confident=all_confident,
        mistaken_items=mistaken_items,
    )

    badge = None
    if state["streak"] > 0 and state["streak"] % 5 == 0:
        badge = "Recycling Hero ğŸ�… (5 confident scans in a row!)"

    if all_confident:
        msg = (
            f"Awesome, {user_id}! All parts were sorted confidently. "
            f"You earned {total_points} points. ğŸŒ±"
        )
    else:
        msg = (
            f"Nice work, {user_id}! You earned {total_points} points. "
            "Some parts used the AI backup tool; double-check tricky items in Ottawa's official tools."
        )

    METRICS.inc("scans")

    log_event(
        "CoachAgent.summary",
        user_id=user_id,
        total_points=total_points,
        new_streak=state["streak"],
        badge=badge or "None",
    )

    return CoachResponse(
        message=msg,
        total_points=total_points,
        new_streak=state["streak"],
        badge_unlocked=badge,
        user_state=state,
    )


# ============================================
# 11. Bin Assets (for user-facing UI)
# ============================================

BIN_ASSETS = {
    "Green Bin (Organics)": {
        "name": "Green Bin",
        "emoji": "ğŸŸ©",
        "image_path": "green_bin.png",  # replace with actual asset paths if you have them
    },
    "Recycling (Blue Bin)": {
        "name": "Blue Bin (Recycling)",
        "emoji": "ğŸŸ¦",
        "image_path": "blue_bin.png",
    },
    "Garbage": {
        "name": "Garbage",
        "emoji": "â¬›",
        "image_path": "garbage_bin.png",
    },
    "Hazardous Waste": {
        "name": "Hazardous Waste",
        "emoji": "ğŸŸ¥",
        "image_path": "hazard_bin.png",
    },
    "Special Collection / Call 3-1-1": {
        "name": "Special Collection",
        "emoji": "ğŸŸ¨",
        "image_path": "special_collection.png",
    },
    "Refund/Return": {
        "name": "Return for Refund",
        "emoji": "ğŸŸª",
        "image_path": "refund_bin.png",
    },
    "Varies by material": {
        "name": "Check Material",
        "emoji": "âšª",
        "image_path": "varies_bin.png",
    },
    "Unknown (AI backup)": {
        "name": "Unknown â€“ check city tools",
        "emoji": "â�“",
        "image_path": "unknown_bin.png",
    },
}


# ============================================
# 12. Customer-facing Multi-Agent Pipeline
#     Vision Agent â†’ Policy Agent â†’ Coach Agent
#     Minimal clean output for end-users
# ============================================

def scan2sort_user_view(image_path: str, user_id: str = "guest") -> Dict[str, Any]:
    """
    Customer-facing version of the pipeline.

    Shows ONLY:
      - Bin(s) with emoji/image
      - Short explanation per bin
      - Points earned this scan
      - Current streak and badge (if any)

    Hides logs, metrics, and internal debug info from the user.
    Returns a UI-friendly dict for your app/frontend.
    """

    # --- Run internal pipeline ---
    vision_res = vision_agent(image_path)
    policy_decisions = policy_agent(vision_res)
    coach_res = coach_agent(user_id, policy_decisions)

    # Group decisions by bin type
    bins_view: Dict[str, Dict[str, Any]] = {}
    for d in policy_decisions:
        btype = d.bin_type
        if btype not in bins_view:
            asset = BIN_ASSETS.get(btype, {
                "name": btype,
                "emoji": "â�“",
                "image_path": "unknown_bin.png",
            })
            bins_view[btype] = {
                "bin_type": btype,
                "display_name": asset["name"],
                "emoji": asset["emoji"],
                "image_path": asset["image_path"],
                "items": [],
                "explanations": [],
            }
        bins_view[btype]["items"].append(d.item)
        bins_view[btype]["explanations"].append(d.explanation)

    # ---------- Minimal console output for the user ----------
    print("âœ… Scan result\n")

    for binfo in bins_view.values():
        print(f"{binfo['emoji']}  {binfo['display_name']}")
        print("  Items:")
        for item in binfo["items"]:
            print(f"   â€¢ {item}")
        short_expl = binfo["explanations"][0]
        print("  Why:")
        print(f"   {short_expl}")
        print()

    print(f"â­� Points earned this scan: {coach_res.total_points}")
    print(f"ğŸ”¥ Current streak: {coach_res.new_streak}")
    if coach_res.badge_unlocked:
        print(f"ğŸ�… Badge unlocked: {coach_res.badge_unlocked}")

    # ---------- Return clean UI-friendly structure ----------
    return {
        "bins": list(bins_view.values()),
        "points_this_scan": coach_res.total_points,
        "streak": coach_res.new_streak,
        "badge": coach_res.badge_unlocked,
        "raw_description": vision_res.description,
    }


# ============================================
# 13. Quick Logic Test (no image)
# ============================================

print("\n=== Logic test (no image) ===")
logic_test_desc = "human being"
print("Input description:", logic_test_desc)

decisions = classify_item(logic_test_desc)
for d in decisions:
    print(f" - {d.item} â†’ {d.bin_type} ({d.source})")
    print(f"   {d.explanation}\n")

# To test with an actual image in Kaggle:
# 1. Upload an image, e.g. 'coffee_cup_with_lid.jpg'
# 2. Then run:
# result = scan2sort_user_view("coffee_cup_with_lid.jpg", user_id="ottawa-resident-1")



result = scan2sort_user_view("/kaggle/input/tim-horton/tim_hortons.jpg", user_id="ottawa-resident-1")





