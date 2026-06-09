# ---------------------------
# GlowGuard: Skincare Ingredient Checker Agent
# Paste this whole cell into a Kaggle Notebook (or split into cells as you like).
# ---------------------------

# 1) Imports
import re
import json
from difflib import get_close_matches
from collections import defaultdict

# 2) Knowledge bases (small curated lists)
# These lists are intentionally small and illustrative—expand them if you want higher coverage.
GOOD_INGREDIENTS = {
    "hyaluronic acid", "glycerin", "ceramide", "niacinamide", "panthenol",
    "squalane", "shea butter", "aloe vera", "vitamin e", "tocopherol", "azelaic acid"
}

NEUTRAL_INGREDIENTS = {
    "water", "aqua", "fragrance", "parfum", "alcohol", "butylene glycol", "propyl", "PEG"
}

# Ingredients that commonly irritate or are problematic for certain skin types:
# Map ingredient -> set of skin types it can irritate
IRRITANTS = {
    "salicylic acid": {"dry"},           # can be drying (but good for oily/acne)
    "benzoyl peroxide": {"dry", "sensitive"},
    "retinol": {"sensitive"},            # great anti-aging but irritating for sensitive skin
    "fragrance": {"sensitive"},
    "linalool": {"sensitive"},
    "limonene": {"sensitive"},
    "menthol": {"sensitive"},
    "essential oil": {"sensitive"},
    "alcohol": {"dry", "sensitive"},
    "sulfate": {"sensitive"},
    "denatured alcohol": {"dry", "sensitive"},
    "alpha hydroxy acid": {"sensitive", "dry"},
    "glycolic acid": {"sensitive", "dry"},
    "salicylate": {"dry"},
    # add more as needed
}

# Mapping common synonyms to canonical names
SYNONYMS = {
    "aqua": "water",
    "butylene glycol": "butylene glycol",
    "niacinamide": "niacinamide",
    "vitamin e": "tocopherol",
    "tocoferol": "tocopherol",
    "bpo": "benzoyl peroxide",
    "bp": "benzoyl peroxide",
}

# Safety categories default thresholds
# Scoring weights for ingredient categories (tweakable)
WEIGHTS = {
    "good": 2,
    "neutral": 0,
    "irritant_for_skin": -3,   # strong negative if known irritant for that skin type
    "irritant_general": -2,    # negative if irritant but not specific to that skin type
    "unknown": -1
}

# 3) Utility functions
def normalize_ingredient(raw: str) -> str:
    """Lowercase, strip, remove trailing amounts and parentheses and common noisy tokens."""
    s = raw.lower().strip()
    # remove parenthetical content
    s = re.sub(r"\(.*?\)", "", s)
    # remove pct, mg, numbers
    s = re.sub(r"[\d\%]+", "", s)
    # strip punctuation
    s = re.sub(r"[^a-z0-9\s\-\/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # common synonyms
    if s in SYNONYMS:
        s = SYNONYMS[s]
    return s

def split_ingredients(text: str) -> list:
    """Split a typical ingredient list string into candidate ingredient tokens."""
    # Common separators: comma, slash, semicolon
    parts = re.split(r"[,/;•\n]", text)
    cleaned = [normalize_ingredient(p) for p in parts if p.strip()]
    # further split by 'and' if necessary
    final = []
    for c in cleaned:
        sub = re.split(r"\band\b", c)
        for s in sub:
            s2 = s.strip()
            if s2:
                final.append(s2)
    # dedupe while preserving order
    seen = set()
    ordered = []
    for i in final:
        if i not in seen:
            seen.add(i)
            ordered.append(i)
    return ordered

def fuzzy_match(ingredient: str, candidates: set, cutoff=0.8):
    """Try to match ingredient string to a candidate using close matches (difflib)."""
    if not ingredient or len(ingredient) < 2:
        return None
    matches = get_close_matches(ingredient, candidates, n=1, cutoff=cutoff)
    return matches[0] if matches else None

# 4) Core agent logic
def classify_ingredient_for_skin(ingredient: str, skin_type: str):
    """
    Returns a tuple: (category, explanation)
    category in {"good", "neutral", "irritant", "unknown"}
    """
    ing = ingredient.lower()
    # direct canonical mapping
    if ing in GOOD_INGREDIENTS:
        return "good", f"{ingredient} — commonly beneficial (hydration/repair)."
    if ing in NEUTRAL_INGREDIENTS:
        return "neutral", f"{ingredient} — usually neutral or formulation ingredient."
    # check irritants mapping (exact or fuzzy)
    for key in IRRITANTS:
        if key in ing or ing in key:
            # determine if irritant applies to user's skin type
            applies = skin_type.lower() in IRRITANTS[key]
            if applies:
                return "irritant", f"{ingredient} — known to irritate {skin_type} skin (ingredient: {key})."
            else:
                return "irritant", f"{ingredient} — potential irritant (monitor/tolerate cautiously)."
    # fuzzy match against known lists
    fm_good = fuzzy_match(ing, GOOD_INGREDIENTS)
    if fm_good:
        return "good", f"{ingredient} — matched to {fm_good}, considered beneficial."
    fm_neu = fuzzy_match(ing, NEUTRAL_INGREDIENTS)
    if fm_neu:
        return "neutral", f"{ingredient} — matched to {fm_neu}, formulation ingredient."
    fm_irr = fuzzy_match(ing, IRRITANTS.keys())
    if fm_irr:
        applies = skin_type.lower() in IRRITANTS[fm_irr]
        if applies:
            return "irritant", f"{ingredient} — fuzzy matched to irritant {fm_irr} (affects {skin_type})."
        else:
            return "irritant", f"{ingredient} — fuzzy matched to {fm_irr}, possibly an irritant."
    # fallback unknown
    return "unknown", f"{ingredient} — ingredient not in our small KB; patch test recommended."

def compute_score(classifications: list):
    """
    Given list of (ingredient, category) tuples, compute a 0-100 safety score.
    We start from neutral baseline 50, then add/subtract weighted points,
    and clamp to [0, 100].
    """
    score = 50
    for ing, cat, explanation in classifications:
        if cat == "good":
            score += WEIGHTS["good"]
        elif cat == "neutral":
            score += WEIGHTS["neutral"]
        elif cat == "irritant":
            # treat as general irritant (if explanation mentions specific skin type we could weight more)
            if "known to irritate" in explanation:
                score += WEIGHTS["irritant_for_skin"]
            else:
                score += WEIGHTS["irritant_general"]
        elif cat == "unknown":
            score += WEIGHTS["unknown"]
    # normalize to 0-100
    if score < 0:
        score = 0
    if score > 100:
        score = 100
    # convert to integer
    return int(score)

# 5) Top-level pipeline function
def check_ingredients_pipeline(ingredient_text: str, skin_type: str = "normal"):
    """
    Input:
      ingredient_text: full ingredient list (string)
      skin_type: one of {"dry","oily","sensitive","normal","combination"}
    Output: dict with parsed ingredients, classifications, score, verdict, and advice
    """
    parsed = split_ingredients(ingredient_text)
    classifications = []
    for ing in parsed:
        cat, expl = classify_ingredient_for_skin(ing, skin_type)
        classifications.append((ing, cat, expl))

    score = compute_score(classifications)

    # verdict thresholds (tunable)
    if score >= 70:
        verdict = "Safe"
        advice = "Looks generally safe for your skin type. Proceed normally."
    elif 45 <= score < 70:
        verdict = "Patch Test Recommended"
        advice = "Some ingredients may be problematic. Patch test before full use."
    else:
        verdict = "Avoid"
        advice = "Product may irritate your skin. Consider avoiding or consult a dermatologist."

    # assemble structured result
    details = []
    for ing, cat, expl in classifications:
        details.append({
            "ingredient": ing,
            "category": cat,
            "explanation": expl
        })

    result = {
        "input_text": ingredient_text,
        "skin_type": skin_type,
        "score": score,
        "verdict": verdict,
        "advice": advice,
        "details": details
    }
    return result

# 6) Pretty-print helper
def print_result(result: dict):
    print(f"\n=== GlowGuard Result — Skin Type: {result['skin_type'].title()} ===")
    print(f"Safety Score: {result['score']} / 100 — Verdict: {result['verdict']}")
    print(f"Advice: {result['advice']}\n")
    print("Ingredient breakdown:")
    for d in result["details"]:
        print(f" • {d['ingredient'][:60]:60} | {d['category']:9} | {d['explanation']}")
    print("\nJSON (for attaching):")
    print(json.dumps(result, indent=2))

# 7) Demo examples
demo_ingredients = [
    # Example 1: hydrating serum (should be mostly safe for dry)
    ("Aqua, Glycerin, Hyaluronic Acid, Butylene Glycol, Phenoxyethanol, Fragrance", "dry"),

    # Example 2: acne treatment (contains benzoyl peroxide & salicylic)
    ("Water, Benzoyl Peroxide 5%, Salicylic Acid, Isopropyl Myristate, Alcohol Denat., Sodium Hydroxide", "sensitive"),

    # Example 3: exfoliant (glycolic)
    ("Water, Glycolic Acid (AHA), Lactic Acid, Aloe Vera, Fragrance, Limonene", "dry"),
]

# Run demos
for text, skin in demo_ingredients:
    res = check_ingredients_pipeline(text, skin)
    print_result(res)

# 8) Example: Using as a function in Kaggle Notebook
# Save to variable & show JSON for attaching in writeup or for tests
example_product = demo_ingredients[0][0]
example_skin = "dry"
glowguard_output = check_ingredients_pipeline(example_product, example_skin)
# If needed, you can save glowguard_output to a JSON attachment file:
# import json
# with open("glowguard_output.json","w") as f:
#     json.dump(glowguard_output, f, indent=2)

# End of cell


