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


# StyleSphere AI â€” Enhanced Boutique Demo (v2)
# Integrates: recommend_matches, shopping cart, improved images, size advisor, styling/UI tweaks,
# semantic search (keyword-based), and an in-app analytics dashboard.
# Single-file Gradio app. Install: pip install google-generativeai gradio pillow requests

import os
import json
import random
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import requests
from io import BytesIO
from PIL import Image, ImageOps

import google.generativeai as genai
import gradio as gr

# ---------------- Config ----------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", None)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("âœ… Gemini configured from environment variable.")
else:
    print("âš ï¸� No GEMINI_API_KEY found. StylistAgent will use a rule-based fallback.")

# ---------------- Logging / Analytics (in-memory) ----------------
ANALYTICS_LOGS: List[dict] = []

def log_event(agent_name: str, event: str, data: Optional[dict] = None):
    payload = {"time": datetime.utcnow().isoformat() + "Z", "agent": agent_name, "event": event, "data": data or {}}
    ANALYTICS_LOGS.append(payload)
    print(f"[LOG] {json.dumps(payload)}")

# ---------------- Data models & demo catalog ----------------
@dataclass
class Product:
    id: str
    name: str
    brand: str
    price: float
    image_url: str
    colors: List[str]
    categories: List[str]
    sizes: List[str] = field(default_factory=lambda: ["XS","S","M","L","XL"])  # demo


PRODUCTS = [
    Product(id="1", name="Pastel Linen Shirt", brand="Breeze", price=40.0,
            image_url="https://th.bing.com/th/id/R.a363ffe5a170b30b77fc152db85849b3?rik=82ub7fnWdHXwmg&riu=http%3a%2f%2flovebrand.com%2fcdn%2fshop%2ffiles%2fMens_Pastel_Pink_Abaco_Linen_Shirt_front_view.jpg%3fv%3d1726838486&ehk=8EhBkBWeuV2lmv8CvXIgzDlxRWfIyxY37YvNvr2I3KQ%3d&risl=&pid=ImgRaw&r=0", colors=["pastel-pink","white"], categories=["top","smart-casual","pastel"]),
    Product(id="2", name="Cream Knit Sweater", brand="CozyHome", price=55.0,
            image_url="https://static.vecteezy.com/system/resources/previews/047/590/091/non_2x/close-up-of-a-cozy-cream-colored-cable-knit-sweater-png.png", colors=["cream"], categories=["top","casual","neutral"]),
    Product(id="3", name="White Cotton Tee", brand="Everyday", price=20.0,
            image_url="https://i5.walmartimages.com/seo/White-Tshirt-for-Men-Gildan-2000-Men-T-Shirt-Cotton-Men-Shirt-Original-Men-s-Shirts-Best-Mens-Classic-Short-Sleeve-Tee_c9ecef4f-8e1e-4f24-8786-e88f1846255c.a79d87519fef1e4e9d626ff3b446742c.jpeg", colors=["white"], categories=["top","casual","basic"]),
    Product(id="4", name="Soft Blue Blouse", brand="Skyline", price=45.0,
            image_url="https://tse4.mm.bing.net/th/id/OIP.SvJrur0g-1CPKaeIE_k1WAHaLH?rs=1&pid=ImgDetMain&o=7&rm=3", colors=["pastel-blue"], categories=["top","smart-casual","pastel"]),
    Product(id="5", name="White Chino Pants", brand="UrbanEase", price=45.0,
            image_url="https://tse4.mm.bing.net/th/id/OIP.35l8L5anX9swsILPD_SUwQHaKE?rs=1&pid=ImgDetMain&o=7&rm=3", colors=["white"], categories=["bottom","smart-casual","neutral"]),
    Product(id="6", name="Beige Wide-Leg Trousers", brand="MinimalCo", price=60.0,
            image_url="https://tse3.mm.bing.net/th/id/OIP.WvvNfVStpqXpa6PuXbna1AHaKR?rs=1&pid=ImgDetMain&o=7&rm=3", colors=["beige"], categories=["bottom","formal","minimal"]),
    Product(id="7", name="Light Wash Mom Jeans", brand="DenimDays", price=50.0,
            image_url="https://cdn-img.prettylittlething.com/e/f/e/a/efea09069977b3be7e02500b8a2c4cc2f80cdeb9_cms6132_5.jpg", colors=["light-blue"], categories=["bottom","casual"]),
    Product(id="8", name="Pastel Maxi Dress", brand="Sunset Bloom", price=70.0,
            image_url="https://th.bing.com/th/id/OIP.WVparMyHoA5QWyPPwTs1QAHaKN?o=7rm=3&rs=1&pid=ImgDetMain&o=7&rm=3", colors=["pastel-blue","pastel-pink"], categories=["dress","occasion","wedding","pastel"]),
    Product(id="9", name="Satin Slip Dress", brand="GlowWear", price=75.0,
            image_url="https://th.bing.com/th/id/R.86bf43045c50dc89d844e585fba178a4?rik=9M0O4wWxceSrpg&riu=http%3a%2f%2fwww.umelondon.com%2fcdn%2fshop%2fproducts%2fmidi-satin-slip-dress-with-split-655748.jpg%3fv%3d1680430438&ehk=cp0Xeik6cNqOYNdIL2HSlQsdnqpczw3dcFtPQL0acpE%3d&risl=&pid=ImgRaw&r=0", colors=["champagne"], categories=["dress","evening","date"]),
    Product(id="10", name="Floral Day Dress", brand="Gardenia", price=55.0,
            image_url="https://tse1.mm.bing.net/th/id/OIP.UCuzXkkzghWWPWfzgzGfjAHaLW?rs=1&pid=ImgDetMain&o=7&rm=3", colors=["white","pastel-pink"], categories=["dress","casual","day"]),
    Product(id="11", name="Tan Loafers", brand="StepRight", price=50.0,
            image_url="data:image/webp;base64,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", colors=["tan"], categories=["shoes","smart-casual"]),
    Product(id="12", name="Strappy Sandals", brand="SoleWave", price=35.0,
            image_url="https://tse4.mm.bing.net/th/id/OIP.xg78cTeqIENLF7ZkIbwUDQHaJ3?w=1340&h=1785&rs=1&pid=ImgDetMain&o=7&rm=3", colors=["beige"], categories=["shoes","occasion","summer"]),
    Product(id="13", name="White Sneakers", brand="StreetStep", price=45.0,
            image_url="https://tse3.mm.bing.net/th/id/OIP.8z5ASaHJDnVdGB3BbUiRSgHaHa?rs=1&pid=ImgDetMain&o=7&rm=3", colors=["white"], categories=["shoes","casual"]),
    Product(id="14", name="Statement Pearl Earrings", brand="Aurora", price=20.0,
            image_url="https://i.etsystatic.com/18666804/r/il/5b31b3/2064840521/il_1588xN.2064840521_97ik.jpg", colors=["white"], categories=["accessory","occasion"]),
    Product(id="15", name="Cream Tote Bag", brand="DailyCarry", price=40.0,
            image_url="https://tse1.explicit.bing.net/th/id/OIP.sV7hjbIbQhkxsDDHwjxwCgHaFj?rs=1&pid=ImgDetMain&o=7&rm=3", colors=["cream"], categories=["accessory","casual","minimal"]),
    Product(id="16", name="Gold Layered Necklace", brand="GlowWear", price=22.0,
            image_url="https://via.placeholder.com/600x600.png?text=Gold+Necklace", colors=["gold"], categories=["accessory"]),
]

# ---------------- Utilities ----------------
def product_by_id(pid: str) -> Optional[Product]:
    for p in PRODUCTS:
        if p.id == str(pid):
            return p
    return None


def safe_fetch_image(url: str, size: Tuple[int,int]=(400,400)):
    """Fetch image and return a PIL Image; on failure return a neutral placeholder image."""
    try:
        resp = requests.get(url, timeout=4)
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        img = ImageOps.fit(img, size, Image.LANCZOS)
        return img
    except Exception as e:
        log_event("Image", "fetch_error", {"url": url, "error": str(e)})
        # generate a simple placeholder
        placeholder = Image.new("RGB", size, (250,250,253))
        return placeholder


def product_search(max_price: Optional[float] = None, min_price: Optional[float] = None,
                   colors: Optional[List[str]] = None, categories: Optional[List[str]] = None, text_query: Optional[str] = None, limit: int = 20) -> List[Product]:
    results = []
    q = (text_query or "").lower().strip()
    for p in PRODUCTS:
        if max_price is not None and p.price > max_price:
            continue
        if min_price is not None and p.price < min_price:
            continue
        if colors and not any(c in p.colors for c in colors):
            continue
        if categories and not any(cat in p.categories for cat in categories):
            continue
        # simple semantic-ish match: check words in name, brand, categories
        if q:
            score = 0
            for tok in q.split():
                if tok in p.name.lower() or tok in p.brand.lower():
                    score += 2
                if tok in " ".join(p.categories):
                    score += 1
                if tok in ",".join(p.colors):
                    score += 1
            if score == 0:
                continue
        results.append(p)
    log_event("product_search", "filtered", {"query": text_query, "count": len(results)})
    return results[:limit]


def color_harmony_score(items: List[Product]) -> float:
    if not items:
        return 0.0
    score = 0
    for p in items:
        for c in p.colors:
            if "pastel" in c or c in ["white", "beige", "tan", "cream"]:
                score += 2
            else:
                score += 1
    return min(1.0, score / (len(items) * 3))

# ---------------- Session & simple persistent store ----------------
@dataclass
class UserPreferences:
    budget: Optional[float] = None
    styles: List[str] = field(default_factory=list)
    disliked_colors: List[str] = field(default_factory=list)
    preferred_fit: Optional[str] = None

@dataclass
class WardrobeItem:
    id: str
    name: str
    colors: List[str]
    category: str
    notes: str = ""

@dataclass
class SessionState:
    user_id: str
    preferences: UserPreferences = field(default_factory=UserPreferences)
    wardrobe: List[WardrobeItem] = field(default_factory=list)
    last_outfits: List[dict] = field(default_factory=list)
    saved_outfits: List[dict] = field(default_factory=list)
    cart: List[dict] = field(default_factory=list)  # items with qty & size

class SessionService:
    def __init__(self):
        self.sessions: Dict[str, SessionState] = {}
    def get_session(self, user_id: str) -> SessionState:
        if user_id not in self.sessions:
            self.sessions[user_id] = SessionState(user_id=user_id)
            log_event("SessionService", "create_session", {"user_id": user_id})
        return self.sessions[user_id]

session_service = SessionService()
DEMO_USER_ID = "demo_user"

# ---------------- Agents (stylist fallback) ----------------
class StylistAgent:
    def __init__(self):
        self.model = genai.GenerativeModel("gemini-1.5-pro") if GEMINI_API_KEY else None
    def create_outfit(self, user_message: str, session: SessionState, candidate_products: List[Product]) -> dict:
        log_event("StylistAgent", "create_outfit_start", {"msg": user_message})
        return self._fallback_outfit(user_message, session, candidate_products)
    def _fallback_outfit(self, user_message: str, session: SessionState, candidate_products: List[Product]) -> dict:
        items = []
        lower = user_message.lower()
        if "wedding" in lower:
            dresses = [p for p in candidate_products if "dress" in p.categories]
            shoes = [p for p in candidate_products if "shoes" in p.categories]
            accessories = [p for p in candidate_products if "accessory" in p.categories]
            if dresses:
                items.append(random.choice(dresses).id)
            if shoes:
                items.append(random.choice(shoes).id)
            if accessories:
                items.append(random.choice(accessories).id)
        else:
            tops = [p for p in candidate_products if "top" in p.categories]
            bottoms = [p for p in candidate_products if "bottom" in p.categories]
            shoes = [p for p in candidate_products if "shoes" in p.categories]
            if tops:
                items.append(random.choice(tops).id)
            if bottoms:
                items.append(random.choice(bottoms).id)
            if shoes:
                items.append(random.choice(shoes).id)
        total_price = sum(product_by_id(pid).price for pid in items)
        return {"name": "Simple Styled Look", "description": "A simple fallback outfit.", "item_ids": items, "style_tags": ["fallback"], "estimated_price": total_price}

stylist_agent = StylistAgent()

# ---------------- Budget & closet agents ----------------
class BudgetAgent:
    def optimize_outfit(self, outfit: dict, session: SessionState) -> dict:
        budget = session.preferences.budget
        if budget is None:
            return outfit
        ids = outfit["item_ids"]
        items = [product_by_id(pid) for pid in ids if product_by_id(pid)]
        total = sum(p.price for p in items)
        if total <= budget:
            outfit["budget_status"] = f"âœ… Within budget (${total:.2f} of ${budget:.2f})"
            return outfit
        items_sorted = sorted(items, key=lambda p: p.price, reverse=True)
        for expensive in items_sorted:
            candidates = product_search(max_price=expensive.price - 10, categories=expensive.categories, limit=3)
            if not candidates:
                continue
            cheaper = candidates[0]
            outfit["item_ids"].remove(expensive.id)
            outfit["item_ids"].append(cheaper.id)
            break
        new_items = [product_by_id(pid) for pid in outfit["item_ids"] if product_by_id(pid)]
        total_new = sum(p.price for p in new_items)
        status = "âœ…" if total_new <= budget else "âš ï¸�"
        outfit["estimated_price"] = total_new
        outfit["budget_status"] = f"{status} Adjusted price: ${total_new:.2f} (budget ${budget:.2f})"
        return outfit

budget_agent = BudgetAgent()

class ClosetAgent:
    def add_item(self, session: SessionState, name: str, colors: List[str], category: str, notes: str = ""):
        wid = f"w{len(session.wardrobe)+1}"
        item = WardrobeItem(id=wid, name=name, colors=colors, category=category, notes=notes)
        session.wardrobe.append(item)
        log_event("ClosetAgent", "add_item", {"id": wid})
        return item

closet_agent = ClosetAgent()

# ---------------- Trend agent ----------------
class TrendAgent:
    def __init__(self):
        self.trends = []
        self.last_refreshed = None
    def refresh_trends(self):
        log_event("TrendAgent", "refresh_start", {})
        time.sleep(0.2)
        now = datetime.utcnow().isoformat() + "Z"
        self.last_refreshed = now
        self.trends = [
            {"name": "Soft Pastel Wedding Guest", "vibe": "Light, romantic, flowy silhouettes in pastel tones.", "tags": ["pastel","wedding","romantic"]},
            {"name": "Minimal Resort Linen", "vibe": "Crisp whites and linens, relaxed tailoring, beach-perfect.", "tags": ["minimal","linen","resort"]},
            {"name": "Statement Accessories", "vibe": "Clean base outfits with bold earrings and bags.", "tags": ["accessories","statement","elevated-basics"]},
        ]
        log_event("TrendAgent", "refresh_done", {"count": len(self.trends)})
        return self.trends, now
    def get_trends(self):
        if not self.trends:
            trends, _ = self.refresh_trends()
            return trends
        return self.trends

trend_agent = TrendAgent()

# ---------------- Evaluation & UI helpers ----------------
COLOR_MAP = {"pastel-pink":"#ffc1cc","pastel-blue":"#c4e1ff","white":"#ffffff","cream":"#f5f5e8","beige":"#f5f5dc","tan":"#d2b48c","light-blue":"#cde4ff","gold":"#f3c969","default":"#cccccc"}

def color_to_hex(c: str) -> str:
    return COLOR_MAP.get(c.lower(), COLOR_MAP["default"])


def build_products_html(items: list, show_match_buttons: bool = False) -> str:
    if not items:
        return "<div style='font-size:13px; opacity:0.7;'>No products to show.</div>"
    cards = []
    for p in items:
        img_url = p.get("image_url", "")
        name = p.get("name", "Item")
        brand = p.get("brand", "")
        price = p.get("price", 0)
        colors = ", ".join(p.get("colors", []))
        match_btn = "<div style='font-size:12px;opacity:0.85;margin-top:6px;'>Click for matches</div>" if show_match_buttons else ""
        cards.append(f"""
        <div style="min-width: 180px; max-width: 220px; background:#ffffff; border-radius:14px; border:1px solid #f3e8ff; box-shadow:0 6px 16px rgba(15,23,42,0.06); padding:10px; display:flex; flex-direction:column; gap:6px;">
          <div style="width:100%;border-radius:10px;overflow:hidden;background:#f9fafb;"><img src=\"{img_url}\" alt=\"{name}\" style=\"width:100%;height:180px;object-fit:cover;display:block;\" /></div>
          <div style="font-size:12px;font-weight:600;line-height:1.2;">{name}</div>
          <div style="font-size:11px;opacity:0.7;">{brand}</div>
          <div style="font-size:12px;font-weight:600;color:#4b164c;">${price:.2f}</div>
          <div style="font-size:11px;opacity:0.7;">{colors}</div>
          {match_btn}
        </div>
        """)
    return f"<div style='overflow-x:auto;padding:4px 0 8px 0;'><div style='display:flex;flex-wrap:nowrap;gap:12px;'>{''.join(cards)}</div></div>"


def build_outfit_board_image(items: list):
    if not items:
        return None
    try:
        imgs = []
        target_h = 220
        for p in items:
            url = p.get("image_url")
            if not url:
                continue
            resp = requests.get(url, timeout=5)
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            w, h = img.size
            new_w = int(w * (target_h / h))
            imgs.append(img.resize((new_w, target_h)))
        if not imgs:
            return None
        total_w = sum(img.size[0] for img in imgs)
        board = Image.new("RGB", (total_w, target_h), (250, 250, 255))
        x = 0
        for img in imgs:
            board.paste(img, (x, 0))
            x += img.size[0]
        return board
    except Exception as e:
        log_event("UI", "board_image_error", {"error": str(e)})
        return None

# ---------------- Closet bias ----------------
def apply_closet_bias(session: SessionState, candidates: List[Product]) -> List[Product]:
    closet_colors = set()
    for item in session.wardrobe:
        closet_colors.update(item.colors)
    if not closet_colors:
        return candidates
    return sorted(candidates, key=lambda p: len(closet_colors & set(p.colors)), reverse=True)

# ---------------- Recommend matches ----------------
def score_candidate_combo(top: Product, bottom: Product, shoe: Product, accessory: Product, session: SessionState):
    items = [top, bottom, shoe, accessory]
    color_score = color_harmony_score(items)
    top_cats = set(top.categories)
    common = 0
    for other in (bottom, shoe, accessory):
        if top_cats & set(other.categories):
            common += 1
    category_score = common / 3.0
    total = sum(p.price for p in items)
    target = top.price * 2.0
    price_diff = abs(total - target) / max(target, 1.0)
    price_score = max(0.0, 1.0 - price_diff)
    score = 0.5 * color_score + 0.3 * category_score + 0.2 * price_score
    explain = f"{top.name} pairs with {bottom.name}, {shoe.name} and {accessory.name} â€” color harmony {color_score:.2f}, vibe match {category_score:.2f}."
    return score, explain, total


def recommend_matches(product_id: str, user_id: str = DEMO_USER_ID, limit: int = 3):
    session = session_service.get_session(user_id)
    top = product_by_id(product_id)
    if not top:
        return []
    bottoms = [p for p in PRODUCTS if "bottom" in p.categories]
    shoes = [p for p in PRODUCTS if "shoes" in p.categories]
    accessories = [p for p in PRODUCTS if "accessory" in p.categories]
    bottoms = apply_closet_bias(session, bottoms)
    shoes = apply_closet_bias(session, shoes)
    accessories = apply_closet_bias(session, accessories)
    combos = []
    for b in bottoms[:6]:
        for s in shoes[:6]:
            for a in accessories[:6]:
                score, explain, total = score_candidate_combo(top, b, s, a, session)
                combos.append({"score": score, "explain": explain, "estimated_price": total, "items": [b.id, s.id, a.id], "items_full": [b.__dict__, s.__dict__, a.__dict__]})
    combos_sorted = sorted(combos, key=lambda c: c["score"], reverse=True)[:limit]
    results = []
    for idx, c in enumerate(combos_sorted, start=1):
        title = f"Match #{idx}"
        one_liner = f"{top.name} + {c['items_full'][0]['name']} + {c['items_full'][1]['name']} â€” {c['explain'].split('â€”')[-1].strip()}"
        results.append({"title": title, "one_liner": one_liner, "item_ids": [top.id] + c["items"], "estimated_price": c["estimated_price"], "score": c["score"], "images": [top.image_url] + [it["image_url"] for it in c["items_full"]], "items_full": c["items_full"]})
    log_event("recommend_matches", "generated", {"top": top.id, "user": user_id, "count": len(results)})
    return results

# ---------------- Shopping cart ----------------
def add_to_cart(user_id: str, product_id: str, qty: int = 1, size: Optional[str] = None):
    session = session_service.get_session(user_id)
    p = product_by_id(product_id)
    if not p:
        return "Product not found."
    # if same item+size exists, increment
    for it in session.cart:
        if it.get("product_id") == p.id and it.get("size") == size:
            it["qty"] += qty
            log_event("Cart", "update_qty", {"user": user_id, "product": p.id, "qty": it["qty"], "size": size})
            return "Updated cart."
    session.cart.append({"product_id": p.id, "qty": qty, "size": size})
    log_event("Cart", "add", {"user": user_id, "product": p.id, "qty": qty, "size": size})
    return "Added to cart."


def cart_summary(user_id: str):
    session = session_service.get_session(user_id)
    items = []
    total = 0.0
    for it in session.cart:
        p = product_by_id(it["product_id"])
        if not p:
            continue
        subtotal = p.price * it["qty"]
        items.append({"id": p.id, "name": p.name, "brand": p.brand, "price": p.price, "qty": it["qty"], "size": it.get("size"), "subtotal": subtotal})
        total += subtotal
    return items, total


def remove_from_cart(user_id: str, product_id: str, size: Optional[str] = None):
    session = session_service.get_session(user_id)
    before = len(session.cart)
    session.cart = [it for it in session.cart if not (it["product_id"] == product_id and it.get("size") == size)]
    after = len(session.cart)
    log_event("Cart", "remove", {"user": user_id, "product": product_id, "size": size, "removed": before - after})
    return "Removed item(s) from cart."

# ---------------- Size advisor (simple heuristic) ----------------
def size_advice_for_product(product: Product, height_cm: int, weight_kg: int, usual_size: Optional[str] = None) -> str:
    # naive mapping just for demo purposes
    bmi = weight_kg / ((height_cm/100) ** 2) if height_cm and weight_kg else 22
    # pick size by BMI thresholds (toy example)
    if bmi < 19:
        size = "S"
    elif bmi < 24:
        size = "M"
    elif bmi < 29:
        size = "L"
    else:
        size = "XL"
    # prefer usual_size if provided and exists in product
    if usual_size and usual_size in product.sizes:
        preferred = usual_size
    else:
        preferred = size if size in product.sizes else product.sizes[min(len(product.sizes)-1, 2)]
    advice = f"Based on height {height_cm}cm and weight {weight_kg}kg, we suggest size **{preferred}** (approx. BMI {bmi:.1f})."
    log_event("SizeAdvisor", "advice", {"product": product.id, "height": height_cm, "weight": weight_kg, "advice_size": preferred})
    return advice

# ---------------- Semantic-ish search helper ----------------
SYNONYMS = {
    "flowy": ["flowy","flowing","flowy"],
    "pastel": ["pastel","soft","light"],
    "linen": ["linen"],
    "casual": ["casual","everyday","day"],
}

def expand_query_terms(q: str) -> List[str]:
    terms = q.lower().split()
    expanded = set(terms)
    for t in terms:
        for k, vals in SYNONYMS.items():
            if t == k or t in vals:
                expanded.update(vals)
    return list(expanded)

# ---------------- Recommend matches UI helpers ----------------
def build_match_cards_html(matches: list) -> str:
    if not matches:
        return "<div style='font-size:13px; opacity:0.7;'>No matches found.</div>"
    cards = []
    for m in matches:
        imgs_html = "".join([f"<img src='{u}' style='width:80px;height:80px;object-fit:cover;border-radius:8px;margin-right:6px;' />" for u in m['images']])
        cards.append(f"""
        <div style="min-width:240px;max-width:280px;background:#fff;border-radius:14px;border:1px solid #f3e8ff;padding:12px;display:flex;flex-direction:column;gap:8px;">
          <div style="font-weight:650;color:#4b164c;">{m['title']}</div>
          <div style="font-size:12px;opacity:0.9;">{m['one_liner']}</div>
          <div style="display:flex;align-items:center;margin-top:6px;">{imgs_html}</div>
          <div style="display:flex;justify-content:space-between;align-items:center;margin-top:8px;">
            <div style="font-weight:600;color:#4b164c;">${m['estimated_price']:.2f}</div>
            <div style="font-size:12px;opacity:0.85;">Score: {m['score']:.2f}</div>
          </div>
        </div>
        """)
    return f"<div style='overflow-x:auto;padding:4px 0 8px 0;'><div style='display:flex;gap:12px;'>{''.join(cards)}</div></div>"

# ---------------- Saved outfits ----------------

def save_match_as_outfit(user_id: str, match: dict):
    session = session_service.get_session(user_id)
    items = [product_by_id(pid).__dict__ for pid in match["item_ids"] if product_by_id(pid)]
    session.saved_outfits.append({"outfit": {"name": match.get("title", "Matched Look"), "item_ids": match["item_ids"], "estimated_price": match.get("estimated_price", 0.0), "style_tags": []}, "items": items})
    log_event("SavedMatch", "save", {"user": user_id, "title": match.get("title")})
    return f"Saved '{match.get('title', 'Look')}' to your Saved Looks."

# ---------------- Orchestrator simplified ----------------
class Orchestrator:
    def handle_message(self, user_id: str, message: str) -> dict:
        session = session_service.get_session(user_id)
        # detect budget in message (simple)
        words = message.lower().replace("$", "").split()
        nums = [w for w in words if w.replace('.', '', 1).isdigit()]
        if nums:
            try:
                session.preferences.budget = float(nums[0])
            except:
                pass
        candidates = product_search(max_price=session.preferences.budget, limit=20) if session.preferences.budget else PRODUCTS
        candidates = apply_closet_bias(session, candidates)
        outfit = stylist_agent.create_outfit(message, session, candidates)
        outfit = budget_agent.optimize_outfit(outfit, session)
        items = [product_by_id(pid).__dict__ for pid in outfit.get("item_ids", []) if product_by_id(pid)]
        outfit["scores"] = {"overall": 0.8, "color_harmony": 0.9}
        session.last_outfits.append(outfit)
        log_event("Orchestrator", "outfit_generated", {"user": user_id, "name": outfit.get('name')})
        return {"type": "outfit", "outfit": outfit, "items": items}

orchestrator = Orchestrator()

# ---------------- Gradio UI ----------------

theme = gr.themes.Soft(primary_hue="pink", secondary_hue="indigo", radius_size="lg").set(
    body_background_fill="#faf5ff",
    body_text_color="#111827",
    block_background_fill="#ffffff",
    block_border_width="1px",
    block_border_color="#f3e8ff",
)

with gr.Blocks(title="StyleSphere AI â€” Enhanced Boutique", theme=theme) as demo:
    gr.Markdown("""
# ğŸŒ¸ StyleSphere AI â€” Enhanced Boutique
Now with: cart, size advisor, semantic-ish search, improved images, analytics, and match-this-top flow.
""")

    with gr.Tab("Shop"):
        with gr.Row():
            with gr.Column(scale=3):
                search_box = gr.Textbox(label="Search (try: 'flowy pastel top')", placeholder="Search products or describe a look")
                search_btn = gr.Button("Search")
                results_html = gr.HTML(label="Results")
                # catalog sections
                gr.Markdown("**Catalog**")
                catalog_all_html = gr.HTML(label="All products")
            with gr.Column(scale=1):
                gr.Markdown("### ğŸ›’ Cart & Size Advisor")
                cart_md = gr.Markdown(label="Cart")
                checkout_btn = gr.Button("Checkout (simulate)")
                gr.Markdown("---")
                gr.Markdown("### Size Advisor")
                sa_product = gr.Dropdown(choices=[], label="Product for size advice")
                sa_height = gr.Number(label="Height (cm)", value=170)
                sa_weight = gr.Number(label="Weight (kg)", value=65)
                sa_usize = gr.Dropdown(choices=["","XS","S","M","L","XL"], label="Your usual size (optional)")
                sa_btn = gr.Button("Get size advice")
                sa_out = gr.Markdown()

        # match flow quick access
        with gr.Row():
            with gr.Column(scale=2):
                top_select = gr.Dropdown(choices=[], label="Select a top to find matches")
                find_matches_btn = gr.Button("Find matches for this top")
            with gr.Column(scale=2):
                matches_html = gr.HTML(label="Matches")

    with gr.Tab("Stylist Chat"):
        with gr.Row():
            with gr.Column(scale=3):
                chat_input = gr.Textbox(label="Tell the stylist what you need", placeholder="e.g. outfit for office meeting under $100 in neutral tones")
                chat_send = gr.Button("Ask stylist")
                chat_out = gr.Markdown()
            with gr.Column(scale=1):
                gr.Markdown("### Quick Actions")
                quick_top = gr.Dropdown(choices=[], label="Quick top")
                quick_match = gr.Button("Find matches for quick top")

    with gr.Tab("My Closet"):
        with gr.Row():
            with gr.Column(scale=1):
                cname = gr.Textbox(label="Item name")
                ccolors = gr.Textbox(label="Colors (comma-separated)")
                ccat = gr.Dropdown(["top","bottom","dress","shoes","accessory","outerwear"], label="Category", value="top")
                cnotes = gr.Textbox(label="Notes")
                add_closet_btn = gr.Button("Add to Closet")
                add_closet_msg = gr.Markdown()
            with gr.Column(scale=2):
                closet_html = gr.HTML()

    with gr.Tab("Saved Looks"):
        saved_html = gr.HTML()
        refresh_saved_btn = gr.Button("Refresh saved looks")

    with gr.Tab("Trends & Analytics"):
        with gr.Row():
            with gr.Column():
                trends_info = gr.Markdown()
                trends_html = gr.HTML()
            with gr.Column():
                gr.Markdown("### Admin Analytics (in-memory)")
                logs_json = gr.Textbox(label="Recent logs (JSON)")
                clear_logs_btn = gr.Button("Clear logs")

    # Hidden state
    matches_state = gr.State([])

    # ---------- wiring ----------
    def load_catalog_all():
        items = [p.__dict__ for p in PRODUCTS]
        return build_products_html(items, show_match_buttons=True)

    demo.load(load_catalog_all, inputs=None, outputs=[catalog_all_html])

    def populate_product_dropdowns():
        opts = [(p.id, f"{p.name} â€” ${p.price:.2f}") for p in PRODUCTS if "top" in p.categories]
        # Gradio Dropdown accepts list of strings or tuples depending on version; return list of (value, label) tuples
        return [o for o in opts]

    demo.load(populate_product_dropdowns, inputs=None, outputs=[top_select, quick_top])

    # Search
    def on_search(q: str):
        if not q:
            items = [p.__dict__ for p in PRODUCTS]
            return build_products_html(items)
        expanded = expand_query_terms(q)
        # do multiple queries and union results
        found = []
        for term in expanded:
            found += product_search(text_query=term, limit=20)
        # dedupe
        seen = set()
        uniq = []
        for p in found:
            if p.id not in seen:
                seen.add(p.id)
                uniq.append(p.__dict__)
        log_event("Search", "query", {"query": q, "expanded": expanded, "found": len(uniq)})
        return build_products_html(uniq)

    search_btn.click(on_search, inputs=[search_box], outputs=[results_html])

    # Size advisor
    def load_sa_product_choices():
        return [(p.id, f"{p.name} â€” ${p.price:.2f}") for p in PRODUCTS]

    demo.load(load_sa_product_choices, inputs=None, outputs=[sa_product])

    def on_size_advice(prod_choice, h, w, usual):
        if not prod_choice:
            return "Please select a product."
        p = product_by_id(prod_choice)
        if not p:
            return "Product not found."
        advice = size_advice_for_product(p, int(h or 0), int(w or 0), usual)
        return advice

    sa_btn.click(on_size_advice, inputs=[sa_product, sa_height, sa_weight, sa_usize], outputs=[sa_out])

    # Match flow wiring
    def on_find_matches(top_id):
        if not top_id:
            return "Select a top.", "<div style='font-size:13px; opacity:0.7;'>No matches yet.</div>", []
        matches = recommend_matches(top_id, DEMO_USER_ID, limit=3)
        html = build_match_cards_html(matches)
        return f"Showing {len(matches)} matches.", html, matches

    find_matches_btn.click(on_find_matches, inputs=[top_select], outputs=[gr.Markdown(), matches_html, matches_state])

    # Cart wiring: add buttons in product cards are static HTML (not interactive). Provide explicit add-to-cart helper UI for demo
    def add_to_cart_ui(product_id: str, qty: int, size: str):
        if not product_id:
            return "Select a product id to add.", *cart_summary_ui(DEMO_USER_ID)
        msg = add_to_cart(DEMO_USER_ID, product_id, qty or 1, size if size else None)
        return msg, *cart_summary_ui(DEMO_USER_ID)

    def cart_summary_ui(user_id: str):
        session = session_service.get_session(user_id)
        items = []
        total = 0.0
        for it in session.cart:
            p = product_by_id(it["product_id"])
            if not p:
                continue
            subtotal = p.price * it["qty"]
            items.append({"id": p.id, "name": p.name, "brand": p.brand, "price": p.price, "qty": it["qty"], "size": it.get("size"), "subtotal": subtotal})
            total += subtotal
        return "\n".join(lines), items, total

    # wire a small add-to-cart form (product id input) to demonstrate cart
    add_cart_pid = gr.Dropdown(choices=[(p.id, f"{p.name} â€” ${p.price:.2f}") for p in PRODUCTS], label="Add product to cart")
    add_cart_qty = gr.Slider(minimum=1, maximum=5, step=1, value=1, label="Qty")
    add_cart_size = gr.Dropdown(choices=["","XS","S","M","L","XL"], label="Size (optional)")
    add_cart_btn = gr.Button("Add to cart")

    def add_cart_click(pid, qty, size):
        msg = add_to_cart(DEMO_USER_ID, pid, int(qty or 1), size if size else None)
        log_event("UI", "add_cart_click", {"product": pid, "qty": qty, "size": size})
        cart_text, _, _ = cart_summary_ui(DEMO_USER_ID)
        return msg, cart_text

    add_cart_btn.click(add_cart_click, inputs=[add_cart_pid, add_cart_qty, add_cart_size], outputs=[gr.Markdown(), cart_md])

    checkout_btn.click(lambda: (log_event("Cart","checkout",{"user":DEMO_USER_ID}), "Checkout simulated â€” thanks!"), inputs=None, outputs=[gr.Markdown()])

    # Closet add
    def show_closet_html():
        session = session_service.get_session(DEMO_USER_ID)
        items = session.wardrobe
        if not items:
            return "<div style='font-size:13px; opacity:0.7;'>Your closet is empty. Add a few favorites!</div>"
        cards = []
        for w in items:
            colors = ", ".join(w.colors)
            cards.append(f"""
            <div style="min-width: 180px; max-width: 220px; background:#ffffff; border-radius:14px; border:1px solid #f3e8ff; box-shadow:0 6px 16px rgba(15,23,42,0.06); padding:10px; display:flex; flex-direction:column; gap:4px;">
              <div style="font-size:12px;font-weight:600;">{w.name}</div>
              <div style="font-size:11px;opacity:0.7;">Category: {w.category}</div>
              <div style="font-size:11px;opacity:0.7;">Colors: {colors}</div>
              <div style="font-size:11px;opacity:0.7;">{w.notes}</div>
            </div>
            """)
        return f"<div style='overflow-x:auto;padding:4px 0 8px 0;'><div style='display:flex;flex-wrap:nowrap;gap:12px;'>{''.join(cards)}</div></div>"

    demo.load(show_closet_html, inputs=None, outputs=[closet_html])

    def add_closet_item_ui(name, colors_text, category, notes):
        session = session_service.get_session(DEMO_USER_ID)
        if not name:
            return "Please provide a name.", show_closet_html()
        colors = [c.strip() for c in (colors_text or "").split(",") if c.strip()]
        item = closet_agent.add_item(session, name, colors, category, notes)
        return f"Added '{item.name}' to your closet.", show_closet_html()

    add_closet_btn.click(add_closet_item_ui, inputs=[cname, ccolors, ccat, cnotes], outputs=[add_closet_msg, closet_html])

    # Saved looks
    def load_saved_outfits_ui():
        session = session_service.get_session(DEMO_USER_ID)
        saved_list = session.saved_outfits
        if not saved_list:
            return "<div style='font-size:13px; opacity:0.7;'>You haven't saved any looks yet.</div>"
        cards = []
        for idx, entry in enumerate(saved_list, start=1):
            outfit = entry.get('outfit', {})
            items = entry.get('items', [])
            name = outfit.get('name', f"Look #{idx}")
            total = outfit.get('estimated_price', 0.0)
            count = len(items)
            tags = ", ".join(outfit.get('style_tags', []))
            cards.append(f"""
            <div style="min-width: 220px; max-width: 260px; background:#ffffff; border-radius:16px; border:1px solid #f3e8ff; box-shadow:0 6px 16px rgba(15,23,42,0.06); padding:12px; display:flex; flex-direction:column; gap:6px;">
              <div style="font-size:13px;font-weight:650;color:#4b164c;">{name}</div>
              <div style="font-size:11px;opacity:0.85;">{count} piece(s) Â· Total ${total:.2f}</div>
              <div style="font-size:11px;opacity:0.8;"><span style="font-weight:600;">Tags:</span> {tags or 'â€”'}</div>
            </div>
            """)
        return f"<div style='overflow-x:auto;padding:4px 0 8px 0;'><div style='display:flex;flex-wrap:nowrap;gap:12px;'>{''.join(cards)}</div></div>"

    demo.load(load_saved_outfits_ui, inputs=None, outputs=[saved_html])
    refresh_saved_btn.click(load_saved_outfits_ui, inputs=None, outputs=[saved_html])

    # Trends & analytics
    def load_trends_ui():
        trends = trend_agent.get_trends()
        ts = trend_agent.last_refreshed
        info = f"Last refreshed at: {ts}" if ts else "Trends not refreshed yet."
        cards = []
        for t in trends:
            cards.append(f"""
            <div style="min-width: 220px; max-width: 260px; background:#ffffff; border-radius:16px; border:1px solid #f3e8ff; box-shadow:0 6px 16px rgba(15,23,42,0.06); padding:12px; display:flex; flex-direction:column; gap:6px;">
              <div style="font-size:13px;font-weight:650;color:#4b164c;">{t['name']}</div>
              <div style="font-size:11px;opacity:0.85;">{t['vibe']}</div>
              <div style="font-size:11px;opacity:0.8;"><span style="font-weight:600;">Tags:</span> {', '.join(t['tags'])}</div>
            </div>
            """)
        html = f"<div style='overflow-x:auto;padding:4px 0 8px 0;'><div style='display:flex;gap:12px;'>{''.join(cards)}</div></div>"
        return info, html

    demo.load(load_trends_ui, inputs=None, outputs=[trends_info, trends_html])

    def get_logs_json():
        return json.dumps(ANALYTICS_LOGS[-200:], indent=2)

    demo.load(get_logs_json, inputs=None, outputs=[logs_json])

    def clear_logs():
        global ANALYTICS_LOGS
        ANALYTICS_LOGS = []
        return "Cleared logs." , get_logs_json()

    clear_logs_btn.click(clear_logs, inputs=None, outputs=[gr.Markdown(), logs_json])

    # Stylist chat
    def on_chat_send(text: str):
        if not text:
            return "Say something to the stylist."
        res = orchestrator.handle_message(DEMO_USER_ID, text)
        if res.get('type') == 'outfit':
            outfit = res['outfit']
            items = [product_by_id(pid).__dict__ for pid in outfit.get('item_ids', []) if product_by_id(pid)]
            md = f"**{outfit.get('name')}**\n\n{outfit.get('description')}\n\nTotal: ${outfit.get('estimated_price',0):.2f}\n\nWhy: A balanced look with good color harmony."
            return md
        return "I can help style outfits."

    chat_send.click(on_chat_send, inputs=[chat_input], outputs=[chat_out])

    # Quick match from chat panel
    def quick_match_click(top_choice):
        matches = recommend_matches(top_choice, DEMO_USER_ID, limit=3)
        return build_match_cards_html(matches)

    quick_match.click(quick_match_click, inputs=[quick_top], outputs=[gr.HTML()])

    demo.launch()

# End of file

