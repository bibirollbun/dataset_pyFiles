# Session memory to store interaction history
session_memory = {"last_list": None, "last_output": None}


# Expanded Categorization Rules
CATEGORIES = {
    "groceries": ["milk", "bread", "eggs", "apple", "banana", "rice", "tomato", "vegetable", "fruit", "snack"],
    "household": ["soap", "detergent", "cleaner", "sponge", "towel", "tissue", "garbage bag"],
    "personal_care": ["shampoo", "toothpaste", "lotion", "cream", "facewash", "deodorant"],
    "electronics": ["charger", "headphone", "battery", "usb", "cable", "adapter", "power bank"],
    "kitchen_tools": ["pan", "knife", "spatula", "plate", "bowl", "container", "cutting board"],
    "school_supplies": ["notebook", "pen", "pencil", "marker", "glue", "ruler", "eraser"],
    "clothes_and_footwear": ["shirt", "t-shirt", "jeans", "shoe", "socks", "jacket", "hoodie"],
    "skincare": ["moisturizer", "sunscreen", "cleanser", "toner", "serum", "mask"],
}



# Custom Tool: Categorizer
def categorize_items(items):
    result = {category: [] for category in CATEGORIES}
    result["uncategorized"] = []

    for item in items:
        added = False
        lower_item = item.lower()
        for category, keywords in CATEGORIES.items():
            if any(k in lower_item for k in keywords):
                result[category].append(item)
                added = True
                break
        if not added:
            result["uncategorized"].append(item)
    return result



# Planner Agent
def planner(items):
    if not items:
        return "EMPTY_LIST"
    return "CATEGORIZE"



# Final Output Agent
def format_output(result):
    output = "ğŸ›�ï¸� Organized Shopping List\n\n"
    for category, items in result.items():
        output += f"**{category.replace('_', ' ').title()}**:\n"
        if items:
            for i in items:
                output += f" - {i}\n"
        else:
            output += " (none)\n"
        output += "\n"
    return output



# Main Agent Pipeline
def shopping_agent(input_text):
    items = [i.strip() for i in input_text.split(",") if i.strip()]
    session_memory["last_list"] = items
    decision = planner(items)
    if decision == "EMPTY_LIST":
        return "Please enter some shopping items."
    categorized = categorize_items(items)
    final_output = format_output(categorized)
    session_memory["last_output"] = final_output
    return final_output



# Demo example (edit items to test)
print(shopping_agent("milk, charger, notebook, shampoo, pan, sunscreen, jeans, apples, glue, towel, serum"))



# Simple evaluation
print("Last Input:", session_memory["last_list"])
print("\nLast Output:\n", session_memory["last_output"])
assert isinstance(session_memory["last_list"], list)
assert "Organized Shopping List" in session_memory["last_output"]
print("\nâœ” Evaluation successful â€” agent is working correctly.")


