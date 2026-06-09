# Simple Health Information Agent
# Works offline, no external libraries required.

disease_data = {
    "fever": {
        "symptoms": ["High temperature", "Headache", "Body pain", "Fatigue"],
        "causes": ["Infection", "Flu", "Bacterial illness"],
        "solutions": ["Drink water", "Paracetamol", "Rest", "Cold compress"],
        "prevention": ["Avoid infections", "Wash hands", "Healthy diet"]
    },
    "diabetes": {
        "symptoms": ["Frequent urination", "Increased thirst", "Fatigue"],
        "causes": ["Insulin resistance", "Genetics", "Obesity"],
        "solutions": ["Control sugar intake", "Exercise", "Medication"],
        "prevention": ["Healthy lifestyle", "Avoid sugar", "Regular checkups"]
    },
    "cold": {
        "symptoms": ["Runny nose", "Sneezing", "Cough"],
        "causes": ["Virus infection"],
        "solutions": ["Steam inhalation", "Warm water", "Rest"],
        "prevention": ["Wash hands", "Avoid cold drinks", "Wear warm clothes"]
    }
}


def health_agent(query):
    query = query.lower()
    for disease, data in disease_data.items():
        if disease in query:
            print(f"\n--- {disease.capitalize()} Information ---")
            print("Symptoms:", ", ".join(data["symptoms"]))
            print("Causes:", ", ".join(data["causes"]))
            print("Solutions:", ", ".join(data["solutions"]))
            print("Prevention:", ", ".join(data["prevention"]))
            return
    print("Sorry, disease not found in the database.")


# ---- Example Usage ----
while True:
    user_input = input("\nAsk about any disease (type 'exit' to quit): ")
    if user_input.lower() == "exit":
        break
    health_agent(user_input)


