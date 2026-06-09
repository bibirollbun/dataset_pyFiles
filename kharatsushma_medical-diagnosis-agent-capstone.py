import gradio as gr

# Medicine database
medicine_db = {
    "paracetamol": {
        "used_for": ["fever", "headache", "body pain"],
        "home_remedies": ["stay hydrated", "rest", "sponge bath for high fever"]
    },
    "cetirizine": {
        "used_for": ["cold", "sneezing", "allergy"],
        "home_remedies": ["steam inhalation", "warm water", "avoid cold drinks"]
    },
    "ors": {
        "used_for": ["dehydration", "loose motion"],
        "home_remedies": ["drink ORS", "hydration with coconut water"]
    }
}

# Agent function
def health_agent(symptoms, medicines):
    symptoms_list = [s.strip().lower() for s in symptoms.split(",") if s.strip()]
    medicines_list = [m.strip().lower() for m in medicines.split(",") if m.strip()]
    response = ""

    for med in medicines_list:
        if med in medicine_db:
            info = medicine_db[med]
            response += f"<b>Medicine:</b> {med.capitalize()}<br>"
            response += f"<b>Common use cases:</b> {', '.join(info['used_for'])}<br>"

            matches = [s for s in symptoms_list if s in info['used_for']]
            if matches:
                response += f"<span style='color:green;'>✅ Your symptoms match: {', '.join(matches)}</span><br>"
            else:
                response += f"<span style='color:red;'>⚠️ No strong match found. Please consult a doctor.</span><br>"

            response += f"<span style='color:brown;'>Home remedies: {', '.join(info['home_remedies'])}</span><br>"
            response += "See a doctor if symptoms worsen or don’t improve.<br><br>"
        else:
            response += f"<span style='color:red;'>❌ No safe info available for {med}. Please consult a doctor.</span><br><br>"

    # Previous entries
    response += f"<b>Previous entries in this session:</b><br>Symptoms: {', '.join(symptoms_list)}<br>Medicines: {', '.join(medicines_list)}<br>"

    return response

# Gradio interface
demo = gr.Interface(
    fn=health_agent,
    inputs=[
        gr.Text(label="Enter symptoms (comma separated)"),
        gr.Text(label="Enter medicine names (comma separated)")
    ],
    outputs=gr.HTML(),
    title="Medical Diagnosis Agent",
    description="Enter your symptoms and medicines to get guidance and home remedies. Matching symptoms are green, warnings red, home remedies brown."
)

demo.launch()


