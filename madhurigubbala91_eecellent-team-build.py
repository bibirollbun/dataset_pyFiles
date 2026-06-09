# Simple Health AI Agent
class HealthAgent:
    def __init__(self):
        # risk scoring (toy model)
        self.symptom_weights = {
            "fever": 2,
            "cough": 1,
            "chest_pain": 5,
            "fatigue": 1,
            "shortness_of_breath": 5,
            "headache": 1
        }

    def assess_risk(self, symptoms):
        score = 0
        for s in symptoms:
            score += self.symptom_weights.get(s, 0)
        return score

    def recommend(self, symptoms):
        score = self.assess_risk(symptoms)

        if score >= 7:
            return "High risk âš ï¸� â€” Seek medical attention immediately."
        elif score >= 3:
            return "Medium risk â€” Monitor symptoms and consider telehealth."
        else:
            return "Low risk â€” Rest, hydrate, and observe."

# Example use
agent = HealthAgent()
symptoms = ["cough", "fever", "fatigue"]

print("Symptoms:", symptoms)
print("Recommendation:", agent.recommend(symptoms))



ğŸŒ± Real-life Sustainability Example: Energy-Saving Home Agent
âœ… Python Example: AI Agent Saving Home Electricity

A rule-based + adaptive agent that tries to reduce electricity usage.


class EnergyAgent:
    def __init__(self):
        self.energy_threshold = 5  # kWh target per hour

    def sense(self, current_usage):
        return current_usage

    def decide(self, usage):
        if usage > self.energy_threshold:
            return "turn_off_heater"
        else:
            return "keep_all_on"

    def act(self, decision):
        if decision == "turn_off_heater":
            return "Heater turned off to save energy."
        return "Energy usage normal. No action taken."

# Simulate environment
agent = EnergyAgent()

usage_now = 7.2  # current kWh
decision = agent.decide(usage_now)
result = agent.act(decision)

print("Usage:", usage_now)
print("Decision:", decision)
print("Action:", result)



ğŸŒ¿ Java Example: Sustainability Agent (Simple Rule-Based AI)


public class SustainabilityAgent {
    private double threshold = 6.0;

    public String decide(double co2Level) {
        if (co2Level > threshold) {
            return "Increase ventilation";
        } else {
            return "Air quality normal";
        }
    }

    public static void main(String[] args) {
        SustainabilityAgent agent = new SustainabilityAgent();

        double currentCO2 = 8.5;

        System.out.println("CO2 Level: " + currentCO2);
        System.out.println("Action: " + agent.decide(currentCO2));
    }
}


