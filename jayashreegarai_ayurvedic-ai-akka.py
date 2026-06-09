# 1ï¸�âƒ£ Import Libraries
import matplotlib.pyplot as plt
from ipywidgets import widgets, VBox, HBox, Output
import json
from datetime import datetime
from IPython.display import display, HTML

# File for persistent memory
USER_DATA_FILE = "akka_user_data.json"



# 2ï¸�âƒ£ Dosha Quiz Widgets
dosha_questions = {
    "digestion": ["Poor", "Moderate", "Good"],
    "sleep": ["Poor", "Moderate", "Good"],
    "skin": ["Dry", "Oily", "Combination"],
    "energy": ["Low", "Medium", "High"],
    "cravings": ["Yes", "No"]
}

quiz_widgets = {q: widgets.Dropdown(options=opts, description=q.capitalize()) for q, opts in dosha_questions.items()}
display(VBox(list(quiz_widgets.values())))



# 3ï¸�âƒ£ Season Selection Widget
season_widget = widgets.Dropdown(
    options=["Winter", "Summer", "Monsoon", "Spring", "Autumn"],
    description="Season:"
)
display(season_widget)



# 4ï¸�âƒ£ DoshaAnalyzer Class
class DoshaAnalyzer:
    """Analyze dosha based on quiz answers and persist results."""
    def __init__(self, answers):
        self.answers = answers
        self.vata = 33
        self.pitta = 33
        self.kapha = 33
        self.dominant = None

    def calculate_dosha(self):
        if self.answers['digestion'] == "Poor": self.vata += 10
        if self.answers['digestion'] == "Good": self.kapha += 5
        if self.answers['sleep'] == "Poor": self.vata += 10
        if self.answers['sleep'] == "Good": self.kapha += 5
        if self.answers['skin'] == "Dry": self.vata += 10
        if self.answers['skin'] == "Oily": self.pitta += 10
        if self.answers['energy'] == "Low": self.kapha += 10
        if self.answers['cravings'] == "Yes": self.pitta += 5
        
        dominant_score = max(self.vata, self.pitta, self.kapha)
        if dominant_score == self.vata:
            self.dominant = "Vata"
        elif dominant_score == self.pitta:
            self.dominant = "Pitta"
        else:
            self.dominant = "Kapha"
        return self.dominant
    
    def get_percentages(self):
        return {"Vata": self.vata, "Pitta": self.pitta, "Kapha": self.kapha}

    def save_user_data(self, season):
        data = {
            "answers": self.answers,
            "dominant": self.dominant,
            "percentages": self.get_percentages(),
            "season": season,
            "timestamp": str(datetime.now())
        }
        try:
            with open(USER_DATA_FILE, "w") as f:
                json.dump(data, f)
        except:
            pass

    @staticmethod
    def load_user_data():
        try:
            with open(USER_DATA_FILE, "r") as f:
                return json.load(f)
        except:
            return None



# 5ï¸�âƒ£ DinacharyaPlanner Class
class DinacharyaPlanner:
    """Plan daily routine based on dominant dosha and seasonal guidance."""
    routines = {
        "Vata": [
            "Wake up early, meditate 10 min",
            "Warm sesame oil massage (Abhyanga)",
            "Herbal tea: Ginger/Cinnamon",
            "Breakfast: Warm porridge",
            "Lunch: Steamed vegetables + grains",
            "Light evening walk",
            "Dinner: Soups or khichdi",
            "Sleep by 10 PM"
        ],
        "Pitta": [
            "Wake up early, cool breathing exercises",
            "Coconut/sunflower oil massage",
            "Herbal tea: Peppermint/Chamomile",
            "Breakfast: Fruits + grains",
            "Lunch: Salads + rice + lentils",
            "Evening meditation",
            "Dinner: Light cooked meals",
            "Sleep by 10 PM"
        ],
        "Kapha": [
            "Wake up early, brisk walk",
            "Mustard/sesame oil massage",
            "Herbal tea: Ginger/Tulsi",
            "Breakfast: Warm cooked grains",
            "Lunch: Light vegetables",
            "Evening yoga",
            "Dinner: Light meal",
            "Sleep by 10 PM"
        ]
    }
    
    seasonal_routines = {
        "Winter": {
            "Vata": "Warm foods, oil massage, grounding meditation.",
            "Pitta": "Moderate activity, light cooling foods, calm meditation.",
            "Kapha": "Increase warm spices, brisk walks, light meals."
        },
        "Summer": {
            "Vata": "Stay hydrated, light cooling foods, morning yoga.",
            "Pitta": "Stay cool, reduce heat-inducing foods, calming meditation.",
            "Kapha": "Moderate exercise, avoid heavy foods, cooling herbs."
        },
        "Monsoon": {
            "Vata": "Avoid dampness, warm foods, herbal teas.",
            "Pitta": "Light meals, cooling drinks, calm breathing.",
            "Kapha": "Stay active, avoid heavy/sticky foods, warm spices."
        },
        "Spring": {
            "Vata": "Increase grounding foods, gentle exercise, herbal teas.",
            "Pitta": "Cooling foods, calm meditation, moderate walks.",
            "Kapha": "Light meals, energizing exercises, dry herbs."
        },
        "Autumn": {
            "Vata": "Warm oils, grounding meditation, cooked meals.",
            "Pitta": "Cooling foods, calm yoga, hydration.",
            "Kapha": "Light warm meals, brisk walks, herbal teas."
        }
    }
    
    @classmethod
    def get_routine(cls, dosha):
        return cls.routines.get(dosha, cls.routines["Vata"])
    
    @classmethod
    def get_seasonal_advice(cls, dosha, season):
        return cls.seasonal_routines.get(season, {}).get(dosha, "")



# 6ï¸�âƒ£ WellnessAdvisor Class
class WellnessAdvisor:
    remedies = {
        "Vata": "Stay warm, drink warm teas, follow calm routine, sesame oil massage.",
        "Pitta": "Stay cool, avoid spicy foods, use cooling teas, meditation.",
        "Kapha": "Stay active, light meals, dry teas like ginger/Tulsi, brisk walks."
    }
    
    teas = {
        "Vata": "Ginger/Cinnamon tea",
        "Pitta": "Peppermint/Chamomile tea",
        "Kapha": "Ginger/Tulsi tea"
    }
    
    meditation = {
        "Vata": "Slow deep belly breathing, grounding meditation",
        "Pitta": "Alternate nostril breathing, calming meditation",
        "Kapha": "Energizing yoga, short meditation sessions"
    }
    
    food_plan = {
        "Vata": "Warm cooked grains, soups, steamed veggies, avoid raw/cold foods",
        "Pitta": "Cooling foods, salads, fruits, avoid spicy/oily foods",
        "Kapha": "Light cooked meals, warm teas, avoid heavy dairy and sugar"
    }
    
    @classmethod
    def get_remedy(cls, dosha): return cls.remedies.get(dosha)
    @classmethod
    def get_tea(cls, dosha): return cls.teas.get(dosha)
    @classmethod
    def get_meditation(cls, dosha): return cls.meditation.get(dosha)
    @classmethod
    def get_food_plan(cls, dosha): return cls.food_plan.get(dosha)



# 7ï¸�âƒ£ Womenâ€™s Health Advisor
women_problem_input = widgets.Text(description="Concern:", placeholder="e.g., cramps, bloating, mood swings")
women_problem_button = widgets.Button(description="Get Advice ğŸŒ¸")
women_problem_output = Output()

def womens_menstrual_advice(dosha, season, concern):
    advice_dict = {
        "Vata": {
            "cramps": f"Warm compress, sesame oil massage, ginger tea, light yoga, warm meals. Seasonal tip: {DinacharyaPlanner.get_seasonal_advice('Vata', season)}"
        },
        "Pitta": {
            "cramps": f"Cool compress, avoid spicy foods, chamomile tea, light meditation. Seasonal tip: {DinacharyaPlanner.get_seasonal_advice('Pitta', season)}"
        },
        "Kapha": {
            "cramps": f"Light exercise, warm compress, tulsi/ginger tea, brisk walk, light meals. Seasonal tip: {DinacharyaPlanner.get_seasonal_advice('Kapha', season)}"
        }
    }
    return advice_dict.get(dosha, {}).get(concern.lower(), f"Gentle exercise, warm tea, and rest. Seasonal tip: {DinacharyaPlanner.get_seasonal_advice(dosha, season)}")

def on_women_problem_submit(b):
    with women_problem_output:
        women_problem_output.clear_output()
        dosha = analyzer.dominant
        season = season_widget.value
        advice = womens_menstrual_advice(dosha, season, women_problem_input.value)
        display(HTML(f"<h4>ğŸŒ¿ Akka's Advice for {women_problem_input.value.capitalize()}:</h4><p>{advice}</p>"))

women_problem_button.on_click(on_women_problem_submit)
display(VBox([women_problem_input, women_problem_button, women_problem_output]))



# 8ï¸�âƒ£ Visualization
def plot_dosha_percentages(percentages):
    names = list(percentages.keys())
    values = list(percentages.values())
    plt.figure(figsize=(6,4))
    plt.bar(names, values, color=['#FFB347','#FF6961','#77DD77'])
    plt.title("Dosha Percentages")
    plt.ylabel("Percentage")
    plt.ylim(0,100)
    plt.show()



# 9ï¸�âƒ£ Run Button & Integration
output = Output()
run_button = widgets.Button(description="Run Ayurvedic AI Akka ğŸŒ¿")

def on_run_click(b):
    global analyzer
    with output:
        output.clear_output()
        # Collect quiz answers
        answers = {k: w.value for k, w in quiz_widgets.items()}
        season = season_widget.value
        analyzer = DoshaAnalyzer(answers)
        dominant = analyzer.calculate_dosha()
        percentages = analyzer.get_percentages()
        analyzer.save_user_data(season)
        
        # HTML output
        html_content = f"""
        <h3>ğŸ’› Dominant Dosha: {dominant}</h3>
        <p>Dosha Percentages: {percentages}</p>
        <h4>âœ¨ Personalized Daily Routine:</h4>
        <ol>"""
        for act in DinacharyaPlanner.get_routine(dominant):
            html_content += f"<li>{act}</li>"
        html_content += "</ol>"
        html_content += f"""
        <h4>ğŸŒ¸ Seasonal (Ritucharya) Advice:</h4><p>{DinacharyaPlanner.get_seasonal_advice(dominant, season)}</p>
        <h4>ğŸ’› Remedies & Tips:</h4><p>{WellnessAdvisor.get_remedy(dominant)}</p>
        <h4>ğŸ�µ Herbal Tea:</h4><p>{WellnessAdvisor.get_tea(dominant)}</p>
        <h4>ğŸ§˜ Meditation & Breathing:</h4><p>{WellnessAdvisor.get_meditation(dominant)}</p>
        <h4>ğŸ¥— Food Plan:</h4><p>{WellnessAdvisor.get_food_plan(dominant)}</p>
        """
        display(HTML(html_content))
        plot_dosha_percentages(percentages)
        display(VBox([women_problem_input, women_problem_button, women_problem_output]))

run_button.on_click(on_run_click)
display(run_button, output)

# âœ… Pre-filled Demo Run if no input
user_data = DoshaAnalyzer.load_user_data()
if user_data:
    analyzer = DoshaAnalyzer(user_data['answers'])
    analyzer.dominant = user_data['dominant']
    percentages_demo = user_data['percentages']
    season_demo = user_data['season']
    print(f"ğŸ’› Previous Dominant Dosha: {analyzer.dominant}")
    print(f"Dosha Percentages: {percentages_demo}")
    print(f"Season: {season_demo}")
else:
    # Demo
    sample_answers = {"digestion":"Good","sleep":"Moderate","skin":"Dry","energy":"Medium","cravings":"No"}
    analyzer = DoshaAnalyzer(sample_answers)
    dominant_demo = analyzer.calculate_dosha()
    percentages_demo = analyzer.get_percentages()
    season_demo = "Winter"
    analyzer.save_user_data(season_demo)
    print(f"ğŸ’› Demo Dominant Dosha: {dominant_demo}")
    print(f"Dosha Percentages: {percentages_demo}")
    print(f"Season: {season_demo}")


