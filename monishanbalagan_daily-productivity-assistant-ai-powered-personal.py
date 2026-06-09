# Install dependencies 

!pip install requests
!pip install ipywidgets
!pip install python-dotenv

print("Dependencies are available. No installation required on Kaggle.")


# %% 
# Essential imports and Kaggle Secrets setup
import os
import json
import time
from datetime import datetime
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import ssl

# Kaggle Secrets setup
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    print("âœ… Kaggle Secrets client initialized successfully!")
except Exception as e:
    print(f"âš ï¸� Kaggle Secrets not available: {e}")
    user_secrets = None

# IPython display imports
try:
    from IPython.display import display, Markdown, HTML, clear_output
    print("âœ… IPython display modules imported successfully!")
except Exception as e:
    print(f"âš ï¸� IPython display modules not available: {e}")

# Gemini API setup
gemini_api_key = None
if user_secrets:
    try:
        gemini_api_key = user_secrets.get_secret("gemini-api-key")
        if gemini_api_key:
            print("âœ… Gemini API key retrieved successfully!")
            from google.generativeai import GenerativeModel, configure
            configure(api_key=gemini_api_key)
        else:
            print("â�Œ Gemini API key not found in Kaggle Secrets")
    except Exception as e:
        print(f"â�Œ Error getting Gemini API key: {e}")
else:
    print("â�Œ Kaggle Secrets client not available")


try:
    from kaggle_secrets import UserSecretsClient
    _KAGGLE_SECRETS = UserSecretsClient()
except Exception:
    _KAGGLE_SECRETS = None

class WeatherService:
    """
    Inline weather helper.
    - Looks for 'openweather-api-key' in Kaggle Secrets or environment variables.
    - Uses OpenWeatherMap geocoding + current weather endpoints when key is available.
    - Returns a readable text summary or None if everything fails.
    """
    def __init__(self, secret_name="openweather-api-key"):
        self.secret_name = secret_name
        self.api_key = self._get_secret(secret_name)

    def _get_secret(self, name):
        if _KAGGLE_SECRETS:
            try:
                v = _KAGGLE_SECRETS.get_secret(name)
                if v and v.strip():
                    return v.strip()
            except Exception:
                pass
        return os.environ.get(name)

    def _geocode(self, location):
        """Return (lat, lon, city, country) or None on failure."""
        if not self.api_key:
            return None
        clean = location.split(",")[0].strip()
        url = f"http://api.openweathermap.org/geo/1.0/direct?q={clean}&limit=1&appid={self.api_key}"
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
            if not data:
                return None
            d = data[0]
            return d.get("lat"), d.get("lon"), d.get("name"), d.get("country")
        except Exception:
            return None

    def get_weather(self, location="San Francisco, CA"):
        """
        Return a formatted weather summary string.
        If OpenWeather is unavailable, returns a deterministic fallback summary.
        """
        coords = self._geocode(location)
        if not coords:
            return self._fallback_weather(location)

        lat, lon, city, country = coords
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=imperial&appid={self.api_key}"
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            d = r.json()

            temp = d.get("main", {}).get("temp")
            feels = d.get("main", {}).get("feels_like")
            hum = d.get("main", {}).get("humidity")
            wind = d.get("wind", {}).get("speed")
            desc = d.get("weather", [{}])[0].get("description", "").title()

            summary = (
                f"Current Weather for {city}, {country}\n"
                f"Temperature: {temp:.1f} Â°F (feels like {feels:.1f} Â°F)\n"
                f"Conditions: {desc}\n"
                f"Wind: {wind} mph\n"
                f"Humidity: {hum}%\n"
            )

            alerts = d.get("alerts")
            if alerts:
                summary += "\nAlerts:\n"
                for a in alerts:
                    summary += f"- {a.get('event')}: {a.get('description','')[:200]}\n"

            return summary

        except Exception:
            return self._fallback_weather(location)

    def _fallback_weather(self, location):
        """Deterministic fallback so planner still gives reasonable advice."""
        now = datetime.now()
        month = now.month
        if month in (12, 1, 2):
            temps = (52, 60) 
            cond = "Cool, partly cloudy"
            outfit = "Light jacket"
        elif month in (6, 7, 8):
            temps = (78, 90)  
            cond = "Warm and sunny"
            outfit = "T-shirt and light pants"
        else:
            temps = (62, 75)
            cond = "Mild, partly cloudy"
            outfit = "Layered clothing"

        temp = sum(temps) / 2
        summary = (
            f"Weather (fallback) for {location}\n"
            f"Estimated Temperature: {temp:.0f} Â°F\n"
            f"Conditions: {cond}\n"
            f"Suggested outfit: {outfit}\n"
            f"Note: This is a fallback summary â€” supply `openweather-api-key` for real data.\n"
        )
        return summary

# Example usage 
ws = WeatherService()
print(ws.get_weather("India"))


_MODEL_AVAILABLE = False
try:
    from google.generativeai import GenerativeModel, configure
    try:
        if 'UserSecretsClient' in globals() and _KAGGLE_SECRETS is not None:
            _gkey = _KAGGLE_SECRETS.get_secret("gemini-api-key")
        else:
            _gkey = os.environ.get("gemini-api-key")
    except Exception:
        _gkey = os.environ.get("gemini-api-key")
    if _gkey:
        try:
            configure(api_key=_gkey)
            _MODEL_AVAILABLE = True
        except Exception:
            _MODEL_AVAILABLE = False
except Exception:
    _MODEL_AVAILABLE = False

def _safe_json_load(text):
    try:
        return json.loads(text)
    except Exception:
        return None

def prioritize_tasks(tasks, model_override=False):
    """
    Prioritize tasks into {'high': [...], 'normal': [...]}.
    - tasks: list[str] or str (comma/newline separated)
    - model_override: force model use (if configured)
    """
    if isinstance(tasks, str):
        items = [t.strip() for line in tasks.splitlines() for t in line.split(",") for t in [line] if t.strip()]
        if len(items) == 1 and "," in tasks:
            items = [t.strip() for t in tasks.split(",") if t.strip()]
    else:
        items = list(tasks)

    items = [t for t in items if t]

    if not items:
        return {"high": [], "normal": []}

    if _MODEL_AVAILABLE and model_override:
        try:
            model = GenerativeModel("gemini-2.5-flash-lite")
            prompt = f"Prioritize these tasks for today and return valid JSON with keys 'high' and 'normal':\n{json.dumps(items)}"
            resp = model.generate_content(prompt)
            text = getattr(resp, "text", str(resp))
            parsed = _safe_json_load(text)
            if isinstance(parsed, dict) and ("high" in parsed or "normal" in parsed):
                # coerce lists
                parsed["high"] = parsed.get("high", []) or []
                parsed["normal"] = parsed.get("normal", []) or []
                return {"high": list(parsed["high"]), "normal": list(parsed["normal"])}
        except Exception:
            pass  

    high = items[:3]
    normal = items[3:]
    return {"high": high, "normal": normal}

# Quick examples:
print(prioritize_tasks("Write report, Email boss, Gym, Grocery shopping, Read"))
print(prioritize_tasks(["Write report","Email boss","Gym","Grocery"]))


# %% 
# Fix the syntax error in the DailyPlannerAgent class

# Define the fixed class with corrected syntax and indentation
class DailyPlannerAgent:
    """
    DailyPlannerAgent:
    Complete implementation with weather, email, storage, and web interface support
    """

    def __init__(self):
        self.session_memory = {
            "user_preferences": {
                "location": "San Francisco, CA",
                "dietary_restrictions": "none",
                "work_schedule": "9 AM - 5 PM",
                "clothing_style": "business casual",
                "email": None
            },
            "conversation_history": [],
            "daily_plans": [],         # list of dicts {date, plan}
            "current_weather": None,
            "prioritized_tasks": None
        }

        # Try to attach WeatherService instance if class exists
        WS = globals().get("WeatherService")
        if WS and callable(WS):
            try:
                self.weather_service = WS()
            except Exception:
                self.weather_service = None
        else:
            self.weather_service = None

        # Model availability flag
        self.model_available = bool(globals().get("_MODEL_AVAILABLE", False))

    # ---------- Preferences ----------
    def set_user_email(self, email_address):
        """Save a Gmail address to use for sending plans (basic validation)."""
        if isinstance(email_address, str) and "@" in email_address and "gmail.com" in email_address.lower():
            self.session_memory["user_preferences"]["email"] = email_address
            print("âœ… Email set to:", email_address)
            return True
        print("â�Œ Invalid email address. Use a Gmail address for the app-password flow.")
        return False

    def set_location(self, location):
        """Update user's preferred location for weather lookups."""
        if isinstance(location, str) and location.strip():
            self.session_memory["user_preferences"]["location"] = location.strip()
            print("âœ… Location set to:", location.strip())
            return True
        return False

    # ---------- Weather ----------
    def get_weather(self, location=None):
        """
        Get weather summary using OpenWeatherMap API
        """
        if location is None:
            location = self.session_memory['user_preferences']['location']
        
        print(f"ğŸŒ¤ï¸� Getting real-time weather for: {location}")
        
        try:
            # Get OpenWeatherMap API key from Kaggle Secrets
            from kaggle_secrets import UserSecretsClient
            user_secrets = UserSecretsClient()
            weather_api_key = user_secrets.get_secret("openweather-api-key")
            
            # Clean location string
            clean_location = location.split(',')[0].strip()
            
            # Get coordinates
            geocode_url = f"http://api.openweathermap.org/geo/1.0/direct?q={clean_location}&limit=1&appid={weather_api_key}"
            geocode_response = requests.get(geocode_url)
            geocode_data = geocode_response.json()
            
            if not geocode_data:
                print(f"âš ï¸� Could not find coordinates for {clean_location}, using fallback method")
                return self._get_weather_fallback(location)
            
            lat = geocode_data[0]['lat']
            lon = geocode_data[0]['lon']
            city_name = geocode_data[0]['name']
            country = geocode_data[0]['country']
            
            # Get weather data - FIXED: Removed extra spaces around =
            weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=imperial&appid={weather_api_key}"
            weather_response = requests.get(weather_url)
            weather_data = weather_response.json()
            
            if weather_response.status_code != 200:
                print(f"âš ï¸� Weather API returned error: {weather_data.get('message', 'Unknown error')}")
                return self._get_weather_fallback(location)
            
            # Extract weather info
            temp = weather_data['main']['temp']
            feels_like = weather_data['main']['feels_like']
            humidity = weather_data['main']['humidity']
            wind_speed = weather_data['wind']['speed']
            description = weather_data['weather'][0]['description']
            
            # Weather alerts
            alerts = []
            if 'alerts' in weather_data:
                for alert in weather_data['alerts']:
                    alerts.append(f"{alert['event']}: {alert['description']}")
            
            # Format
            weather_info = f"""
            **Current Weather for {city_name}, {country}**
            
            ğŸŒ¡ï¸� **Temperature:** {temp:.1f}Â°F (Feels like {feels_like:.1f}Â°F)
            ğŸŒ¤ï¸� **Conditions:** {description.title()}
            ğŸ’¨ **Wind Speed:** {wind_speed} mph
            ğŸ’§ **Humidity:** {humidity}%
            """
            
            if alerts:
                weather_info += "\nğŸš¨ **Weather Alerts:**\n" + "\n".join([f"- {alert}" for alert in alerts])
            
            # Save
            self.session_memory['current_weather'] = weather_info
            print("âœ… Real-time weather data retrieved successfully!")
            return weather_info
            
        except Exception as e:
            print(f"â�Œ Real weather API failed: {e}")
            print("ğŸ”„ Falling back to Gemini weather generation...")
            return self._get_weather_fallback(location)

    def _get_weather_fallback(self, location):
        """Fallback method using Gemini when API fails"""
        prompt = f"""
        You are a weather assistant. Get the current weather forecast for {location} for today.
        Return the weather information in a structured format with:
        - Temperature (in Fahrenheit)
        - Conditions (sunny, rainy, cloudy, etc.)
        - Wind speed
        - Any weather alerts
        
        Be concise and factual. Since this is a fallback, you can make reasonable assumptions.
        """
        
        try:
            from google.generativeai import GenerativeModel
            model = GenerativeModel('gemini-2.5-flash-lite')
            response = model.generate_content(prompt)
            weather_info = response.text
            self.session_memory['current_weather'] = weather_info
            return weather_info
        except Exception as e:
            print(f"â�Œ Fallback weather generation failed: {e}")
            return """
            **Weather Report (Fallback)**
            
            ğŸŒ¡ï¸� Temperature: 70Â°F
            ğŸŒ¤ï¸� Conditions: Partly Cloudy
            ğŸ’¨ Wind Speed: 10 mph
            ğŸ’§ Humidity: 65%
            ğŸš¨ Weather Alerts: None
            """

    # ---------- Calendar helper ----------
    def get_todays_events(self):
        """Return a small human-friendly schedule skeleton based on weekday/time."""
        now = datetime.now()
        hour = now.hour
        weekday = now.weekday()
        is_weekend = weekday >= 5

        if is_weekend:
            events = [
                "ğŸŒ… Morning routine (7:00 - 8:00)",
                "ğŸ�® Leisure / hobbies (10:00 - 12:00)", 
                "ğŸŒ³ Afternoon errands (15:00 - 17:00)"
            ]
        else:
            if 5 <= hour < 9:
                events = ["ğŸŒ… Morning routine (6:30 - 7:30)", "ğŸš— Commute (8:00 - 9:00)", "ğŸ’» Work start (9:00)"]
            elif 9 <= hour < 13:
                events = ["ğŸ“Š Morning work block", "ğŸ‘¥ Team meeting", "ğŸ�½ï¸� Lunch break"]
            elif 13 <= hour < 17:
                events = ["ğŸ’» Afternoon work block", "ğŸ�¯ Project time"]
            else:
                events = ["ğŸŒ™ Wrap up & evening routine"]

        text = f"**Today's Schedule ({now.strftime('%A, %B %d, %Y')}):**\n"
        for e in events:
            text += f"- {e}\n"
        return text

    # ---------- Task prioritization ----------
    def prioritize_tasks(self, tasks):
        """Task Prioritization Agent"""
        from google.generativeai import GenerativeModel
        model = GenerativeModel('gemini-2.5-flash-lite')
        
        prompt = f"""
        You are a productivity expert. Prioritize these tasks for today:
        {tasks}
        
        Return a ranked list in JSON format with:
        - task_name
        - priority (High/Medium/Low) 
        - estimated_time_minutes
        - reason
        
        Keep it concise and practical.
        """
        
        response = model.generate_content(prompt)
        
        try:
            prioritized_tasks = json.loads(response.text)
        except json.JSONDecodeError:
            prioritized_tasks = response.text
        
        self.session_memory['prioritized_tasks'] = prioritized_tasks
        return prioritized_tasks

    # ---------- Plan generation ----------
    def generate_daily_plan(self, weather_info, tasks_info):
        """Summary Agent: Create comprehensive daily plan"""
        from google.generativeai import GenerativeModel
        model = GenerativeModel('gemini-2.5-flash-lite')
        
        user_prefs = self.session_memory['user_preferences']
        calendar_events = self.get_todays_events()
        
        # Format tasks info for prompt
        if isinstance(tasks_info, list):
            tasks_formatted = json.dumps(tasks_info, indent=2)
        else:
            tasks_formatted = str(tasks_info)
        
        prompt = f"""
        You are a personal concierge creating a daily plan. Create a comprehensive daily plan for the user based on:
        
        **Weather Information:**
        {weather_info}
        
        **Today's Schedule:**
        {calendar_events}
        
        **Prioritized Tasks:**
        {tasks_formatted}
        
        **User Preferences:**
        - Location: {user_prefs['location']}
        - Dietary Restrictions: {user_prefs['dietary_restrictions']}
        - Work Schedule: {user_prefs['work_schedule']}
        - Clothing Style: {user_prefs['clothing_style']}
        
        **Create a plan that includes:**
        1. **Morning Routine** (wake up time, breakfast suggestion based on dietary needs)
        2. **Outfit Recommendation** (based on weather and clothing style)
        3. **Task Integration** (when to do each prioritized task within the schedule)
        4. **Meal Suggestions** (lunch/dinner ideas considering dietary restrictions)
        5. **Evening Wind-down** (relaxation suggestions)
        6. **Pro Tips** (weather-specific advice, productivity hacks)
        
        Format the response as a beautiful, readable daily plan with clear sections.
        Use markdown formatting for better readability.
        """
        
        response = model.generate_content(prompt)
        daily_plan = response.text
        
        # Store in session memory for future reference
        self.session_memory['daily_plans'].append({
            'date': datetime.now().strftime('%Y-%m-%d'),
            'plan': daily_plan
        })
        
        return daily_plan

    # ---------- Email Methods ----------
    def send_daily_plan_email(self, daily_plan, recipient_email=None):
        """
        Send the daily plan via email using Gmail SMTP
        """
        if recipient_email is None:
            recipient_email = self.session_memory['user_preferences'].get('email', None)
        
        if not recipient_email:
            print("ğŸ“§ No email address configured. Skipping email notification.")
            return False, "No email address configured"
        
        print(f"ğŸ“§ Sending daily plan to: {recipient_email}")
        
        try:
            # Get email credentials from Kaggle Secrets
            from kaggle_secrets import UserSecretsClient
            user_secrets = UserSecretsClient()
            app_password = user_secrets.get_secret("email-app-password")
            
            if not app_password:
                print("â�Œ No email app password found in Kaggle Secrets")
                return False, "No email app password found in Kaggle Secrets"
            
            # Extract Gmail address
            gmail_user = recipient_email.split('@')[0] + "@gmail.com"
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = gmail_user
            msg['To'] = recipient_email
            msg['Subject'] = f"ğŸ“… Your Daily Plan - {datetime.now().strftime('%B %d, %Y')}"
            
            # FIX: Move string replacements OUTSIDE the f-string to avoid syntax error
            daily_plan_html = daily_plan.replace('\n', '<br>').replace('**', '<strong>')
            
            # Create HTML version - CORRECTED SYNTAX
            html_content = f"""
            <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="max-width: 800px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 10px;">
                    <h1 style="color: #2c3e50; text-align: center; border-bottom: 2px solid #3498db; padding-bottom: 10px;">
                        ğŸ“… Your Daily Productivity Plan
                    </h1>
                    <p style="text-align: center; color: #7f8c8d; margin-bottom: 20px;">
                        Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
                    </p>
                    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 8px; margin: 15px 0;">
                        {daily_plan_html}
                    </div>
                    <div style="margin-top: 20px; padding-top: 15px; border-top: 1px solid #eee; text-align: center;">
                        <p style="color: #7f8c8d; font-size: 0.9em;">
                            Powered by AI Daily Planner Agent | Kaggle Capstone Project
                        </p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Create plain text version
            text_content = f"Your Daily Plan - {datetime.now().strftime('%B %d, %Y')}\n\n" + daily_plan
            
            # Attach both versions
            msg.attach(MIMEText(text_content, 'plain'))
            msg.attach(MIMEText(html_content, 'html'))
            
            # Send email
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as server:
                server.login(gmail_user, app_password)
                server.sendmail(gmail_user, recipient_email, msg.as_string())
            
            print("âœ… Email sent successfully!")
            return True, "Email sent successfully"
            
        except Exception as e:
            print(f"â�Œ Email sending failed: {e}")
            return False, f"Email sending failed: {e}"

    # ---------- Storage Methods ----------
    def save_daily_plan_locally(self, daily_plan, plan_date=None):
        """
        Save daily plan to local Kaggle storage (persists across sessions)
        """
        if plan_date is None:
            plan_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"ğŸ“� Saving daily plan to local storage for {plan_date}...")
        
        try:
            import os
            storage_dir = "/kaggle/working/daily_plans"
            if not os.path.exists(storage_dir):
                os.makedirs(storage_dir)
                print(f"âœ… Created storage directory: {storage_dir}")
            
            # Create filename
            filename = f"Daily_Plan_{plan_date}.md"
            file_path = os.path.join(storage_dir, filename)
            
            # Prepare content with markdown formatting
            content = f"""
            # ğŸ“… Daily Productivity Plan
            **Date:** {plan_date}  
            **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
            **Agent Version:** 1.0
            
            ---
            
            ## ğŸŒ¤ï¸� Weather Summary
            {self.session_memory.get('current_weather', 'Weather data not available')}
            
            ---
            
            ## ğŸ“‹ Today's Plan
            {daily_plan}
            
            ---
            
            ## ğŸ’¾ Storage Information
            - **Location:** `{file_path}`
            - **Storage Type:** Kaggle Local Storage
            - **Persistence:** Files persist across notebook sessions
            - **Access:** Download from Kaggle notebook output files
            
            ---
            
            *Powered by AI Daily Planner Agent - Kaggle Capstone Project*
            """
            
            # Save file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"âœ… âœ… SAVED SUCCESSFULLY: {file_path}")
            print(f"ğŸ’¡ You can download this file from the Kaggle notebook's output pane!")
            
            # Also save a simple text version
            text_path = os.path.join(storage_dir, f"Daily_Plan_{plan_date}.txt")
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(f"Daily Plan for {plan_date}\n" + "="*50 + "\n\n" + daily_plan)
            
            return True
            
        except Exception as e:
            print(f"â�Œ â�Œ Local save failed: {e}")
            print("âš ï¸� Could not save daily plan. Displaying plan in notebook instead.")
            return False

    def get_saved_plans(self):
        """List all saved daily plans"""
        try:
            import os
            storage_dir = "/kaggle/working/daily_plans"
            
            if not os.path.exists(storage_dir):
                return "**No saved plans found.**\nPlans will be saved to `/kaggle/working/daily_plans/` after generation."
            
            files = [f for f in os.listdir(storage_dir) if f.startswith('Daily_Plan_')]
            
            if not files:
                return "**No saved plans found.**\nGenerate a daily plan to create your first saved file."
            
            # Sort by date (newest first)
            files.sort(reverse=True)
            
            result = f"**ğŸ“� Found {len(files)} saved daily plans:**\n\n"
            for file in files:
                result += f"- ğŸ“„ `{file}`\n"
            
            result += f"\nğŸ’¡ **Access your plans:**\n"
            result += f"- View in Kaggle notebook file browser\n"
            result += f"- Download from output files\n"
            result += f"- Access via `/kaggle/working/daily_plans/{files[0]}`"
            
            return result
            
        except Exception as e:
            return f"â�Œ Error listing saved plans: {e}"

    # ---------- Main Orchestrator ----------
    def run_daily_planner(self, custom_tasks=None, custom_location=None):
        """Main orchestrator: Run the sequential agent system"""
        
        print("\n" + "="*50)
        print("ğŸ”„ STARTING DAILY PLANNER")
        print("="*50)
        
        # Step 1: Get weather
        print("\nğŸŒ¤ï¸� Weather Agent: Getting current weather...")
        weather_info = self.get_weather(custom_location)
        display(Markdown(f"**Weather Report:**\n{weather_info}"))
        print("-" * 30)
        
        # Step 2: Get tasks
        print("\nğŸ“‹ Task Prioritization Agent: Ranking your tasks...")
        if custom_tasks is None:
            default_tasks = [
                "Prepare presentation for team meeting",
                "Respond to important emails", 
                "Go to gym",
                "Grocery shopping",
                "Read 30 minutes",
                "Call mom",
                "Work on project deadline"
            ]
            tasks_input = ", ".join(default_tasks)
        else:
            tasks_input = ", ".join(custom_tasks)
        
        tasks_info = self.prioritize_tasks(tasks_input)
        
        if isinstance(tasks_info, list):
            tasks_display = json.dumps(tasks_info, indent=2)
        else:
            tasks_display = tasks_info
        
        display(Markdown(f"**Prioritized Tasks:**\n```json\n{tasks_display}\n```"))
        print("-" * 30)
        
        # Step 3: Generate plan
        print("\nğŸ�¯ Summary Agent: Creating your daily plan...")
        daily_plan = self.generate_daily_plan(weather_info, tasks_info)
        
        # Step 4: Save to storage
        print("\nğŸ“� Saving to local storage...")
        self.save_daily_plan_locally(daily_plan)
        
        # Step 5: Send email if configured
        email = self.session_memory['user_preferences'].get('email')
        if email:
            print("\nğŸ“§ Sending via email...")
            self.send_daily_plan_email(daily_plan, email)
        else:
            print("\nğŸ’¡ Tip: Set email with agent.set_user_email('you@gmail.com') to receive daily plans!")
        
        print("\n" + "="*50)
        print("âœ… DAILY PLAN COMPLETE!")
        print("="*50)
        return daily_plan

# Recreate the agent with the fixed class
agent = DailyPlannerAgent()

print("âœ… âœ… DailyPlannerAgent successfully defined!")



# %% 
# Test your fixed agent

# Set your email
agent.set_user_email("monishanbalagan7777@gmail.com")

# Generate and send a plan
print("\n Generating and sending daily plan...")
plan = agent.run_daily_planner()

# Display the plan
display(Markdown("#  Your Daily Plan"))
display(Markdown(plan))

print("\n All methods working correctly!")


# %% 
# Create a fresh agent instance
agent = DailyPlannerAgent()

# Set your email so the plan can be mailed
agent.set_user_email("monishanbalagan7777@gmail.com")

# Generate a daily plan (corrected method call)
print("\n Generating daily plan...")
daily_plan = agent.run_daily_planner()  # No try_send_email parameter needed

print("\n" + "="*60)
print(" PLAN GENERATED SUCCESSFULLY")
print("="*60)

# Display plan preview
print("\n=== PLAN PREVIEW (first 1000 characters) ===\n")
print(daily_plan[:1000])
print("\n" + "="*60)

# Optional: Send email with the generated plan
print("\n Sending plan via email...")
email_success, email_msg = agent.send_daily_plan_email(daily_plan, agent.session_memory['user_preferences']['email'])
print(f" Email Status: {' Success' if email_success else ' Failed'}")
print(f" Message: {email_msg}")

print(" TEST COMPLETED SUCCESSFULLY!")


# Import required libraries
from IPython.display import display, HTML, Markdown, clear_output
import ipywidgets as widgets
from datetime import datetime

# Create a new agent instance for UI use
ui_agent = DailyPlannerAgent()

header = widgets.HTML("<h2 style='color:#2c3e50; text-align:center;'>Daily Planner Web Interface</h2>")

tasks_label = widgets.HTML("<b>Tasks (one per line):</b>")
tasks_input = widgets.Textarea(
    value="Prepare presentation\nRespond to emails\nGo to gym\nRead 30 minutes",
    layout=widgets.Layout(width="100%", height="100px")
)

location_input = widgets.Text(value="San Francisco, CA", description="Location:")

email_toggle = widgets.Checkbox(value=False, description="Send to email")
email_input = widgets.Text(value="your_email@gmail.com", description="Email:")

prefs_label = widgets.HTML("<b>Preferences:</b>")

diet_dropdown = widgets.Dropdown(
    options=["none", "vegetarian", "vegan", "gluten-free"],
    value="none",
    description="Diet:",
    layout=widgets.Layout(width="300px")
)

style_dropdown = widgets.Dropdown(
    options=["business casual", "casual", "formal", "athletic"],
    value="business casual",
    description="Clothing:",
    layout=widgets.Layout(width="300px")
)

work_input = widgets.Text(value="9 AM - 5 PM", description="Work Hours:")

run_btn = widgets.Button(description="Generate Plan", button_style="success")
output_box = widgets.Output()


def on_run_clicked(b):
    with output_box:
        clear_output()
        
        ui_agent.set_location(location_input.value.strip())
        ui_agent.session_memory["user_preferences"]["dietary_restrictions"] = diet_dropdown.value
        ui_agent.session_memory["user_preferences"]["clothing_style"] = style_dropdown.value
        ui_agent.session_memory["user_preferences"]["work_schedule"] = work_input.value.strip()
        
        # Set email if toggle is enabled and valid Gmail address
        if email_toggle.value and "@gmail.com" in email_input.value:
            ui_agent.set_user_email(email_input.value)
        
        tasks = [t.strip() for t in tasks_input.value.split("\n") if t.strip()]
        
        # Generate plan (email will be sent automatically if email is configured)
        daily_plan = ui_agent.run_daily_planner(custom_tasks=tasks)
        
        display(Markdown("### Daily Plan"))
        display(Markdown(daily_plan))
        
        # Show email status based on whether email was configured
        if email_toggle.value and "@gmail.com" in email_input.value:
            display(HTML("<p style='color:green'>Email notification enabled</p>"))
        elif email_toggle.value:
            display(HTML("<p style='color:red'>Email not sent - invalid Gmail address</p>"))

run_btn.on_click(on_run_clicked)

ui = widgets.VBox([
    header,
    tasks_label, tasks_input,
    location_input,
    prefs_label, diet_dropdown, style_dropdown, work_input,
    email_toggle, email_input,
    run_btn,
    output_box
])

display(ui)

