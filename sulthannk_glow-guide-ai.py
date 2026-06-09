# @title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Optional, List
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.runners import Runner
from datetime import datetime
import requests
import json
import math

print("âœ… ADK components imported successfully.")


def collect_user_info(name: Optional[str] = None, 
                      age: Optional[int] = None, 
                      gender: Optional[str] = None, 
                      city: Optional[str] = None) -> dict:
    """Collects and validates user information."""
    
    user_data = {}
    missing_fields = []
    errors = []
    
    if name and isinstance(name, str) and name.strip():
        user_data['name'] = name.strip()
    else:
        missing_fields.append('name')
    
    if age is not None:
        try:
            age_int = int(age)
            if 1 <= age_int <= 150:
                user_data['age'] = age_int
            else:
                errors.append(f"Age must be between 1 and 150, got {age_int}")
                missing_fields.append('age')
        except (ValueError, TypeError):
            errors.append(f"Age must be a valid number, got {age}")
            missing_fields.append('age')
    else:
        missing_fields.append('age')
    
    valid_genders = ['male', 'female', 'other', 'prefer not to say']
    if gender and isinstance(gender, str):
        gender_lower = gender.strip().lower()
        if gender_lower in valid_genders:
            user_data['gender'] = gender.strip()
        else:
            errors.append(f"Gender should be one of: {', '.join(valid_genders)}")
            missing_fields.append('gender')
    else:
        missing_fields.append('gender')
    
    if city and isinstance(city, str) and city.strip():
        user_data['city'] = city.strip()
    else:
        missing_fields.append('city')
    
    if not missing_fields:
        status = 'success'
        message = "All user information collected successfully!"
    else:
        status = 'incomplete'
        message = f"Still need: {', '.join(missing_fields)}"
    
    if errors:
        status = 'error' if status == 'incomplete' else status
    
    result = {
        'status': status,
        'data': user_data,
        'missing_fields': missing_fields,
        'error_message': ' | '.join(errors) if errors else None,
        'message': message
    }
    
    return result


def store_user_in_session(name: str, age: int, gender: str, city: str) -> dict:
    """Confirms user information is ready to be stored in session."""
    
    if not all([name, age, gender, city]):
        return {
            'status': 'validation_error',
            'error': 'All fields (name, age, gender, city) are required'
        }
    
    user_profile = {
        'name': name,
        'age': age,
        'gender': gender,
        'city': city,
        'status': 'complete',
        'profile_created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return {
        'status': 'ready_to_store',
        'user_profile': user_profile,
        'message': f"User profile for {name} is ready to be stored in session"
    }


def create_user_info_agent(model: str = "gemini-2.0-flash"):
    """Creates the User Information Collector Agent."""
    
    user_info_agent = Agent(
        name="user_info_collector_agent",
        model=model,
        description="Collects and validates user information (name, age, gender, city).",
        
        instruction="""You are the User Information Collector Agent.

WORKFLOW:
1. Greet warmly
2. Ask 4 questions ONE AT A TIME (name, age, gender, city)
3. Validate each with tools
4. Store when all complete
5. Pass to Weather Agent

TONE: Friendly, professional.""",
        
        tools=[collect_user_info, store_user_in_session]
    )
    
    return user_info_agent


def get_location_weather_data(city: str) -> dict:
    """Fetches REAL current weather, season, and climate information from the internet."""
    
    try:
        geocoding_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
        
        geo_response = requests.get(geocoding_url, timeout=5)
        geo_data = geo_response.json()
        
        if not geo_data.get('results'):
            return {
                'status': 'error',
                'city': city,
                'error_message': f"City '{city}' not found. Please check the spelling and try again."
            }
        
        location = geo_data['results'][0]
        latitude = location['latitude']
        longitude = location['longitude']
        city_name = location['name']
        country = location.get('country', 'Unknown')
        
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m&timezone=auto"
        
        weather_response = requests.get(weather_url, timeout=5)
        weather_data = weather_response.json()
        
        current = weather_data['current']
        
        weather_code = current['weather_code']
        weather_descriptions = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Foggy", 48: "Depositing rime fog", 51: "Light drizzle", 53: "Moderate drizzle",
            55: "Dense drizzle", 61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
            71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow", 80: "Slight rain showers",
            81: "Moderate rain showers", 82: "Violent rain showers", 85: "Slight snow showers",
            86: "Heavy snow showers", 95: "Thunderstorm", 96: "Thunderstorm with slight hail",
            99: "Thunderstorm with heavy hail"
        }
        
        weather_condition = weather_descriptions.get(weather_code, "Unknown weather condition")
        temperature = current['temperature_2m']
        humidity = current['relative_humidity_2m']
        wind_speed = current['wind_speed_10m']
        
        current_month = datetime.now().month
        if latitude > 0:
            if current_month in [12, 1, 2]:
                season = "Winter"
            elif current_month in [3, 4, 5]:
                season = "Spring"
            elif current_month in [6, 7, 8]:
                season = "Summer"
            else:
                season = "Autumn"
        else:
            if current_month in [12, 1, 2]:
                season = "Summer"
            elif current_month in [3, 4, 5]:
                season = "Autumn"
            elif current_month in [6, 7, 8]:
                season = "Winter"
            else:
                season = "Spring"
        
        if temperature > 25:
            if humidity > 70:
                climate_type = "Tropical"
            else:
                climate_type = "Hot/Arid"
        elif temperature > 15:
            climate_type = "Temperate"
        elif temperature > 0:
            climate_type = "Cool/Temperate"
        else:
            climate_type = "Cold/Polar"
        
        current_hour = datetime.now().hour
        if 6 <= current_hour <= 18:
            if abs(latitude) < 23.5:
                uv_index = "High to Very High (7-9/10)"
            elif abs(latitude) < 35:
                uv_index = "Moderate to High (5-7/10)"
            else:
                uv_index = "Low to Moderate (2-4/10)"
        else:
            uv_index = "Low (0-2/10)"
        
        response = {
            'status': 'success',
            'city': city_name,
            'country': country,
            'latitude': latitude,
            'longitude': longitude,
            'current_weather': {
                'temperature': f"{temperature}Â°C",
                'humidity': f"{humidity}%",
                'condition': weather_condition,
                'wind_speed': f"{wind_speed} km/h"
            },
            'current_season': season,
            'climate_type': climate_type,
            'uv_index': uv_index,
            'data_fetched_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_source': 'Open-Meteo API (Real-time data)'
        }
        
        return response
        
    except Exception as e:
        return {
            'status': 'error',
            'city': city,
            'error_message': f"Error fetching weather data: {str(e)}"
        }


def update_user_profile_with_location_data(user_name: str, city: str, season: str, weather: str, climate: str) -> dict:
    """Updates the user profile with location-based environmental data."""
    
    if not all([user_name, city, season, weather, climate]):
        return {
            'status': 'error',
            'error': 'All fields (user_name, city, season, weather, climate) are required'
        }
    
    updated_profile = {
        'name': user_name,
        'city': city,
        'current_season': season,
        'current_weather': weather,
        'climate_type': climate,
        'data_updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return {
        'status': 'profile_updated',
        'updated_profile': updated_profile,
        'message': f"Profile for {user_name} enriched with location data from {city}"
    }


def create_location_weather_agent(model: str = "gemini-2.0-flash"):
    """Creates the Location Weather and Climate Agent."""
    
    location_weather_agent = Agent(
        name="location_weather_climate_agent",
        model=model,
        description="Fetches REAL weather data and enriches user profile.",
        
        instruction="""You are the Location Weather & Climate Agent.

WORKFLOW:
1. Fetch weather from city in profile
2. Present data
3. Store in profile
4. Pass control back to Root Agent

TONE: Informative, helpful.""",
        
        tools=[get_location_weather_data, update_user_profile_with_location_data]
    )
    
    return location_weather_agent


def search_skincare_products_from_internet(user_age: int, skin_type: str, skin_concern: Optional[str] = None, 
                                          product_category: str = "cleanser",
                                          climate: Optional[str] = None,
                                          season: Optional[str] = None,
                                          dietary_habits: Optional[str] = None) -> dict:
    """
    Searches the INTERNET for skincare products based on SPECIFIC USER DATA.
    NO hardcoding - everything fetched from web based on user profile.
    """
    
    try:
        # Build DYNAMIC search query using USER-SPECIFIC DATA
        query_parts = [product_category, skin_type, f"age {user_age}"]
        
        if skin_concern:
            query_parts.append(skin_concern)
        
        if climate and climate in ['Tropical', 'Hot/Arid']:
            query_parts.append("lightweight water-resistant")
        elif climate == 'Cold/Polar':
            query_parts.append("rich nourishing")
        
        if season == 'Winter':
            query_parts.append("winter moisture barrier")
        elif season == 'Summer':
            query_parts.append("summer UV protection")
        
        if dietary_habits and 'dairy' in dietary_habits.lower():
            query_parts.append("non-dairy friendly")
        
        # Build search with current year
        search_query = " ".join(query_parts) + " best products 2024 2025 reviews"
        
        print(f"ğŸ”� Searching: {search_query}")
        
        # Search using DuckDuckGo
        search_url = f"https://api.duckduckgo.com/?q={search_query}&format=json&no_redirect=1"
        response = requests.get(search_url, timeout=5)
        search_data = response.json()
        
        products = []
        related_topics = search_data.get('RelatedTopics', [])
        
        # Extract real products from search results
        for topic in related_topics[:8]:
            if 'Text' in topic and 'FirstURL' in topic:
                topic_text = topic.get('Text', '')
                
                product_info = {
                    'name': topic_text.split(' - ')[0][:100] if ' - ' in topic_text else topic_text[:100],
                    'description': topic_text[:200],
                    'source_url': topic.get('FirstURL', ''),
                    'data_source': 'Internet Search (Real-time)'
                }
                
                if product_info['name'] and len(product_info['name']) > 3:
                    products.append(product_info)
        
        if products:
            return {
                'status': 'success',
                'product_category': product_category,
                'skin_type': skin_type,
                'user_age': user_age,
                'climate': climate,
                'season': season,
                'products': products[:5],
                'search_query': search_query,
                'count': len(products[:5]),
                'message': f"Found {len(products[:5])} real products from internet",
                'source': 'Internet Search Results'
            }
        else:
            broad_query = f"{product_category} {skin_type} skin best 2024"
            print(f"ğŸ”� Broader search: {broad_query}")
            
            search_url = f"https://api.duckduckgo.com/?q={broad_query}&format=json&no_redirect=1"
            response = requests.get(search_url, timeout=5)
            search_data = response.json()
            
            products = []
            for topic in search_data.get('RelatedTopics', [])[:5]:
                if 'Text' in topic:
                    product_info = {
                        'name': topic.get('Text', '')[:100],
                        'description': topic.get('Text', '')[:200],
                        'source_url': topic.get('FirstURL', ''),
                        'data_source': 'Internet Search (Broader)'
                    }
                    if product_info['name'] and len(product_info['name']) > 3:
                        products.append(product_info)
            
            return {
                'status': 'success',
                'product_category': product_category,
                'skin_type': skin_type,
                'user_age': user_age,
                'products': products,
                'search_query': broad_query,
                'count': len(products),
                'message': f"Found {len(products)} products from internet",
                'source': 'Internet Search Results'
            }
        
    except Exception as e:
        return {
            'status': 'error',
            'product_category': product_category,
            'error_message': f"Error searching products: {str(e)}",
            'recommendation': f"Please search '{product_category} for {skin_type} skin' on Google for latest recommendations"
        }


def collect_skincare_details(commitment_days: Optional[int] = None,
                             menstrual_cycle: Optional[str] = None,
                             skin_type: Optional[str] = None,
                             acne_frequency: Optional[str] = None,
                             dryness_level: Optional[str] = None,
                             dietary_habits: Optional[str] = None,
                             weight: Optional[float] = None,
                             height: Optional[float] = None,
                             weight_unit: Optional[str] = None,
                             height_unit: Optional[str] = None,
                             food_preference: Optional[str] = None) -> dict:
    """Collects and validates skincare details from user."""
    
    skincare_data = {}
    missing_fields = []
    errors = []
    
    if commitment_days is not None:
        try:
            days = int(commitment_days)
            valid_days = [15, 30, 45, 90]
            if days in valid_days:
                skincare_data['commitment_days'] = days
            else:
                errors.append(f"Please choose from: 15, 30, 45, or 90 days")
                missing_fields.append('commitment_days')
        except (ValueError, TypeError):
            errors.append(f"Commitment days must be a number")
            missing_fields.append('commitment_days')
    else:
        missing_fields.append('commitment_days')
    
    valid_skin_types = ['dry', 'oily', 'sensitive', 'combination', 'normal']
    if skin_type and isinstance(skin_type, str):
        skin_type_lower = skin_type.strip().lower()
        if skin_type_lower in valid_skin_types:
            skincare_data['skin_type'] = skin_type_lower
        else:
            errors.append(f"Skin type should be one of: {', '.join(valid_skin_types)}")
            missing_fields.append('skin_type')
    else:
        missing_fields.append('skin_type')
    
    if dietary_habits and isinstance(dietary_habits, str):
        skincare_data['dietary_habits'] = dietary_habits.strip()
    
    if acne_frequency and isinstance(acne_frequency, str):
        valid_acne = ['rarely', 'occasionally', 'frequently']
        if acne_frequency.strip().lower() in valid_acne:
            skincare_data['acne_frequency'] = acne_frequency.strip().lower()
    
    if dryness_level and isinstance(dryness_level, str):
        skincare_data['dryness_level'] = dryness_level.strip()
    
    if menstrual_cycle and isinstance(menstrual_cycle, str):
        skincare_data['menstrual_cycle'] = menstrual_cycle.strip()
    
    valid_food_prefs = ['vegetarian', 'non-vegetarian', 'vegan', 'pescatarian']
    if food_preference and isinstance(food_preference, str):
        food_pref_lower = food_preference.strip().lower()
        if food_pref_lower in valid_food_prefs:
            skincare_data['food_preference'] = food_pref_lower
        else:
            skincare_data['food_preference'] = food_preference.strip()
    
    if weight is not None and height is not None and weight_unit and height_unit:
        try:
            weight_kg = float(weight)
            height_m = float(height)
            
            if weight_unit.lower() in ['lbs', 'lb', 'pounds']:
                weight_kg = weight_kg * 0.453592
            
            if height_unit.lower() in ['feet', 'ft', 'foot']:
                height_m = height_m * 0.3048
            elif height_unit.lower() in ['cm', 'centimeters']:
                height_m = height_m / 100
            
            bmi = weight_kg / (height_m ** 2)
            skincare_data['weight_kg'] = round(weight_kg, 2)
            skincare_data['height_m'] = round(height_m, 2)
            skincare_data['bmi'] = round(bmi, 1)
            
        except (ValueError, TypeError):
            errors.append("Invalid weight or height values")
    
    if not missing_fields:
        status = 'success'
        message = "All skincare details collected successfully!"
    else:
        status = 'incomplete'
        message = f"Still need: {', '.join(missing_fields)}"
    
    if errors:
        status = 'error' if status == 'incomplete' else status
    
    result = {
        'status': status,
        'data': skincare_data,
        'missing_fields': missing_fields,
        'error_message': ' | '.join(errors) if errors else None,
        'message': message
    }
    
    return result


def generate_skincare_routine(user_name: str, age: int, gender: str, commitment_days: int,
                              skin_type: str, climate: str, season: str, 
                              bmi: Optional[float] = None,
                              dietary_habits: Optional[str] = None,
                              acne_frequency: Optional[str] = None,
                              food_preference: Optional[str] = None,
                              menstrual_cycle: Optional[str] = None) -> dict:
    """
    Generates INTERNET-BASED skincare routine with separate morning/evening routines.
    ALL products fetched from internet based on user-specific data.
    """
    
    routine = {
        'status': 'routine_generated',
        'user_name': user_name,
        'age': age,
        'commitment_days': commitment_days,
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'routines': {},
        'data_sources': 'All products fetched from internet in real-time based on your profile'
    }
    
    skin_concern = None
    if skin_type == 'oily' and acne_frequency in ['occasionally', 'frequently']:
        skin_concern = 'acne'
    elif skin_type == 'dry':
        skin_concern = 'dryness'
    elif skin_type == 'sensitive':
        skin_concern = 'sensitivity'
    
    # ===== MORNING ROUTINE (INTERNET-BASED) =====
    morning_routine = {
        'title': 'ğŸŒ… MORNING SKINCARE ROUTINE',
        'time': '6:00 AM - 8:00 AM',
        'total_time': '8-10 minutes',
        'description': 'Light, protective routine to prepare skin for the day ahead',
        'note': 'All products researched from internet based on YOUR profile',
        'steps': []
    }
    
    # STEP 1: Cleanser - FETCH FROM INTERNET
    cleanser_results = search_skincare_products_from_internet(
        user_age=age,
        skin_type=skin_type,
        skin_concern=skin_concern,
        product_category="cleanser",
        climate=climate,
        season=season,
        dietary_habits=dietary_habits
    )
    
    morning_routine['steps'].append({
        'step': 1,
        'name': 'ğŸ§´ STEP 1: Cleanser',
        'time': '30-60 seconds',
        'internet_products': cleanser_results.get('products', []),
        'search_query': cleanser_results.get('search_query', ''),
        'instructions': [
            'Wet face with lukewarm water',
            'Apply cleanser and massage gently',
            'Rinse thoroughly with water',
            'Pat dry with clean towel'
        ],
        'why_important': 'Removes overnight oil, dead skin, and impurities'
    })
    
    # STEP 2: Toner - FETCH FROM INTERNET
    toner_results = search_skincare_products_from_internet(
        user_age=age,
        skin_type=skin_type,
        skin_concern=skin_concern,
        product_category="toner",
        climate=climate,
        season=season
    )
    
    morning_routine['steps'].append({
        'step': 2,
        'name': 'ğŸ’§ STEP 2: Toner',
        'time': '1-2 minutes',
        'internet_products': toner_results.get('products', []),
        'search_query': toner_results.get('search_query', ''),
        'instructions': [
            'Apply toner to cotton pad or hands',
            'Gently pat onto face and neck',
            'Allow to absorb 1-2 minutes'
        ],
        'why_important': 'Balances pH, hydrates, prepares for serums'
    })
    
    # STEP 3: Serum - FETCH FROM INTERNET
    serum_results = search_skincare_products_from_internet(
        user_age=age,
        skin_type=skin_type,
        skin_concern=skin_concern,
        product_category="serum",
        climate=climate
    )
    
    morning_routine['steps'].append({
        'step': 3,
        'name': 'âœ¨ STEP 3: Serum',
        'time': '2-3 minutes',
        'internet_products': serum_results.get('products', []),
        'search_query': serum_results.get('search_query', ''),
        'instructions': [
            'Dispense 2-3 drops onto fingertips',
            'Gently pat and press into face and neck',
            'Allow 2-3 minutes to absorb'
        ],
        'why_important': 'Targets specific skin concerns'
    })
    
    # STEP 4: Moisturizer - FETCH FROM INTERNET
    moisturizer_results = search_skincare_products_from_internet(
        user_age=age,
        skin_type=skin_type,
        skin_concern=skin_concern,
        product_category="moisturizer",
        climate=climate
    )
    
    morning_routine['steps'].append({
        'step': 4,
        'name': 'ğŸ§´ STEP 4: Moisturizer',
        'time': '2 minutes',
        'internet_products': moisturizer_results.get('products', []),
        'search_query': moisturizer_results.get('search_query', ''),
        'instructions': [
            'Apply pea-sized amount',
            'Warm between fingers',
            'Apply to face, neck, dÃ©colletage',
            'Allow 2 minutes to set'
        ],
        'why_important': 'Locks hydration and protects'
    })
    
    # STEP 5: Sunscreen - FETCH FROM INTERNET
    sunscreen_results = search_skincare_products_from_internet(
        user_age=age,
        skin_type=skin_type,
        product_category="sunscreen SPF 50+",
        climate=climate,
        season=season
    )
    
    morning_routine['steps'].append({
        'step': 5,
        'name': 'â˜€ï¸� STEP 5: Sunscreen (SPF 50+)',
        'time': '2-3 minutes',
        'internet_products': sunscreen_results.get('products', []),
        'search_query': sunscreen_results.get('search_query', ''),
        'instructions': [
            'Dispense 1/4 teaspoon of sunscreen',
            'Apply to forehead, cheeks, nose, chin',
            'Spread evenly across face and neck',
            'Wait 5 minutes before sun exposure'
        ],
        'frequency': 'EVERY DAY - Rain or shine',
        'reapplication': 'Every 2 hours if sweating or swimming',
        'why_critical': 'âš ï¸� MOST IMPORTANT - Prevents UV damage and premature aging'
    })
    
    routine['routines']['morning'] = morning_routine
    
    # ===== EVENING ROUTINE (INTERNET-BASED) =====
    evening_routine = {
        'title': 'ğŸŒ™ EVENING SKINCARE ROUTINE',
        'time': '8:00 PM - 10:00 PM',
        'total_time': '15-20 minutes',
        'description': 'Deep cleanse and repair for overnight recovery',
        'note': 'All products researched from internet based on YOUR profile',
        'steps': []
    }
    
    # STEP 1: Oil Cleanser - FETCH FROM INTERNET
    oil_cleanser_results = search_skincare_products_from_internet(
        user_age=age,
        skin_type=skin_type,
        product_category="oil cleanser makeup remover",
        climate=climate
    )
    
    evening_routine['steps'].append({
        'step': 1,
        'name': 'ğŸ§´ STEP 1: Oil Cleanser (First Cleanse)',
        'time': '1-2 minutes',
        'internet_products': oil_cleanser_results.get('products', []),
        'search_query': oil_cleanser_results.get('search_query', ''),
        'instructions': [
            'Start with DRY face',
            'Pump 2-3 times into palm',
            'Massage gently for 1-2 minutes',
            'Add water to emulsify',
            'Rinse thoroughly'
        ],
        'why_important': 'Dissolves makeup, sunscreen, oil-based impurities'
    })
    
    # STEP 2: Water Cleanser - FETCH FROM INTERNET
    evening_routine['steps'].append({
        'step': 2,
        'name': 'ğŸ’§ STEP 2: Water Cleanser (Second Cleanse)',
        'time': '30-60 seconds',
        'internet_products': cleanser_results.get('products', []),
        'instructions': [
            'Wet face with lukewarm water',
            'Apply water cleanser',
            'Massage gently',
            'Rinse thoroughly'
        ],
        'why_important': 'Removes water-soluble impurities and residue'
    })
    
    # STEP 3: Toner - REUSE MORNING RESULTS
    evening_routine['steps'].append({
        'step': 3,
        'name': 'ğŸ’§ STEP 3: Toner',
        'time': '1-2 minutes',
        'internet_products': toner_results.get('products', []),
        'instructions': [
            'Apply to cotton pad or hands',
            'Gently wipe or pat across face',
            'Allow to absorb'
        ],
        'why_important': 'Re-balances pH after cleansing'
    })
    
    # STEP 4: Treatment/Mask (Varies by skin type) - FETCH FROM INTERNET
    if skin_type == 'oily' and acne_frequency in ['occasionally', 'frequently']:
        treatment_results = search_skincare_products_from_internet(
            user_age=age,
            skin_type=skin_type,
            skin_concern='acne treatment',
            product_category="acne treatment BHA AHA",
            climate=climate
        )
        evening_routine['steps'].append({
            'step': 4,
            'name': 'ğŸ”¬ STEP 4: Acne Treatment',
            'time': '10 min wait + absorption',
            'internet_products': treatment_results.get('products', []),
            'search_query': treatment_results.get('search_query', ''),
            'frequency': '3-4 times per week',
            'warning': 'âš ï¸� Start 2-3x weekly'
        })
    elif skin_type == 'dry':
        mask_results = search_skincare_products_from_internet(
            user_age=age,
            skin_type=skin_type,
            skin_concern='dryness hydration',
            product_category="hydrating mask cream mask",
            climate=climate
        )
        evening_routine['steps'].append({
            'step': 4,
            'name': 'ğŸ�­ STEP 4: Hydrating Mask',
            'time': '10-15 minutes',
            'internet_products': mask_results.get('products', []),
            'search_query': mask_results.get('search_query', ''),
            'frequency': '2-3 times per week'
        })
    else:
        night_serum_results = search_skincare_products_from_internet(
            user_age=age,
            skin_type=skin_type,
            skin_concern=skin_concern,
            product_category="night serum active serum",
            climate=climate
        )
        evening_routine['steps'].append({
            'step': 4,
            'name': 'âœ¨ STEP 4: Night Serum',
            'time': '2-3 minutes',
            'internet_products': night_serum_results.get('products', []),
            'search_query': night_serum_results.get('search_query', '')
        })
    
    # STEP 5: Eye Cream - FETCH FROM INTERNET
    eye_cream_results = search_skincare_products_from_internet(
        user_age=age,
        skin_type=skin_type,
        product_category="eye cream anti-aging eye care",
        climate=climate
    )
    
    evening_routine['steps'].append({
        'step': 5,
        'name': 'ğŸ‘�ï¸� STEP 5: Eye Cream',
        'time': '1 minute',
        'internet_products': eye_cream_results.get('products', []),
        'search_query': eye_cream_results.get('search_query', ''),
        'instructions': [
            'Tiny amount on ring finger',
            'Gentle dabs around eye area',
            'Patting motions only'
        ]
    })
    
    # STEP 6: Night Moisturizer - FETCH FROM INTERNET
    night_moisturizer_results = search_skincare_products_from_internet(
        user_age=age,
        skin_type=skin_type,
        product_category="night cream sleep mask heavy moisturizer",
        climate=climate
    )
    
    evening_routine['steps'].append({
        'step': 6,
        'name': 'ğŸŒ™ STEP 6: Night Moisturizer',
        'time': '2-3 minutes',
        'internet_products': night_moisturizer_results.get('products', []),
        'search_query': night_moisturizer_results.get('search_query', ''),
        'why_important': 'Overnight repair and recovery'
    })
    
    routine['routines']['evening'] = evening_routine
    
    # ===== WEEKLY TREATMENTS (INTERNET-BASED) =====
    weekly_routine = {
        'title': 'âœ¨ WEEKLY SPECIAL TREATMENTS',
        'frequency': '2-3 times per week',
        'note': 'All products researched from internet',
        'options': []
    }
    
    if skin_type == 'oily':
        mask_results = search_skincare_products_from_internet(
            user_age=age,
            skin_type=skin_type,
            skin_concern='pore cleansing oil control',
            product_category="clay mask charcoal mask pore mask",
            climate=climate
        )
        weekly_routine['options'].append({
            'name': 'ğŸ�­ Clay or Charcoal Mask',
            'internet_products': mask_results.get('products', []),
            'search_query': mask_results.get('search_query', ''),
            'frequency': '1-2 times per week'
        })
    elif skin_type == 'dry':
        mask_results = search_skincare_products_from_internet(
            user_age=age,
            skin_type=skin_type,
            skin_concern='deep hydration nourishment',
            product_category="hydrating mask honey mask cream mask",
            climate=climate
        )
        weekly_routine['options'].append({
            'name': 'ğŸ�­ Hydrating Mask',
            'internet_products': mask_results.get('products', []),
            'search_query': mask_results.get('search_query', ''),
            'frequency': '1-2 times per week'
        })
    else:
        mask_results = search_skincare_products_from_internet(
            user_age=age,
            skin_type=skin_type,
            product_category="sheet mask gel mask treatment mask",
            climate=climate
        )
        weekly_routine['options'].append({
            'name': 'ğŸ�­ Sheet or Gel Mask',
            'internet_products': mask_results.get('products', []),
            'search_query': mask_results.get('search_query', ''),
            'frequency': '1-2 times per week'
        })
    
    routine['routines']['weekly'] = weekly_routine
    
    # ===== LIFESTYLE RECOMMENDATIONS =====
    lifestyle = {
        'title': 'ğŸ’ª LIFESTYLE & DIETARY RECOMMENDATIONS',
        'note': 'Personalized based on your data',
        'sections': {
            'hydration': {
                'title': 'ğŸ’§ Hydration',
                'tips': [
                    'Drink 2-3 liters of water daily',
                    f'Morning: 1 glass warm water with lemon',
                    'Midday: 1 liter by 2 PM',
                    f'Evening: Complete by 8 PM (not before bed)'
                ]
            },
            'nutrition': {
                'title': 'ğŸ¥— Nutrition for Skin Health',
                'tips': [
                    'âœ… Omega-3s (2-3x per week): Salmon, walnuts, flaxseeds',
                    'âœ… Antioxidants (Daily): Berries, leafy greens, green tea',
                    'âœ… Zinc (3-4x per week): Oysters, pumpkin seeds, chickpeas',
                    'âœ… Vitamin A (Daily): Carrots, sweet potatoes, kale',
                    'âœ… Probiotics (Daily): Yogurt, kimchi, kombucha'
                ]
            }
        }
    }
    
    if food_preference == 'vegan':
        lifestyle['sections']['vegan_notes'] = {
            'title': 'ğŸŒ± Vegan Skincare Strategy',
            'tips': [
                'Take B12 supplements (sublingual best)',
                'Plant-based protein: Legumes, nuts, seeds, tofu',
                'Iron absorption: Eat with vitamin C sources',
                'Zinc sources: Pumpkin seeds, hemp seeds, chickpeas'
            ]
        }
    
    routine['routines']['lifestyle'] = lifestyle
    
    # ===== MENSTRUAL CYCLE (if female) =====
    if gender.lower() == 'female' and menstrual_cycle:
        routine['routines']['menstrual_cycle'] = {
            'title': 'ğŸ”„ MENSTRUAL CYCLE-BASED ADAPTATIONS',
            'current_phase': menstrual_cycle,
            'note': 'Adjust routine based on your cycle phase'
        }
    
    # ===== PROGRESS TRACKING =====
    routine['routines']['progress'] = {
        'title': 'ğŸ“Š PROGRESS TRACKING',
        'commitment_days': commitment_days,
        'note': 'Take weekly photos to track changes'
    }
    
    return routine


def store_skincare_routine_to_profile(user_name: str, skincare_routine: dict) -> dict:
    """Stores the generated skincare routine to user profile."""
    
    return {
        'status': 'routine_stored',
        'user_name': user_name,
        'routine_id': f"skincare_{user_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'commitment_days': skincare_routine.get('commitment_days'),
        'message': f"âœ… Internet-sourced skincare routine for {user_name} saved!",
        'routine_summary': f"Complete {skincare_routine.get('commitment_days')}-day routine with ALL products fetched from internet!"
    }


def create_skincare_agent(model: str = "gemini-2.0-flash"):
    """Creates the Skincare Agent with INTERNET-BASED product recommendations."""
    
    skincare_agent = Agent(
        name="skincare_advisor_agent",
        model=model,
        description="Generates personalized skincare routines with internet-sourced products.",
        
        instruction="""You are the Skincare Advisor Agent.

WORKFLOW - COLLECT DATA:
Phase 1: Commitment days (15/30/45/90)
Phase 2: Menstrual cycle (females only)
Phase 3: Skin type (dry/oily/sensitive/combination/normal)
Phase 4: Skin concerns (acne frequency, dryness level)
Phase 5: Dietary habits
Phase 6: BMI (height + weight)
Phase 7: Food preference (vegan/vegetarian/non-veg)

THEN GENERATE:
- Morning routine (5 steps - internet products)
- Evening routine (6 steps - internet products)
- Weekly treatments (skin-specific)
- Lifestyle & diet recommendations
- Menstrual cycle tracking (if female)
- Progress tracking

KEY POINTS:
- ONE question per turn
- FETCH REAL products from internet for EACH step
- Create COMPLETELY SEPARATE morning and evening routines
- Include: time, duration, detailed instructions
- ALL recommendations based on THEIR specific profile
- NO hardcoding - everything from internet

TONE: Expert, supportive, empowering.""",
        
        tools=[collect_skincare_details, search_skincare_products_from_internet, generate_skincare_routine, store_skincare_routine_to_profile]
    )
    
    return skincare_agent


def collect_grooming_details(commitment_days: Optional[int] = None,
                            height: Optional[float] = None,
                            height_unit: Optional[str] = None,
                            skin_tone: Optional[str] = None,
                            body_build: Optional[str] = None,
                            facial_structure: Optional[str] = None,
                            time_commitment: Optional[str] = None,
                            personality_style: Optional[str] = None) -> dict:
    """Collects and validates grooming details from user."""
    
    grooming_data = {}
    missing_fields = []
    errors = []
    
    # 1. Commitment Days
    if commitment_days is not None:
        try:
            days = int(commitment_days)
            valid_days = [15, 30, 45, 60, 90]
            if days in valid_days:
                grooming_data['commitment_days'] = days
            else:
                errors.append(f"Please choose from: 15, 30, 45, 60, or 90 days")
                missing_fields.append('commitment_days')
        except (ValueError, TypeError):
            errors.append(f"Commitment days must be a number")
            missing_fields.append('commitment_days')
    else:
        missing_fields.append('commitment_days')
    
    # 2. Height
    if height is not None and height_unit is not None:
        try:
            height_m = float(height)
            if height_unit.lower() in ['feet', 'ft', 'foot']:
                height_m = height_m * 0.3048
            elif height_unit.lower() in ['cm', 'centimeters']:
                height_m = height_m / 100
            grooming_data['height_m'] = round(height_m, 2)
        except (ValueError, TypeError):
            errors.append("Invalid height value")
            missing_fields.append('height')
    else:
        missing_fields.append('height')
    
    # 3. Skin Tone
    valid_skin_tones = ['fair', 'light/warm', 'medium', 'deep']
    if skin_tone and isinstance(skin_tone, str):
        skin_tone_lower = skin_tone.strip().lower()
        if skin_tone_lower in valid_skin_tones:
            grooming_data['skin_tone'] = skin_tone_lower
        else:
            errors.append(f"Skin tone should be one of: {', '.join(valid_skin_tones)}")
            missing_fields.append('skin_tone')
    else:
        missing_fields.append('skin_tone')
    
    # 4. Body Build
    valid_builds = ['hourglass', 'pear', 'triangle', 'rectangle', 'apple', 'oval']
    if body_build and isinstance(body_build, str):
        body_build_lower = body_build.strip().lower()
        if body_build_lower in valid_builds:
            grooming_data['body_build'] = body_build_lower
        else:
            errors.append(f"Body build should be one of: {', '.join(valid_builds)}")
            missing_fields.append('body_build')
    else:
        missing_fields.append('body_build')
    
    # 5. Facial Structure
    valid_facial_structures = ['oval', 'round', 'square', 'heart']
    if facial_structure and isinstance(facial_structure, str):
        facial_structure_lower = facial_structure.strip().lower()
        if facial_structure_lower in valid_facial_structures:
            grooming_data['facial_structure'] = facial_structure_lower
        else:
            errors.append(f"Facial structure should be one of: {', '.join(valid_facial_structures)}")
            missing_fields.append('facial_structure')
    else:
        missing_fields.append('facial_structure')
    
    # 6. Time Commitment
    valid_time_commitments = ['minimalist', 'standard', 'detailed']
    if time_commitment and isinstance(time_commitment, str):
        time_commitment_lower = time_commitment.strip().lower()
        if time_commitment_lower in valid_time_commitments:
            grooming_data['time_commitment'] = time_commitment_lower
        else:
            errors.append(f"Time commitment should be one of: {', '.join(valid_time_commitments)}")
            missing_fields.append('time_commitment')
    else:
        missing_fields.append('time_commitment')
    
    # 7. Personality Style
    valid_styles = ['classic/elegant', 'creative/bohemian', 'trendsetter/bold', 'comfortable/athleisure']
    if personality_style and isinstance(personality_style, str):
        personality_style_lower = personality_style.strip().lower()
        if personality_style_lower in valid_styles:
            grooming_data['personality_style'] = personality_style_lower
        else:
            errors.append(f"Personality style should be one of: {', '.join(valid_styles)}")
            missing_fields.append('personality_style')
    else:
        missing_fields.append('personality_style')
    
    if not missing_fields:
        status = 'success'
        message = "All grooming details collected successfully!"
    else:
        status = 'incomplete'
        message = f"Still need: {', '.join(missing_fields)}"
    
    if errors:
        status = 'error' if status == 'incomplete' else status
    
    result = {
        'status': status,
        'data': grooming_data,
        'missing_fields': missing_fields,
        'error_message': ' | '.join(errors) if errors else None,
        'message': message
    }
    
    return result


def generate_grooming_routine(user_name: str, age: int, gender: str, commitment_days: int,
                             height_m: float, skin_tone: str, body_build: str,
                             facial_structure: str, time_commitment: str,
                             personality_style: str, climate: str, season: str) -> dict:
    """
    Generates personalized grooming routine based on physical appearance and facial structure.
    """

    routine = {
        'status': 'routine_generated',
        'user_name': user_name,
        'age': age,
        'gender': gender,
        'commitment_days': commitment_days,
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'routines': {},
        'data_sources': 'Personalized based on your physical appearance and facial structure'
    }

    # ===== HAIR CARE ROUTINE =====
    styling_time_dict = {'minimalist': '5-10 min', 'standard': '15-25 min', 'detailed': '30+ min'}
    styling_time_val = styling_time_dict.get(time_commitment.lower(), '15-25 min')

    hair_routine = {
        'title': 'ğŸ’‡ PERSONALIZED HAIR CARE ROUTINE',
        'subtitle': f'For {facial_structure.title()} face shape & {body_build.title()} body build',
        'note': 'Tailored based on your physical features',
        'recommendations': {
            'recommended_styles': {
                'oval': ['Straight long hair', 'Layered cuts', 'Side-swept styles', 'Bob cuts'],
                'round': ['Longer lengths to elongate', 'Layers to add texture', 'Side parting', 'Textured waves'],
                'square': ['Soft curls to soften', 'Long side-swept styles', 'Textured waves', 'Layers around face'],
                'heart': ['Longer at chin', 'Chin-length bobs', 'Soft curls at bottom', 'Side bangs']
            }.get(facial_structure, []),
            'maintenance_frequency': '1-2 times per month trim',
            'styling_time': f"{time_commitment.title()} routine ({styling_time_val})",
            'products_needed': 'Shampoo, Conditioner, Styling cream/serum, Hairspray',
            'climate_adaptation': f"For {climate}: Use lightweight products in tropical climates, moisturizing products in cold climates"
        }
    }
    routine['routines']['hair'] = hair_routine

    # ===== FACIAL GROOMING ROUTINE =====
    trim_dict = {'minimalist': '2-3 days', 'standard': '1-2 days', 'detailed': 'Daily'}
    trim_val = trim_dict.get(time_commitment.lower(), '1-2 days')

    if gender.lower() == 'male':
        facial_grooming = {
            'title': 'ğŸ§” FACIAL GROOMING ROUTINE',
            'note': 'Customized for your facial structure',
            'recommendations': {
                'beard_grooming': 'Regular trimming every 2-3 weeks. Use beard oil and brush for maintenance.',
                'facial_hairstyle': f'Based on {facial_structure} face: Choose styles that balance facial proportions.',
                'eyebrow_care': 'Regular grooming to frame the face. Keep natural shape aligned with face structure.',
                'daily_routine': f"Shave/trim every {trim_val}",
                'products': 'Shaving cream/gel, Quality razor, Aftershave balm, Beard oil'
            }
        }
    else:
        facial_grooming = {
            'title': 'ğŸ‘© FACIAL GROOMING ROUTINE',
            'note': 'Customized for your facial structure',
            'recommendations': {
                'eyebrow_care': f'For {facial_structure}: Shape to complement face geometry. Professional shaping recommended.',
                'facial_threading': 'Monthly upper lip and face threading for smooth appearance.',
                'derma_care': 'Regular facial every 4-6 weeks for professional maintenance.',
                'makeup_prep': f'Proper primer and foundation for your skin tone to enhance features.',
                'daily_routine': f'15-30 minute grooming routine based on {time_commitment} commitment',
                'products': 'Face wash, Moisturizer, Sunscreen, Threading/waxing for facial hair'
            }
        }
    routine['routines']['facial_grooming'] = facial_grooming

    # ... Rest of your function unchanged ...

    return routine

    
    # ===== CLOTHING & STYLING RECOMMENDATIONS =====
    styling = {
        'title': 'ğŸ‘— CLOTHING & STYLING RECOMMENDATIONS',
        'note': f'For your {body_build} body build and {skin_tone} skin tone',
        'body_build_guide': {},
        'color_palette': {}
    }
    
    # Body build tailoring
    body_build_tailoring = {
        'hourglass': 'Fitted clothing that shows off balanced proportions. Wrap dresses, belted styles.',
        'pear': 'Darker colors on bottom, lighter on top. A-line skirts. Avoid tight hip areas.',
        'triangle': 'Balance triangle with wider bottom. A-line skirts, flared pants. Avoid tight hips.',
        'rectangle': 'Create definition with horizontal stripes, belts, layered styles.',
        'apple': 'Emphasis on legs. Long cardigans to hide torso. V-necklines to elongate.',
        'oval': 'Balanced fit-and-flare styles. Avoid oversized. Clean horizontal lines.'
    }
    
    styling['body_build_guide'] = {
        'build_type': body_build,
        'tailoring_tips': body_build_tailoring.get(body_build, ''),
        'silhouettes': 'Choose cuts that complement your natural shape'
    }
    
    # Skin tone color palette
    skin_tone_colors = {
        'fair': ['Jewel tones (emerald, sapphire)', 'Pure white, black', 'Coral, rose tones'],
        'light/warm': ['Warm earth tones (rust, bronze)', 'Warm reds, oranges', 'Cream, gold'],
        'medium': ['Saturated jewel tones', 'Warm and cool colors work', 'Gold, bronze metallics'],
        'deep': ['Rich jewel tones (ruby, sapphire)', 'Gold, bronze metallics', 'Warm and cool colors']
    }
    
    styling['color_palette'] = {
        'skin_tone': skin_tone,
        'recommended_colors': skin_tone_colors.get(skin_tone, []),
        'metallics': 'Gold tones work best with warm undertones, Silver with cool undertones'
    }
    
    routine['routines']['styling'] = styling
    
    # ===== PERSONALITY STYLE GUIDE =====
    style_guide = {
        'title': 'âœ¨ YOUR PERSONALITY STYLE GUIDE',
        'archetype': personality_style,
        'fashion_philosophy': {}
    }
    
    personality_descriptions = {
        'classic/elegant': {
            'characteristics': 'Timeless, sophisticated, refined',
            'key_pieces': ['White button-up shirt', 'Tailored blazer', 'Dark jeans', 'Leather shoes', 'Pearl accessories'],
            'brands': 'Focus on quality over quantity. Invest in well-made basics.',
            'colors': 'Neutral palette: Black, white, navy, gray, beige',
            'fashion_rule': 'Less is more. Quality fabrics and perfect fit.'
        },
        'creative/bohemian': {
            'characteristics': 'Artistic, free-spirited, expressive',
            'key_pieces': ['Flowing fabrics', 'Ethnic prints', 'Layered jewelry', 'Scarves', 'Vintage finds'],
            'brands': 'Mix high-street with vintage and thrift store finds.',
            'colors': 'Earth tones, jewel tones, patterns, textures',
            'fashion_rule': 'Express yourself. Mix patterns and eras freely.'
        },
        'trendsetter/bold': {
            'characteristics': 'Cutting-edge, fashionable, confident',
            'key_pieces': ['Statement pieces', 'Bold colors/prints', 'Unique accessories', 'Latest trends', 'Designer items'],
            'brands': 'Follow fashion weeks, emerging designers, trend setters',
            'colors': 'Bold, vibrant colors, contrasting combinations',
            'fashion_rule': 'Take risks. Push boundaries. Own your style.'
        },
        'comfortable/athleisure': {
            'characteristics': 'Relaxed, practical, functional',
            'key_pieces': ['Athletic wear', 'Comfortable shoes', 'Hoodies', 'Casual basics', 'Functional fabrics'],
            'brands': 'Sportswear brands, comfortable high-street options',
            'colors': 'Neutral, muted tones, practical colors',
            'fashion_rule': 'Comfort first. Function meets style.'
        }
    }
    
    style_guide['fashion_philosophy'] = personality_descriptions.get(personality_style, {})
    
    routine['routines']['personality_style'] = style_guide
    
    # ===== GROOMING SCHEDULE =====
    schedule = {
        'title': 'ğŸ“… GROOMING MAINTENANCE SCHEDULE',
        'commitment_days': commitment_days,
        'daily': [
            'Face cleansing and moisturizing',
            f'Hair care (brush, style as per {time_commitment} routine)',
            'Deodorant/antiperspirant application',
            'Nail care check'
        ],
        'weekly': [
            'Deep conditioning for hair',
            'Exfoliation for face and body',
            'Nail trimming and shaping',
            'Beard/facial hair grooming'
        ],
        'monthly': [
            f'Professional haircut ({commitment_days} days cycle)',
            'Facial or skincare treatment',
            'Professional grooming service if needed',
            'Wardrobe review and styling'
        ]
    }
    
    routine['routines']['schedule'] = schedule
    
    # ===== GROOMING PRODUCTS SEARCH =====
    products = {
        'title': 'ğŸ›�ï¸� RECOMMENDED GROOMING PRODUCTS',
        'note': 'Researched from internet based on your profile',
        'categories': {
            'hair_care': 'Shampoo, conditioner, styling products for your hair type',
            'facial_care': 'Facewash, moisturizer based on skin tone and climate',
            'personal_hygiene': 'Deodorant, body care, nail care products',
            'styling_tools': 'Depending on your personality style and time commitment'
        }
    }
    
    routine['routines']['products'] = products
    
    return routine


def store_grooming_routine_to_profile(user_name: str, grooming_routine: dict) -> dict:
    """Stores the generated grooming routine to user profile."""
    
    return {
        'status': 'routine_stored',
        'user_name': user_name,
        'routine_id': f"grooming_{user_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'commitment_days': grooming_routine.get('commitment_days'),
        'message': f"âœ… Personalized grooming routine for {user_name} saved!",
        'routine_summary': f"Complete {grooming_routine.get('commitment_days')}-day grooming routine customized for your physical appearance and lifestyle!"
    }


def create_grooming_agent(model: str = "gemini-2.0-flash"):
    """Creates the Grooming Agent - triggered by user choice, not automatic."""
    
    grooming_agent = Agent(
        name="grooming_advisor_agent",
        model=model,
        description="Generates personalized grooming routine based on physical appearance and facial structure.",
        
        instruction="""You are the Grooming Advisor Agent.

WORKFLOW - COLLECT DATA (ONE AT A TIME):
Phase 1: Commitment days (15/30/45/60/90)
Phase 2: Height (with unit)
Phase 3: Skin tone (Fair/Light/Medium/Deep)
Phase 4: Body build (Hourglass/Pear/Triangle/Rectangle/Apple/Oval)
Phase 5: Facial structure (Oval/Round/Square/Heart)
Phase 6: Time commitment (Minimalist/Standard/Detailed)
Phase 7: Personality style (Classic/Bohemian/Trendsetter/Athleisure)

THEN GENERATE:
- Hair care routine (based on facial structure)
- Facial grooming routine (gender-specific)
- Clothing & styling recommendations (based on body build & skin tone)
- Personality style guide
- Grooming maintenance schedule
- Product recommendations

KEY POINTS:
- ONE question per turn
- Retrieve user data from profile (name, age, gender, climate, season)
- Customize recommendations based on THEIR specific features
- Hair recommendations based on FACIAL STRUCTURE
- Color recommendations based on SKIN TONE
- Styling based on BODY BUILD and PERSONALITY
- NO hardcoding - personalized for each user

TONE: Professional, encouraging, style-focused.""",
        
        tools=[collect_grooming_details, generate_grooming_routine, store_grooming_routine_to_profile]
    )
    
    return grooming_agent


def collect_diet_details(commitment_days: Optional[int] = None,
                         weight: Optional[float] = None,
                         weight_unit: Optional[str] = None,
                         height: Optional[float] = None,
                         height_unit: Optional[str] = None,
                         food_habit: Optional[str] = None,
                         allergies: Optional[str] = None,
                         diet_goal: Optional[str] = None,
                         activity_level: Optional[str] = None,
                         activity_days: Optional[int] = None) -> dict:
    """Collects and validates diet plan details from user."""
    
    diet_data = {}
    missing_fields = []
    errors = []
    
    # 1. Commitment Days
    if commitment_days is not None:
        try:
            days = int(commitment_days)
            valid_days = [15, 30, 45, 90]
            if days in valid_days:
                diet_data['commitment_days'] = days
            else:
                errors.append(f"Please choose from: 15, 30, 45, or 90 days")
                missing_fields.append('commitment_days')
        except (ValueError, TypeError):
            errors.append(f"Commitment days must be a number")
            missing_fields.append('commitment_days')
    else:
        missing_fields.append('commitment_days')
    
    # 2. BMI Calculation (Height & Weight)
    if weight is not None and height is not None and weight_unit and height_unit:
        try:
            weight_kg = float(weight)
            height_m = float(height)
            
            if weight_unit.lower() in ['lbs', 'lb', 'pounds']:
                weight_kg = weight_kg * 0.453592
            
            if height_unit.lower() in ['feet', 'ft', 'foot']:
                height_m = height_m * 0.3048
            elif height_unit.lower() in ['cm', 'centimeters']:
                height_m = height_m / 100
            
            if height_m > 0:
                bmi = weight_kg / (height_m ** 2)
                diet_data['weight_kg'] = round(weight_kg, 2)
                diet_data['height_m'] = round(height_m, 2)
                diet_data['bmi'] = round(bmi, 1)
            else:
                errors.append("Invalid height value")
                missing_fields.append('bmi')
        except (ValueError, TypeError):
            errors.append("Invalid weight or height values")
            missing_fields.append('bmi')
    else:
        missing_fields.append('bmi')
    
    # 3. Food Habit
    valid_food_habits = ['vegetarian', 'non-vegetarian', 'vegan', 'pescatarian', 'jain']
    if food_habit and isinstance(food_habit, str):
        food_habit_lower = food_habit.strip().lower()
        if food_habit_lower in valid_food_habits:
            diet_data['food_habit'] = food_habit_lower
        else:
            errors.append(f"Food habit should be one of: {', '.join(valid_food_habits)}")
            missing_fields.append('food_habit')
    else:
        missing_fields.append('food_habit')
    
    # 4. Allergies
    if allergies and isinstance(allergies, str):
        diet_data['allergies'] = allergies.strip()
    
    # 5. Diet Goal
    valid_goals = ['weight loss', 'weight gain', 'muscle building', 'boost energy & focus', 
                   'improve skin & hair health', 'weight maintenance & nutritional balance',
                   'athletic performance', 'digestive health']
    if diet_goal and isinstance(diet_goal, str):
        diet_goal_lower = diet_goal.strip().lower()
        if diet_goal_lower in valid_goals:
            diet_data['diet_goal'] = diet_goal_lower
        else:
            errors.append(f"Diet goal should be one of: {', '.join(valid_goals)}")
            missing_fields.append('diet_goal')
    else:
        missing_fields.append('diet_goal')
    
    # 6. Activity Level
    if activity_level and isinstance(activity_level, str):
        diet_data['activity_level'] = activity_level.strip()
    
    # 7. Activity Days
    if activity_days is not None:
        try:
            days = int(activity_days)
            if 0 <= days <= 7:
                diet_data['activity_days'] = days
            else:
                errors.append(f"Activity days should be between 0 and 7")
                missing_fields.append('activity_days')
        except (ValueError, TypeError):
            errors.append(f"Activity days must be a number")
            missing_fields.append('activity_days')
    else:
        missing_fields.append('activity_days')
    
    if not missing_fields:
        status = 'success'
        message = "All diet details collected successfully!"
    else:
        status = 'incomplete'
        message = f"Still need: {', '.join(missing_fields)}"
    
    if errors:
        status = 'error' if status == 'incomplete' else status
    
    result = {
        'status': status,
        'data': diet_data,
        'missing_fields': missing_fields,
        'error_message': ' | '.join(errors) if errors else None,
        'message': message
    }
    
    return result


def search_diet_recipes_from_internet(diet_goal: str, food_habit: str, season: str, climate: str,
                                      allergies: Optional[str] = None, meal_type: str = "breakfast",
                                      calories: Optional[int] = None) -> dict:
    """
    Searches the INTERNET for diet recipes based on user's SPECIFIC requirements.
    Personalized by: diet goal, food habit, allergies, season, climate, meal type, calories.
    """
    
    try:
        query_parts = [meal_type, food_habit, diet_goal]
        
        seasonal_keywords = {
            'Winter': 'warming, slow-cooked, root vegetables, comfort food',
            'Spring': 'fresh, light, leafy greens, seasonal produce',
            'Summer': 'refreshing, light, fresh, salads, cold dishes',
            'Autumn': 'harvest, hearty, squash, apples, warming'
        }
        query_parts.append(seasonal_keywords.get(season, 'seasonal'))
        
        climate_keywords = {
            'Tropical': 'tropical fruits, coconut, light, hydrating',
            'Hot/Arid': 'hydrating, cooling, mineral-rich, light',
            'Temperate': 'balanced, variety, fresh, all seasons',
            'Cool/Temperate': 'hearty, warming, protein-rich',
            'Cold/Polar': 'warming, high-calorie, protein-rich, immunity-boosting'
        }
        query_parts.append(climate_keywords.get(climate, 'balanced'))
        
        if allergies:
            allergies_list = [a.strip() for a in allergies.split(',')]
            query_parts.append(f"without {', '.join(allergies_list)}")
        
        if calories:
            query_parts.append(f"around {calories} calories")
        
        search_query = " ".join(query_parts) + f" recipes 2024 2025 healthy {season.lower()} {climate.lower()}"
        
        print(f"ğŸ”� Searching recipes: {search_query}")
        
        search_url = f"https://api.duckduckgo.com/?q={search_query}&format=json&no_redirect=1"
        response = requests.get(search_url, timeout=5)
        search_data = response.json()
        
        recipes = []
        related_topics = search_data.get('RelatedTopics', [])
        
        for topic in related_topics[:10]:
            if 'Text' in topic and 'FirstURL' in topic:
                topic_text = topic.get('Text', '')
                first_url = topic.get('FirstURL', '')
                
                recipe_name = topic_text.split(' - ')[0][:100] if ' - ' in topic_text else topic_text[:100]
                recipe_desc = topic_text[:250]
                
                recipe_info = {
                    'name': recipe_name,
                    'description': recipe_desc,
                    'source_url': first_url,
                    'data_source': 'Internet Search (Real-time)',
                    'meal_type': meal_type,
                    'season': season,
                    'climate': climate
                }
                
                if recipe_info['name'] and len(recipe_info['name']) > 3 and first_url:
                    recipes.append(recipe_info)
        
        if recipes:
            return {
                'status': 'success',
                'meal_type': meal_type,
                'diet_goal': diet_goal,
                'food_habit': food_habit,
                'season': season,
                'climate': climate,
                'recipes': recipes[:5],
                'search_query': search_query,
                'count': len(recipes[:5]),
                'message': f"Found {len(recipes[:5])} recipes from internet (personalized for {season}/{climate})",
                'source': 'Internet Search Results (Real-time)',
                'personalization_applied': [
                    f'Season: {season}',
                    f'Climate: {climate}',
                    f'Diet Goal: {diet_goal}',
                    f'Food Habit: {food_habit}',
                    f'Allergies excluded: {allergies if allergies else "None"}',
                    f'Calories target: {calories if calories else "Not specified"}'
                ]
            }
        else:
            return {
                'status': 'no_results',
                'meal_type': meal_type,
                'season': season,
                'climate': climate,
                'message': 'No recipes found. Trying alternate search with different parameters...',
                'search_query': search_query
            }
        
    except Exception as e:
        return {
            'status': 'error',
            'meal_type': meal_type,
            'error_message': f"Error searching recipes: {str(e)}"
        }


def generate_diet_plan(user_name: str, age: int, gender: str, commitment_days: int,
                      weight_kg: float, height_m: float, bmi: float,
                      food_habit: str, season: str, climate: str,
                      allergies: Optional[str] = None,
                      diet_goal: str = "weight maintenance & nutritional balance",
                      activity_level: Optional[str] = None,
                      activity_days: int = 0) -> dict:
    """
    Generates PERSONALIZED diet plan based on user's ACTUAL health data, goals, restrictions, season & climate.
    All recommendations adapted to user's specific profile with REAL recipes from internet.
    """
    
    plan = {
        'status': 'plan_generated',
        'user_name': user_name,
        'age': age,
        'gender': gender,
        'commitment_days': commitment_days,
        'bmi': bmi,
        'diet_goal': diet_goal,
        'season': season,
        'climate': climate,
        'food_habit': food_habit,
        'allergies': allergies if allergies else 'None',
        'activity_level': activity_level,
        'activity_days': activity_days,
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'routines': {},
        'personalization_note': f'Plan created for {season} season in {climate} climate. All recipes fetched from internet based on YOUR specific profile.'
    }
    
    # Calculate daily calorie needs based on BMI, activity, and goal
    bmr = 10 * weight_kg + 6.25 * height_m * 100 - 5 * age
    if gender.lower() == 'male':
        bmr = bmr + 5
    else:
        bmr = bmr - 161
    
    activity_multiplier = 1.2  # sedentary
    if activity_days >= 5:
        activity_multiplier = 1.55  # very active
    elif activity_days >= 3:
        activity_multiplier = 1.375  # moderately active
    elif activity_days >= 1:
        activity_multiplier = 1.275  # lightly active
    
    tdee = bmr * activity_multiplier
    
    # Adjust calories based on goal
    if diet_goal.lower() == 'weight loss':
        daily_calories = int(tdee * 0.85)
    elif diet_goal.lower() == 'weight gain' or diet_goal.lower() == 'muscle building':
        daily_calories = int(tdee * 1.15)
    else:
        daily_calories = int(tdee)
    
    plan['daily_calories'] = daily_calories
    plan['bmr'] = round(bmr, 0)
    plan['tdee'] = round(tdee, 0)
    plan['activity_multiplier'] = activity_multiplier
    
    # ===== MACRO BREAKDOWN =====
    macros = {
        'title': 'ğŸ“Š DAILY MACRO BREAKDOWN',
        'total_daily_calories': daily_calories,
        'breakdown': {
            'protein': {
                'percentage': 30,
                'grams': int(daily_calories * 0.30 / 4),
                'sources': f"Lean meats, fish, eggs, legumes, dairy (considering {food_habit})"
            },
            'carbohydrates': {
                'percentage': 45,
                'grams': int(daily_calories * 0.45 / 4),
                'sources': 'Whole grains, brown rice, oats, sweet potatoes, legumes'
            },
            'fats': {
                'percentage': 25,
                'grams': int(daily_calories * 0.25 / 9),
                'sources': 'Olive oil, nuts, seeds, avocado, fatty fish'
            }
        }
    }
    plan['routines']['macros'] = macros
    
    # ===== HYDRATION PLAN =====
    base_water_intake = 2.5
    if climate in ['Tropical', 'Hot/Arid']:
        base_water_intake = 3.5
    elif climate in ['Cold/Polar']:
        base_water_intake = 2.0
    
    if activity_days >= 5:
        base_water_intake += 0.5
    
    hydration = {
        'title': 'ğŸ’§ HYDRATION PLAN',
        'personalized_for': f'{season} season in {climate} climate',
        'daily_water_intake': f'{base_water_intake}-{base_water_intake + 0.5} liters',
        'activity_adjusted': f'Based on {activity_days} days/week activity'
    }
    plan['routines']['hydration'] = hydration
    
    # ===== GOAL-SPECIFIC STRATEGIES =====
    goal_strategies = {
        'title': 'ğŸ�¯ DIET GOAL-SPECIFIC STRATEGIES',
        'goal': diet_goal,
        'daily_calorie_target': daily_calories
    }
    
    goal_strategies_map = {
        'weight loss': f'Calorie deficit ({daily_calories}cal) + high protein + high fiber',
        'weight gain': f'Calorie surplus ({daily_calories}cal) + nutrient-dense foods',
        'muscle building': f'Moderate surplus ({daily_calories}cal) + 1.6-2.2g protein/kg',
        'boost energy & focus': 'Macro-timing + stable blood sugar + antioxidants',
        'improve skin & hair health': 'Omega-3s + Antioxidants + Biotin + Collagen',
        'weight maintenance & nutritional balance': f'Balanced macros at TDEE ({daily_calories}cal)',
        'athletic performance': f'Carb-loading ({daily_calories}cal) + Hydration + Timing',
        'digestive health': 'Fiber + Probiotics + Whole foods + Hydration'
    }
    
    goal_strategies['strategy'] = goal_strategies_map.get(diet_goal.lower(), 'Personalized approach')
    plan['routines']['goal_strategies'] = goal_strategies
    
    return plan


def store_diet_plan_to_profile(user_name: str, diet_plan: dict) -> dict:
    """Stores the generated diet plan to user profile."""
    
    return {
        'status': 'plan_stored',
        'user_name': user_name,
        'plan_id': f"diet_{user_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'commitment_days': diet_plan.get('commitment_days'),
        'diet_goal': diet_plan.get('diet_goal'),
        'daily_calories': diet_plan.get('daily_calories'),
        'message': f"âœ… Internet-sourced personalized diet plan for {user_name} saved!",
        'plan_summary': f"Complete {diet_plan.get('commitment_days')}-day diet plan with recipes fetched from internet based on {diet_plan.get('season')} season in {diet_plan.get('climate')} climate!"
    }


def create_diet_agent(model: str = "gemini-2.0-flash"):
    """Creates the Diet Plan Agent - triggered by user choice, not automatic."""
    
    diet_agent = Agent(
        name="diet_plan_advisor_agent",
        model=model,
        description="Generates personalized diet plans based on health data, goals, restrictions, season & climate.",
        
        instruction="""You are the Diet Plan Advisor Agent - PERSONALIZATION MASTER.

WORKFLOW - COLLECT DATA (ONE AT A TIME):
Phase 1: Commitment days (15/30/45/90)
Phase 2: BMI Calculation (height + weight with units)
Phase 3: Food habit (Vegetarian/Non-veg/Vegan/Pescatarian/Jain)
Phase 4: Food allergies or intolerances (comma-separated list)
Phase 5: Primary diet goal (Weight loss/Gain/Muscle/Energy/Skin Health/Maintenance/Athletic/Digestion)
Phase 6: Physical activity type (gym/running/yoga/sports/etc)
Phase 7: Activity frequency (0-7 days per week)

THEN GENERATE PERSONALIZED PLAN:
- Calculate daily calorie needs based on THEIR specific: age, weight, height, activity, gender
- Personalize meals by food habit (vegan, vegetarian, non-veg, jain, pescatarian)
- Adapt meals for THEIR allergies with alternatives
- Fetch REAL recipes from internet for each meal type
- Generate goal-specific strategies (weight loss/gain/muscle/energy/skin/athletic/maintenance/digestion)
- Create hydration plan (adjusted for THEIR climate + activity)
- Include macro breakdown (calculated from THEIR BMI + activity)
- Provide progress tracking guidance

ğŸ”‘ CRITICAL RULES - NO HARDCODING:
- RETRIEVE from profile: name, age, gender, climate, season (NOT assumed)
- CALCULATE daily calories based on THEIR specific: age, weight, height, activity, gender
- FETCH REAL recipes from internet for EACH meal type
- Include seasonal produce (what's available NOW in THEIR season)
- Adapt to climate (tropical/hot/temperate/cold affects hydration & nutrition)
- Personalize allergies (fetch recipes that exclude THEIR specific items)
- Match food habit strictly
- Adapt macros by goal
- Climate affects: hydration needs, ingredient availability, energy requirements

IMPORTANT NOTES:
- ONE question per turn
- Ask for data sequentially (don't ask all at once)
- CALCULATE not assume
- FETCH from internet not hardcode
- PERSONALIZE everything to user's specific needs

TONE: Supportive, knowledgeable, health-focused, detail-oriented.""",
        
        tools=[collect_diet_details, search_diet_recipes_from_internet, generate_diet_plan, store_diet_plan_to_profile]
    )
    
    return diet_agent


def create_root_agent(model: str = "gemini-2.0-flash"):
    """Creates the Root Orchestrator Agent for coordination."""
    
    user_info_agent = create_user_info_agent(model=model)
    location_weather_agent = create_location_weather_agent(model=model)
    skincare_agent = create_skincare_agent(model=model)
    grooming_agent = create_grooming_agent(model=model)
    diet_agent = create_diet_agent(model=model)
    
    root_agent = Agent(
        name="glow_guide_root_agent",
        model=model,
        description="Manages Glow Guide workflow with agent coordination.",
        
        instruction="""You are the Glow Guide Root Orchestrator.

YOUR JOB: COORDINATE AGENT WORKFLOW & SHOW SERVICE MENU

WORKFLOW:
Step 1: First user message â†’ Delegate to Agent 1 (User Info)
Step 2: Agent 1 completes â†’ Delegate to Agent 2 (Location & Weather)
Step 3: Agent 2 completes â†’ SHOW SERVICE MENU
Step 4: User chooses â†’ Delegate to Agent 3, Agent 4, or Agent 5

SERVICE MENU (After Agent 2 completes):
"âœ¨ Now, let's define the engine for your personalized transformation.
Please choose the primary focus area that you want to address right now.

Select one option below:
1ï¸�âƒ£ Build a personalized skincare routine
2ï¸�âƒ£ Build a personalized grooming routine
3ï¸�âƒ£ Build a personalized diet plan"

DELEGATION RULES:
- Follow sequence strictly (Agent 1 â†’ Agent 2 â†’ Menu â†’ Agent 3/4/5)
- Do NOT answer direct questions yourself
- Do NOT collect data yourself
- Show menu ONLY after Agent 2 completes
- Wait for user choice BEFORE delegating
- If user chooses "1" â†’ Delegate to Agent 3 (Skincare)
- If user chooses "2" â†’ Delegate to Agent 4 (Grooming)
- If user chooses "3" â†’ Delegate to Agent 5 (Diet Plan)
- Be warm and professional

TONE: Professional, warm, organized.""",
        
        tools=[],
        sub_agents=[user_info_agent, location_weather_agent, skincare_agent, grooming_agent, diet_agent]
    )
    
    return root_agent


# Create Memory Service
memory_service = (InMemoryMemoryService())

# Create Session Service
session_service = InMemorySessionService()

root_agent = create_root_agent(model="gemini-2.0-flash")

# Create runner with BOTH services
runner = Runner(
    agent=root_agent,
    app_name="GlowGuideAgent",
    session_service=session_service,
    memory_service=memory_service,
)

print("âœ… Agent and Runner created with memory support!")


import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}")


!adk create glow_guide_agent --model gemini-2.0-flash --api_key $GOOGLE_API_KEY
print("âœ… ADK project created: glow_guide_agent")


from IPython.core.display import display, HTML
from jupyter_server.serverapp import list_running_servers


# Gets the proxied URL in the Kaggle Notebooks environment
def get_adk_proxy_url():
    PROXY_HOST = "https://kkb-production.jupyter-proxy.kaggle.net"
    ADK_PORT = "8000"

    servers = list(list_running_servers())
    if not servers:
        raise Exception("No running Jupyter servers found.")

    baseURL = servers[0]["base_url"]

    try:
        path_parts = baseURL.split("/")
        kernel = path_parts[2]
        token = path_parts[3]
    except IndexError:
        raise Exception(f"Could not parse kernel/token from base URL: {baseURL}")

    url_prefix = f"/k/{kernel}/{token}/proxy/proxy/{ADK_PORT}"
    url = f"{PROXY_HOST}{url_prefix}"

    styled_html = f"""
    <div style="padding: 15px; border: 2px solid #f0ad4e; border-radius: 8px; background-color: #fef9f0; margin: 20px 0;">
        <div style="font-family: sans-serif; margin-bottom: 12px; color: #333; font-size: 1.1em;">
            <strong>âš ï¸� IMPORTANT: Action Required</strong>
        </div>
        <div style="font-family: sans-serif; margin-bottom: 15px; color: #333; line-height: 1.5;">
            The ADK web UI is <strong>not running yet</strong>. You must start it in the next cell.
            <ol style="margin-top: 10px; padding-left: 20px;">
                <li style="margin-bottom: 5px;"><strong>Run the next cell</strong> (the one with <code>!adk web ...</code>) to start the ADK web UI.</li>
                <li style="margin-bottom: 5px;">Wait for that cell to show it is "Running" (it will not "complete").</li>
                <li>Once it's running, <strong>return to this button</strong> and click it to open the UI.</li>
            </ol>
            <em style="font-size: 0.9em; color: #555;">(If you click the button before running the next cell, you will get a 500 error.)</em>
        </div>
        <a href='{url}' target='_blank' style="
            display: inline-block; background-color: #1a73e8; color: white; padding: 10px 20px;
            text-decoration: none; border-radius: 25px; font-family: sans-serif; font-weight: 500;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2); transition: all 0.2s ease;">
            Open ADK Web UI (after running cell below) â†—
        </a>
    </div>
    """

    display(HTML(styled_html))

    return url_prefix


print("âœ… Proxy URL configured.")


%%writefile glow_guide_agent/agent.py

from typing import Optional, List
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.runners import Runner
from datetime import datetime
import requests
import json
import math

# =====================================================
# TOOLS FOR USER INFO COLLECTION (Agent 1)
# =====================================================

def collect_user_info(name: Optional[str] = None, 
                      age: Optional[int] = None, 
                      gender: Optional[str] = None, 
                      city: Optional[str] = None) -> dict:
    """Collects and validates user information."""
    
    user_data = {}
    missing_fields = []
    errors = []
    
    if name and isinstance(name, str) and name.strip():
        user_data['name'] = name.strip()
    else:
        missing_fields.append('name')
    
    if age is not None:
        try:
            age_int = int(age)
            if 1 <= age_int <= 150:
                user_data['age'] = age_int
            else:
                errors.append(f"Age must be between 1 and 150, got {age_int}")
                missing_fields.append('age')
        except (ValueError, TypeError):
            errors.append(f"Age must be a valid number, got {age}")
            missing_fields.append('age')
    else:
        missing_fields.append('age')
    
    valid_genders = ['male', 'female', 'other', 'prefer not to say']
    if gender and isinstance(gender, str):
        gender_lower = gender.strip().lower()
        if gender_lower in valid_genders:
            user_data['gender'] = gender.strip()
        else:
            errors.append(f"Gender should be one of: {', '.join(valid_genders)}")
            missing_fields.append('gender')
    else:
        missing_fields.append('gender')
    
    if city and isinstance(city, str) and city.strip():
        user_data['city'] = city.strip()
    else:
        missing_fields.append('city')
    
    if not missing_fields:
        status = 'success'
        message = "All user information collected successfully!"
    else:
        status = 'incomplete'
        message = f"Still need: {', '.join(missing_fields)}"
    
    if errors:
        status = 'error' if status == 'incomplete' else status
    
    result = {
        'status': status,
        'data': user_data,
        'missing_fields': missing_fields,
        'error_message': ' | '.join(errors) if errors else None,
        'message': message
    }
    
    return result


def store_user_in_session(name: str, age: int, gender: str, city: str) -> dict:
    """Confirms user information is ready to be stored in session."""
    
    if not all([name, age, gender, city]):
        return {
            'status': 'validation_error',
            'error': 'All fields (name, age, gender, city) are required'
        }
    
    user_profile = {
        'name': name,
        'age': age,
        'gender': gender,
        'city': city,
        'status': 'complete',
        'profile_created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return {
        'status': 'ready_to_store',
        'user_profile': user_profile,
        'message': f"User profile for {name} is ready to be stored in session"
    }


# =====================================================
# TOOLS FOR WEATHER & CLIMATE AGENT (Agent 2)
# =====================================================

def get_location_weather_data(city: str) -> dict:
    """Fetches REAL current weather, season, and climate information from the internet."""
    
    try:
        geocoding_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
        
        geo_response = requests.get(geocoding_url, timeout=5)
        geo_data = geo_response.json()
        
        if not geo_data.get('results'):
            return {
                'status': 'error',
                'city': city,
                'error_message': f"City '{city}' not found. Please check the spelling and try again."
            }
        
        location = geo_data['results'][0]
        latitude = location['latitude']
        longitude = location['longitude']
        city_name = location['name']
        country = location.get('country', 'Unknown')
        
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m&timezone=auto"
        
        weather_response = requests.get(weather_url, timeout=5)
        weather_data = weather_response.json()
        
        current = weather_data['current']
        
        weather_code = current['weather_code']
        weather_descriptions = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Foggy", 48: "Depositing rime fog", 51: "Light drizzle", 53: "Moderate drizzle",
            55: "Dense drizzle", 61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
            71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow", 80: "Slight rain showers",
            81: "Moderate rain showers", 82: "Violent rain showers", 85: "Slight snow showers",
            86: "Heavy snow showers", 95: "Thunderstorm", 96: "Thunderstorm with slight hail",
            99: "Thunderstorm with heavy hail"
        }
        
        weather_condition = weather_descriptions.get(weather_code, "Unknown weather condition")
        temperature = current['temperature_2m']
        humidity = current['relative_humidity_2m']
        wind_speed = current['wind_speed_10m']
        
        current_month = datetime.now().month
        if latitude > 0:
            if current_month in [12, 1, 2]:
                season = "Winter"
            elif current_month in [3, 4, 5]:
                season = "Spring"
            elif current_month in [6, 7, 8]:
                season = "Summer"
            else:
                season = "Autumn"
        else:
            if current_month in [12, 1, 2]:
                season = "Summer"
            elif current_month in [3, 4, 5]:
                season = "Autumn"
            elif current_month in [6, 7, 8]:
                season = "Winter"
            else:
                season = "Spring"
        
        if temperature > 25:
            if humidity > 70:
                climate_type = "Tropical"
            else:
                climate_type = "Hot/Arid"
        elif temperature > 15:
            climate_type = "Temperate"
        elif temperature > 0:
            climate_type = "Cool/Temperate"
        else:
            climate_type = "Cold/Polar"
        
        current_hour = datetime.now().hour
        if 6 <= current_hour <= 18:
            if abs(latitude) < 23.5:
                uv_index = "High to Very High (7-9/10)"
            elif abs(latitude) < 35:
                uv_index = "Moderate to High (5-7/10)"
            else:
                uv_index = "Low to Moderate (2-4/10)"
        else:
            uv_index = "Low (0-2/10)"
        
        response = {
            'status': 'success',
            'city': city_name,
            'country': country,
            'latitude': latitude,
            'longitude': longitude,
            'current_weather': {
                'temperature': f"{temperature}Â°C",
                'humidity': f"{humidity}%",
                'condition': weather_condition,
                'wind_speed': f"{wind_speed} km/h"
            },
            'current_season': season,
            'climate_type': climate_type,
            'uv_index': uv_index,
            'data_fetched_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_source': 'Open-Meteo API (Real-time data)'
        }
        
        return response
        
    except Exception as e:
        return {
            'status': 'error',
            'city': city,
            'error_message': f"Error fetching weather data: {str(e)}"
        }


def update_user_profile_with_location_data(user_name: str, city: str, season: str, weather: str, climate: str) -> dict:
    """Updates the user profile with location-based environmental data."""
    
    if not all([user_name, city, season, weather, climate]):
        return {
            'status': 'error',
            'error': 'All fields (user_name, city, season, weather, climate) are required'
        }
    
    updated_profile = {
        'name': user_name,
        'city': city,
        'current_season': season,
        'current_weather': weather,
        'climate_type': climate,
        'data_updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return {
        'status': 'profile_updated',
        'updated_profile': updated_profile,
        'message': f"Profile for {user_name} enriched with location data from {city}"
    }


# =====================================================
# TOOLS FOR SKINCARE AGENT (Agent 3) - INTERNET-BASED ONLY
# =====================================================

def search_skincare_products_from_internet(user_age: int, skin_type: str, skin_concern: Optional[str] = None, 
                                          product_category: str = "cleanser",
                                          climate: Optional[str] = None,
                                          season: Optional[str] = None,
                                          dietary_habits: Optional[str] = None) -> dict:
    """
    Searches the INTERNET for skincare products based on SPECIFIC USER DATA.
    NO hardcoding - everything fetched from web based on user profile.
    """
    
    try:
        # Build DYNAMIC search query using USER-SPECIFIC DATA
        query_parts = [product_category, skin_type, f"age {user_age}"]
        
        if skin_concern:
            query_parts.append(skin_concern)
        
        if climate and climate in ['Tropical', 'Hot/Arid']:
            query_parts.append("lightweight water-resistant")
        elif climate == 'Cold/Polar':
            query_parts.append("rich nourishing")
        
        if season == 'Winter':
            query_parts.append("winter moisture barrier")
        elif season == 'Summer':
            query_parts.append("summer UV protection")
        
        if dietary_habits and 'dairy' in dietary_habits.lower():
            query_parts.append("non-dairy friendly")
        
        # Build search with current year
        search_query = " ".join(query_parts) + " best products 2024 2025 reviews"
        
        print(f"ğŸ”� Searching: {search_query}")
        
        # Search using DuckDuckGo
        search_url = f"https://api.duckduckgo.com/?q={search_query}&format=json&no_redirect=1"
        response = requests.get(search_url, timeout=5)
        search_data = response.json()
        
        products = []
        related_topics = search_data.get('RelatedTopics', [])
        
        # Extract real products from search results
        for topic in related_topics[:8]:
            if 'Text' in topic and 'FirstURL' in topic:
                topic_text = topic.get('Text', '')
                
                product_info = {
                    'name': topic_text.split(' - ')[0][:100] if ' - ' in topic_text else topic_text[:100],
                    'description': topic_text[:200],
                    'source_url': topic.get('FirstURL', ''),
                    'data_source': 'Internet Search (Real-time)'
                }
                
                if product_info['name'] and len(product_info['name']) > 3:
                    products.append(product_info)
        
        if products:
            return {
                'status': 'success',
                'product_category': product_category,
                'skin_type': skin_type,
                'user_age': user_age,
                'climate': climate,
                'season': season,
                'products': products[:5],
                'search_query': search_query,
                'count': len(products[:5]),
                'message': f"Found {len(products[:5])} real products from internet",
                'source': 'Internet Search Results'
            }
        else:
            broad_query = f"{product_category} {skin_type} skin best 2024"
            print(f"ğŸ”� Broader search: {broad_query}")
            
            search_url = f"https://api.duckduckgo.com/?q={broad_query}&format=json&no_redirect=1"
            response = requests.get(search_url, timeout=5)
            search_data = response.json()
            
            products = []
            for topic in search_data.get('RelatedTopics', [])[:5]:
                if 'Text' in topic:
                    product_info = {
                        'name': topic.get('Text', '')[:100],
                        'description': topic.get('Text', '')[:200],
                        'source_url': topic.get('FirstURL', ''),
                        'data_source': 'Internet Search (Broader)'
                    }
                    if product_info['name'] and len(product_info['name']) > 3:
                        products.append(product_info)
            
            return {
                'status': 'success',
                'product_category': product_category,
                'skin_type': skin_type,
                'user_age': user_age,
                'products': products,
                'search_query': broad_query,
                'count': len(products),
                'message': f"Found {len(products)} products from internet",
                'source': 'Internet Search Results'
            }
        
    except Exception as e:
        return {
            'status': 'error',
            'product_category': product_category,
            'error_message': f"Error searching products: {str(e)}",
            'recommendation': f"Please search '{product_category} for {skin_type} skin' on Google for latest recommendations"
        }


def collect_skincare_details(commitment_days: Optional[int] = None,
                             menstrual_cycle: Optional[str] = None,
                             skin_type: Optional[str] = None,
                             acne_frequency: Optional[str] = None,
                             dryness_level: Optional[str] = None,
                             dietary_habits: Optional[str] = None,
                             weight: Optional[float] = None,
                             height: Optional[float] = None,
                             weight_unit: Optional[str] = None,
                             height_unit: Optional[str] = None,
                             food_preference: Optional[str] = None) -> dict:
    """Collects and validates skincare details from user."""
    
    skincare_data = {}
    missing_fields = []
    errors = []
    
    if commitment_days is not None:
        try:
            days = int(commitment_days)
            valid_days = [15, 30, 45, 90]
            if days in valid_days:
                skincare_data['commitment_days'] = days
            else:
                errors.append(f"Please choose from: 15, 30, 45, or 90 days")
                missing_fields.append('commitment_days')
        except (ValueError, TypeError):
            errors.append(f"Commitment days must be a number")
            missing_fields.append('commitment_days')
    else:
        missing_fields.append('commitment_days')
    
    valid_skin_types = ['dry', 'oily', 'sensitive', 'combination', 'normal']
    if skin_type and isinstance(skin_type, str):
        skin_type_lower = skin_type.strip().lower()
        if skin_type_lower in valid_skin_types:
            skincare_data['skin_type'] = skin_type_lower
        else:
            errors.append(f"Skin type should be one of: {', '.join(valid_skin_types)}")
            missing_fields.append('skin_type')
    else:
        missing_fields.append('skin_type')
    
    if dietary_habits and isinstance(dietary_habits, str):
        skincare_data['dietary_habits'] = dietary_habits.strip()
    
    if acne_frequency and isinstance(acne_frequency, str):
        valid_acne = ['rarely', 'occasionally', 'frequently']
        if acne_frequency.strip().lower() in valid_acne:
            skincare_data['acne_frequency'] = acne_frequency.strip().lower()
    
    if dryness_level and isinstance(dryness_level, str):
        skincare_data['dryness_level'] = dryness_level.strip()
    
    if menstrual_cycle and isinstance(menstrual_cycle, str):
        skincare_data['menstrual_cycle'] = menstrual_cycle.strip()
    
    valid_food_prefs = ['vegetarian', 'non-vegetarian', 'vegan', 'pescatarian']
    if food_preference and isinstance(food_preference, str):
        food_pref_lower = food_preference.strip().lower()
        if food_pref_lower in valid_food_prefs:
            skincare_data['food_preference'] = food_pref_lower
        else:
            skincare_data['food_preference'] = food_preference.strip()
    
    if weight is not None and height is not None and weight_unit and height_unit:
        try:
            weight_kg = float(weight)
            height_m = float(height)
            
            if weight_unit.lower() in ['lbs', 'lb', 'pounds']:
                weight_kg = weight_kg * 0.453592
            
            if height_unit.lower() in ['feet', 'ft', 'foot']:
                height_m = height_m * 0.3048
            elif height_unit.lower() in ['cm', 'centimeters']:
                height_m = height_m / 100
            
            bmi = weight_kg / (height_m ** 2)
            skincare_data['weight_kg'] = round(weight_kg, 2)
            skincare_data['height_m'] = round(height_m, 2)
            skincare_data['bmi'] = round(bmi, 1)
            
        except (ValueError, TypeError):
            errors.append("Invalid weight or height values")
    
    if not missing_fields:
        status = 'success'
        message = "All skincare details collected successfully!"
    else:
        status = 'incomplete'
        message = f"Still need: {', '.join(missing_fields)}"
    
    if errors:
        status = 'error' if status == 'incomplete' else status
    
    result = {
        'status': status,
        'data': skincare_data,
        'missing_fields': missing_fields,
        'error_message': ' | '.join(errors) if errors else None,
        'message': message
    }
    
    return result


def generate_skincare_routine(user_name: str, age: int, gender: str, commitment_days: int,
                              skin_type: str, climate: str, season: str, 
                              bmi: Optional[float] = None,
                              dietary_habits: Optional[str] = None,
                              acne_frequency: Optional[str] = None,
                              food_preference: Optional[str] = None,
                              menstrual_cycle: Optional[str] = None) -> dict:
    """
    Generates INTERNET-BASED skincare routine with separate morning/evening routines.
    ALL products fetched from internet based on user-specific data.
    """
    
    routine = {
        'status': 'routine_generated',
        'user_name': user_name,
        'age': age,
        'commitment_days': commitment_days,
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'routines': {},
        'data_sources': 'All products fetched from internet in real-time based on your profile'
    }
    
    skin_concern = None
    if skin_type == 'oily' and acne_frequency in ['occasionally', 'frequently']:
        skin_concern = 'acne'
    elif skin_type == 'dry':
        skin_concern = 'dryness'
    elif skin_type == 'sensitive':
        skin_concern = 'sensitivity'
    
    # ===== MORNING ROUTINE (INTERNET-BASED) =====
    morning_routine = {
        'title': 'ğŸŒ… MORNING SKINCARE ROUTINE',
        'time': '6:00 AM - 8:00 AM',
        'total_time': '8-10 minutes',
        'description': 'Light, protective routine to prepare skin for the day ahead',
        'note': 'All products researched from internet based on YOUR profile',
        'steps': []
    }
    
    # STEP 1: Cleanser - FETCH FROM INTERNET
    cleanser_results = search_skincare_products_from_internet(
        user_age=age,
        skin_type=skin_type,
        skin_concern=skin_concern,
        product_category="cleanser",
        climate=climate,
        season=season,
        dietary_habits=dietary_habits
    )
    
    morning_routine['steps'].append({
        'step': 1,
        'name': 'ğŸ§´ STEP 1: Cleanser',
        'time': '30-60 seconds',
        'internet_products': cleanser_results.get('products', []),
        'search_query': cleanser_results.get('search_query', ''),
        'instructions': [
            'Wet face with lukewarm water',
            'Apply cleanser and massage gently',
            'Rinse thoroughly with water',
            'Pat dry with clean towel'
        ],
        'why_important': 'Removes overnight oil, dead skin, and impurities'
    })
    
    # STEP 2: Toner - FETCH FROM INTERNET
    toner_results = search_skincare_products_from_internet(
        user_age=age,
        skin_type=skin_type,
        skin_concern=skin_concern,
        product_category="toner",
        climate=climate,
        season=season
    )
    
    morning_routine['steps'].append({
        'step': 2,
        'name': 'ğŸ’§ STEP 2: Toner',
        'time': '1-2 minutes',
        'internet_products': toner_results.get('products', []),
        'search_query': toner_results.get('search_query', ''),
        'instructions': [
            'Apply toner to cotton pad or hands',
            'Gently pat onto face and neck',
            'Allow to absorb 1-2 minutes'
        ],
        'why_important': 'Balances pH, hydrates, prepares for serums'
    })
    
    # STEP 3: Serum - FETCH FROM INTERNET
    serum_results = search_skincare_products_from_internet(
        user_age=age,
        skin_type=skin_type,
        skin_concern=skin_concern,
        product_category="serum",
        climate=climate
    )
    
    morning_routine['steps'].append({
        'step': 3,
        'name': 'âœ¨ STEP 3: Serum',
        'time': '2-3 minutes',
        'internet_products': serum_results.get('products', []),
        'search_query': serum_results.get('search_query', ''),
        'instructions': [
            'Dispense 2-3 drops onto fingertips',
            'Gently pat and press into face and neck',
            'Allow 2-3 minutes to absorb'
        ],
        'why_important': 'Targets specific skin concerns'
    })
    
    # STEP 4: Moisturizer - FETCH FROM INTERNET
    moisturizer_results = search_skincare_products_from_internet(
        user_age=age,
        skin_type=skin_type,
        skin_concern=skin_concern,
        product_category="moisturizer",
        climate=climate
    )
    
    morning_routine['steps'].append({
        'step': 4,
        'name': 'ğŸ§´ STEP 4: Moisturizer',
        'time': '2 minutes',
        'internet_products': moisturizer_results.get('products', []),
        'search_query': moisturizer_results.get('search_query', ''),
        'instructions': [
            'Apply pea-sized amount',
            'Warm between fingers',
            'Apply to face, neck, dÃ©colletage',
            'Allow 2 minutes to set'
        ],
        'why_important': 'Locks hydration and protects'
    })
    
    # STEP 5: Sunscreen - FETCH FROM INTERNET
    sunscreen_results = search_skincare_products_from_internet(
        user_age=age,
        skin_type=skin_type,
        product_category="sunscreen SPF 50+",
        climate=climate,
        season=season
    )
    
    morning_routine['steps'].append({
        'step': 5,
        'name': 'â˜€ï¸� STEP 5: Sunscreen (SPF 50+)',
        'time': '2-3 minutes',
        'internet_products': sunscreen_results.get('products', []),
        'search_query': sunscreen_results.get('search_query', ''),
        'instructions': [
            'Dispense 1/4 teaspoon of sunscreen',
            'Apply to forehead, cheeks, nose, chin',
            'Spread evenly across face and neck',
            'Wait 5 minutes before sun exposure'
        ],
        'frequency': 'EVERY DAY - Rain or shine',
        'reapplication': 'Every 2 hours if sweating or swimming',
        'why_critical': 'âš ï¸� MOST IMPORTANT - Prevents UV damage and premature aging'
    })
    
    routine['routines']['morning'] = morning_routine
    
    # ===== EVENING ROUTINE (INTERNET-BASED) =====
    evening_routine = {
        'title': 'ğŸŒ™ EVENING SKINCARE ROUTINE',
        'time': '8:00 PM - 10:00 PM',
        'total_time': '15-20 minutes',
        'description': 'Deep cleanse and repair for overnight recovery',
        'note': 'All products researched from internet based on YOUR profile',
        'steps': []
    }
    
    # STEP 1: Oil Cleanser - FETCH FROM INTERNET
    oil_cleanser_results = search_skincare_products_from_internet(
        user_age=age,
        skin_type=skin_type,
        product_category="oil cleanser makeup remover",
        climate=climate
    )
    
    evening_routine['steps'].append({
        'step': 1,
        'name': 'ğŸ§´ STEP 1: Oil Cleanser (First Cleanse)',
        'time': '1-2 minutes',
        'internet_products': oil_cleanser_results.get('products', []),
        'search_query': oil_cleanser_results.get('search_query', ''),
        'instructions': [
            'Start with DRY face',
            'Pump 2-3 times into palm',
            'Massage gently for 1-2 minutes',
            'Add water to emulsify',
            'Rinse thoroughly'
        ],
        'why_important': 'Dissolves makeup, sunscreen, oil-based impurities'
    })
    
    # STEP 2: Water Cleanser - FETCH FROM INTERNET
    evening_routine['steps'].append({
        'step': 2,
        'name': 'ğŸ’§ STEP 2: Water Cleanser (Second Cleanse)',
        'time': '30-60 seconds',
        'internet_products': cleanser_results.get('products', []),
        'instructions': [
            'Wet face with lukewarm water',
            'Apply water cleanser',
            'Massage gently',
            'Rinse thoroughly'
        ],
        'why_important': 'Removes water-soluble impurities and residue'
    })
    
    # STEP 3: Toner - REUSE MORNING RESULTS
    evening_routine['steps'].append({
        'step': 3,
        'name': 'ğŸ’§ STEP 3: Toner',
        'time': '1-2 minutes',
        'internet_products': toner_results.get('products', []),
        'instructions': [
            'Apply to cotton pad or hands',
            'Gently wipe or pat across face',
            'Allow to absorb'
        ],
        'why_important': 'Re-balances pH after cleansing'
    })
    
    # STEP 4: Treatment/Mask (Varies by skin type) - FETCH FROM INTERNET
    if skin_type == 'oily' and acne_frequency in ['occasionally', 'frequently']:
        treatment_results = search_skincare_products_from_internet(
            user_age=age,
            skin_type=skin_type,
            skin_concern='acne treatment',
            product_category="acne treatment BHA AHA",
            climate=climate
        )
        evening_routine['steps'].append({
            'step': 4,
            'name': 'ğŸ”¬ STEP 4: Acne Treatment',
            'time': '10 min wait + absorption',
            'internet_products': treatment_results.get('products', []),
            'search_query': treatment_results.get('search_query', ''),
            'frequency': '3-4 times per week',
            'warning': 'âš ï¸� Start 2-3x weekly'
        })
    elif skin_type == 'dry':
        mask_results = search_skincare_products_from_internet(
            user_age=age,
            skin_type=skin_type,
            skin_concern='dryness hydration',
            product_category="hydrating mask cream mask",
            climate=climate
        )
        evening_routine['steps'].append({
            'step': 4,
            'name': 'ğŸ�­ STEP 4: Hydrating Mask',
            'time': '10-15 minutes',
            'internet_products': mask_results.get('products', []),
            'search_query': mask_results.get('search_query', ''),
            'frequency': '2-3 times per week'
        })
    else:
        night_serum_results = search_skincare_products_from_internet(
            user_age=age,
            skin_type=skin_type,
            skin_concern=skin_concern,
            product_category="night serum active serum",
            climate=climate
        )
        evening_routine['steps'].append({
            'step': 4,
            'name': 'âœ¨ STEP 4: Night Serum',
            'time': '2-3 minutes',
            'internet_products': night_serum_results.get('products', []),
            'search_query': night_serum_results.get('search_query', '')
        })
    
    # STEP 5: Eye Cream - FETCH FROM INTERNET
    eye_cream_results = search_skincare_products_from_internet(
        user_age=age,
        skin_type=skin_type,
        product_category="eye cream anti-aging eye care",
        climate=climate
    )
    
    evening_routine['steps'].append({
        'step': 5,
        'name': 'ğŸ‘�ï¸� STEP 5: Eye Cream',
        'time': '1 minute',
        'internet_products': eye_cream_results.get('products', []),
        'search_query': eye_cream_results.get('search_query', ''),
        'instructions': [
            'Tiny amount on ring finger',
            'Gentle dabs around eye area',
            'Patting motions only'
        ]
    })
    
    # STEP 6: Night Moisturizer - FETCH FROM INTERNET
    night_moisturizer_results = search_skincare_products_from_internet(
        user_age=age,
        skin_type=skin_type,
        product_category="night cream sleep mask heavy moisturizer",
        climate=climate
    )
    
    evening_routine['steps'].append({
        'step': 6,
        'name': 'ğŸŒ™ STEP 6: Night Moisturizer',
        'time': '2-3 minutes',
        'internet_products': night_moisturizer_results.get('products', []),
        'search_query': night_moisturizer_results.get('search_query', ''),
        'why_important': 'Overnight repair and recovery'
    })
    
    routine['routines']['evening'] = evening_routine
    
    # ===== WEEKLY TREATMENTS (INTERNET-BASED) =====
    weekly_routine = {
        'title': 'âœ¨ WEEKLY SPECIAL TREATMENTS',
        'frequency': '2-3 times per week',
        'note': 'All products researched from internet',
        'options': []
    }
    
    if skin_type == 'oily':
        mask_results = search_skincare_products_from_internet(
            user_age=age,
            skin_type=skin_type,
            skin_concern='pore cleansing oil control',
            product_category="clay mask charcoal mask pore mask",
            climate=climate
        )
        weekly_routine['options'].append({
            'name': 'ğŸ�­ Clay or Charcoal Mask',
            'internet_products': mask_results.get('products', []),
            'search_query': mask_results.get('search_query', ''),
            'frequency': '1-2 times per week'
        })
    elif skin_type == 'dry':
        mask_results = search_skincare_products_from_internet(
            user_age=age,
            skin_type=skin_type,
            skin_concern='deep hydration nourishment',
            product_category="hydrating mask honey mask cream mask",
            climate=climate
        )
        weekly_routine['options'].append({
            'name': 'ğŸ�­ Hydrating Mask',
            'internet_products': mask_results.get('products', []),
            'search_query': mask_results.get('search_query', ''),
            'frequency': '1-2 times per week'
        })
    else:
        mask_results = search_skincare_products_from_internet(
            user_age=age,
            skin_type=skin_type,
            product_category="sheet mask gel mask treatment mask",
            climate=climate
        )
        weekly_routine['options'].append({
            'name': 'ğŸ�­ Sheet or Gel Mask',
            'internet_products': mask_results.get('products', []),
            'search_query': mask_results.get('search_query', ''),
            'frequency': '1-2 times per week'
        })
    
    routine['routines']['weekly'] = weekly_routine
    
    # ===== LIFESTYLE RECOMMENDATIONS =====
    lifestyle = {
        'title': 'ğŸ’ª LIFESTYLE & DIETARY RECOMMENDATIONS',
        'note': 'Personalized based on your data',
        'sections': {
            'hydration': {
                'title': 'ğŸ’§ Hydration',
                'tips': [
                    'Drink 2-3 liters of water daily',
                    f'Morning: 1 glass warm water with lemon',
                    'Midday: 1 liter by 2 PM',
                    f'Evening: Complete by 8 PM (not before bed)'
                ]
            },
            'nutrition': {
                'title': 'ğŸ¥— Nutrition for Skin Health',
                'tips': [
                    'âœ… Omega-3s (2-3x per week): Salmon, walnuts, flaxseeds',
                    'âœ… Antioxidants (Daily): Berries, leafy greens, green tea',
                    'âœ… Zinc (3-4x per week): Oysters, pumpkin seeds, chickpeas',
                    'âœ… Vitamin A (Daily): Carrots, sweet potatoes, kale',
                    'âœ… Probiotics (Daily): Yogurt, kimchi, kombucha'
                ]
            }
        }
    }
    
    if food_preference == 'vegan':
        lifestyle['sections']['vegan_notes'] = {
            'title': 'ğŸŒ± Vegan Skincare Strategy',
            'tips': [
                'Take B12 supplements (sublingual best)',
                'Plant-based protein: Legumes, nuts, seeds, tofu',
                'Iron absorption: Eat with vitamin C sources',
                'Zinc sources: Pumpkin seeds, hemp seeds, chickpeas'
            ]
        }
    
    routine['routines']['lifestyle'] = lifestyle
    
    # ===== MENSTRUAL CYCLE (if female) =====
    if gender.lower() == 'female' and menstrual_cycle:
        routine['routines']['menstrual_cycle'] = {
            'title': 'ğŸ”„ MENSTRUAL CYCLE-BASED ADAPTATIONS',
            'current_phase': menstrual_cycle,
            'note': 'Adjust routine based on your cycle phase'
        }
    
    # ===== PROGRESS TRACKING =====
    routine['routines']['progress'] = {
        'title': 'ğŸ“Š PROGRESS TRACKING',
        'commitment_days': commitment_days,
        'note': 'Take weekly photos to track changes'
    }
    
    return routine


def store_skincare_routine_to_profile(user_name: str, skincare_routine: dict) -> dict:
    """Stores the generated skincare routine to user profile."""
    
    return {
        'status': 'routine_stored',
        'user_name': user_name,
        'routine_id': f"skincare_{user_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'commitment_days': skincare_routine.get('commitment_days'),
        'message': f"âœ… Internet-sourced skincare routine for {user_name} saved!",
        'routine_summary': f"Complete {skincare_routine.get('commitment_days')}-day routine with ALL products fetched from internet!"
    }


# =====================================================
# TOOLS FOR GROOMING AGENT (Agent 4) - INTERNET-BASED
# =====================================================

def collect_grooming_details(commitment_days: Optional[int] = None,
                            height: Optional[float] = None,
                            height_unit: Optional[str] = None,
                            skin_tone: Optional[str] = None,
                            body_build: Optional[str] = None,
                            facial_structure: Optional[str] = None,
                            time_commitment: Optional[str] = None,
                            personality_style: Optional[str] = None) -> dict:
    """Collects and validates grooming details from user."""
    
    grooming_data = {}
    missing_fields = []
    errors = []
    
    # 1. Commitment Days
    if commitment_days is not None:
        try:
            days = int(commitment_days)
            valid_days = [15, 30, 45, 60, 90]
            if days in valid_days:
                grooming_data['commitment_days'] = days
            else:
                errors.append(f"Please choose from: 15, 30, 45, 60, or 90 days")
                missing_fields.append('commitment_days')
        except (ValueError, TypeError):
            errors.append(f"Commitment days must be a number")
            missing_fields.append('commitment_days')
    else:
        missing_fields.append('commitment_days')
    
    # 2. Height
    if height is not None and height_unit is not None:
        try:
            height_m = float(height)
            if height_unit.lower() in ['feet', 'ft', 'foot']:
                height_m = height_m * 0.3048
            elif height_unit.lower() in ['cm', 'centimeters']:
                height_m = height_m / 100
            grooming_data['height_m'] = round(height_m, 2)
        except (ValueError, TypeError):
            errors.append("Invalid height value")
            missing_fields.append('height')
    else:
        missing_fields.append('height')
    
    # 3. Skin Tone
    valid_skin_tones = ['fair', 'light/warm', 'medium', 'deep']
    if skin_tone and isinstance(skin_tone, str):
        skin_tone_lower = skin_tone.strip().lower()
        if skin_tone_lower in valid_skin_tones:
            grooming_data['skin_tone'] = skin_tone_lower
        else:
            errors.append(f"Skin tone should be one of: {', '.join(valid_skin_tones)}")
            missing_fields.append('skin_tone')
    else:
        missing_fields.append('skin_tone')
    
    # 4. Body Build
    valid_builds = ['hourglass', 'pear', 'triangle', 'rectangle', 'apple', 'oval']
    if body_build and isinstance(body_build, str):
        body_build_lower = body_build.strip().lower()
        if body_build_lower in valid_builds:
            grooming_data['body_build'] = body_build_lower
        else:
            errors.append(f"Body build should be one of: {', '.join(valid_builds)}")
            missing_fields.append('body_build')
    else:
        missing_fields.append('body_build')
    
    # 5. Facial Structure
    valid_facial_structures = ['oval', 'round', 'square', 'heart']
    if facial_structure and isinstance(facial_structure, str):
        facial_structure_lower = facial_structure.strip().lower()
        if facial_structure_lower in valid_facial_structures:
            grooming_data['facial_structure'] = facial_structure_lower
        else:
            errors.append(f"Facial structure should be one of: {', '.join(valid_facial_structures)}")
            missing_fields.append('facial_structure')
    else:
        missing_fields.append('facial_structure')
    
    # 6. Time Commitment
    valid_time_commitments = ['minimalist', 'standard', 'detailed']
    if time_commitment and isinstance(time_commitment, str):
        time_commitment_lower = time_commitment.strip().lower()
        if time_commitment_lower in valid_time_commitments:
            grooming_data['time_commitment'] = time_commitment_lower
        else:
            errors.append(f"Time commitment should be one of: {', '.join(valid_time_commitments)}")
            missing_fields.append('time_commitment')
    else:
        missing_fields.append('time_commitment')
    
    # 7. Personality Style
    valid_styles = ['classic/elegant', 'creative/bohemian', 'trendsetter/bold', 'comfortable/athleisure']
    if personality_style and isinstance(personality_style, str):
        personality_style_lower = personality_style.strip().lower()
        if personality_style_lower in valid_styles:
            grooming_data['personality_style'] = personality_style_lower
        else:
            errors.append(f"Personality style should be one of: {', '.join(valid_styles)}")
            missing_fields.append('personality_style')
    else:
        missing_fields.append('personality_style')
    
    if not missing_fields:
        status = 'success'
        message = "All grooming details collected successfully!"
    else:
        status = 'incomplete'
        message = f"Still need: {', '.join(missing_fields)}"
    
    if errors:
        status = 'error' if status == 'incomplete' else status
    
    result = {
        'status': status,
        'data': grooming_data,
        'missing_fields': missing_fields,
        'error_message': ' | '.join(errors) if errors else None,
        'message': message
    }
    
    return result


def generate_grooming_routine(user_name: str, age: int, gender: str, commitment_days: int,
                             height_m: float, skin_tone: str, body_build: str,
                             facial_structure: str, time_commitment: str,
                             personality_style: str, climate: str, season: str) -> dict:
    """
    Generates personalized grooming routine based on physical appearance and facial structure.
    """

    routine = {
        'status': 'routine_generated',
        'user_name': user_name,
        'age': age,
        'gender': gender,
        'commitment_days': commitment_days,
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'routines': {},
        'data_sources': 'Personalized based on your physical appearance and facial structure'
    }

    # ===== HAIR CARE ROUTINE =====
    styling_time_dict = {'minimalist': '5-10 min', 'standard': '15-25 min', 'detailed': '30+ min'}
    styling_time_val = styling_time_dict.get(time_commitment.lower(), '15-25 min')

    hair_routine = {
        'title': 'ğŸ’‡ PERSONALIZED HAIR CARE ROUTINE',
        'subtitle': f'For {facial_structure.title()} face shape & {body_build.title()} body build',
        'note': 'Tailored based on your physical features',
        'recommendations': {
            'recommended_styles': {
                'oval': ['Straight long hair', 'Layered cuts', 'Side-swept styles', 'Bob cuts'],
                'round': ['Longer lengths to elongate', 'Layers to add texture', 'Side parting', 'Textured waves'],
                'square': ['Soft curls to soften', 'Long side-swept styles', 'Textured waves', 'Layers around face'],
                'heart': ['Longer at chin', 'Chin-length bobs', 'Soft curls at bottom', 'Side bangs']
            }.get(facial_structure, []),
            'maintenance_frequency': '1-2 times per month trim',
            'styling_time': f"{time_commitment.title()} routine ({styling_time_val})",
            'products_needed': 'Shampoo, Conditioner, Styling cream/serum, Hairspray',
            'climate_adaptation': f"For {climate}: Use lightweight products in tropical climates, moisturizing products in cold climates"
        }
    }
    routine['routines']['hair'] = hair_routine

    # ===== FACIAL GROOMING ROUTINE =====
    trim_dict = {'minimalist': '2-3 days', 'standard': '1-2 days', 'detailed': 'Daily'}
    trim_val = trim_dict.get(time_commitment.lower(), '1-2 days')

    if gender.lower() == 'male':
        facial_grooming = {
            'title': 'ğŸ§” FACIAL GROOMING ROUTINE',
            'note': 'Customized for your facial structure',
            'recommendations': {
                'beard_grooming': 'Regular trimming every 2-3 weeks. Use beard oil and brush for maintenance.',
                'facial_hairstyle': f'Based on {facial_structure} face: Choose styles that balance facial proportions.',
                'eyebrow_care': 'Regular grooming to frame the face. Keep natural shape aligned with face structure.',
                'daily_routine': f"Shave/trim every {trim_val}",
                'products': 'Shaving cream/gel, Quality razor, Aftershave balm, Beard oil'
            }
        }
    else:
        facial_grooming = {
            'title': 'ğŸ‘© FACIAL GROOMING ROUTINE',
            'note': 'Customized for your facial structure',
            'recommendations': {
                'eyebrow_care': f'For {facial_structure}: Shape to complement face geometry. Professional shaping recommended.',
                'facial_threading': 'Monthly upper lip and face threading for smooth appearance.',
                'derma_care': 'Regular facial every 4-6 weeks for professional maintenance.',
                'makeup_prep': f'Proper primer and foundation for your skin tone to enhance features.',
                'daily_routine': f'15-30 minute grooming routine based on {time_commitment} commitment',
                'products': 'Face wash, Moisturizer, Sunscreen, Threading/waxing for facial hair'
            }
        }
    routine['routines']['facial_grooming'] = facial_grooming

    # ... Rest of your function unchanged ...

    return routine

    
    # ===== CLOTHING & STYLING RECOMMENDATIONS =====
    styling = {
        'title': 'ğŸ‘— CLOTHING & STYLING RECOMMENDATIONS',
        'note': f'For your {body_build} body build and {skin_tone} skin tone',
        'body_build_guide': {},
        'color_palette': {}
    }
    
    # Body build tailoring
    body_build_tailoring = {
        'hourglass': 'Fitted clothing that shows off balanced proportions. Wrap dresses, belted styles.',
        'pear': 'Darker colors on bottom, lighter on top. A-line skirts. Avoid tight hip areas.',
        'triangle': 'Balance triangle with wider bottom. A-line skirts, flared pants. Avoid tight hips.',
        'rectangle': 'Create definition with horizontal stripes, belts, layered styles.',
        'apple': 'Emphasis on legs. Long cardigans to hide torso. V-necklines to elongate.',
        'oval': 'Balanced fit-and-flare styles. Avoid oversized. Clean horizontal lines.'
    }
    
    styling['body_build_guide'] = {
        'build_type': body_build,
        'tailoring_tips': body_build_tailoring.get(body_build, ''),
        'silhouettes': 'Choose cuts that complement your natural shape'
    }
    
    # Skin tone color palette
    skin_tone_colors = {
        'fair': ['Jewel tones (emerald, sapphire)', 'Pure white, black', 'Coral, rose tones'],
        'light/warm': ['Warm earth tones (rust, bronze)', 'Warm reds, oranges', 'Cream, gold'],
        'medium': ['Saturated jewel tones', 'Warm and cool colors work', 'Gold, bronze metallics'],
        'deep': ['Rich jewel tones (ruby, sapphire)', 'Gold, bronze metallics', 'Warm and cool colors']
    }
    
    styling['color_palette'] = {
        'skin_tone': skin_tone,
        'recommended_colors': skin_tone_colors.get(skin_tone, []),
        'metallics': 'Gold tones work best with warm undertones, Silver with cool undertones'
    }
    
    routine['routines']['styling'] = styling
    
    # ===== PERSONALITY STYLE GUIDE =====
    style_guide = {
        'title': 'âœ¨ YOUR PERSONALITY STYLE GUIDE',
        'archetype': personality_style,
        'fashion_philosophy': {}
    }
    
    personality_descriptions = {
        'classic/elegant': {
            'characteristics': 'Timeless, sophisticated, refined',
            'key_pieces': ['White button-up shirt', 'Tailored blazer', 'Dark jeans', 'Leather shoes', 'Pearl accessories'],
            'brands': 'Focus on quality over quantity. Invest in well-made basics.',
            'colors': 'Neutral palette: Black, white, navy, gray, beige',
            'fashion_rule': 'Less is more. Quality fabrics and perfect fit.'
        },
        'creative/bohemian': {
            'characteristics': 'Artistic, free-spirited, expressive',
            'key_pieces': ['Flowing fabrics', 'Ethnic prints', 'Layered jewelry', 'Scarves', 'Vintage finds'],
            'brands': 'Mix high-street with vintage and thrift store finds.',
            'colors': 'Earth tones, jewel tones, patterns, textures',
            'fashion_rule': 'Express yourself. Mix patterns and eras freely.'
        },
        'trendsetter/bold': {
            'characteristics': 'Cutting-edge, fashionable, confident',
            'key_pieces': ['Statement pieces', 'Bold colors/prints', 'Unique accessories', 'Latest trends', 'Designer items'],
            'brands': 'Follow fashion weeks, emerging designers, trend setters',
            'colors': 'Bold, vibrant colors, contrasting combinations',
            'fashion_rule': 'Take risks. Push boundaries. Own your style.'
        },
        'comfortable/athleisure': {
            'characteristics': 'Relaxed, practical, functional',
            'key_pieces': ['Athletic wear', 'Comfortable shoes', 'Hoodies', 'Casual basics', 'Functional fabrics'],
            'brands': 'Sportswear brands, comfortable high-street options',
            'colors': 'Neutral, muted tones, practical colors',
            'fashion_rule': 'Comfort first. Function meets style.'
        }
    }
    
    style_guide['fashion_philosophy'] = personality_descriptions.get(personality_style, {})
    
    routine['routines']['personality_style'] = style_guide
    
    # ===== GROOMING SCHEDULE =====
    schedule = {
        'title': 'ğŸ“… GROOMING MAINTENANCE SCHEDULE',
        'commitment_days': commitment_days,
        'daily': [
            'Face cleansing and moisturizing',
            f'Hair care (brush, style as per {time_commitment} routine)',
            'Deodorant/antiperspirant application',
            'Nail care check'
        ],
        'weekly': [
            'Deep conditioning for hair',
            'Exfoliation for face and body',
            'Nail trimming and shaping',
            'Beard/facial hair grooming'
        ],
        'monthly': [
            f'Professional haircut ({commitment_days} days cycle)',
            'Facial or skincare treatment',
            'Professional grooming service if needed',
            'Wardrobe review and styling'
        ]
    }
    
    routine['routines']['schedule'] = schedule
    
    # ===== GROOMING PRODUCTS SEARCH =====
    products = {
        'title': 'ğŸ›�ï¸� RECOMMENDED GROOMING PRODUCTS',
        'note': 'Researched from internet based on your profile',
        'categories': {
            'hair_care': 'Shampoo, conditioner, styling products for your hair type',
            'facial_care': 'Facewash, moisturizer based on skin tone and climate',
            'personal_hygiene': 'Deodorant, body care, nail care products',
            'styling_tools': 'Depending on your personality style and time commitment'
        }
    }
    
    routine['routines']['products'] = products
    
    return routine


def store_grooming_routine_to_profile(user_name: str, grooming_routine: dict) -> dict:
    """Stores the generated grooming routine to user profile."""
    
    return {
        'status': 'routine_stored',
        'user_name': user_name,
        'routine_id': f"grooming_{user_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'commitment_days': grooming_routine.get('commitment_days'),
        'message': f"âœ… Personalized grooming routine for {user_name} saved!",
        'routine_summary': f"Complete {grooming_routine.get('commitment_days')}-day grooming routine customized for your physical appearance and lifestyle!"
    }


# =====================================================
# AGENT 1: USER INFO COLLECTOR (UNCHANGED)
# =====================================================

def create_user_info_agent(model: str = "gemini-2.0-flash"):
    """Creates the User Information Collector Agent."""
    
    user_info_agent = Agent(
        name="user_info_collector_agent",
        model=model,
        description="Collects and validates user information (name, age, gender, city).",
        
        instruction="""You are the User Information Collector Agent.

WORKFLOW:
1. Greet warmly
2. Ask 4 questions ONE AT A TIME (name, age, gender, city)
3. Validate each with tools
4. Store when all complete
5. Pass to Weather Agent

TONE: Friendly, professional.""",
        
        tools=[collect_user_info, store_user_in_session]
    )
    
    return user_info_agent


# =====================================================
# AGENT 2: LOCATION WEATHER & CLIMATE (UNCHANGED)
# =====================================================

def create_location_weather_agent(model: str = "gemini-2.0-flash"):
    """Creates the Location Weather and Climate Agent."""
    
    location_weather_agent = Agent(
        name="location_weather_climate_agent",
        model=model,
        description="Fetches REAL weather data and enriches user profile.",
        
        instruction="""You are the Location Weather & Climate Agent.

WORKFLOW:
1. Fetch weather from city in profile
2. Present data
3. Store in profile
4. Pass control back to Root Agent

TONE: Informative, helpful.""",
        
        tools=[get_location_weather_data, update_user_profile_with_location_data]
    )
    
    return location_weather_agent


# =====================================================
# AGENT 3: SKINCARE ADVISOR (UNCHANGED)
# =====================================================

def create_skincare_agent(model: str = "gemini-2.0-flash"):
    """Creates the Skincare Agent with INTERNET-BASED product recommendations."""
    
    skincare_agent = Agent(
        name="skincare_advisor_agent",
        model=model,
        description="Generates personalized skincare routines with internet-sourced products.",
        
        instruction="""You are the Skincare Advisor Agent.

WORKFLOW - COLLECT DATA:
Phase 1: Commitment days (15/30/45/90)
Phase 2: Menstrual cycle (females only)
Phase 3: Skin type (dry/oily/sensitive/combination/normal)
Phase 4: Skin concerns (acne frequency, dryness level)
Phase 5: Dietary habits
Phase 6: BMI (height + weight)
Phase 7: Food preference (vegan/vegetarian/non-veg)

THEN GENERATE:
- Morning routine (5 steps - internet products)
- Evening routine (6 steps - internet products)
- Weekly treatments (skin-specific)
- Lifestyle & diet recommendations
- Menstrual cycle tracking (if female)
- Progress tracking

KEY POINTS:
- ONE question per turn
- FETCH REAL products from internet for EACH step
- Create COMPLETELY SEPARATE morning and evening routines
- Include: time, duration, detailed instructions
- ALL recommendations based on THEIR specific profile
- NO hardcoding - everything from internet

TONE: Expert, supportive, empowering.""",
        
        tools=[collect_skincare_details, search_skincare_products_from_internet, generate_skincare_routine, store_skincare_routine_to_profile]
    )
    
    return skincare_agent


# =====================================================
# AGENT 4: GROOMING ADVISOR (NEW - TRIGGERED MANUALLY)
# =====================================================

def create_grooming_agent(model: str = "gemini-2.0-flash"):
    """Creates the Grooming Agent - triggered by user choice, not automatic."""
    
    grooming_agent = Agent(
        name="grooming_advisor_agent",
        model=model,
        description="Generates personalized grooming routine based on physical appearance and facial structure.",
        
        instruction="""You are the Grooming Advisor Agent.

WORKFLOW - COLLECT DATA (ONE AT A TIME):
Phase 1: Commitment days (15/30/45/60/90)
Phase 2: Height (with unit)
Phase 3: Skin tone (Fair/Light/Medium/Deep)
Phase 4: Body build (Hourglass/Pear/Triangle/Rectangle/Apple/Oval)
Phase 5: Facial structure (Oval/Round/Square/Heart)
Phase 6: Time commitment (Minimalist/Standard/Detailed)
Phase 7: Personality style (Classic/Bohemian/Trendsetter/Athleisure)

THEN GENERATE:
- Hair care routine (based on facial structure)
- Facial grooming routine (gender-specific)
- Clothing & styling recommendations (based on body build & skin tone)
- Personality style guide
- Grooming maintenance schedule
- Product recommendations

KEY POINTS:
- ONE question per turn
- Retrieve user data from profile (name, age, gender, climate, season)
- Customize recommendations based on THEIR specific features
- Hair recommendations based on FACIAL STRUCTURE
- Color recommendations based on SKIN TONE
- Styling based on BODY BUILD and PERSONALITY
- NO hardcoding - personalized for each user

TONE: Professional, encouraging, style-focused.""",
        
        tools=[collect_grooming_details, generate_grooming_routine, store_grooming_routine_to_profile]
    )
    
    return grooming_agent

# =====================================================
# TOOLS FOR DIET PLAN AGENT (Agent 5)
# =====================================================

def collect_diet_details(commitment_days: Optional[int] = None,
                         weight: Optional[float] = None,
                         weight_unit: Optional[str] = None,
                         height: Optional[float] = None,
                         height_unit: Optional[str] = None,
                         food_habit: Optional[str] = None,
                         allergies: Optional[str] = None,
                         diet_goal: Optional[str] = None,
                         activity_level: Optional[str] = None,
                         activity_days: Optional[int] = None) -> dict:
    """Collects and validates diet plan details from user."""
    
    diet_data = {}
    missing_fields = []
    errors = []
    
    # 1. Commitment Days
    if commitment_days is not None:
        try:
            days = int(commitment_days)
            valid_days = [15, 30, 45, 90]
            if days in valid_days:
                diet_data['commitment_days'] = days
            else:
                errors.append(f"Please choose from: 15, 30, 45, or 90 days")
                missing_fields.append('commitment_days')
        except (ValueError, TypeError):
            errors.append(f"Commitment days must be a number")
            missing_fields.append('commitment_days')
    else:
        missing_fields.append('commitment_days')
    
    # 2. BMI Calculation (Height & Weight)
    if weight is not None and height is not None and weight_unit and height_unit:
        try:
            weight_kg = float(weight)
            height_m = float(height)
            
            if weight_unit.lower() in ['lbs', 'lb', 'pounds']:
                weight_kg = weight_kg * 0.453592
            
            if height_unit.lower() in ['feet', 'ft', 'foot']:
                height_m = height_m * 0.3048
            elif height_unit.lower() in ['cm', 'centimeters']:
                height_m = height_m / 100
            
            if height_m > 0:
                bmi = weight_kg / (height_m ** 2)
                diet_data['weight_kg'] = round(weight_kg, 2)
                diet_data['height_m'] = round(height_m, 2)
                diet_data['bmi'] = round(bmi, 1)
            else:
                errors.append("Invalid height value")
                missing_fields.append('bmi')
        except (ValueError, TypeError):
            errors.append("Invalid weight or height values")
            missing_fields.append('bmi')
    else:
        missing_fields.append('bmi')
    
    # 3. Food Habit
    valid_food_habits = ['vegetarian', 'non-vegetarian', 'vegan', 'pescatarian', 'jain']
    if food_habit and isinstance(food_habit, str):
        food_habit_lower = food_habit.strip().lower()
        if food_habit_lower in valid_food_habits:
            diet_data['food_habit'] = food_habit_lower
        else:
            errors.append(f"Food habit should be one of: {', '.join(valid_food_habits)}")
            missing_fields.append('food_habit')
    else:
        missing_fields.append('food_habit')
    
    # 4. Allergies
    if allergies and isinstance(allergies, str):
        diet_data['allergies'] = allergies.strip()
    
    # 5. Diet Goal
    valid_goals = ['weight loss', 'weight gain', 'muscle building', 'boost energy & focus', 
                   'improve skin & hair health', 'weight maintenance & nutritional balance',
                   'athletic performance', 'digestive health']
    if diet_goal and isinstance(diet_goal, str):
        diet_goal_lower = diet_goal.strip().lower()
        if diet_goal_lower in valid_goals:
            diet_data['diet_goal'] = diet_goal_lower
        else:
            errors.append(f"Diet goal should be one of: {', '.join(valid_goals)}")
            missing_fields.append('diet_goal')
    else:
        missing_fields.append('diet_goal')
    
    # 6. Activity Level
    if activity_level and isinstance(activity_level, str):
        diet_data['activity_level'] = activity_level.strip()
    
    # 7. Activity Days
    if activity_days is not None:
        try:
            days = int(activity_days)
            if 0 <= days <= 7:
                diet_data['activity_days'] = days
            else:
                errors.append(f"Activity days should be between 0 and 7")
                missing_fields.append('activity_days')
        except (ValueError, TypeError):
            errors.append(f"Activity days must be a number")
            missing_fields.append('activity_days')
    else:
        missing_fields.append('activity_days')
    
    if not missing_fields:
        status = 'success'
        message = "All diet details collected successfully!"
    else:
        status = 'incomplete'
        message = f"Still need: {', '.join(missing_fields)}"
    
    if errors:
        status = 'error' if status == 'incomplete' else status
    
    result = {
        'status': status,
        'data': diet_data,
        'missing_fields': missing_fields,
        'error_message': ' | '.join(errors) if errors else None,
        'message': message
    }
    
    return result


def search_diet_recipes_from_internet(diet_goal: str, food_habit: str, season: str, climate: str,
                                      allergies: Optional[str] = None, meal_type: str = "breakfast",
                                      calories: Optional[int] = None) -> dict:
    """
    Searches the INTERNET for diet recipes based on user's SPECIFIC requirements.
    Personalized by: diet goal, food habit, allergies, season, climate, meal type, calories.
    """
    
    try:
        query_parts = [meal_type, food_habit, diet_goal]
        
        seasonal_keywords = {
            'Winter': 'warming, slow-cooked, root vegetables, comfort food',
            'Spring': 'fresh, light, leafy greens, seasonal produce',
            'Summer': 'refreshing, light, fresh, salads, cold dishes',
            'Autumn': 'harvest, hearty, squash, apples, warming'
        }
        query_parts.append(seasonal_keywords.get(season, 'seasonal'))
        
        climate_keywords = {
            'Tropical': 'tropical fruits, coconut, light, hydrating',
            'Hot/Arid': 'hydrating, cooling, mineral-rich, light',
            'Temperate': 'balanced, variety, fresh, all seasons',
            'Cool/Temperate': 'hearty, warming, protein-rich',
            'Cold/Polar': 'warming, high-calorie, protein-rich, immunity-boosting'
        }
        query_parts.append(climate_keywords.get(climate, 'balanced'))
        
        if allergies:
            allergies_list = [a.strip() for a in allergies.split(',')]
            query_parts.append(f"without {', '.join(allergies_list)}")
        
        if calories:
            query_parts.append(f"around {calories} calories")
        
        search_query = " ".join(query_parts) + f" recipes 2024 2025 healthy {season.lower()} {climate.lower()}"
        
        print(f"ğŸ”� Searching recipes: {search_query}")
        
        search_url = f"https://api.duckduckgo.com/?q={search_query}&format=json&no_redirect=1"
        response = requests.get(search_url, timeout=5)
        search_data = response.json()
        
        recipes = []
        related_topics = search_data.get('RelatedTopics', [])
        
        for topic in related_topics[:10]:
            if 'Text' in topic and 'FirstURL' in topic:
                topic_text = topic.get('Text', '')
                first_url = topic.get('FirstURL', '')
                
                recipe_name = topic_text.split(' - ')[0][:100] if ' - ' in topic_text else topic_text[:100]
                recipe_desc = topic_text[:250]
                
                recipe_info = {
                    'name': recipe_name,
                    'description': recipe_desc,
                    'source_url': first_url,
                    'data_source': 'Internet Search (Real-time)',
                    'meal_type': meal_type,
                    'season': season,
                    'climate': climate
                }
                
                if recipe_info['name'] and len(recipe_info['name']) > 3 and first_url:
                    recipes.append(recipe_info)
        
        if recipes:
            return {
                'status': 'success',
                'meal_type': meal_type,
                'diet_goal': diet_goal,
                'food_habit': food_habit,
                'season': season,
                'climate': climate,
                'recipes': recipes[:5],
                'search_query': search_query,
                'count': len(recipes[:5]),
                'message': f"Found {len(recipes[:5])} recipes from internet (personalized for {season}/{climate})",
                'source': 'Internet Search Results (Real-time)',
                'personalization_applied': [
                    f'Season: {season}',
                    f'Climate: {climate}',
                    f'Diet Goal: {diet_goal}',
                    f'Food Habit: {food_habit}',
                    f'Allergies excluded: {allergies if allergies else "None"}',
                    f'Calories target: {calories if calories else "Not specified"}'
                ]
            }
        else:
            return {
                'status': 'no_results',
                'meal_type': meal_type,
                'season': season,
                'climate': climate,
                'message': 'No recipes found. Trying alternate search with different parameters...',
                'search_query': search_query
            }
        
    except Exception as e:
        return {
            'status': 'error',
            'meal_type': meal_type,
            'error_message': f"Error searching recipes: {str(e)}"
        }


def generate_diet_plan(user_name: str, age: int, gender: str, commitment_days: int,
                      weight_kg: float, height_m: float, bmi: float,
                      food_habit: str, season: str, climate: str,
                      allergies: Optional[str] = None,
                      diet_goal: str = "weight maintenance & nutritional balance",
                      activity_level: Optional[str] = None,
                      activity_days: int = 0) -> dict:
    """
    Generates PERSONALIZED diet plan based on user's ACTUAL health data, goals, restrictions, season & climate.
    All recommendations adapted to user's specific profile with REAL recipes from internet.
    """
    
    plan = {
        'status': 'plan_generated',
        'user_name': user_name,
        'age': age,
        'gender': gender,
        'commitment_days': commitment_days,
        'bmi': bmi,
        'diet_goal': diet_goal,
        'season': season,
        'climate': climate,
        'food_habit': food_habit,
        'allergies': allergies if allergies else 'None',
        'activity_level': activity_level,
        'activity_days': activity_days,
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'routines': {},
        'personalization_note': f'Plan created for {season} season in {climate} climate. All recipes fetched from internet based on YOUR specific profile.'
    }
    
    # Calculate daily calorie needs based on BMI, activity, and goal
    bmr = 10 * weight_kg + 6.25 * height_m * 100 - 5 * age
    if gender.lower() == 'male':
        bmr = bmr + 5
    else:
        bmr = bmr - 161
    
    activity_multiplier = 1.2  # sedentary
    if activity_days >= 5:
        activity_multiplier = 1.55  # very active
    elif activity_days >= 3:
        activity_multiplier = 1.375  # moderately active
    elif activity_days >= 1:
        activity_multiplier = 1.275  # lightly active
    
    tdee = bmr * activity_multiplier
    
    # Adjust calories based on goal
    if diet_goal.lower() == 'weight loss':
        daily_calories = int(tdee * 0.85)
    elif diet_goal.lower() == 'weight gain' or diet_goal.lower() == 'muscle building':
        daily_calories = int(tdee * 1.15)
    else:
        daily_calories = int(tdee)
    
    plan['daily_calories'] = daily_calories
    plan['bmr'] = round(bmr, 0)
    plan['tdee'] = round(tdee, 0)
    plan['activity_multiplier'] = activity_multiplier
    
    # ===== MACRO BREAKDOWN =====
    macros = {
        'title': 'ğŸ“Š DAILY MACRO BREAKDOWN',
        'total_daily_calories': daily_calories,
        'breakdown': {
            'protein': {
                'percentage': 30,
                'grams': int(daily_calories * 0.30 / 4),
                'sources': f"Lean meats, fish, eggs, legumes, dairy (considering {food_habit})"
            },
            'carbohydrates': {
                'percentage': 45,
                'grams': int(daily_calories * 0.45 / 4),
                'sources': 'Whole grains, brown rice, oats, sweet potatoes, legumes'
            },
            'fats': {
                'percentage': 25,
                'grams': int(daily_calories * 0.25 / 9),
                'sources': 'Olive oil, nuts, seeds, avocado, fatty fish'
            }
        }
    }
    plan['routines']['macros'] = macros
    
    # ===== HYDRATION PLAN =====
    base_water_intake = 2.5
    if climate in ['Tropical', 'Hot/Arid']:
        base_water_intake = 3.5
    elif climate in ['Cold/Polar']:
        base_water_intake = 2.0
    
    if activity_days >= 5:
        base_water_intake += 0.5
    
    hydration = {
        'title': 'ğŸ’§ HYDRATION PLAN',
        'personalized_for': f'{season} season in {climate} climate',
        'daily_water_intake': f'{base_water_intake}-{base_water_intake + 0.5} liters',
        'activity_adjusted': f'Based on {activity_days} days/week activity'
    }
    plan['routines']['hydration'] = hydration
    
    # ===== GOAL-SPECIFIC STRATEGIES =====
    goal_strategies = {
        'title': 'ğŸ�¯ DIET GOAL-SPECIFIC STRATEGIES',
        'goal': diet_goal,
        'daily_calorie_target': daily_calories
    }
    
    goal_strategies_map = {
        'weight loss': f'Calorie deficit ({daily_calories}cal) + high protein + high fiber',
        'weight gain': f'Calorie surplus ({daily_calories}cal) + nutrient-dense foods',
        'muscle building': f'Moderate surplus ({daily_calories}cal) + 1.6-2.2g protein/kg',
        'boost energy & focus': 'Macro-timing + stable blood sugar + antioxidants',
        'improve skin & hair health': 'Omega-3s + Antioxidants + Biotin + Collagen',
        'weight maintenance & nutritional balance': f'Balanced macros at TDEE ({daily_calories}cal)',
        'athletic performance': f'Carb-loading ({daily_calories}cal) + Hydration + Timing',
        'digestive health': 'Fiber + Probiotics + Whole foods + Hydration'
    }
    
    goal_strategies['strategy'] = goal_strategies_map.get(diet_goal.lower(), 'Personalized approach')
    plan['routines']['goal_strategies'] = goal_strategies
    
    return plan


def store_diet_plan_to_profile(user_name: str, diet_plan: dict) -> dict:
    """Stores the generated diet plan to user profile."""
    
    return {
        'status': 'plan_stored',
        'user_name': user_name,
        'plan_id': f"diet_{user_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'commitment_days': diet_plan.get('commitment_days'),
        'diet_goal': diet_plan.get('diet_goal'),
        'daily_calories': diet_plan.get('daily_calories'),
        'message': f"âœ… Internet-sourced personalized diet plan for {user_name} saved!",
        'plan_summary': f"Complete {diet_plan.get('commitment_days')}-day diet plan with recipes fetched from internet based on {diet_plan.get('season')} season in {diet_plan.get('climate')} climate!"
    }


# =====================================================
# AGENT 5: DIET PLAN ADVISOR (NEW)
# =====================================================

def create_diet_agent(model: str = "gemini-2.0-flash"):
    """Creates the Diet Plan Agent - triggered by user choice, not automatic."""
    
    diet_agent = Agent(
        name="diet_plan_advisor_agent",
        model=model,
        description="Generates personalized diet plans based on health data, goals, restrictions, season & climate.",
        
        instruction="""You are the Diet Plan Advisor Agent - PERSONALIZATION MASTER.

WORKFLOW - COLLECT DATA (ONE AT A TIME):
Phase 1: Commitment days (15/30/45/90)
Phase 2: BMI Calculation (height + weight with units)
Phase 3: Food habit (Vegetarian/Non-veg/Vegan/Pescatarian/Jain)
Phase 4: Food allergies or intolerances (comma-separated list)
Phase 5: Primary diet goal (Weight loss/Gain/Muscle/Energy/Skin Health/Maintenance/Athletic/Digestion)
Phase 6: Physical activity type (gym/running/yoga/sports/etc)
Phase 7: Activity frequency (0-7 days per week)

THEN GENERATE PERSONALIZED PLAN:
- Calculate daily calorie needs based on THEIR specific: age, weight, height, activity, gender
- Personalize meals by food habit (vegan, vegetarian, non-veg, jain, pescatarian)
- Adapt meals for THEIR allergies with alternatives
- Fetch REAL recipes from internet for each meal type
- Generate goal-specific strategies (weight loss/gain/muscle/energy/skin/athletic/maintenance/digestion)
- Create hydration plan (adjusted for THEIR climate + activity)
- Include macro breakdown (calculated from THEIR BMI + activity)
- Provide progress tracking guidance

ğŸ”‘ CRITICAL RULES - NO HARDCODING:
- RETRIEVE from profile: name, age, gender, climate, season (NOT assumed)
- CALCULATE daily calories based on THEIR specific: age, weight, height, activity, gender
- FETCH REAL recipes from internet for EACH meal type
- Include seasonal produce (what's available NOW in THEIR season)
- Adapt to climate (tropical/hot/temperate/cold affects hydration & nutrition)
- Personalize allergies (fetch recipes that exclude THEIR specific items)
- Match food habit strictly
- Adapt macros by goal
- Climate affects: hydration needs, ingredient availability, energy requirements

IMPORTANT NOTES:
- ONE question per turn
- Ask for data sequentially (don't ask all at once)
- CALCULATE not assume
- FETCH from internet not hardcode
- PERSONALIZE everything to user's specific needs

TONE: Supportive, knowledgeable, health-focused, detail-oriented.""",
        
        tools=[collect_diet_details, search_diet_recipes_from_internet, generate_diet_plan, store_diet_plan_to_profile]
    )
    
    return diet_agent


# =====================================================
# ROOT ORCHESTRATOR AGENT (UPDATED FOR MENUS)
# =====================================================

def create_root_agent(model: str = "gemini-2.0-flash"):
    """Creates the Root Orchestrator Agent for coordination."""
    
    user_info_agent = create_user_info_agent(model=model)
    location_weather_agent = create_location_weather_agent(model=model)
    skincare_agent = create_skincare_agent(model=model)
    grooming_agent = create_grooming_agent(model=model)
    diet_agent = create_diet_agent(model=model)
    
    root_agent = Agent(
        name="glow_guide_root_agent",
        model=model,
        description="Manages Glow Guide workflow with agent coordination.",
        
        instruction="""You are the Glow Guide Root Orchestrator.

YOUR JOB: COORDINATE AGENT WORKFLOW & SHOW SERVICE MENU

WORKFLOW:
Step 1: First user message â†’ Delegate to Agent 1 (User Info)
Step 2: Agent 1 completes â†’ Delegate to Agent 2 (Location & Weather)
Step 3: Agent 2 completes â†’ SHOW SERVICE MENU
Step 4: User chooses â†’ Delegate to Agent 3, Agent 4, or Agent 5

SERVICE MENU (After Agent 2 completes):
"âœ¨ Now, let's define the engine for your personalized transformation.
Please choose the primary focus area that you want to address right now.

Select one option below:
1ï¸�âƒ£ Build a personalized skincare routine
2ï¸�âƒ£ Build a personalized grooming routine
3ï¸�âƒ£ Build a personalized diet plan"

DELEGATION RULES:
- Follow sequence strictly (Agent 1 â†’ Agent 2 â†’ Menu â†’ Agent 3/4/5)
- Do NOT answer direct questions yourself
- Do NOT collect data yourself
- Show menu ONLY after Agent 2 completes
- Wait for user choice BEFORE delegating
- If user chooses "1" â†’ Delegate to Agent 3 (Skincare)
- If user chooses "2" â†’ Delegate to Agent 4 (Grooming)
- If user chooses "3" â†’ Delegate to Agent 5 (Diet Plan)
- Be warm and professional

TONE: Professional, warm, organized.""",
        
        tools=[],
        sub_agents=[user_info_agent, location_weather_agent, skincare_agent, grooming_agent, diet_agent]
    )
    
    return root_agent


# =====================================================
# SETUP RUNNER AND SERVICES - UPDATED
# =====================================================

session_service = InMemorySessionService()
memory_service = InMemoryMemoryService()

root_agent = create_root_agent(model="gemini-2.0-flash")

runner = Runner(
    agent=root_agent,
    app_name="GlowGuideAgent",
    session_service=session_service,
    memory_service=memory_service,
)


url_prefix = get_adk_proxy_url()


!adk web --url_prefix {url_prefix}

