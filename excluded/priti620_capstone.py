# Environmental Sustainability Multi-Agent System
# Demonstrating course concepts: Multi-agent system, Tools, Sessions & Memory

import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any
import requests

# ==================== SESSIONS & MEMORY ====================
class InMemorySessionService:
    def __init__(self):
        self.sessions = {}
        self.memory_bank = {}
    
    def create_session(self, session_id: str, initial_state: Dict):
        self.sessions[session_id] = {
            'state': initial_state,
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        }
    
    def update_session(self, session_id: str, updates: Dict):
        if session_id in self.sessions:
            self.sessions[session_id]['state'].update(updates)
            self.sessions[session_id]['updated_at'] = datetime.now()
    
    def get_session(self, session_id: str) -> Dict:
        return self.sessions.get(session_id, {})
    
    def store_memory(self, key: str, data: Any):
        self.memory_bank[key] = {
            'data': data,
            'timestamp': datetime.now()
        }
    
    def retrieve_memory(self, key: str) -> Any:
        return self.memory_bank.get(key, {})

# ==================== TOOLS ====================
class SustainabilityTools:
    """Custom tools for environmental analysis"""
    
    @staticmethod
    def carbon_footprint_calculator(activity: str, value: float) -> Dict:
        """Calculate carbon footprint for various activities"""
        carbon_factors = {
            'electricity_kwh': 0.5,  # kg CO2 per kWh
            'gasoline_liter': 2.3,   # kg CO2 per liter
            'flight_km': 0.09,       # kg CO2 per km
            'waste_kg': 0.5          # kg CO2 per kg waste
        }
        
        factor = carbon_factors.get(activity, 0)
        return {
            'activity': activity,
            'value': value,
            'carbon_emission_kg': value * factor,
            'equivalent_trees': round((value * factor) / 21, 2)  # trees needed to offset
        }
    
    @staticmethod
    def water_usage_calculator(activity: str, value: float) -> Dict:
        """Calculate water usage for various activities"""
        water_factors = {
            'shower_minutes': 7.5,    # liters per minute
            'laundry_load': 50,       # liters per load
            'dishwashing': 20,        # liters per session
            'gardening_hour': 100     # liters per hour
        }
        
        factor = water_factors.get(activity, 0)
        return {
            'activity': activity,
            'value': value,
            'water_usage_liters': value * factor
        }

# ==================== AGENTS ====================
class DataCollectionAgent:
    """Agent for collecting environmental data"""
    
    def __init__(self, tools: SustainabilityTools):
        self.tools = tools
        self.role = "Environmental Data Collector"
    
    def process_household_data(self, household_data: Dict) -> Dict:
        """Process household sustainability data"""
        results = {
            'carbon_footprint': [],
            'water_usage': [],
            'recommendations': []
        }
        
        # Calculate carbon footprint
        for activity, value in household_data.get('energy_usage', {}).items():
            result = self.tools.carbon_footprint_calculator(activity, value)
            results['carbon_footprint'].append(result)
        
        # Calculate water usage
        for activity, value in household_data.get('water_usage', {}).items():
            result = self.tools.water_usage_calculator(activity, value)
            results['water_usage'].append(result)
        
        return results

class AnalysisAgent:
    """Agent for analyzing sustainability data"""
    
    def __init__(self):
        self.role = "Sustainability Analyst"
    
    def analyze_sustainability(self, data: Dict) -> Dict:
        """Analyze environmental impact and provide insights"""
        total_carbon = sum(item['carbon_emission_kg'] for item in data['carbon_footprint'])
        total_water = sum(item['water_usage_liters'] for item in data['water_usage'])
        
        analysis = {
            'total_carbon_emission_kg': total_carbon,
            'total_water_usage_liters': total_water,
            'carbon_rating': self._get_carbon_rating(total_carbon),
            'water_efficiency': self._get_water_efficiency(total_water),
            'improvement_areas': self._identify_improvement_areas(data),
            'sustainability_score': self._calculate_sustainability_score(total_carbon, total_water)
        }
        
        return analysis
    
    def _get_carbon_rating(self, carbon: float) -> str:
        if carbon < 100: return "Excellent"
        elif carbon < 500: return "Good"
        elif carbon < 1000: return "Average"
        else: return "Needs Improvement"
    
    def _get_water_efficiency(self, water: float) -> str:
        if water < 500: return "Efficient"
        elif water < 1000: return "Moderate"
        else: return "Inefficient"
    
    def _identify_improvement_areas(self, data: Dict) -> List[str]:
        areas = []
        for item in data['carbon_footprint']:
            if item['carbon_emission_kg'] > 100:
                areas.append(f"Reduce {item['activity']}")
        return areas
    
    def _calculate_sustainability_score(self, carbon: float, water: float) -> int:
        base_score = 100
        carbon_penalty = min(carbon / 10, 50)
        water_penalty = min(water / 100, 30)
        return max(0, base_score - carbon_penalty - water_penalty)

class RecommendationAgent:
    """Agent for providing sustainability recommendations"""
    
    def __init__(self):
        self.role = "Sustainability Advisor"
        self.recommendations_db = {
            'high_carbon': [
                "Switch to renewable energy sources",
                "Use public transportation or carpool",
                "Reduce air travel when possible"
            ],
            'high_water': [
                "Install water-efficient fixtures",
                "Collect rainwater for gardening",
                "Fix leaky faucets promptly"
            ],
            'general': [
                "Plant native trees in your area",
                "Use reusable bags and containers",
                "Compost organic waste"
            ]
        }
    
    def generate_recommendations(self, analysis: Dict) -> Dict:
        """Generate personalized sustainability recommendations"""
        recs = {
            'priority_actions': [],
            'long_term_goals': [],
            'immediate_actions': []
        }
        
        # Priority actions based on analysis
        if analysis['carbon_rating'] in ["Average", "Needs Improvement"]:
            recs['priority_actions'].extend(self.recommendations_db['high_carbon'])
        
        if analysis['water_efficiency'] in ["Moderate", "Inefficient"]:
            recs['priority_actions'].extend(self.recommendations_db['high_water'])
        
        # Long term goals
        recs['long_term_goals'] = [
            f"Achieve sustainability score of {analysis['sustainability_score'] + 20}",
            "Reduce carbon footprint by 25% in 6 months",
            "Implement water conservation system"
        ]
        
        # Immediate actions
        recs['immediate_actions'] = self.recommendations_db['general']
        
        return recs

# ==================== MULTI-AGENT SYSTEM ====================
class SustainabilityMultiAgentSystem:
    """Multi-agent system for environmental sustainability analysis"""
    
    def __init__(self):
        self.session_service = InMemorySessionService()
        self.tools = SustainabilityTools()
        
        # Initialize agents
        self.data_agent = DataCollectionAgent(self.tools)
        self.analysis_agent = AnalysisAgent()
        self.recommendation_agent = RecommendationAgent()
        
        # Store system memory
        self.session_service.store_memory('system_config', {
            'agents': [agent.role for agent in [self.data_agent, self.analysis_agent, self.recommendation_agent]],
            'tools_available': ['carbon_footprint_calculator', 'water_usage_calculator'],
            'created_at': datetime.now()
        })
    
    def process_household_sustainability(self, household_id: str, household_data: Dict) -> Dict:
        """Process household data through the multi-agent pipeline"""
        
        # Create session
        self.session_service.create_session(household_id, {
            'initial_data': household_data,
            'processing_start': datetime.now()
        })
        
        print(f"Processing household {household_id} through multi-agent system...")
        
        # SEQUENTIAL AGENTS PIPELINE
        # Agent 1: Data Collection
        print("1. Data Collection Agent processing...")
        processed_data = self.data_agent.process_household_data(household_data)
        self.session_service.update_session(household_id, {'processed_data': processed_data})
        
        # Agent 2: Analysis
        print("2. Analysis Agent processing...")
        analysis = self.analysis_agent.analyze_sustainability(processed_data)
        self.session_service.update_session(household_id, {'analysis': analysis})
        
        # Agent 3: Recommendations
        print("3. Recommendation Agent processing...")
        recommendations = self.recommendation_agent.generate_recommendations(analysis)
        self.session_service.update_session(household_id, {'recommendations': recommendations})
        
        # Final results
        final_result = {
            'household_id': household_id,
            'processed_data': processed_data,
            'analysis': analysis,
            'recommendations': recommendations,
            'session_info': self.session_service.get_session(household_id)
        }
        
        # Store in memory bank
        self.session_service.store_memory(f'result_{household_id}', final_result)
        
        return final_result
    
    def get_session_history(self, household_id: str) -> Dict:
        """Retrieve session history from memory"""
        return self.session_service.get_session(household_id)

# ==================== DEMONSTRATION ====================
def demonstrate_system():
    """Demonstrate the multi-agent sustainability system"""
    
    print("ğŸŒ� SUSTAINABILITY MULTI-AGENT SYSTEM DEMONSTRATION")
    print("=" * 50)
    
    # Initialize the multi-agent system
    sustainability_system = SustainabilityMultiAgentSystem()
    
    # Sample household data
    household_data = {
        'energy_usage': {
            'electricity_kwh': 300,
            'gasoline_liter': 100,
            'flight_km': 2000
        },
        'water_usage': {
            'shower_minutes': 300,
            'laundry_load': 8,
            'gardening_hour': 10
        }
    }
    
    # Process through multi-agent system
    result = sustainability_system.process_household_sustainability("household_001", household_data)
    
    print("\n RESULTS:")
    print(f"Carbon Emission: {result['analysis']['total_carbon_emission_kg']:.2f} kg CO2")
    print(f"Water Usage: {result['analysis']['total_water_usage_liters']:.2f} liters")
    print(f"Sustainability Score: {result['analysis']['sustainability_score']}/100")
    print(f"Carbon Rating: {result['analysis']['carbon_rating']}")
    print(f"Water Efficiency: {result['analysis']['water_efficiency']}")
    
    print("\n RECOMMENDATIONS:")
    print("Priority Actions:")
    for action in result['recommendations']['priority_actions'][:3]:
        print(f"  â€¢ {action}")
    
    print("\n SESSION MEMORY RETRIEVAL:")
    session_history = sustainability_system.get_session_history("household_001")
    print(f"Session created: {session_history.get('created_at')}")
    print(f"Last updated: {session_history.get('updated_at')}")
    
    return result

# Run demonstration
if __name__ == "__main__":
    results = demonstrate_system()
    
    # Create visualization data
    df_data = []
    for item in results['processed_data']['carbon_footprint']:
        df_data.append({
            'Activity': item['activity'],
            'Carbon_Emission_kg': item['carbon_emission_kg'],
            'Equivalent_Trees': item['equivalent_trees']
        })
    
    df = pd.DataFrame(df_data)
    print("\n CARBON FOOTPRINT BREAKDOWN:")
    print(df.to_string(index=False))


