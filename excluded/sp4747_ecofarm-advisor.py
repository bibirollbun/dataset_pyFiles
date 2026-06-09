import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
from enum import Enum

class CropType(Enum):
    CORN = "corn"
    WHEAT = "wheat"
    SOYBEAN = "soybean"
    VEGETABLES = "vegetables"

@dataclass
class FarmData:
    location: str
    size_hectares: float
    soil_type: str
    current_crops: List[CropType]
    water_source: str

@dataclass
class SustainabilityScore:
    water_usage: int
    soil_health: int
    biodiversity: int
    carbon_footprint: int
    overall: int

class AgriculturalTools:
    """Custom tools for agricultural data analysis"""
    
    @staticmethod
    def calculate_water_requirements(crop: CropType, area: float, season: str) -> float:
        water_rates = {
            CropType.CORN: 5000,
            CropType.WHEAT: 4000,
            CropType.SOYBEAN: 4500,
            CropType.VEGETABLES: 6000
        }
        
        seasonal_adjustment = {
            "spring": 1.0,
            "summer": 1.3,
            "fall": 0.8,
            "winter": 0.5
        }
        
        base_water = water_rates.get(crop, 5000)
        return base_water * area * seasonal_adjustment.get(season, 1.0)
    
    @staticmethod
    def assess_soil_health(soil_type: str, recent_crops: List[CropType]) -> int:
        soil_scores = {
            "clay": 7,
            "loam": 9,
            "sandy": 5,
            "silt": 8
        }
        
        unique_crops = len(set(recent_crops))
        diversity_bonus = min(2, unique_crops - 1)
        
        base_score = soil_scores.get(soil_type, 5)
        return min(10, base_score + diversity_bonus)

class DataCollectorAgent:
    def __init__(self):
        self.tools = AgriculturalTools()
        self.logger = logging.getLogger("DataCollectorAgent")
    
    async def collect_farm_data(self, farmer_input: Dict[str, Any]) -> FarmData:
        self.logger.info("Collecting farm data...")
        
        # Convert string crops to CropType enum
        current_crops = []
        for crop in farmer_input.get("current_crops", []):
            if isinstance(crop, str):
                try:
                    current_crops.append(CropType(crop))
                except ValueError:
                    # Handle unknown crop types gracefully
                    self.logger.warning(f"Unknown crop type: {crop}")
                    continue
            else:
                current_crops.append(crop)
        
        farm_data = FarmData(
            location=farmer_input.get("location", "Unknown"),
            size_hectares=farmer_input.get("size_hectares", 0),
            soil_type=farmer_input.get("soil_type", "loam"),
            current_crops=current_crops,
            water_source=farmer_input.get("water_source", "well")
        )
        
        self.logger.info(f"Collected data for farm in {farm_data.location}")
        return farm_data

class AnalysisAgent:
    def __init__(self):
        self.tools = AgriculturalTools()
        self.logger = logging.getLogger("AnalysisAgent")
    
    async def analyze_sustainability(self, farm_data: FarmData) -> SustainabilityScore:
        self.logger.info("Analyzing farm sustainability...")
        
        # Run analyses concurrently
        water_score, soil_score, biodiversity_score, carbon_score = await asyncio.gather(
            self._analyze_water_usage(farm_data),
            self._analyze_soil_health(farm_data),
            self._analyze_biodiversity(farm_data),
            self._analyze_carbon_footprint(farm_data)
        )
        
        overall_score = (water_score + soil_score + biodiversity_score + carbon_score) // 4
        
        return SustainabilityScore(
            water_usage=water_score,
            soil_health=soil_score,
            biodiversity=biodiversity_score,
            carbon_footprint=carbon_score,
            overall=overall_score
        )
    
    async def _analyze_water_usage(self, farm_data: FarmData) -> int:
        if not farm_data.current_crops:
            return 5  # Default score if no crops
        
        total_water = sum(
            self.tools.calculate_water_requirements(crop, farm_data.size_hectares, "spring")
            for crop in farm_data.current_crops
        )
        
        if total_water < 10000:
            return 9
        elif total_water < 20000:
            return 7
        elif total_water < 30000:
            return 5
        else:
            return 3
    
    async def _analyze_soil_health(self, farm_data: FarmData) -> int:
        return self.tools.assess_soil_health(farm_data.soil_type, farm_data.current_crops)
    
    async def _analyze_biodiversity(self, farm_data: FarmData) -> int:
        crop_diversity = len(set(farm_data.current_crops))
        if crop_diversity >= 4:
            return 9
        elif crop_diversity >= 3:
            return 7
        elif crop_diversity >= 2:
            return 5
        else:
            return 3
    
    async def _analyze_carbon_footprint(self, farm_data: FarmData) -> int:
        score = 6
        if farm_data.water_source == "rainwater":
            score += 2
        if len(farm_data.current_crops) > 1:
            score += 1
        if farm_data.size_hectares < 10:
            score += 1
        return min(10, score)

class SustainabilityAdvisorAgent:
    def __init__(self):
        self.logger = logging.getLogger("SustainabilityAdvisorAgent")
        self.recommendation_knowledge = {
            "water_conservation": [
                "Implement drip irrigation systems",
                "Collect and use rainwater",
                "Use mulch to reduce evaporation",
                "Schedule irrigation during cooler hours"
            ],
            "soil_health": [
                "Practice crop rotation",
                "Use cover crops during off-season",
                "Apply organic compost",
                "Reduce tillage practices"
            ],
            "biodiversity": [
                "Plant native species in border areas",
                "Implement intercropping",
                "Maintain hedgerows for wildlife",
                "Create pollinator-friendly habitats"
            ]
        }
    
    async def generate_recommendations(self, farm_data: FarmData, 
                                     sustainability_score: SustainabilityScore) -> Dict[str, List[str]]:
        self.logger.info("Generating sustainability recommendations...")
        
        recommendations = {}
        
        if sustainability_score.water_usage < 7:
            recommendations["water_management"] = self._get_water_recommendations()
        
        if sustainability_score.soil_health < 7:
            recommendations["soil_improvement"] = self._get_soil_recommendations()
        
        if sustainability_score.biodiversity < 7:
            recommendations["biodiversity"] = self._get_biodiversity_recommendations()
        
        if sustainability_score.carbon_footprint < 7:
            recommendations["carbon_reduction"] = self._get_carbon_recommendations()
        
        recommendations["general"] = [
            "Monitor soil moisture regularly",
            "Use integrated pest management",
            "Keep detailed farm records for continuous improvement"
        ]
        
        return recommendations
    
    def _get_water_recommendations(self) -> List[str]:
        return self.recommendation_knowledge["water_conservation"]
    
    def _get_soil_recommendations(self) -> List[str]:
        return self.recommendation_knowledge["soil_health"]
    
    def _get_biodiversity_recommendations(self) -> List[str]:
        return self.recommendation_knowledge["biodiversity"]
    
    def _get_carbon_recommendations(self) -> List[str]:
        return [
            "Use renewable energy for farm operations",
            "Implement agroforestry practices",
            "Reduce synthetic fertilizer use",
            "Optimize machinery use to reduce fuel consumption"
        ]

class ReportGeneratorAgent:
    def __init__(self):
        self.logger = logging.getLogger("ReportGeneratorAgent")
    
    async def generate_report(self, farm_data: FarmData, 
                            sustainability_score: SustainabilityScore,
                            recommendations: Dict[str, List[str]]) -> str:
        self.logger.info("Generating farm sustainability report...")
        
        crop_names = ', '.join([crop.value for crop in farm_data.current_crops]) if farm_data.current_crops else "None"
        
        report = f"""
ECOFARM ADVISOR SUSTAINABILITY REPORT
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

FARM OVERVIEW:
- Location: {farm_data.location}
- Size: {farm_data.size_hectares} hectares
- Soil Type: {farm_data.soil_type}
- Current Crops: {crop_names}
- Water Source: {farm_data.water_source}

SUSTAINABILITY SCORE: {sustainability_score.overall}/10

DETAILED SCORES:
- Water Usage: {sustainability_score.water_usage}/10
- Soil Health: {sustainability_score.soil_health}/10
- Biodiversity: {sustainability_score.biodiversity}/10
- Carbon Footprint: {sustainability_score.carbon_footprint}/10

RECOMMENDATIONS:
"""
        
        for category, recs in recommendations.items():
            report += f"\n{category.upper().replace('_', ' ')}:\n"
            for i, rec in enumerate(recs, 1):
                report += f"  {i}. {rec}\n"
        
        report += "\nNEXT STEPS:\n"
        report += "1. Prioritize recommendations based on your farm's specific needs\n"
        report += "2. Create an implementation timeline\n"
        report += "3. Monitor progress and adjust practices as needed\n"
        report += "4. Schedule follow-up assessment in 6 months\n"
        
        return report

class EcoFarmAdvisorSystem:
    def __init__(self):
        self.data_collector = DataCollectorAgent()
        self.analysis_agent = AnalysisAgent()
        self.advisor_agent = SustainabilityAdvisorAgent()
        self.report_generator = ReportGeneratorAgent()
        
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self._setup_logging()
    
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
    
    async def create_session(self, farmer_id: str) -> str:
        session_id = f"{farmer_id}_{datetime.now().timestamp()}"
        self.sessions[session_id] = {
            "farmer_id": farmer_id,
            "created_at": datetime.now(),
            "farm_data": None,
            "sustainability_score": None,
            "recommendations": None
        }
        logging.info(f"Created new session: {session_id}")
        return session_id
    
    async def process_farm_assessment(self, session_id: str, farmer_input: Dict[str, Any]) -> str:
        try:
            logging.info(f"Starting farm assessment for session: {session_id}")
            
            farm_data = await self.data_collector.collect_farm_data(farmer_input)
            self.sessions[session_id]["farm_data"] = farm_data
            
            sustainability_score = await self.analysis_agent.analyze_sustainability(farm_data)
            self.sessions[session_id]["sustainability_score"] = sustainability_score
            
            recommendations = await self.advisor_agent.generate_recommendations(
                farm_data, sustainability_score
            )
            self.sessions[session_id]["recommendations"] = recommendations
            
            report = await self.report_generator.generate_report(
                farm_data, sustainability_score, recommendations
            )
            
            logging.info(f"Completed farm assessment for session: {session_id}")
            return report
            
        except Exception as e:
            logging.error(f"Error in farm assessment: {str(e)}")
            return f"Error processing farm assessment: {str(e)}"
    
    def get_session_history(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self.sessions.get(session_id)

# Jupyter-compatible demo function
async def demo_ecofarm_system():
    """Demo function that can be run in Jupyter"""
    print("ðŸŒ± EcoFarm Advisor - Sustainable Agriculture Assistant")
    print("=" * 50)
    
    ecofarm_system = EcoFarmAdvisorSystem()
    session_id = await ecofarm_system.create_session("farmer_123")
    
    farmer_input = {
        "location": "California Central Valley",
        "size_hectares": 15.5,
        "soil_type": "loam",
        "current_crops": ["corn", "wheat"],  # Using strings instead of enum
        "water_source": "well"
    }
    
    print("Processing farm assessment...")
    report = await ecofarm_system.process_farm_assessment(session_id, farmer_input)
    
    print("\n" + "=" * 50)
    print("ASSESSMENT COMPLETE!")
    print("=" * 50)
    print(report)
    
    # Show session memory functionality
    session_history = ecofarm_system.get_session_history(session_id)
    if session_history:
        print(f"\nðŸ“Š Session stored for farmer: {session_history['farmer_id']}")
        print(f"ðŸ“… Assessment date: {session_history['created_at']}")
    
    return report

# Method 1: For Jupyter - using existing event loop
def run_demo_jupyter():
    """Run the demo in Jupyter notebook using existing event loop"""
    try:
        # Try to get the current event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is running, use create_task
            task = loop.create_task(demo_ecofarm_system())
            return task
        else:
            # If loop exists but isn't running, run until complete
            return loop.run_until_complete(demo_ecofarm_system())
    except RuntimeError:
        # No event loop, create new one
        return asyncio.run(demo_ecofarm_system())

# Method 2: Simple await in Jupyter (if you're in an async context)
async def run_demo_simple():
    """Simple async function to run in Jupyter with await"""
    return await demo_ecofarm_system()

# Method 3: Synchronous wrapper
def run_demo_sync():
    """Synchronous wrapper for the demo"""
    import asyncio
    return asyncio.run(demo_ecofarm_system())

# Choose the method that works for your environment:

# OPTION 1: If you're in a regular Python script or Colab:
# result = run_demo_sync()

# OPTION 2: If you're in Jupyter and want to use the existing loop:
# result = run_demo_jupyter()

# OPTION 3: If you're in an async context in Jupyter:
# result = await run_demo_simple()

# Let's try Option 2 for Jupyter:
print("Starting EcoFarm Advisor Demo...")
result = run_demo_jupyter()

