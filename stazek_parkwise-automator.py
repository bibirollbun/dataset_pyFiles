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


## Dummy Dataset Generation for ParkWise Automator

# =============================================================================
# SETUP
# =============================================================================
# Core data manipulation and visualization
import pandas as pd
import numpy as np
import hashlib
from datetime import datetime, timedelta
import warnings
import random
import logging
from typing import Dict, Optional
import os

warnings.filterwarnings('ignore')

# Configuration for dummy data
N_SLOTS = 1000  # Total number of parking slots
N_LOGS_RAW = 25000  # Number of entries in raw logs/payments
N_LOGS_CLEANED = 24500  # Number of cleaned logs (some failed entries/exits)
N_ZONES = 5

# Define Zone and Slot details
ZONE_DETAILS = {
    'A': {'capacity': 250, 'rate_multiplier': 1.0, 'is_premium': False, 'floor': 'Ground'},
    'B': {'capacity': 250, 'rate_multiplier': 1.0, 'is_premium': False, 'floor': 'Level 1'},
    'C': {'capacity': 100, 'rate_multiplier': 1.5, 'is_premium': True, 'floor': 'Level 2'},
    'D': {'capacity': 300, 'rate_multiplier': 0.8, 'is_premium': False, 'floor': 'Basement 1'},
    'E': {'capacity': 100, 'rate_multiplier': 1.2, 'is_premium': True, 'floor': 'Rooftop'},
}

SLOT_TYPES = ['standard'] * 7 + ['premium'] * 2 + ['disabled'] * 1
SLOT_STATUSES = ['available'] * 8 + ['occupied'] * 2 + ['maintenance'] * 0
VEHICLE_PREFIXES = ['MH', 'KA', 'DL', 'TN']
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2024, 11, 28)

print("=" * 80)
print("ParkWise Automator - DUMMY DATASET CREATOR")
print("=" * 80)

# Create a dummy path for simulation (Kaggle context)
class MockKagglePath:
    def __init__(self):
        # Create a temporary directory structure to simulate Kaggle environment
        self.base_dir = "temp_kaggle_input"
        self.data_path = os.path.join(self.base_dir, "mall-parking-analysis")
        os.makedirs(self.data_path, exist_ok=True)
        print(f"\nCreated mock data path: {self.data_path}")
        
    def get_path(self):
        return self.data_path

mock_path_manager = MockKagglePath()
DATA_PATH = mock_path_manager.get_path() + "/"


# =============================================================================
# 1. PARKING INFRASTRUCTURE (parking_zones.csv, parking_slots.csv)
# =============================================================================

## A. parking_zones.csv
print("\n1. Generating parking_zones.csv...")
zones_data = []
for zone_id, details in ZONE_DETAILS.items():
    zones_data.append({
        'zone_id': zone_id,
        'floor': details['floor'],
        'total_capacity': details['capacity'],
        'rate_multiplier': details['rate_multiplier'],
        'is_premium': details['is_premium'],
        'description': f"Parking Zone {zone_id} on {details['floor']}"
    })
parking_zones = pd.DataFrame(zones_data)
parking_zones.to_csv(f"{DATA_PATH}parking_zones.csv", index=False)
print(f"   Shape: {parking_zones.shape}")


## B. parking_slots.csv
print("2. Generating parking_slots.csv...")
slots_data = []
slot_counter = 1
for zone_id, details in ZONE_DETAILS.items():
    for i in range(details['capacity']):
        slot_id = f"{zone_id}-{i+1:03d}"
        slot_type = random.choice(SLOT_TYPES)
        status = random.choice(SLOT_STATUSES)
        
        # Ensure a good mix of available/occupied slots initially
        if status == 'occupied':
            last_entry = START_DATE + (END_DATE - START_DATE) * np.random.rand()
            last_entry = last_entry.replace(minute=np.random.randint(0, 60), second=np.random.randint(0, 60))
        else:
            last_entry = None
            
        slots_data.append({
            'slot_id': slot_id,
            'zone_id': zone_id,
            'floor': details['floor'],
            'slot_type': slot_type,
            'status': status,
            'is_reserved_for_disabled': (slot_type == 'disabled'),
            'last_occupied_time': last_entry
        })
parking_slots = pd.DataFrame(slots_data)
parking_slots.to_csv(f"{DATA_PATH}parking_slots.csv", index=False)
print(f"   Shape: {parking_slots.shape}")


# =============================================================================
# CELL 2: AGENT FOUNDATION DEFINITIONS
# =============================================================================
# NOTE: This cell assumes necessary imports (like Enum, datetime, typing, pandas, etc.)
# have been successfully executed in CELL 1.

print("\n" + "=" * 80)
print("CELL 2: AGENT FOUNDATION DEFINITIONS")
print("=" * 80)

from enum import Enum # <-- Explicitly ensuring Enum is defined if Cell 1 was skipped
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np
import hashlib
import logging

logger = logging.getLogger('ParkWise')

# --- ENUMERATIONS ---
class AgentType(Enum):
    ORCHESTRATOR = "Orchestrator"
    VALIDATOR = "EntryValidator"
    ALLOCATOR = "SlotAllocator"
    MONITOR = "OccupancyMonitor"
    REMINDER = "PaymentReminder"
    ANALYTICS = "Analytics"

class VehicleStatus(Enum):
    PENDING = "pending"
    PARKED = "parked"
    PAYMENT_PENDING = "payment_pending"
    EXITED = "exited"


# --- DATA STRUCTURES (Models) ---
class ParkingSession:
    def __init__(self, session_id, vehicle_number, status, entry_time, assigned_slot, zone, context: Dict):
        self.session_id = session_id
        self.vehicle_number = vehicle_number
        self.status = status
        self.entry_time = entry_time
        self.exit_time: Optional[datetime] = None
        self.assigned_slot = assigned_slot
        self.zone = zone
        self.context = context

class AgentMetrics:
    def __init__(self, agent_id: str, agent_type: AgentType):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.total_requests = 0
        self.successful_requests = 0
        self.processing_times_ms: List[float] = []
        self.avg_processing_time_ms = 0.0

    def update_success(self, processing_time_ms: float):
        self.total_requests += 1
        self.successful_requests += 1
        self.processing_times_ms.append(processing_time_ms)
        self._recalculate_avg()

    def update_failure(self):
        self.total_requests += 1
        self._recalculate_avg()

    def _recalculate_avg(self):
        if self.processing_times_ms:
            self.avg_processing_time_ms = sum(self.processing_times_ms) / len(self.processing_times_ms)

    def get_success_rate(self) -> float:
        if self.total_requests == 0:
            return 100.0
        return (self.successful_requests / self.total_requests) * 100.0


# --- MOCK TOOLS (Data Access Layer) ---
class MockParkingDB:
    """Simulates the real-time database for slots."""
    def __init__(self, initial_slots: pd.DataFrame):
        # Use a dictionary for fast lookup and state management
        self.slots_state: Dict[str, Dict] = initial_slots.set_index('slot_id').to_dict('index')

    def get_available_slots(self, zone_id: Optional[str] = None) -> List[Dict]:
        slots = []
        for slot_id, data in self.slots_state.items():
            if data['status'] == 'available':
                if not zone_id or data['zone_id'] == zone_id:
                    slots.append(data)
        return slots

    def update_slot_status(self, slot_id: str, status: str) -> bool:
        if slot_id in self.slots_state:
            self.slots_state[slot_id]['status'] = status
            return True
        return False

# --- MOCK MEMORY BANK ---
class MemoryBank:
    """Simulates a long-term memory store for non-operational data."""
    def __init__(self):
        self.frequent_vehicles: Dict[str, int] = {} # vehicle_number -> visit_count
        self.customer_issues: List[Dict] = []
        self.historical_rates: Dict[str, float] = {}
    
    def record_visit(self, vehicle_number: str):
        self.frequent_vehicles[vehicle_number] = self.frequent_vehicles.get(vehicle_number, 0) + 1

print("\nâœ“ Agent Foundation (Enums, Metrics, Mock DB) defined.")


# =============================================================================
# CELL 3: Dummy Data Generation and Cleaning
# =============================================================================

print("\n" + "=" * 80)
print("CELL 3: DUMMY DATA GENERATION")
print("=" * 80)

# 1. SLOTS (parking_slots.csv)
print("1. Generating parking_slots.csv...")
zones = ['A', 'B', 'C', 'D', 'E']
zone_config = {
    'A': {'capacity': 150, 'rate_multiplier': 1.2, 'type': 'premium'}, # High price, high demand
    'B': {'capacity': 100, 'rate_multiplier': 1.0, 'type': 'standard'},
    'C': {'capacity': 100, 'rate_multiplier': 1.0, 'type': 'standard'},
    'D': {'capacity': 100, 'rate_multiplier': 0.8, 'type': 'economy'}, # Low price, low demand
    'E': {'capacity': 50, 'rate_multiplier': 1.5, 'type': 'EV_only'}   # Special
}
slots_data = []
for zone_id, config in zone_config.items():
    for i in range(config['capacity']):
        slots_data.append({
            'slot_id': f"{zone_id}-{i+1:03d}",
            'zone_id': zone_id,
            'status': np.random.choice(['available', 'occupied'], p=[0.2, 0.8]),
            'slot_type': config['type']
        })

parking_slots = pd.DataFrame(slots_data)
parking_slots.to_csv(f"{DATA_PATH}parking_slots.csv", index=False)
print(f"   Shape: {parking_slots.shape}")
zones_data = list(zone_config.values()) 

# 2. LOGS (parking_logs_25000.csv, parking_logs_cleaned.csv)
print("2. Generating parking_logs_25000.csv (Raw Logs)...")
vehicle_numbers = [f"KA{i:02d}XX{np.random.randint(1000, 9999)}" for i in range(1000)]
start_time = datetime.now() - timedelta(days=90)
end_time = datetime.now()

log_data = []
for i in range(N_LOGS_RAW):
    entry = start_time + timedelta(seconds=np.random.randint(0, (end_time - start_time).total_seconds()))
    duration_hours = np.random.lognormal(mean=0.5, sigma=0.5) * 1.5 # Skewed towards shorter stays
    exit_ = entry + timedelta(hours=duration_hours)
    
    # Ensure exit time is not in the future for simulation
    if exit_ > datetime.now():
        exit_ = datetime.now() - timedelta(minutes=np.random.randint(1, 60))

    session_id = f"SESS_{i+1:05d}"
    
    # Randomly select a slot
    slot_id = np.random.choice(parking_slots['slot_id'])
    
    log_data.append({
        'session_id': session_id,
        'vehicle_number': np.random.choice(vehicle_numbers),
        'entry_time': entry,
        'exit_time': exit_,
        'assigned_slot': slot_id,
        'duration_minutes': (exit_ - entry).total_seconds() / 60
    })

parking_logs = pd.DataFrame(log_data)
parking_logs.to_csv(f"{DATA_PATH}parking_logs_25000.csv", index=False)
print(f"   Shape: {parking_logs.shape}")

# --- Data Cleaning and Feature Engineering ---
print("\nâš™ï¸� Step 3: Cleaning Logs and Handling Errors...")

# Convert to datetime (crucial for calculations)
parking_logs['entry_time'] = pd.to_datetime(parking_logs['entry_time'])
parking_logs['exit_time'] = pd.to_datetime(parking_logs['exit_time'])

# Error Simulation 1: Missing Entry/Exit Data (3% of logs)
error_indices = parking_logs.sample(frac=0.03).index
parking_logs.loc[error_indices, ['entry_time', 'exit_time']] = np.nan
parking_logs.dropna(subset=['entry_time', 'exit_time'], inplace=True)

# Error Simulation 2: Invalid Duration (Exit before Entry)
invalid_duration_indices = parking_logs[parking_logs['duration_minutes'] <= 0].index
parking_logs.drop(invalid_duration_indices, inplace=True)

print(f"   After cleaning: {parking_logs.shape}")


# --- Cost Calculation Simulation (for historical payment data generation) ---
print("\nâš™ï¸� Step 4: Simulating Final Cost Calculation...")

def calculate_dummy_cost(row, pricing_config):
    """Simulates cost calculation based on the logic in CELL 5's CostCalculatorTool."""
    entry = row['entry_time']
    exit_ = row['exit_time']
    if pd.isna(entry) or pd.isna(exit_):
        return 0.0

    duration = (exit_ - entry).total_seconds() / 60
    
    # Get zone ID from slot (e.g., A-001 -> A)
    slot_id = row.get('assigned_slot', 'A-001')
    zone_id = slot_id.split('-')[0]
    
    # Simple rate mapping for simulation
    rate = 50.0 * pricing_config[zone_id]['rate_multiplier']
    
    # 30 min free duration
    billable_minutes = max(0, duration - 30)
    billable_hours = np.ceil(billable_minutes / 60)
    
    base_cost = billable_hours * rate
    
    # Random member status (10% chance)
    is_member = np.random.rand() < 0.10
    discount = base_cost * 0.20 if is_member else 0
    final_cost = round(base_cost - discount, 2)
    
    return final_cost

# Apply cost calculation
# Uses the already defined zone_config dictionary which maps zone_id to config dict
pricing_config = zone_config 
parking_logs['final_cost'] = parking_logs.apply(
    lambda row: calculate_dummy_cost(row, pricing_config), axis=1
)

# Rename the cleaned DataFrame for saving and consistency
parking_logs_cleaned = parking_logs

# Save the cleaned and enriched data
parking_logs_cleaned.to_csv(f"{DATA_PATH}parking_logs_cleaned.csv", index=False)

print(f"âœ“ Data cleaning and cost simulation complete.")
print(f"  Cleaned logs shape: {parking_logs_cleaned.shape}")
print(f"  Sample of final costs:\n{parking_logs_cleaned[['duration_minutes', 'final_cost']].tail()}")


# 3. PAYMENTS (payments_25000.csv, payments_cleaned.csv)
print("5. Generating payments_25000.csv (Raw Payments)...")
# Prepare payments_raw (map 1:1 with parking_logs)
payments_raw = parking_logs[['session_id', 'vehicle_number', 'entry_time', 'exit_time', 'final_cost']].copy()
payments_raw.rename(columns={'final_cost': 'payment_amount'}, inplace=True)
payments_raw['payment_id'] = [f"PAY_{i+1:05d}" for i in range(len(payments_raw))]
payments_raw['payment_time'] = payments_raw['exit_time'] + timedelta(minutes=np.random.randint(0, 15))

# Introduce payment errors (e.g., missing payment/cancellation)
payments_raw['status'] = 'paid'
payments_raw.loc[payments_raw.sample(frac=0.03).index, 'status'] = 'pending' # 3% pending
payments_raw.loc[payments_raw.sample(frac=0.01).index, 'status'] = 'cancelled' # 1% cancelled

payments_raw.to_csv(f"{DATA_PATH}payments_25000.csv", index=False)
print(f"   Shape: {payments_raw.shape}")

## B. payments_cleaned.csv
print("6. Generating payments_cleaned.csv (Cleaned Payments)...")
payments_cleaned = payments_raw[payments_raw['status'] == 'paid'].copy()
cleaned_session_ids = parking_logs_cleaned['session_id'].unique()
payments_cleaned = payments_cleaned[payments_cleaned['session_id'].isin(cleaned_session_ids)].copy()

payments_cleaned['entry_time'] = pd.to_datetime(payments_cleaned['entry_time'])
payments_cleaned['exit_time'] = pd.to_datetime(payments_cleaned['exit_time'])
payments_cleaned['payment_time'] = pd.to_datetime(payments_cleaned['payment_time'])

# Enhance Payments DF for Analytics Agent
# CORRECTED LINES: Define outcomes and probabilities separately
payment_outcomes = ['Card', 'UPI', 'Cash']
payment_probabilities = [0.6, 0.3, 0.1] # Must sum to 1.0

payments_cleaned['payment_method'] = np.random.choice(
    payment_outcomes, 
    size=len(payments_cleaned), 
    p=payment_probabilities
)

payments_cleaned.to_csv(f"{DATA_PATH}payments_cleaned.csv", index=False)
print(f"   Shape: {payments_cleaned.shape}")

print("\nâœ“ All dummy datasets generated and saved to mock path for project execution!")
print("=" * 80)


# =============================================================================
# CELL 4: Entry Validation and Database Agents
# =============================================================================

print("\n" + "=" * 80)
print("CELL 4: ENTRY VALIDATION AND DATABASE AGENTS")
print("=" * 80)

# Import 'random' for robust choice selection in the allocator agent
import random 

# --- RE-DEFINING THE CRUCIAL MOCK TOOL WITH THE FIX ---
# (Usually in Cell 2, but redefined here to ensure the latest version is used)
class MockParkingDB:
    """Simulates the real-time database for slots."""
    def __init__(self, initial_slots: pd.DataFrame):
        # Use a dictionary for fast lookup and state management
        self.slots_state: Dict[str, Dict] = initial_slots.set_index('slot_id').to_dict('index')

    def get_available_slots(self, zone_id: Optional[str] = None) -> List[Dict]:
        slots = []
        for slot_id, data in self.slots_state.items():
            if data['status'] == 'available':
                if not zone_id or data['zone_id'] == zone_id:
                    # FIX: Explicitly include 'slot_id' in the returned dictionary
                    slot_data_copy = data.copy()
                    slot_data_copy['slot_id'] = slot_id
                    slots.append(slot_data_copy)
        return slots

    def update_slot_status(self, slot_id: str, status: str) -> bool:
        if slot_id in self.slots_state:
            self.slots_state[slot_id]['status'] = status
            return True
        return False
# -----------------------------------------------------


# Initialize global DB and Memory (using data from CELL 3)
# Assumes other necessary classes (AgentMetrics, AgentType, MemoryBank) are available.
parking_db = MockParkingDB(parking_slots)
membership_db = {vn: (i % 5 == 0) for i, vn in enumerate(parking_logs_cleaned['vehicle_number'].unique())}
memory_bank = MemoryBank()
# Pre-populate memory bank with some frequent visitors for simulation
for vn in parking_logs_cleaned['vehicle_number'].unique()[:50]:
    for _ in range(np.random.randint(5, 20)):
        memory_bank.record_visit(vn)

print("âœ“ Parking DB, Membership DB, and Memory Bank initialized.")


# --- Agent 1: EntryValidatorAgent (Sequential) ---
class EntryValidatorAgent:
    """Agent responsible for License Plate Recognition (LPR) and membership check."""
    
    def __init__(self):
        self.agent_type = AgentType.VALIDATOR
        self.agent_id = f"validator_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}"
        self.metrics = AgentMetrics(self.agent_id, self.agent_type)
        
        # Tool Simulation: Mock OCR results based on logs
        self.log_vehicle_numbers = parking_logs_cleaned['vehicle_number'].tolist()
        logger.info(f"Initialized {self.agent_type.value} agent: {self.agent_id}")
    
    def _run_lpr_tool(self, image_path: str) -> Optional[str]:
        """Simulates an OCR/LPR call, randomly returning a valid number or None."""
        if np.random.rand() < 0.95: # 95% success rate
            return np.random.choice(self.log_vehicle_numbers)
        return None

    def validate_entry(self, vehicle_image_path: str, membership_db: Dict[str, bool]) -> Dict:
        """Core workflow: LPR -> Validation -> Membership Check."""
        start_time = datetime.now()
        
        vehicle_number = self._run_lpr_tool(vehicle_image_path)
        
        if not vehicle_number:
            self.metrics.update_failure()
            return {'success': False, 'reason': 'LPR failed to detect plate.'}
        
        # Check for banned list (Simulated by checking for a specific pattern)
        if 'XX00' in vehicle_number: 
            self.metrics.update_failure()
            return {'success': False, 'reason': 'Vehicle is on the restricted list.'}
        
        is_member = membership_db.get(vehicle_number, False)
        
        # Record visit in Memory Bank (for future frequency analysis)
        memory_bank.record_visit(vehicle_number)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        self.metrics.update_success(processing_time)
        
        return {
            'success': True,
            'vehicle_number': vehicle_number,
            'is_member': is_member
        }

entry_validator = EntryValidatorAgent()


# --- Agent 2: SmartSlotAllocatorAgent (Sequential) ---
class SmartSlotAllocatorAgent:
    """Agent responsible for assigning the optimal available parking slot."""
    
    def __init__(self, parking_db: MockParkingDB):
        self.agent_type = AgentType.ALLOCATOR
        self.agent_id = f"allocator_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}"
        self.parking_db = parking_db
        self.metrics = AgentMetrics(self.agent_id, self.agent_type)
        logger.info(f"Initialized {self.agent_type.value} agent: {self.agent_id}")
        
    def allocate_slot(self, vehicle_number: str) -> Dict:
        """Core workflow: Find optimal slot based on availability and heuristics."""
        start_time = datetime.now()
        
        available_slots = self.parking_db.get_available_slots()
        
        if not available_slots:
            self.metrics.update_failure()
            return {'success': False, 'reason': 'No available slots in the facility.'}
        
        # Use random.choice for robust selection of a dictionary object
        chosen_slot = random.choice(available_slots)

        # Heuristic 1: Prioritize EV slots if vehicle is known to be EV (simulated)
        if vehicle_number.endswith('EV'):
            ev_slots = [s for s in available_slots if s['slot_type'] == 'EV_only']
            if ev_slots:
                chosen_slot = random.choice(ev_slots)
        
        # Heuristic 2: For frequent visitors (simulated check on MemoryBank)
        elif memory_bank.frequent_vehicles.get(vehicle_number, 0) > 10:
            # Assign them to a zone with historically lower average duration (Zone C/D)
            preferred_slots = [s for s in available_slots if s['zone_id'] in ['C', 'D']]
            if preferred_slots:
                chosen_slot = random.choice(preferred_slots)
        
        # Update DB status
        # This line is now safe because get_available_slots now includes 'slot_id'.
        self.parking_db.update_slot_status(chosen_slot['slot_id'], 'occupied')
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        self.metrics.update_success(processing_time)
        
        return {
            'success': True,
            'slot': chosen_slot
        }

slot_allocator = SmartSlotAllocatorAgent(parking_db)
print("âœ“ EntryValidator and SmartSlotAllocator agents initialized.")


# =============================================================================
# CELL 5: Monitoring and Payment Agents
# =============================================================================

print("\n" + "=" * 80)
print("CELL 5: MONITORING AND PAYMENT AGENTS")
print("=" * 80)

# --- Agent 3: OccupancyMonitorAgent (Parallel) ---
class OccupancyMonitorAgent:
    """Agent responsible for parallel checks on system health and real-time occupancy."""
    
    def __init__(self, parking_db: MockParkingDB, memory_bank: MemoryBank):
        self.agent_type = AgentType.MONITOR
        self.agent_id = f"monitor_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}"
        self.parking_db = parking_db
        self.memory_bank = memory_bank
        self.metrics = AgentMetrics(self.agent_id, self.agent_type)
        logger.info(f"Initialized {self.agent_type.value} agent: {self.agent_id}")
        
    def _check_sensor_pings(self) -> Dict:
        """Simulates parallel sensor data aggregation."""
        # Using a global constant that should be defined in Cell 1 or 3
        if 'N_SLOTS' in globals():
            total_slots = N_SLOTS 
        else:
            total_slots = len(self.parking_db.slots_state)
            
        occupied_slots = len([s for s in self.parking_db.slots_state.values() if s['status'] == 'occupied'])
        return {
            'occupied_slots': occupied_slots,
            'occupancy_rate': (occupied_slots / total_slots) * 100
        }
        
    def _analyze_recent_events(self) -> Dict:
        """Simulates log processing for recent activity."""
        # Since we're in a single script run, this just returns dummy data
        recent_entries = np.random.randint(2, 5)
        recent_exits = np.random.randint(1, 4)
        return {
            'recent_count': recent_entries,
            'anomalies_detected': np.random.rand() < 0.05 # 5% chance of anomaly
        }
        
    def monitor_parallel(self) -> Dict:
        """Executes parallel monitoring tasks and aggregates results."""
        start_time = datetime.now()
        
        # Simulate parallel execution of checks
        sensor_status = self._check_sensor_pings()
        entry_events = self._analyze_recent_events()
        exit_events = self._analyze_recent_events()
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        self.metrics.update_success(processing_time)
        
        return {
            'sensor_pings': sensor_status,
            'entry_events': entry_events,
            'exit_events': exit_events,
            'last_check_ms': processing_time
        }

occupancy_monitor = OccupancyMonitorAgent(parking_db, memory_bank)


# --- Agent 4: PaymentReminderAgent (Loop) ---
class CostCalculatorTool:
    """Tool used by PaymentReminderAgent for cost logic (similar to CELL 3's logic)."""
    def __init__(self, pricing_config: Dict):
        self.pricing_config = pricing_config
        self.base_rate = 50.0 # Base rate per billable hour

    def calculate_cost(self, session: ParkingSession, exit_time: datetime) -> Dict:
        duration_minutes = (exit_time - session.entry_time).total_seconds() / 60
        zone_id = session.zone
        
        # Access rate_multiplier using zone_id as the key in pricing_config
        rate = self.base_rate * self.pricing_config[zone_id]['rate_multiplier']
        
        # Logic: 30 min free, billable hours (rounded up)
        billable_minutes = max(0, duration_minutes - 30)
        billable_hours = np.ceil(billable_minutes / 60)
        
        base_cost = billable_hours * rate
        
        discount = 0
        if session.context.get('is_member'):
            discount = base_cost * 0.20 # 20% member discount
            
        final_cost = round(base_cost - discount, 2)
        
        return {
            'amount': final_cost,
            'cost_breakdown': {
                'duration_minutes': int(duration_minutes),
                'billable_hours': int(billable_hours),
                'base_rate': rate,
                'discount': round(discount, 2)
            }
        }


class PaymentReminderAgent:
    """Agent responsible for initiating payment and sending reminders/invoices."""
    
    def __init__(self, pricing_config: Dict):
        self.agent_type = AgentType.REMINDER
        self.agent_id = f"payment_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}"
        self.cost_calculator = CostCalculatorTool(pricing_config)
        self.metrics = AgentMetrics(self.agent_id, self.agent_type)
        logger.info(f"Initialized {self.agent_type.value} agent: {self.agent_id}")
        
    def process_exit(self, session: ParkingSession) -> Dict:
        """Core workflow: Calculate cost and generate payment ID."""
        start_time = datetime.now()
        
        cost_result = self.cost_calculator.calculate_cost(session, session.exit_time)
        amount = cost_result['amount']
        
        # Simulate payment ID generation
        payment_id = f"PAY_{hashlib.md5(f'{session.session_id}{datetime.now()}'.encode()).hexdigest()[:10]}"
        
        # Simulate sending invoice/reminder (Loop functionality is simulated here)
        logger.info(f"Reminder loop: Sending invoice for {session.vehicle_number}, amount: â‚¹{amount}")
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        self.metrics.update_success(processing_time)
        
        return {
            'success': True,
            'amount': amount,
            'payment_id': payment_id,
            'cost_breakdown': cost_result['cost_breakdown']
        }
        
# FIX: Use zone_config (zone_id: config dict) which was defined in Cell 3
# NOTE: This assumes zone_config is a global variable available here.
pricing_config = zone_config 
payment_reminder = PaymentReminderAgent(pricing_config)
print("âœ“ OccupancyMonitor and PaymentReminder agents initialized.")


# =============================================================================
# CELL 6: Orchestrator Definition
# =============================================================================

print("\n" + "=" * 80)
print("CELL 6: PARKING ORCHESTRATOR AGENT - DEFINITION")
print("=" * 80)

class ParkingOrchestratorAgent:
    """
    Main orchestrator that coordinates all sub-agents.
    Manages sessions, delegates tasks, and maintains state.
    """
    
    def __init__(
        self,
        entry_validator: EntryValidatorAgent,
        slot_allocator: SmartSlotAllocatorAgent,
        occupancy_monitor: OccupancyMonitorAgent,
        payment_reminder: PaymentReminderAgent,
        memory_bank: MemoryBank
    ):
        self.agent_type = AgentType.ORCHESTRATOR
        self.agent_id = f"orchestrator_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}"
        
        # Sub-agents
        self.entry_validator = entry_validator
        self.slot_allocator = slot_allocator
        self.occupancy_monitor = occupancy_monitor
        self.payment_reminder = payment_reminder
        
        # Memory and state
        self.memory_bank = memory_bank
        self.active_sessions: Dict[str, ParkingSession] = {}
        self.metrics = AgentMetrics(self.agent_id, self.agent_type)
        
        logger.info(f"Initialized {self.agent_type.value} agent: {self.agent_id}")
        print(f"\nğŸ�¯ Orchestrator {self.agent_id} is now active")
    
    def process_vehicle_entry(self, vehicle_image_path: str) -> Dict:
        """
        Complete vehicle entry workflow.
        Delegates to: EntryValidator â†’ SlotAllocator
        """
        logger.info(f"{self.agent_id}: Processing vehicle entry")
        start_time = datetime.now()
        
        # Step 1: Validate entry
        validation_result = self.entry_validator.validate_entry(
            vehicle_image_path, 
            membership_db
        )
        
        if not validation_result['success']:
            self.metrics.update_failure()
            return {
                'success': False,
                'stage': 'validation',
                'reason': validation_result.get('reason', 'Validation failed')
            }
        
        vehicle_number = validation_result['vehicle_number']
        
        # Step 2: Allocate slot
        allocation_result = self.slot_allocator.allocate_slot(vehicle_number)
        
        if not allocation_result['success']:
            self.metrics.update_failure()
            return {
                'success': False,
                'stage': 'allocation',
                'reason': allocation_result.get('reason', 'No slots available')
            }
        
        # Step 3: Create session
        session_id = f"SESSION_{hashlib.md5(f'{vehicle_number}{datetime.now()}'.encode()).hexdigest()[:12]}"
        
        session = ParkingSession(
            session_id=session_id,
            vehicle_number=vehicle_number,
            status=VehicleStatus.PARKED,
            entry_time=datetime.now(),
            assigned_slot=allocation_result['slot']['slot_id'],
            zone=allocation_result['slot']['zone_id'],
            context={
                'is_member': validation_result['is_member'],
                'slot_type': allocation_result['slot'].get('slot_type', 'standard')
            }
        )
        
        self.active_sessions[session_id] = session
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        self.metrics.update_success(processing_time)
        
        result = {
            'success': True,
            'session_id': session_id,
            'vehicle_number': vehicle_number,
            'assigned_slot': session.assigned_slot,
            'zone': session.zone,
            'is_member': validation_result['is_member'],
            'processing_time_ms': processing_time
        }
        
        logger.info(f"{self.agent_id}: Entry processed - {vehicle_number} â†’ {session.assigned_slot}")
        return result
    
    def process_vehicle_exit(self, session_id: str) -> Dict:
        """
        Complete vehicle exit workflow.
        Delegates to: PaymentReminder
        """
        logger.info(f"{self.agent_id}: Processing vehicle exit")
        start_time = datetime.now()
        
        if session_id not in self.active_sessions:
            self.metrics.update_failure()
            return {
                'success': False,
                'reason': 'Session not found'
            }
        
        session = self.active_sessions[session_id]
        session.exit_time = datetime.now()
        session.status = VehicleStatus.PAYMENT_PENDING
        
        # Calculate payment
        payment_result = self.payment_reminder.process_exit(session)
        
        if not payment_result['success']:
            self.metrics.update_failure()
            return {
                'success': False,
                'stage': 'payment',
                'reason': payment_result.get('reason', 'Payment calculation failed')
            }
        
        # Update slot status
        parking_db.update_slot_status(session.assigned_slot, 'available')
        
        # Clean up session and record completion
        del self.active_sessions[session_id]
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        self.metrics.update_success(processing_time)
        
        result = {
            'success': True,
            'session_id': session_id,
            'vehicle_number': session.vehicle_number,
            'duration_minutes': payment_result['cost_breakdown']['duration_minutes'],
            'payment_amount': payment_result['amount'],
            'payment_id': payment_result['payment_id']
        }
        
        logger.info(f"{self.agent_id}: Exit processed - {session.vehicle_number} - â‚¹{payment_result['amount']:.2f}")
        return result
    
    def get_system_status(self) -> Dict:
        """
        Get overall system status with agent metrics.
        Feature: Observability
        """
        # Collect metrics from all agents
        agent_metrics = {
            'orchestrator': {
                'success_rate': self.metrics.get_success_rate(),
                'avg_processing_time_ms': self.metrics.avg_processing_time_ms,
                'total_requests': self.metrics.total_requests
            },
            'entry_validator': {
                'success_rate': self.entry_validator.metrics.get_success_rate(),
                'avg_processing_time_ms': self.entry_validator.metrics.avg_processing_time_ms,
                'total_requests': self.entry_validator.metrics.total_requests
            },
            'slot_allocator': {
                'success_rate': self.slot_allocator.metrics.get_success_rate(),
                'avg_processing_time_ms': self.slot_allocator.metrics.avg_processing_time_ms,
                'total_requests': self.slot_allocator.metrics.total_requests
            },
            'payment_reminder': {
                'success_rate': self.payment_reminder.metrics.get_success_rate(),
                'avg_processing_time_ms': self.payment_reminder.metrics.avg_processing_time_ms,
                'total_requests': self.payment_reminder.metrics.total_requests
            }
        }
        
        # Get occupancy status
        occupancy_status = self.occupancy_monitor.monitor_parallel()
        
        return {
            'active_sessions': len(self.active_sessions),
            'agent_metrics': agent_metrics,
            'occupancy': occupancy_status,
            'memory_stats': {
                'frequent_vehicles': len(self.memory_bank.frequent_vehicles),
                'recorded_issues': len(self.memory_bank.customer_issues)
            }
        }

print("âœ“ Orchestrator class defined.")


# =============================================================================
# CELL 7: Parking Orchestrator Agent - Main Controller (Initialization)
# =============================================================================

print("\n" + "=" * 80)
print("CELL 7: ORCHESTRATOR INITIALIZATION")
print("=" * 80)

# Initialize orchestrator
orchestrator = ParkingOrchestratorAgent(
    entry_validator=entry_validator,
    slot_allocator=slot_allocator,
    occupancy_monitor=occupancy_monitor,
    payment_reminder=payment_reminder,
    memory_bank=memory_bank
)

print(f"\nâœ“ Orchestrator initialized and ready")
print(f"  Agent ID: {orchestrator.agent_id}")
print(f"  Sub-agents: 4 active")
print(f"  Memory bank: Ready")


# =============================================================================
# CELL 8: ORCHESTRATOR AGENT AND ENTRY SIMULATION
# =============================================================================

print("\n" + "=" * 80)
print("CELL 8: ORCHESTRATOR AGENT AND ENTRY SIMULATION")
print("=" * 80)

# NOTE: Need to ensure 'timedelta' is imported if it wasn't in Cell 1
from datetime import datetime, timedelta 
import hashlib
from typing import Dict, List, Optional
import numpy as np
import time # Ensure time is available if the simulation loop is moved here

# --- Agent 5: OrchestratorAgent (Main Control) ---
class OrchestratorAgent:
    """The central agent responsible for sequencing the entry/exit workflows."""
    
    def __init__(self, entry_validator: EntryValidatorAgent, slot_allocator: SmartSlotAllocatorAgent, 
                 monitor: OccupancyMonitorAgent, reminder: PaymentReminderAgent, 
                 parking_db: MockParkingDB):
        self.agent_type = AgentType.ORCHESTRATOR
        self.agent_id = f"orchestrator_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}"
        self.entry_validator = entry_validator
        self.slot_allocator = slot_allocator
        self.monitor = monitor
        self.reminder = reminder
        self.parking_db = parking_db
        self.metrics = AgentMetrics(self.agent_id, self.agent_type)
        self.active_sessions: Dict[str, ParkingSession] = {} # vehicle_number -> ParkingSession
        logger.info(f"Initialized {self.agent_type.value} agent: {self.agent_id}")

    # FIX: Add the missing method required by Cell 10
    def get_system_status(self) -> Dict:
        """Delegates the real-time monitoring task to the OccupancyMonitorAgent."""
        # This calls the monitor's parallel check
        return self.monitor.monitor_parallel()

    def process_vehicle_entry(self, vehicle_image_path: str) -> Dict:
        """
        Main workflow for a vehicle entering the facility.
        1. Validate entry and get vehicle details.
        2. Allocate slot.
        3. Create and store a new session.
        """
        start_time = datetime.now()
        
        # Step 1: Validate entry
        # NOTE: membership_db is assumed to be a global variable defined in Cell 4
        validation_result = self.entry_validator.validate_entry(vehicle_image_path, membership_db)
        
        if not validation_result['success']:
            self.metrics.update_failure()
            return {'success': False, 'reason': f"Entry failed: {validation_result['reason']}"}
        
        vehicle_number = validation_result['vehicle_number']
        
        # Step 2: Allocate slot
        allocation_result = self.slot_allocator.allocate_slot(vehicle_number)
        
        if not allocation_result['success']:
            self.metrics.update_failure()
            return {'success': False, 'reason': f"Allocation failed: {allocation_result['reason']}"}
            
        chosen_slot = allocation_result['slot']
        
        # Step 3: Create and store session
        session_id = f"RUN_{hashlib.md5(f'{vehicle_number}{datetime.now()}'.encode()).hexdigest()[:10]}"
        
        new_session = ParkingSession(
            session_id=session_id,
            vehicle_number=vehicle_number,
            status=VehicleStatus.PARKED,
            entry_time=datetime.now(),
            assigned_slot=chosen_slot['slot_id'],
            zone=chosen_slot['zone_id'],
            context={'is_member': validation_result['is_member']}
        )
        self.active_sessions[vehicle_number] = new_session
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        self.metrics.update_success(processing_time)

        return {
            'success': True,
            'session_id': session_id,
            'slot_id': chosen_slot['slot_id'],
            'zone': chosen_slot['zone_id'],
            'time_ms': processing_time
        }

    def process_vehicle_exit(self, vehicle_number: str) -> Dict:
        """
        Main workflow for a vehicle exiting the facility.
        1. Retrieve session.
        2. Calculate cost (via PaymentReminder agent).
        3. Clear session and update slot status.
        """
        start_time = datetime.now()

        if vehicle_number not in self.active_sessions:
            self.metrics.update_failure()
            return {'success': False, 'reason': f"No active session found for {vehicle_number}."}

        session = self.active_sessions.pop(vehicle_number)
        # Simulate a stay duration for cost calculation
        session.exit_time = datetime.now() + timedelta(minutes=np.random.randint(10, 180)) 
        session.status = VehicleStatus.PAYMENT_PENDING
        
        # Step 1: Calculate cost and initiate payment process
        payment_result = self.reminder.process_exit(session)
        
        # Step 2: Clear slot and update session status to EXIT
        self.parking_db.update_slot_status(session.assigned_slot, 'available')
        session.status = VehicleStatus.EXITED
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        self.metrics.update_success(processing_time)

        return {
            'success': True,
            'vehicle_number': vehicle_number,
            'cost': payment_result['amount'],
            'time_ms': processing_time
        }


# Initialize the Orchestrator with all defined agents
orchestrator = OrchestratorAgent(
    entry_validator, 
    slot_allocator, 
    occupancy_monitor, 
    payment_reminder, 
    parking_db
)

print(f"âœ“ Orchestrator Agent ({orchestrator.agent_id}) initialized, ready for simulation.")

# --- SIMULATION PREPARATION (For the next cell) ---
# Prepare dummy image paths (using vehicle numbers as proxies for images)
all_vehicles = parking_logs_cleaned['vehicle_number'].unique()
# Simulate a batch of 20 vehicles entering
N_SIMULATION_ENTRIES = 20
dummy_image_paths = np.random.choice(all_vehicles, N_SIMULATION_ENTRIES, replace=False)

print(f"\nSimulation ready. Preparing to process {N_SIMULATION_ENTRIES} vehicle entries in the next cell (Cell 10).")
print("--------------------------------------------------------------------------------")


# =============================================================================
# CELL 9: ANALYTICS AGENT AND REPORT GENERATION
# =============================================================================

print("\n" + "=" * 80)
print("CELL 9: ANALYTICS AGENT AND REPORT GENERATION")
print("=" * 80)

# FIX: Ensure 'json' is defined for the print statements later in the cell
import json 

# Load dataframes required by the Analytics Agent
try:
    # Assumes these files were saved in Cell 3
    parking_logs = pd.read_csv(f"{DATA_PATH}parking_logs_cleaned.csv")
    payments_data = pd.read_csv(f"{DATA_PATH}payments_cleaned.csv")
    slots_data = pd.read_csv(f"{DATA_PATH}parking_slots.csv")
    print("âœ“ Analytics data loaded successfully.")
    
    # --- Data Preparation for Analysis ---
    # The logs need the zone_id column for grouping, extracted from 'assigned_slot'
    parking_logs['zone_id'] = parking_logs['assigned_slot'].apply(lambda x: x.split('-')[0])
    print("âœ“ 'zone_id' column extracted and added to parking logs for analysis.")
    # -----------------------------------

except Exception as e:
    print(f"Error loading analytics data: {e}. Ensure Cell 3 ran successfully.")
    parking_logs = pd.DataFrame() # Create empty DF to prevent crashing


# --- Agent 6: AnalyticsAgent (Report Generator) ---
class AnalyticsAgent:
    """Agent responsible for generating historical and performance reports."""
    
    def __init__(self, logs: pd.DataFrame, payments: pd.DataFrame, slots: pd.DataFrame):
        self.agent_type = AgentType.ANALYTICS
        self.agent_id = f"analytics_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}"
        self.logs = logs
        self.payments = payments
        self.slots = slots
        self.metrics = AgentMetrics(self.agent_id, self.agent_type)
        logger.info(f"Initialized {self.agent_type.value} agent: {self.agent_id}")

    def generate_occupancy_report(self) -> Dict:
        """Generates a summary of parking activity and utilization."""
        start_time = datetime.now()
        
        if self.logs.empty:
            return {'status': 'error', 'message': 'No log data available for report.'}
            
        # 1. Total traffic
        total_sessions = len(self.logs)
        unique_vehicles = self.logs['vehicle_number'].nunique()
        
        # 2. Daily average traffic
        self.logs['entry_date'] = pd.to_datetime(self.logs['entry_time']).dt.date
        num_days = self.logs['entry_date'].nunique()
        avg_daily_sessions = round(total_sessions / num_days, 1) if num_days > 0 else 0
        
        # 3. Zone-wise duration
        zone_duration = self.logs.groupby('zone_id')['duration_minutes'].mean().round(1).to_dict()
        
        # 4. Slot utilization
        slot_traffic = self.logs['assigned_slot'].value_counts()
        most_used_slot = slot_traffic.index[0]
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        self.metrics.update_success(processing_time)
        
        return {
            'status': 'success',
            'period_days': num_days,
            'total_sessions': total_sessions,
            'unique_vehicles': unique_vehicles,
            'avg_daily_sessions': avg_daily_sessions,
            'avg_duration_by_zone_min': zone_duration,
            'most_used_slot': most_used_slot,
            'processing_time_ms': processing_time
        }

    def generate_financial_report(self) -> Dict:
        """Generates a summary of revenue and payment methods."""
        start_time = datetime.now()
        
        if self.payments.empty:
            return {'status': 'error', 'message': 'No payment data available for report.'}

        # 1. Total Revenue
        total_revenue = self.payments['payment_amount'].sum().round(2)
        
        # 2. Avg transaction value
        avg_transaction = self.payments['payment_amount'].mean().round(2)
        
        # 3. Payment method split
        payment_split = (self.payments['payment_method'].value_counts(normalize=True) * 100).round(1).to_dict()
        
        # 4. Member vs Non-member revenue (requires joining logs back)
        revenue_df = self.payments.merge(
            self.logs[['session_id', 'zone_id']], 
            on='session_id', 
            how='left'
        )
        # Note: 'is_member' flag is not directly in payments/logs, so we skip member analysis for this mock.
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        self.metrics.update_success(processing_time)
        
        return {
            'status': 'success',
            'total_revenue_inr': total_revenue,
            'total_transactions': len(self.payments),
            'avg_transaction_inr': avg_transaction,
            'payment_method_split_pct': payment_split,
            'processing_time_ms': processing_time
        }

# Initialize the Analytics Agent
analytics_agent = AnalyticsAgent(parking_logs, payments_data, slots_data)

print("\n--- Generating Reports ---")
# 1. Occupancy Report
occupancy_report = analytics_agent.generate_occupancy_report()
print("\n--- Occupancy and Traffic Report ğŸ“Š ---")
# Use the imported json module
print(json.dumps(occupancy_report, indent=4))

# 2. Financial Report
financial_report = analytics_agent.generate_financial_report()
print("\n--- Financial Report ğŸ’° ---")
# Use the imported json module
print(json.dumps(financial_report, indent=4))
print("----------------------------------------")


# =============================================================================
# CELL 10: SIMULATION RESULTS AND CONCLUSION
# =============================================================================

print("\n" + "=" * 80)
print("CELL 10: SIMULATION RESULTS AND CONCLUSION")
print("=" * 80)

# FIX: Import the 'time' module for time.sleep()
import time 
import json # Also ensuring json is imported if needed for the print statements

# --- Entry Simulation ---
entry_results = []
# NOTE: N_SIMULATION_ENTRIES and dummy_image_paths are assumed to be defined in Cell 8
print(f"--- Running Entry Simulation for {N_SIMULATION_ENTRIES} Vehicles ---")

for i in range(N_SIMULATION_ENTRIES):
    # Use the dummy vehicle number as a proxy for the image path
    image_path = dummy_image_paths[i] 
    
    # Process the entry using the Orchestrator
    result = orchestrator.process_vehicle_entry(image_path)
    
    # Simulate a small delay for realistic asynchronous processing
    time.sleep(0.01) 
    
    entry_results.append(result)

# --- Exit Simulation (Simulating Exits for half the sessions) ---
exit_results = []
vehicles_to_exit = list(orchestrator.active_sessions.keys())[:N_SIMULATION_ENTRIES // 2]
print(f"--- Running Exit Simulation for {len(vehicles_to_exit)} Vehicles ---")

for vehicle in vehicles_to_exit:
    result = orchestrator.process_vehicle_exit(vehicle)
    exit_results.append(result)
    time.sleep(0.01)

print("âœ“ Entry and Exit simulations complete.")

# --- Aggregating Metrics ---
# NOTE: Agents are assumed to be defined in previous cells (e.g., orchestrator, entry_validator, etc.)
all_metrics = [
    orchestrator.metrics, 
    entry_validator.metrics, 
    slot_allocator.metrics, 
    occupancy_monitor.metrics, 
    payment_reminder.metrics,
    analytics_agent.metrics
]

# Calculate total requests processed
total_requests = sum(m.total_requests for m in all_metrics)
total_success = sum(m.successful_requests for m in all_metrics)

# FIX: Use the built-in round() function instead of the .round() method
overall_success_rate = round((total_success / total_requests * 100), 2) if total_requests > 0 else 0.0

# Calculate average time per agent
agent_performance = {}
for m in all_metrics:
    agent_performance[m.agent_type.value] = {
        'avg_time_ms': round(m.avg_processing_time_ms, 2),
        'success_rate_pct': round(m.get_success_rate(), 2), # AgentMetrics.get_success_rate should return float
        'total_requests': m.total_requests
    }

print("\n--- Summary of Agent Performance ---")
print(json.dumps(agent_performance, indent=4))
print(f"\nOverall Agent Success Rate: **{overall_success_rate}%** across {total_requests} requests.")

# --- Final Conclusion and Key Takeaways ---
print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

# Re-run Occupancy report to reflect simulation activity 
occupancy_report = analytics_agent.generate_occupancy_report() 

print("\n**Key Takeaways:**")
# NOTE: The Orchestrator's get_system_status() method is assumed to be fixed in Cell 8
print(f"1.  **Live System Status:** {orchestrator.get_system_status()['sensor_pings']['occupied_slots']} slots are currently occupied.") 
# NOTE: financial_report is assumed to be available from Cell 9
print(f"2.  **Revenue Insight:** Total simulated revenue for this historical period was **â‚¹{financial_report['total_revenue_inr']}**.") 
print(f"3.  **Efficiency:** All agents achieved high success rates in the simulated environment.")
print(f"4.  **Overall Throughput:** Processed **{total_requests}** agent requests with an **{overall_success_rate}%** success rate.")

print("\nâœ“ Agent simulation completed. The system demonstrated robust functionality and high efficiency.")
print("=" * 80)

