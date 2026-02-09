"""
Household Type Definitions

Defines 5 distinct household profiles with different:
- Occupancy patterns
- Device ownership
- Activity schedules
- Energy consumption characteristics
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple
from enum import Enum


class HouseholdType(Enum):
    SINGLE_PROFESSIONAL = "single_professional"
    YOUNG_COUPLE = "young_couple"
    FAMILY_WITH_CHILDREN = "family_with_children"
    LARGE_FAMILY = "large_family"
    RETIRED_COUPLE = "retired_couple"


@dataclass
class ActivityPeriod:
    """Represents a period of household activity"""
    start_hour: int
    end_hour: int
    occupancy_prob: float  # Probability someone is home and active
    

@dataclass
class HouseholdProfile:
    """Complete household profile"""
    name: str
    household_type: HouseholdType
    num_occupants: int
    description: str
    
    # Device ownership (device_id -> quantity)
    devices: Dict[str, int]
    
    # Activity periods (weekday)
    weekday_schedule: List[ActivityPeriod]
    
    # Activity periods (weekend)
    weekend_schedule: List[ActivityPeriod]
    
    # Device usage multipliers (device_id -> multiplier)
    device_multipliers: Dict[str, float]
    
    # Has EV
    has_ev: bool = False
    

# ==================== HOUSEHOLD PROFILES ====================

SINGLE_PROFESSIONAL = HouseholdProfile(
    name="Single Professional",
    household_type=HouseholdType.SINGLE_PROFESSIONAL,
    num_occupants=1,
    description="Working adult, 25-45 years old, away during work hours",
    
    devices={
        # Always on
        "refrigerator": 1,
        "wifi_router": 1,
        "security_system": 1,
        
        # HVAC
        "heat_pump": 1,
        "water_heater": 1,
        
        # Cooking
        "electric_stove": 1,
        "microwave": 1,
        "kettle": 1,
        "coffee_maker": 1,
        "toaster": 1,
        
        # Laundry
        "washing_machine": 1,
        
        # Entertainment
        "tv_living_room": 1,
        "tv_bedroom": 1,
        "laptop": 1,
        
        # Lighting
        "light_living_room": 1,
        "light_kitchen": 1,
        "light_bedroom": 1,
        "light_bathroom": 1,
        
        # Personal care
        "hair_dryer": 1,
        
        # EV
        "ev_charger_level2": 1,
    },
    
    weekday_schedule=[
        ActivityPeriod(0, 6, 0.9),    # Sleeping
        ActivityPeriod(6, 8, 0.95),   # Morning routine
        ActivityPeriod(8, 17, 0.1),   # At work
        ActivityPeriod(17, 19, 0.9),  # Evening activities
        ActivityPeriod(19, 23, 0.95), # Home evening
        ActivityPeriod(23, 24, 0.9),  # Late evening
    ],
    
    weekend_schedule=[
        ActivityPeriod(0, 8, 0.9),    # Sleeping in
        ActivityPeriod(8, 12, 0.8),   # Morning activities
        ActivityPeriod(12, 18, 0.6),  # Out and about
        ActivityPeriod(18, 24, 0.9),  # Home evening
    ],
    
    device_multipliers={
        "washing_machine": 0.5,  # Less laundry
        "dishwasher": 0.4,       # Less dishes
        "gaming_console": 0.3,   # Occasional gaming
    },
    
    has_ev=True,
)


YOUNG_COUPLE = HouseholdProfile(
    name="Young Couple",
    household_type=HouseholdType.YOUNG_COUPLE,
    num_occupants=2,
    description="Two working adults, 25-40 years old, active lifestyle",
    
    devices={
        # Always on
        "refrigerator": 1,
        "freezer": 1,
        "wifi_router": 1,
        "security_system": 1,
        
        # HVAC
        "heat_pump": 1,
        "water_heater": 1,
        
        # Cooking
        "electric_stove": 1,
        "oven": 1,
        "microwave": 1,
        "kettle": 1,
        "coffee_maker": 1,
        "toaster": 1,
        
        # Major appliances
        "washing_machine": 1,
        "dryer": 1,
        "dishwasher": 1,
        
        # Entertainment
        "tv_living_room": 1,
        "tv_bedroom": 1,
        "gaming_console": 1,
        "stereo_system": 1,
        "laptop": 2,
        
        # Lighting
        "light_living_room": 1,
        "light_kitchen": 1,
        "light_bedroom": 1,
        "light_bathroom": 1,
        "light_outdoor": 1,
        
        # Cleaning
        "vacuum_cleaner": 1,
        "iron": 1,
        
        # Personal care
        "hair_dryer": 1,
        "electric_shaver": 1,
        
        # EV
        "ev_charger_level2": 1,
    },
    
    weekday_schedule=[
        ActivityPeriod(0, 6, 0.95),   # Sleeping
        ActivityPeriod(6, 9, 0.9),    # Staggered morning routines
        ActivityPeriod(9, 17, 0.15),  # At work
        ActivityPeriod(17, 20, 0.9),  # Evening cooking/activities
        ActivityPeriod(20, 24, 0.95), # Home evening
    ],
    
    weekend_schedule=[
        ActivityPeriod(0, 9, 0.9),    # Sleeping in
        ActivityPeriod(9, 12, 0.85),  # Breakfast, chores
        ActivityPeriod(12, 17, 0.5),  # Out for lunch/activities
        ActivityPeriod(17, 24, 0.9),  # Home evening, cooking
    ],
    
    device_multipliers={
        "gaming_console": 1.2,
        "stereo_system": 1.3,
        "oven": 1.2,  # More cooking
    },
    
    has_ev=True,
)


FAMILY_WITH_CHILDREN = HouseholdProfile(
    name="Family with Children",
    household_type=HouseholdType.FAMILY_WITH_CHILDREN,
    num_occupants=4,  # 2 adults + 2 children
    description="Family with 2-3 kids (ages 5-15), high activity level",
    
    devices={
        # Always on
        "refrigerator": 1,
        "freezer": 1,
        "wifi_router": 1,
        "security_system": 1,
        
        # HVAC
        "heat_pump": 1,
        "boiler": 1,
        "water_heater": 1,
        
        # Cooking
        "electric_stove": 1,
        "oven": 1,
        "microwave": 1,
        "kettle": 1,
        "coffee_maker": 1,
        "toaster": 1,
        
        # Major appliances
        "washing_machine": 1,
        "dryer": 1,
        "dishwasher": 1,
        
        # Entertainment
        "tv_living_room": 1,
        "tv_bedroom": 1,
        "gaming_console": 1,
        "stereo_system": 1,
        "desktop_computer": 1,
        "laptop": 2,
        "printer": 1,
        
        # Lighting
        "light_living_room": 1,
        "light_kitchen": 1,
        "light_bedroom": 2,  # Multiple bedrooms
        "light_bathroom": 1,
        "light_outdoor": 1,
        
        # Cleaning
        "vacuum_cleaner": 1,
        "iron": 1,
        
        # Personal care
        "hair_dryer": 1,
        "electric_shaver": 1,
    },
    
    weekday_schedule=[
        ActivityPeriod(0, 6, 0.95),   # Sleeping
        ActivityPeriod(6, 9, 0.98),   # Busy morning (breakfast, school prep)
        ActivityPeriod(9, 15, 0.3),   # Parents work, kids at school
        ActivityPeriod(15, 17, 0.8),  # Kids home from school
        ActivityPeriod(17, 21, 0.98), # Family dinner, homework, activities
        ActivityPeriod(21, 24, 0.9),  # Evening wind-down
    ],
    
    weekend_schedule=[
        ActivityPeriod(0, 8, 0.95),   # Sleeping
        ActivityPeriod(8, 12, 0.9),   # Breakfast, cleaning
        ActivityPeriod(12, 18, 0.6),  # Mixed (some out, some home)
        ActivityPeriod(18, 22, 0.95), # Family time
        ActivityPeriod(22, 24, 0.9),  # Evening
    ],
    
    device_multipliers={
        "washing_machine": 1.8,  # Lots of laundry
        "dryer": 1.8,
        "dishwasher": 1.5,
        "microwave": 1.4,
        "gaming_console": 1.8,
        "light_bedroom": 1.5,
        "water_heater": 1.4,
    },
    
    has_ev=False,  # No EV for this profile
)


LARGE_FAMILY = HouseholdProfile(
    name="Large Family",
    household_type=HouseholdType.LARGE_FAMILY,
    num_occupants=6,  # 2 adults + 4 children
    description="Large family with 3-4 kids, very high energy usage",
    
    devices={
        # Always on
        "refrigerator": 1,
        "freezer": 1,
        "wifi_router": 1,
        "security_system": 1,
        
        # HVAC
        "heat_pump": 1,
        "boiler": 1,
        "water_heater": 1,
        "electric_heater": 1,  # Extra heating
        
        # Cooking
        "electric_stove": 1,
        "oven": 1,
        "microwave": 1,
        "kettle": 1,
        "coffee_maker": 1,
        "toaster": 1,
        
        # Major appliances
        "washing_machine": 1,
        "dryer": 1,
        "dishwasher": 1,
        
        # Entertainment
        "tv_living_room": 1,
        "tv_bedroom": 2,  # Multiple TVs
        "gaming_console": 1,
        "stereo_system": 1,
        "desktop_computer": 1,
        "laptop": 2,
        "printer": 1,
        
        # Lighting
        "light_living_room": 1,
        "light_kitchen": 1,
        "light_bedroom": 3,  # Multiple bedrooms
        "light_bathroom": 2,  # Multiple bathrooms
        "light_outdoor": 1,
        
        # Cleaning
        "vacuum_cleaner": 1,
        "iron": 1,
        
        # Personal care
        "hair_dryer": 2,  # Multiple users
        "electric_shaver": 1,
    },
    
    weekday_schedule=[
        ActivityPeriod(0, 6, 0.98),   # Sleeping
        ActivityPeriod(6, 9, 0.99),   # Very busy morning
        ActivityPeriod(9, 15, 0.4),   # Reduced activity
        ActivityPeriod(15, 17, 0.9),  # Kids home
        ActivityPeriod(17, 22, 0.99), # Very active evening
        ActivityPeriod(22, 24, 0.95), # Evening
    ],
    
    weekend_schedule=[
        ActivityPeriod(0, 8, 0.95),   # Sleeping
        ActivityPeriod(8, 22, 0.85),  # High activity all day
        ActivityPeriod(22, 24, 0.95), # Evening
    ],
    
    device_multipliers={
        "washing_machine": 2.5,
        "dryer": 2.5,
        "dishwasher": 2.0,
        "microwave": 1.8,
        "electric_stove": 1.6,
        "oven": 1.5,
        "gaming_console": 2.0,
        "water_heater": 1.8,
        "boiler": 1.5,
    },
    
    has_ev=False,
)


RETIRED_COUPLE = HouseholdProfile(
    name="Retired Couple",
    household_type=HouseholdType.RETIRED_COUPLE,
    num_occupants=2,
    description="Elderly couple, home most of the day, moderate usage",
    
    devices={
        # Always on
        "refrigerator": 1,
        "freezer": 1,
        "wifi_router": 1,
        "security_system": 1,
        
        # HVAC
        "heat_pump": 1,
        "boiler": 1,
        "water_heater": 1,
        
        # Cooking
        "electric_stove": 1,
        "oven": 1,
        "microwave": 1,
        "kettle": 1,
        "coffee_maker": 1,
        "toaster": 1,
        
        # Major appliances
        "washing_machine": 1,
        "dryer": 1,
        "dishwasher": 1,
        
        # Entertainment
        "tv_living_room": 1,
        "tv_bedroom": 1,
        "stereo_system": 1,
        "laptop": 1,
        
        # Lighting
        "light_living_room": 1,
        "light_kitchen": 1,
        "light_bedroom": 1,
        "light_bathroom": 1,
        "light_outdoor": 1,
        
        # Cleaning
        "vacuum_cleaner": 1,
        "iron": 1,
        
        # Personal care
        "hair_dryer": 1,
        "electric_shaver": 1,
    },
    
    weekday_schedule=[
        ActivityPeriod(0, 7, 0.95),   # Sleeping
        ActivityPeriod(7, 10, 0.95),  # Morning routine
        ActivityPeriod(10, 17, 0.85), # Home during day
        ActivityPeriod(17, 22, 0.95), # Evening
        ActivityPeriod(22, 24, 0.9),  # Bedtime
    ],
    
    weekend_schedule=[
        ActivityPeriod(0, 7, 0.95),   # Sleeping
        ActivityPeriod(7, 22, 0.9),   # Consistently home
        ActivityPeriod(22, 24, 0.9),  # Evening
    ],
    
    device_multipliers={
        "tv_living_room": 1.6,  # More TV watching
        "stereo_system": 1.4,
        "boiler": 1.3,  # More heating
        "heat_pump": 1.2,
        "gaming_console": 0.0,  # No gaming
        "laptop": 0.6,  # Less computer use
    },
    
    has_ev=False,  # No EV
)


# Dictionary of all household profiles
HOUSEHOLD_PROFILES = {
    HouseholdType.SINGLE_PROFESSIONAL: SINGLE_PROFESSIONAL,
    HouseholdType.YOUNG_COUPLE: YOUNG_COUPLE,
    HouseholdType.FAMILY_WITH_CHILDREN: FAMILY_WITH_CHILDREN,
    HouseholdType.LARGE_FAMILY: LARGE_FAMILY,
    HouseholdType.RETIRED_COUPLE: RETIRED_COUPLE,
}


def get_household_profile(household_type: HouseholdType) -> HouseholdProfile:
    """Get a household profile by type"""
    return HOUSEHOLD_PROFILES[household_type]
