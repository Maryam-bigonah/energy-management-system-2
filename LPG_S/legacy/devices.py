"""
Comprehensive Device Database for Household Load Profile Generation

Power ratings and usage patterns based on:
- US Department of Energy appliance data
- European Commission energy labels
- Manufacturer specifications
- Academic research on residential energy consumption
"""

from dataclasses import dataclass
from typing import List, Tuple
from enum import Enum


class DeviceCategory(Enum):
    ALWAYS_ON = "always_on"
    HVAC = "hvac"
    MAJOR_APPLIANCE = "major_appliance"
    COOKING = "cooking"
    CLEANING = "cleaning"
    ENTERTAINMENT = "entertainment"
    LIGHTING = "lighting"
    PERSONAL_CARE = "personal_care"
    OFFICE = "office"
    EV_CHARGING = "ev_charging"


class UsagePattern(Enum):
    CONTINUOUS = "continuous"
    DAILY = "daily"
    WEEKLY = "weekly"
    SEASONAL = "seasonal"
    OCCASIONAL = "occasional"


@dataclass
class Device:
    """Represents a household electrical device"""
    name: str
    category: DeviceCategory
    standby_power_w: float  # Standby power consumption (W)
    active_power_w: float   # Active power consumption (W)
    peak_power_w: float     # Peak power consumption (W)
    avg_duration_min: float # Average usage duration (minutes)
    pattern: UsagePattern
    time_preferences: List[Tuple[int, int]]  # Preferred usage hours (start, end)
    daily_frequency: float  # Expected uses per day
    seasonal_factor: dict   # Multiplier by season
    weekday_factor: float   # Weekday vs weekend ratio
    
    
# Complete device database
DEVICE_DATABASE = {
    # ============ ALWAYS ON DEVICES ============
    "refrigerator": Device(
        name="Refrigerator",
        category=DeviceCategory.ALWAYS_ON,
        standby_power_w=3,  # Between compressor cycles
        active_power_w=180,
        peak_power_w=220,
        avg_duration_min=25,  # Compressor runs 25 min per cycle
        pattern=UsagePattern.DAILY,
        time_preferences=[(0, 24)],
        daily_frequency=40,  # Cycles ~every 36 minutes (25 min on, 11 min off)
        seasonal_factor={"winter": 0.9, "spring": 1.0, "summer": 1.2, "fall": 1.0},
        weekday_factor=1.0
    ),
    
    "freezer": Device(
        name="Freezer",
        category=DeviceCategory.ALWAYS_ON,
        standby_power_w=2,  # Between compressor cycles
        active_power_w=120,
        peak_power_w=150,
        avg_duration_min=20,  # Compressor runs 20 min per cycle
        pattern=UsagePattern.DAILY,
        time_preferences=[(0, 24)],
        daily_frequency=48,  # Cycles ~every 30 minutes (20 min on, 10 min off)
        seasonal_factor={"winter": 0.85, "spring": 1.0, "summer": 1.15, "fall": 1.0},
        weekday_factor=1.0
    ),
    
    "wifi_router": Device(
        name="WiFi Router",
        category=DeviceCategory.ALWAYS_ON,
        standby_power_w=8,
        active_power_w=12,
        peak_power_w=15,
        avg_duration_min=1440,
        pattern=UsagePattern.CONTINUOUS,
        time_preferences=[(0, 24)],
        daily_frequency=1,
        seasonal_factor={"winter": 1.0, "spring": 1.0, "summer": 1.0, "fall": 1.0},
        weekday_factor=1.0
    ),
    
    "security_system": Device(
        name="Security System",
        category=DeviceCategory.ALWAYS_ON,
        standby_power_w=5,
        active_power_w=8,
        peak_power_w=10,
        avg_duration_min=1440,
        pattern=UsagePattern.CONTINUOUS,
        time_preferences=[(0, 24)],
        daily_frequency=1,
        seasonal_factor={"winter": 1.0, "spring": 1.0, "summer": 1.0, "fall": 1.0},
        weekday_factor=1.0
    ),
    
    # ============ HVAC SYSTEMS ============
    "heat_pump": Device(
        name="Heat Pump",
        category=DeviceCategory.HVAC,
        standby_power_w=20,
        active_power_w=3500,
        peak_power_w=5000,
        avg_duration_min=180,  # Cycles throughout day
        pattern=UsagePattern.DAILY,
        time_preferences=[(0, 24)],
        daily_frequency=8,
        seasonal_factor={"winter": 1.8, "spring": 0.6, "summer": 1.5, "fall": 0.7},
        weekday_factor=1.0
    ),
    
    "central_ac": Device(
        name="Central Air Conditioning",
        category=DeviceCategory.HVAC,
        standby_power_w=0,  # Completely off in winter
        active_power_w=3000,
        peak_power_w=4500,
        avg_duration_min=120,
        pattern=UsagePattern.SEASONAL,
        time_preferences=[(12, 22)],
        daily_frequency=6,
        seasonal_factor={"winter": 0.0, "spring": 0.3, "summer": 2.0, "fall": 0.4},
        weekday_factor=0.8  # More usage on weekends when home
    ),
    
    "electric_heater": Device(
        name="Space Heater",
        category=DeviceCategory.HVAC,
        standby_power_w=0,
        active_power_w=1500,
        peak_power_w=1500,
        avg_duration_min=120,
        pattern=UsagePattern.SEASONAL,
        time_preferences=[(6, 9), (17, 23)],
        daily_frequency=2,
        seasonal_factor={"winter": 2.5, "spring": 0.5, "summer": 0.0, "fall": 0.8},
        weekday_factor=0.7
    ),
    
    "water_heater": Device(
        name="Electric Water Heater",
        category=DeviceCategory.HVAC,
        standby_power_w=50,
        active_power_w=4000,
        peak_power_w=4500,
        avg_duration_min=90,  # Total heating time per day
        pattern=UsagePattern.DAILY,
        time_preferences=[(6, 9), (17, 22)],
        daily_frequency=4,
        seasonal_factor={"winter": 1.3, "spring": 1.0, "summer": 0.8, "fall": 1.1},
        weekday_factor=1.0
    ),
    
    "boiler": Device(
        name="Electric Boiler",
        category=DeviceCategory.HVAC,
        standby_power_w=30,
        active_power_w=9000,
        peak_power_w=12000,
        avg_duration_min=60,
        pattern=UsagePattern.DAILY,
        time_preferences=[(5, 8), (16, 22)],
        daily_frequency=3,
        seasonal_factor={"winter": 2.0, "spring": 0.8, "summer": 0.3, "fall": 1.2},
        weekday_factor=1.0
    ),
    
    # ============ MAJOR APPLIANCES ============
    "washing_machine": Device(
        name="Washing Machine",
        category=DeviceCategory.MAJOR_APPLIANCE,
        standby_power_w=0,  # 0W when not in use
        active_power_w=500,
        peak_power_w=2000,  # Heating water
        avg_duration_min=60,
        pattern=UsagePattern.WEEKLY,
        time_preferences=[(8, 12), (14, 18)],
        daily_frequency=0.5,  # 3-4 times per week
        seasonal_factor={"winter": 1.0, "spring": 1.0, "summer": 1.1, "fall": 1.0},
        weekday_factor=0.6  # More on weekends
    ),
    
    "dryer": Device(
        name="Clothes Dryer",
        category=DeviceCategory.MAJOR_APPLIANCE,
        standby_power_w=0,  # 0W when not in use
        active_power_w=3000,
        peak_power_w=3400,
        avg_duration_min=50,
        pattern=UsagePattern.WEEKLY,
        time_preferences=[(9, 13), (15, 19)],
        daily_frequency=0.4,
        seasonal_factor={"winter": 1.3, "spring": 1.0, "summer": 0.7, "fall": 1.1},
        weekday_factor=0.6
    ),
    
    "dishwasher": Device(
        name="Dishwasher",
        category=DeviceCategory.MAJOR_APPLIANCE,
        standby_power_w=0,  # 0W when not in use
        active_power_w=1200,
        peak_power_w=1800,
        avg_duration_min=90,
        pattern=UsagePattern.DAILY,
        time_preferences=[(20, 23)],
        daily_frequency=0.8,
        seasonal_factor={"winter": 1.0, "spring": 1.0, "summer": 1.0, "fall": 1.0},
        weekday_factor=1.0
    ),
    
    # ============ COOKING APPLIANCES ============
    "electric_stove": Device(
        name="Electric Stove",
        category=DeviceCategory.COOKING,
        standby_power_w=0,
        active_power_w=2400,
        peak_power_w=3000,
        avg_duration_min=25,
        pattern=UsagePattern.DAILY,
        time_preferences=[(7, 9), (12, 13), (17, 20)],
        daily_frequency=2.5,
        seasonal_factor={"winter": 1.1, "spring": 1.0, "summer": 0.9, "fall": 1.0},
        weekday_factor=1.0
    ),
    
    "oven": Device(
        name="Electric Oven",
        category=DeviceCategory.COOKING,
        standby_power_w=0,
        active_power_w=2400,
        peak_power_w=3600,
        avg_duration_min=45,
        pattern=UsagePattern.WEEKLY,
        time_preferences=[(17, 20)],
        daily_frequency=0.4,
        seasonal_factor={"winter": 1.2, "spring": 1.0, "summer": 0.8, "fall": 1.0},
        weekday_factor=0.5  # More on weekends
    ),
    
    "microwave": Device(
        name="Microwave",
        category=DeviceCategory.COOKING,
        standby_power_w=0,  # Modern energy-efficient models
        active_power_w=1200,
        peak_power_w=1400,
        avg_duration_min=5,
        pattern=UsagePattern.DAILY,
        time_preferences=[(7, 9), (12, 14), (17, 21)],
        daily_frequency=3,
        seasonal_factor={"winter": 1.0, "spring": 1.0, "summer": 1.1, "fall": 1.0},
        weekday_factor=1.1
    ),
    
    "kettle": Device(
        name="Electric Kettle",
        category=DeviceCategory.COOKING,
        standby_power_w=0,
        active_power_w=2000,
        peak_power_w=2200,
        avg_duration_min=3,
        pattern=UsagePattern.DAILY,
        time_preferences=[(6, 9), (14, 16), (19, 21)],
        daily_frequency=4,
        seasonal_factor={"winter": 1.3, "spring": 1.0, "summer": 0.8, "fall": 1.1},
        weekday_factor=1.2  # More during work-from-home
    ),
    
    "coffee_maker": Device(
        name="Coffee Maker",
        category=DeviceCategory.COOKING,
        standby_power_w=0,  # Unplugged when not in use
        active_power_w=1000,
        peak_power_w=1200,
        avg_duration_min=8,
        pattern=UsagePattern.DAILY,
        time_preferences=[(6, 9), (13, 15)],
        daily_frequency=2,
        seasonal_factor={"winter": 1.1, "spring": 1.0, "summer": 0.9, "fall": 1.0},
        weekday_factor=1.3
    ),
    
    "toaster": Device(
        name="Toaster",
        category=DeviceCategory.COOKING,
        standby_power_w=0,
        active_power_w=1200,
        peak_power_w=1400,
        avg_duration_min=3,
        pattern=UsagePattern.DAILY,
        time_preferences=[(6, 9)],
        daily_frequency=1.5,
        seasonal_factor={"winter": 1.0, "spring": 1.0, "summer": 1.0, "fall": 1.0},
        weekday_factor=1.2
    ),
    
    # ============ CLEANING DEVICES ============
    "vacuum_cleaner": Device(
        name="Vacuum Cleaner",
        category=DeviceCategory.CLEANING,
        standby_power_w=0,
        active_power_w=1200,
        peak_power_w=1400,
        avg_duration_min=25,
        pattern=UsagePattern.WEEKLY,
        time_preferences=[(9, 12), (14, 17)],
        daily_frequency=0.3,  # 2 times per week
        seasonal_factor={"winter": 1.0, "spring": 1.1, "summer": 1.0, "fall": 1.0},
        weekday_factor=0.4  # Mostly weekends
    ),
    
    "iron": Device(
        name="Iron",
        category=DeviceCategory.CLEANING,
        standby_power_w=0,
        active_power_w=1500,
        peak_power_w=1800,
        avg_duration_min=20,
        pattern=UsagePattern.WEEKLY,
        time_preferences=[(9, 11), (19, 21)],
        daily_frequency=0.3,
        seasonal_factor={"winter": 1.0, "spring": 1.0, "summer": 1.0, "fall": 1.0},
        weekday_factor=0.5
    ),
    
    # ============ ENTERTAINMENT DEVICES ============
    "tv_living_room": Device(
        name="TV (Living Room)",
        category=DeviceCategory.ENTERTAINMENT,
        standby_power_w=2,
        active_power_w=120,
        peak_power_w=150,
        avg_duration_min=210,  # 3.5 hours average
        pattern=UsagePattern.DAILY,
        time_preferences=[(18, 23)],
        daily_frequency=1.2,
        seasonal_factor={"winter": 1.3, "spring": 1.0, "summer": 0.8, "fall": 1.1},
        weekday_factor=0.8
    ),
    
    "tv_bedroom": Device(
        name="TV (Bedroom)",
        category=DeviceCategory.ENTERTAINMENT,
        standby_power_w=2,
        active_power_w=80,
        peak_power_w=100,
        avg_duration_min=90,
        pattern=UsagePattern.DAILY,
        time_preferences=[(21, 24)],
        daily_frequency=0.8,
        seasonal_factor={"winter": 1.2, "spring": 1.0, "summer": 0.9, "fall": 1.0},
        weekday_factor=1.0
    ),
    
    "gaming_console": Device(
        name="Gaming Console (PlayStation/Xbox)",
        category=DeviceCategory.ENTERTAINMENT,
        standby_power_w=10,
        active_power_w=150,
        peak_power_w=180,
        avg_duration_min=120,
        pattern=UsagePattern.DAILY,
        time_preferences=[(15, 18), (19, 23)],
        daily_frequency=0.6,
        seasonal_factor={"winter": 1.3, "spring": 1.0, "summer": 0.9, "fall": 1.1},
        weekday_factor=0.7
    ),
    
    "stereo_system": Device(
        name="Stereo System",
        category=DeviceCategory.ENTERTAINMENT,
        standby_power_w=5,
        active_power_w=80,
        peak_power_w=120,
        avg_duration_min=90,
        pattern=UsagePattern.DAILY,
        time_preferences=[(8, 10), (18, 22)],
        daily_frequency=0.7,
        seasonal_factor={"winter": 1.0, "spring": 1.0, "summer": 1.1, "fall": 1.0},
        weekday_factor=0.8
    ),
    
    # ============ OFFICE/COMPUTING ============
    "desktop_computer": Device(
        name="Desktop Computer",
        category=DeviceCategory.OFFICE,
        standby_power_w=5,
        active_power_w=200,
        peak_power_w=300,
        avg_duration_min=180,
        pattern=UsagePattern.DAILY,
        time_preferences=[(8, 12), (19, 23)],
        daily_frequency=1.5,
        seasonal_factor={"winter": 1.0, "spring": 1.0, "summer": 1.0, "fall": 1.0},
        weekday_factor=1.3  # More on weekdays (work)
    ),
    
    "laptop": Device(
        name="Laptop",
        category=DeviceCategory.OFFICE,
        standby_power_w=2,
        active_power_w=60,
        peak_power_w=90,
        avg_duration_min=240,
        pattern=UsagePattern.DAILY,
        time_preferences=[(8, 17), (19, 22)],
        daily_frequency=1.8,
        seasonal_factor={"winter": 1.0, "spring": 1.0, "summer": 1.0, "fall": 1.0},
        weekday_factor=1.5
    ),
    
    "printer": Device(
        name="Printer",
        category=DeviceCategory.OFFICE,
        standby_power_w=5,
        active_power_w=300,
        peak_power_w=400,
        avg_duration_min=5,
        pattern=UsagePattern.WEEKLY,
        time_preferences=[(9, 17)],
        daily_frequency=0.3,
        seasonal_factor={"winter": 1.0, "spring": 1.0, "summer": 1.0, "fall": 1.0},
        weekday_factor=1.8
    ),
    
    # ============ LIGHTING ============
    "light_living_room": Device(
        name="Living Room Lights",
        category=DeviceCategory.LIGHTING,
        standby_power_w=0,
        active_power_w=60,  # LED bulbs
        peak_power_w=60,
        avg_duration_min=240,
        pattern=UsagePattern.DAILY,
        time_preferences=[(6, 9), (17, 23)],
        daily_frequency=2,
        seasonal_factor={"winter": 1.4, "spring": 1.0, "summer": 0.7, "fall": 1.2},
        weekday_factor=0.9
    ),
    
    "light_kitchen": Device(
        name="Kitchen Lights",
        category=DeviceCategory.LIGHTING,
        standby_power_w=0,
        active_power_w=40,
        peak_power_w=40,
        avg_duration_min=180,
        pattern=UsagePattern.DAILY,
        time_preferences=[(6, 9), (17, 22)],
        daily_frequency=2.5,
        seasonal_factor={"winter": 1.4, "spring": 1.0, "summer": 0.7, "fall": 1.2},
        weekday_factor=1.0
    ),
    
    "light_bedroom": Device(
        name="Bedroom Lights",
        category=DeviceCategory.LIGHTING,
        standby_power_w=0,
        active_power_w=30,
        peak_power_w=30,
        avg_duration_min=90,
        pattern=UsagePattern.DAILY,
        time_preferences=[(6, 8), (21, 24)],
        daily_frequency=2,
        seasonal_factor={"winter": 1.4, "spring": 1.0, "summer": 0.7, "fall": 1.2},
        weekday_factor=1.0
    ),
    
    "light_bathroom": Device(
        name="Bathroom Lights",
        category=DeviceCategory.LIGHTING,
        standby_power_w=0,
        active_power_w=25,
        peak_power_w=25,
        avg_duration_min=60,
        pattern=UsagePattern.DAILY,
        time_preferences=[(6, 9), (21, 23)],
        daily_frequency=4,
        seasonal_factor={"winter": 1.3, "spring": 1.0, "summer": 0.8, "fall": 1.1},
        weekday_factor=1.0
    ),
    
    "light_outdoor": Device(
        name="Outdoor Lights",
        category=DeviceCategory.LIGHTING,
        standby_power_w=0,
        active_power_w=35,
        peak_power_w=35,
        avg_duration_min=480,  # Dusk to dawn
        pattern=UsagePattern.DAILY,
        time_preferences=[(17, 7)],  # Evening to morning
        daily_frequency=1,
        seasonal_factor={"winter": 1.6, "spring": 1.0, "summer": 0.6, "fall": 1.2},
        weekday_factor=1.0
    ),
    
    # ============ PERSONAL CARE ============
    "hair_dryer": Device(
        name="Hair Dryer",
        category=DeviceCategory.PERSONAL_CARE,
        standby_power_w=0,
        active_power_w=1800,
        peak_power_w=2000,
        avg_duration_min=8,
        pattern=UsagePattern.DAILY,
        time_preferences=[(6, 9), (20, 22)],
        daily_frequency=1.2,
        seasonal_factor={"winter": 1.1, "spring": 1.0, "summer": 0.9, "fall": 1.0},
        weekday_factor=1.3
    ),
    
    "electric_shaver": Device(
        name="Electric Shaver",
        category=DeviceCategory.PERSONAL_CARE,
        standby_power_w=1,
        active_power_w=15,
        peak_power_w=20,
        avg_duration_min=5,
        pattern=UsagePattern.DAILY,
        time_preferences=[(6, 9)],
        daily_frequency=0.8,
        seasonal_factor={"winter": 1.0, "spring": 1.0, "summer": 1.0, "fall": 1.0},
        weekday_factor=1.4
    ),
    
    # ============ EV CHARGING ============
    "ev_charger_level2": Device(
        name="EV Charger (Level 2, 7.2kW)",
        category=DeviceCategory.EV_CHARGING,
        standby_power_w=5,
        active_power_w=7200,
        peak_power_w=7400,
        avg_duration_min=180,  # 3 hours average charge
        pattern=UsagePattern.DAILY,
        time_preferences=[(22, 6)],  # Overnight charging
        daily_frequency=0.7,  # Not every day
        seasonal_factor={"winter": 1.2, "spring": 1.0, "summer": 0.9, "fall": 1.0},
        weekday_factor=1.3  # More charging on work days
    ),
}


def get_device(device_id: str) -> Device:
    """Retrieve a device from the database"""
    if device_id not in DEVICE_DATABASE:
        raise ValueError(f"Device '{device_id}' not found in database")
    return DEVICE_DATABASE[device_id]


def get_devices_by_category(category: DeviceCategory) -> dict:
    """Get all devices in a specific category"""
    return {k: v for k, v in DEVICE_DATABASE.items() if v.category == category}
