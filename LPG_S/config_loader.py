"""
Configuration Loader and Validator

Loads device and household definitions from YAML files with comprehensive validation.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class DeviceConfig:
    """Device configuration"""
    id: str
    name: str
    category: str
    pattern: str
    standby_power_w: float
    active_power_w: float
    peak_power_w: float
    avg_duration_min: float
    time_preferences: List[List[int]]
    seasonal_factor: Dict[str, float]
    weekday_factor: float
    requires_occupancy: bool
    expected_uses_per_day: Optional[float] = None
    expected_uses_per_week: Optional[float] = None
    expected_uses_per_month: Optional[float] = None
    cycle_on_min: Optional[float] = None
    cycle_off_min: Optional[float] = None
    
    # New metadata for Professor's layers
    shiftable: bool = False
    max_delay_min: int = 0
    comfort_class: str = "none" # low, medium, high
    service_type: str = "electrical" # space_heating, water_heating, cooking, electrical


@dataclass
class HouseholdConfig:
    """Household configuration"""
    id: str
    name: str
    occupants: int
    description: str
    has_ev: bool
    devices: Dict[str, int]
    weekday_schedule: List[Dict[str, Any]]
    weekend_schedule: List[Dict[str, Any]]
    device_multipliers: Dict[str, float]


class ConfigValidator:
    """Validates configuration files"""
    
    REQUIRED_SEASONS = {"winter", "spring", "summer", "fall"}
    VALID_PATTERNS = {"continuous", "cycling", "daily", "weekly", "seasonal", "occasional"}
    VALID_CATEGORIES = {
        "always_on", "hvac", "major_appliance", "cooking", 
        "cleaning", "entertainment", "lighting", "personal_care", 
        "office", "ev_charging"
    }
    
    @staticmethod
    def validate_device(device_id: str, device_data: Dict[str, Any]) -> List[str]:
        """Validate a single device definition. Returns list of errors."""
        errors = []
        
        # Determine pattern first
        pattern = device_data.get('pattern')
        if pattern not in ConfigValidator.VALID_PATTERNS:
            # Will be caught later, but needed for conditional validation
            pass
            
        # Common required fields
        required_fields = [
            'id', 'name', 'category', 'pattern',
            'standby_power_w', 'active_power_w', 'peak_power_w',
            'time_preferences', 'seasonal_factor', 'weekday_factor', 
            'requires_occupancy'
        ]
        
        # Conditional required fields
        if pattern not in ['cycling', 'continuous']:
             required_fields.append('avg_duration_min')
        
        for field in required_fields:
            if field not in device_data:
                errors.append(f"Device '{device_id}': missing required field '{field}'")
        
        if errors:
            return errors  # Don't validate further if missing required fields
        
        # Validate ID matches
        if device_data['id'] != device_id:
            errors.append(f"Device '{device_id}': id field '{device_data['id']}' doesn't match key")
        
        # Validate category
        if device_data['category'] not in ConfigValidator.VALID_CATEGORIES:
            errors.append(f"Device '{device_id}': invalid category '{device_data['category']}'")
        
        # Validate pattern
        if device_data['pattern'] not in ConfigValidator.VALID_PATTERNS:
            errors.append(f"Device '{device_id}': invalid pattern '{device_data['pattern']}'")
        
        # Validate power values
        if device_data['standby_power_w'] < 0:
            errors.append(f"Device '{device_id}': standby_power_w must be >= 0")
        if device_data['active_power_w'] <= 0:
            errors.append(f"Device '{device_id}': active_power_w must be > 0")
        if device_data['peak_power_w'] < device_data['active_power_w']:
            errors.append(f"Device '{device_id}': peak_power_w must be >= active_power_w")
        
        # Validate duration if present
        if 'avg_duration_min' in device_data and device_data['avg_duration_min'] <= 0:
            errors.append(f"Device '{device_id}': avg_duration_min must be > 0")
        
        # Validate time preferences
        for idx, window in enumerate(device_data['time_preferences']):
            if not isinstance(window, list) or len(window) != 2:
                errors.append(f"Device '{device_id}': time_preferences[{idx}] must be [start, end]")
            else:
                start, end = window
                if not (0 <= start <= 24 and 0 <= end <= 24):
                    errors.append(f"Device '{device_id}': time window [{start}, {end}] out of range [0, 24]")
        
        # Validate seasonal factors
        seasonal = device_data['seasonal_factor']
        if not isinstance(seasonal, dict):
            errors.append(f"Device '{device_id}': seasonal_factor must be a dict")
        else:
            missing_seasons = ConfigValidator.REQUIRED_SEASONS - set(seasonal.keys())
            if missing_seasons:
                errors.append(f"Device '{device_id}': missing seasons in seasonal_factor: {missing_seasons}")
            
            for season, factor in seasonal.items():
                if not isinstance(factor, (int, float)) or factor < 0:
                    errors.append(f"Device '{device_id}': seasonal_factor[{season}] must be >= 0")
        
        # Validate weekday factor
        if device_data['weekday_factor'] < 0:
            errors.append(f"Device '{device_id}': weekday_factor must be >= 0")
        
        # Validate requires_occupancy
        if not isinstance(device_data['requires_occupancy'], bool):
            errors.append(f"Device '{device_id}': requires_occupancy must be boolean")
        
        # Validate expected uses (not needed for cycling or continuous)
        if pattern not in ['cycling', 'continuous']:
            has_uses = any([
                'expected_uses_per_day' in device_data,
                'expected_uses_per_week' in device_data,
                'expected_uses_per_month' in device_data
            ])
            if not has_uses:
                errors.append(f"Device '{device_id}': must have expected_uses_per_day/week/month")
            
            # IMPOSSIBLE RUNTIME CHECK (Phase 2F)
            if 'expected_uses_per_day' in device_data and 'avg_duration_min' in device_data:
                daily_uses = device_data['expected_uses_per_day']
                duration = device_data['avg_duration_min']
                total_minutes = daily_uses * duration
                
                # Hard limit: 1440 minutes/day
                if total_minutes > 1440:
                    errors.append(f"Device '{device_id}': impossible runtime {total_minutes:.1f} min/day (>1440)")
                
                # Soft high-power limit: warning/error for high usage
                if device_data['active_power_w'] > 1000 and total_minutes > 600:
                    errors.append(f"Device '{device_id}': excessive high-power runtime {total_minutes:.1f} min/day (>600)")
                
                # Absurd energy check (>30 kWh/day for non-HVAC/EV) - Optional heuristic
                daily_kwh = (device_data['active_power_w'] * total_minutes / 60) / 1000
                if daily_kwh > 30 and device_data['category'] not in ['hvac', 'ev_charging']:
                    errors.append(f"Device '{device_id}': absurd energy {daily_kwh:.1f} kWh/day for non-HVAC/EV")

        # Validate cycling parameters
        if pattern == 'cycling':
            if 'cycle_on_min' not in device_data or 'cycle_off_min' not in device_data:
                errors.append(f"Device '{device_id}': cycling pattern requires both cycle_on_min and cycle_off_min")
            else:
                if device_data['cycle_on_min'] <= 0:
                    errors.append(f"Device '{device_id}': cycle_on_min must be > 0")
                if device_data['cycle_off_min'] <= 0:
                    errors.append(f"Device '{device_id}': cycle_off_min must be > 0")
                    
        return errors
    
    @staticmethod
    def validate_household(household_id: str, household_data: Dict[str, Any], 
                          valid_device_ids: set) -> List[str]:
        """Validate a single household definition. Returns list of errors."""
        errors = []
        
        # Required fields
        required_fields = [
            'id', 'name', 'occupants', 'description', 'has_ev',
            'devices', 'weekday_schedule', 'weekend_schedule', 'device_multipliers'
        ]
        
        for field in required_fields:
            if field not in household_data:
                errors.append(f"Household '{household_id}': missing required field '{field}'")
        
        if errors:
            return errors
        
        # Validate ID matches
        if household_data['id'] != household_id:
            errors.append(f"Household '{household_id}': id field '{household_data['id']}' doesn't match key")
        
        # Validate occupants
        if not isinstance(household_data['occupants'], int) or household_data['occupants'] < 1:
            errors.append(f"Household '{household_id}': occupants must be integer >= 1")
        
        # Validate has_ev
        if not isinstance(household_data['has_ev'], bool):
            errors.append(f"Household '{household_id}': has_ev must be boolean")
        
        # Validate devices
        devices = household_data['devices']
        if not isinstance(devices, dict):
            errors.append(f"Household '{household_id}': devices must be a dict")
        else:
            for device_id, quantity in devices.items():
                if device_id not in valid_device_ids:
                    errors.append(f"Household '{household_id}': unknown device '{device_id}'")
                if not isinstance(quantity, int) or quantity < 0:
                    errors.append(f"Household '{household_id}': device quantity for '{device_id}' must be int >= 0")
        
        # Validate schedules
        for schedule_name in ['weekday_schedule', 'weekend_schedule']:
            schedule = household_data[schedule_name]
            if not isinstance(schedule, list):
                errors.append(f"Household '{household_id}': {schedule_name} must be a list")
                continue
            
            for idx, slot in enumerate(schedule):
                if not all(k in slot for k in ['start_hour', 'end_hour', 'occupancy_prob']):
                    errors.append(f"Household '{household_id}': {schedule_name}[{idx}] missing required keys")
                else:
                    if not (0 <= slot['start_hour'] <= 24):
                        errors.append(f"Household '{household_id}': {schedule_name}[{idx}] start_hour out of range")
                    if not (0 <= slot['end_hour'] <= 24):
                        errors.append(f"Household '{household_id}': {schedule_name}[{idx}] end_hour out of range")
                    if not (0 <= slot['occupancy_prob'] <= 1):
                        errors.append(f"Household '{household_id}': {schedule_name}[{idx}] occupancy_prob must be [0, 1]")
        
        # Validate device multipliers
        multipliers = household_data['device_multipliers']
        if not isinstance(multipliers, dict):
            errors.append(f"Household '{household_id}': device_multipliers must be a dict")
        else:
            for device_id, multiplier in multipliers.items():
                if device_id not in valid_device_ids:
                    errors.append(f"Household '{household_id}': unknown device in multipliers '{device_id}'")
                if not isinstance(multiplier, (int, float)) or multiplier < 0:
                    errors.append(f"Household '{household_id}': multiplier for '{device_id}' must be >= 0")
        
        # Conflict Checks (Phase 2)
        # 1. Primary Heating: Max 1 of [heat_pump, boiler, electric_heater, gas_boiler]
        heating_set = {'heat_pump', 'boiler', 'electric_heater', 'gas_boiler'}
        present_heating = [d for d in household_data['devices'] if d in heating_set]
        if len(present_heating) > 1:
            errors.append(f"Household '{household_id}': multiple primary heating systems found {present_heating}. Must have at most ONE.")
            
        # 2. Water Heating: Max 1 of [water_heater, boiler_dhw]
        water_set = {'water_heater', 'heat_pump_water_heater'}
        present_water = [d for d in household_data['devices'] if d in water_set]
        if len(present_water) > 1:
            errors.append(f"Household '{household_id}': multiple water heating systems found {present_water}. Must have at most ONE.")

        return errors


class ConfigLoader:
    """Loads and validates YAML configuration files"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.devices: Dict[str, DeviceConfig] = {}
        self.households: Dict[str, HouseholdConfig] = {}
    
    def load_devices(self) -> Dict[str, DeviceConfig]:
        """Load and validate device configuration"""
        devices_file = self.config_dir / "devices.yaml"
        
        if not devices_file.exists():
            raise FileNotFoundError(f"Device config not found: {devices_file}")
        
        with open(devices_file, 'r') as f:
            data = yaml.safe_load(f)
        
        if 'devices' not in data:
            raise ValueError("devices.yaml must contain 'devices' key")
        
        # Validate all devices
        all_errors = []
        for device_id, device_data in data['devices'].items():
            errors = ConfigValidator.validate_device(device_id, device_data)
            all_errors.extend(errors)
        
        if all_errors:
            error_msg = "Device configuration validation failed:\n" + "\n".join(f"  - {e}" for e in all_errors)
            raise ValueError(error_msg)
        
        # Load into DeviceConfig objects
        for device_id, device_data in data['devices'].items():
            self.devices[device_id] = DeviceConfig(
                id=device_data['id'],
                name=device_data['name'],
                category=device_data['category'],
                pattern=device_data['pattern'],
                standby_power_w=device_data['standby_power_w'],
                active_power_w=device_data['active_power_w'],
                peak_power_w=device_data['peak_power_w'],
                avg_duration_min=device_data.get('avg_duration_min', 0),
                time_preferences=device_data['time_preferences'],
                seasonal_factor=device_data['seasonal_factor'],
                weekday_factor=device_data['weekday_factor'],
                requires_occupancy=device_data['requires_occupancy'],
                expected_uses_per_day=device_data.get('expected_uses_per_day'),
                expected_uses_per_week=device_data.get('expected_uses_per_week'),
                expected_uses_per_month=device_data.get('expected_uses_per_month'),
                cycle_on_min=device_data.get('cycle_on_min'),
                cycle_off_min=device_data.get('cycle_off_min'),
                shiftable=device_data.get('shiftable', False),
                max_delay_min=device_data.get('max_delay_min', 0),
                comfort_class=device_data.get('comfort_class', 'none'),
                service_type=device_data.get('service_type', 'electrical')
            )
        
        print(f"✓ Loaded {len(self.devices)} devices from {devices_file}")
        return self.devices
    
    def load_households(self) -> Dict[str, HouseholdConfig]:
        """Load and validate household configuration"""
        households_file = self.config_dir / "households.yaml"
        
        if not households_file.exists():
            raise FileNotFoundError(f"Household config not found: {households_file}")
        
        with open(households_file, 'r') as f:
            data = yaml.safe_load(f)
        
        if 'households' not in data:
            raise ValueError("households.yaml must contain 'households' key")
        
        # Validate all households
        valid_device_ids = set(self.devices.keys())
        all_errors = []
        for household_id, household_data in data['households'].items():
            errors = ConfigValidator.validate_household(household_id, household_data, valid_device_ids)
            all_errors.extend(errors)
        
        if all_errors:
            error_msg = "Household configuration validation failed:\n" + "\n".join(f"  - {e}" for e in all_errors)
            raise ValueError(error_msg)
        
        # Load into HouseholdConfig objects
        for household_id, household_data in data['households'].items():
            self.households[household_id] = HouseholdConfig(
                id=household_data['id'],
                name=household_data['name'],
                occupants=household_data['occupants'],
                description=household_data['description'],
                has_ev=household_data['has_ev'],
                devices=household_data['devices'],
                weekday_schedule=household_data['weekday_schedule'],
                weekend_schedule=household_data['weekend_schedule'],
                device_multipliers=household_data['device_multipliers']
            )
        
        print(f"✓ Loaded {len(self.households)} households from {households_file}")
        return self.households
    
    def load_all(self) -> tuple[Dict[str, DeviceConfig], Dict[str, HouseholdConfig]]:
        """Load and validate all configuration files"""
        print(f"\nLoading configuration from {self.config_dir}/")
        self.load_devices()
        self.load_households()
        print(f"✓ Configuration loaded successfully\n")
        return self.devices, self.households


def load_config(config_dir: str = "config") -> tuple[Dict[str, DeviceConfig], Dict[str, HouseholdConfig]]:
    """
    Convenience function to load all configuration
    
    Args:
        config_dir: Path to configuration directory
        
    Returns:
        (devices_dict, households_dict)
        
    Raises:
        FileNotFoundError: If config files don't exist
        ValueError: If validation fails
    """
    loader = ConfigLoader(config_dir)
    return loader.load_all()
