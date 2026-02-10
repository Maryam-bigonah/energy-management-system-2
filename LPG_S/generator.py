"""
Load Profile Generator Engine - Physics-Correct Energy Modeling

Generates synthetic household electricity consumption profiles with:
- Partial-interval accounting for resolution-invariant energy
- Proper startup peak modeling (energy bumps, not full-interval)
- Cycling duty fractions for realistic fridge/freezer behavior
- Average power output (W) suitable for energy calculation
"""

import numpy as np
import pandas as pd
import yaml
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from tqdm import tqdm

from config_loader import DeviceConfig, HouseholdConfig, load_config
from behavior import (
    get_season, is_weekend, is_holiday,
    get_occupancy_probability, should_device_turn_on,
    get_cycling_duty_fraction, get_device_duration, add_noise
)
from scheduler import QuotaScheduler, ScheduledEvent


class DeviceState:
    """Tracks the state of a device over time with partial-interval support"""
    
    def __init__(self, device_id: str, device_config: DeviceConfig):
        self.device_id = device_id
        self.device_config = device_config
        self.is_on = False
        self.turn_on_time = None
        self.remaining_minutes = 0.0  # Changed from time_active
        self.has_had_startup_peak = False
        
        # For cycling devices (fridge/freezer)
        self.is_cycling = device_config.pattern == 'cycling'
        self.phase_offset = None  # Will be set externally with seeded RNG if cycling
        
        # New: Tracking current event for scheduled tasks
        self.current_event_id = None
    
    def turn_on(self, duration_minutes: float, event_id: str = None):
        """Turn device on with specified duration"""
        self.is_on = True
        self.remaining_minutes = duration_minutes
        self.has_had_startup_peak = False
        self.current_event_id = event_id
    
    def turn_off(self):
        """Turn device off"""
        self.is_on = False
        self.remaining_minutes = 0.0
        self.has_had_startup_peak = False
        self.current_event_id = None
    
    def get_active_fraction(self, interval_minutes: float) -> float:
        """
        Get fraction of interval this device is active
        
        Returns:
            Fraction in [0, 1] representing portion of interval device is ON
        """
        if not self.is_on:
            return 0.0
        
        # Active for min(remaining, interval)
        active_minutes = min(self.remaining_minutes, interval_minutes)
        return active_minutes / interval_minutes
    
    def advance(self, interval_minutes: float) -> bool:
        """
        Advance device state by interval
        
        Returns:
            True if device should turn off after this step
        """
        if not self.is_on:
            return False
        
        self.remaining_minutes -= interval_minutes
        
        # Device turns off if no time remaining
        if self.remaining_minutes <= 0:
            return True
        
        return False


def compute_average_power_with_peak(
    device_config: DeviceConfig,
    active_fraction: float,
    interval_minutes: float,
    is_first_interval: bool,
    random_state: np.random.RandomState
) -> float:
    """
    Compute average power for interval including startup peak if applicable
    
    Peak is modeled as energy bump over short duration, then converted to
    average power over the interval.
    
    Args:
        device_config: Device configuration
        active_fraction: Fraction of interval device is active [0, 1]
        interval_minutes: Interval duration in minutes
        is_first_interval: Whether this is the first interval after turn-on
        random_state: Random state for variance
        
    Returns:
        Average power in watts for this interval
    """
    if active_fraction == 0:
        return device_config.standby_power_w
    
    # Get peak parameters (with defaults)
    enable_peak = getattr(device_config, 'enable_startup_peak', True)
    peak_duration_sec = getattr(device_config, 'peak_duration_seconds', 15.0)
    
    # Active time in seconds
    active_seconds = active_fraction * interval_minutes * 60
    
    # Compute power during active portion
    if enable_peak and is_first_interval and active_seconds > peak_duration_sec:
        # Peak for short duration, then normal active power
        peak_energy_wh = device_config.peak_power_w * (peak_duration_sec / 3600)
        normal_energy_wh = device_config.active_power_w * ((active_seconds - peak_duration_sec) / 3600)
        total_active_energy_wh = peak_energy_wh + normal_energy_wh
        avg_active_power = total_active_energy_wh / (active_seconds / 3600)
    else:
        # No peak, just active power with variance
        variance = device_config.active_power_w * 0.1
        avg_active_power = random_state.normal(device_config.active_power_w, variance)
        avg_active_power = np.clip(avg_active_power, 
                                   device_config.active_power_w * 0.8,
                                   device_config.peak_power_w)
    
    # Blend active and standby based on fraction
    avg_power = (active_fraction * avg_active_power + 
                 (1 - active_fraction) * device_config.standby_power_w)
    
    return avg_power


class LoadProfileGenerator:
    """Generate synthetic household load profiles from YAML configs"""
    
    def __init__(
        self,
        household_config: HouseholdConfig,
        devices_config: Dict[str, DeviceConfig],
        start_date: datetime,
        end_date: datetime,
        interval_minutes: int = 10,
        random_seed: Optional[int] = None,
        config_dir: str = "config"
    ):
        """
        Initialize generator
        
        Args:
            household_config: HouseholdConfig object
            devices_config: Dict of DeviceConfig objects
            start_date: Start date for simulation
            end_date: End date for simulation
            interval_minutes: Time resolution in minutes
            random_seed: Random seed for reproducibility
            config_dir: Path to config directory (needed for behavior profiles)
        """
        self.household = household_config
        self.devices_config = devices_config
        self.start_date = start_date
        self.end_date = end_date
        self.interval_minutes = interval_minutes
        self.config_dir = config_dir
        
        # Random state for reproducibility
        self.random_state = np.random.RandomState(random_seed)
        seed_val = random_seed if random_seed is not None else 42
        
        # Initialize device instances
        self.device_instances = []
        for device_id, quantity in household_config.devices.items():
            for i in range(quantity):
                device_config = devices_config[device_id]
                device_state = DeviceState(device_id, device_config)
                
                # CRITICAL: Set phase offset using SEEDED random state
                if device_state.is_cycling:
                    device_state.phase_offset = self.random_state.random()
                
                self.device_instances.append(device_state)
        
        # Storage for time series data
        self.timestamps = []
        self.device_powers = {inst.device_id: [] for inst in self.device_instances}
        self.total_powers = []
        
        self.activations = []  # List of {device_id, start_time, duration, energy, ...}
        
        # LOAD BEHAVIOR PROFILES & SCHEDULE TASKS
        self.scheduled_events: List[ScheduledEvent] = []
        self._load_and_schedule_tasks(seed_val)

    def _load_and_schedule_tasks(self, seed: int):
        """Load behavior profiles and pre-calculate task schedules"""
        behavior_file = os.path.join(self.config_dir, "behavior_profiles.yaml")
        if not os.path.exists(behavior_file):
            print(f"Warning: {behavior_file} not found. Using legacy probabilistic logic only.")
            return

        with open(behavior_file, 'r') as f:
            data = yaml.safe_load(f)
            
        profiles = data.get('behavior_profiles', {})
        hh_profile = profiles.get(self.household.id)
        
        if hh_profile:
            print(f"Loaded behavior profile for {self.household.name}")
            scheduler = QuotaScheduler(hh_profile, self.devices_config, self.household, self.start_date.year, seed)
            generated_events = scheduler.generate_schedule()
            
            # Filter events within simulation period
            self.scheduled_events = [
                e for e in generated_events 
                if self.start_date <= e.start_time < self.end_date
            ]
            print(f"Scheduled {len(self.scheduled_events)} tasks for this period.")
        else:
            print(f"No specific behavior profile found for {self.household.id}")

    def generate(self) -> (pd.DataFrame, List[Dict]):
        """
        Generate complete load profile
        
        Returns:
            Tuple of (DataFrame, ActivationsList)
        """
        print(f"\nGenerating load profile for {self.household.name}")
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Resolution: {self.interval_minutes} minutes")
        print(f"Devices: {len(self.device_instances)}")
        
        # Generate time series
        current_time = self.start_date
        interval_delta = timedelta(minutes=self.interval_minutes)
        min_steps = int((self.end_date - self.start_date).total_seconds() / 60 / self.interval_minutes)
        
        # Track simulation time offset
        time_offset_minutes = 0.0
        
        # Sort scheduled events by start time for efficient monotonic consumption
        sorted_events = sorted(self.scheduled_events, key=lambda x: x.start_time)
        event_idx = 0
        
        # Simulate each time step
        for step in tqdm(range(min_steps), desc="Simulating"):
            # Check for new events starting in this interval [current_time, next_time)
            next_time = current_time + interval_delta
            
            active_events_this_step = []
            while event_idx < len(sorted_events):
                evt = sorted_events[event_idx]
                if evt.start_time >= next_time:
                    break # Future event
                if evt.start_time >= current_time:
                    # Event starts effectively within this interval
                    active_events_this_step.append(evt)
                event_idx += 1
            
            self._simulate_timestep(current_time, time_offset_minutes, active_events_this_step)
            current_time = next_time
            time_offset_minutes += self.interval_minutes
        
        # Build DataFrame
        df = self._build_dataframe()
        return df, self.activations
    
    def _simulate_timestep(self, current_time: datetime, time_offset_minutes: float, new_events: List[ScheduledEvent]):
        """Simulate one time step with partial-interval accounting"""
        hour = current_time.hour
        is_weekday_val = not is_weekend(current_time) and not is_holiday(current_time)
        
        # Get occupancy probability
        occupancy_prob = get_occupancy_probability(
            hour,
            is_weekday_val,
            self.household.weekday_schedule,
            self.household.weekend_schedule
        )
        
        # Track power for this timestep
        timestep_power = {}
        total_power = 0
        
        # Consumed new events (to prevent multi-instance taking same event if not appropriate)
        events_to_process = new_events.copy()
        
        # Simulate each device instance
        for device_inst in self.device_instances:
            device_config = device_inst.device_config
            device_id = device_inst.device_id
            
            # Handle cycling devices (deterministic duty cycle with fraction)
            if device_inst.is_cycling:
                duty_fraction = get_cycling_duty_fraction(
                    time_offset_minutes,
                    self.interval_minutes,
                    device_config.cycle_on_min,
                    device_config.cycle_off_min,
                    device_inst.phase_offset
                )
                
                # Weighted average of active and standby
                variance = device_config.active_power_w * 0.05
                active_power = self.random_state.normal(device_config.active_power_w, variance)
                active_power = np.clip(active_power, 
                                      device_config.active_power_w * 0.9,
                                      device_config.peak_power_w)
                
                power = (duty_fraction * active_power + 
                        (1 - duty_fraction) * device_config.standby_power_w)
                
                # Apply seasonal factor
                season = get_season(current_time)
                seasonal_factor = device_config.seasonal_factor.get(season, 1.0)
                power = power * seasonal_factor
            
            # Handle continuous devices (always on)
            elif device_config.pattern == 'continuous':
                power = device_config.active_power_w
                # Small variance
                power += self.random_state.normal(0, power * 0.05)
            
            # Handle regular/task devices
            else:
                # Check for schedule trigger
                matched_event = None
                if not device_inst.is_on:
                    for i, evt in enumerate(events_to_process):
                        if evt.device_id == device_id:
                            matched_event = evt
                            events_to_process.pop(i) # Consume event
                            break
                            
                # Determine if first interval (for peak detection)
                is_first_interval = device_inst.is_on and device_inst.remaining_minutes > (device_inst.remaining_minutes - self.interval_minutes + 1e-6)
                
                should_legacy_trigger = False
                
                # Start device if event triggered
                if matched_event:
                    duration = matched_event.duration_min
                    device_inst.turn_on(duration, event_id=f"{device_id}_{len(self.activations)}")
                    is_first_interval = True
                    
                    # Log activation
                    self.activations.append({
                        'task_id': device_inst.current_event_id,
                        'device_id': device_id,
                        'start_time': current_time, 
                        'duration_min': duration,
                        'energy_kwh': matched_event.energy_kwh,
                        'max_delay_min': getattr(device_config, 'max_delay_min', 0),
                        'comfort_class': getattr(device_config, 'comfort_class', 'none')
                    })
                
                # Determine if this device is a task device covered by behavior profiles
                # Task devices include: washing_machine, dryer, dishwasher, vacuum_cleaner, iron, 
                # electric_stove, oven, ev_charger_level2 (anything in behavior_profiles.yaml)
                is_task_device = device_config.pattern in ['daily', 'weekly']
                has_scheduler = len(self.scheduled_events) > 0  # Behavior profile was loaded
                
                # Check legacy probabilistic logic ONLY for non-task devices
                # Examples: entertainment devices (TV, gaming), lighting, office equipment
                # These don't have quotas/schedules and use probabilistic activation
                should_legacy_trigger = False
                
                if not device_inst.is_on and device_config.pattern not in ['standby', 'cycling', 'continuous']:
                    # Only use legacy logic if:
                    # 1. Device is NOT a scheduled task device (no behavior profile), OR
                    # 2. No scheduler exists at all for this household
                    if not has_scheduler or not is_task_device:
                        multiplier = self.household.device_multipliers.get(device_id, 1.0)
                        should_legacy_trigger = should_device_turn_on(
                            device_config,
                            current_time,
                            occupancy_prob,
                            multiplier,
                            self.interval_minutes,
                            self.random_state
                        )
                
                if should_legacy_trigger:
                    duration = get_device_duration(device_config, self.random_state)
                    device_inst.turn_on(duration, event_id=f"{device_id}_legacy_{len(self.activations)}")
                    is_first_interval = True
                    
                    if getattr(device_config, 'shiftable', False):
                        avg_power_kw = device_config.active_power_w / 1000.0
                        energy_kwh = avg_power_kw * (duration / 60.0)
                        self.activations.append({
                            'task_id': device_inst.current_event_id,
                            'device_id': device_id,
                            'start_time': current_time,
                            'duration_min': duration,
                            'energy_kwh': energy_kwh,
                            'max_delay_min': getattr(device_config, 'max_delay_min', 0),
                            'comfort_class': getattr(device_config, 'comfort_class', 'none')
                        })

                # Get active fraction for this interval
                active_fraction = device_inst.get_active_fraction(self.interval_minutes)
                
                # Compute average power with peak logic
                power = compute_average_power_with_peak(
                    device_config,
                    active_fraction,
                    self.interval_minutes,
                    is_first_interval,
                    self.random_state
                )
                
                # Advance device state
                should_turn_off = device_inst.advance(self.interval_minutes)
                if should_turn_off:
                    device_inst.turn_off()
            
            # Add noise
            power = add_noise(power, self.random_state)
            
            # Accumulate power
            if device_id not in timestep_power:
                timestep_power[device_id] = 0
            timestep_power[device_id] += power
            total_power += power
        
        # Store results
        self.timestamps.append(current_time)
        self.total_powers.append(total_power)
        for device_id in timestep_power:
            self.device_powers[device_id].append(timestep_power[device_id])
        
        # Fill in zeros for devices that didn't run
        for device_id in self.device_powers:
            if device_id not in timestep_power:
                self.device_powers[device_id].append(0)
    
    def _build_dataframe(self) -> pd.DataFrame:
        """Build final DataFrame with average power (W) columns"""
        data = {
            'timestamp': self.timestamps,
            'total_power_W': self.total_powers
        }
        
        # Add device columns (with sanitized names)
        for device_id, powers in self.device_powers.items():
            device_name = self.devices_config[device_id].name
            col_name = device_name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("/", "_")
            data[col_name + "_W"] = powers
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        self._print_summary(df)
        
        return df
    
    def _print_summary(self, df: pd.DataFrame):
        """Print summary statistics"""
        print(f"\n{'='*60}")
        print("GENERATION SUMMARY")
        print(f"{'='*60}")
        print(f"Household: {self.household.name}")
        
        duration_days = (self.end_date - self.start_date).days
        print(f"Simulation period: {duration_days} days")
        
        # Energy calculations (CORRECT: average power * time)
        interval_hours = self.interval_minutes / 60
        total_energy_kwh = (df['total_power_W'].sum() * interval_hours) / 1000
        avg_daily_kwh = total_energy_kwh / duration_days if duration_days > 0 else 0
        
        print(f"Total energy: {total_energy_kwh:.1f} kWh")
        print(f"Average daily: {avg_daily_kwh:.1f} kWh/day")
        
        # Peak and average demand
        peak_kw = df['total_power_W'].max() / 1000
        avg_kw = df['total_power_W'].mean() / 1000
        load_factor = (avg_kw / peak_kw * 100) if peak_kw > 0 else 0
        
        print(f"Peak demand: {peak_kw:.2f} kW")
        print(f"Average demand: {avg_kw:.2f} kW")
        print(f"Load factor: {load_factor:.1f}%")
        
        # Top energy consumers
        print(f"\nTop 5 energy consumers:")
        device_energies = {}
        for col in df.columns:
            if col.endswith('_W') and col != 'total_power_W':
                energy_kwh = (df[col].sum() * interval_hours) / 1000
                device_name = col[:-2].replace("_", " ")
                device_energies[device_name] = energy_kwh
        
        top_devices = sorted(device_energies.items(), key=lambda x: x[1], reverse=True)[:5]
        for idx, (device_name, energy) in enumerate(top_devices, 1):
            pct = (energy / total_energy_kwh * 100) if total_energy_kwh > 0 else 0
            print(f"  {idx}. {device_name}: {energy:.1f} kWh ({pct:.1f}%)")
        
        print(f"{'='*60}\n")


def generate_household_profile(
    household_id: str,
    start_date: datetime,
    end_date: datetime,
    interval_minutes: int = 10,
    config_dir: str = "config",
    random_seed: Optional[int] = None,
    return_activations: bool = False
) -> pd.DataFrame:
    """
    Generate a single household profile (PUBLIC API)
    
    Args:
        household_id: ID of household to generate
        start_date: Start date
        end_date: End date
        interval_minutes: Resolution in minutes
        config_dir: Path to config directory
        random_seed: Random seed for reproducibility
        return_activations: If True, returns (df, activations)
        
    Returns:
        DataFrame with timestamp index and average power (W) columns
        Energy can be computed as: kWh = sum(P_W * interval_minutes/60) / 1000
    """
    devices, households = load_config(config_dir)
    
    if household_id not in households:
        raise ValueError(f"Unknown household: {household_id}")
    
    generator = LoadProfileGenerator(
        households[household_id],
        devices,
        start_date,
        end_date,
        interval_minutes,
        random_seed,
        config_dir # Passed correctly now
    )
    
    # generate() now returns (df, activations)
    df, activations = generator.generate()
    
    if return_activations:
        return df, activations
        
    return df
