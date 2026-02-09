"""
Behavioral Models for Realistic Device Usage

Implements probabilistic models for:
- Time-of-day usage patterns
- Seasonal variations
- Day-of-week effects
- Cycling behavior for always-on devices
- Resolution-independent probability calculations
"""

import numpy as np
from datetime import datetime
from typing import Optional
import calendar


def get_season(date: datetime) -> str:
    """Determine season from date (Northern Hemisphere)"""
    month = date.month
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    else:  # 9, 10, 11
        return "fall"


def is_weekend(date: datetime) -> bool:
    """Check if date is weekend"""
    return date.weekday() >= 5


def is_holiday(date: datetime) -> bool:
    """Simple holiday detection (major US holidays)"""
    # New Year's Day
    if date.month == 1 and date.day == 1:
        return True
    # Independence Day
    if date.month == 7 and date.day == 4:
        return True
    # Christmas
    if date.month == 12 and date.day == 25:
        return True
    # Thanksgiving (4th Thursday of November)
    if date.month == 11:
        cal = calendar.monthcalendar(date.year, 11)
        thursdays = [week[3] for week in cal if week[3] != 0]
        if len(thursdays) >= 4 and date.day == thursdays[3]:
            return True
    return False


def get_occupancy_probability(
    hour: int,
    is_weekday: bool,
    weekday_schedule: list,
    weekend_schedule: list
) -> float:
    """
    Get probability someone is home and active at a given hour
    
    Args:
        hour: Hour of day (0-23)
        is_weekday: Whether it's a weekday
        weekday_schedule: List of schedule dicts for weekdays
        weekend_schedule: List of schedule dicts for weekends
        
    Returns:
        Probability (0-1) that household is active
    """
    schedule = weekday_schedule if is_weekday else weekend_schedule
    
    for period in schedule:
        start_hour = period['start_hour']
        end_hour = period['end_hour']
        occupancy_prob = period['occupancy_prob']
        
        # Handle periods that cross midnight
        if start_hour <= end_hour:
            if start_hour <= hour < end_hour:
                return occupancy_prob
        else:  # Crosses midnight
            if hour >= start_hour or hour < end_hour:
                return occupancy_prob
    
    return 0.5  # Default if not found


def get_time_usage_probability(
    hour: int,
    minute: int,
    time_preferences: list
) -> float:
    """
    Calculate usage probability based on time preferences
    
    Args:
        hour: Hour (0-23)
        minute: Minute (0-59)
        time_preferences: List of [start_hour, end_hour] windows
        
    Returns:
        Probability multiplier (0-1)
    """
    current_time = hour + minute / 60
    
    # Default very low probability
    base_prob = 0.05
    max_prob = 1.0
    
    # Check if current time falls within any preference window
    for window in time_preferences:
        start_hour, end_hour = window
        
        if start_hour <= end_hour:
            # Normal window
            if start_hour <= current_time < end_hour:
                # Peak in middle of window
                window_center = (start_hour + end_hour) / 2
                window_width = end_hour - start_hour
                
                # Gaussian peak centered on window
                distance = abs(current_time - window_center)
                prob = max_prob * np.exp(-0.5 * (distance / (window_width / 4))**2)
                return max(prob, base_prob)
        else:
            # Window crosses midnight
            if current_time >= start_hour or current_time < end_hour:
                # Adjust for midnight crossing
                if current_time >= start_hour:
                    adjusted_time = current_time - start_hour
                else:
                    adjusted_time = current_time + (24 - start_hour)
                
                window_width = (24 - start_hour) + end_hour
                window_center = window_width / 2
                distance = abs(adjusted_time - window_center)
                prob = max_prob * np.exp(-0.5 * (distance / (window_width / 4))**2)
                return max(prob, base_prob)
    
    return base_prob


def should_device_turn_on(
    device,
    current_time: datetime,
    occupancy_prob: float,
    device_multiplier: float,
    interval_minutes: int,
    random_state: np.random.RandomState
) -> bool:
    """
    Probabilistic decision: should device turn on at this time step?
    
    RESOLUTION-INDEPENDENT: Uses interval_minutes to compute correct probability.
    
    Args:
        device: DeviceConfig object
        current_time: Current datetime
        occupancy_prob: Probability house is occupied
        device_multiplier: Household-specific usage multiplier
        interval_minutes: Resolution in minutes (e.g., 10, 15, 60)
        random_state: Numpy random state for reproducibility
        
    Returns:
        True if device should turn on
    """
    # Handle cycling devices separately
    if device.pattern == 'cycling':
        # These are handled deterministically by the generator
        return False
    
    # Handle continuous devices (stay on always)
    if device.pattern == 'continuous':
        return True
    
    hour = current_time.hour
    minute = current_time.minute
    
    # Get seasonal factor
    season = get_season(current_time)
    seasonal_factor = device.seasonal_factor.get(season, 1.0)
    
    # If seasonal factor is 0, device is completely off
    if seasonal_factor == 0.0:
        return False
    
    # Weekday factor
    is_weekday_val = not is_weekend(current_time) and not is_holiday(current_time)
    weekday_factor = device.weekday_factor if is_weekday_val else (2.0 - device.weekday_factor)
    
    # Time-of-day probability
    time_prob = get_time_usage_probability(hour, minute, device.time_preferences)
    
    # Calculate expected uses per day
    if device.expected_uses_per_day is not None:
        uses_per_day = device.expected_uses_per_day
    elif device.expected_uses_per_week is not None:
        uses_per_day = device.expected_uses_per_week / 7
    elif device.expected_uses_per_month is not None:
        uses_per_day = device.expected_uses_per_month / 30
    else:
        uses_per_day = 1.0
    
    # CRITICAL FIX: Compute intervals per day from resolution
    intervals_per_day = (24 * 60) / interval_minutes
    
    # Base probability per interval
    base_prob = uses_per_day / intervals_per_day
    
    # Occupancy multiplier (only for devices that require it)
    if device.requires_occupancy:
        occupancy_multiplier = occupancy_prob
    else:
        occupancy_multiplier = 1.0
    
    # Combine all factors
    total_prob = (
        base_prob *
        time_prob *
        occupancy_multiplier *
        seasonal_factor *
        weekday_factor *
        device_multiplier
    )
    
    # Cap probability at 95% to prevent certainty
    total_prob = min(total_prob, 0.95)
    
    # Random decision
    return random_state.random() < total_prob


def get_cycling_state(
    time_offset_minutes: float,
    cycle_on_min: float,
    cycle_off_min: float,
    phase_offset: float
) -> bool:
    """
    Determine if a cycling device (fridge/freezer) is currently ON
    
    Uses deterministic duty cycle with phase offset for realistic cycling.
    
    Args:
        time_offset_minutes: Minutes since simulation start
        cycle_on_min: Duration compressor runs (minutes)
        cycle_off_min: Duration compressor is off (minutes)
        phase_offset: Random phase offset (0-1) to desynchronize devices
        
    Returns:
        True if compressor should be ON
    """
    cycle_period = cycle_on_min + cycle_off_min
    
    # Apply phase offset
    adjusted_time = time_offset_minutes + (phase_offset * cycle_period)
    
    # Position in cycle
    position_in_cycle = adjusted_time % cycle_period
    
    # ON during first part of cycle
    return position_in_cycle < cycle_on_min


def get_cycling_duty_fraction(
    time_offset_minutes: float,
    interval_minutes: float,
    cycle_on_min: float,
    cycle_off_min: float,
    phase_offset: float
) -> float:
    """
    Compute fraction of interval that cycling device is ON
    
    This handles intervals that span multiple cycles or partial cycles,
    providing resolution-correct energy accounting.
    
    Args:
        time_offset_minutes: Current time offset from start of simulation
        interval_minutes: Duration of current interval
        cycle_on_min: Duration compressor runs
        cycle_off_min: Duration compressor is off
        phase_offset: Random phase offset (0-1)
        
    Returns:
        Fraction of interval spent ON [0, 1]
    """
    cycle_period = cycle_on_min + cycle_off_min
    
    # Apply phase offset
    start_time = time_offset_minutes + (phase_offset * cycle_period)
    end_time = start_time + interval_minutes
    
    # For very short cycles or very long intervals, integrate properly
    on_time = 0.0
    
    # Sample at sub-interval resolution for accuracy
    num_samples = max(10, int(interval_minutes / 5))  # At least 10 samples
    dt = interval_minutes / num_samples
    
    for i in range(num_samples):
        t = start_time + i * dt
        position = t % cycle_period
        if position < cycle_on_min:
            on_time += dt
    
    return on_time / interval_minutes


def get_device_duration(
    device,
    random_state: np.random.RandomState
) -> int:
    """
    Sample device usage duration in minutes
    
    Returns:
        Duration in minutes
    """
    # Add variance to average duration (±30%)
    std_dev = device.avg_duration_min * 0.3
    duration = random_state.normal(device.avg_duration_min, std_dev)
    
    # Ensure reasonable bounds
    min_duration = max(5, device.avg_duration_min * 0.5)
    max_duration = device.avg_duration_min * 2.0
    
    duration = np.clip(duration, min_duration, max_duration)
    
    return int(duration)


def get_device_power(
    device,
    is_active: bool,
    time_since_start: int,
    total_duration: int,
    random_state: np.random.RandomState
) -> float:
    """
    Get instantaneous power consumption
    
    Args:
        device: DeviceConfig object
        is_active: Whether device is currently on
        time_since_start: Minutes since device turned on
        total_duration: Total expected duration
        random_state: Random state
        
    Returns:
        Power in watts
    """
    if not is_active:
        return device.standby_power_w
    
    # Some devices have startup peaks
    if time_since_start < 2:  # First 2 minutes
        # Chance of peak power
        if random_state.random() < 0.3:
            power = device.peak_power_w
        else:
            power = device.active_power_w
    else:
        # Normal active power with small variance
        variance = device.active_power_w * 0.1
        power = random_state.normal(device.active_power_w, variance)
        power = np.clip(power, device.active_power_w * 0.8, device.peak_power_w)
    
    return power


def add_noise(power: float, random_state: np.random.RandomState) -> float:
    """
    Add realistic measurement noise
    
    Args:
        power: True power value
        random_state: Random state
        
    Returns:
        Power with noise
    """
    # Add small Gaussian noise (±2%)
    noise = random_state.normal(0, power * 0.02)
    return max(0, power + noise)
