
import random
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from behavior import get_occupancy_probability, is_weekend, is_holiday

logger = logging.getLogger(__name__)

@dataclass
class ScheduledEvent:
    device_id: str
    start_time: datetime
    end_time: datetime
    duration_min: int
    energy_kwh: float

class QuotaScheduler:
    """
    Schedules task-based device events based on weekly quotas and behavior profiles.
    Replaces probabilistic turn-on logic for 'task' devices.
    """
    
    def __init__(
        self, 
        behavior_profile: Dict, 
        devices_config: Dict[str, Any], 
        household_config: Any, # New: Need full household config for occupancy schedules
        year: int, 
        random_seed: int = 42
    ):
        self.profile = behavior_profile
        self.devices = devices_config
        self.household = household_config
        self.year = year
        self.rng = random.Random(random_seed)
        self.np_rng = np.random.RandomState(random_seed)
        self.events: List[ScheduledEvent] = []
        
    def generate_schedule(self) -> List[ScheduledEvent]:
        """
        Generate a full year's schedule for all devices in the profile.
        """
        self.events = []
        
        # 1. Identify task devices from the profile
        task_devices = self.profile.get('weekly_device_usage', {}).keys()
        
        # 2. Process independent devices first
        independent_devices = [d for d in task_devices 
                             if 'linked_to' not in self.profile['weekly_device_usage'][d]]
                             
        for device_id in independent_devices:
            if device_id in self.devices:
                self._schedule_device_year(device_id)
                
        # 3. Process dependent/linked devices (e.g., dryer after washer)
        dependent_devices = [d for d in task_devices 
                           if 'linked_to' in self.profile['weekly_device_usage'][d]]
                           
        for device_id in dependent_devices:
            if device_id in self.devices:
                self._schedule_dependent_device_year(device_id)
                
        # Sort events by start time
        self.events.sort(key=lambda x: x.start_time)
        return self.events

    def _schedule_device_year(self, device_id: str):
        """Schedule a single independent device for the whole year"""
        config = self.profile['weekly_device_usage'][device_id]
        dev_meta = self.devices[device_id]
        
        # Iterate through weeks
        current_date = datetime(self.year, 1, 1)
        while current_date.year == self.year:
            # Determine quota for this week
            mean_uses = config.get('mean_uses_per_week', 3.0)
            std_uses = config.get('std_uses_per_week', 0.0)
            
            # Sample N events for this week
            if std_uses > 0:
                n_events = int(round(self.np_rng.normal(mean_uses, std_uses)))
            else:
                n_events = int(round(mean_uses))
            
            n_events = max(0, n_events)
            
            # Distribute events across days
            week_events = self._distribute_events_in_week(n_events, config, current_date, dev_meta)
            self.events.extend(week_events)
            
            # Advance to next week
            current_date += timedelta(days=7)

    def _schedule_dependent_device_year(self, device_id: str):
        """Schedule a device that depends on another (e.g., Dryer follows Washer)"""
        config = self.profile['weekly_device_usage'][device_id]
        parent_id = config['linked_to']
        link_window_hours = config.get('link_window_hours', 24)
        dev_meta = self.devices[device_id]
        
        # Get all parent events
        parent_events = [e for e in self.events if e.device_id == parent_id]
        
        # Occupancy thresholds
        threshold_weekday = self.profile.get('min_occupancy_to_start_weekday', 0.0)
        threshold_weekend = self.profile.get('min_occupancy_to_start_weekend', 0.0)

        for p_event in parent_events:
            parent_mean = self.profile['weekly_device_usage'][parent_id]['mean_uses_per_week']
            child_mean = config['mean_uses_per_week']
            follow_prob = min(1.0, child_mean / parent_mean) if parent_mean > 0 else 0
            
            if self.rng.random() < follow_prob:
                success = False
                attempts = 0
                max_attempts = 10
                
                while not success and attempts < max_attempts:
                    attempts += 1
                    delay_min = self.rng.randint(10, link_window_hours * 60)
                    start_time = p_event.end_time + timedelta(minutes=delay_min)
                    
                    # Check occupancy if required
                    requires_presence = getattr(dev_meta, 'requires_presence_to_start', getattr(dev_meta, 'requires_occupancy', False))
                    can_start_unattended = getattr(dev_meta, 'can_start_unattended', False)
                    
                    if requires_presence and not can_start_unattended:
                        is_wd = not is_weekend(start_time) and not is_holiday(start_time)
                        threshold = threshold_weekday if is_wd else threshold_weekend
                        
                        prob = get_occupancy_probability(
                            start_time.hour, 
                            is_wd, 
                            self.household.weekday_schedule, 
                            self.household.weekend_schedule
                        )
                        
                        if prob < threshold:
                            continue # Try another delay
                            
                    # Valid
                    duration = int(self.np_rng.normal(dev_meta.avg_duration_min, dev_meta.avg_duration_min * 0.1))
                    duration = max(10, duration)
                    energy = (dev_meta.active_power_w * duration / 60) / 1000
                    
                    self.events.append(ScheduledEvent(
                        device_id=device_id,
                        start_time=start_time,
                        end_time=start_time + timedelta(minutes=duration),
                        duration_min=duration,
                        energy_kwh=energy
                    ))
                    success = True

    def _distribute_events_in_week(self, n_events: int, config: Dict, week_start: datetime, dev_meta: Any) -> List[ScheduledEvent]:
        """Distribute N events into specific days and times of the week"""
        events = []
        days_dist = config.get('preferred_days', {'mon': 1})
        days_map = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
        
        # Convert map to probability list
        probs = [days_dist.get(d, 0) for d in days_map]
        total_p = sum(probs)
        if total_p == 0:
            probs = [1/7] * 7
        else:
            probs = [p/total_p for p in probs]
            
        # Select days
        if n_events > 0:
            selected_indices = self.np_rng.choice(7, size=n_events, p=probs)
            
            # Group by day to enforce max_uses_per_day
            from collections import Counter
            day_counts = Counter(selected_indices)
            
            max_per_day = config.get('max_uses_per_day', 99)
            
            for day_idx, count in day_counts.items():
                actual_count = min(count, max_per_day)
                day_date = week_start + timedelta(days=int(day_idx))
                
                # Schedule times for this day
                day_events = self._schedule_times_for_day(actual_count, config, day_date, dev_meta)
                events.extend(day_events)
                
        return events

    def _schedule_times_for_day(self, count: int, config: Dict, day_date: datetime, dev_meta: Any) -> List[ScheduledEvent]:
        events = []
        
        # Determine day type
        is_wd = not is_weekend(day_date) and not is_holiday(day_date)
        
        # Select windows based on day type
        if is_wd:
             windows = config.get('start_time_windows_weekday', config.get('start_time_windows', [[8, 22]]))
             threshold = self.profile.get('min_occupancy_to_start_weekday', 0.0)
        else:
             windows = config.get('start_time_windows_weekend', config.get('start_time_windows', [[8, 22]]))
             threshold = self.profile.get('min_occupancy_to_start_weekend', 0.0)
             
        min_gap = config.get('min_gap_min', 60)
        
        last_end_time = day_date # Start of day
        
        requires_presence = getattr(dev_meta, 'requires_presence_to_start', getattr(dev_meta, 'requires_occupancy', False))
        can_start_unattended = getattr(dev_meta, 'can_start_unattended', False)
        
        for _ in range(count):
            slot_found = False
            
            # Attempt to find valid slot in PREFERRED windows
            limit_attempts = 20
            for _ in range(limit_attempts):
                # Pick a window
                if len(windows) > 0: 
                    w = self.rng.choice(windows)
                    start_h, end_h = w
                else: 
                    start_h, end_h = 8, 22
                
                hour = self.rng.randint(start_h, end_h - 1) if end_h > start_h else start_h
                minute = self.rng.randint(0, 59)
                hour = max(0, min(23, hour))
                
                start_dt = day_date.replace(hour=hour, minute=minute)
                
                # Check gap
                if start_dt < last_end_time + timedelta(minutes=min_gap if len(events)>0 else 0):
                    continue

                # Check Occupancy
                if requires_presence and not can_start_unattended:
                    prob = get_occupancy_probability(
                        start_dt.hour, 
                        is_wd, 
                        self.household.weekday_schedule, 
                        self.household.weekend_schedule
                    )
                    if prob < threshold:
                        continue # Resample
                
                # Valid slot found
                slot_found = True
                self._add_event(events, dev_meta, start_dt)
                last_end_time = events[-1].end_time
                break
            
            if not slot_found and requires_presence:
                 # FALLBACK: Ignore preference windows, find ANY high occupancy time in the day
                 # This prevents skipping tasks just because preferred window (e.g. evening) was full or unlucky
                 fallback_attempts = 50
                 for _ in range(fallback_attempts):
                     hour = self.rng.randint(6, 23) # Reasonable waking hours
                     minute = self.rng.randint(0, 59)
                     start_dt = day_date.replace(hour=hour, minute=minute)
                     
                     if start_dt < last_end_time + timedelta(minutes=min_gap if len(events)>0 else 0): continue
                     
                     prob = get_occupancy_probability(
                        start_dt.hour, is_wd, self.household.weekday_schedule, self.household.weekend_schedule
                     )
                     if prob >= threshold:
                         self._add_event(events, dev_meta, start_dt)
                         last_end_time = events[-1].end_time
                         slot_found = True
                         break
            
            # If still not found, we effectively drop this event instance (quota reduced for this week)
            # This is acceptable to avoid irrational starts.
                
        return events

    def _add_event(self, events_list, dev_meta, start_dt):
        duration = int(self.np_rng.normal(dev_meta.avg_duration_min, dev_meta.avg_duration_min * 0.1))
        duration = max(10, duration)
        end_dt = start_dt + timedelta(minutes=duration)
        energy_est = (dev_meta.active_power_w * duration / 60) / 1000
        
        events_list.append(ScheduledEvent(
            device_id=dev_meta.id,
            start_time=start_dt,
            end_time=end_dt,
            duration_min=duration,
            energy_kwh=energy_est
        ))

