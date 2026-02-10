#!/usr/bin/env python3
"""
Irrationality Report - Detects irrational device activations
Ensures devices requiring presence don't start when nobody is home.

EXIT CODE:
  0 - All checks passed
  1 - Violations detected (FAIL)
"""

import argparse
import pandas as pd
import numpy as np
import yaml
import os
from datetime import datetime, timedelta
from generator import generate_household_profile
from config_loader import load_config
from behavior import get_occupancy_probability, is_weekend, is_holiday

def check_irrationality(household_id: str, start_date: datetime, end_date: datetime, random_seed: int = 42):
    """
    Check for irrational device activations (devices starting when nobody is home).
    
    Returns:
        (num_violations, total_events, violations_list)
    """
    print(f"\n{'='*70}")
    print(f"IRRATIONALITY CHECK: {household_id}")
    print(f"{'='*70}")
    
    # Load configs
    devices, households = load_config("config")
    
    if household_id not in households:
        print(f"ERROR: Unknown household {household_id}")
        return 0, 0, []
        
    household = households[household_id]
    
    # Load behavior profile for thresholds
    with open("config/behavior_profiles.yaml", "r") as f:
        profiles = yaml.safe_load(f)['behavior_profiles']
        
    if household_id not in profiles:
        print(f"WARNING: No behavior profile for {household_id}, skipping")
        return 0, 0, []
        
    profile = profiles[household_id]
    threshold_weekday = profile.get('min_occupancy_to_start_weekday', 0.0)
    threshold_weekend = profile.get('min_occupancy_to_start_weekend', 0.0)
    
    # Generate profile with activations
    print(f"Generating profile from {start_date.date()} to {end_date.date()}...")
    df, activations = generate_household_profile(
        household_id,
        start_date,
        end_date,
        interval_minutes=10,
        random_seed=random_seed,
        return_activations=True
    )
    
    if not activations:
        print("No task activations found.")
        return 0, 0, []
    
    # Convert to DataFrame for analysis
    act_df = pd.DataFrame(activations)
    
    violations = []
    
    print(f"\nChecking {len(act_df)} task activations for occupancy violations...")
    
    for idx, act in act_df.iterrows():
        device_id = act['device_id']
        start_time = pd.to_datetime(act['start_time'])
        
        # Get device config
        if device_id not in devices:
            continue
            
        dev_config = devices[device_id]
        
        # Check if device requires presence
        requires_presence = getattr(dev_config, 'requires_presence_to_start', 
                                   getattr(dev_config, 'requires_occupancy', False))
        can_start_unattended = getattr(dev_config, 'can_start_unattended', False)
        
        if not requires_presence or can_start_unattended:
            continue  # This device can start anytime
            
        # Check occupancy at start time
        # Treat holidays as weekend if configured
        treat_holidays_as_weekend = profile.get('treat_holidays_as_weekend', True)
        is_holiday_today = is_holiday(start_time)
        
        if treat_holidays_as_weekend and is_holiday_today:
            is_wd = False  # Treat holiday as weekend
        else:
            is_wd = not is_weekend(start_time) and not is_holiday_today
            
        threshold = threshold_weekday if is_wd else threshold_weekend
        
        occupancy_prob = get_occupancy_probability(
            start_time.hour,
            is_wd,
            household.weekday_schedule,
            household.weekend_schedule
        )
        
        # Violation if occupancy is below threshold
        if occupancy_prob < threshold:
            violations.append({
                'device_id': device_id,
                'start_time': start_time,
                'day_type': 'weekday' if is_wd else 'weekend',
                'occupancy_prob': occupancy_prob,
                'threshold': threshold,
                'violation_severity': threshold - occupancy_prob
            })
    
    # Report results
    total_checked = len([a for a in activations if getattr(devices.get(a['device_id']), 
                        'requires_presence_to_start', False)])
    
    print(f"\n--- RESULTS ---")
    print(f"Total events checked: {total_checked}")
    print(f"Violations found: {len(violations)}")
    
    if violations:
        print(f"\n{'='*70}")
        print("VIOLATIONS DETECTED (devices started when nobody home):")
        print(f"{'='*70}")
        
        # Group by device
        df_viol = pd.DataFrame(violations)
        for device_id in df_viol['device_id'].unique():
            dev_violations = df_viol[df_viol['device_id'] == device_id]
            print(f"\n{device_id}:")
            print(f"  Violations: {len(dev_violations)}")
            print(f"  Example timestamps:")
            for _, v in dev_violations.head(3).iterrows():
                print(f"    - {v['start_time']} ({v['day_type']}) "
                      f"occupancy={v['occupancy_prob']:.2f} < threshold={v['threshold']:.2f}")
        
        print(f"\n{'='*70}")
        print("❌ IRRATIONALITY CHECK FAILED")
        print(f"{'='*70}\n")
        return len(violations), total_checked, violations
    else:
        print("\n✓ ALL CHECKS PASSED - No irrational activations detected\n")
        return 0, total_checked, []


def main():
    parser = argparse.ArgumentParser(description='Check for irrational device activations')
    parser.add_argument('--household', type=str, default=None, 
                       help='Specific household to check (default: all)')
    parser.add_argument('--days', type=int, default=7,
                       help='Number of days to check (default: 7)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    if args.household:
        households_to_check = [args.household]
    else:
        households_to_check = ['single_professional', 'young_couple', 'family_with_children', 
                              'large_family', 'retired_couple']
    
    total_violations = 0
    
    start_date = datetime(2024, 1, 15)  # Mid-month to avoid holidays
    end_date = start_date + timedelta(days=args.days)
    
    for hh_id in households_to_check:
        num_violations, total_events, violations = check_irrationality(
            hh_id, start_date, end_date, args.seed
        )
        total_violations += num_violations
    
    print(f"\n{'='*70}")
    if total_violations > 0:
        print(f"OVERALL RESULT: {total_violations} VIOLATIONS DETECTED")
        print(f"{'='*70}\n")
        exit(1)
    else:
        print("OVERALL RESULT: ALL HOUSEHOLDS PASSED")
        print(f"{'='*70}\n")
        exit(0)


if __name__ == "__main__":
    main()
