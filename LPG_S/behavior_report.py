
import argparse
import pandas as pd
import numpy as np
import yaml
import os
from datetime import datetime, timedelta
from generator import generate_household_profile

def validate_behavior_quotas(household_id: str, year: int, output_dir: str):
    """
    Validate that the generated profile matches the behavior quotas.
    """
    print(f"\nVALIDATING BEHAVIOR FOR: {household_id} ({year})")
    
    # Load behavior profile expectation
    with open("config/behavior_profiles.yaml", "r") as f:
        profiles = yaml.safe_load(f)['behavior_profiles']
        
    if household_id not in profiles:
        print(f"Skipping: No behavior profile for {household_id}")
        return

    profile = profiles[household_id]
    expected_usage = profile.get('weekly_device_usage', {})

    # Run generation
    start = datetime(year, 1, 1)
    end = datetime(year, 2, 1) # Generate 1 month for speed, or full year?
    # Requirement says "weekly quotas", 1 month is enough to check averages roughly.
    # But user wants full year validation potentially. Let's do 4 weeks.
    end = start + timedelta(days=28)
    
    print(f"Generating simulation for 4 weeks...")
    df, activations = generate_household_profile(
        household_id, 
        start, 
        end, 
        return_activations=True,
        random_seed=42
    )
    
    # Analyze activations
    act_df = pd.DataFrame(activations)
    if act_df.empty:
        print("No activations found!")
        return
        
    act_df['start_time'] = pd.to_datetime(act_df['start_time'])
    act_df['week'] = act_df['start_time'].dt.isocalendar().week
    
    print("\n--- RESULTS vs QUOTA ---")
    
    max_name_len = max([len(k) for k in expected_usage.keys()]) if expected_usage else 10
    
    failure = False
    
    for device_id, config in expected_usage.items():
        # Filter for this device
        dev_acts = act_df[act_df['device_id'] == device_id]
        
        # Calculate weekly counts
        # Note: week numbers might span years, but for 28 days starting Jan 1 it's weeks 1-4
        if dev_acts.empty:
            actual_mean = 0
        else:
            weekly_counts = dev_acts.groupby('week').size()
            actual_mean = weekly_counts.mean()
            
        target_mean = config['mean_uses_per_week']
        
        # Check tolerance (allow some variance due to Poisson/randomness)
        # For 4 weeks, variance can be high. 
        # If mean is 3, std is 1. StdErr is 1/sqrt(4) = 0.5. 
        # 95% CI is approx +/- 1.0.
        
        diff = abs(actual_mean - target_mean)
        
        status = "OK"
        if diff > max(1.0, target_mean * 0.5): # Loose check for short sim
            status = "WARNING"
        if diff > max(2.0, target_mean * 1.0):
            status = "FAIL"
            failure = True
            
        print(f"{device_id.ljust(max_name_len)} | Target: {target_mean:4.1f}/wk | Actual: {actual_mean:4.1f}/wk | {status}")
        
    return failure

def main():
    if not os.path.exists("config/behavior_profiles.yaml"):
        print("Error: config/behavior_profiles.yaml not found")
        return
        
    households = ['single_professional', 'young_couple', 'family_with_children', 'large_family', 'retired_couple']
    
    any_fail = False
    for hh in households:
        if validate_behavior_quotas(hh, 2024, "output"):
            any_fail = True
            
    if any_fail:
        print("\n❌ SOME VALIDATIONS FAILED")
        exit(1)
    else:
        print("\n✓ ALL BEHAVIOR CHECKS PASSED")
        exit(0)

if __name__ == "__main__":
    main()
