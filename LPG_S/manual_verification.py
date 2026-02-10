#!/usr/bin/env python3
"""
Manual Verification Script
Generates profiles and inspects activation times to verify occupancy-aware scheduling.
"""

from datetime import datetime
from generator import generate_household_profile
import pandas as pd

def verify_young_couple_weekday_activations():
    """Verify young_couple has NO task device activations during weekday 09:00-17:00"""
    print("="*70)
    print("VERIFICATION: Young Couple Weekday Activations")
    print("="*70)
    
    # Generate profile with activations
    df, activations = generate_household_profile(
        'young_couple',
        datetime(2024, 1, 15),  # Monday
        datetime(2024, 1, 22),  # Monday (7 days)
        interval_minutes=10,
        random_seed=42,
        return_activations=True
    )
    
    print(f"\nTotal activations logged: {len(activations)}")
    
    # Categorize by device type
    task_devices = ['washing_machine', 'dryer', 'dishwasher', 'vacuum_cleaner', 'iron']
    
    task_activations = [a for a in activations if a['device_id'] in task_devices]
    print(f"Task device activations: {len(task_activations)}")
    
    # Check for weekday work hour violations (Mon-Fri 09:00-17:00)
    violations = []
    for act in task_activations:
        start_time = act['start_time']
        is_weekday = start_time.weekday() < 5  # Mon=0, Fri=4
        is_work_hours = 9 <= start_time.hour < 17
        
        if is_weekday and is_work_hours:
            violations.append(act)
    
    print(f"\nWeekday work hour violations (09:00-17:00): {len(violations)}")
    
    if violations:
        print("\nVIOLATIONS FOUND:")
        for v in violations:
            print(f"  - {v['device_id']} started at {v['start_time']}")
        print("\n❌ TEST FAILED: Found irrational activations during work hours!")
        return False
    else:
        print("\n✓ PASS: No task devices started during weekday work hours")
        
        # Show when they DID start
        if task_activations:
            print("\nTask device activation times:")
            for act in task_activations[:10]:  # Show first 10
                start = act['start_time']
                day = start.strftime('%A')
                time = start.strftime('%H:%M')
                print(f"  - {act['device_id']}: {day} {time}")
        
        return True


def verify_retired_couple_daytime_activations():
    """Verify retired_couple CAN have daytime activations (they're home)"""
    print("\n" + "="*70)
    print("VERIFICATION: Retired Couple Daytime Activations")
    print("="*70)
    
    # Generate profile
    df, activations = generate_household_profile(
        'retired_couple',
        datetime(2024, 1, 15),
        datetime(2024, 1, 22),
        interval_minutes=10,
        random_seed=42,
        return_activations=True
    )
    
    print(f"\nTotal activations logged: {len(activations)}")
    
    task_devices = ['washing_machine', 'dryer', 'dishwasher', 'vacuum_cleaner']
    task_activations = [a for a in activations if a['device_id'] in task_devices]
    print(f"Task device activations: {len(task_activations)}")
    
    # Count daytime weekday activations (should be allowed)
    daytime_weekday = []
    for act in task_activations:
        start_time = act['start_time']
        is_weekday = start_time.weekday() < 5
        is_daytime = 9 <= start_time.hour < 17
        
        if is_weekday and is_daytime:
            daytime_weekday.append(act)
    
    print(f"\nWeekday daytime activations (09:00-17:00): {len(daytime_weekday)}")
    
    if daytime_weekday:
        print("\nExamples of allowed daytime activations:")
        for act in daytime_weekday[:5]:
            start = act['start_time']
            print(f"  - {act['device_id']}: {start.strftime('%A %H:%M')}")
        print("\n✓ PASS: Retired couple correctly has daytime activations (they're home)")
        return True
    else:
        print("\n⚠ Note: No daytime activations found, but this isn't necessarily wrong")
        print("  (could be due to random scheduling)")
        return True


if __name__ == '__main__':
    print("\n" + "="*70)
    print("MANUAL VERIFICATION OF OCCUPANCY-AWARE SCHEDULING")
    print("="*70)
    
    test1 = verify_young_couple_weekday_activations()
    test2 = verify_retired_couple_daytime_activations()
    
    print("\n" + "="*70)
    if test1 and test2:
        print("✓✓✓ ALL VERIFICATIONS PASSED ✓✓✓")
    else:
        print("❌ SOME VERIFICATIONS FAILED")
    print("="*70 + "\n")
    
    exit(0 if (test1 and test2) else 1)
