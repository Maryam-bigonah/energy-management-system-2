#!/usr/bin/env python3
"""
Household Services Analyzer
============================
Converts sub-metered household device loads into service-based format for optimization.

This script processes multi-carrier household data (electricity, water, gas) and:
1. Creates a Device/Service Inventory table (blueprint for optimization)
2. Extracts service events for hard-shiftable appliances (dishwasher, washing machine)
3. Generates baseline non-shiftable electricity profile
4. Outputs one comprehensive Excel workbook with all results

Author: Energy Management System Project
Date: 2026-02-07
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - EDIT THESE PATHS
# ============================================================================

# Input CSV files
ELECTRICITY_CSV = "/Volumes/M@RI@/CHS04 Retired Couple, no work/Reports/DeviceDurationCurves.Electricity.csv"
COLDWATER_CSV = "/Volumes/M@RI@/CHS04 Retired Couple, no work/Reports/DeviceDurationCurves.Cold Water.csv"
WARMWATER_CSV = "/Volumes/M@RI@/CHS04 Retired Couple, no work/Reports/DeviceDurationCurves.Warm Water.csv"
HOTWATER_CSV = "/Volumes/M@RI@/CHS04 Retired Couple, no work/Reports/DeviceDurationCurves.Hot water.csv"
GASOLINE_CSV = "/Volumes/M@RI@/CHS04 Retired Couple, no work/Reports/DeviceDurationCurves.Gasoline.csv"

# Output directory for CSV files
OUTPUT_DIR = "/Users/mariabigonah/Desktop/thesis/anti/pv/load"
OUTPUT_PREFIX = "services_RetiredCouple"

# Event detection parameters
POWER_THRESHOLD_W = 10  # Minimum power to consider device active
WATER_THRESHOLD_LPM = 0.01  # Minimum water flow to consider device active (L/min)
MIN_GAP_MINUTES = 5  # Merge events if gap is less than this
MIN_EVENT_DURATION = 5  # Minimum event duration in minutes

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def normalize_column_name(col):
    """Normalize column names: strip spaces, lowercase, remove units in brackets"""
    col = str(col).strip()
    # Remove units in brackets like [W], [L/min], etc.
    if '[' in col and ']' in col:
        col = col.split('[')[0].strip()
    return col

def load_csv_robust(filepath, carrier_name):
    """Load CSV with robust column name handling"""
    try:
        df = pd.read_csv(filepath, sep=';' if ';' in open(filepath).readline() else ',')
        print(f"✓ Loaded {carrier_name}: {filepath}")
        print(f"  Shape: {df.shape}, Columns: {len(df.columns)}")
        
        # Normalize column names
        df.columns = [normalize_column_name(c) for c in df.columns]
        
        # Try to parse time column
        time_col = None
        for col in ['Time', 'time', 'DateTime', 'datetime', 'Timestamp']:
            if col.lower() in [c.lower() for c in df.columns]:
                time_col = col
                break
        
        if time_col:
            df['datetime'] = pd.to_datetime(df[time_col], errors='coerce')
            if df['datetime'].isna().all():
                # Create dummy datetime starting from 2000-01-01
                print(f"  Warning: Could not parse time column, creating dummy datetime")
                df['datetime'] = pd.date_range('2000-01-01', periods=len(df), freq='1min')
        else:
            # No time column, create dummy
            print(f"  Warning: No time column found, creating dummy datetime")
            df['datetime'] = pd.date_range('2000-01-01', periods=len(df), freq='1min')
        
        return df
    except Exception as e:
        print(f"✗ Error loading {carrier_name}: {e}")
        return None

def detect_service_events(power_series, water_series, device_name, datetime_series):
    """
    Detect discrete service events from power and water time series
    
    Args:
        power_series: Power consumption in Watts (or None)
        water_series: Water flow in L/min (or None)
        device_name: Name of the device
        datetime_series: Datetime index
        
    Returns:
        List of event dictionaries
    """
    # Create binary active signal
    active = pd.Series(False, index=datetime_series.index)
    
    if power_series is not None:
        active |= (power_series > POWER_THRESHOLD_W)
    
    if water_series is not None:
        active |= (water_series > WATER_THRESHOLD_LPM)
    
    # Find transitions
    transitions = active.astype(int).diff()
    start_indices = transitions[transitions == 1].index
    end_indices = transitions[transitions == -1].index
    
    # Handle edge cases
    if active.iloc[0]:
        start_indices = pd.Index([0]).append(start_indices)
    if active.iloc[-1]:
        end_indices = end_indices.append(pd.Index([len(active) - 1]))
    
    # Match starts with ends
    events = []
    for start_idx in start_indices:
        # Find next end
        matching_ends = end_indices[end_indices > start_idx]
        if len(matching_ends) == 0:
            end_idx = len(active) - 1
        else:
            end_idx = matching_ends[0]
        
        # Calculate duration
        start_time = datetime_series.iloc[start_idx]
        end_time = datetime_series.iloc[end_idx]
        duration_min = (end_idx - start_idx)  # Assuming 1-minute resolution
        
        if duration_min >= MIN_EVENT_DURATION:
            # Calculate energy and water
            event_slice = slice(start_idx, end_idx + 1)
            
            electricity_kWh = 0
            if power_series is not None:
                # Sum power in W, convert to kWh (1 min = 1/60 h)
                electricity_kWh = power_series.iloc[event_slice].sum() / 1000 / 60
            
            coldwater_L = 0
            if water_series is not None:
                # Sum flow in L/min * 1 min
                coldwater_L = water_series.iloc[event_slice].sum()
            
            events.append({
                'device_name': device_name,
                'start_datetime': start_time,
                'end_datetime': end_time,
                'duration_min': duration_min,
                'electricity_kWh': round(electricity_kWh, 4),
                'coldwater_L': round(coldwater_L, 2)
            })
    
    # Merge close events (gaps < MIN_GAP_MINUTES)
    merged_events = []
    if events:
        current = events[0].copy()
        
        for next_event in events[1:]:
            gap_min = (next_event['start_datetime'] - current['end_datetime']).total_seconds() / 60
            
            if gap_min <= MIN_GAP_MINUTES:
                # Merge
                current['end_datetime'] = next_event['end_datetime']
                current['duration_min'] += next_event['duration_min']
                current['electricity_kWh'] += next_event['electricity_kWh']
                current['coldwater_L'] += next_event['coldwater_L']
            else:
                # Save current and start new
                merged_events.append(current)
                current = next_event.copy()
        
        merged_events.append(current)
    
    return merged_events

# ============================================================================
# STEP 1: CREATE SERVICE INVENTORY TABLE
# ============================================================================

def create_service_inventory():
    """Create the service inventory blueprint for the optimization model"""
    
    inventory = []
    
    # Non-shiftable services
    inventory.append({
        'ServiceName': 'Lighting',
        'DeviceColumnName(s)': 'Various lighting devices',
        'EnergyCarrier(s)': 'Electricity',
        'ShiftabilityClass': 'Non-shiftable',
        'WhatIsFixed': 'User occupancy pattern, lux requirements',
        'WhatIsDecision': 'None (baseline)',
        'Constraints': 'User-driven, immediate need',
        'Notes': 'Follows occupancy pattern, cannot be shifted'
    })
    
    inventory.append({
        'ServiceName': 'Cooking',
        'DeviceColumnName(s)': 'Cooker, Oven, Microwave, etc.',
        'EnergyCarrier(s)': 'Electricity',
        'ShiftabilityClass': 'Non-shiftable',
        'WhatIsFixed': 'Meal time preferences, cooking duration',
        'WhatIsDecision': 'None (baseline)',
        'Constraints': 'User-driven at meal times',
        'Notes': 'Tied to meal schedule, cannot defer cooking'
    })
    
    inventory.append({
        'ServiceName': 'Entertainment',
        'DeviceColumnName(s)': 'TV, Audio, Set-top box',
        'EnergyCarrier(s)': 'Electricity',
        'ShiftabilityClass': 'Non-shiftable',
        'WhatIsFixed': 'User viewing/listening preferences',
        'WhatIsDecision': 'None (baseline)',
        'Constraints': 'User-driven, real-time service',
        'Notes': 'Follows user leisure time, immediate consumption'
    })
    
    inventory.append({
        'ServiceName': 'ICT (Information & Communication)',
        'DeviceColumnName(s)': 'Router, Computer, Phone charger',
        'EnergyCarrier(s)': 'Electricity',
        'ShiftabilityClass': 'Non-shiftable',
        'WhatIsFixed': 'Always-on connectivity, user device usage',
        'WhatIsDecision': 'None (baseline)',
        'Constraints': 'Continuous or user-driven',
        'Notes': 'Router always-on, devices used as needed'
    })
    
    inventory.append({
        'ServiceName': 'Cold Storage (Refrigeration)',
        'DeviceColumnName(s)': 'Refrigerator, Freezer',
        'EnergyCarrier(s)': 'Electricity',
        'ShiftabilityClass': 'Non-shiftable',
        'WhatIsFixed': 'Temperature setpoint, thermal inertia',
        'WhatIsDecision': 'None (baseline, thermostat-controlled)',
        'Constraints': 'Must maintain food safety temperature',
        'Notes': 'Continuous cooling, temperature-controlled'
    })
    
    inventory.append({
        'ServiceName': 'Hygiene Water (Sink, Shower)',
        'DeviceColumnName(s)': 'Sink taps, Shower',
        'EnergyCarrier(s)': 'ColdWater, WarmWater, HotWater',
        'ShiftabilityClass': 'Non-shiftable',
        'WhatIsFixed': 'User hygiene schedule, comfort preferences',
        'WhatIsDecision': 'None (baseline)',
        'Constraints': 'User-driven, immediate need',
        'Notes': 'Personal hygiene cannot be deferred'
    })
    
    inventory.append({
        'ServiceName': 'Sanitation Water (Toilet)',
        'DeviceColumnName(s)': 'Toilet flush',
        'EnergyCarrier(s)': 'ColdWater',
        'ShiftabilityClass': 'Non-shiftable',
        'WhatIsFixed': 'User biological needs',
        'WhatIsDecision': 'None (baseline)',
        'Constraints': 'User-driven, immediate need',
        'Notes': 'Biological necessity, cannot be shifted'
    })
    
    inventory.append({
        'ServiceName': 'Mobility Fuel',
        'DeviceColumnName(s)': 'Car refueling',
        'EnergyCarrier(s)': 'Gasoline',
        'ShiftabilityClass': 'Non-shiftable',
        'WhatIsFixed': 'User travel schedule',
        'WhatIsDecision': 'None (baseline for gas vehicles)',
        'Constraints': 'Refueling as needed for travel',
        'Notes': 'Gasoline car, travel schedule driven'
    })
    
    # Hard-shiftable services
    inventory.append({
        'ServiceName': 'Dishwashing Service',
        'DeviceColumnName(s)': 'Dishwasher',
        'EnergyCarrier(s)': 'Electricity, ColdWater',
        'ShiftabilityClass': 'Hard-shiftable',
        'WhatIsFixed': 'Cycle energy profile, water consumption, duration once started',
        'WhatIsDecision': 'Start time (within acceptable window)',
        'Constraints': 'Must complete within 24h, user may prefer daytime completion',
        'Notes': 'Discrete task with fixed profile, only start time is flexible'
    })
    
    inventory.append({
        'ServiceName': 'Laundry Service',
        'DeviceColumnName(s)': 'Washing Machine',
        'EnergyCarrier(s)': 'Electricity, ColdWater',
        'ShiftabilityClass': 'Hard-shiftable',
        'WhatIsFixed': 'Cycle energy profile, water consumption, duration once started',
        'WhatIsDecision': 'Start time (within acceptable window)',
        'Constraints': 'Must complete within 24h, clothes should not sit wet too long',
        'Notes': 'Discrete task with fixed profile, only start time is flexible'
    })
    
    # Note: Soft-shiftable services (space heating, DHW tank, EV) would go here if present
    # For retired couple dataset, check if these exist
    
    return pd.DataFrame(inventory)

# ============================================================================
# MAIN PROCESSING
# ============================================================================

def main():
    print("="*80)
    print("HOUSEHOLD SERVICES ANALYZER - Retired Couple Dataset")
    print("="*80)
    print()
    
    # Load all CSV files
    print("STEP 1: Loading CSV files...")
    print("-" * 80)
    electricity_df = load_csv_robust(ELECTRICITY_CSV, "Electricity")
    coldwater_df = load_csv_robust(COLDWATER_CSV, "Cold Water")
    warmwater_df = load_csv_robust(WARMWATER_CSV, "Warm Water")
    hotwater_df = load_csv_robust(HOTWATER_CSV, "Hot Water")
    gasoline_df = load_csv_robust(GASOLINE_CSV, "Gasoline")
    print()
    
    # Create service inventory
    print("STEP 2: Creating Service Inventory...")
    print("-" * 80)
    inventory_df = create_service_inventory()
    print(f"✓ Created inventory with {len(inventory_df)} services")
    print()
    
    # Extract service events for hard-shiftable appliances
    print("STEP 3: Extracting Service Events...")
    print("-" * 80)
    
    all_events = []
    
    # Dishwasher events
    if electricity_df is not None:
        # Find dishwasher column
        dishwasher_col = None
        for col in electricity_df.columns:
            if 'dishwasher' in col.lower():
                dishwasher_col = col
                break
        
        if dishwasher_col:
            print(f"  Found dishwasher column: {dishwasher_col}")
            power_series = electricity_df[dishwasher_col]
            
            # Find dishwasher in cold water if available
            water_series = None
            if coldwater_df is not None:
                for col in coldwater_df.columns:
                    if 'dishwasher' in col.lower():
                        water_series = coldwater_df[col]
                        print(f"  Found dishwasher water column: {col}")
                        break
            
            dishwasher_events = detect_service_events(
                power_series, water_series, 'Dishwasher', electricity_df['datetime']
            )
            
            for event in dishwasher_events:
                event['service_name'] = 'Dishwashing Service'
                event['shiftability'] = 'Hard-shiftable'
                event['carriers'] = 'Electricity, ColdWater'
                all_events.append(event)
            
            print(f"  ✓ Detected {len(dishwasher_events)} dishwasher events")
        else:
            print("  ✗ No dishwasher column found")
    
    # Washing machine events
    if electricity_df is not None:
        # Find washing machine column
        wm_col = None
        for col in electricity_df.columns:
            if 'washing' in col.lower() or 'wasch' in col.lower():
                wm_col = col
                break
        
        if wm_col:
            print(f"  Found washing machine column: {wm_col}")
            power_series = electricity_df[wm_col]
            
            # Find washing machine in cold water if available
            water_series = None
            if coldwater_df is not None:
                for col in coldwater_df.columns:
                    if 'washing' in col.lower() or 'wasch' in col.lower():
                        water_series = coldwater_df[col]
                        print(f"  Found washing machine water column: {col}")
                        break
            
            wm_events = detect_service_events(
                power_series, water_series, 'Washing Machine', electricity_df['datetime']
            )
            
            for event in wm_events:
                event['service_name'] = 'Laundry Service'
                event['shiftability'] = 'Hard-shiftable'
                event['carriers'] = 'Electricity, ColdWater'
                all_events.append(event)
            
            print(f"  ✓ Detected {len(wm_events)} washing machine events")
        else:
            print("  ✗ No washing machine column found")
    
    events_df = pd.DataFrame(all_events)
    if len(events_df) > 0:
        events_df['household_id'] = 'CHS04_RetiredCouple'
        events_df = events_df[['household_id', 'service_name', 'device_name', 
                               'start_datetime', 'end_datetime', 'duration_min',
                               'electricity_kWh', 'coldwater_L', 'shiftability', 'carriers']]
    print()
    
    # Create non-shiftable baseline
    print("STEP 4: Creating Non-Shiftable Baseline...")
    print("-" * 80)
    
    if electricity_df is not None:
        # Sum all electricity columns EXCEPT dishwasher and washing machine
        shiftable_devices = []
        for col in electricity_df.columns:
            if 'dishwasher' in col.lower() or 'washing' in col.lower() or 'wasch' in col.lower():
                shiftable_devices.append(col)
        
        # Get all device columns (exclude datetime and time)
        device_cols = [c for c in electricity_df.columns 
                      if c not in ['datetime', 'Time', 'time'] and c not in shiftable_devices]
        
        # Filter for numeric columns only
        numeric_device_cols = []
        for col in device_cols:
            if pd.api.types.is_numeric_dtype(electricity_df[col]):
                numeric_device_cols.append(col)
        
        nonshiftable_df = pd.DataFrame({
            'datetime': electricity_df['datetime'],
            'NonShiftable_Electricity_W': electricity_df[numeric_device_cols].sum(axis=1)
        })
        
        print(f"  ✓ Created baseline from {len(device_cols)} non-shiftable devices")
        print(f"    Excluded shiftable: {shiftable_devices}")
    else:
        nonshiftable_df = None
        print("  ✗ No electricity data available")
    print()
    
    # Create summary statistics
    print("STEP 5: Creating Summary Statistics...")
    print("-" * 80)
    
    summary_data = []
    
    if len(events_df) > 0:
        for service in events_df['service_name'].unique():
            service_events = events_df[events_df['service_name'] == service]
            summary_data.append({
                'Service': service,
                'Total Events': len(service_events),
                'Mean Duration (min)': service_events['duration_min'].mean(),
                'Median Duration (min)': service_events['duration_min'].median(),
                'Total Electricity (kWh)': service_events['electricity_kWh'].sum(),
                'Total Cold Water (L)': service_events['coldwater_L'].sum()
            })
    
    if nonshiftable_df is not None:
        baseline_kwh = nonshiftable_df['NonShiftable_Electricity_W'].sum() / 1000 / 60
        summary_data.append({
            'Service': 'Non-Shiftable Baseline',
            'Total Events': '-',
            'Mean Duration (min)': '-',
            'Median Duration (min)': '-',
            'Total Electricity (kWh)': baseline_kwh,
            'Total Cold Water (L)': '-'
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(f"  ✓ Created summary with {len(summary_df)} entries")
    print()
    
    # Export to single CSV file - Service Events Only
    print("STEP 6: Exporting Service Events to CSV...")
    print("-" * 80)
    
    output_file = f"{OUTPUT_DIR}/{OUTPUT_PREFIX}_events.csv"
    
    if len(events_df) > 0:
        # Ensure columns are in exact order requested
        events_df = events_df[['household_id', 'service_name', 'device_name', 
                               'start_datetime', 'end_datetime', 'duration_min',
                               'electricity_kWh', 'coldwater_L', 'shiftability', 'carriers']]
        events_df.to_csv(output_file, index=False)
        print(f"  ✓ Service Events CSV created: {OUTPUT_PREFIX}_events.csv")
        print(f"    Rows: {len(events_df)}")
        print(f"    Columns: {', '.join(events_df.columns)}")
    else:
        # Create empty file with correct headers
        empty_df = pd.DataFrame(columns=['household_id', 'service_name', 'device_name', 
                                        'start_datetime', 'end_datetime', 'duration_min',
                                        'electricity_kWh', 'coldwater_L', 'shiftability', 'carriers'])
        empty_df.to_csv(output_file, index=False)
        print(f"  ⚠ No events detected, created empty CSV with headers")
    
    print()
    print("="*80)
    print(f"✓ COMPLETED! Output saved to:")
    print(f"  {output_file}")
    print("="*80)
    
    # Display summary
    print("\nSUMMARY STATISTICS:")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
