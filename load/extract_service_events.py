#!/usr/bin/env python3
"""
Convert shiftable appliance load profiles into discrete service events.

This script detects ON/OFF transitions in appliance power consumption to identify
individual service events (e.g., one dishwasher cycle, one washing machine run).

Output: services_events.csv with columns:
- event_id
- device
- date
- start_time_original
- end_time_original
- duration_hours
- energy_kWh
- preferred_tariff
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


# Device name patterns to extract (shiftable appliances only)
SHIFTABLE_DEVICES = {
    'Dishwasher': 'Dishwasher',
    'Washing Machine': 'Washing Machine',
    'Dryer': 'Dryer',
    'Vacuum Cleaner Robot': 'Roomba',  # Pattern to match
}

# Energy threshold to determine ON state (kWh per timestep)
ENERGY_THRESHOLD = 0.01  # 10 Wh minimum to avoid noise

# Minimum energy per complete event to filter out standby/noise
MIN_EVENT_ENERGY = {
    'Dishwasher': 0.5,          # Typical: 0.8-1.2 kWh
    'Washing Machine': 0.4,      # Typical: 0.6-1.0 kWh
    'Dryer': 1.0,                # Typical: 1.5-2.5 kWh (dryers are high energy!)
    'Vacuum Cleaner Robot': 0.05 # Small device, but real cycles are ~0.08+ kWh
}


def merge_duplicate_columns(df, pattern):
    """
    Merge duplicate columns with same device name (e.g., 'Device [kWh]', 'Device [kWh].1').
    
    Args:
        df: DataFrame with device columns
        pattern: Text pattern to match column names
    
    Returns:
        Series: Merged data (sum of all duplicate columns)
    """
    # Find all columns containing the pattern
    matching_cols = [col for col in df.columns if pattern in col]
    
    if not matching_cols:
        return None
    
    # Sum all matching columns
    merged = df[matching_cols].sum(axis=1)
    
    return merged


def detect_service_events(time_series, device_name, energy_threshold=ENERGY_THRESHOLD):
    """
    Detect service events from a time series of energy consumption.
    
    An event is a continuous period where energy > threshold.
    
    Args:
        time_series: DataFrame with 'Time' and 'Energy' columns
        device_name: Name of the device
        energy_threshold: Minimum energy to consider device ON
    
    Returns:
        List of event dictionaries
    """
    events = []
    event_id = 1
    in_event = False
    event_start_idx = None
    event_energy = 0.0
    
    # Get minimum event energy for this device
    min_event_energy = MIN_EVENT_ENERGY.get(device_name, 0.1)
    
    for idx, row in time_series.iterrows():
        is_on = row['Energy'] > energy_threshold
        
        if is_on and not in_event:
            # Event starts
            in_event = True
            event_start_idx = idx
            event_energy = row['Energy']
            
        elif is_on and in_event:
            # Event continues
            event_energy += row['Energy']
            
        elif not is_on and in_event:
            # Event ends
            in_event = False
            
            # Get start and end times
            start_time = time_series.loc[event_start_idx, 'Time']
            end_time = row['Time']
            
            # Calculate duration in hours
            # Assuming hourly data, duration = number of timesteps
            num_timesteps = idx - event_start_idx
            duration_hours = num_timesteps  # For hourly data
            
            # Only record events with reasonable duration and energy (device-specific filter)
            if duration_hours > 0 and event_energy >= min_event_energy:
                events.append({
                    'event_id': f"{device_name.replace(' ', '')}_{event_id:03d}",
                    'device': device_name,
                    'date': start_time.date(),
                    'start_time_original': start_time,
                    'end_time_original': end_time,
                    'duration_hours': duration_hours,
                    'energy_kWh': round(event_energy, 4),
                    'target_tariff': 'F3'  # Target for optimization, not actual tariff
                })
                event_id += 1
            
            event_energy = 0.0
    
    # Handle case where event is still ongoing at end of data
    if in_event and event_start_idx is not None:
        start_time = time_series.loc[event_start_idx, 'Time']
        end_time = time_series.iloc[-1]['Time']
        num_timesteps = len(time_series) - 1 - event_start_idx
        duration_hours = num_timesteps
        
        if duration_hours > 0 and event_energy >= min_event_energy:
            events.append({
                'event_id': f"{device_name.replace(' ', '')}_{event_id:03d}",
                'device': device_name,
                'date': start_time.date(),
                'start_time_original': start_time,
                'end_time_original': end_time,
                'duration_hours': duration_hours,
                'energy_kWh': round(event_energy, 4),
                'target_tariff': 'F3'  # Target for optimization, not actual tariff
            })
    
    return events


def main():
    # Read the CSV file
    csv_path = Path('/Users/mariabigonah/Desktop/DeviceProfiles_Couple both at Work.Electricity.csv')
    print(f"Reading CSV file: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path, sep=';', encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, sep=';', encoding='latin-1')
    
    # Parse time column
    df['Time'] = pd.to_datetime(df['Time'], format='%m/%d/%Y %H:%M', errors='coerce')
    
    print(f"Total rows: {len(df)}")
    print(f"Date range: {df['Time'].min()} to {df['Time'].max()}")
    
    # Extract and merge data for each shiftable device
    all_events = []
    
    print("\n" + "="*80)
    print("EXTRACTING SERVICE EVENTS")
    print("="*80)
    
    for device_name, pattern in SHIFTABLE_DEVICES.items():
        print(f"\n{device_name} ({pattern}):")
        print("-" * 40)
        
        # Merge duplicate columns
        merged_series = merge_duplicate_columns(df, pattern)
        
        if merged_series is None or merged_series.sum() == 0:
            print(f"  ⚠️  No data found for {device_name}")
            continue
        
        # Create time series DataFrame
        time_series = pd.DataFrame({
            'Time': df['Time'],
            'Energy': merged_series
        })
        
        # Detect events
        events = detect_service_events(time_series, device_name)
        
        print(f"  ✓ Found {len(events)} events")
        print(f"  Total energy: {sum(e['energy_kWh'] for e in events):.2f} kWh")
        if events:
            avg_duration = np.mean([e['duration_hours'] for e in events])
            avg_energy = np.mean([e['energy_kWh'] for e in events])
            print(f"  Avg duration: {avg_duration:.1f} hours")
            print(f"  Avg energy per event: {avg_energy:.2f} kWh")
        
        all_events.extend(events)
    
    # Create DataFrame from all events
    if not all_events:
        print("\n⚠️  No service events detected!")
        return
    
    events_df = pd.DataFrame(all_events)
    
    # Sort by start time
    events_df = events_df.sort_values('start_time_original').reset_index(drop=True)
    
    # Save to CSV
    output_path = Path('/Users/mariabigonah/Desktop/thesis/anti/pv/services_events.csv')
    events_df.to_csv(output_path, index=False)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nTotal service events detected: {len(events_df)}")
    print(f"Output saved to: {output_path}")
    
    print("\nEvents per device:")
    for device in events_df['device'].unique():
        device_events = events_df[events_df['device'] == device]
        print(f"  {device}: {len(device_events)} events, {device_events['energy_kWh'].sum():.2f} kWh total")
    
    print("\nEnergy per device (for validation):")
    for device in events_df['device'].unique():
        device_events = events_df[events_df['device'] == device]
        print(f"  {device}:")
        print(f"    Min energy: {device_events['energy_kWh'].min():.2f} kWh")
        print(f"    Max energy: {device_events['energy_kWh'].max():.2f} kWh")
        print(f"    Median energy: {device_events['energy_kWh'].median():.2f} kWh")
        print(f"    Events/week: {len(device_events) / 52:.1f}")
    
    print("\nSample events (first 10):")
    print(events_df[['event_id', 'device', 'start_time_original', 'duration_hours', 'energy_kWh']].head(10).to_string(index=False))
    
    print("\n" + "="*80)
    print("VALIDATION CHECKS")
    print("="*80)
    
    # Check for suspiciously low-energy events
    suspicious = events_df[events_df['energy_kWh'] < 0.3]
    if len(suspicious) > 0:
        print(f"\n⚠️  Warning: {len(suspicious)} events with energy < 0.3 kWh detected")
        print("    These may be noise or partial cycles. Consider reviewing thresholds.")
    else:
        print("\n✅ All events meet minimum energy thresholds")
    
    # Check daily frequency
    events_df['date_only'] = pd.to_datetime(events_df['date'])
    events_per_day = events_df.groupby(['date_only', 'device']).size()
    max_per_day = events_per_day.groupby('device').max()
    print(f"\nMax events per day per device:")
    for device, count in max_per_day.items():
        status = "✅" if count <= 3 else "⚠️"
        print(f"  {status} {device}: {count}")
    
    print("\n" + "="*80)
    print("PROCESSING COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
