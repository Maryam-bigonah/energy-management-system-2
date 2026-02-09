"""
Export module for Professor's Dataset Layers
Handles export of shiftable tasks and thermal service demands.
"""

import pandas as pd
import csv
from pathlib import Path
from typing import List, Dict
from datetime import timedelta

def export_shiftable_tasks(
    activations: List[Dict], 
    household_id: str, 
    year: int, 
    output_dir: str
):
    """
    Export list of activations as shiftable tasks CSV
    
    Args:
        activations: List of activation dicts
        household_id: Household ID
        year: Year of simulation
        output_dir: Output directory
    """
    if not activations:
        print(f"No shiftable tasks to export for {household_id}")
        return

    output_path = Path(output_dir) / f"tasks_{household_id}_{year}.csv"
    
    # Add calculated fields for constraints
    export_rows = []
    for task in activations:
        start = task['start_time']
        duration = task['duration_min']
        max_delay = task['max_delay_min']
        
        # Calculate Earliest Start and Latest End
        # Earliest Start: Current time (or earlier if pre-planning allowed, but let's stick to arrival)
        earliest_start = start
        
        # Latest End: Start + Duration + Max Delay
        latest_end = start + timedelta(minutes=duration + max_delay)
        
        row = {
            'task_id': task['task_id'],
            'household_id': household_id,
            'device_id': task['device_id'],
            'energy_kwh': round(task['energy_kwh'], 4),
            'duration_min': round(duration, 1),
            'start_time': start,
            'end_time': start + timedelta(minutes=duration),
            'earliest_start': earliest_start,
            'latest_end': latest_end,
            'max_delay_min': max_delay,
            'comfort_class': task['comfort_class']
        }
        export_rows.append(row)
        
    # Convert to DataFrame for easy CSV write
    df = pd.DataFrame(export_rows)
    df.to_csv(output_path, index=False)
    print(f"✓ Exported {len(df)} shiftable tasks for {household_id} to {output_path}")


def export_thermal_services(
    df: pd.DataFrame,
    devices_config: Dict,
    household_id: str,
    year: int,
    output_dir: str,
    interval_minutes: int
):
    """
    Export technology-agnostic service demands
    
    Args:
        df: Load profile DataFrame
        devices_config: Device configurations
        household_id: Household ID
        year: Year
        output_dir: Output directory
        interval_minutes: Resolution
    """
    output_path = Path(output_dir) / f"services_{household_id}_{year}.csv"
    
    # 1. Identify Service Columns
    service_cols = {
        'space_heating': [],
        'water_heating': [],
        'cooking': [],
        'ev_charging': []
    }
    
    # Map DataFrame columns back to devices
    col_to_device = {}
    for device_id, cfg in devices_config.items():
        dev_name = cfg.name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("/", "_")
        col_name = f"{dev_name}_W"
        col_to_device[col_name] = device_id
        
        # Categorize by service_type
        stype = getattr(cfg, 'service_type', 'electrical')
        if stype in service_cols:
            service_cols[stype].append(col_name)
    
    # 2. Calculate Service Demands (kWth or kWh equivalent)
    # Default assumptions (can be refined with equipment_catalog later)
    # Heat Pump COP ~3.0 (seasonal avg), Resistance ~1.0
    
    # Create new DataFrame
    services_df = pd.DataFrame(index=df.index)
    
    # Electrical Non-Shiftable (Base Load)
    # Start with Total and subtract known services
    # OR sum known non-service columns. Let's sum non-service columns for accuracy.
    electrical_cols = [c for c in df.columns if c.endswith('_W') and c != 'total_power_W']
    for stype, cols in service_cols.items():
        for c in cols:
            if c in electrical_cols:
                electrical_cols.remove(c)
    
    services_df['electric_base_load_W'] = df[electrical_cols].sum(axis=1)
    
    # Space Heating (kWth)
    # Assume Heat Pumps have COP=3.0, Boilers=1.0 (Electric)
    space_heat_joules = 0
    for col in service_cols['space_heating']:
        if col in df.columns:
            dev_id = col_to_device[col]
            # Simple COP heuristic for now (Heat Pump vs Resistance)
            is_hp = 'heat_pump' in dev_id
            cop = 3.0 if is_hp else 1.0
            space_heat_joules += df[col] * cop
            
    services_df['space_heating_kWth'] = space_heat_joules / 1000.0
    
    # Water Heating (kWth)
    water_heat_joules = 0
    for col in service_cols['water_heating']:
        if col in df.columns:
            # Water heaters usually resistance (COP=1) unless hybrid
            water_heat_joules += df[col] * 1.0
    services_df['water_heating_kWth'] = water_heat_joules / 1000.0
    
    # Cooking (kWth equivalent)
    cooking_joules = 0
    for col in service_cols['cooking']:
        if col in df.columns:
            cooking_joules += df[col] * 1.0
    services_df['cooking_kWth'] = cooking_joules / 1000.0
    
    # EV Charging (kWh delivered to battery)
    ev_joules = 0
    for col in service_cols['ev_charging']:
        if col in df.columns:
            ev_joules += df[col] * 0.9  # 90% efficiency
    services_df['ev_charging_kWh'] = (ev_joules / 1000.0) * (interval_minutes / 60.0)
    
    services_df.to_csv(output_path)
    print(f"✓ Exported thermal services for {household_id} to {output_path}")
