#!/usr/bin/env python3
"""
Separate device profiles by shiftability category.
Categories:
- Non-shiftable: Lighting, cooking, entertainment, IT, continuous cooling, personal care
- Hard shiftable: Dishwasher, washing machine, dryer, vacuum robot, garden tools
- Soft shiftable: Electric vehicle (not present in this dataset)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Define device categorization based on column name patterns
DEVICE_CATEGORIES = {
    'non_shiftable': {
        'patterns': [
            'Light', 'light',  # All lighting
            'TV', 'PlayStation', 'CD/DVD',  # Entertainment
            'SAT Receiver', 'Router', 'Server', 'Laptop',  # IT equipment
            'Kl 20 LA 65',  # Fridge/freezer (continuous cooling)
            'Stove', 'stove', 'Microwave', 'Kettle', 'Coffee Machine', 'Nespresso',  # Cooking
            'Toaster', 'Egg Cooker', 'Juicer', 'Blender', 'Hand Mixer', 'Handmixer',  # Food prep
            'Food Slicer', 'AFK BM-2N', 'Single Stove Plate',  # More cooking
            'Extractor Hood', 'Kitchen radio', 'AEG KRC',  # Kitchen equipment
            'Hair Dryer', 'Electric Razor', 'Electric Toothbrush',  # Personal care
            'Steam Iron', 'Humidifier', 'Air Conditioning',  # Comfort
            'Hometrainer', 'Treadmill', 'Christopeit',  # Exercise (non-deferrable)
            'Miele DG', 'Miele DA', 'Bauknecht GTE', 'Bauknecht Heko',  # Kitchen appliances
            'Clatronic', 'Grundig', 'Milk Foamer', 'Eye Glass Cleaner'  # Small appliances
        ],
        'devices': []
    },
    'hard_shiftable': {
        'patterns': [
            'Dishwasher',  # Hard shiftable
            'Washing Machine',  # Hard shiftable
            'Dryer',  # Hard shiftable
            'Vacuum Cleaner Robot', 'Roomba',  # Hard shiftable
            'Hedge Trimmer',  # Garden tool
            'Lawn Mower', 'Atika LH'  # Garden tool
        ],
        'devices': []
    },
    'soft_shiftable': {
        'patterns': [
            'Electric Vehicle', 'EV', 'Charger', 'Car'  # Not in this dataset
        ],
        'devices': []
    }
}


def categorize_column(col_name):
    """Categorize a column based on its name."""
    # Skip metadata columns
    if col_name in ['Electricity.Timestep', 'Time']:
        return 'metadata'
    
    # Check hard shiftable first (more specific)
    for pattern in DEVICE_CATEGORIES['hard_shiftable']['patterns']:
        if pattern in col_name:
            return 'hard_shiftable'
    
    # Check non-shiftable
    for pattern in DEVICE_CATEGORIES['non_shiftable']['patterns']:
        if pattern in col_name:
            return 'non_shiftable'
    
    # Check soft shiftable
    for pattern in DEVICE_CATEGORIES['soft_shiftable']['patterns']:
        if pattern in col_name:
            return 'soft_shiftable'
    
    # If no match, default to non-shiftable (conservative approach)
    print(f"Warning: No category match for '{col_name}', defaulting to non-shiftable")
    return 'non_shiftable'


def main():
    # Read the CSV file
    csv_path = Path('/Users/mariabigonah/Desktop/DeviceProfiles_Couple both at Work.Electricity.csv')
    print(f"Reading CSV file: {csv_path}")
    
    # Read with error handling for encoding
    try:
        df = pd.read_csv(csv_path, sep=';', encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, sep=';', encoding='latin-1')
    
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    
    # Categorize all columns
    categorized_cols = {
        'metadata': [],
        'non_shiftable': [],
        'hard_shiftable': [],
        'soft_shiftable': []
    }
    
    for col in df.columns:
        category = categorize_column(col)
        categorized_cols[category].append(col)
    
    # Print categorization summary
    print("\n" + "="*80)
    print("CATEGORIZATION SUMMARY")
    print("="*80)
    print(f"\nMetadata columns: {len(categorized_cols['metadata'])}")
    print(f"  {categorized_cols['metadata'][:5]}")  # Show first 5
    
    print(f"\nNon-shiftable devices: {len(categorized_cols['non_shiftable'])}")
    print(f"  Examples: {categorized_cols['non_shiftable'][:5]}")
    
    print(f"\nHard shiftable devices: {len(categorized_cols['hard_shiftable'])}")
    print(f"  All: {categorized_cols['hard_shiftable']}")
    
    print(f"\nSoft shiftable devices: {len(categorized_cols['soft_shiftable'])}")
    print(f"  All: {categorized_cols['soft_shiftable']}")
    
    # Create separate dataframes
    output_dir = Path('/Users/mariabigonah/Desktop/thesis/anti/pv')
    
    # Save metadata + non-shiftable
    non_shift_cols = categorized_cols['metadata'] + categorized_cols['non_shiftable']
    df_non_shift = df[non_shift_cols].copy()
    output_non = output_dir / 'load_profile_non_shiftable.csv'
    df_non_shift.to_csv(output_non, index=False)
    print(f"\nSaved non-shiftable loads to: {output_non}")
    print(f"  Shape: {df_non_shift.shape}")
    
    # Save metadata + hard shiftable
    hard_shift_cols = categorized_cols['metadata'] + categorized_cols['hard_shiftable']
    df_hard_shift = df[hard_shift_cols].copy()
    output_hard = output_dir / 'load_profile_hard_shiftable.csv'
    df_hard_shift.to_csv(output_hard, index=False)
    print(f"\nSaved hard shiftable loads to: {output_hard}")
    print(f"  Shape: {df_hard_shift.shape}")
    
    # Create aggregated load profiles
    print("\n" + "="*80)
    print("CREATING AGGREGATED LOAD PROFILES")
    print("="*80)
    
    # Aggregate non-shiftable (sum across all non-shiftable devices per timestep)
    df_agg = df[categorized_cols['metadata']].copy()
    df_agg['NonShiftable_Total_kWh'] = df[categorized_cols['non_shiftable']].sum(axis=1)
    df_agg['HardShiftable_Total_kWh'] = df[categorized_cols['hard_shiftable']].sum(axis=1)
    df_agg['Total_Load_kWh'] = df_agg['NonShiftable_Total_kWh'] + df_agg['HardShiftable_Total_kWh']
    
    output_agg = output_dir / 'load_profile_aggregated.csv'
    df_agg.to_csv(output_agg, index=False)
    print(f"\nSaved aggregated load profile to: {output_agg}")
    print(f"  Shape: {df_agg.shape}")
    
    # Print statistics
    print("\n" + "="*80)
    print("LOAD STATISTICS")
    print("="*80)
    print(f"\nNon-shiftable load:")
    print(f"  Total energy: {df_agg['NonShiftable_Total_kWh'].sum():.2f} kWh")
    print(f"  Mean power: {df_agg['NonShiftable_Total_kWh'].mean():.4f} kWh/timestep")
    print(f"  Peak power: {df_agg['NonShiftable_Total_kWh'].max():.4f} kWh/timestep")
    
    print(f"\nHard shiftable load:")
    print(f"  Total energy: {df_agg['HardShiftable_Total_kWh'].sum():.2f} kWh")
    print(f"  Mean power: {df_agg['HardShiftable_Total_kWh'].mean():.4f} kWh/timestep")
    print(f"  Peak power: {df_agg['HardShiftable_Total_kWh'].max():.4f} kWh/timestep")
    
    print(f"\nTotal household load:")
    print(f"  Total energy: {df_agg['Total_Load_kWh'].sum():.2f} kWh")
    print(f"  Mean power: {df_agg['Total_Load_kWh'].mean():.4f} kWh/timestep")
    print(f"  Peak power: {df_agg['Total_Load_kWh'].max():.4f} kWh/timestep")
    
    # Calculate shiftability percentage
    non_shift_pct = (df_agg['NonShiftable_Total_kWh'].sum() / df_agg['Total_Load_kWh'].sum()) * 100
    hard_shift_pct = (df_agg['HardShiftable_Total_kWh'].sum() / df_agg['Total_Load_kWh'].sum()) * 100
    
    print(f"\nShiftability breakdown:")
    print(f"  Non-shiftable: {non_shift_pct:.1f}%")
    print(f"  Hard shiftable: {hard_shift_pct:.1f}%")
    print(f"  Soft shiftable: 0.0% (no EV in dataset)")
    
    # Save categorization mapping
    mapping_file = output_dir / 'device_category_mapping.txt'
    with open(mapping_file, 'w') as f:
        f.write("DEVICE CATEGORIZATION MAPPING\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"NON-SHIFTABLE DEVICES ({len(categorized_cols['non_shiftable'])} devices):\n")
        f.write("-"*80 + "\n")
        for col in sorted(categorized_cols['non_shiftable']):
            f.write(f"  - {col}\n")
        
        f.write(f"\n\nHARD SHIFTABLE DEVICES ({len(categorized_cols['hard_shiftable'])} devices):\n")
        f.write("-"*80 + "\n")
        for col in sorted(categorized_cols['hard_shiftable']):
            f.write(f"  - {col}\n")
        
        f.write(f"\n\nSOFT SHIFTABLE DEVICES ({len(categorized_cols['soft_shiftable'])} devices):\n")
        f.write("-"*80 + "\n")
        if categorized_cols['soft_shiftable']:
            for col in sorted(categorized_cols['soft_shiftable']):
                f.write(f"  - {col}\n")
        else:
            f.write("  - None (no EV in this dataset)\n")
    
    print(f"\nSaved device categorization mapping to: {mapping_file}")
    print("\n" + "="*80)
    print("PROCESSING COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
