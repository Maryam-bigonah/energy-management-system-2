"""
Scenario 0 - Baseline Cost Calculation
Complete implementation for thesis baseline scenario

Processes:
1. Load 4 family types (2024, 10-min resolution)
2. Resample to hourly, shift to 2025
3. Apply TOU pricing (ARERA F1/F2/F3)
4. Calculate baseline costs
5. Generate thesis-ready outputs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# Configuration
FAMILY_TYPES = {
    'A': 'young_couple_load_profile.csv',
    'B': 'retired_couple_load_profile.csv',
    'C': 'family_with_children_load_profile.csv',
    'D': 'large_family_load_profile.csv'
}

PRICE_FILE = 'arera_fixed_prices_2025.csv'
OUTPUT_DIR = 'scenario0_results'
UNITS_PER_FAMILY = 5  # 5 apartments per family type
TOTAL_APARTMENTS = 20

def load_and_preprocess_profiles():
    """Step 1: Load and preprocess all family profiles"""
    
    print("="*70)
    print("SCENARIO 0 - BASELINE DATA PREPROCESSING")
    print("="*70)
    
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    profiles = {}
    
    for family_id, filename in FAMILY_TYPES.items():
        print(f"\n### Processing Family Type {family_id} ###")
        print(f"Loading: {filename}")
        
        # Load 10-minute data from 2024
        df = pd.read_csv(filename, parse_dates=['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"  Original resolution: {len(df)} records")
        print(f"  Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Get service columns (exclude timestamp and total)
        service_cols = [c for c in df.columns if c not in ['timestamp', 'total_power_W']]
        print(f"  Services: {len(service_cols)}")
        
        # SANITY CHECK: Compare total_power_W with sum of services
        if 'total_power_W' in df.columns:
            computed_total = df[service_cols].sum(axis=1)
            diff = np.abs(df['total_power_W'] - computed_total)
            max_diff = diff.max()
            if max_diff > 1.0:  # Allow 1W tolerance
                print(f"  ⚠ WARNING: Max difference between total_power_W and sum: {max_diff:.2f} W")
            else:
                print(f"  ✓ Sanity check passed: total matches sum of services")
        
        # Resample to hourly (sum power over hour, then divide by 6 for average, then convert to kWh)
        # For 10-min data: 6 samples/hour
        # Energy (kWh) = Average Power (kW) × time (h)
        # Average Power (kW) = Sum of 10-min power (W) / 6 samples / 1000
        df.set_index('timestamp', inplace=True)
        
        # Resample: sum gives total W across all samples
        # Then divide by 6 to get avg W, divide by 1000 for kW, multiply by 1h for kWh
        hourly_2024 = df[service_cols].resample('1h').sum() / 6 / 1000  # Now in kWh
        
        # Also compute total if not doing sanity check
        hourly_2024['total_kwh'] = hourly_2024.sum(axis=1)
        
        print(f"  Resampled to: {len(hourly_2024)} hourly records (2024 - leap year)")
        
        # Create 2025 full index (8760 hours, non-leap year)
        full_index_2025 = pd.date_range('2025-01-01', '2025-12-31 23:00', freq='1h')
        
        # Map 2024 data to 2025 by matching day-of-year and hour
        # For 2024 leap year (366 days), we skip Feb 29 data when mapping to 2025
        hourly_2024_reset = hourly_2024.reset_index()
        hourly_2024_reset['doy'] = hourly_2024_reset['timestamp'].dt.dayofyear
        hourly_2024_reset['hour'] = hourly_2024_reset['timestamp'].dt.hour
        
        # For 2025 index
        mapping_df = pd.DataFrame({'timestamp_2025': full_index_2025})
        mapping_df['doy'] = mapping_df['timestamp_2025'].dt.dayofyear
        mapping_df['hour'] = mapping_df['timestamp_2025'].dt.hour
        
        # Merge (skip Feb 29 from 2024 if it exists)
        hourly = mapping_df.merge(
            hourly_2024_reset.drop(columns=['timestamp']),
            on=['doy', 'hour'],
            how='left'
        )
        
        # Fill any NaN with 0 (shouldn't happen but safety)
        hourly.fillna(0, inplace=True)
        hourly.set_index('timestamp_2025', inplace=True)
        hourly = hourly.drop(columns=['doy', 'hour'])
        
        print(f"  Final: {len(hourly)} hours in 2025")
        print(f"  Annual energy: {hourly['total_kwh'].sum():.1f} kWh")
        
        hourly.reset_index(inplace=True)
        hourly.rename(columns={'timestamp_2025': 'timestamp'}, inplace=True)
        
        profiles[family_id] = hourly
    
    return profiles

def load_prices():
    """Step 2: Load TOU prices"""
    
    print("\n" + "="*70)
    print("LOADING TOU PRICES")
    print("="*70)
    
    prices = pd.read_csv(PRICE_FILE, parse_dates=['timestamp'])
    
    print(f"Price records: {len(prices)}")
    print(f"Period: {prices['timestamp'].min()} to {prices['timestamp'].max()}")
    
    # Show price distribution by band
    band_summary = prices.groupby('arera_band')['price_eur_per_kwh'].agg(['first', 'count'])
    band_summary['hours'] = band_summary['count']
    band_summary['price_€/kWh'] = band_summary['first']
    
    print("\nTOU Band Summary:")
    print(band_summary[['hours', 'price_€/kWh']])
    
    # Verify 8760 hours
    if len(prices) != 8760:
        print(f"⚠ WARNING: Expected 8760 hours, got {len(prices)}")
    
    return prices

def calculate_baseline_costs(profiles, prices):
    """Step 3: Calculate baseline costs for each family type"""
    
    print("\n" + "="*70)
    print("BASELINE COST CALCULATION")
    print("="*70)
    
    results = {}
    
    for family_id, hourly_df in profiles.items():
        print(f"\n### Family Type {family_id} ###")
        
        # Merge with prices
        merged = hourly_df.merge(prices[['timestamp', 'arera_band', 'price_eur_per_kwh']], 
                                 on='timestamp', how='left')
        
        # Calculate hourly cost: c_h,t = p(t) × E_h,t
        merged['hourly_cost_€'] = merged['price_eur_per_kwh'] * merged['total_kwh']
        
        # Annual totals
        annual_kwh = merged['total_kwh'].sum()
        annual_cost = merged['hourly_cost_€'].sum()
        
        print(f"  Annual energy: {annual_kwh:.1f} kWh")
        print(f"  Annual cost (1 unit): {annual_cost:.2f} €")
        print(f"  Average price: {annual_cost/annual_kwh:.6f} €/kWh")
        
        # Breakdown by TOU band
        band_breakdown = merged.groupby('arera_band').agg({
            'total_kwh': 'sum',
            'hourly_cost_€': 'sum'
        }).reset_index()
        band_breakdown.columns = ['Band', 'kWh', 'Cost_€']
        band_breakdown['Avg_Price_€/kWh'] = band_breakdown['Cost_€'] / band_breakdown['kWh']
        
        print("\n  By TOU Band:")
        print(band_breakdown.to_string(index=False))
        
        results[family_id] = {
            'merged_data': merged,
            'annual_kwh': annual_kwh,
            'annual_cost_€': annual_cost,
            'band_breakdown': band_breakdown
        }
    
    return results

def calculate_building_totals(results):
    """Step 4: Calculate building-wide totals"""
    
    print("\n" + "="*70)
    print("BUILDING-WIDE TOTALS (20 apartments)")
    print("="*70)
    
    building_totals = {
        'by_family': {},
        'total_kwh': 0,
        'total_cost_€': 0
    }
    
    for family_id in FAMILY_TYPES.keys():
        per_unit_kwh = results[family_id]['annual_kwh']
        per_unit_cost = results[family_id]['annual_cost_€']
        
        total_kwh = per_unit_kwh * UNITS_PER_FAMILY
        total_cost = per_unit_cost * UNITS_PER_FAMILY
        
        building_totals['by_family'][family_id] = {
            'units': UNITS_PER_FAMILY,
            'kwh_per_unit': per_unit_kwh,
            'cost_per_unit_€': per_unit_cost,
            'total_kwh': total_kwh,
            'total_cost_€': total_cost
        }
        
        building_totals['total_kwh'] += total_kwh
        building_totals['total_cost_€'] += total_cost
        
        print(f"\nFamily {family_id} ({UNITS_PER_FAMILY} units):")
        print(f"  Total energy: {total_kwh:,.0f} kWh")
        print(f"  Total cost: {total_cost:,.2f} €")
    
    print(f"\n{'='*70}")
    print(f"BUILDING TOTAL ({TOTAL_APARTMENTS} apartments):")
    print(f"  Total energy: {building_totals['total_kwh']:,.0f} kWh")
    print(f"  Total cost: {building_totals['total_cost_€']:,.2f} €")
    print(f"  Average cost per apartment: {building_totals['total_cost_€']/TOTAL_APARTMENTS:,.2f} €")
    print(f"{'='*70}")
    
    return building_totals

def save_results_tables(results, building_totals):
    """Step 5: Save results to CSV"""
    
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    # Table 1: Per-family summary
    family_summary = []
    for family_id in FAMILY_TYPES.keys():
        row = {
            'Family_Type': family_id,
            'Annual_kWh_per_unit': results[family_id]['annual_kwh'],
            'Annual_Cost_€_per_unit': results[family_id]['annual_cost_€'],
            'Units': UNITS_PER_FAMILY,
            'Total_kWh': building_totals['by_family'][family_id]['total_kwh'],
            'Total_Cost_€': building_totals['by_family'][family_id]['total_cost_€']
        }
        family_summary.append(row)
    
    df_family = pd.DataFrame(family_summary)
    df_family.to_csv(f'{OUTPUT_DIR}/scenario0_family_summary.csv', index=False)
    print(f"  ✓ Saved: {OUTPUT_DIR}/scenario0_family_summary.csv")
    
    # Table 2: By TOU band (all families combined)
    all_bands = []
    for family_id in FAMILY_TYPES.keys():
        band_df = results[family_id]['band_breakdown'].copy()
        band_df['Family'] = family_id
        band_df['kWh_total'] = band_df['kWh'] * UNITS_PER_FAMILY
        band_df['Cost_€_total'] = band_df['Cost_€'] * UNITS_PER_FAMILY
        all_bands.append(band_df)
    
    df_bands = pd.concat(all_bands, ignore_index=True)
    df_bands.to_csv(f'{OUTPUT_DIR}/scenario0_band_breakdown.csv', index=False)
    print(f"  ✓ Saved: {OUTPUT_DIR}/scenario0_band_breakdown.csv")
    
    # Table 3: Building total
    building_df = pd.DataFrame([{
        'Total_Apartments': TOTAL_APARTMENTS,
        'Total_Annual_kWh': building_totals['total_kwh'],
        'Total_Annual_Cost_€': building_totals['total_cost_€'],
        'Avg_Cost_per_Apartment_€': building_totals['total_cost_€'] / TOTAL_APARTMENTS
    }])
    building_df.to_csv(f'{OUTPUT_DIR}/scenario0_building_total.csv', index=False)
    print(f"  ✓ Saved: {OUTPUT_DIR}/scenario0_building_total.csv")

def create_validation_plot(results, prices):
    """Step 6: Create one-week validation plot"""
    
    print("\n" + "="*70)
    print("CREATING VALIDATION PLOT")
    print("="*70)
    
    # Use Family A for visualization
    week_data = results['A']['merged_data'].head(7*24).copy()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    
    # Plot 1: Demand and Price
    ax1_price = ax1.twinx()
    
    ax1.plot(week_data['timestamp'], week_data['total_kwh'], 
            linewidth=2, color='steelblue', label='Power Demand')
    ax1.fill_between(week_data['timestamp'], 0, week_data['total_kwh'], 
                     alpha=0.3, color='steelblue')
    ax1.set_ylabel('Energy (kWh/hour)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    ax1_price.plot(week_data['timestamp'], week_data['price_eur_per_kwh'], 
                   linewidth=2, color='red', label='Price', linestyle='--')
    ax1_price.set_ylabel('Price (€/kWh)', fontsize=12, fontweight='bold', color='red')
    ax1_price.tick_params(axis='y', labelcolor='red')
    ax1_price.legend(loc='upper right')
    
    ax1.set_title('Scenario 0 Baseline - One Week (Family A)', 
                 fontsize=14, fontweight='bold')
    
    # Plot 2: TOU Bands
    band_colors = {'F1': '#e74c3c', 'F2': '#f39c12', 'F3': '#27ae60'}
    for band in ['F1', 'F2', 'F3']:
        band_mask = week_data['arera_band'] == band
        ax2.scatter(week_data.loc[band_mask, 'timestamp'], 
                   [band]*band_mask.sum(),
                   c=band_colors[band], label=band, s=30, alpha=0.8)
    
    ax2.set_ylabel('TOU Band', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date/Time', fontsize=12, fontweight='bold')
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['F3 (off-peak)', 'F2 (mid)', 'F1 (peak)'])
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/scenario0_validation_week.png', dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved: {OUTPUT_DIR}/scenario0_validation_week.png")
    plt.close()

def main():
    """Main execution"""
    
    # Step 1: Load and preprocess
    profiles = load_and_preprocess_profiles()
    
    # Step 2: Load prices
    prices = load_prices()
    
    # Step 3: Calculate costs
    results = calculate_baseline_costs(profiles, prices)
    
    # Step 4: Building totals
    building_totals = calculate_building_totals(results)
    
    # Step 5: Save tables
    save_results_tables(results, building_totals)
    
    # Step 6: Validation plot
    create_validation_plot(results, prices)
    
    print("\n" + "="*70)
    print("✓ SCENARIO 0 BASELINE ANALYSIS COMPLETE")
    print(f"✓ All results saved to: {OUTPUT_DIR}/")
    print("="*70)

if __name__ == '__main__':
    main()
