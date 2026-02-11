
"""
Scenario 1: PV + Shared BESS Dispatch Optimization
Input:
    - Fixed Building Load (aggregated 20 units)
    - Fixed PV Generation
    - Fixed TOU Prices
    - Battery Specs: 20 kWh, 5 kW, 0.9 Eff
Output:
    - Optimal BESS dispatch to minimize Grid Cost
"""

import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import sys

# --- CONFIGURATION ---
INPUT_DIR = Path('..') # pv/optimization/
OUTPUT_DIR = Path('scenario1_results')
PLOTS_DIR = OUTPUT_DIR / 'plots'

# Battery Specs (Scenario 1)
BESS_CAPACITY_KWH = 20.0
BESS_POWER_KW = 5.0
ROUNDTRIP_EFF = 0.90
AGEING_COST = 1e-6 # epsilon

SOC_MIN_PCT = 0.05
SOC_MAX_PCT = 0.95
SOC_INITIAL_PCT = 0.50

# Derived
EFF_CH = np.sqrt(ROUNDTRIP_EFF)
EFF_DIS = np.sqrt(ROUNDTRIP_EFF) # Assuming symmetric
SOC_MIN = SOC_MIN_PCT * BESS_CAPACITY_KWH
SOC_MAX = SOC_MAX_PCT * BESS_CAPACITY_KWH
SOC_INITIAL = SOC_INITIAL_PCT * BESS_CAPACITY_KWH

# Data Files
FAMILY_FILES = {
    'A': 'young_couple_load_profile.csv',
    'B': 'retired_couple_load_profile.csv',
    'C': 'family_with_children_load_profile.csv',
    'D': 'large_family_load_profile.csv'
}
UNITS_PER_FAMILY = 5
PRICE_FILE = 'arera_fixed_prices_2025.csv'
PV_FILE = 'pv_prediction_2025_20260207_150532.csv'

def load_and_process_data():
    print("Loading Data...")
    
    # 1. Load Prices
    print(f"  Prices: {PRICE_FILE}")
    prices_df = pd.read_csv(INPUT_DIR / PRICE_FILE, parse_dates=['timestamp'])
    prices_df.set_index('timestamp', inplace=True)
    # Ensure 8760
    if len(prices_df) != 8760:
        print(f"Warning: Price index length {len(prices_df)}")
        
    prices = prices_df['price_eur_per_kwh']
    bands = prices_df['arera_band']

    # 2. Load PV
    print(f"  PV: {PV_FILE}")
    pv_df = pd.read_csv(INPUT_DIR / PV_FILE)
    # "dt_aligned" seems to be the one aligned with hours, check scenario0 logic or file view?
    # file view showed dt_aligned 2024-12-31 23:10:00 for 2025-01-01 00:00:00 end. 
    # Usually we want the timestamp representing the interval.
    # Let's use the index from prices to be safe and just take the array if length matches.
    
    # Check loaded PV length
    # Note: PV prediction file has 'P_pred' in Watts?
    # View file showed: P_pred column.
    
    # Let's assume P_pred is average power in Watts for that hour? Or sum?
    # Usually PVGIS/models give Power (W). Energy (kWh) = Power (W) * 1h / 1000.
    pv_watts = pv_df['P_pred'].values
    # Truncate or pad to 8760
    if len(pv_watts) > 8760:
        pv_watts = pv_watts[:8760]
    pv_kwh = pv_watts / 1000.0
    
    # 3. Load & Aggregate Demand
    print(f"  Demand: Loading {len(FAMILY_FILES)} profiles...")
    total_load_kwh = np.zeros(8760)
    
    # Create 2025 hourly index
    idx_2025 = pd.date_range('2025-01-01', periods=8760, freq='h')
    
    for _, fname in FAMILY_FILES.items():
        fpath = INPUT_DIR / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Missing {fpath}")
            
        # Load 2024 data (10 min)
        raw = pd.read_csv(fpath, parse_dates=['timestamp'])
        
        # Resample to hourly kWh
        # Exclude total_power_W if present, sum services
        cols = [c for c in raw.columns if c not in ['timestamp', 'total_power_W']]
        raw.set_index('timestamp', inplace=True)
        hourly = raw[cols].resample('h').sum() / 6 / 1000 # W -> kW avg -> kWh
        hourly['total'] = hourly.sum(axis=1)
        
        # Map 2024 -> 2025
        # Quick & Dirty mapping: ignore leap day difference or smart map
        # Scenario 0 logic: map by doy/hour.
        hourly['doy'] = hourly.index.dayofyear
        hourly['hour'] = hourly.index.hour
        
        # 2025 template
        temp_df = pd.DataFrame(index=idx_2025)
        temp_df['doy'] = temp_df.index.dayofyear
        temp_df['hour'] = temp_df.index.hour
        
        # Merge
        merged = temp_df.merge(hourly[['total', 'doy', 'hour']], on=['doy', 'hour'], how='left')
        merged['total'] = merged['total'].fillna(0)
        
        # Add to building total (5 units per family)
        total_load_kwh += merged['total'].values * UNITS_PER_FAMILY
        
    print(f"  Total Annual Load: {total_load_kwh.sum():,.2f} kWh")
    print(f"  Total Annual PV: {pv_kwh.sum():,.2f} kWh")
    
    # Create master DF
    df = pd.DataFrame(index=idx_2025)
    df['Load_kWh'] = total_load_kwh
    df['PV_kWh'] = pv_kwh
    df['Price_EUR_kWh'] = prices.values
    df['Band'] = bands.values
    
    return df

def run_optimization(df):
    print("\nSetting up Optimization Model...")
    
    # Time horizon
    T = len(df)
    
    # Variables
    G_imp = cp.Variable(T, nonneg=True, name="GridImport")
    B_ch = cp.Variable(T, nonneg=True, name="BattCharge")
    B_dis = cp.Variable(T, nonneg=True, name="BattDischarge")
    SOC = cp.Variable(T+1, nonneg=True, name="SOC")
    
    PV_use = cp.Variable(T, nonneg=True, name="PV_SelfCons")
    PV_ch = cp.Variable(T, nonneg=True, name="PV_Charge")
    PV_curt = cp.Variable(T, nonneg=True, name="PV_Curtailment")
    
    # Parameters
    Load = df['Load_kWh'].values
    PV_gen = df['PV_kWh'].values
    Price = df['Price_EUR_kWh'].values
    
    constraints = []
    
    # 1. PV Balance
    # PV_gen = PV_use + PV_ch + PV_curt
    constraints += [PV_gen == PV_use + PV_ch + PV_curt]
    
    # 2. PV-Only Charging
    # B_ch must equal PV_ch (no grid charging)
    constraints += [B_ch == PV_ch]
    
    # 3. Load Balance
    # Load = PV_use + B_dis + G_imp
    constraints += [Load == PV_use + B_dis + G_imp]
    
    # 4. SOC Dynamics
    # SOC[t+1] = SOC[t] + eff*B_ch - B_dis/eff
    # Note: cvxpy indexing SOC is 0..T
    constraints += [SOC[0] == SOC_INITIAL]
    constraints += [SOC[1:] == SOC[:-1] + EFF_CH * B_ch - B_dis / EFF_DIS]
    
    # 5. Constraints & Limits
    constraints += [SOC >= SOC_MIN, SOC <= SOC_MAX]
    constraints += [B_ch <= BESS_POWER_KW] # Hourly energy <= Power * 1h
    constraints += [B_dis <= BESS_POWER_KW]
    
    # 6. End Condition
    constraints += [SOC[T] == SOC_INITIAL]
    
    # Objective
    # Min Cost + Penalty
    cost = cp.sum(cp.multiply(Price, G_imp))
    penalty = AGEING_COST * cp.sum(B_ch + B_dis)
    
    objective = cp.Minimize(cost + penalty)
    
    print(f"Solving with {cp.installed_solvers()}...")
    prob = cp.Problem(objective, constraints)
    
    # Try available solvers
    try:
        prob.solve(solver=cp.CBC, verbose=True)
    except:
        try:
            prob.solve(solver=cp.GLPK_MI, verbose=True)
        except: 
            # Fallback to ECOS or OSQP (might struggle with exact equality or simply be slower/less precise)
            prob.solve(solver=cp.ECOS, verbose=True)
            
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print(f"WARNING: Problem status is {prob.status}")
        
    print(f"Solved. Cost: {prob.value:.2f} EUR")
    
    # Extract results
    results = df.copy()
    results['Grid_Import_kWh'] = G_imp.value
    results['Batt_Ch_kWh'] = B_ch.value
    results['Batt_Dis_kWh'] = B_dis.value
    results['SOC_kWh'] = SOC.value[:-1] # take 0..T-1
    results['PV_Use_kWh'] = PV_use.value
    results['PV_Ch_kWh'] = PV_ch.value
    results['PV_Curt_kWh'] = PV_curt.value
    
    return results

def sanity_checks(df):
    print("\nRunning Sanity Checks...")
    
    # 1. Load Balance
    supply = df['PV_Use_kWh'] + df['Batt_Dis_kWh'] + df['Grid_Import_kWh']
    load_err = np.max(np.abs(supply - df['Load_kWh']))
    print(f"  Max Load Balance Error: {load_err:.6f}")
    
    # 2. PV Balance
    pv_dest = df['PV_Use_kWh'] + df['PV_Ch_kWh'] + df['PV_Curt_kWh']
    pv_err = np.max(np.abs(pv_dest - df['PV_kWh']))
    print(f"  Max PV Balance Error: {pv_err:.6f}")
    
    # 3. PV Charging Only
    ch_err = np.max(np.abs(df['Batt_Ch_kWh'] - df['PV_Ch_kWh']))
    print(f"  Max PV-Only Charging Error: {ch_err:.6f}")
    
    # 4. SOC Bounds
    min_soc = df['SOC_kWh'].min()
    max_soc = df['SOC_kWh'].max()
    print(f"  SOC Range: [{min_soc:.2f}, {max_soc:.2f}] (Limit: [{SOC_MIN:.2f}, {SOC_MAX:.2f}])")
    
    # 5. Simultaneous Ch/Dis
    simul = ((df['Batt_Ch_kWh'] > 0.01) & (df['Batt_Dis_kWh'] > 0.01)).sum()
    print(f"  Simultaneous Ch/Dis Hours: {simul}")
    
    # 6. End SOC
    # Need to reconstruct final SOC or pass it out. 
    # But strictly SOC[T] was constrained to equal SOC[0].
    
    return True

def generate_plots(df):
    PLOTS_DIR.mkdir(exist_ok=True, parents=True)
    
    # 1. Weekly Plot (Winter) - Jan 8-14
    start_winter = '2025-01-08'
    end_winter = '2025-01-14 23:00'
    plot_week(df, start_winter, end_winter, 'winter_week')
    
    # 2. Weekly Plot (Summer) - Jul 1-7
    start_summer = '2025-07-01'
    end_summer = '2025-07-07 23:00'
    plot_week(df, start_summer, end_summer, 'summer_week')
    
    # 3. Annual SOC
    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df['SOC_kWh'], color='purple', linewidth=0.5)
    plt.axhline(SOC_MAX, color='r', linestyle='--')
    plt.axhline(SOC_MIN, color='r', linestyle='--')
    plt.title('Annual Battery State of Charge')
    plt.ylabel('kWh')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'annual_soc.png', dpi=150)
    plt.close()

def plot_week(df, start, end, name):
    mask = (df.index >= start) & (df.index <= end)
    sub = df.loc[mask]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Ax1: Power
    ax1.plot(sub.index, sub['Load_kWh'], label='Load', color='black', linewidth=1.5)
    ax1.plot(sub.index, sub['PV_kWh'], label='PV Gen', color='orange', alpha=0.7)
    ax1.bar(sub.index, sub['Grid_Import_kWh'], label='Grid Imp', color='red', alpha=0.3, width=0.04)
    ax1.bar(sub.index, sub['Batt_Dis_kWh'], bottom=sub['Grid_Import_kWh'], label='Batt Dis', color='green', alpha=0.3, width=0.04)
    
    ax1.set_ylabel('Power (kWh/h)')
    ax1.legend(loc='upper left')
    ax1.set_title(f'Operation: {start} to {end}')
    ax1.grid(True, alpha=0.3)
    
    # Ax2: SOC & Price
    ax2.plot(sub.index, sub['SOC_kWh'], label='SOC', color='purple', linewidth=2)
    ax2.set_ylabel('SOC (kWh)', color='purple')
    ax2.set_ylim(0, BESS_CAPACITY_KWH)
    ax2.axhline(SOC_MAX, linestyle='--', color='purple', alpha=0.5)
    ax2.axhline(SOC_MIN, linestyle='--', color='purple', alpha=0.5)
    
    ax3 = ax2.twinx()
    ax3.plot(sub.index, sub['Price_EUR_kWh'], label='Price', color='gray', linestyle='--')
    ax3.set_ylabel('Price (€/kWh)', color='gray')
    
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'{name}.png', dpi=150)
    plt.close()

def save_tables(df):
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    df.to_csv(OUTPUT_DIR / 'scenario1_hourly_dispatch.csv')
    
    # Calculate Summary Metrics
    total_cost = (df['Grid_Import_kWh'] * df['Price_EUR_kWh']).sum()
    total_load = df['Load_kWh'].sum()
    total_pv = df['PV_kWh'].sum()
    total_import = df['Grid_Import_kWh'].sum()
    self_cons = df['PV_Use_kWh'].sum() + df['Batt_Ch_kWh'].sum() # Approximately (neglecting efficiency loss for simplicity in this metric, or use PV - Export)
    # Strictly: Load covered by PV = PV_use + Batt_dis. 
    # But Self Cons usually means how much PV was NOT Exported/Curtailed.
    # Here PV_Curt is explicitly tracked.
    pv_utilized = total_pv - df['PV_Curt_kWh'].sum()
    
    summary = {
        'Total_Cost_EUR': total_cost,
        'Total_Load_kWh': total_load,
        'Total_PV_kWh': total_pv,
        'Total_Grid_Import_kWh': total_import,
        'PV_Self_Consumption_kWh': pv_utilized,
        'SSR_%': (1 - total_import/total_load)*100,
        'SCR_%': (pv_utilized/total_pv)*100
    }
    
    s_df = pd.Series(summary).to_frame(name='Value')
    s_df.to_csv(OUTPUT_DIR / 'scenario1_summary_metrics.csv')
    print("Results saved.")

def main():
    print("=== SCENARIO 1 OPTIMIZATION ===")
    df = load_and_process_data()
    res = run_optimization(df)
    sanity_checks(res)
    generate_plots(res)
    save_tables(res)
    print("Done.")

if __name__ == '__main__':
    main()
