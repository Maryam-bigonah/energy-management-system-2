"""
Scenario 2: PV + Shared BESS + Service Flexibility Optimization

This extends Scenario 1 by allowing flexible services to shift in time:
- Non-interruptible cycles: Washing Machine, Dishwasher, Dryer
- Interruptible services: EV Charging

Penalty mechanism discourages scheduling in F1 (peak) hours.

Input:
    - Fixed Building Base Load (inflexible services)
    - Flexible Service Demands (with time windows)
    - Fixed PV Generation
    - Fixed TOU Prices + Penalty Weights
    - Battery Specs (same as Scenario 1)
    
Output:
    - Optimal BESS dispatch + Service Schedules to minimize Cost + Penalty
"""

import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime, timedelta

# --- CONFIGURATION ---
INPUT_DIR = Path('..') # pv/optimization/
OUTPUT_DIR = Path('scenario2_results')
PLOTS_DIR = OUTPUT_DIR / 'plots'

# Battery Specs (Same as Scenario 1)
BESS_CAPACITY_KWH = 20.0
BESS_POWER_KW = 5.0
ROUNDTRIP_EFF = 0.90
AGEING_COST = 1e-6

SOC_MIN_PCT = 0.05
SOC_MAX_PCT = 0.95
SOC_INITIAL_PCT = 0.50

# Derived
EFF_CH = np.sqrt(ROUNDTRIP_EFF)
EFF_DIS = np.sqrt(ROUNDTRIP_EFF)
SOC_MIN = SOC_MIN_PCT * BESS_CAPACITY_KWH
SOC_MAX = SOC_MAX_PCT * BESS_CAPACITY_KWH
SOC_INITIAL = SOC_INITIAL_PCT * BESS_CAPACITY_KWH

# Penalty Weights (EUR/kWh equivalent for discomfort)
# These should be >> electricity prices to prioritize comfort
PENALTY_F1 = 1.0     # High penalty for peak hours
PENALTY_F2 = 0.3     # Medium penalty for mid-peak
PENALTY_F3 = 0.0     # No penalty for off-peak

# Large multiplier to make penalty dominant
RHO = 100.0  # Penalty weight multiplier

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

# Service Definitions
# Non-Interruptible Cycles (per apartment per day, averaged)
APPLIANCE_CYCLES = {
    'washing_machine': {
        'duration_h': 2,
        'power_kw': 0.5,  # Average during cycle
        'daily_cycles': 1.0,  # Average cycles per day per 20 apartments
        'window_start_h': 6,  # Earliest start (6 AM)
        'window_end_h': 22,   # Latest start (10 PM)
    },
    'dishwasher': {
        'duration_h': 2,
        'power_kw': 1.2,
        'daily_cycles': 1.5,
        'window_start_h': 7,
        'window_end_h': 23,
    },
    'dryer': {
        'duration_h': 1.5,
        'power_kw': 2.0,
        'daily_cycles': 0.8,
        'window_start_h': 8,
        'window_end_h': 22,
    }
}

# EV Charging (Interruptible)
EV_CONFIG = {
    'charger_power_kw': 7.0,  # Max charging power
    'daily_energy_kwh': 11.2,  # Average daily requirement (building-wide for EV owners)
    'arrival_hour': 18,  # Typical arrival at 6 PM
    'departure_hour': 7,  # Departure next day at 7 AM
}


def load_and_process_data():
    """Load prices, PV, and building load data."""
    print("Loading Data...")
    
    # 1. Load Prices
    print(f"  Prices: {PRICE_FILE}")
    prices_df = pd.read_csv(INPUT_DIR / PRICE_FILE, parse_dates=['timestamp'])
    prices_df.set_index('timestamp', inplace=True)
    
    if len(prices_df) != 8760:
        print(f"Warning: Price index length {len(prices_df)}")
        
    prices = prices_df['price_eur_per_kwh']
    bands = prices_df['arera_band']
    
    # Create penalty weights based on bands
    penalty_weights = np.zeros(8760)
    penalty_weights[bands == 'F1'] = PENALTY_F1
    penalty_weights[bands == 'F2'] = PENALTY_F2
    penalty_weights[bands == 'F3'] = PENALTY_F3

    # 2. Load PV
    print(f"  PV: {PV_FILE}")
    pv_df = pd.read_csv(INPUT_DIR / PV_FILE)
    pv_watts = pv_df['P_pred'].values
    if len(pv_watts) > 8760:
        pv_watts = pv_watts[:8760]
    pv_kwh = pv_watts / 1000.0
    
    # 3. Load & Aggregate BASE Load (non-flexible services)
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
        cols = [c for c in raw.columns if c not in ['timestamp', 'total_power_W']]
        raw.set_index('timestamp', inplace=True)
        hourly = raw[cols].resample('h').sum() / 6 / 1000
        hourly['total'] = hourly.sum(axis=1)
        
        # Map 2024 -> 2025
        hourly['doy'] = hourly.index.dayofyear
        hourly['hour'] = hourly.index.hour
        
        temp_df = pd.DataFrame(index=idx_2025)
        temp_df['doy'] = temp_df.index.dayofyear
        temp_df['hour'] = temp_df.index.hour
        
        merged = temp_df.merge(hourly[['total', 'doy', 'hour']], on=['doy', 'hour'], how='left')
        merged['total'] = merged['total'].fillna(0)
        
        total_load_kwh += merged['total'].values * UNITS_PER_FAMILY
        
    # For Scenario 2, we extract flexible loads from total
    # SIMPLIFIED: Assume flexible loads are additive on top of base
    # In reality, we'd need to decompose the original profiles
    # For now, we treat the original load as BASE LOAD
    
    print(f"  Total Annual Base Load: {total_load_kwh.sum():,.2f} kWh")
    print(f"  Total Annual PV: {pv_kwh.sum():,.2f} kWh")
    
    # Create master DF
    df = pd.DataFrame(index=idx_2025)
    df['Load_Base_kWh'] = total_load_kwh
    df['PV_kWh'] = pv_kwh
    df['Price_EUR_kWh'] = prices.values
    df['Band'] = bands.values
    df['Penalty_Weight'] = penalty_weights
    df['Hour'] = df.index.hour
    
    return df


def create_flexible_service_profiles(df):
    """
    Create flexible service demand profiles.
    
    Returns dictionaries defining when services can run and their energy requirements.
    """
    T = len(df)
    
    # For non-interruptible cycles, we create a simplified daily pattern
    # In a full implementation, this would be probabilistic or based on historical data
    
    # SIMPLIFIED: Create a few representative cycles per week
    # For demonstration, we'll create daily cycles
    
    cycles = []
    
    # Example: 1 washing machine cycle per day for the building
    for day in range(365):
        start_hour = day * 24 + 10  # Default to 10 AM
        if start_hour + 2 < T:
            cycles.append({
                'type': 'washing_machine',
                'duration': 2,
                'energy_kwh': 1.0,
                'window_start': day * 24 + 6,
                'window_end': day * 24 + 22,
            })
    
    # EV Charging: Daily requirement
    ev_sessions = []
    for day in range(365):
        arrival = day * 24 + EV_CONFIG['arrival_hour']
        departure = (day + 1) * 24 + EV_CONFIG['departure_hour']
        if departure < T:
            ev_sessions.append({
                'arrival': arrival,
                'departure': departure,
                'energy_required_kwh': EV_CONFIG['daily_energy_kwh'],
                'max_power_kw': EV_CONFIG['charger_power_kw'],
            })
    
    return cycles, ev_sessions


def run_optimization(df, cycles, ev_sessions):
    """
    Run the TRUE MILP optimization with binary cycle start times and daily EV windows.
    """
    print("\nSetting up MILP Model with Shiftable Services...")
    
    T = len(df)
    
    # ==========================================
    # DECISION VARIABLES
    # ==========================================
    
    # Grid & Battery (same as Scenario 1)
    G_imp = cp.Variable(T, nonneg=True, name="GridImport")
    B_ch = cp.Variable(T, nonneg=True, name="BattCharge")
    B_dis = cp.Variable(T, nonneg=True, name="BattDischarge")
    SOC = cp.Variable(T+1, nonneg=True, name="SOC")
    
    PV_use = cp.Variable(T, nonneg=True, name="PV_SelfCons")
    PV_ch = cp.Variable(T, nonneg=True, name="PV_Charge")
    PV_curt = cp.Variable(T, nonneg=True, name="PV_Curtailment")
    
    # ==========================================
    # APPLIANCE CYCLE BINARY VARIABLES
    # ==========================================
    # For each cycle, create binary start variable
    cycle_starts = []
    for i, cycle in enumerate(cycles):
        # Binary: y[t] = 1 if cycle i starts at hour t
        window_start = cycle['window_start']
        window_end = cycle['window_end']
        duration = cycle['duration']
        
        # Can only start if there's room for full cycle before window closes
        valid_start_hours = list(range(window_start, min(window_end - duration + 1, T - duration)))
        
        if len(valid_start_hours) > 0:
            # Create binary variable for valid start times
            y_cycle = cp.Variable(len(valid_start_hours), boolean=True, name=f"cycle_{i}_start")
            cycle_starts.append({
                'var': y_cycle,
                'cycle': cycle,
                'valid_hours': valid_start_hours,
                'index': i
            })
    
    # ==========================================
    # EV CHARGING VARIABLES (DAILY WINDOWS)
    # ==========================================
    # For each EV session, create power variable within window
    ev_power_vars = []
    for i, session in enumerate(ev_sessions):
        arrival = session['arrival']
        departure = session['departure']
        
        if departure <= T:
            window_length = departure - arrival
            # Power variable for this session
            ev_power = cp.Variable(window_length, nonneg=True, name=f"EV_session_{i}")
            ev_power_vars.append({
                'var': ev_power,
                'session': session,
                'arrival': arrival,
                'departure': departure
            })
    
    # Parameters
    Load_base = df['Load_Base_kWh'].values
    PV_gen = df['PV_kWh'].values
    Price = df['Price_EUR_kWh'].values
    Penalty_w = df['Penalty_Weight'].values
    
    constraints = []
    
    # ==========================================
    # CONSTRAINTS
    # ==========================================
    
    # 1. PV Balance
    constraints += [PV_gen == PV_use + PV_ch + PV_curt]
    
    # 2. PV-Only Charging
    constraints += [B_ch == PV_ch]
    
    # 3. Build Flexible Load from Cycles
    Flex_load_appliances = np.zeros(T)
    
    for cycle_info in cycle_starts:
        y = cycle_info['var']
        cycle = cycle_info['cycle']
        valid_hours = cycle_info['valid_hours']
        duration = cycle['duration']
        power = cycle['energy_kwh'] / duration  # Average power during cycle
        
        # Must start exactly once
        constraints += [cp.sum(y) == 1]
        
        # Build load profile: for each hour t, sum contributions from all possible starts
        for t in range(T):
            # Check which starts would make this hour active
            active_starts = []
            for idx, start_hour in enumerate(valid_hours):
                if start_hour <= t < start_hour + duration:
                    active_starts.append(idx)
            
            if len(active_starts) > 0:
                # This hour gets power if any of these starts were chosen
                Flex_load_appliances[t] += power * cp.sum([y[idx] for idx in active_starts])
    
    # 4. Build EV Load from Sessions
    EV_total_power = np.zeros(T)
    
    for ev_info in ev_power_vars:
        ev_p = ev_info['var']
        session = ev_info['session']
        arrival = ev_info['arrival']
        departure = ev_info['departure']
        
        # Must deliver required energy
        constraints += [cp.sum(ev_p) == session['energy_required_kwh']]
        
        # Power limit
        constraints += [ev_p <= session['max_power_kw']]
        
        # Map to global time array
        for local_t, global_t in enumerate(range(arrival, departure)):
            if global_t < T:
                EV_total_power[global_t] += ev_p[local_t]
    
    # 5. Total Load Balance
    Total_load = Load_base + Flex_load_appliances + EV_total_power
    constraints += [Total_load == PV_use + B_dis + G_imp]
    
    # 6. SOC Dynamics
    constraints += [SOC[0] == SOC_INITIAL]
    constraints += [SOC[1:] == SOC[:-1] + EFF_CH * B_ch - B_dis / EFF_DIS]
    
    # 7. SOC & Power Limits
    constraints += [SOC >= SOC_MIN, SOC <= SOC_MAX]
    constraints += [B_ch <= BESS_POWER_KW]
    constraints += [B_dis <= BESS_POWER_KW]
    
    # 8. End Condition
    constraints += [SOC[T] >= SOC_INITIAL]
    
    # ==========================================
    # OBJECTIVE FUNCTION
    # ==========================================
    
    # Economic cost
    cost = cp.sum(cp.multiply(Price, G_imp))
    
    # Battery aging penalty
    battery_penalty = AGEING_COST * cp.sum(B_ch + B_dis)
    
    # SERVICE FLEXIBILITY PENALTY (Comfort-Dominant)
    # Penalize appliances
    appliance_penalty = cp.sum(cp.multiply(Penalty_w, Flex_load_appliances))
    
    # Penalize EV charging
    ev_penalty = cp.sum(cp.multiply(Penalty_w, EV_total_power))
    
    flexible_penalty = appliance_penalty + ev_penalty
    
    # Combined objective (penalty-dominant)
    objective = cp.Minimize(cost + battery_penalty + RHO * flexible_penalty)
    
    # ==========================================
    # SOLVE
    # ==========================================
    
    print(f"Problem has {len([v for v in prob.variables() if v.is_boolean()])} binary variables")
    print(f"Solving with {cp.installed_solvers()}...")
    
    prob = cp.Problem(objective, constraints)
    
    # MILP solvers
    try:
        print("Attempting CBC solver...")
        prob.solve(solver=cp.CBC, verbose=True, maximumSeconds=300)
    except Exception as e:
        print(f"CBC failed: {e}")
        try:
            print("Attempting GLPK_MI solver...")
            prob.solve(solver=cp.GLPK_MI, verbose=True)
        except Exception as e:
            print(f"GLPK_MI failed: {e}")
            print("Falling back to ECOS (may not handle binaries well)...")
            prob.solve(solver=cp.ECOS, verbose=True)
            
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print(f"WARNING: Problem status is {prob.status}")
        if prob.status == "infeasible":
            print("Model is INFEASIBLE. Check constraints.")
            return None
    else:
        print(f"\n✓ Solved. Total Objective: {prob.value:.2f}")
        print(f"  Cost Component: {cost.value:.2f} EUR")
        print(f"  Penalty Component: {(RHO * flexible_penalty).value:.2f}")
    
    # Extract results
    results = df.copy()
    results['Grid_Import_kWh'] = G_imp.value
    results['Batt_Ch_kWh'] = B_ch.value
    results['Batt_Dis_kWh'] = B_dis.value
    results['SOC_kWh'] = SOC.value[:-1]
    results['PV_Use_kWh'] = PV_use.value
    results['PV_Ch_kWh'] = PV_ch.value
    results['PV_Curt_kWh'] = PV_curt.value
    
    # Extract flexible loads (need to evaluate expressions)
    flex_app_val = np.zeros(T)
    for t in range(T):
        if isinstance(Flex_load_appliances[t], (int, float)):
            flex_app_val[t] = Flex_load_appliances[t]
        else:
            flex_app_val[t] = Flex_load_appliances[t].value if hasattr(Flex_load_appliances[t], 'value') else 0
    
    ev_val = np.zeros(T)
    for t in range(T):
        if isinstance(EV_total_power[t], (int, float)):
            ev_val[t] = EV_total_power[t]
        else:
            ev_val[t] = EV_total_power[t].value if hasattr(EV_total_power[t], 'value') else 0
    
    results['Flex_Load_kWh'] = flex_app_val
    results['EV_Power_kWh'] = ev_val
    results['Total_Load_kWh'] = Load_base + flex_app_val + ev_val
    
    # Print cycle start times (for verification)
    print("\n=== CYCLE START TIMES ===")
    for cycle_info in cycle_starts:
        y_val = cycle_info['var'].value
        valid_hours = cycle_info['valid_hours']
        cycle = cycle_info['cycle']
        
        if y_val is not None:
            chosen_idx = np.argmax(y_val)
            if y_val[chosen_idx] > 0.5:
                start_hour = valid_hours[chosen_idx]
                start_time = df.index[start_hour]
                print(f"{cycle['type']}: starts at hour {start_hour} ({start_time})")
    
    return results


def generate_comparison_plots(df_scen1, df_scen2):
    """Generate comparison plots between Scenario 1 and Scenario 2."""
    PLOTS_DIR.mkdir(exist_ok=True, parents=True)
    
    # Plot 1: Weekly comparison (Winter)
    start = '2025-01-08'
    end = '2025-01-14 23:00'
    
    mask = (df_scen2.index >= start) & (df_scen2.index <= end)
    sub2 = df_scen2.loc[mask]
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Ax1: Load Profile Comparison
    axes[0].plot(sub2.index, sub2['Load_Base_kWh'], label='Base Load', color='black', linewidth=1)
    axes[0].plot(sub2.index, sub2['Total_Load_kWh'], label='Total Load (with Flex)', color='blue', linewidth=1.5)
    axes[0].fill_between(sub2.index, sub2['Load_Base_kWh'], sub2['Total_Load_kWh'], 
                          alpha=0.3, color='cyan', label='Flexible Services')
    axes[0].set_ylabel('Load (kWh/h)')
    axes[0].legend()
    axes[0].set_title('Scenario 2: Load Profile with Flexible Services')
    axes[0].grid(True, alpha=0.3)
    
    # Ax2: Service Breakdown
    axes[1].bar(sub2.index, sub2['Flex_Load_kWh'], label='Appliances', color='orange', alpha=0.6, width=0.04)
    axes[1].bar(sub2.index, sub2['EV_Power_kWh'], bottom=sub2['Flex_Load_kWh'], 
                label='EV Charging', color='green', alpha=0.6, width=0.04)
    axes[1].set_ylabel('Flexible Load (kWh/h)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Ax3: Price bands
    axes[2].scatter(sub2.index, sub2['Price_EUR_kWh'], c=sub2['Band'].map({'F1': 'red', 'F2': 'orange', 'F3': 'green'}),
                    s=20, alpha=0.7)
    axes[2].set_ylabel('Price (€/kWh)')
    axes[2].set_xlabel('Time')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'winter_week_flexibility.png', dpi=150)
    plt.close()
    
    print("Plots generated.")


def save_results(df):
    """Save results to CSV."""
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    df.to_csv(OUTPUT_DIR / 'scenario2_hourly_dispatch.csv')
    
    # Summary metrics
    total_cost = (df['Grid_Import_kWh'] * df['Price_EUR_kWh']).sum()
    total_load = df['Total_Load_kWh'].sum()
    total_flex = (df['Flex_Load_kWh'] + df['EV_Power_kWh']).sum()
    
    # Flexible energy by band
    flex_F1 = df[df['Band'] == 'F1'][['Flex_Load_kWh', 'EV_Power_kWh']].sum().sum()
    flex_F2 = df[df['Band'] == 'F2'][['Flex_Load_kWh', 'EV_Power_kWh']].sum().sum()
    flex_F3 = df[df['Band'] == 'F3'][['Flex_Load_kWh', 'EV_Power_kWh']].sum().sum()
    
    summary = {
        'Total_Cost_EUR': total_cost,
        'Total_Load_kWh': total_load,
        'Total_Flexible_kWh': total_flex,
        'Flexible_in_F1_kWh': flex_F1,
        'Flexible_in_F2_kWh': flex_F2,
        'Flexible_in_F3_kWh': flex_F3,
        'Percent_Flex_in_F1': (flex_F1 / total_flex * 100) if total_flex > 0 else 0,
        'Percent_Flex_in_F2': (flex_F2 / total_flex * 100) if total_flex > 0 else 0,
        'Percent_Flex_in_F3': (flex_F3 / total_flex * 100) if total_flex > 0 else 0,
    }
    
    pd.Series(summary).to_frame(name='Value').to_csv(OUTPUT_DIR / 'scenario2_summary_metrics.csv')
    
    print("\n=== SCENARIO 2 SUMMARY ===")
    print(f"Total Cost: €{total_cost:,.2f}")
    print(f"Total Flexible Energy: {total_flex:,.2f} kWh")
    print(f"  In F1: {flex_F1:,.2f} kWh ({flex_F1/total_flex*100:.1f}%)")
    print(f"  In F2: {flex_F2:,.2f} kWh ({flex_F2/total_flex*100:.1f}%)")
    print(f"  In F3: {flex_F3:,.2f} kWh ({flex_F3/total_flex*100:.1f}%)")


def main():
    print("=== SCENARIO 2: PV + BESS + SERVICE FLEXIBILITY ===\n")
    
    # Load data
    df = load_and_process_data()
    
    # Create flexible service definitions
    cycles, ev_sessions = create_flexible_service_profiles(df)
    print(f"\nCreated {len(cycles)} appliance cycles and {len(ev_sessions)} EV charging sessions")
    
    # Run optimization
    results = run_optimization(df, cycles, ev_sessions)
    
    # Generate plots
    generate_comparison_plots(df, results)
    
    # Save results
    save_results(results)
    
    print("\nDone.")


if __name__ == '__main__':
    main()
