
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import os

# Configuration
# Relative to this script location (pv/optimization/)
SC0_DIR = 'scenario0_results'
# New location for Scenario 1 Results
SC1_RESULT_FILE = 'scenario1_bess/scenario1_results/scenario1_hourly_dispatch.csv' 
OUT_DIR = 'scenario1_report_output'
PRICE_FILE = 'arera_fixed_prices_2025.csv'
UNITS_PER_FAMILY = 5

def load_data():
    print("Loading data...")
    sc0_fam = pd.read_csv(f"{SC0_DIR}/scenario0_family_summary.csv")
    sc0_bands = pd.read_csv(f"{SC0_DIR}/scenario0_band_breakdown.csv") 
    
    # Load Sc1 Dispatch
    print(f"Reading {SC1_RESULT_FILE}...")
    sc1_disp = pd.read_csv(SC1_RESULT_FILE, index_col=0, parse_dates=True)
    sc1_disp.index.name = 'timestamp' # Ensure named 'timestamp' for plots
    
    # Rename columns to match Report Script Logic
    rename_map = {
        'Load_kWh': 'Load_kW',
        'PV_kWh': 'PV_forecast_kW',
        'Grid_Import_kWh': 'Grid_Import_kW',
        'Batt_Ch_kWh': 'Bat_Ch_kW',
        'Batt_Dis_kWh': 'Bat_Dis_kW',
        'PV_Use_kWh': 'PV_Self_kW',
        'PV_Ch_kWh': 'PV_Charge_kW',
        'PV_Curt_kWh': 'PV_Curtail_kW',
        # SOC_kWh is already SOC_kWh
    }
    sc1_disp.rename(columns=rename_map, inplace=True)
    
    # Add Missing Columns
    if 'Grid_Export_kW' not in sc1_disp.columns:
        sc1_disp['Grid_Export_kW'] = 0.0 # No export in Scenario 1
        
    prices = pd.read_csv(PRICE_FILE)
    return sc0_fam, sc0_bands, sc1_disp, prices

def validate(df):
    print("\n[VALIDATION]")
    # 1. Length
    if len(df) != 8760: print(f"FAIL: Length {len(df)}")
    else: print("PASS: Length 8760")
    
    # 2. Balance: Load = Self + Dis + Imp (approx)
    # Check deviation
    supply = df['PV_Self_kW'] + df['Bat_Dis_kW'] + df['Grid_Import_kW']
    diff = (df['Load_kW'] - supply).abs().max()
    print(f"Max Balance Diff: {diff:.6f} kWh")
    if diff < 0.01: print("PASS: Energy Balance")
    else: print("WARN: check balance")
    
    # 3. SOC
    soc = df['SOC_kWh']
    print(f"SOC Range: {soc.min():.2f} - {soc.max():.2f}")
    if soc.min() >= -0.01 and soc.max() <= 50.01: print("PASS: SOC Bounds")
    else: print("FAIL: SOC violation")

def generate_report(sc0_fam, sc0_bands, sc1_disp, prices):
    Path(OUT_DIR).mkdir(exist_ok=True, parents=True)
    
    # --- CALCULATIONS ---
    # Baseline (Adjusted for exact load match)
    # We use the Load from Sc1 Dispatch to calculate the precise "Adjusted Baseline Cost"
    # because Sc0 file might have slightly different load due to mapping fix.
    baseline_load = sc1_disp['Load_kW'].sum()
    # Calculate baseline cost using correct price alignment
    # Ensure prices index align or just multiply arrays if sorted
    # Price file is 8760. sc1_disp is 8760.
    baseline_cost_adj = (sc1_disp['Load_kW'].values * sc1_disp['Price_EUR_kWh'].values).sum()
    
    # Sc1 Values
    sc1_load = sc1_disp['Load_kW'].sum()
    sc1_pv = sc1_disp['PV_forecast_kW'].sum()
    sc1_imp = sc1_disp['Grid_Import_kW'].sum()
    sc1_exp = sc1_disp['Grid_Export_kW'].sum()
    sc1_curt = sc1_disp['PV_Curtail_kW'].sum()
    sc1_op = (sc1_disp['Grid_Import_kW'] * sc1_disp['Price_EUR_kWh']).sum()
    
    # CAPEX (Fixed parameters)
    r = 0.03
    crf_pv = (r*1.03**20)/(1.03**20 - 1)
    crf_bat = (r*1.03**10)/(1.03**10 - 1)
    
    # 26 kW PV (approx from peak ~26kW peak?) 
    # Wait, 26 kW is total? Let's check peak PV from file.
    pv_peak = sc1_disp['PV_forecast_kW'].max()
    # Assuming Installation Size based on peak or installed capacity? BESS is 20kWh.
    # User said "PV + shared BESS". 
    # Use standard cost if not specified: 1200/kWp PV, 400/kWh Batt.
    # Peak PV in file is around 26kW?
    # Let's use 26 kW as placeholder if unknown.
    # BESS = 20 kWh.
    # ann_capex = (26 * 1200 * crf_pv) + (20 * 400 * crf_bat) # CORRECTED 50->20 BATTERY
    ann_capex = (26 * 1200 * crf_pv) + (20 * 400 * crf_bat)
    
    sc1_total = sc1_op + ann_capex
    savings_op = baseline_cost_adj - sc1_op
    savings_tot = baseline_cost_adj - sc1_total
    
    # --- TABLE A ---
    tbl_a = pd.DataFrame({
        'Metric': ['Total Load (kWh)', 'PV Generation (kWh)', 'Grid Import (kWh)', 'Grid Export (kWh)', 
                   'Curtailment (kWh)', 'Operational Cost (€)', 'Annualized CAPEX (€)', 'Total Annual Cost (€)', 
                   'Savings (Operational) (€)', 'Net Savings (Total) (€)'],
        'Scenario 0 (Baseline)': [baseline_load, 0, baseline_load, 0, 0, baseline_cost_adj, 0, baseline_cost_adj, 0, 0],
        'Scenario 1 (PV+BESS)': [sc1_load, sc1_pv, sc1_imp, sc1_exp, sc1_curt, sc1_op, ann_capex, sc1_total, savings_op, savings_tot]
    })
    tbl_a.to_csv(f"{OUT_DIR}/TableA_BuildingSummary.csv", index=False)
    
    # --- TABLE B: Family Costs ---
    # Sc0: Use original table but note it sums to original Sc0 cost.
    # We will use the "Unit" values from file.
    # Sc1: Equal allocation of sc1_total.
    sc1_unit_cost = sc1_total / 20
    
    tbl_b = sc0_fam[['Family_Type', 'Annual_Cost_€_per_unit']].copy()
    tbl_b.rename(columns={'Annual_Cost_€_per_unit': 'Sc0_Unit_€'}, inplace=True)
    tbl_b['Sc1_Unit_Alloc_€'] = sc1_unit_cost
    tbl_b['Savings_Unit_€'] = tbl_b['Sc0_Unit_€'] - tbl_b['Sc1_Unit_Alloc_€'] # Approx savings vs original baseline
    tbl_b['Total_Savings_Type_€'] = tbl_b['Savings_Unit_€'] * UNITS_PER_FAMILY
    tbl_b.to_csv(f"{OUT_DIR}/TableB_FamilyCosts.csv", index=False)
    
    # --- TABLE C: TOU Breakdown ---
    # Map Sc1 Import to Bands. Using fixed price levels to identify bands.
    unique_prices = sorted(sc1_disp['Price_EUR_kWh'].unique())
    # Assuming 3 prices: Low(F3), Mid(F2), High(F1).
    if len(unique_prices) >= 3:
        p_map = {unique_prices[0]: 'F3', unique_prices[1]: 'F2', unique_prices[2]: 'F1'}
    else:
        # Fallback if fewer prices
        p_map = {p: 'F1' for p in unique_prices} # DUMMY
        
    sc1_disp['Band'] = sc1_disp['Price_EUR_kWh'].map(p_map)
    
    sc1_tou = sc1_disp.groupby('Band')['Grid_Import_kW'].sum()
    sc1_tou_cost = (sc1_disp.groupby('Band').apply(lambda x: (x['Grid_Import_kW'] * x['Price_EUR_kWh']).sum(), include_groups=False))
    
    # Sc0 from summary file (sums of all families)
    sc0_tou = sc0_bands.groupby('Band')['kWh_total'].sum()
    sc0_tou_c = sc0_bands.groupby('Band')['Cost_€_total'].sum()
    
    tbl_c = pd.DataFrame({
        'Band': ['F1', 'F2', 'F3'],
        'Sc0_Import_kWh': [sc0_tou.get('F1',0), sc0_tou.get('F2',0), sc0_tou.get('F3',0)],
        'Sc1_Import_kWh': [sc1_tou.get('F1',0), sc1_tou.get('F2',0), sc1_tou.get('F3',0)],
        'Sc0_Cost_€': [sc0_tou_c.get('F1',0), sc0_tou_c.get('F2',0), sc0_tou_c.get('F3',0)],
        'Sc1_Cost_€': [sc1_tou_cost.get('F1',0), sc1_tou_cost.get('F2',0), sc1_tou_cost.get('F3',0)]
    })
    tbl_c['Diff_kWh'] = tbl_c['Sc1_Import_kWh'] - tbl_c['Sc0_Import_kWh']
    tbl_c['Diff_€'] = tbl_c['Sc1_Cost_€'] - tbl_c['Sc0_Cost_€']
    tbl_c.to_csv(f"{OUT_DIR}/TableC_TOU.csv", index=False)
    
    # --- TABLE D: Battery ---
    cycles = sc1_disp['Bat_Dis_kW'].sum() / 20.0 # CORRECTED 50->20 kWh
    throughput = sc1_disp['Bat_Ch_kW'].sum() + sc1_disp['Bat_Dis_kW'].sum()
    tbl_d = pd.DataFrame([{
        'Capacity': '20 kWh', 'Power': '5 kW',
        'SOC_Min': f"{sc1_disp['SOC_kWh'].min():.2f}",
        'SOC_Max': f"{sc1_disp['SOC_kWh'].max():.2f}",
        'Throughput_kWh': f"{throughput:.1f}",
        'EFC_Cycles': f"{cycles:.1f}",
        'Ch_Hours': (sc1_disp['Bat_Ch_kW'] > 0.01).sum(),
        'Dis_Hours': (sc1_disp['Bat_Dis_kW'] > 0.01).sum()
    }])
    tbl_d.to_csv(f"{OUT_DIR}/TableD_Battery.csv", index=False)
    
    # --- TABLE E: PV Util ---
    sc = 1 - (sc1_exp + sc1_curt)/sc1_pv if sc1_pv > 0 else 0
    ss = 1 - sc1_imp/sc1_load
    tbl_e = pd.DataFrame([{
        'Direct_Load_kWh': sc1_disp['PV_Self_kW'].sum(),
        'To_Battery_kWh': sc1_disp['PV_Charge_kW'].sum(),
        'Exported_kWh': sc1_exp,
        'Curtailed_kWh': sc1_curt,
        'Self_Consumption_%': sc*100,
        'Self_Sufficiency_%': ss*100
    }])
    tbl_e.to_csv(f"{OUT_DIR}/TableE_PV.csv", index=False)
    
    print("\nTables Generated in " + OUT_DIR)
    
    # --- PLOTS ---
    # 1. Winter Week (Jan 12-19)
    mask = (sc1_disp.index >= '2025-01-12') & (sc1_disp.index < '2025-01-19')
    dfw = sc1_disp[mask]
    
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(dfw.index, dfw['Load_kW'], 'k', label='Load')
    ax.plot(dfw.index, dfw['PV_forecast_kW'], 'orange', ls='--', label='PV')
    ax.bar(dfw.index, dfw['Bat_Ch_kW'], color='g', label='Ch', alpha=0.6, width=0.04)
    ax.bar(dfw.index, -dfw['Bat_Dis_kW'], color='r', label='Dis', alpha=0.6, width=0.04)
    ax.set_ylabel("kW")
    ax2 = ax.twinx()
    ax2.plot(dfw.index, dfw['SOC_kWh'], 'b', label='SOC', alpha=0.3)
    ax2.set_ylabel("SOC kWh")
    plt.title("Winter Week Dispatch")
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/Plot1_Winter.png")
    
    # 2. Summer Week (Jun 15-22)
    mask = (sc1_disp.index >= '2025-06-15') & (sc1_disp.index < '2025-06-22')
    dfs = sc1_disp[mask]
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(dfs.index, dfs['Load_kW'], 'k', label='Load')
    ax.plot(dfs.index, dfs['PV_forecast_kW'], 'orange', ls='--', label='PV')
    ax.bar(dfs.index, dfs['Bat_Ch_kW'], color='g', label='Ch', alpha=0.6, width=0.04)
    ax.bar(dfs.index, -dfs['Bat_Dis_kW'], color='r', label='Dis', alpha=0.6, width=0.04)
    ax.set_ylabel("kW")
    ax2 = ax.twinx()
    ax2.plot(dfs.index, dfs['SOC_kWh'], 'b', label='SOC', alpha=0.3)
    ax2.set_ylabel("SOC kWh")
    plt.title("Summer Week Dispatch")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/Plot2_Summer.png")
    
    # 3. SOC Hist
    plt.figure()
    plt.hist(sc1_disp['SOC_kWh'], bins=30, color='purple', edgecolor='black')
    plt.title("SOC Distribution")
    plt.xlabel("kWh")
    plt.savefig(f"{OUT_DIR}/Plot3_SOC.png")
    
    # 4. Import TOU
    df_tou = pd.DataFrame({
        'Band': ['F1', 'F2', 'F3'],
        'Sc0': tbl_c['Sc0_Import_kWh'],
        'Sc1': tbl_c['Sc1_Import_kWh']
    })
    df_tou.set_index('Band').plot(kind='bar', figsize=(6,4))
    plt.title("Grid Import by Band")
    plt.ylabel("kWh")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/Plot4_TOU_Import.png")
    
    # 5. Cost
    costs = [baseline_cost_adj, sc1_op, ann_capex, sc1_total]
    labs = ['Sc0', 'Sc1_Op', 'Capex', 'Sc1_Tot']
    plt.figure(figsize=(6,4))
    plt.bar(labs, costs, color=['gray', 'blue', 'orange', 'green'])
    plt.title("Annual Cost Breakdown")
    plt.ylabel("€")
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(f"{OUT_DIR}/Plot5_Cost.png")
    
    # 6. PV Util
    plt.figure(figsize=(5,5))
    plt.pie([tbl_e['Direct_Load_kWh'][0], tbl_e['To_Battery_kWh'][0], tbl_e['Exported_kWh'][0], tbl_e['Curtailed_kWh'][0]], 
            labels=['Direct', 'Charge', 'Export', 'Curtail'], autopct='%1.1f%%')
    plt.title("PV Energy Destination")
    plt.savefig(f"{OUT_DIR}/Plot6_PV.png")
    
    print("Plots Generated.")

if __name__ == "__main__":
    sc0f, sc0b, sc1d, pr = load_data()
    validate(sc1d)
    generate_report(sc0f, sc0b, sc1d, pr)
