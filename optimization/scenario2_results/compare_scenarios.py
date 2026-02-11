"""
Scenario 1b vs Scenario 2 Comparison Analysis

This script computes the KEY THESIS METRIC:
    ΔC_flex = Extra cost reduction from service flexibility beyond PV+BESS

Operational Cost:
    C = Σ p(t) · GridImport(t)  [€/year]

Metrics:
    C1b = Scenario 1 operational cost (PV + BESS, fixed services)
    C2  = Scenario 2 operational cost (PV + BESS + service shifting)
    
    ΔC_flex = C1b - C2  [€/year]
    %FlexGain = (C1b - C2) / C1b · 100  [%]

This answers: "How much extra savings do we get by allowing service shifting 
(within comfort limits), on top of what the battery alone provides?"

IMPORTANT: C2 is computed from GRID COST ONLY (not including penalty),
since penalty represents comfort/discomfort, not euros.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Paths
SCENARIO1_DIR = Path('../scenario1_bess/scenario1_results')
SCENARIO2_DIR = Path('.')
OUTPUT_DIR = Path('comparison_results')

SCENARIO1_FILE = SCENARIO1_DIR / 'scenario1_hourly_dispatch.csv'
SCENARIO2_FILE = SCENARIO2_DIR / 'scenario2_hourly_dispatch.csv'


def load_results():
    """Load Scenario 1 and 2 hourly results."""
    print("Loading results...")
    
    if not SCENARIO1_FILE.exists():
        raise FileNotFoundError(f"Scenario 1 results not found: {SCENARIO1_FILE}")
    
    if not SCENARIO2_FILE.exists():
        raise FileNotFoundError(f"Scenario 2 results not found: {SCENARIO2_FILE}")
    
    s1 = pd.read_csv(SCENARIO1_FILE, parse_dates=[0], index_col=0)
    s2 = pd.read_csv(SCENARIO2_FILE, parse_dates=[0], index_col=0)
    
    print(f"  Scenario 1: {len(s1)} hours")
    print(f"  Scenario 2: {len(s2)} hours")
    
    return s1, s2


def compute_operational_costs(s1, s2):
    """
    Compute operational costs (grid cost only).
    
    Returns:
        C1b: Scenario 1 operational cost [€]
        C2: Scenario 2 operational cost [€]
    """
    print("\nComputing Operational Costs (Grid Only)...")
    
    # Scenario 1: PV + BESS, fixed services
    C1b = (s1['Grid_Import_kWh'] * s1['Price_EUR_kWh']).sum()
    
    # Scenario 2: PV + BESS + service shifting
    # CRITICAL: Use grid cost only, NOT penalty
    C2 = (s2['Grid_Import_kWh'] * s2['Price_EUR_kWh']).sum()
    
    print(f"  C1b (Scenario 1 - PV+BESS fixed): €{C1b:,.2f}")
    print(f"  C2  (Scenario 2 - PV+BESS+shift): €{C2:,.2f}")
    
    return C1b, C2


def compute_flexibility_value(C1b, C2):
    """
    Compute the extra value from service flexibility.
    
    Returns:
        delta: Extra savings [€/year]
        pct: Extra savings as % of C1b
    """
    print("\nComputing Flexibility Value...")
    
    delta_C_flex = C1b - C2
    pct_flex_gain = 100 * delta_C_flex / C1b if C1b > 0 else 0
    
    print(f"  ΔC_flex (Extra saving from flexibility): €{delta_C_flex:,.2f}/year")
    print(f"  %FlexGain (Extra saving %): {pct_flex_gain:.2f}%")
    
    return delta_C_flex, pct_flex_gain


def compute_comfort_penalty(s2):
    """
    Compute the comfort/penalty metric for Scenario 2.
    
    This is separate from cost and represents the discomfort trade-off.
    """
    print("\nComputing Comfort Penalty (Scenario 2 only)...")
    
    if 'Penalty_Weight' not in s2.columns:
        print("  Warning: No penalty weights found. Skipping penalty calculation.")
        return None
    
    # Flexible service energy
    flex_energy = s2.get('Flex_Load_kWh', 0) + s2.get('EV_Power_kWh', 0)
    
    # Total penalty
    penalty = (s2['Penalty_Weight'] * flex_energy).sum()
    
    print(f"  Total Comfort Penalty: {penalty:,.2f} [penalty units]")
    
    return penalty


def compute_energy_metrics(s1, s2):
    """Compute additional energy-related metrics."""
    print("\nComputing Energy Metrics...")
    
    metrics = {}
    
    # Grid imports
    metrics['Grid_Import_S1_kWh'] = s1['Grid_Import_kWh'].sum()
    metrics['Grid_Import_S2_kWh'] = s2['Grid_Import_kWh'].sum()
    metrics['Grid_Reduction_kWh'] = metrics['Grid_Import_S1_kWh'] - metrics['Grid_Import_S2_kWh']
    metrics['Grid_Reduction_pct'] = 100 * metrics['Grid_Reduction_kWh'] / metrics['Grid_Import_S1_kWh']
    
    # PV self-consumption
    if 'PV_Use_kWh' in s1.columns:
        metrics['PV_SelfCons_S1_kWh'] = s1['PV_Use_kWh'].sum() + s1.get('PV_Ch_kWh', 0).sum()
        metrics['PV_SelfCons_S2_kWh'] = s2['PV_Use_kWh'].sum() + s2.get('PV_Ch_kWh', 0).sum()
    
    # Battery throughput
    if 'Batt_Ch_kWh' in s1.columns:
        metrics['Batt_Throughput_S1_kWh'] = s1['Batt_Ch_kWh'].sum() + s1['Batt_Dis_kWh'].sum()
        metrics['Batt_Throughput_S2_kWh'] = s2['Batt_Ch_kWh'].sum() + s2['Batt_Dis_kWh'].sum()
    
    # Flexible energy distribution by TOU band (Scenario 2)
    if 'Band' in s2.columns:
        flex_energy = s2.get('Flex_Load_kWh', 0) + s2.get('EV_Power_kWh', 0)
        
        flex_F1 = flex_energy[s2['Band'] == 'F1'].sum()
        flex_F2 = flex_energy[s2['Band'] == 'F2'].sum()
        flex_F3 = flex_energy[s2['Band'] == 'F3'].sum()
        total_flex = flex_F1 + flex_F2 + flex_F3
        
        if total_flex > 0:
            metrics['Flex_in_F1_kWh'] = flex_F1
            metrics['Flex_in_F2_kWh'] = flex_F2
            metrics['Flex_in_F3_kWh'] = flex_F3
            metrics['Flex_in_F1_pct'] = 100 * flex_F1 / total_flex
            metrics['Flex_in_F2_pct'] = 100 * flex_F2 / total_flex
            metrics['Flex_in_F3_pct'] = 100 * flex_F3 / total_flex
    
    for k, v in metrics.items():
        print(f"  {k}: {v:,.2f}")
    
    return metrics


def create_comparison_table(C1b, C2, delta_C_flex, pct_flex_gain, penalty, metrics):
    """Create a summary comparison table for the thesis."""
    print("\n=== COMPARISON TABLE (THESIS) ===")
    
    data = {
        'Metric': [
            'Operational Cost (Grid)',
            'Extra Savings from Flexibility',
            'Extra Savings %',
            'Comfort Penalty (Scenario 2)',
            'Grid Import',
            'Grid Import Reduction',
            'Flexible Energy in F1',
            'Flexible Energy in F2',
            'Flexible Energy in F3',
        ],
        'Scenario 1 (PV+BESS)': [
            f"€{C1b:,.2f}",
            '-',
            '-',
            '-',
            f"{metrics.get('Grid_Import_S1_kWh', 0):,.0f} kWh",
            '-',
            '-',
            '-',
            '-',
        ],
        'Scenario 2 (PV+BESS+Flex)': [
            f"€{C2:,.2f}",
            f"€{delta_C_flex:,.2f}",
            f"{pct_flex_gain:.2f}%",
            f"{penalty:,.2f}" if penalty is not None else 'N/A',
            f"{metrics.get('Grid_Import_S2_kWh', 0):,.0f} kWh",
            f"{metrics.get('Grid_Reduction_kWh', 0):,.0f} kWh ({metrics.get('Grid_Reduction_pct', 0):.1f}%)",
            f"{metrics.get('Flex_in_F1_kWh', 0):,.0f} kWh ({metrics.get('Flex_in_F1_pct', 0):.1f}%)",
            f"{metrics.get('Flex_in_F2_kWh', 0):,.0f} kWh ({metrics.get('Flex_in_F2_pct', 0):.1f}%)",
            f"{metrics.get('Flex_in_F3_kWh', 0):,.0f} kWh ({metrics.get('Flex_in_F3_pct', 0):.1f}%)",
        ]
    }
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    
    return df


def save_results(df, C1b, C2, delta_C_flex, pct_flex_gain, penalty, metrics):
    """Save comparison results."""
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    # Save comparison table
    df.to_csv(OUTPUT_DIR / 'scenario_comparison_table.csv', index=False)
    
    # Save detailed metrics
    summary = {
        'C1b_EUR': C1b,
        'C2_EUR': C2,
        'Delta_C_flex_EUR': delta_C_flex,
        'Pct_FlexGain': pct_flex_gain,
        'Comfort_Penalty_S2': penalty if penalty is not None else np.nan,
        **metrics
    }
    
    pd.Series(summary).to_frame(name='Value').to_csv(OUTPUT_DIR / 'detailed_metrics.csv')
    
    print(f"\nResults saved to {OUTPUT_DIR}/")


def create_visualization(s1, s2, delta_C_flex, pct_flex_gain):
    """Create comparison visualization."""
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Cost Comparison
    ax = axes[0, 0]
    scenarios = ['Scenario 1\n(PV+BESS)', 'Scenario 2\n(PV+BESS+Flex)']
    costs = [(s1['Grid_Import_kWh'] * s1['Price_EUR_kWh']).sum(),
             (s2['Grid_Import_kWh'] * s2['Price_EUR_kWh']).sum()]
    
    bars = ax.bar(scenarios, costs, color=['#1f77b4', '#2ca02c'])
    ax.set_ylabel('Annual Operational Cost (€)')
    ax.set_title('Cost Comparison')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, cost in zip(bars, costs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'€{cost:,.0f}',
                ha='center', va='bottom', fontsize=10)
    
    # Add savings annotation
    ax.annotate(f'Savings:\n€{delta_C_flex:,.0f}\n({pct_flex_gain:.1f}%)',
                xy=(1, costs[1]), xytext=(0.5, costs[0] * 0.95),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, color='red', ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    # Plot 2: Grid Import Comparison
    ax = axes[0, 1]
    imports = [s1['Grid_Import_kWh'].sum(), s2['Grid_Import_kWh'].sum()]
    bars = ax.bar(scenarios, imports, color=['#1f77b4', '#2ca02c'])
    ax.set_ylabel('Annual Grid Import (kWh)')
    ax.set_title('Grid Import Comparison')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, imp in zip(bars, imports):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{imp:,.0f} kWh',
                ha='center', va='bottom', fontsize=10)
    
    # Plot 3: Flexible Energy Distribution (Scenario 2)
    if 'Band' in s2.columns:
        ax = axes[1, 0]
        flex_energy = s2.get('Flex_Load_kWh', 0) + s2.get('EV_Power_kWh', 0)
        
        bands = ['F1\n(Peak)', 'F2\n(Mid)', 'F3\n(Off-peak)']
        energies = [
            flex_energy[s2['Band'] == 'F1'].sum(),
            flex_energy[s2['Band'] == 'F2'].sum(),
            flex_energy[s2['Band'] == 'F3'].sum()
        ]
        colors_band = ['#d62728', '#ff7f0e', '#2ca02c']
        
        bars = ax.bar(bands, energies, color=colors_band, alpha=0.7)
        ax.set_ylabel('Flexible Service Energy (kWh)')
        ax.set_title('Flexible Energy Distribution by TOU Band')
        ax.grid(axis='y', alpha=0.3)
        
        total = sum(energies)
        for bar, energy in zip(bars, energies):
            height = bar.get_height()
            pct = 100 * energy / total if total > 0 else 0
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{energy:,.0f} kWh\n({pct:.1f}%)',
                    ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Weekly Load Comparison (Sample week)
    ax = axes[1, 1]
    sample_start = '2025-01-08'
    sample_end = '2025-01-14 23:00'
    
    mask = (s2.index >= sample_start) & (s2.index <= sample_end)
    week_s1 = s1.loc[mask]
    week_s2 = s2.loc[mask]
    
    ax.plot(week_s1.index, week_s1['Load_Base_kWh'], 
            label='S1: Fixed Load', color='blue', linewidth=1.5, alpha=0.7)
    ax.plot(week_s2.index, week_s2['Total_Load_kWh'], 
            label='S2: Optimized Load', color='green', linewidth=1.5, alpha=0.7)
    ax.fill_between(week_s2.index, 
                     week_s2['Load_Base_kWh'], 
                     week_s2['Total_Load_kWh'],
                     alpha=0.3, color='cyan', label='Shifted Services')
    
    ax.set_ylabel('Load (kWh/h)')
    ax.set_xlabel('Time')
    ax.set_title('Sample Week Load Profile Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'scenario_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {OUTPUT_DIR}/scenario_comparison.png")


def main():
    print("=" * 60)
    print("SCENARIO 1 vs SCENARIO 2 COMPARISON ANALYSIS")
    print("Computing ΔC_flex: Value of Service Flexibility")
    print("=" * 60)
    
    # Load
    s1, s2 = load_results()
    
    # Compute operational costs (grid only)
    C1b, C2 = compute_operational_costs(s1, s2)
    
    # Compute flexibility value
    delta_C_flex, pct_flex_gain = compute_flexibility_value(C1b, C2)
    
    # Compute comfort penalty (separate metric)
    penalty = compute_comfort_penalty(s2)
    
    # Additional energy metrics
    metrics = compute_energy_metrics(s1, s2)
    
    # Create comparison table
    df = create_comparison_table(C1b, C2, delta_C_flex, pct_flex_gain, penalty, metrics)
    
    # Save results
    save_results(df, C1b, C2, delta_C_flex, pct_flex_gain, penalty, metrics)
    
    # Create visualization
    try:
        create_visualization(s1, s2, delta_C_flex, pct_flex_gain)
    except Exception as e:
        print(f"Warning: Visualization failed: {e}")
    
    print("\n" + "=" * 60)
    print("KEY RESULT FOR THESIS:")
    print(f"  ΔC_flex = €{delta_C_flex:,.2f}/year ({pct_flex_gain:.2f}%)")
    print("This is the extra cost reduction from service flexibility")
    print("on top of what PV+BESS alone provides.")
    print("=" * 60)


if __name__ == '__main__':
    main()
