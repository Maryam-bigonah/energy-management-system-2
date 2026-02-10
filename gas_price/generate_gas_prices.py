"""
Generate ARERA Monthly Gas Price Dataset for 2025
Creates CSV with official C_MEM,m values (€/Smc)
"""

import pandas as pd

# ARERA C_MEM,m values for 2025 (€/Smc)
gas_prices = {
    'Year': [2025] * 12,
    'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
    'C_MEM,m (€/Smc)': [
        0.533576,  # Jan
        0.566178,  # Feb
        0.455069,  # Mar
        0.402365,  # Apr
        0.403010,  # May
        0.418839,  # Jun
        0.392478,  # Jul
        0.380886,  # Aug
        0.373358,  # Sep
        0.353669,  # Oct
        0.348704,  # Nov
        0.327985   # Dec
    ],
    'Source': ['ARERA – C_MEM,m (tutela vulnerabili)'] * 12,
    'Notes': ['Monthly value, not hourly. Wholesale PSV day-ahead monthly average converted to €/Smc (ARERA uses PCS=0.038520 GJ/Smc).'] * 12
}

# Create DataFrame
df = pd.DataFrame(gas_prices)

# Save to CSV
output_file = 'gas_prices_2025.csv'
df.to_csv(output_file, index=False)

print("="*70)
print("ARERA GAS PRICE DATASET (2025)")
print("="*70)
print(f"\n✓ Saved to: {output_file}")
print(f"\nDataset Summary:")
print(f"  Months: 12")
print(f"  Lowest price:  {df['C_MEM,m (€/Smc)'].min():.6f} €/Smc (December)")
print(f"  Highest price: {df['C_MEM,m (€/Smc)'].max():.6f} €/Smc (February)")
print(f"  Average price: {df['C_MEM,m (€/Smc)'].mean():.6f} €/Smc")

print("\n" + "="*70)
print("MARKDOWN TABLE FOR THESIS")
print("="*70)
print()
print(df.to_markdown(index=False))
print()
print("="*70)
