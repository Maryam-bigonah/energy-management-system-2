"""
Generate Fixed ARERA Band Prices for 2025
Creates a CSV with hourly data showing:
- Timestamp
- ARERA band (F1/F2/F3)
- Fixed price per band
- Holiday flag
"""

import pandas as pd
from datetime import datetime, timedelta
import holidays

# Fixed prices from December 2025 offer (€/kWh)
PRICES = {
    'F1': 0.129865,  # Mon-Fri 08:00-19:00 (peak)
    'F2': 0.120466,  # Mon-Fri 07:00-08:00, 19:00-23:00 + Sat 07:00-23:00 (mid-peak)
    'F3': 0.106513   # Nights, Sundays, holidays (off-peak)
}

def classify_arera_band(ts, it_holidays):
    """
    Classify timestamp into ARERA band
    
    F1: Mon-Fri 08:00-19:00 (excluding holidays)
    F2: Mon-Fri 07:00-08:00, 19:00-23:00 + Sat 07:00-23:00 (excluding holidays)
    F3: Mon-Sat 00:00-07:00, 23:00-24:00 + Sundays + holidays (all hours)
    """
    is_holiday = (ts.date() in it_holidays)
    dow = ts.weekday()  # 0=Mon ... 6=Sun
    h = ts.hour
    
    # Sundays or holidays -> F3 (all hours)
    if dow == 6 or is_holiday:
        return 'F3'
    
    # Saturday
    if dow == 5:
        if 7 <= h < 23:
            return 'F2'
        else:
            return 'F3'
    
    # Mon-Fri (not holiday)
    if 8 <= h < 19:
        return 'F1'
    elif (7 <= h < 8) or (19 <= h < 23):
        return 'F2'
    else:
        return 'F3'

def generate_fixed_price_dataset(year=2025):
    """Generate full year hourly dataset with fixed ARERA prices"""
    
    print(f"Generating ARERA band price dataset for {year}...")
    
    # Initialize Italian holidays
    it_holidays = holidays.Italy(years=year)
    
    # Generate hourly timestamps for the year
    start_date = datetime(year, 1, 1, 0, 0, 0)
    end_date = datetime(year + 1, 1, 1, 0, 0, 0)
    
    timestamps = []
    current = start_date
    while current < end_date:
        timestamps.append(current)
        current += timedelta(hours=1)
    
    print(f"  Generated {len(timestamps)} hourly timestamps")
    
    # Create DataFrame
    df = pd.DataFrame({'timestamp': timestamps})
    
    # Classify bands
    print("  Classifying ARERA bands...")
    df['arera_band'] = df['timestamp'].apply(lambda x: classify_arera_band(x, it_holidays))
    
    # Add holiday flag
    df['is_holiday'] = df['timestamp'].apply(lambda x: x.date() in it_holidays)
    
    # Add fixed price based on band
    df['price_eur_per_kwh'] = df['arera_band'].map(PRICES)
    
    # Add day name and hour for clarity
    df['day_of_week'] = df['timestamp'].dt.day_name()
    df['hour'] = df['timestamp'].dt.hour
    df['date'] = df['timestamp'].dt.date
    
    # Reorder columns
    df = df[[
        'timestamp',
        'date',
        'hour',
        'day_of_week',
        'arera_band',
        'price_eur_per_kwh',
        'is_holiday'
    ]]
    
    # Print statistics
    print("\n" + "="*70)
    print("DATASET SUMMARY")
    print("="*70)
    print(f"Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Total hours: {len(df)}")
    print(f"\nFixed Prices:")
    print(f"  F1 (peak):     {PRICES['F1']:.6f} €/kWh")
    print(f"  F2 (mid-peak): {PRICES['F2']:.6f} €/kWh")
    print(f"  F3 (off-peak): {PRICES['F3']:.6f} €/kWh")
    
    print(f"\nBand Distribution:")
    band_counts = df['arera_band'].value_counts()
    for band in ['F1', 'F2', 'F3']:
        count = band_counts[band]
        pct = count / len(df) * 100
        print(f"  {band}: {count:5d} hours ({pct:5.1f}%)")
    
    print(f"\nHolidays in {year}:")
    holiday_dates = sorted([date for date in it_holidays.keys() if date.year == year])
    for date in holiday_dates:
        print(f"  {date.strftime('%Y-%m-%d (%A)')}: {it_holidays[date]}")
    
    # Save to CSV
    output_file = f"arera_fixed_prices_{year}.csv"
    df.to_csv(output_file, index=False)
    
    print("\n" + "="*70)
    print(f"✓ Saved to: {output_file}")
    print("="*70)
    
    return df

if __name__ == '__main__':
    df = generate_fixed_price_dataset(2025)
