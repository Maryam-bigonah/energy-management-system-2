"""
Visualize generated load profiles

Creates plots showing:
- Daily power profiles
- Weekly patterns
- Monthly comparisons
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import argparse


def plot_household_profile(csv_file: str, output_dir: str = "plots", show_plots: bool = False):
    """
    Create visualization plots for a household profile
    
    Args:
        csv_file: Path to CSV file
        output_dir: Directory to save plots
        show_plots: If True, display plots interactively (in addition to saving)
    """
    # Load data
    df = pd.read_csv(csv_file, parse_dates=['timestamp'], index_col='timestamp')
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    household_name = Path(csv_file).stem.replace('_load_profile', '').replace('_', ' ').title()
    
    # 1. One week profile (use first week of data)
    fig, ax = plt.subplots(figsize=(14, 6))
    start_date = df.index.min()
    end_date = start_date + pd.Timedelta(days=7)
    week_data = df[start_date:end_date]
    ax.plot(week_data.index, week_data['total_power_W'] / 1000, linewidth=0.8)
    ax.set_title(f'{household_name} - One Week Profile', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date/Time')
    ax.set_ylabel('Power (kW)')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\\n%H:%M'))
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{Path(csv_file).stem}_week.png", dpi=150)
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # 2. Average daily profile (weekday vs weekend)
    df['hour'] = df.index.hour + df.index.minute / 60
    df['is_weekend'] = df.index.dayofweek >= 5
    
    weekday_profile = df[~df['is_weekend']].groupby('hour')['total_power_W'].mean() / 1000
    weekend_profile = df[df['is_weekend']].groupby('hour')['total_power_W'].mean() / 1000
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(weekday_profile.index, weekday_profile.values, label='Weekday', linewidth=2)
    ax.plot(weekend_profile.index, weekend_profile.values, label='Weekend', linewidth=2)
    ax.set_title(f'{household_name} - Average Daily Profile', fontsize=14, fontweight='bold')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Average Power (kW)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 24)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{Path(csv_file).stem}_daily.png", dpi=150)
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # 3. Monthly consumption
    # Detect interval from timestamp differences
    if len(df) > 1:
        interval_minutes = (df.index[1] - df.index[0]).total_seconds() / 60
    else:
        interval_minutes = 10  # Default fallback
    
    interval_hours = interval_minutes / 60
    df['month'] = df.index.month
    monthly_kwh = df.groupby('month')['total_power_W'].sum() / 1000 * interval_hours
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(monthly_kwh.index, monthly_kwh.values, color='steelblue', alpha=0.8)
    ax.set_title(f'{household_name} - Monthly Energy Consumption', fontsize=14, fontweight='bold')
    ax.set_xlabel('Month')
    ax.set_ylabel('Energy (kWh)')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{Path(csv_file).stem}_monthly.png", dpi=150)
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    print(f"Created plots for {household_name}")
    print(f"  - One week profile")
    print(f"  - Average daily profile")
    print(f"  - Monthly consumption")
    if show_plots:
        print(f"  Displayed interactively and saved to: {output_dir}/\\n")
    else:
        print(f"  Saved to: {output_dir}/\\n")


def main():
    parser = argparse.ArgumentParser(description="Visualize load profiles")
    parser.add_argument('csv_file', nargs='?', help='CSV file to plot')
    parser.add_argument('--all', action='store_true', help='Plot all profiles in output/')
    parser.add_argument('--output', type=str, default='plots', help='Output directory for plots')
    parser.add_argument('--show', action='store_true', help='Display plots interactively (in addition to saving)')
    
    args = parser.parse_args()
    
    if args.all:
        csv_files = list(Path('output').glob('*_load_profile.csv'))
        print(f"Found {len(csv_files)} profiles to plot\\n")
        for csv_file in csv_files:
            plot_household_profile(str(csv_file), args.output, args.show)
    elif args.csv_file:
        plot_household_profile(args.csv_file, args.output, args.show)
    else:
        parser.error("Must specify csv_file or --all")
    
    if args.show:
        print(f"\\nAll plots displayed interactively and saved to: {args.output}/")
    else:
        print(f"\\nAll plots saved to: {args.output}/")


if __name__ == '__main__':
    main()

