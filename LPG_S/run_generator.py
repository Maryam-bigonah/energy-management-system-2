"""
Command-line interface for load profile generator

Usage:
    python run_generator.py --all
    python run_generator.py --household single_professional --year 2024
    python run_generator.py --household family_with_children --start 2024-01-01 --end 2024-02-01
"""

import argparse
from datetime import datetime
from pathlib import Path

from generator import generate_household_profile
from config_loader import load_config


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic household load profiles")
    
    parser.add_argument('--household', type=str, help='Household type to generate')
    parser.add_argument('--all', action='store_true', help='Generate all household types')
    parser.add_argument('--year', type=int, default=2024, help='Year to simulate (default: 2024)')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--interval', type=int, default=10, help='Time resolution in minutes (default: 10)')
    parser.add_argument('--output', type=str, default='output', help='Output directory (default: output)')
    parser.add_argument('--config-dir', type=str, default='config', help='Config directory (default: config)')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    
    parser.add_argument('--export', type=str, default='all', choices=['all', 'tasks', 'services', 'none'], 
                        help='Export additional datasets (default: all)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Load configuration
    devices, households = load_config(args.config_dir)
    
    from export import export_shiftable_tasks, export_thermal_services
    
    # Determine date range
    if args.start and args.end:
        start_date = datetime.strptime(args.start, '%Y-%m-%d')
        end_date = datetime.strptime(args.end, '%Y-%m-%d')
    else:
        # Full year
        start_date = datetime(args.year, 1, 1)
        end_date = datetime(args.year + 1, 1, 1)
    
    if args.all:
        households_to_process = list(households.keys())
    elif args.household:
        if args.household not in households:
            parser.error(f"Household '{args.household}' not found in config")
        households_to_process = [args.household]
    else:
        parser.error("Must specify --household or --all")

    print("="*60)
    print(f"GENERATING PROFILES: {len(households_to_process)} Households")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Interval: {args.interval} minutes")
    print(f"Export: {args.export}")
    print("="*60)
    
    for hh_id in households_to_process:
        print(f"\nProcessing: {hh_id}...")
        
        # Call Generator
        df, activations = generate_household_profile(
            hh_id,
            start_date,
            end_date,
            args.interval,
            args.config_dir,
            args.seed,
            return_activations=True
        )
        
        # Save Load Profile
        output_file = output_dir / f"{hh_id}_load_profile.csv"
        df.to_csv(output_file)
        print(f"Saved load profile: {output_file}")
        
        # Export Additional Layers
        if args.export in ['all', 'tasks']:
            export_shiftable_tasks(activations, hh_id, args.year, args.output)
            
        if args.export in ['all', 'services']:
            export_thermal_services(df, devices, hh_id, args.year, args.output, args.interval)

    print(f"\n\n{'='*60}")
    print("ALL TASKS COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == '__main__':
    main()
