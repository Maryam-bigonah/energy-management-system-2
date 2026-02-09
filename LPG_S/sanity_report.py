"""
Sanity Report Generator

Generates profiles for each household and validates against calibration targets.
Outputs detailed reports showing which devices are outside realistic ranges.
"""

import pandas as pd
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict

from generator import generate_household_profile
from config_loader import load_config


def load_calibration_targets(targets_file: str = "config/calibration_targets.yaml") -> Dict:
    """Load calibration targets"""
    with open(targets_file, 'r') as f:
        return yaml.safe_load(f)


def analyze_device_in_profile(
    df: pd.DataFrame,
    device_col: str,
    interval_minutes: float,
    days: int,
    device_cfg: 'DeviceConfig' = None
) -> Dict:
    """
    Analyze a single device's behavior in the profile
    
    Returns dict with:
        mean_W, max_W, kwh_per_day, annual_kwh_estimate,
        runtime_hours_per_day, num_activations
    """
    if device_col not in df.columns:
        return {}
    
    power_series = df[device_col]
    interval_hours = interval_minutes / 60
    
    # Basic stats
    mean_w = power_series.mean()
    max_w = power_series.max()
    
    # Energy
    total_kwh = (power_series.sum() * interval_hours) / 1000
    kwh_per_day = total_kwh / days if days > 0 else 0
    annual_kwh_estimate = kwh_per_day * 365
    
    # Runtime (time spent above standby threshold)
    threshold = 1.0
    if device_cfg:
        threshold = device_cfg.standby_power_w + 1.0
    elif max_w > 0:
        threshold = max_w * 0.1
        
    active_intervals = (power_series > threshold).sum()
    runtime_hours_per_day = (active_intervals * interval_hours) / days if days > 0 else 0
    
    # Activations (count transitions from low to high)
    is_active = (power_series > threshold).astype(int)
    activations = (is_active.diff() > 0).sum()
    activations_per_day = activations / days if days > 0 else 0
    
    # Estimate duty cycle
    duty_cycle = 0
    if device_cfg and device_cfg.pattern == 'cycling':
        # Physics-based duty calculation
        # avg = duty * active + (1-duty) * standby
        # duty * (active - standby) = avg - standby
        denom = device_cfg.active_power_w - device_cfg.standby_power_w
        if denom > 0:
            duty_cycle = (mean_w - device_cfg.standby_power_w) / denom
            duty_cycle = max(0.0, min(1.0, duty_cycle))
    elif max_w > 0:
        # Fallback approximation
        duty_cycle = mean_w / max_w
    
    return {
        'mean_W': mean_w,
        'max_W': max_w,
        'kwh_per_day': kwh_per_day,
        'annual_kwh_estimate': annual_kwh_estimate,
        'runtime_hours_per_day': runtime_hours_per_day,
        'activations_per_day': activations_per_day,
        'duty_cycle': duty_cycle
    }


def check_device_against_targets(device_id: str, stats: Dict, targets: Dict) -> Dict:
    """
    Check if device stats are within target ranges
    
    Returns dict with:
        status: 'PASS' | 'WARN' | 'FAIL'
        issues: list of issue descriptions
        suggestions: list of suggested fixes
    """
    if device_id not in targets['devices']:
        return {'status': 'UNKNOWN', 'issues': ['No calibration target'], 'suggestions': []}
    
    target = targets['devices'][device_id]
    issues = []
    suggestions = []
    
    # Check average power
    if 'avg_power_w_range' in target:
        min_w, max_w = target['avg_power_w_range']
        if stats['mean_W'] < min_w:
            issues.append(f"Avg power {stats['mean_W']:.1f}W < {min_w}W")
            suggestions.append("Increase active_power_w or duty cycle")
        elif stats['mean_W'] > max_w:
            issues.append(f"Avg power {stats['mean_W']:.1f}W > {max_w}W")
            suggestions.append("Reduce active_power_w or duty cycle")
    
    # Check annual energy
    if 'annual_kwh_range' in target:
        min_kwh, max_kwh = target['annual_kwh_range']
        if stats['annual_kwh_estimate'] < min_kwh:
            issues.append(f"Annual {stats['annual_kwh_estimate']:.0f} kWh < {min_kwh} kWh")
            suggestions.append("Increase power or usage frequency")
        elif stats['annual_kwh_estimate'] > max_kwh:
            issues.append(f"Annual {stats['annual_kwh_estimate']:.0f} kWh > {max_kwh} kWh")
            suggestions.append("Reduce power or usage frequency")
    
    # Check daily runtime
    if 'daily_runtime_hours_range' in target:
        min_h, max_h = target['daily_runtime_hours_range']
        if stats['runtime_hours_per_day'] > max_h:
            issues.append(f"Runtime {stats['runtime_hours_per_day']:.1f}h > {max_h}h/day")
            suggestions.append("Reduce expected_uses_per_day or avg_duration_min")
    
    # Check max daily energy
    if 'max_daily_kwh' in target:
        if stats['kwh_per_day'] > target['max_daily_kwh']:
            issues.append(f"Daily energy {stats['kwh_per_day']:.2f} kWh > {target['max_daily_kwh']} kWh")
            suggestions.append("Reduce power or frequency significantly")
    
    # Check duty cycle for cycling devices
    if 'duty_cycle_range' in target:
        min_duty, max_duty = target['duty_cycle_range']
        if stats['duty_cycle'] > max_duty:
            issues.append(f"Duty cycle {stats['duty_cycle']:.2f} > {max_duty}")
            suggestions.append("Increase cycle_off_min or decrease cycle_on_min")
    
    # Determine status
    if not issues:
        status = 'PASS'
    elif len(issues) <= 1 or stats['kwh_per_day'] < 5:  # Minor issues or low energy device
        status = 'WARN'
    else:
        status = 'FAIL'
    
    return {
        'status': status,
        'issues': issues,
        'suggestions': suggestions
    }


def generate_sanity_report(
    household_id: str,
    season: str,
    devices: Dict = None,
    config_dir: str = "config",
    seed: int = 42
):
    """Generate and analyze a weekly profile for one household in one season"""
    
    # Pick a week for the season
    season_dates = {
        'winter': (datetime(2024, 1, 15), datetime(2024, 1, 22)),
        'spring': (datetime(2024, 4, 15), datetime(2024, 4, 22)),
        'summer': (datetime(2024, 7, 15), datetime(2024, 7, 22)),
        'fall': (datetime(2024, 10, 15), datetime(2024, 10, 22))
    }
    
    start, end = season_dates[season]
    
    print(f"\n{'='*70}")
    print(f"Analyzing: {household_id} - {season}")
    print(f"{'='*70}")
    
    # Generate profile
    df = generate_household_profile(
        household_id, start, end,
        interval_minutes=10,
        config_dir=config_dir,
        random_seed=seed
    )
    
    days = (end - start).days
    interval_minutes = 10
    
    # Load targets
    targets = load_calibration_targets()
    
    # Use passed devices or load if missing (fallback)
    if devices is None:
        devices, _ = load_config(config_dir)
    
    # Build reverse mapping from column name to device ID
    # Must match generator.py _build_dataframe logic
    col_to_id = {}
    devices_config = devices  # From load_config or arg
    
    for d_id, d_cfg in devices_config.items():
        # Replicate generator's sanitization
        sanitized = d_cfg.name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("/", "_")
        col_to_id[sanitized] = d_id
        col_to_id[sanitized + "_W"] = d_id  # Handle suffix

    # Analyze each device
    device_results = []
    
    for col in df.columns:
        if col == 'total_power_W':
            continue
            
        # Identify device ID
        clean_col = col[:-2] if col.endswith('_W') else col
        device_id = col_to_id.get(clean_col)
        
        if not device_id:
            # Fallback fuzzy match (should not happen if logic matches)
            device_id = clean_col.lower()
            
        device_cfg = devices_config.get(device_id)
        
        # Analyze
        stats = analyze_device_in_profile(df, col, interval_minutes, days, device_cfg)
        
        if stats and stats['kwh_per_day'] > 0.01:  # Skip negligible devices
            check = check_device_against_targets(device_id, stats, targets)
            
            device_results.append({
                'device': clean_col,
                'device_id': device_id,
                'status': check['status'],
                'mean_W': stats['mean_W'],
                'kwh_per_day': stats['kwh_per_day'],
                'annual_kwh': stats['annual_kwh_estimate'],
                'runtime_h_per_day': stats['runtime_hours_per_day'],
                'uses_per_day': stats['activations_per_day'],
                'duty_cycle': stats['duty_cycle'],
                'issues': '; '.join(check['issues']) if check['issues'] else '',
                'suggestions': '; '.join(check['suggestions']) if check['suggestions'] else ''
            })
    
    # Sort by daily energy
    device_results.sort(key=lambda x: x['kwh_per_day'], reverse=True)
    
    # Print summary
    print(f"\nDevice Analysis (sorted by daily energy):")
    print(f"{'Device':<30} {'Status':<6} {'kWh/day':<10} {'Annual kWh':<12} {'Issues'}")
    print("-" * 120)
    
    for result in device_results:
        status_marker = {
            'PASS': '✓',
            'WARN': '⚠',
            'FAIL': '✗',
            'UNKNOWN': '?'
        }.get(result['status'], '?')
        
        print(f"{result['device']:<30} {status_marker:<6} {result['kwh_per_day']:<10.2f} "
              f"{result['annual_kwh']:<12.0f} {result['issues']}")
    
    # Count statuses
    status_counts = {}
    for r in device_results:
        status_counts[r['status']] = status_counts.get(r['status'], 0) + 1
    
    print(f"\n Summary: {status_counts.get('PASS', 0)} PASS, "
          f"{status_counts.get('WARN', 0)} WARN, "
          f"{status_counts.get('FAIL', 0)} FAIL")
    
    return device_results


def main():
    print("\n" + "="*70)
    print("SANITY REPORT - Device Calibration Validation")
    print("="*70)
    print("\nGenerating test profiles and checking against calibration targets...")
    
    # Load config once
    print("Loading configuration...")
    devices, households_config = load_config("config")
    
    households = ['single_professional', 'young_couple', 'family_with_children', 
                  'large_family', 'retired_couple']
    seasons = ['winter', 'summer']  # Check extremes
    
    all_results = []
    
    for household in households:
        for season in seasons:
            results = generate_sanity_report(household, season, devices=devices)
            all_results.extend(results)
    
    # Generate CSV report
    report_df = pd.DataFrame(all_results)
    report_file = 'sanity_report.csv'
    report_df.to_csv(report_file, index=False)
    
    print(f"\n{'='*70}")
    print(f"Full report saved to: {report_file}")
    print("="*70)
    
    # Print overall summary
    print("\nOVERALL SUMMARY:")
    total = len(all_results)
    passed = len([r for r in all_results if r['status'] == 'PASS'])
    warned = len([r for r in all_results if r['status'] == 'WARN'])
    failed = len([r for r in all_results if r['status'] == 'FAIL'])
    
    print(f"  Total device-season combinations analyzed: {total}")
    print(f"  ✓ PASS: {passed} ({passed/total*100:.1f}%)")
    print(f"  ⚠ WARN: {warned} ({warned/total*100:.1f}%)")
    print(f"  ✗ FAIL: {failed} ({failed/total*100:.1f}%)")
    
    if failed > 0:
        print("\n⚠ FAILURES DETECTED - Review sanity_report.csv for details")
        print("Common fixes:")
        print("  - HVAC/water heating: Reduce expected_uses_per_day or avg_duration_min")
        print("  - Cycling devices: Adjust cycle_on_min/cycle_off_min ratios")
        print("  - Short-use devices: Parameters OK, partial-interval accounting should fix")
    else:
        print("\n✓ All devices within acceptable ranges!")


if __name__ == '__main__':
    main()
