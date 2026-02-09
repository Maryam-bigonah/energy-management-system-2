# Synthetic Household Load Profile Generator

**Production-ready, calibrated, and resolution-independent load profile generator for household electricity consumption.**

## 📊 Output Format

CSV with columns:
- `timestamp`: Datetime index
- `total_power_W`: **Average Power** (Watts) over the interval
- `<Device>_W`: **Average Power** (Watts) for each device

 **Important**:
 - values represent the *average power* during the interval, not instantaneous peak.
 - To calculate Energy (kWh):
   `Energy_kWh = Power_W * (Interval_Minutes / 60) / 1000`
 - This ensures energy is conserved regardless of resolution (1 min vs 60 min).

## ✨ Key Features

- **YAML Configuration**: All devices and households defined in editable YAML files (`config/`).
- **Physics-Based Modeling**: Partial-interval energy accounting ensuring 1-minute vs 10-minute resolution results match within <1%.
- **35+ Calibrated Devices**: From cycling fridges (with correct duty cycles) to EV chargers and HVAC.
- **Advanced Cycling Logic**: Realistic compressor behavior for Fridges, Freezers, Heat Pumps, and Water Heaters.
- **Seasonal Intelligence**: HVAC and seasonal devices automatically adjust based on season (Winter/Spring/Summer/Fall).
- **Automated Calibration**: Built-in `sanity_report.py` validates every device against realistic energy targets.
- **5 Household Types**: Calibrated profiles for Single Professional, Couples, Families, and Retirees.
- **Reproducible**: Seeded RNG for identical results across runs.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate all households (1 week example)
python run_generator.py --all --start 2024-01-01 --end 2024-01-08

# Generate specific household
python run_generator.py --household family_with_children --year 2024

# Calibration & Sanity Check (Validate device parameters)
python sanity_report.py

# Validate Output Consistency (Resolution tests)
python validate_outputs.py

# Visualize Results
python visualize.py output/family_with_children_load_profile.csv
```

## 📁 Project Structure

```
config/
├── devices.yaml          # Device definitions (power, pattern, cycles)
├── households.yaml       # Household profiles (inventory, schedules)
└── calibration_targets.yaml # Targets for sanity checking

config_loader.py          # YAML loader with validation
behavior.py               # Behavioral models & cycling logic
generator.py              # Main simulation engine (Physics-based)
run_generator.py          # CLI interface
sanity_report.py          # Calibration validation tool
validate_outputs.py       # Consistency test suite
visualize.py              # Plotting utility

output/                   # Generated CSV files
plots/                    # Visualization output
```

## 🔧 Configuration

### Devices (`config/devices.yaml`)

Define devices with precise physics parameters:

```yaml
refrigerator:
  id: refrigerator
  category: always_on
  pattern: cycling              # cycling, continuous, daily, weekly, seasonal
  standby_power_w: 3
  active_power_w: 180
  peak_power_w: 220
  cycle_on_min: 20              # 33% duty cycle (20m ON, 40m OFF)
  cycle_off_min: 40
  seasonal_factor:
    winter: 0.9
    summer: 1.2
```

### Households (`config/households.yaml`)

Define household inventory and schedules:

```yaml
family_with_children:
  devices:
    refrigerator: 1
    heat_pump: 1
    water_heater: 1
  weekday_schedule:
    - start_hour: 17
      end_hour: 21
      occupancy_prob: 0.98
```

## ✅ Validation & Calibration

The system includes a robust validation suite:

1.  **Sanity Report** (`sanity_report.py`):
    - Generates 1-week profiles for Winter/Summer.
    - Compares every device's Avg Power, Daily kWh, and Runtime against `calibration_targets.yaml`.
    - Reports PASS/WARN/FAIL (currently 100% PASS/WARN).

2.  **Resolution Independence** (`validate_outputs.py`):
    - Ensures 1-minute, 10-minute, and 30-minute simulations produce consistent total energy (<1% difference).
    - Validates correct cycling behavior (transitions) and seasonal logic.

## 📈 Generated Profiles (Calibrated)

| Household | Occupants | Daily Avg (kWh) | Top Consumer |
|-----------|-----------|-----------------|--------------|
| Single Professional | 1 | ~52 | Heat Pump (82%) |
| Young Couple | 2 | ~51 | Heat Pump (83%) |
| Family with Children | 4 | ~50 | Heat Pump (85%) |
| Large Family | 6 | ~65 | Heat Pump (75%) |
| Retired Couple | 2 | ~51 | Heat Pump (83%) |

*Note: Heat Pump consumption dominates in Winter/Summer peaks. Milder seasons will show lower usage.*

## 🔬 Technical Details

### Partial-Interval Accounting
The generator calculates exact energy usage even when devices switch mid-interval.
- If a 1000W device runs for 3 minutes in a 10-minute interval:
  `Energy = 1000W * (3/60)h = 50 Wh`
  `Avg Power = 50 Wh / (10/60)h = 300 W`
- This ensures 1-minute and 10-minute resolutions match perfectly.

### Deterministic Cycling
Thermostatic devices (Fridge, HVAC, Water Heater) use deterministic duty cycles based on time offsets. This guarantees that a device turned ON at 10:05 in a 1-minute simulation is also accounted for correctly in a 10-minute simulation covering 10:00-10:10.

## 📄 License

Open source - use freely for research or commercial applications.
