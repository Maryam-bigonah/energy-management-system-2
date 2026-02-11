# Scenario 2: Service Flexibility Optimization

This directory contains the implementation of **Scenario 2**, which extends Scenario 1 by adding **Active Demand Management** through service scheduling optimization.

## Files

- `scenario2_model.py` - Main MILP optimization model with shiftable services
- `SCENARIO_2_THESIS_SECTION.md` - Thesis documentation
- `scenario2_hourly_dispatch.csv` - Hourly results (generated after running)
- `scenario2_summary_metrics.csv` - Summary KPIs (generated after running)
- `plots/` - Visualization outputs

## Key Features

### 1. Non-Interruptible Cycles (MILP with Binary Variables)
Appliances with complete cycles that must run uninterrupted once started:
- **Washing Machine**: 2h cycles, 6 AM - 10 PM window
- **Dishwasher**: 2h cycles, 7 AM - 11 PM window  
- **Dryer**: 1.5h cycles, 8 AM - 10 PM window

**Implementation**: Binary start-time variables `y[t]` enforce that each cycle:
- Starts exactly once within its allowed window
- Runs for its full duration without interruption
- Cannot start if there isn't enough time before the window closes

### 2. EV Charging Flexibility (Daily Windows)
Interruptible service with daily constraints:
- 7 kW max charging power
- 11.2 kWh daily energy requirement per session
- **Daily arrival/departure windows**: Each day has its own EV session with specific arrival (6 PM) and departure (7 AM next day) times
- Can only charge within the daily window (enforced at 0 outside window)
- Must meet exact daily energy target

### 3. Penalty Mechanism (Comfort-Dominant)
Comfort-based penalties discourage scheduling during expensive/inconvenient hours:
- **F1 (Peak)**: 1.0 €/kWh penalty (High)
- **F2 (Mid-peak)**: 0.3 €/kWh penalty (Medium)
- **F3 (Off-peak)**: 0.0 €/kWh penalty (None)

The objective function uses a penalty multiplier `RHO = 100` to make comfort dominant over cost, ensuring services shift to F3 unless deadlines force otherwise.

### 4. Integration with PV + BESS (Reused from Scenario 1)
All battery and PV constraints from Scenario 1 are preserved:
- 20 kWh battery capacity
- 5 kW power limit
- PV-only charging (no grid charging)
- 90% round-trip efficiency
- SOC bounds and end-of-year SOC constraint

## How to Run

```bash
cd optimization/scenario2_results
python scenario2_model.py
```

**Requirements:**
- Python 3.8+
- cvxpy
- pandas
- numpy
- matplotlib

Install dependencies:
```bash
pip install cvxpy pandas numpy matplotlib
```

## Expected Outputs

1. **Cost Savings**: Additional savings beyond Scenario 1 from load shifting
2. **Flexible Energy Distribution**: 
   - High % in F3 (off-peak)
   - Minimal in F1 (peak)
3. **Service Schedules**: Optimized start times for appliances and EV charging profiles
4. **Visualizations**: 
   - Weekly load profiles showing service shifting
   - Flexible load breakdown by time of day
   - Before/after comparison with Scenario 1

## Comparison to Scenario 1

| Aspect | Scenario 1 | Scenario 2 |
|--------|-----------|-----------|
| **Demand Profile** | Fixed | Flexible (optimized) |
| **Service Timing** | Historical | Optimally shifted with binary start times |
| **Optimization Type** | LP (Linear Program) | **MILP (Mixed-Integer LP)** |
| **Decision Variables** | Grid + Battery (continuous) | Grid + Battery + Binary cycle starts + EV power (mixed) |
| **Constraints** | Energy balance only | Energy balance + Cycle completion + Daily EV windows |
| **Value Proposition** | Hardware (BESS) | Software (Smart scheduling) |

## Implementation Details

### MILP Complexity
- **Binary Variables**: One per cycle per valid start time (365 cycles × ~15 hour window = ~5,500 binaries)
- **EV Variables**: Continuous power variables for each hour within daily windows
- **Solver**: CBC or GLPK_MI (specialized MILP solvers required)
- **Solution Time**: Minutes to hours depending on problem size and solver

### Cycle Start Logic
For each appliance cycle, the model creates binary variables `y[t]` for each valid start time. The constraints ensure:
```
Σ y[t] = 1  (must start exactly once)
Load[t] = Σ (power × y[t-k]) for k=0..duration  (load during cycle)
```

### Daily EV Windows
Each day creates a separate EV session with:
- Local power variables `EV_power[arrival:departure]`
- Energy constraint: `Σ EV_power[arrival:departure] = daily_target`
- Window enforcement: Power = 0 outside [arrival, departure]

## Notes

- This is a **true MILP** with binary start-time variables for non-interruptible cycles
- **Daily EV constraints** enforce arrival/departure windows and energy targets per day
- Perfect foresight (deterministic optimization) is assumed
- Solver performance depends on number of cycles and EV sessions
- Penalty weights can be tuned to adjust comfort vs. cost tradeoff

## Computing the Value of Flexibility (ΔC_flex)

After running both Scenario 1 and Scenario 2, use the comparison script to compute the key thesis metric:

```bash
python compare_scenarios.py
```

This computes:
- **ΔC_flex = C1b - C2** (Extra savings from flexibility, €/year)
- **%FlexGain = (ΔC_flex / C1b) × 100** (Extra savings as %)

Where:
- **C1b** = Scenario 1 operational cost (PV+BESS with fixed services)
- **C2** = Scenario 2 operational cost (PV+BESS with optimized service shifting)

**Important**: Operational cost = `Σ p(t) · GridImport(t)` (grid cost only, NOT including penalty)

The script also reports:
- **Comfort Penalty**: Separate metric for service discomfort/inconvenience
- **Flexible energy distribution**: % of flexible services in F1/F2/F3 bands
- **Grid import reduction**: How much less energy is imported from the grid

This answers: *"How much extra cost reduction do we get by allowing service shifting (within comfort limits), on top of what PV+BESS alone provides?"*
