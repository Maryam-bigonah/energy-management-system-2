# Energy Management System

Residential energy management system with PV power forecasting, appliance load classification, and service event extraction for MPC-based optimization.

## Project Overview

This project develops an energy management system for residential households with photovoltaic (PV) systems and shiftable appliances. The system:

1. **Forecasts PV power generation** for 2025 using historical data and weather forecasts
2. **Classifies household appliances** by shiftability (non-shiftable, hard shiftable, soft shiftable)
3. **Extracts discrete service events** from load profiles for task-based MPC optimization
4. **Enables optimal scheduling** of appliances to maximize PV self-consumption or minimize costs

## Key Components

### 1. PV Power Forecasting (`pvgis_xgboost_regressor.py`)

XGBoost-based model for predicting PV power output:
- **Training data**: PVGIS historical timeseries (2018-2023)
- **Features**: POA irradiance components (beam, diffuse, ground-reflected), sun elevation, temperature, wind speed
- **Validation**: Train on 2018-2021, validate on 2022, test on 2023
- **Output**: Hourly PV power predictions for 2025

**Metrics (2023 test set, daylight hours)**:
- R²: 0.95+
- RMSE: ~4-5 kW
- MAE: ~2-3 kW

### 2. Appliance Classification (`appliance_shiftability_classification.md`)

Classification of 42 household appliances based on shiftability for demand response:

- **Non-shiftable (35 devices)**: Lighting, cooking appliances, entertainment, IT equipment, continuous cooling, personal care
- **Hard shiftable (6 devices)**: Dishwasher, washing machine, dryer, vacuum robot, garden tools
- **Soft shiftable (1 device)**: Electric vehicle charging (not in current dataset)

### 3. Load Profile Separation (`separate_devices_by_shiftability.py`)

Categorizes device profiles from CSV data:
- **Output files**:
  - `load_profile_non_shiftable.csv`: Non-shiftable appliance data
  - `load_profile_hard_shiftable.csv`: Hard shiftable appliance data
  - `load_profile_aggregated.csv`: Total load by category

**Results (annual totals)**:
- Non-shiftable: 2,241 kWh (76.2%)
- Hard shiftable: 699 kWh (23.8%)

### 4. Service Event Extraction (`extract_service_events.py`)

Converts load profiles into discrete service events for task-based optimization:
- **Detection method**: ON/OFF transitions with device-specific energy thresholds
- **Output**: `services_events.csv` with 514 service events

**Filtering thresholds**:
- Dishwasher: ≥0.5 kWh
- Washing machine: ≥0.4 kWh
- Dryer: ≥1.0 kWh
- Vacuum robot: ≥0.05 kWh

**Event statistics**:
- Dishwasher: 195 events, ~0.98 kWh avg, 2.9h duration
- Washing machine: 132 events, ~0.82 kWh avg, 3.1h duration
- Dryer: 140 events, ~1.78 kWh avg, 3.1h duration
- Vacuum robot: 47 events, ~0.09 kWh avg, 3.4h duration

## Data Files

### Input Data
- `Timeseries_45.044_7.639_E5_83kWp_crystSi_14_42deg_-3deg_2018_2023.csv`: PVGIS historical PV data (2018-2023)
- `open-meteo-45.10N7.70E239m.csv`: Open-Meteo weather forecast (2025)
- `DeviceProfiles_Couple both at Work.Electricity.csv`: Household appliance load profiles (2016)

### Output Data
- `pv_prediction_2025.csv`: Hourly PV power predictions for 2025
- `services_events.csv`: Discrete service events for optimization
- `load_profile_aggregated.csv`: Aggregated load profiles by shiftability category
- `device_category_mapping.txt`: Device-to-category mapping

## System Specifications

**PV System**:
- Capacity: 83.2 kWp crystalline silicon
- Tilt: 42° (optimum)
- Azimuth: -3° (optimum)
- System losses: 14%
- Location: 45.044°N, 7.639°E (Turin, Italy area)

**Household Profile**:
- Type: Couple both at work
- Annual consumption: 2,940 kWh
- Peak load: 4.8 kW

## Usage

### PV Power Forecasting

```bash
python3 pvgis_xgboost_regressor.py \
  --pvgis_csv Timeseries_45.044_7.639_E5_83kWp_crystSi_14_42deg_-3deg_2018_2023.csv \
  --meteo_csv open-meteo-45.10N7.70E239m.csv \
  --out_csv pv_prediction_2025.csv \
  --albedo 0.2
```

### Device Load Separation

```bash
python3 separate_devices_by_shiftability.py
```

### Service Event Extraction

```bash
python3 extract_service_events.py
```

## MPC Integration

The service events dataset enables task-based MPC optimization:

1. **Load services file**: Each row represents one discrete task (appliance cycle)
2. **Define scheduling windows**: Time constraints per device type
3. **Optimize start times**: Align with PV generation or tariff structure
4. **Apply constraints**:
   - Fixed duration per task
   - No overlapping tasks for same appliance
   - Sequencing (e.g., dryer after washing machine)
   - Deadline compliance (24h completion window)

**Objective functions**:
- Maximize PV self-consumption
- Minimize electricity cost (align with F3 tariff)
- Valley-filling (schedule during low demand periods)

## Dependencies

```
pandas
numpy
scikit-learn
xgboost>=3.0
matplotlib
```

## License

MIT

## Author

Maryam Bigonah
