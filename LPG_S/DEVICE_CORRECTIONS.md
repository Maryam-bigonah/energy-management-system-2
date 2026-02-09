# Device Power Corrections - Summary

## Changes Made

Based on your feedback, I've corrected the standby power consumption for devices that should be **completely off (0W)** when not in use.

### Fixed Devices

**Major Appliances** (now 0W when off):
- Washing Machine: 3W → **0W**
- Clothes Dryer: 3W → **0W**  
- Dishwasher: 3W → **0W**

**Small Appliances** (now 0W when unplugged):
- Coffee Maker: 2W → **0W**
- Microwave: 3W → **0W**

**HVAC** (seasonal corrections):
- Central AC: 15W → **0W** (completely off in winter with seasonal_factor=0.0)

### Devices That Still Have Standby Power (Realistic)

These devices typically remain plugged in and consume vampire power:

**Electronics:**
- TV (Living Room): 2W standby
- TV (Bedroom): 2W standby
- Gaming Console: 10W standby (realistic for PlayStation/Xbox)
- Stereo System: 5W standby
- Desktop Computer: 5W standby
- Laptop: 2W standby (when charging)
- Printer: 5W standby
- EV Charger: 5W standby

**Always-On Devices:**
- WiFi Router: 8-12W (continuous)
- Security System: 5-8W (continuous)
- Refrigerator: 150-200W (cycling compressor)
- Freezer: 100-150W (cycling compressor)

**HVAC with Standby:**
- Heat Pump: 20W (thermostat control)
- Water Heater: 50W (maintaining temperature)
- Boiler: 30W (pilot/control system)

### Seasonal Behavior Verification

**Winter (Dec-Feb):**
- ✓ Central AC: **0W** (not running, seasonal_factor = 0.0)
- ✓ Space Heater: **High usage** (seasonal_factor = 2.5)
- ✓ Heat Pump: **High usage** (seasonal_factor = 1.8)
- ✓ Boiler: **High usage** (seasonal_factor = 2.0)

**Summer (Jun-Aug):**
- ✓ Central AC: **High usage** (seasonal_factor = 2.0)
- ✓ Space Heater: **0W** (not running, seasonal_factor = 0.0)
- ✓ Heat Pump: **Moderate** (cooling mode, seasonal_factor = 1.5)
- ✓ Boiler: **Low usage** (hot water only, seasonal_factor = 0.3)

**Spring/Fall:**
- Moderate HVAC usage
- Reduced heating/cooling needs

## Updated Results (2024 Full Year)

### 1. Single Professional
- Daily: 36.5 kWh/day (slightly reduced from 36.7 due to removed standby power)
- Top consumer: Heat Pump (59%)

### 2. Young Couple  
- Daily: 47.2 kWh/day
- Top consumer: Heat Pump (56%)

### 3. Family with Children
- Daily: **53.5 kWh/day** (down from 53.9)
- Top: Heat Pump (52.1%), Heater (15.2%), Boiler (14.4%)

### 4. Large Family
- Daily: **59.6 kWh/day** (down from 60.0)
- Top: Heat Pump (48%), Boiler (17%), Heater (15.4%)

### 5. Retired Couple
- Daily: **58.6 kWh/day** (down from 58.9)
- Top: Heat Pump (58.4%), Boiler (16.6%)

## Validation

I verified sample data showing:
- ✅ Coffee maker: **0.0W** at 3 AM (not in use)
- ✅ Washing machine: **0.0W** when not running
- ✅ Dryer: **0.0W** when not running
- ✅ Vacuum: **0.0W** when not in use
- ✅ Hair dryer: **0.0W** when not in use
- ✅ Gaming console: **~10W** when in standby (realistic)

## Impact

The corrections removed approximately **0.3-0.4 kWh/day** of vampire power from major appliances, resulting in:
- More realistic baseline consumption
- Clearer distinction between active and inactive periods
- Better seasonal HVAC patterns (AC truly off in winter)

All profiles have been regenerated with these corrections!
