# Scenario 0 Thesis Section - Complete Contents

## File: SCENARIO_0_THESIS_SECTION.md (326 lines)

### Table of Contents

**1. Definition**
- Baseline scenario explanation
- Grid-only operation
- No optimization, no DER, no flexibility
- Reference cost C₀ = €39,563.85

**2. Data Preprocessing**
- 2.1 Input Data (4 family types, 10-min resolution)
- 2.2 Temporal Resampling (10-min → hourly)
- 2.3 Sanity Check (total power consistency)
- 2.4 Year Translation (2024 → 2025, leap year handling)
- 2.5 Validation (8,760 hours verified)

**3. TOU Pricing for 2025**
- 3.1 TOU Band Definitions (F1/F2/F3)
- 3.2 Italian National Holidays (11 days, all F3)
- 3.3 Fixed TOU Prices (F1: 0.1299, F2: 0.1205, F3: 0.1065 €/kWh)

**4. Baseline Cost Calculation**
- 4.1 Hourly Cost (equations)
- 4.2 Annual Baseline Cost per Family Type
- 4.3 Building-Wide Baseline Cost (20 apartments)
- 4.4 Band-Specific Breakdown (F1/F2/F3 split)

**5. What is Included vs Excluded**
- 5.1 Included (time-varying energy cost only)
- 5.2 Excluded (fixed charges, taxes, VAT - rationale provided)

**6. Results**
- 6.1 Annual Baseline Energy Consumption (4 families)
- 6.2 Building-Wide Totals (338,200 kWh, €39,563.85)
- 6.3 Energy Consumption by TOU Band (detailed tables)
- 6.4 Key Observations:
  - 6.4.1 Young Couples Highest Consumption (EV analysis with service breakdown table)
  - 6.4.2 Retired Couples: Lowest Total, Highest Peak Usage
  - 6.4.3 Family Types C and D Similar Patterns
  - 6.4.4 Off-Peak Consumption = Optimization Potential
- 6.5 Sanity Checks (validation confirmed)

**7. Baseline Cost for Optimization**
- Final boxed result: C₀ = €39,563.85
- Next steps preview (Scenario 1)

---

## Supporting Files Created

### Results Directory: `scenario0_results/`

1. **scenario0_family_summary.csv** - Annual kWh and € per family type
2. **scenario0_band_breakdown.csv** - Energy by F1/F2/F3 bands  
3. **scenario0_building_total.csv** - Building-wide totals
4. **scenario0_validation_week.png** - Week plot with demand + TOU bands
5. **seasonal_profiles.png** - 4 families × 4 seasons daily profiles
6. **monthly_comparison.png** - Monthly bar chart comparison
7. **service_breakdown_by_family.png** - Top 10 services per family

### Python Script

**scenario0_baseline.py** - Complete implementation:
- Data loading and preprocessing
- Leap year handling (2024→2025)
- TOU price mapping
- Cost calculations
- Validation plots generation

---

## Key Features

✅ **Complete equations** in LaTeX format  
✅ **All 4 family types** analyzed in detail  
✅ **EV charging impact** explained with service-level breakdown  
✅ **Seasonal patterns** documented with plots  
✅ **TOU pricing** fully integrated with Italian holidays  
✅ **Validation plots** showing correctness  
✅ **Ready to copy** directly into thesis document  

---

## How to Use

1. **Copy sections** from SCENARIO_0_THESIS_SECTION.md into thesis
2. **Embed plots** using paths in scenario0_results/
3. **Reference tables** from CSV files
4. **Cite baseline cost**: C₀ = €39,563.85 for all future scenarios

This is your **complete, thesis-ready Scenario 0 documentation**!
