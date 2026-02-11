# Scenario 1: Shared PV + Battery (Optimization)

## 1. System Context and Definition

Scenario 1 introduces a centralized **Shared Photovoltaic (PV)** system and a **Shared Battery Energy Storage System (BESS)** to the 20-apartment building. The primary objective is to minimize the total annual cost (operational energy procurement + annualized investment) by optimizing the battery dispatch operation.

**Key constraints for Scenario 1:**
- **Fixed Service Demand**: No load shifting or variation in user behavior (services $E_{h,t}$ remain identical to Scenario 0).
- **No Fuel Switching**: All heating/cooking remains electric.
- **PV-Only Charging**: The battery charges exclusively from the PV system to maximize self-consumption (Green Charging).
- **Shared Assets**: The PV and BESS are building-level assets; costs and savings are allocated financially to the 20 apartments.

### 1.1 System Sizing
Based on the available rooftop area and preliminary analysis (peak generation consistent with modeled layout):
- **PV System Capacity ($P_{PV}^{nom}$)**: **26 kWp**
- **Battery Capacity ($E_{BESS}^{max}$)**: **50 kWh**
- **Battery Power ($P_{BESS}^{max}$)**: **25 kW**

## 2. Inputs and Methodology

### 2.1 Inputs (2025 Hourly)
The optimization is performed over a full year ($T = 8,760$ hours) using the following hourly profiles:

1.  **Building Load ($E^{build}_t$)**: The aggregation of all 20 apartment profiles (Scenario 0):
    $$
    E^{build}_t = \sum_{h \in \{A,B,C,D\}} 5 \times E_{h,t} \quad [\text{kWh}]
    $$
2.  **PV Generation Forecast ($PV^{fc}_t$)**: Hourly availability profile for 2025, generated via PVGIS/Open-Meteo simulations for Turin. This is treated as a deterministic input (perfect forecast) for the optimization.
3.  **Electricity Prices ($p_t$)**: The same 2025 ARERA Time-of-Use (TOU) tariffs used in Scenario 0.
4.  **Export Price ($r^{exp}_t$)**: Set to **0 €/kWh** to strictly value self-consumption and simplify the economic boundary (excess is curtailed or exported without revenue in this conservative baseline).

### 2.2 Optimization Decisions and Constraints
The problem is formulated as a linear program (LP) to solve the optimal dispatch schedule.

**Decision Variables (for each hour $t$):**
- $G^{imp}_t$: Grid import [kWh]
- $G^{exp}_t$: Grid export [kWh]
- $PV^{self}_t, PV^{ch}_t$: PV used for load / charging [kWh]
- $B^{ch}_t, B^{dis}_t$: Battery charge / discharge [kWh]
- $SOC_t$: State of Charge [kWh]

**Objective Function:**
Minimize Total Annual Operational Cost with a penalty term for battery degradation (Method B):
$$
\min \sum_{t=1}^{T} \left( p_t G^{imp}_t - r^{exp}_t G^{exp}_t + \epsilon (B^{ch}_t + B^{dis}_t) \right)
$$
where $\epsilon = 10^{-6}$ €/kWh is a small penalty to prevent simultaneous charging/discharging and unnecessary cycling (Method B).

**Constraints:**
1.  **PV Energy Balance**:
    $$ PV^{fc}_t = PV^{self}_t + PV^{ch}_t + G^{exp}_t $$
2.  **Building Demand Balance**:
    $$ E^{build}_t = PV^{self}_t + B^{dis}_t + G^{imp}_t $$
3.  **Battery Charging Source**:
    $$ B^{ch}_t = PV^{ch}_t \quad (\text{PV-only charging enforcement}) $$
4.  **SOC Dynamics**:
    $$ SOC_t = SOC_{t-1} + \eta_{ch} B^{ch}_t - \frac{1}{\eta_{dis}} B^{dis}_t $$
5.  **Physical Limits**:
    $$ SOC^{min} \le SOC_t \le SOC^{max} $$
    $$ 0 \le B^{ch}_t, B^{dis}_t \le P_{BESS}^{max} $$
6.  **Cyclic Condition**:
    $$ SOC_{8760} \ge SOC_0 $$

### 2.3 Economic Evaluation (CAPEX)
To determine feasibility, the annualized investment cost is added to the operational cost:
$$
C_{1,total} = C_{1}^{op} + C_{PV}^{ann} + C_{BESS}^{ann}
$$
where annualized CAPEX is calculated using a discount rate $r=3\%$:
- **PV**: 1200 €/kWp, 20-year lifetime.
- **Battery**: 400 €/kWh, 10-year lifetime.

## 3. Results and outputs

### 3.1 Building-Level Performance
The optimized dispatch yields significant operational savings, though the net economic benefit is constrained by the investment costs.

| Metric | Scenario 0 (Baseline)* | Scenario 1 (PV + BESS) | Difference |
|--------|-----------------------|------------------------|------------|
| **Operational Cost** | **€ 39,537.39** | **€ 34,153.00** | **- € 5,384.39** |
| Total Import | 338,234 kWh | 299,642 kWh | - 38,592 kWh |
| Self-Consumption | 0% | **100.0%** | + 100% |
| **Annualized CAPEX** | € 0.00 | € 4,441.74 | + € 4,441.74 |
| **Total Annual Cost** | **€ 39,537.39** | **€ 38,594.74** | **- € 942.65** |

*\*Note: Baseline cost re-evaluated for exact load profile (338,234 kWh) generated with month-day mapping logic.*

**Key Insight**: The system achieves **100% PV self-consumption**. This is due to the modest PV size (26 kWp) relative to the building's aggregate demand (~38 kW average), ensuring all generated energy is immediately consumed or stored for evening use without generating excess for export.

### 3.2 Allocation to Family Types
Costs are allocated using an **Equal Share** principle ($1/20$ per apartment) as a representative financial settlement rule for shared assets.

**Annual Cost per Apartment (Allocated):**
| Family Type | Baseline Cost ($C_0$) | Scenario 1 Cost ($C_1$) | Annual Savings |
|-------------|-----------------------|-------------------------|----------------|
| **Allocated Share** | (varies by type) | **€ 1,929.74** | (varies) |
| **Average** | € 1,976.87 | € 1,929.74 | **€ 47.13** |

*(Note: While the absolute cost $C_1$ is shared equally here for reporting, in practice, savings might be distributed based on millesimi or actual consumption contribution. The optimization secures a small but positive net saving of ~€47/year per apartment after paying for the infrastructure.)*

### 3.3 System Operation Visualization

**Typical Dispatch (June Week)**:
The following plot illustrates the battery activity during a representative summer week. The battery charges (green bars) during peak solar hours and discharges (red bars) during the evening peak (F1/F2 boundary), effectively flattening the net grid import.

![Scenario 1 Validation Plot](file:///Users/mariabigonah/Desktop/thesis/anti/pv/optimization/scenario1_results/scenario1_validation_plot.png)

**State of Charge (SOC)**:
The TOU-aware optimization prioritizes discharging during high-price hours (F1/F2) to maximize value, adhering to the physical limits of the 50 kWh capacity.

---

## 4. Conclusion for Scenario 1
The optimization confirms that a **shared 26 kWp PV + 50 kWh BESS** system is technically feasible and economically positive ($+€943/year$ net benefit) under 2025 prices. The 100% self-consumption rate validates the sizing for a self-sufficiency focused operation, though simpler rules or larger sizing might impact these margins. This sets the stage for **Scenario 2**, where demand flexibility will be introduced to further improve these economics.
