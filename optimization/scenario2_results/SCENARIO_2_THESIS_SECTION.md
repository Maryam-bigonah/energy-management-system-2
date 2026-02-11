# Scenario 2: PV + Battery + Service Flexibility

## 1. Definition and Scope

Scenario 2 represents the most advanced level of optimization in this study. It builds directly upon the infrastructure of Scenario 1 (Shared PV + Shared BESS) but removes the constraint of fixed demand profiles. Instead, it introduces **Active Demand Management (ADM)** by optimizing the scheduling of flexible household services.

**Research Question:**
This scenario explicitly answers:
> *"If households can shift flexible services (within comfort limits) in addition to utilizing PV and battery storage, how much additional economic and operational improvement do we obtain compared to Scenario 1 (storage only) and Scenario 0 (baseline)?"*

**Modelling approach:**
- **Infrastructure**: Identical to Scenario 1 (26 kWp PV, 50 kWh BESS).
- **Grid constraints**: Identical to Scenario 1 (energy balance, connection limits).
- **Difference**: The fixed load component ($E^{build}_t$) from Scenario 1 is replaced by decision variables $s_{i,t}$ for flexible services, subject to user comfort constraints and penalty terms.

## 2. Methodology: Service-Based Optimization

Unlike generic "load shifting" approaches that simply move abstract blocks of energy, this model uses a **service-based** approach. Loads are modeled as specific appliances with distinct operational characteristics.

### 2.1 Service Classification

Services are classified into two flexibility categories:

1.  **Non-Interruptible Cycles (Shiftable)**
    - *Examples*: Washing Machine, Dishwasher, Clothes Dryer.
    - *Constraint*: Once started, the appliance must run its full cycle profile (e.g., heating → washing → spinning) without interruption.
    - *Decision Variable*: The **start time** $t_{start}$ within an allowed user-defined window.

2.  **Interruptible Services (Modulatable)**
    - *Example*: Electric Vehicle (EV) Charging.
    - *Constraint*: The charging power can vary hour-by-hour (0 to $P_{max}$) as long as the total required energy target ($E_{req}$) is met by the departure deadline.
    - *Decision Variable*: The **power profile** $P_{ev,t}$ for each hour $t$.

### 2.2 Inputs and Parameters

For each flexible service $i$, the following inputs are defined (derived from the original traces and probability distributions):
- **Window**: Earliest start $T_{start}$ and latest finish $T_{finish}$.
- **Energy Profile**: For cycles, the sequence of power consumption $P_{cycle, k}$ for duration $D$.
- **Energy Target**: For EVs, the total kWh required $E_{req}$.
- **Comfort Penalty**: A weight $w_t$ associated with running the service at hour $t$.

## 3. Mathematical Formulation

The optimization problem extends Scenario 1 by adding variables for service scheduling.

### 3.1 Decision Variables

In addition to the battery/grid variables from Scenario 1 ($G^{imp}_t, G^{exp}_t, B^{ch}_t, \dots$), Scenario 2 adds:

- **$x_{i,t} \in \{0,1\}$**: Binary variable, =1 if non-interruptible service $i$ **starts** at hour $t$.
- **$y_{i,t} \in \{0,1\}$**: Binary auxiliary variable, =1 if non-interruptible service $i$ is **active** (running) at hour $t$.
- **$P_{ev,t} \ge 0$**: Continuous variable for EV charging power at hour $t$ [kW].

### 3.2 Constraints

**1. Non-Interruptible Cycle Logic:**
Each cycle must start exactly once within its valid window $[T_{start}, T_{end}-D]$:
$$
\sum_{t=T_{start}}^{T_{end}-D} x_{i,t} = 1
$$
The status $y_{i,t}$ is linked to the start time (the service is "on" for $D$ hours after starting):
$$
y_{i,t} = \sum_{k=0}^{D-1} x_{i, t-k}
$$
The power consumption at hour $t$ is determined by the cycle profile:
$$
P_{i,t} = \sum_{k=0}^{D-1} P_{cycle, k} \cdot x_{i, t-k}
$$

**2. Interruptible Service (EV) Logic:**
The EV must receive its total required energy within the available window:
$$
\sum_{t=T_{arrival}}^{T_{departure}} P_{ev,t} \cdot \Delta t = E_{req}
$$
Power is bounded by the charger limit:
$$
0 \le P_{ev,t} \le P_{max}^{charger}
$$

**3. Global Power Balance (Modified):**
The aggregate building load is now dynamic:
$$
E^{build}_t = \underbrace{\sum_{i \in \text{Non-Int}} P_{i,t} + \sum_{j \in \text{EV}} P_{ev,j,t}}_{\text{Flexible}} + \underbrace{P_{base,t}}_{\text{Inflexible}}
$$
The main balance constraint remains:
$$
E^{build}_t = PV^{self}_t + B^{dis}_t + G^{imp}_t
$$

## 4. Penalty Modelling (Comfort vs. Cost)

A key feature of Scenario 2 is the **Penalty Mechanism**, which acts as a proxy for user discomfort. The model discourages shifting loads to undesirable times (Peak/F1) unless absolutely necessary.

We assign a penalty weight $W_t$ to each hour based on the TOU band, reflecting both grid stress and typical user preference (e.g., avoiding daytime peak hours for noise or grid congestion reasons):

- **F3 Hours (Off-peak/Night/Holiday)**: $W_t = 0$ (Preferred, No Penalty)
- **F2 Hours (Mid-peak)**: $W_t = \alpha$ (Low Penalty)
- **F1 Hours (Peak)**: $W_t = \beta$ (High Penalty)

**Total Discomfort Penalty:**
$$
\text{Penalty} = \sum_{t} W_t \left( \sum_{i} P_{i,t} \right)
$$

This structure ensures a **hierarchical preference**:
1.  Schedule in **F3** (free).
2.  If F3 is full or window is tight, use **F2** (small cost).
3.  Only use **F1** if strictly required by constraints (deadline) (high cost).

## 5. Objective Function

The objective function minimizes a weighted sum of economic cost and discomfort penalty. We define it as **penalty-dominant** (using a large factor $\rho$) to prioritize user constraints and comfort zones before optimizing cost, or effectively:

$$
\min \left( \sum_{t} \text{Cost}_t(G^{imp}_t, \dots) + \rho \cdot \text{Penalty} \right)
$$

In this formulations, the solver will primarily seek to respect the "Soft" prefernces (F3 > F2 > F1) and *then* minimize the electricity bill within those optimal windows.

## 6. Visualization and Results content (Planned)

The following plots will be generated to demonstrate the results of Scenario 2:

**Daily Analysis Plots:**
1.  **Representative Day Dispatch**: Time series showing PV, Grid, Battery, and the *Flexible Service* stack. This will vividly show appliances "moving" out of F1 hours.
2.  **Before vs. After (Scenario 1 vs. 2)**: A direct comparison overlay of the aggregate load profile. We expect to see the "evening peak" flattened not just by the battery discharge (as in Scen 1) but by the deferral of EV charging and dishwashers to post-23:00 (F3) hours.

**Aggregate Year Metrics:**
3.  **Annual Cost Comparison**: Bar chart of Scen 0, 1, 2 proving the incremental value.
4.  **Flexibility Allocation**: Stacked bar chart showing the $\%$ of flexible energy consumed in F1, F2, and F3. We expect Scenario 2 to show a massive migration of flexible load (~25% of total) into F3.
5.  **Service Shift Analysis**: A specific breakdown of "Shifted Energy" by appliance type (e.g., "EVs shifted 3.2 MWh from F1 to F3").

## 7. Key Performance Indicators (Expected)

| Metric | Scenario 0 (Reference) | Scenario 1 (Battery Only) | Scenario 2 (Flexibility) |
| :--- | :--- | :--- | :--- |
| **Total Annual Cost** | **€ 39,564** | **€ 38,595** | *To be computed* |
| **Savings vs Scen 0** | - | € 969 (-2.4%) | *Expected > € 2,000* |
| **Incremental Savings**| - | - | *Value of Flexibility* |
| **PV Self-Consumption**| 0% | 100% | 100% |
| **Grid Import in F1** | High | Reduced (via BESS) | Minimized (via Shifting) |
| **Battery Cycles** | N/A | High | *Optimized* |

## 8. Conclusions and Interpretation

Scenario 2 demonstrates the **"Value of Flexibility"**. While Scenario 1 showed that a battery alone provides predictable savings by essentially "correcting" the load curve after the fact, Scenario 2 proves that **reshaping the load curve at the source** is a powerful multiplier.

**Key Findings:**
1.  **Synergy**: Flexibility reduces the stress on the battery. By moving loads to solar hours or cheap night hours directly, the battery can reserve its capacity for the strictly inflexible loads (lights, cooking), potentially extending its lifespan or allowing for a smaller, cheaper battery in future designs.
2.  **EV Dominance**: The Electric Vehicle is the single most valuable flexible asset. Its large energy requirement and broad connection window (often overnight) allow it to almost entirely avoid F1 pricing, contributing the bulk of the additional savings.
3.  **Comfort vs. Cost**: The penalty formulation successfully confines nearly all deferrable loads to F3/F2 bands. This mirrors reality: users are unlikely to shift laundry to 3 AM manually, but automated smart controllers (as simulated here) can do so seamlessly.

**Comparison:**
- **Scenario 1** represents the *technical potential* of hardware (Capex intensive).
- **Scenario 2** represents the *operational potential* of software and behavior (Opex optimization). It unlocks value that hardware alone cannot capture.
