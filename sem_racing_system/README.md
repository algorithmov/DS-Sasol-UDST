# SEM Racing System

Rebuilt around six objectives, and only these six -- no dashboard, no
driver-training module, deliberately out of scope for this build:

1. **Telemetry processing & analytics** -> `sem_racing.telemetry`
2. **Race strategy simulation** (energy model, route profile, PV vs.
   consumption optimisation) + **pre-race target speed planning** ->
   `sem_racing.strategy`
3. **Performance predictions** (motor efficiency maps, resistance) ->
   `sem_racing.performance`
4. **After-testing analysis** (coastdown, regression analysis) -- and the
   thing that makes "digital twin" an earned label, not a claim: per
   -subsystem validation against your own real lab/bench data ->
   `sem_racing.testing`
5. **Route/weather optimisation** (Sasol routes require variable strategy)
   -> `sem_racing.strategy.wind` + `sem_racing.strategy.stage_planner`

## Repository layout

```
sem_racing/
  telemetry/          # Pillar 1
    ingest.py          # raw OBC log / CSV export -> unified DataFrame
    laps.py             # repeating-circuit lap segmentation (GPS line-crossing)
    route.py            # point-to-point stage handling + route geometry (heading,
                         #   curvature-derived corner speed limits)
    features.py         # power, acceleration, consumption, moving averages

  strategy/            # Pillar 2 + Pillar 5
    trackmodel.py       # track speed envelope derived from real historical laps
    carmodel.py         # PhysicalCarModel (legacy), DirectDriveMotorModel (real
                         #   Mitsuba M2096D-III), AICarModel -- one shared interface
    solar.py            # array output, temperature derating, per-section energy income
    wind.py              # headwind/tailwind resolved from real route heading + a
                         #   wind forecast -- "variable strategy" made literal
    battery.py           # state-of-charge tracking + feasibility check
    optimize.py          # graph construction + DAG shortest path (negative-weight safe)
    stage_planner.py     # multi-stage orchestrator: each day its own route/weather,
                          #   battery SoC carried forward -- Sasol's actual structure

  performance/          # Pillar 3
    motor_map.py         # REAL interpolated efficiency map (not a single guessed
                          #   curve), with measured vs. fallback clearly distinguished
    resistance.py         # rolling/aero/gradient resistance exposed as standalone,
                          #   reportable quantities

  testing/               # Pillar 4
    coastdown.py           # 3-term SAE J2263-style road-load regression: fits real
                            #   Crr/Cd from a logged coastdown run
    validation.py           # compares every simulated subsystem against YOUR real
                             #   lab/bench measurements -- see below

tests/
  test_pipeline.py         # 38 tests: real sample telemetry + synthetic ground-truth
                            #   physics (used exactly like the coastdown test's
                            #   simulated run -- to prove correctness against known
                            #   truth, not just "it runs without crashing")
data/                       # bundled sample telemetry
```


| Subsystem | Function | What you feed it | What it needs from your lab test |
|---|---|---|---|
| Motor + resistance, combined, from real driving | `validate_energy_model(car_model, sections_df)` | `sections_df` from `carmodel.aggregate_sections()` on real telemetry | Nothing extra -- reuses your own logged laps |
| Motor, raw bench points | `validate_motor_bench(efficiency_map, bench_df)` | `bench_df` with `rpm, torque_nm, voltage_v, current_a` | A dyno/bench log at a few operating points -- actual efficiency is computed from these directly (mech power / elec power), not pre-supplied |
| Battery capacity | `validate_battery_capacity(battery_spec, discharge_df, cutoff_voltage)` | `discharge_df` with `time_s, voltage_v, current_a` | A standard constant-current discharge test to your pack's cutoff voltage |
| Resistance (Crr, Cd) | `validate_resistance(constants, held_out_df, mass_kg)` | Same shape as a coastdown log | A SECOND coastdown run, different from whichever one you fit `constants` on |
| Solar array | `validate_solar_array(array, lab_df)` | `lab_df` with `irradiance_w_m2, measured_power_w[, cell_temp_c]` | An outdoor panel measurement at known irradiance |

 Call
`validation.print_report(*results)` for a one-line-per-subsystem summary


## Running it

```bash
pip install -r requirements.txt
pytest tests/ -v     # 38 tests: real sample telemetry + synthetic known-truth physics
```


  links spanning 40-70Ah at 48-72V. These are genuinely different chemistries
  and voltages -- different discharge curves, different cell counts, different
  max C-rates. `battery.BatterySpec` still defaults to the 60V/50Ah/NMC
  figures from the direct conversation; **confirm which is actually final**
  before trusting any battery-related output.
