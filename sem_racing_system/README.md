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

## Validating the digital twin against real tests (Pillar 4)

This is the part that turns "digital twin" from a name into something
earned: every simulated subsystem has an independent `validate_*` function
in `testing/validation.py` that takes YOUR real measurements and reports
real error statistics (MAE, RMSE, MAPE, R², bias) -- not just "does it run."

**Important distinction: validation is not calibration.** `testing.coastdown`
*fits* new Crr/Cd from data. Everything in `testing.validation` *predicts*
with whatever coefficients you already have, then reports how wrong it was.
If you fit and validate on the same run, you're measuring how well the
model fits its own training data, not whether it generalizes -- always use
a held-out run you didn't calibrate on for anything you call a validation.

| Subsystem | Function | What you feed it | What it needs from your lab test |
|---|---|---|---|
| Motor + resistance, combined, from real driving | `validate_energy_model(car_model, sections_df)` | `sections_df` from `carmodel.aggregate_sections()` on real telemetry | Nothing extra -- reuses your own logged laps |
| Motor, raw bench points | `validate_motor_bench(efficiency_map, bench_df)` | `bench_df` with `rpm, torque_nm, voltage_v, current_a` | A dyno/bench log at a few operating points -- actual efficiency is computed from these directly (mech power / elec power), not pre-supplied |
| Battery capacity | `validate_battery_capacity(battery_spec, discharge_df, cutoff_voltage)` | `discharge_df` with `time_s, voltage_v, current_a` | A standard constant-current discharge test to your pack's cutoff voltage |
| Resistance (Crr, Cd) | `validate_resistance(constants, held_out_df, mass_kg)` | Same shape as a coastdown log | A SECOND coastdown run, different from whichever one you fit `constants` on |
| Solar array | `validate_solar_array(array, lab_df)` | `lab_df` with `irradiance_w_m2, measured_power_w[, cell_temp_c]` | An outdoor panel measurement at known irradiance |

Each one is independent -- run just the motor validation against your dyno
session without needing a battery test done yet, or vice versa. Call
`validation.print_report(*results)` for a one-line-per-subsystem summary
once you've run whichever ones you have data for.

## Design decisions worth knowing about

- **One ingest path, two file formats.** `ingest.load()` auto-detects raw
  semicolon-delimited OBC logs vs. already-processed CSV exports.
- **`laps.ensure_laps()` is idempotent** for repeating circuits; `route.py`
  handles point-to-point stages separately, since `laps.py`'s line-crossing
  detection has no meaning on a road you only drive once.
- **`trackmodel.build_track_config()`** derives a speed envelope from real
  historical laps instead of hand-typed waypoints; **`route.build_stage_config()`**
  does the equivalent from pure route geometry (curvature-derived corner
  limits, `v_max = sqrt(mu*g*R)`) when no historical laps exist yet at all.
- **Rolling starts, not just standing starts** -- `optimize.py` searches
  from/to whichever speeds are actually feasible at each end rather than
  assuming a dead stop.
- **Car model is a swappable interface.** `PhysicalCarModel` (legacy,
  geared brushed-DC bootcamp placeholder), `DirectDriveMotorModel` (the
  club's real Mitsuba M2096D-III, no gearbox, real power/efficiency
  ceiling), and `AICarModel` (trained regressor) all implement
  `.predict(gradient_angle, v1, v2, section_length, headwind_ms=0.0)`.
- **Motor efficiency is a real, swappable map (`performance.motor_map`)**,
  not a single hardcoded curve. `from_datasheet_default()` is explicitly
  flagged `is_measured=False` -- an approximation anchored on one published
  point, not data. `from_measurements()` builds a real map (1D rpm-only or
  full 2D rpm+torque) from dyno points the moment you have them.
- **Wind exists now (`strategy.wind`), and it's route-dependent.** The same
  wind forecast produces a headwind on one section and a tailwind further
  along just because the road turned -- resolved via the route's real
  heading, not a single flat assumption.
- **Multi-stage variable strategy is a real orchestrator
  (`strategy.stage_planner`)**, not a manual loop -- each stage gets its
  own route/weather, and battery state of charge carries forward
  realistically (or resets, if you explicitly tell it your event guarantees
  overnight recharging -- see `recharge_between_stages_to`'s docstring for
  why that default is deliberately conservative, not assumed).
- **Coastdown regression matches the real SAE J2263 methodology**: a
  3-term road-load fit (`A + B*v + C*v^2`), not a simplified 2-term version.
  The linear term is reported (bearing/drivetrain friction) but not applied
  to `CarConstants`, since the current force equations have no place for it
  yet -- an honest, stated gap rather than a silently dropped one.
- **The solver is a DAG shortest path, not Dijkstra** -- solar income makes
  edge weights go negative, which breaks Dijkstra's correctness guarantee.
  Verified empirically (not just argued theoretically) that the two
  algorithms diverge on real data once that happens.
- **No battery current-limit checking yet** -- `battery.py` tracks energy
  (state of charge), not current. Real discharge-current limits weren't
  supplied for the chosen pack; adding them is a matter of extending
  `simulate_soc` once you have them, not a redesign.

## Running it

```bash
pip install -r requirements.txt
pytest tests/ -v     # 38 tests: real sample telemetry + synthetic known-truth physics
```

No dashboard app in this build -- everything above is a library, meant to
be driven from your own scripts/notebooks or wired into whatever tooling
you build next.

## What's still explicitly open

- Wheel diameter, drag coefficient, frontal area: still need real values
  in `CarConstants`/`MotorConstants` (coastdown testing gives you Crr/Cd;
  frontal area still needs a tape measure or CAD figure).
- Battery discharge-current limits: not yet enforced anywhere.
- Solar array size vs. the class rules: still needs finalizing (see
  earlier conversation -- the array as originally diagrammed was ~2x the
  legal limit).
- Motor efficiency map: still running on the fallback approximation until
  real dyno points are fed into `from_measurements()`.
- No round-trip conversion loss on solar income (MPPT + charge/discharge
  losses) -- `net_energy_j` currently assumes solar power converts to
  propulsion energy at 100% efficiency.

## Real data from Mitsuba manual + club purchase docs (latest update)

Three source documents reconciled into the code, with one conflict flagged
rather than silently resolved:

- **Motor efficiency: real chart, not a guess.** `performance/motor_map.py`'s
  `from_mitsuba_eco_mode_chart()` digitizes actual points off Mitsuba's
  published "Motor Chracteristics, 96V ECO Mode" chart (manual page 12) --
  by eye, not precision plot-digitizing software. This is now
  `DirectDriveMotorModel`'s default, replacing the guessed Gaussian curve.
  **Honest discrepancy, not reconciled:** the chart plateaus at ~93-94%
  efficiency out to 30A; the datasheet headline claims ">95% including
  controller." Possibly a POWER-mode vs. ECO-mode difference, possibly a
  higher-current measurement point. Worth asking Mitsuba directly.
- **95% efficiency confirmed as motor+controller combined** -- the manual
  states this explicitly, resolving what was previously an open question.
- **Wheel diameter now computed from the actual tyre choice**, not a guess:
  Michelin Pilot Street 2, front 70/90-17 -> 0.558m OD (rear 80/90-17 ->
  0.576m -- confirm which wheel actually carries the motor).
- **Thermal derating added straight from the manual's fault table**:
  `mitsuba_thermal_derate()` -- 85degC->1/2 power, 95degC->1/4 power,
  105degC->drive stop, wired into `DirectDriveMotorModel.predict()` as an
  optional `controller_temp_c` argument.
- **Rolling resistance (0.0134) is confirmed as an ENGINEERING ESTIMATE**,
  not a measured value -- the tyre document's own methodology (a
  component-hysteresis model with modifiers on a baseline Crr), not a
  manufacturer spec. `testing.coastdown` exists specifically to check this
  estimate against a real coastdown test.
- **BATTERY SPEC CONFLICT -- not resolved, flagged instead:** earlier in
  this conversation you gave 60V / 50Ah / NMC. The newly-uploaded purchase
  document's header says "48V, 60A LFP" (LiFePO4, not NMC) with candidate
  links spanning 40-70Ah at 48-72V. These are genuinely different chemistries
  and voltages -- different discharge curves, different cell counts, different
  max C-rates. `battery.BatterySpec` still defaults to the 60V/50Ah/NMC
  figures from the direct conversation; **confirm which is actually final**
  before trusting any battery-related output.
