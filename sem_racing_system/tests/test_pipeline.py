"""
End-to-end tests for the six-pillar rebuild:
1. Telemetry processing & analytics
2. Race strategy simulation (energy model, route profile, PV vs. consumption)
3. Performance predictions (motor efficiency maps, resistance)
4. After-testing analysis (coastdown, regression analysis)
Plus pre-race target speed planning and route/weather variable strategy,
which are exercised through pillar 2's optimizer and stage planner.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from sem_racing.telemetry import ingest, laps, route, features
from sem_racing.strategy import trackmodel, carmodel, solar, wind, battery, optimize, stage_planner
from sem_racing.performance import motor_map, resistance
from sem_racing.testing import coastdown

DATA = Path(__file__).parent.parent / "data"


# ---------------------------------------------------------------------------
# Pillar 1: Telemetry processing & analytics
# ---------------------------------------------------------------------------

def test_load_export_csv_already_has_laps():
    df = ingest.load_export_csv(DATA / "BE_sample_data.csv")
    assert "lap_lap" in df.columns
    df2 = laps.ensure_laps(df)
    assert df2["lap_lap"].max() >= 1


def test_raw_obc_log_gets_segmented_from_gps_lines():
    df = ingest.load_obc_log(DATA / "obc_2019_07_02_Telemetry.log")
    london_lines = {
        "start": ((-0.469553367, -0.4696881), (51.35103562, 51.3510535)),
        "finish": ((-0.469667683, -0.469823533), (51.35060957, 51.35061913)),
        "lap": ((-0.469633117, -0.46974765), (51.35076913, 51.35077756)),
    }
    segmented = laps.ensure_laps(df, lines=london_lines)
    assert segmented["lap_lap"].max() > 0
    assert segmented["dist"].is_monotonic_increasing


def test_features_enrich_does_not_explode():
    df = ingest.load_export_csv(DATA / "BE_sample_data.csv")
    enriched = features.enrich(df)
    assert "power_w" in enriched.columns
    assert "acceleration" in enriched.columns


def test_route_load_requires_lat_lon():
    with pytest.raises(ValueError):
        route.load_route(pd.DataFrame({"foo": [1, 2]}))


def test_add_stage_distance_is_monotonic_no_crossing_detection_needed():
    df = ingest.load_export_csv(DATA / "BE_sample_data.csv")
    staged = route.add_stage_distance(df)
    assert "stage_dist" in staged.columns
    assert staged["stage_dist"].is_monotonic_increasing


def _synthetic_straight_route(n=20, section_len_deg=0.001):
    lat0, lon0 = 51.0, -0.1
    lats = [lat0 + i * section_len_deg for i in range(n)]
    lons = [lon0] * n
    return pd.DataFrame({"lat": lats, "lon": lons, "elevation_m": np.linspace(0, 50, n)})


def test_corner_speed_lower_than_straight_speed():
    lats = [51.0 + i * 0.001 for i in range(10)] + [51.01] * 10
    lons = [-0.1] * 10 + [-0.1 + i * 0.001 for i in range(10)]
    r = pd.DataFrame({"lat": lats, "lon": lons})
    limits = route.estimate_corner_speed_limits(r, top_speed_kmh=80.0)
    assert limits.iloc[10] < limits.iloc[3]


def test_compute_headings_deg_flags_the_turn():
    lats = [51.0 + i * 0.001 for i in range(10)] + [51.01] * 10
    lons = [-0.1] * 10 + [-0.1 + i * 0.001 for i in range(10)]
    r = pd.DataFrame({"lat": lats, "lon": lons})
    headings = route.compute_headings_deg(r)
    assert abs(((headings[3] - headings[15] + 180) % 360) - 180) > 45


# ---------------------------------------------------------------------------
# Pillar 2: Race strategy simulation (energy model, route profile, PV vs.
# consumption optimisation) + pre-race target speed planning
# ---------------------------------------------------------------------------

def test_track_config_derived_from_real_laps():
    df = ingest.load_export_csv(DATA / "BE_sample_data.csv")
    df = laps.ensure_laps(df)
    df = laps.add_lap_relative_columns(df)
    config = trackmodel.build_track_config(df, section_length=50.0)
    assert set(["s", "vmax", "vmin", "m_above_sea"]).issubset(config.columns)
    assert (config["vmax"] >= config["vmin"]).all()


def test_car_constants_reflect_real_specs():
    c = carmodel.CarConstants()
    assert c.car_mass_no_driver == 300.0
    assert c.coeff_roll_res == pytest.approx(0.0134)


def test_physical_car_model_predicts_plausible_energy_and_time():
    constants = carmodel.CarConstants(torque_max=20.0)
    model = carmodel.PhysicalCarModel(constants)
    energy, time = model.predict(gradient_angle=0.0, v1=5.0, v2=6.0, section_length=50.0)
    assert time > 0
    assert energy > 0


def test_direct_drive_motor_predicts_plausible_energy():
    model = carmodel.DirectDriveMotorModel()
    energy, time = model.predict(gradient_angle=0.0, v1=5.0, v2=6.0, section_length=100.0)
    assert time > 0
    assert energy > 0


def test_direct_drive_motor_respects_peak_power_ceiling():
    model = carmodel.DirectDriveMotorModel(motor=carmodel.MotorConstants(peak_power_w=200.0))
    with pytest.raises(carmodel.ImpossibleState):
        model.predict(gradient_angle=0.0, v1=0.1, v2=20.0, section_length=20.0)


def test_direct_drive_motor_uses_injected_efficiency_map():
    measured = motor_map.MotorEfficiencyMap.from_measurements(
        pd.DataFrame({"rpm": [100, 500, 900], "efficiency": [0.6, 0.9, 0.7]})
    )
    model = carmodel.DirectDriveMotorModel(efficiency_map=measured)
    assert model.efficiency_map.is_measured
    energy, time = model.predict(gradient_angle=0.0, v1=2.0, v2=3.0, section_length=100.0)
    assert energy > 0 and time > 0


def test_direct_drive_motor_headwind_increases_energy_use():
    model = carmodel.DirectDriveMotorModel()
    calm_energy, _ = model.predict(gradient_angle=0.0, v1=8.0, v2=8.0, section_length=200.0, headwind_ms=0.0)
    headwind_energy, _ = model.predict(gradient_angle=0.0, v1=8.0, v2=8.0, section_length=200.0, headwind_ms=5.0)
    tailwind_energy, _ = model.predict(gradient_angle=0.0, v1=8.0, v2=8.0, section_length=200.0, headwind_ms=-5.0)
    assert headwind_energy > calm_energy > tailwind_energy


def test_available_power_scales_with_irradiance_and_temperature():
    array = solar.ArraySpec(rated_power_w=1000.0)
    assert solar.available_power(array, 1000.0) == pytest.approx(1000.0, rel=1e-6)
    assert solar.available_power(array, 500.0) == pytest.approx(500.0, rel=1e-6)
    assert solar.available_power(array, 1000.0, cell_temp_c=60.0) < solar.available_power(array, 1000.0, cell_temp_c=25.0)


def test_energy_income_rewards_slower_sections():
    array = solar.ArraySpec(rated_power_w=1000.0)
    irr = solar.IrradianceProfile.constant(1000.0, stage_length_m=1000.0)
    assert solar.energy_income(array, irr, 0.0, time_s=20.0) > solar.energy_income(array, irr, 0.0, time_s=5.0)


def test_wind_headwind_and_tailwind_signs():
    wc = wind.WindConditions(speed_ms=5.0, direction_from_deg=0.0)
    assert wind.headwind_component(0.0, wc) == pytest.approx(5.0, rel=1e-6)
    assert wind.headwind_component(180.0, wc) == pytest.approx(-5.0, rel=1e-6)


def test_wind_profile_from_route_varies_with_heading():
    lats = [51.0 + i * 0.001 for i in range(10)] + [51.01] * 10
    lons = [-0.1] * 10 + [-0.1 + i * 0.001 for i in range(10)]
    r = pd.DataFrame({"lat": lats, "lon": lons})
    wc = wind.WindConditions(speed_ms=6.0, direction_from_deg=0.0)
    profile = wind.WindProfile.from_route_and_wind(r, wc)
    early = profile.at(50.0)
    late = profile.at(1400.0)
    assert abs(early - late) > 1.0


def test_battery_spec_usable_capacity():
    b = battery.BatterySpec(nominal_voltage_v=60.0, capacity_ah=50.0, usable_fraction=0.9)
    assert b.usable_capacity_wh == pytest.approx(60.0 * 50.0 * 0.9)


def test_full_optimizer_with_solar_and_wind_through_dag_solver():
    r = _synthetic_straight_route(n=15)
    config = route.build_stage_config(r, section_length=200.0, top_speed_kmh=70.0, standing_start=True)
    motor = carmodel.DirectDriveMotorModel()
    array = solar.ArraySpec(rated_power_w=1400.0)
    strong_sun = solar.IrradianceProfile.constant(1000.0, stage_length_m=config["s"].max())
    calm_wind = wind.WindProfile.calm(config["s"].max())
    settings = optimize.OptimizerSettings(section_length=200.0, max_acceleration_steps=20)

    G, coords, s_nodes, e_nodes = optimize.build_speed_graph(config, motor, settings, array, strong_sun, calm_wind)
    profile, result = optimize.solve_optimal_profile(G, config, s_nodes, e_nodes, settings)
    assert result["total_solar_income_j"] > 0
    assert len(profile) == len(config)


def test_battery_soc_depletes_and_flags_infeasible_when_too_small():
    r = _synthetic_straight_route(n=10)
    config = route.build_stage_config(r, section_length=200.0, top_speed_kmh=40.0, standing_start=True)
    motor = carmodel.DirectDriveMotorModel()
    settings = optimize.OptimizerSettings(section_length=200.0, max_acceleration_steps=15)
    G, coords, s_nodes, e_nodes = optimize.build_speed_graph(config, motor, settings)
    _, result = optimize.solve_optimal_profile(G, config, s_nodes, e_nodes, settings)

    ok_battery = battery.BatterySpec(capacity_ah=50.0)
    trace = battery.simulate_soc(result["node_path"], G, ok_battery)
    assert trace["soc_fraction"].is_monotonic_decreasing

    tiny_battery = battery.BatterySpec(nominal_voltage_v=60.0, capacity_ah=0.001)
    trace2 = battery.simulate_soc(result["node_path"], G, tiny_battery)
    feasible, message = battery.check_feasible(trace2)
    assert feasible is False
    assert "depleted" in message.lower()


def test_multi_stage_planner_carries_soc_forward():
    r1 = _synthetic_straight_route(n=10)
    r2 = _synthetic_straight_route(n=10)
    array = solar.ArraySpec(rated_power_w=1400.0)
    stages = [
        stage_planner.StagePlan(
            name="Stage 1", route=r1, section_length=200.0, top_speed_kmh=50.0,
            irradiance=solar.IrradianceProfile.constant(1000.0, stage_length_m=2000.0),
        ),
        stage_planner.StagePlan(
            name="Stage 2", route=r2, section_length=200.0, top_speed_kmh=50.0,
            irradiance=solar.IrradianceProfile.constant(200.0, stage_length_m=2000.0),
        ),
    ]
    car = carmodel.DirectDriveMotorModel()
    batt = battery.BatterySpec(nominal_voltage_v=60.0, capacity_ah=50.0)

    results = stage_planner.plan_stages(
        stages, car, array, batt,
        settings_overrides={"max_acceleration_steps": 20},
        starting_soc_fraction=1.0,
        recharge_between_stages_to=None,
    )
    assert len(results) == 2
    assert results[0].name == "Stage 1"
    assert results[1].soc_trace["soc_fraction"].iloc[0] == pytest.approx(results[0].ending_soc_fraction)


def test_ai_car_model_trains_on_aggregated_sections():
    df = ingest.load_export_csv(DATA / "BE_sample_data.csv")
    df = laps.ensure_laps(df)
    df = laps.add_lap_relative_columns(df, cumulative_cols=("jm3_netjoule",))
    sections = carmodel.aggregate_sections(df, section_length=100.0)
    assert len(sections) > 10

    pipeline = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("linear", MultiOutputRegressor(LinearRegression())),
    ])
    ai_model = carmodel.AICarModel(pipeline).fit(sections)
    assert "rmse" in ai_model.metrics_
    energy, time = ai_model.predict(gradient_angle=0.0, v1=5.0, v2=8.0, section_length=100.0)
    assert time > 0


# ---------------------------------------------------------------------------
# Pillar 3: Performance predictions (motor efficiency maps, resistance)
# ---------------------------------------------------------------------------

def test_motor_efficiency_map_default_is_flagged_unmeasured():
    m = motor_map.MotorEfficiencyMap.from_datasheet_default()
    assert m.is_measured is False
    assert "FALLBACK" in m.describe()
    # interpolation-grid granularity, not exact -- the underlying continuous
    # Gaussian peaks exactly at rated_speed_rpm, but from_datasheet_default's
    # grid gets asymmetrically clamped near rpm=0, so the nearest grid point
    # to the true peak is a hair off. abs tolerance, not the default rel.
    assert m.efficiency_at(810.0) == pytest.approx(0.95, abs=1e-3)
    assert m.efficiency_at(810.0 * 5) < 0.95


def test_motor_efficiency_map_from_measurements_is_flagged_measured():
    m = motor_map.MotorEfficiencyMap.from_measurements(
        pd.DataFrame({"rpm": [200, 800, 1400], "efficiency": [0.7, 0.95, 0.8]})
    )
    assert m.is_measured is True
    assert "bench" in m.describe()
    assert m.efficiency_at(800) == pytest.approx(0.95, rel=1e-6)
    assert m.efficiency_at(50) == pytest.approx(0.7, rel=1e-6)


def test_motor_efficiency_map_2d_uses_torque():
    m = motor_map.MotorEfficiencyMap.from_measurements(
        pd.DataFrame({"rpm": [800, 800], "torque_nm": [5.0, 20.0], "efficiency": [0.95, 0.7]})
    )
    assert m.is_2d
    assert m.efficiency_at(800, torque_nm=5.0) == pytest.approx(0.95, rel=1e-6)
    assert m.efficiency_at(800, torque_nm=20.0) == pytest.approx(0.7, rel=1e-6)


def test_resistance_curve_aero_term_dominates_at_speed():
    c = carmodel.CarConstants()
    curve = resistance.resistance_curve(c, speed_range_ms=np.array([1.0, 10.0, 20.0]))
    assert curve["f_rolling_n"].nunique() == 1
    assert curve["f_aero_n"].iloc[2] > curve["f_aero_n"].iloc[1] > curve["f_aero_n"].iloc[0]


def test_resistance_headwind_increases_aero_force():
    c = carmodel.CarConstants()
    calm = resistance.resistance_at_speed(c, speed_ms=10.0, headwind_ms=0.0)
    headwind = resistance.resistance_at_speed(c, speed_ms=10.0, headwind_ms=5.0)
    assert headwind["f_aero_n"] > calm["f_aero_n"]


# ---------------------------------------------------------------------------
# Pillar 4: After-testing analysis (coastdown, regression analysis)
# ---------------------------------------------------------------------------

def _simulate_coastdown(true_crr=0.0134, true_cd=0.3, area=1.0, mass=300.0, v0=15.0, rho=1.225, g=9.81, dt=0.5, n=60):
    t = [0.0]
    v = [v0]
    for _ in range(n):
        a = -(g * true_crr + (rho * area * true_cd) / (2 * mass) * v[-1] ** 2)
        v_next = max(v[-1] + a * dt, 0.0)
        v.append(v_next)
        t.append(t[-1] + dt)
        if v_next <= 0:
            break
    return pd.DataFrame({"obc_timestamp": t, "gps_speed": np.array(v) * 3.6})


def test_coastdown_regression_recovers_known_coefficients():
    true_crr, true_cd, area, mass = 0.0134, 0.3, 1.0, 300.0
    df = _simulate_coastdown(true_crr, true_cd, area, mass)
    result = coastdown.analyze_coastdown(df, mass_kg=mass, frontal_area_m2=area)
    assert result.r_squared > 0.99
    assert result.coeff_roll_res == pytest.approx(true_crr, rel=0.05)
    assert result.coeff_aero_drag == pytest.approx(true_cd, rel=0.05)


def test_coastdown_rejects_too_few_points():
    df = pd.DataFrame({"obc_timestamp": [0, 1, 2], "gps_speed": [20, 19, 18]})
    with pytest.raises(ValueError):
        coastdown.analyze_coastdown(df, mass_kg=300.0)


def test_apply_to_car_constants_refuses_bad_fit():
    bad_result = coastdown.CoastdownResult(
        a0=0.1, a1=0.0, a2=0.001, coeff_roll_res=0.01, drag_area_m2=0.3,
        coeff_aero_drag=0.3, r_squared=0.2, n_points=20,
    )
    with pytest.raises(ValueError):
        coastdown.apply_to_car_constants(bad_result, carmodel.CarConstants())


def test_apply_to_car_constants_updates_only_resistance_fields():
    base = carmodel.CarConstants(car_mass_no_driver=300.0)
    good_result = coastdown.CoastdownResult(
        a0=0.13, a1=0.0, a2=0.0012, coeff_roll_res=0.0134, drag_area_m2=0.3,
        coeff_aero_drag=0.3, r_squared=0.95, n_points=50,
    )
    updated = coastdown.apply_to_car_constants(good_result, base)
    assert updated.coeff_roll_res == pytest.approx(0.0134)
    assert updated.coeff_aero_drag == pytest.approx(0.3)
    assert updated.car_mass_no_driver == 300.0


# ---------------------------------------------------------------------------
# Digital-twin validation: each subsystem checked against "lab" data
# independently. Real bench/dyno logs aren't available in this repo, so
# these tests generate honest synthetic ground truth (known physics, no
# hidden fudge) the same way test_coastdown_regression_recovers_known_coefficients
# does -- proving the machinery correctly reports near-zero error against
# data generated from its own equations, and correctly reports LARGE error
# when the model and "reality" genuinely disagree.
# ---------------------------------------------------------------------------

from sem_racing.testing import validation


def test_validate_energy_model_perfect_when_data_matches_model():
    model = carmodel.DirectDriveMotorModel()
    rows = []
    for v1, v2 in [(3.0, 4.0), (5.0, 6.0), (7.0, 6.5), (4.0, 4.0)]:
        energy, time = model.predict(0.0, v1, v2, section_length=100.0)
        rows.append({"v1": v1, "v2": v2, "E_Engine": energy, "time": time})
    sections = pd.DataFrame(rows)

    result = validation.validate_energy_model(model, sections)
    assert result.r_squared == pytest.approx(1.0, abs=1e-6)
    assert result.mae == pytest.approx(0.0, abs=1e-6)


def test_validate_energy_model_flags_a_bad_model():
    good_model = carmodel.DirectDriveMotorModel()
    rows = []
    for v1, v2 in [(3.0, 4.0), (5.0, 6.0), (7.0, 6.5), (4.0, 4.0)]:
        energy, time = good_model.predict(0.0, v1, v2, section_length=100.0)
        rows.append({"v1": v1, "v2": v2, "E_Engine": energy, "time": time})
    sections = pd.DataFrame(rows)

    # a deliberately wrong model (way too heavy) should score badly against
    # "real" data generated by the correctly-specified one
    wrong_model = carmodel.DirectDriveMotorModel(car=carmodel.CarConstants(car_mass_no_driver=3000.0))
    result = validation.validate_energy_model(wrong_model, sections)
    assert result.mae > 0
    assert result.r_squared < 0.9


def test_validate_motor_bench_matches_the_map_it_was_built_from():
    m = motor_map.MotorEfficiencyMap.from_measurements(
        pd.DataFrame({"rpm": [200, 800, 1400], "efficiency": [0.7, 0.95, 0.8]})
    )
    # construct a "bench log" whose actual efficiency, by definition, equals
    # the map at those exact rpm points -- solve for current given an
    # arbitrary torque/voltage so mech_power/elec_power == the map's value
    rpm = np.array([200.0, 800.0, 1400.0])
    torque = np.array([2.0, 2.0, 2.0])
    omega = rpm * 2 * np.pi / 60.0
    mech = torque * omega
    voltage = np.array([48.0, 48.0, 48.0])
    eff = np.array([m.efficiency_at(r) for r in rpm])
    current = mech / (voltage * eff)
    bench_df = pd.DataFrame({"rpm": rpm, "torque_nm": torque, "voltage_v": voltage, "current_a": current})

    result = validation.validate_motor_bench(m, bench_df)
    assert result.mae == pytest.approx(0.0, abs=1e-6)


def test_validate_battery_capacity_recovers_known_delivered_energy():
    # simulate a clean constant-current discharge: 10A at a flat 60V for
    # exactly 5 hours, hitting cutoff right at the end
    t = np.linspace(0, 5 * 3600, 200)
    v = np.linspace(65.0, 50.0, 200)  # sags down to cutoff
    i = np.full_like(t, 10.0)
    discharge_df = pd.DataFrame({"time_s": t, "voltage_v": v, "current_a": i})

    b = battery.BatterySpec(nominal_voltage_v=60.0, capacity_ah=50.0, usable_fraction=1.0)
    result = validation.validate_battery_capacity(b, discharge_df, cutoff_voltage=50.0)
    assert result.delivered_ah == pytest.approx(50.0, rel=0.02)  # 10A x 5h = 50Ah
    assert result.cutoff_reached is True


def test_validate_resistance_on_held_out_run_matches_true_physics():
    true_crr, true_cd, area, mass = 0.0134, 0.3, 1.0, 300.0
    held_out_df = _simulate_coastdown(true_crr, true_cd, area, mass, v0=12.0)
    constants = carmodel.CarConstants(
        coeff_roll_res=true_crr, coeff_aero_drag=true_cd, a_aero=area, car_mass_no_driver=mass, driver_mass=0.0
    )
    result = validation.validate_resistance(constants, held_out_df, mass_kg=mass)
    assert result.r_squared > 0.99


def test_validate_solar_array_matches_model_exactly_when_data_is_consistent():
    array = solar.ArraySpec(rated_power_w=1400.0)
    irr = np.array([200.0, 500.0, 1000.0])
    temp = np.array([25.0, 25.0, 25.0])
    measured = np.array([solar.available_power(array, x, t) for x, t in zip(irr, temp)])
    lab_df = pd.DataFrame({"irradiance_w_m2": irr, "cell_temp_c": temp, "measured_power_w": measured})

    result = validation.validate_solar_array(array, lab_df)
    assert result.mae == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Real Mitsuba datasheet chart (digitized) + thermal derating from the manual
# ---------------------------------------------------------------------------

def test_mitsuba_eco_chart_map_is_labeled_digitized_not_measured_or_guessed():
    m = motor_map.MotorEfficiencyMap.from_mitsuba_eco_mode_chart()
    assert m.source == "digitized_datasheet_chart"
    assert m.is_measured is True  # not the fallback tier
    assert "digitized" in m.describe()
    # sanity: efficiency should rise steeply at low current then plateau
    assert m.efficiency_at(rpm=890) < m.efficiency_at(rpm=800)


def test_direct_drive_motor_defaults_to_real_chart_map():
    model = carmodel.DirectDriveMotorModel()
    assert model.efficiency_map.source == "digitized_datasheet_chart"


def test_thermal_derate_matches_manual_fault_table():
    assert motor_map.mitsuba_thermal_derate(70.0) == 1.0
    assert motor_map.mitsuba_thermal_derate(85.0) == 0.5
    assert motor_map.mitsuba_thermal_derate(95.0) == 0.25
    assert motor_map.mitsuba_thermal_derate(105.0) == 0.0
    assert motor_map.mitsuba_thermal_derate(110.0) == 0.0


def test_direct_drive_motor_respects_thermal_derating():
    hot_motor = carmodel.MotorConstants(peak_power_w=1000.0)
    model = carmodel.DirectDriveMotorModel(motor=hot_motor)
    # a demand safely under the 1000W nameplate peak...
    model.predict(gradient_angle=0.0, v1=5.0, v2=6.0, section_length=100.0)
    # ...but the same demand should be refused once an overheated controller
    # (95C -> 1/4 power = 250W) can't deliver it
    with pytest.raises(carmodel.ImpossibleState):
        model.predict(gradient_angle=0.0, v1=5.0, v2=6.0, section_length=100.0, controller_temp_c=95.0)
