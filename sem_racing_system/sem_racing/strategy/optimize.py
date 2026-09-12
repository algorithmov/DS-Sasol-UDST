"""
sem_racing.optimize
====================
Graph construction + shortest-path solve, refactored out of
``BE_SpeedProfileOptimisation.ipynb`` / ``AI_SpeedProfileOptimisation.ipynb``
into functions that take a ``config`` (from ``trackmodel`` or ``route``) and
any object satisfying the car-model ``.predict()`` interface from
``carmodel``.

Two generalisations over the original notebooks:

1. Rolling starts. The notebooks hard-coded a single ``x0_k0`` node at v=0,
   correct only for a standing start. Section 0 is now treated exactly like
   every other section (a *range* of feasible entry speeds); the solver
   searches from/to whichever speeds are actually feasible at each end. A
   config with ``vmin==vmax==0`` at both ends (a standing-start config)
   collapses back to the original single-source/-sink case automatically.

2. Solar income makes edge weights go negative -- important, and it's a
   correctness fix I owed from the first version of this module. Dijkstra's
   algorithm assumes every edge weight is non-negative; once a slow, sunny
   section can have *negative* net cost (you gain more energy than you
   spend), that assumption breaks and Dijkstra can silently return a
   suboptimal path. This graph is a strict layered DAG, though -- every
   edge goes from distance-index x-1 to x, so cycles are structurally
   impossible -- which means a single topological-order relaxation pass
   (classic DAG shortest path) is both correct with negative weights *and*
   faster than Dijkstra. That's what ``solve_optimal_profile`` does now.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

from .carmodel import ImpossibleState
from .solar import ArraySpec, IrradianceProfile, energy_income
from .wind import WindProfile


@dataclass
class OptimizerSettings:
    section_length: float = 50.0     # m, distance resolution
    velocity_spacing: float = 2.0    # km/h, speed resolution
    max_acceleration_steps: int = 4  # how many velocity_spacing steps/section
    factor_time: float = 0.5         # 0..1, weight on going fast
    factor_efficiency: float = 0.5   # 0..1, weight on saving energy
    energy_scale: float = 2500.0     # normalises energy cost vs. time cost
    time_scale: float = 10.0
    cell_temp_c: float = 25.0        # only used if a solar array/irradiance is supplied


def _gradient_angle(config: pd.DataFrame, x: int, section_length: float) -> float:
    rise = config.loc[x, "m_above_sea"] - config.loc[x - 1, "m_above_sea"]
    return math.atan(rise / section_length)


def _nodes_for_section(config: pd.DataFrame, x: int, settings: OptimizerSettings):
    vmin, vmax = config.loc[x, "vmin"], config.loc[x, "vmax"]
    k_min = int(np.ceil(vmin / settings.velocity_spacing))
    k_max = int(np.floor(vmax / settings.velocity_spacing))
    if k_min > k_max:
        # vmax - vmin narrower than one velocity_spacing step (common in low
        # speed / near-stationary bins) -- fall back to the single nearest
        # step so the section still has at least one feasible node.
        k_mid = int(round((vmin + vmax) / 2 / settings.velocity_spacing))
        k_min = k_max = k_mid
    return range(k_min, k_max + 1)


def build_speed_graph(
    config: pd.DataFrame,
    car_model,
    settings: OptimizerSettings,
    array: ArraySpec | None = None,
    irradiance: IrradianceProfile | None = None,
    wind: WindProfile | None = None,
):
    """Build the distance/speed state graph and weight edges by the car
    model's predicted net energy (consumption minus solar income, if an
    ``array``/``irradiance`` are supplied) and time cost for that
    transition. Returns ``(G, coordinates, start_nodes, end_nodes)``.

    Passing ``array``/``irradiance``/``wind`` doesn't change the graph's
    shape -- the state is still just (distance, speed), not (distance,
    speed, elapsed time) -- so both are looked up by *distance into the
    stage*, not true time-of-day. See ``solar.py``'s module docstring for
    why: a fully time-aware version needs elapsed time as part of the
    state, which would triple the state space. This distance-indexed
    approximation is what you get by turning a time-of-day forecast into a
    distance-of-stage forecast using your planned average pace up front.
    """
    s = settings
    G = nx.DiGraph()
    coords = {}
    x_max = len(config.index) - 1

    for k in _nodes_for_section(config, 0, s):
        v = k * s.velocity_spacing
        node_name = f"x0_k{k}"
        G.add_node(node_name, v=v)
        coords[node_name] = (0, v)
    start_nodes = [n for n in G.nodes if n.startswith("x0_k")]

    for x in range(1, x_max + 1):
        gradient_angle = _gradient_angle(config, x, s.section_length)
        section_start_dist = (x - 1) * s.section_length
        headwind_ms = wind.at(section_start_dist) if wind is not None else 0.0
        for k in _nodes_for_section(config, x, s):
            v = k * s.velocity_spacing
            node_name = f"x{x}_k{k}"
            G.add_node(node_name, v=v)
            coords[node_name] = (x * s.section_length, v)

            for kold in range(k - s.max_acceleration_steps, k + s.max_acceleration_steps + 1):
                prev = f"x{x - 1}_k{kold}"
                if not G.has_node(prev):
                    continue
                vold = kold * s.velocity_spacing
                try:
                    energy, time = car_model.predict(
                        gradient_angle, vold / 3.6, v / 3.6, s.section_length, headwind_ms=headwind_ms
                    )
                except ImpossibleState:
                    continue

                income = 0.0
                if array is not None and irradiance is not None:
                    income = energy_income(array, irradiance, section_start_dist, time, s.cell_temp_c)
                net_energy = energy - income  # can be negative: net charging while driving

                cost_econ = net_energy / s.energy_scale
                cost_speed = time / s.time_scale
                weight = s.factor_time * cost_speed + s.factor_efficiency * cost_econ
                G.add_edge(prev, node_name, weight=weight, time=time, energy=energy, income=income)

    end_nodes = [n for n in G.nodes if n.startswith(f"x{x_max}_k")]
    return G, coords, start_nodes, end_nodes


def solve_optimal_profile(G, config: pd.DataFrame, start_nodes, end_nodes, settings: OptimizerSettings):
    """Single-pass topological (DAG) shortest path -- correct even with the
    negative edge weights solar income can introduce, unlike Dijkstra.
    Collapses to the same answer Dijkstra would give whenever all weights
    are non-negative (i.e. no solar income supplied), so this is a strict
    upgrade, not a behaviour change for the no-solar case."""
    if not start_nodes or not end_nodes:
        raise ValueError("No feasible start or end nodes in the graph.")

    order = list(nx.topological_sort(G))
    dist = {n: math.inf for n in order}
    prev: dict[str, str] = {}
    for n in start_nodes:
        if n in dist:
            dist[n] = 0.0

    for u in order:
        du = dist[u]
        if du == math.inf:
            continue
        for v in G.successors(u):
            w = G.edges[u, v]["weight"]
            if du + w < dist[v]:
                dist[v] = du + w
                prev[v] = u

    reachable_ends = [n for n in end_nodes if dist.get(n, math.inf) < math.inf]
    if not reachable_ends:
        raise ValueError(
            "No feasible path from start to finish -- loosen vmin/vmax, "
            "raise max_acceleration_steps, or check the car model's limits "
            "against the speeds this config actually demands."
        )
    target = min(reachable_ends, key=lambda n: dist[n])

    node_path = [target]
    while node_path[-1] not in start_nodes:
        node_path.append(prev[node_path[-1]])
    node_path.reverse()

    speeds = [G.nodes[n]["v"] for n in node_path]
    x_positions = [int(n.split("_")[0][1:]) for n in node_path]
    profile = pd.DataFrame({"speed_kmh": speeds, "distance_m": [x * settings.section_length for x in x_positions]})

    total_time = sum(G.edges[u, v].get("time", 0) for u, v in zip(node_path[:-1], node_path[1:]))
    total_energy = sum(G.edges[u, v].get("energy", 0) for u, v in zip(node_path[:-1], node_path[1:]))
    total_income = sum(G.edges[u, v].get("income", 0) for u, v in zip(node_path[:-1], node_path[1:]))

    return profile, {
        "total_time_s": total_time,
        "total_energy_j": total_energy,
        "total_solar_income_j": total_income,
        "net_energy_j": total_energy - total_income,
        "node_path": node_path,
    }
