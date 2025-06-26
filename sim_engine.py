# -*- coding: utf-8 -*-
"""
Engine interface for SPECTRE simulations.
"""

from simulations.weekly_simulation import WeeklySimulation
from simulations.personnel_simulation import PersonnelSimulation
from simulations.historical_annual import HistoricalAnnualSimulation


def run_weekly_simulation(params, trials):
    """
    Executes a Monte Carlo simulation of weekly sortie generation, honoring weekend repair flags.

    Parameters:
    - params: dict of all simulation inputs, must include 'weekend_duty'
    - trials: number of simulation trials to run

    Returns:
    - A list of trial results (each trial is a list of weekly summaries)
    """
    sim = WeeklySimulation(**params)
    # Pass weekend duty flags into simulate
    weekend = params.get("weekend_duty", None)
    return sim.simulate(trials=trials, weekend_duty=weekend)


def run_personnel_simulation(params, trials):
    """
    Executes a simulation to assess sortie capacity based on available manning.

    Parameters:
    - params: dict of inputs (TAI, O&M days, skill mix, absence rates)
    - trials: number of trials for stochastic variation

    Returns:
    - A list of trial results (each trial is a list of month summaries)
    """
    sim = PersonnelSimulation(**params)
    return sim.simulate(trials=trials)


def run_historical_simulation(params, trials):
    """
    Placeholder for running the historical annual performance simulator.

    Parameters:
    - params: dict of inputs (expected format TBD)
    - trials: number of deterministic or stochastic runs

    Returns:
    - A list of simulation outputs (placeholder or structured results)
    """
    sim = HistoricalAnnualSimulation(**params)
    return sim.simulate(trials=trials)



