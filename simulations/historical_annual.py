# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 08:59:18 2025

@author: scwri
"""
# simulations/historical_annual.py
from simulations.simulation_base import SimulationBase

class HistoricalAnnualSimulation(SimulationBase):
    def validate_params(self):
        pass  # Implement validation later

    def simulate(self, trials=1):
        return [{}] * trials

    def run(self):
        return self.simulate(trials=1)[0]

