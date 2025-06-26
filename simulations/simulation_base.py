# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 08:59:29 2025

@author: scwri
"""
# simulations/simulation_base.py
from abc import ABC, abstractmethod

class SimulationBase(ABC):
    def __init__(self, **params):
        self.params = params
        self.results = None

    @abstractmethod
    def validate_params(self):
        pass

    @abstractmethod
    def run(self):
        pass

    def get_results(self):
        return self.results

