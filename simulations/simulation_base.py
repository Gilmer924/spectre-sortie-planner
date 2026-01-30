# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 08:59:29 2025

@author: scwri
"""
# simulations/simulation_base.py
# from abc import ABC, abstractmethod


# class SimulationBase(ABC):
#     def __init__(self, **params):
#         self.params = params
#         self.results = None

#     @abstractmethod
#     def validate_params(self):
#         pass

#     @abstractmethod
#     def run(self):
#         pass

#     def get_results(self):
#         return self.results

from abc import ABC, abstractmethod
import logging

# Setup a basic logger to catch where things go wrong
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SimulationEngine")

class SimulationBase(ABC):
    def __init__(self, **params):
        self.params = params
        self.results = None

    @abstractmethod
    def validate_params(self):
        """Subclasses must check for required keys here."""
        pass

    @abstractmethod
    def run(self):
        """Subclasses implement the core simulation logic."""
        pass

    def get_results(self):
        return self.results

    def safe_access(self, dictionary, key, default=0.0, min_val=None):
        """
        Helper to grab params or data rows safely.
        Ensures numeric types and prevents zeros where they shouldn't be.
        """
        try:
            val = dictionary.get(key, default)
            # Handle potential None or empty string from DataFrames
            if val is None or val == "":
                return default
            
            numeric_val = float(val)
            
            if min_val is not None:
                return max(numeric_val, min_val)
            return numeric_val
        except (ValueError, TypeError):
            return default

    def execute(self):
        """
        A wrapper for run() that handles top-level crashes 
        and ensures parameters are validated first.
        """
        try:
            self.validate_params()
            self.results = self.run()
            return self.results
        except ZeroDivisionError as e:
            logger.error(f"Execution failed in {self.__class__.__name__}: Mathematical Division by Zero.")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error in {self.__class__.__name__}: {str(e)}")
            raise e