# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 2025

@author: Gilmer Jenkins
"""
# simulations/personnel_simulation.py

import calendar
import numpy as np
from datetime import datetime
from typing import Any, Dict, List

from simulations.simulation_base import SimulationBase

class PersonnelSimulation(SimulationBase):
    """
    Monte Carlo or deterministic simulation of sortie-producing labor capacity.
    
    Tracks:
        - present people after leave/TDY/deploy (accounts for overlap)
        - available person-hours (UTE × O&M days)
        - sorties_supported (hours ÷ labor_hours_per_sortie)
        - ppl_per_ac (people per aircraft)
        - shifts_supported (8-hr shifts per AC)
        - shortfall flag vs. monthly goal
        - bottleneck shop/skill if a shortfall occurs
    """

    def validate_params(self) -> None:
        """Validate simulation input parameters, with informative error messages."""
        p = self.params
        tai = p.get("TAI")
        if not isinstance(tai, (int, float)) or tai <= 0:
            raise ValueError(f"‘TAI’ must be a positive number, got {tai!r}")

        months = p.get("months")
        if not isinstance(months, list) or not months:
            raise ValueError("‘months’ must be a non-empty list of month numbers")

        omd = p.get("om_days")
        if not isinstance(omd, dict):
            raise ValueError("‘om_days’ must be a dict month→O&M-days")

        goals = p.get("month_goals")
        if not isinstance(goals, dict):
            raise ValueError("‘month_goals’ must be a dict month→goal")

        for key in ("leave_rate", "tdy_rate", "deploy_rate"):
            v = p.get(key, 0.0)
            if not (isinstance(v, (int, float)) and 0 <= v <= 1):
                raise ValueError(f"{key!r} must be between 0 and 1 (got {v!r})")

        ute = p.get("ute_rates")
        if not isinstance(ute, dict) or not ute:
            raise ValueError("‘ute_rates’ must be a dict level→hours_per_day")

        lps = p.get("labor_hours_per_sortie")
        if not (isinstance(lps, (int, float)) and lps > 0):
            raise ValueError("‘labor_hours_per_sortie’ must be positive (got {lps!r})")

        wcs = p.get("workcenters")
        if not isinstance(wcs, dict) or not wcs:
            raise ValueError("‘workcenters’ must be a non-empty dict shop→{level:count}")

    def _calculate_absence(self, leave: float, tdy: float, deploy: float) -> float:
        """
        Calculate the combined probability of absence (not just sum)
        - returns 1 - probability of present
        """
        present_prob = (1 - leave) * (1 - tdy) * (1 - deploy)
        return 1 - present_prob

    def simulate(self, trials: int = 1) -> List[List[Dict[str, Any]]]:
        """
        Simulate labor-based sortie production for each month.
        Returns: list of [list-of-month-results-per-trial]
        """
        self.validate_params()
        p = self.params
        months = p["months"]
        omd = p["om_days"]
        goals = p["month_goals"]
        leave = p["leave_rate"]
        tdy = p["tdy_rate"]
        deploy = p["deploy_rate"]
        ute = p["ute_rates"]
        lps = float(p["labor_hours_per_sortie"])
        wcs = p["workcenters"]
        tai = float(p["TAI"])

        stochastic = (trials > 1)
        all_trials = []
        this_year = datetime.now().year

        absence_rate = self._calculate_absence(leave, tdy, deploy)

        for _ in range(trials):
            trial_months = []
            for m in months:
                days_in_month = omd.get(m, calendar.monthrange(this_year, m)[1])

                total_present = 0.0
                total_hours = 0.0

                shop_present = {}
                shop_hours = {}

                # 1. Sum across each shop & skill level, tracking details for bottleneck analysis
                for shop, levels in wcs.items():
                    present_in_shop = 0.0
                    hours_in_shop = 0.0
                    for lvl, assigned in levels.items():
                        if stochastic:
                            absent = np.random.binomial(int(assigned), min(1, absence_rate))
                            present = assigned - absent
                        else:
                            present = assigned * (1 - absence_rate)
                        total_present += present
                        total_hours += present * ute.get(lvl, 0.0) * days_in_month
                        present_in_shop += present
                        hours_in_shop += present * ute.get(lvl, 0.0) * days_in_month
                    shop_present[shop] = present_in_shop
                    shop_hours[shop] = hours_in_shop

                sorties_supported = total_hours / lps if lps > 0 else 0
                ppl_per_ac = total_present / tai if tai > 0 else 0
                shifts_supported = total_hours / (tai * 8.0) if tai > 0 else 0
                shortfall = sorties_supported < goals.get(m, 0)

                # 2. Bottleneck shop (if shortfall)
                limiting_shop = ""
                limiting_shop_sorties = None
                if shortfall:
                    # Which shop is the most limiting by available hours?
                    shop_sortie_capacity = {shop: shop_hours[shop]/lps if lps>0 else 0 for shop in shop_hours}
                    limiting_shop, limiting_shop_sorties = min(
                        shop_sortie_capacity.items(), key=lambda x: x[1])

                trial_months.append({
                    "month": m,
                    "present_people": total_present,
                    "available_hours": total_hours,
                    "sorties_supported": sorties_supported,
                    "ppl_per_ac": ppl_per_ac,
                    "shifts_supported": shifts_supported,
                    "shortfall": shortfall,
                    "limiting_shop": limiting_shop if shortfall else "",
                    "limiting_shop_sorties": limiting_shop_sorties if shortfall else None,
                })
            all_trials.append(trial_months)
        return all_trials

    def run(self) -> List[Dict[str, Any]]:
        """
        Deterministic single-trial shortcut
        Returns: list of monthly results (for one trial)
        """
        return self.simulate(trials=1)[0]

