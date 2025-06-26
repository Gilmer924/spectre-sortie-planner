# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 08:59:05 2025

@author: scwri
"""
# simulations/personnel_simulation.py

import calendar
import numpy as np
from datetime import datetime
from simulations.simulation_base import SimulationBase

class PersonnelSimulation(SimulationBase):
    """
    Monte–Carlo or deterministic simulation of sortie-producing labor capacity
    across a custom set of months.  Tracks:
      - total present people (after leave/TDY/deploy)
      - available person-hours (UTE × O&M days)
      - sorties_supported (hours ÷ labor_hours_per_sortie)
      - ppl_per_ac (people per aircraft)
      - shifts_supported (8-hr shifts per AC)
      - shortfall flag vs. monthly goal
    """

    def validate_params(self):
        p = self.params
        # 1) TAI
        tai = p.get("TAI")
        if not isinstance(tai, (int, float)) or tai <= 0:
            raise ValueError("‘TAI’ must be a positive number")

        # 2) months list
        months = p.get("months")
        if not isinstance(months, list) or not months:
            raise ValueError("‘months’ must be a non-empty list of month numbers")

        # 3) om_days
        omd = p.get("om_days")
        if not isinstance(omd, dict):
            raise ValueError("‘om_days’ must be a dict month→O&M-days")

        # 4) month_goals
        goals = p.get("month_goals")
        if not isinstance(goals, dict):
            raise ValueError("‘month_goals’ must be a dict month→goal")

        # 5) absence rates
        for key in ("leave_rate","tdy_rate","deploy_rate"):
            v = p.get(key, 0.0)
            if not (isinstance(v,(int,float)) and 0<=v<=1):
                raise ValueError(f"{key!r} must be between 0 and 1")

        # 6) ute_rates
        ute = p.get("ute_rates")
        if not isinstance(ute, dict) or not ute:
            raise ValueError("‘ute_rates’ must be a dict level→hours_per_day")

        # 7) labor_hours_per_sortie
        lps = p.get("labor_hours_per_sortie")
        if not (isinstance(lps,(int,float)) and lps>0):
            raise ValueError("‘labor_hours_per_sortie’ must be positive")

        # 8) workcenters
        wcs = p.get("workcenters")
        if not isinstance(wcs, dict) or not wcs:
            raise ValueError("‘workcenters’ must be a non-empty dict shop→{level:count}")

    def simulate(self, trials=1):
        self.validate_params()
        p      = self.params
        months = p["months"]
        omd    = p["om_days"]
        goals  = p["month_goals"]
        leave  = p["leave_rate"]
        tdy    = p["tdy_rate"]
        deploy = p["deploy_rate"]
        ute    = p["ute_rates"]
        lps    = float(p["labor_hours_per_sortie"])
        wcs    = p["workcenters"]
        tai    = float(p["TAI"])

        stochastic = (trials > 1)
        all_trials = []

        # current year, for fallback month-length lookup
        this_year = datetime.now().year

        for _ in range(trials):
            trial_months = []
            for m in months:
                # get O&M days if provided, else full month length
                days_in_month = omd.get(
                    m,
                    calendar.monthrange(this_year, m)[1]
                )

                total_present = 0.0
                total_hours   = 0.0

                # sum across each shop & skill level
                for levels in wcs.values():
                    for lvl, assigned in levels.items():
                        if stochastic:
                            absent  = np.random.binomial(int(assigned),
                                                         min(1, leave+tdy+deploy))
                            present = assigned - absent
                        else:
                            present = assigned * (1 - (leave+tdy+deploy))

                        total_present += present
                        total_hours   += present * ute.get(lvl, 0.0) * days_in_month

                # compute sortie capacity etc.
                sorties_supported = total_hours / lps
                ppl_per_ac        = total_present / tai
                shifts_supported  = total_hours / (tai * 8.0)
                shortfall         = sorties_supported < goals.get(m, 0)

                trial_months.append({
                    "month":             m,
                    "present_people":    total_present,
                    "available_hours":   total_hours,
                    "sorties_supported": sorties_supported,
                    "ppl_per_ac":        ppl_per_ac,
                    "shifts_supported":  shifts_supported,
                    "shortfall":         shortfall
                })
            all_trials.append(trial_months)

        return all_trials

    def run(self):
        # deterministic single-trial shortcut
        return self.simulate(trials=1)[0]
