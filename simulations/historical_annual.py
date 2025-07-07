# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 08:59:18 2025

@author: Gilmer Jenkins
"""
# simulations/historical_annual.py

import numpy as np
import math
import calendar
from simulations.simulation_base import SimulationBase

class HistoricalAnnualSimulation(SimulationBase):
    def validate_params(self):
        required = [
            "TAI", "rates_df", "om_days", "planned_degraders",
            "turn_patterns", "commit_rates", "uncertainty"
        ]
        missing = [k for k in required if k not in self.params]
        if missing:
            raise ValueError("Missing parameter(s): " + ", ".join(missing))

    def simulate(self, trials=500):
        self.validate_params()
        rates = self.params["rates_df"]
        TAI = self.params["TAI"]
        om_days = self.params["om_days"]
        degraders = self.params["planned_degraders"]
        turn_patterns = self.params["turn_patterns"]
        commit_rates = self.params["commit_rates"]
        uncertainty = float(self.params.get("uncertainty", 0.05))  # ±5% default
        spares_percent = self.params.get("spares_pct", 0.2)  # Default to 20%
        commit_thresh = self.params.get("commit_thresh", 0.8) * 100  # Default 80%
        months = list(range(10, 13)) + list(range(1, 10))

        # Monte Carlo simulation: accumulate results for each trial, each month
        trial_results = []
        for t in range(trials):
            monthly = []
            for m in months:
                # --- Pull month data (handle both 12-row and 48-row) ---
                month_rows = rates[rates["month_num"] == m]
                if month_rows.empty:
                    # Defensive: skip if no data for this month
                    monthly.append({
                        "month": m,
                        "scheduled": 0,
                        "flown": 0,
                        "mc_rate": 0,
                        "execution_rate": 0,
                        "attrition_rate": 0,
                        "break_rate": 0,
                        "gab_rate": 0,
                        "spared_gab_rate": 0,
                        "asd": 0,
                        "avg_poss_ac": 0,
                        "avg_fly_ac": 0,
                        "flyable_ac": 0,
                        "spares_needed": 0,
                        "can_hold_spares": False,
                        "commit_pct": 0,
                        "first_go": 0,
                    })
                    continue

                # If multiple rows, pick a random one (for 48-row support); else use .iloc[0]
                if len(month_rows) > 1:
                    r = month_rows.sample(1).iloc[0]
                else:
                    r = month_rows.iloc[0]

                # --- Add random “noise” to MC and Execution rate for uncertainty ---
                mc_rate = np.clip(np.random.normal(r["mc_rate"], uncertainty), 0, 1)
                exe_rate = np.clip(np.random.normal(r["execution_rate"], uncertainty), 0, 1)

                flyable_ac = max(0, math.floor(TAI - degraders[m]) * mc_rate)
                days = om_days[m]
                pattern = turn_patterns[m].split("x")
                first_go = int(pattern[0])
                sorties_per_day = sum(map(int, pattern))
                scheduled = sorties_per_day * days
                flown = scheduled * exe_rate
                spares_needed = max(1, math.floor(flyable_ac * spares_percent))
                can_hold_spares = (flyable_ac - spares_needed) >= first_go
                commit_pct = (first_go / flyable_ac * 100) if flyable_ac > 0 else 0

                # Store all useful stats for this trial/month
                monthly.append({
                    "month": m,
                    "scheduled": int(scheduled),
                    "flown": int(flown),
                    "mc_rate": mc_rate,
                    "execution_rate": exe_rate,
                    "attrition_rate": r.get("attrition_rate", 0),
                    "break_rate": r.get("break_rate", 0),
                    "gab_rate": r.get("gab_rate", 0),
                    "spared_gab_rate": r.get("spared_gab_rate", 0),
                    "asd": r.get("asd", 0),
                    "avg_poss_ac": r.get("avg_poss_ac", 0),
                    "avg_fly_ac": r.get("avg_fly_ac", 0),
                    "flyable_ac": flyable_ac,
                    "spares_needed": spares_needed,
                    "can_hold_spares": can_hold_spares,
                    "commit_pct": commit_pct,
                    "first_go": first_go,
                })
            trial_results.append(monthly)

        # Now: summarize results per month (mean, CI, etc)
        summary = []
        for idx, m in enumerate(months):
            sched = np.array([trial[idx]["scheduled"] for trial in trial_results])
            flown = np.array([trial[idx]["flown"] for trial in trial_results])
            mc_r  = np.array([trial[idx]["mc_rate"] for trial in trial_results])
            exe_r = np.array([trial[idx]["execution_rate"] for trial in trial_results])
            avg_flyable = np.array([trial[idx]["flyable_ac"] for trial in trial_results])
            overcommit = np.array([trial[idx]["commit_pct"] > commit_thresh for trial in trial_results])
            break_r  = np.array([trial[idx]["break_rate"] for trial in trial_results])
            gab_r    = np.array([trial[idx]["gab_rate"] for trial in trial_results])
            sp_gab_r = np.array([trial[idx]["spared_gab_rate"] for trial in trial_results])
            asd_arr  = np.array([trial[idx]["asd"] for trial in trial_results])

            summary.append({
                "month": m,
                "scheduled_mean": float(np.mean(sched)),
                "scheduled_ci_lo": float(np.percentile(sched, 2.5)),
                "scheduled_ci_hi": float(np.percentile(sched, 97.5)),
                "flown_mean": float(np.mean(flown)),
                "flown_ci_lo": float(np.percentile(flown, 2.5)),
                "flown_ci_hi": float(np.percentile(flown, 97.5)),
                "mc_rate_mean": float(np.mean(mc_r)),
                "execution_rate_mean": float(np.mean(exe_r)),
                "avg_flyable": float(np.mean(avg_flyable)),
                "overcommit_risk": float(np.mean(overcommit)*100),
                # Optional: can add other rates as desired (break, GAB, fixes, etc.)
                "break_rate_mean": float(np.mean(break_r)),
                "gab_rate_mean": float(np.mean(gab_r)),
                "spared_gab_rate_mean": float(np.mean(sp_gab_r)),
                "asd_mean": float(np.mean(asd_arr)),

            })
        return trial_results, [summary]  # (keep in list for compatibility)

    def run(self):
        return self.simulate(trials=500)[0]
