# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 08:59:18 2025

@author: Gilmer Jenkins
"""
# simulations/historical_annual.py

import numpy as np
import pandas as pd
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
        planned_depot_upnr = self.params.get("planned_depot_upnr", {})
        planned_deploy_tails = self.params.get("planned_deploy_tails", {})
        planned_tdy_tails = self.params.get("planned_tdy_tails", {})
        planned_tdy_hours_per_tail = self.params.get("planned_tdy_hours_per_tail", {})

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
                mc_rate = np.clip(np.random.normal(float(r.get("mc_rate", 0.0)), uncertainty), 0, 1)
                exe_rate = np.clip(np.random.normal(float(r.get("execution_rate", 0.0)), uncertainty), 0, 1)
                
                # AA rate is tracked for context / validation (NOT the primary flyable driver).
                # If aa_rate exists in the historical file, we carry it through summaries.
                aa_hist = r.get("aa_rate", np.nan)
                aa_rate = float(aa_hist) if pd.notna(aa_hist) else np.nan
                
                # Planning bins reduce available assigned tails at home (STRUCTURE).
                # Performance is then applied via historical MC rate.
                planned_depot_upnr = self.params.get("planned_depot_upnr", {})
                depot_upnr_tails = int(planned_depot_upnr.get(m, 0))
                
                tai_home = max(0, int(TAI) - int(degraders[m]) - depot_upnr_tails)
                
                # Flyable supply for scheduling is MC-driven on home-station assigned tails:
                flyable_ac = max(0, math.floor(tai_home * mc_rate))

                days = om_days[m]
                pattern = [int(x) for x in str(turn_patterns[m]).lower().split("x") if x.strip().isdigit()]
                first_go_planned = int(pattern[0]) if pattern else 0
                sorties_per_day_planned = int(sum(pattern)) if pattern else 0
                
                # Implied TF from the plan (sorties per committed tail)
                tf_implied = (sorties_per_day_planned / first_go_planned) if first_go_planned > 0 else 0.0
                
                # Commit cap (from planning table)
                commit_cap = float(commit_rates.get(m, 0.65))
                commit_cap = min(max(commit_cap, 0.0), 1.0)
                
                # Max first-go you can actually commit given flyable supply
                first_go_cap = int(math.floor(flyable_ac * commit_cap)) if flyable_ac > 0 else 0
                
                # Achievable first-go cannot exceed plan
                first_go = min(first_go_planned, first_go_cap)
                
                # Capacity-limited sorties/day (do not exceed planned)
                sorties_per_day_cap = first_go * tf_implied
                sorties_per_day_sched = int(min(sorties_per_day_planned, math.floor(sorties_per_day_cap)))
                
                scheduled = sorties_per_day_sched * days
                flown = int(math.floor(scheduled * exe_rate))

                # Store all useful stats for this trial/month
                monthly.append({
                    "month": m,

                    # Core outputs
                    "scheduled": int(scheduled),
                    "flown": int(flown),

                    # Rates
                    "mc_rate": mc_rate,
                    "aa_rate": aa_rate,
                    "execution_rate": exe_rate,
                    "attrition_rate": r.get("attrition_rate", 0),

                    # Maintenance context
                    "break_rate": r.get("break_rate", 0),
                    "gab_rate": r.get("gab_rate", 0),
                    "spared_gab_rate": r.get("spared_gab_rate", 0),

                    # Flying characteristics
                    "asd": r.get("asd", 0),

                    # Historical context
                    "avg_poss_ac": r.get("avg_poss_ac", 0),
                    "avg_fly_ac": r.get("avg_fly_ac", 0),

                    # Supply (this is now authoritative)
                    "flyable_ac": int(flyable_ac),
                
                    # Pattern info
                    "first_go": int(first_go),
                })

            trial_results.append(monthly)

        # Now: summarize results per month (mean, CI, etc)
        summary = []
        for idx, m in enumerate(months):
            sched = np.array([trial[idx]["scheduled"] for trial in trial_results])
            flown = np.array([trial[idx]["flown"] for trial in trial_results])
            mc_r  = np.array([trial[idx]["mc_rate"] for trial in trial_results])
            aa_r = np.array([trial[idx].get("aa_rate", np.nan) for trial in trial_results], dtype=float)
            aa_rate_mean = float(np.nanmean(aa_r)) if np.isfinite(aa_r).any() else float("nan")
            exe_r = np.array([trial[idx]["execution_rate"] for trial in trial_results])
            avg_flyable = np.array([trial[idx].get("flyable_ac", 0) for trial in trial_results])
            
            # commit_pct may not exist (we removed it while we finalize supply-constrained logic)
            commit_arr = np.array([trial[idx].get("commit_pct", np.nan) for trial in trial_results], dtype=float)
            if np.isfinite(commit_arr).any():
                overcommit = np.array(commit_arr > commit_thresh, dtype=float)
                overcommit_risk = float(np.mean(overcommit) * 100.0)
            else:
                overcommit_risk = float("nan")

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
                "aa_rate_mean": aa_rate_mean,
                "execution_rate_mean": float(np.mean(exe_r)),
                "avg_flyable": float(np.mean(avg_flyable)),
                "overcommit_risk": overcommit_risk,

                # Optional: can add other rates as desired (break, GAB, fixes, etc.)
                "break_rate_mean": float(np.mean(break_r)),
                "gab_rate_mean": float(np.mean(gab_r)),
                "spared_gab_rate_mean": float(np.mean(sp_gab_r)),
                "asd_mean": float(np.mean(asd_arr)),

            })
        return trial_results, [summary]  # (keep in list for compatibility)

    def run(self):
        trials = int(self.params.get("trials", 500))
        return self.simulate(trials=trials)[0]  # return trial_results only (back-compat)
