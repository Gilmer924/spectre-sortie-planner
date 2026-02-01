# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 08:59:18 2025

@author: Gilmer Jenkins
"""
# simulations/historical_annual.py

# import numpy as np
# import pandas as pd
# import math
# import calendar
# from simulations.simulation_base import SimulationBase

# class HistoricalAnnualSimulation(SimulationBase):
#     def validate_params(self):
#         required = [
#             "TAI", "rates_df", "om_days", "planned_degraders",
#             "turn_patterns", "commit_rates", "uncertainty"
#         ]
#         missing = [k for k in required if k not in self.params]
#         if missing:
#             raise ValueError("Missing parameter(s): " + ", ".join(missing))

#     def simulate(self, trials=500):
#         self.validate_params()
#         rates = self.params["rates_df"]
#         TAI = self.params["TAI"]
#         om_days = self.params["om_days"]
#         degraders = self.params["planned_degraders"]
#         turn_patterns = self.params["turn_patterns"]
#         commit_rates = self.params["commit_rates"]
#         uncertainty = float(self.params.get("uncertainty", 0.05))  # ±5% default
#         spares_percent = self.params.get("spares_pct", 0.2)  # Default to 20%
#         commit_thresh = self.params.get("commit_thresh", 0.8) * 100  # Default 80%
#         months = list(range(10, 13)) + list(range(1, 10))
#         planned_depot_upnr = self.params.get("planned_depot_upnr", {})
#         planned_deploy_tails = self.params.get("planned_deploy_tails", {})
#         planned_tdy_tails = self.params.get("planned_tdy_tails", {})
#         planned_tdy_hours_per_tail = self.params.get("planned_tdy_hours_per_tail", {})

#         # Monte Carlo simulation: accumulate results for each trial, each month
#         trial_results = []

#         for t in range(trials):
#             monthly = []

#             for m in months:
#                 # --- Pull month data ---
#                 month_rows = rates[rates["month_num"] == m]
#                 if month_rows.empty:
#                     monthly.append({
#                         "month": m,
#                         "scheduled": 0,
#                         "flown": 0,
#                         "mc_rate_hist": 0.0,
#                         "mc_rate_sim": 0.0,
#                         "execution_rate": 0.0,
#                         "flyable_ac": 0,
#                         "possessed_home": 0,
#                         "nmc_sched": 0,
#                         "nmc_unsch": 0.0,
#                         "first_go": 0,
#                     })
#                     continue

#                 r = month_rows.sample(1).iloc[0] if len(month_rows) > 1 else month_rows.iloc[0]

#                 # -----------------------------
#                 # HISTORICAL PERFORMANCE INPUTS
#                 # -----------------------------
#                 mc_hist = float(r.get("mc_rate", 0.0))
#                 exe_rate = np.clip(
#                     np.random.normal(float(r.get("execution_rate", 0.0)), uncertainty),
#                     0, 1
#                 )

#                 break_rate = float(r.get("break_rate", 0.0))

#                 fix8  = float(r.get("fix_rate_8hr", 0.0))
#                 fix12 = float(r.get("fix_rate_12hr", 0.0))
#                 fix24 = float(r.get("fix_rate_24hr", 0.0))

#                 # -----------------------------
#                 # STRUCTURE: HOME-STATION POOL
#                 # -----------------------------
#                 depot_upnr = int(planned_depot_upnr.get(m, 0))
#                 deploy     = int(planned_deploy_tails.get(m, 0))
#                 tdy        = int(planned_tdy_tails.get(m, 0))

#                 possessed_home = max(0, int(TAI) - depot_upnr - deploy - tdy)

#                 # -----------------------------
#                 # PLANNED MAINTENANCE (NMC)
#                 # -----------------------------
#                 nmc_sched = max(0, int(degraders.get(m, 0)))

#                 fix_sum = fix8 + fix12 + fix24
#                 if fix_sum > 0:
#                     p8, p12, p24 = fix8 / fix_sum, fix12 / fix_sum, fix24 / fix_sum
#                     exp_fix_days = (1*p8 + 2*p12 + 3*p24)
#                 else:
#                     exp_fix_days = 2.0  # safe default

#                 days = max(1, int(om_days.get(m, 0)))

#                 pattern = [int(x) for x in str(turn_patterns[m]).split("x") if x.isdigit()]
#                 sorties_per_day_planned = sum(pattern)
#                 scheduled_planned = sorties_per_day_planned * days

#                 monthly_breaks = scheduled_planned * break_rate
#                 nmc_unsch = (monthly_breaks * exp_fix_days) / days

#                 nmc_total = min(float(possessed_home), float(nmc_sched) + float(nmc_unsch))

#                 # -----------------------------
#                 # SIMULATED MC (THIS IS THE FIX)
#                 # -----------------------------
#                 mc_sim = (
#                     (possessed_home - nmc_total) / possessed_home
#                     if possessed_home > 0 else 0.0
#                 )
#                 mc_sim = float(np.clip(mc_sim, 0, 1))

#                 flyable_ac = int(math.floor(possessed_home * mc_sim))

#                 # -----------------------------
#                 # SORTIE CAPACITY
#                 # -----------------------------
#                 first_go_planned = pattern[0] if pattern else 0
#                 commit_cap = min(max(float(commit_rates.get(m, 0.65)), 0.0), 1.0)

#                 first_go_cap = int(math.floor(flyable_ac * commit_cap))
#                 first_go = min(first_go_planned, first_go_cap)

#                 tf = (sorties_per_day_planned / first_go_planned) if first_go_planned > 0 else 0.0
#                 sorties_per_day = int(min(sorties_per_day_planned, math.floor(first_go * tf)))

#                 scheduled = sorties_per_day * days
#                 flown = int(math.floor(scheduled * exe_rate))

#                 # -----------------------------
#                 # STORE RESULTS
#                 # -----------------------------
#                 monthly.append({
#                     "month": m,

#                     # Core outputs
#                     "scheduled": scheduled,
#                     "flown": flown,

#                     # MC lines (SEPARATED, BY DESIGN)
#                     "mc_rate_hist": mc_hist,
#                     "mc_rate_sim": mc_sim,

#                     # Structure & maintenance
#                     "possessed_home": possessed_home,
#                     "nmc_sched": nmc_sched,
#                     "nmc_unsch": nmc_unsch,
#                     "flyable_ac": flyable_ac,

#                     # I added these back in
#                     "asd": float(r.get("asd", 0.0)),
#                     "break_rate": float(r.get("break_rate", 0.0)),
#                     "gab_rate": float(r.get("gab_rate", 0.0)),
#                     "spared_gab_rate": float(r.get("spared_gab_rate", 0.0)),

#                     # Planning context
#                     "first_go": first_go,
#                     "execution_rate": exe_rate,
#                 })

#             trial_results.append(monthly)

#         # Now: summarize results per month (mean, CI, etc)
#         summary = []
#         for idx, m in enumerate(months):
#             sched = np.array([trial[idx]["scheduled"] for trial in trial_results], dtype=float)
#             flown = np.array([trial[idx]["flown"] for trial in trial_results], dtype=float)
        
#             # New: split MC concepts
#             mc_hist = np.array([trial[idx].get("mc_rate_hist", np.nan) for trial in trial_results], dtype=float)
#             mc_sim  = np.array([trial[idx].get("mc_rate_sim", np.nan)  for trial in trial_results], dtype=float)
        
#             exe_r = np.array([trial[idx].get("execution_rate", np.nan) for trial in trial_results], dtype=float)
        
#             flyable = np.array([trial[idx].get("flyable_ac", 0) for trial in trial_results], dtype=float)
        
#             poss_home = np.array([trial[idx].get("possessed_home", 0) for trial in trial_results], dtype=float)
#             nmc_sched = np.array([trial[idx].get("nmc_sched", 0) for trial in trial_results], dtype=float)
#             nmc_unsch = np.array([trial[idx].get("nmc_unsch", 0) for trial in trial_results], dtype=float)
        
#             first_go = np.array([trial[idx].get("first_go", 0) for trial in trial_results], dtype=float)
        
#             # Commit percent derived (no need to store commit_pct in each trial)
#             commit_pct = np.where(flyable > 0, (first_go / flyable) * 100.0, np.nan)
#             overcommit_risk = float(np.nanmean(commit_pct > commit_thresh) * 100.0) if np.isfinite(commit_pct).any() else float("nan")

#             asd_arr = np.array([trial[idx].get("asd", np.nan) for trial in trial_results], dtype=float)
#             break_r = np.array([trial[idx].get("break_rate", np.nan) for trial in trial_results], dtype=float)
#             gab_r = np.array([trial[idx].get("gab_rate", np.nan) for trial in trial_results], dtype=float)
#             sp_gab_r = np.array([trial[idx].get("spared_gab_rate", np.nan) for trial in trial_results], dtype=float)


#             summary.append({
#                 "month": m,

#                 "scheduled_mean": float(np.mean(sched)),
#                 "scheduled_ci_lo": float(np.percentile(sched, 2.5)),
#                 "scheduled_ci_hi": float(np.percentile(sched, 97.5)),

#                 "flown_mean": float(np.mean(flown)),
#                 "flown_ci_lo": float(np.percentile(flown, 2.5)),
#                 "flown_ci_hi": float(np.percentile(flown, 97.5)),

#                 # MC: historical vs simulated (this is what you wanted for the chart cleanup)
#                 "mc_hist_mean": float(np.nanmean(mc_hist)) if np.isfinite(mc_hist).any() else float("nan"),
#                 "mc_sim_mean":  float(np.nanmean(mc_sim))  if np.isfinite(mc_sim).any()  else float("nan"),

#                 "execution_rate_mean": float(np.nanmean(exe_r)) if np.isfinite(exe_r).any() else float("nan"),

#                 "possessed_home_mean": float(np.mean(poss_home)),
#                 "flyable_mean": float(np.mean(flyable)),

#                 # I added these back in as well
#                 "asd_mean": float(np.nanmean(asd_arr)) if np.isfinite(asd_arr).any() else float("nan"),
#                 "break_rate_mean": float(np.nanmean(break_r)) if np.isfinite(break_r).any() else float("nan"),
#                 "gab_rate_mean": float(np.nanmean(gab_r)) if np.isfinite(gab_r).any() else float("nan"),
#                 "spared_gab_rate_mean": float(np.nanmean(sp_gab_r)) if np.isfinite(sp_gab_r).any() else float("nan"),

#                 "nmc_sched_mean": float(np.mean(nmc_sched)),
#                 "nmc_unsch_mean": float(np.mean(nmc_unsch)),

#                 "first_go_mean": float(np.mean(first_go)),
#                 "commit_pct_mean": float(np.nanmean(commit_pct)) if np.isfinite(commit_pct).any() else float("nan"),
#                 "overcommit_risk": overcommit_risk,
#             })

#         return trial_results, [summary]  # keep in list for compatibility

#     def run(self):
#         trials = int(self.params.get("trials", 500))
#         return self.simulate(trials=trials)  # return trial_results only (back-compat)

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
        
        # --- Local Variable Extraction ---
        rates = self.params["rates_df"]
        TAI = self.safe_access(self.params, "TAI", min_val=1.0)
        om_days = self.params["om_days"]
        degraders = self.params["planned_degraders"]
        turn_patterns = self.params["turn_patterns"]
        commit_rates = self.params["commit_rates"]
        
        uncertainty = self.safe_access(self.params, "uncertainty", default=0.05)
        commit_thresh = self.safe_access(self.params, "commit_thresh", default=0.8) * 100
        
        # Preserve original months sequence
        months = list(range(10, 13)) + list(range(1, 10))

        planned_degraders = self.params.get("planned_degraders", {})
        planned_depot_upnr = self.params.get("planned_depot_upnr", {})
        planned_deploy_tails = self.params.get("planned_deploy_tails", {})
        planned_tdy_tails = self.params.get("planned_tdy_tails", {})

        trial_results = []

        # Suppress numpy division warnings globally for the duration of the loop
        with np.errstate(divide='ignore', invalid='ignore'):
            for t in range(trials):
                monthly = []

                for m in months:
                    # --- 1. Robust Month Matching ---
                    m_int = int(m)
                    # Convert to string to ensure we match even if types vary in the dataframe
                    month_rows = rates[rates["month_num"].astype(str).str.contains(str(m_int))]

                    if month_rows.empty:
                        # FALLBACK: Use fleet averages if month is missing
                        if not rates.empty:
                            r = rates.mean(numeric_only=True)
                        else:
                            monthly.append(self._create_empty_month(m))
                            continue
                    else:
                        r = month_rows.sample(1).iloc[0]

                    # --- 2. Historical Inputs (The "Safe" Extraction) ---
                    mc_hist = self.safe_access(r, "mc_rate", default=0.75)
                    
                    # Keep Execution Rate! 
                    exe_base = self.safe_access(r, "execution_rate", default=0.95)
                    exe_rate = np.clip(np.random.normal(float(exe_base), float(uncertainty)), 0, 1)

                    # Scale Break Rate (Handle 25.0 vs 0.25)
                    br_raw = self.safe_access(r, "break_rate", default=0.05)
                    break_rate = br_raw / 100.0 if br_raw > 1.0 else br_raw

                    # --- 3. Fix Rates & Exp Fix Days (Scaled) ---
                    f8 = self.safe_access(r, "fix_rate_8hr", default=0.20)
                    f8 = f8 / 100.0 if f8 > 1.0 else f8
                    
                    f12 = self.safe_access(r, "fix_rate_12hr", default=0.30)
                    f12 = f12 / 100.0 if f12 > 1.0 else f12
                    
                    f24 = self.safe_access(r, "fix_rate_24hr", default=0.20)
                    f24 = f24 / 100.0 if f24 > 1.0 else f24

                    fix_sum = f8 + f12 + f24
                    # Calculate average days an aircraft stays in maintenance
                    exp_fix_days = (1*f8 + 2*f12 + 3*f24) / fix_sum if fix_sum > 1e-9 else 2.0

                    # --- 3. Structural Availability (Corrected for self.params) ---
                    # Pull values from the dictionaries passed in self.params
                    # We use the key names defined in your web_app.py sim_params
                    
                    depot   = self.safe_access(self.params.get("planned_depot_upnr", {}), m)
                    deploy  = self.safe_access(self.params.get("planned_deploy_tails", {}), m)
                    tdy     = self.safe_access(self.params.get("planned_tdy_tails", {}), m)
                    
                    # This captures your "Scheduled MX Tails" from the UI
                    nmc_sched = self.safe_access(self.params.get("planned_degraders", {}), m)
                    
                    # Monthly O&M Days
                    days = self.safe_access(self.params.get("om_days", {}), m, min_val=1.0)

                    # Calculate possessed aircraft remaining at home station
                    possessed_home = max(0.0, float(TAI) - depot - deploy - tdy)

                    # --- REWRITTEN SECTION 4 & 5: FLOW-BASED MC LOGIC ---
                    
                    # 4. Turn Pattern & Maintenance Flow
                    pattern_str = str(turn_patterns.get(m, "0"))
                    pattern = [int(x) for x in pattern_str.split("x") if x.isdigit()] or [0]
                    sorties_planned_day = sum(pattern)
                    
                    # Instead of calculating NMC for the whole month, calculate the "Daily Hangar Load"
                    # nmc_unsch represents the average number of tails down for unscheduled mx on any given day.
                    # (Daily Sorties * Break Rate) = Tails entering hangar per day
                    # (Tails entering * Exp Fix Days) = Average tails staying in hangar
                    nmc_unsch = sorties_planned_day * break_rate * exp_fix_days
                    
                    # Total NMC cannot exceed the tails we actually have at home
                    nmc_total = min(possessed_home, nmc_sched + nmc_unsch)
                    
                    # 5. Simulated MC & Flyable (The "First Go" Capacity)
                    if possessed_home > 0:
                        # mc_sim is the percentage of possessed aircraft NOT in the hangar
                        mc_sim = (possessed_home - nmc_total) / possessed_home
                    else:
                        mc_sim = 0.0
                    
                    mc_sim = float(np.clip(mc_sim, 0, 1))
                    
                    # flyable_ac represents the tails ready for the Morning Go
                    flyable_ac = int(math.floor(possessed_home * mc_sim))
                    
                    # 6. Sortie Capacity & Turn-Forward (RECOVERY CHECK)
                    first_go_planned = pattern[0]
                    commit_cap = np.clip(self.safe_access(commit_rates, m, default=0.65), 0, 1)
                    
                    # We can only launch what we have flyable, capped by our commitment rate
                    first_go = min(first_go_planned, int(math.floor(flyable_ac * commit_cap)))
                    
                    # IMPORTANT: If first_go is 0 because of a high break rate, 
                    # the sim should still show the 'Potential' if maintenance were to catch up.
                    # We ensure the scheduled math doesn't drop to zero unless flyable_ac is truly 0.
                    tf = (sorties_planned_day / first_go_planned) if first_go_planned > 0 else 0.0
                    scheduled = int(min(sorties_planned_day * days, math.floor(first_go * tf * days)))
                    flown = int(math.floor(scheduled * exe_rate))

                    tf = (sorties_planned_day / first_go_planned) if first_go_planned > 0 else 0.0
                    scheduled = int(min(sorties_planned_day * days, math.floor(first_go * tf * days)))
                    flown = int(math.floor(scheduled * exe_rate))

                    # --- Section 7: Maintenance Recovery & Health Checks ---

                    # Calculate how many landed broken and where they went
                    total_breaks_month = flown * break_rate
                    f8_count = total_breaks_month * f8
                    f12_count = total_breaks_month * f12
                    f24_count = total_breaks_month * f24
                    # Anything not fixed in 24 hours is a 'Long Fix'
                    long_fixes = max(0, total_breaks_month - (f8_count + f12_count + f24_count))

                    # Logic for Alerts
                    warnings = []
                    hangar_load_pct = (nmc_total / possessed_home) if possessed_home > 0 else 0.0
                    
                    if hangar_load_pct > 0.85:
                        warnings.append("CRITICAL: Fleet Saturated (Hangar > 85%)")
                    elif hangar_load_pct > 0.60:
                        warnings.append("CAUTION: High Maintenance Backlog")
                    
                    if flyable_ac < (pattern[0] if pattern else 1):
                        warnings.append("INSUFFICIENT ASSETS: Cannot meet Morning Go")

                    # --- Section 8: Final Package (In simulate) ---
                    monthly.append({
                        "month": m,
                        "nmc_sched": float(nmc_sched),
                        "nmc_unsch": float(nmc_unsch),
                        "scheduled": scheduled,
                        "flown": flown,
                        "mc_hist": mc_hist,
                        "mc_sim": mc_sim,
                        "flyable_ac": flyable_ac,
                        
                        # --- ADD THESE LINES ---
                        "depot": depot,      # Saving the value we used (e.g., 2)
                        "deploy": deploy,    # Saving the value we used (e.g., 1)
                        "tdy": tdy,          # Saving the value we used
                        # -----------------------

                        "hangar_load_pct": round(hangar_load_pct * 100, 1),
                        "fixes_8hr": round(f8_count, 1),
                        "fixes_12hr": round(f12_count, 1),
                        "fixes_24hr": round(f24_count, 1),
                        "long_fixes": round(long_fixes, 1),
                        "warnings": " | ".join(warnings) if warnings else "Healthy"
                    })

                trial_results.append(monthly)

        return trial_results, [self._summarize(trial_results, months, commit_thresh)]

    def _create_empty_month(self, m):
        return {
            "month": m, "scheduled": 0, "flown": 0, "mc_rate_hist": 0.0, "mc_rate_sim": 0.0,
            "possessed_home": 0, "nmc_sched": 0, "nmc_unsch": 0.0, "flyable_ac": 0, "first_go": 0
        }

    def _summarize(self, trial_results, months, commit_thresh):
        summary = []
        if not trial_results:
            return summary

        for idx, m in enumerate(months):
            # Helper to safely grab numeric lists across all trials
            def get_vector(key, default=np.nan):
                return np.array([t[idx].get(key, default) for t in trial_results if idx < len(t)], dtype=float)

            # 1. CORE PERFORMANCE VECTORS
            sched    = get_vector("scheduled", 0)
            flown    = get_vector("flown", 0)
            flyable  = get_vector("flyable_ac", 0)
            days_vec = get_vector("days", 20)
            
            # 2. PLANNING INPUT VECTORS 
            # CHANGE THESE to match the keys we just added to monthly.append
            depot_vec  = get_vector("depot", 0)   # Was "planned_depot_upnr"
            deploy_vec = get_vector("deploy", 0)  # Was "planned_deploy_tails"
            tdy_vec    = get_vector("tdy", 0)     # Was "planned_tdy_tails"

            # 3. MAINTENANCE & STATS VECTORS
            mc_sim_vec = get_vector("mc_sim")
            mc_hist_vec = get_vector("mc_hist")
            asd_vec     = get_vector("asd")
            nmc_s       = get_vector("nmc_sched", 0)
            nmc_u       = get_vector("nmc_unsch", 0)

            # 4. FIX CYCLE VECTORS
            f8_vec   = get_vector("fixes_8hr", 0)
            f12_vec  = get_vector("fixes_12hr", 0)
            f24_vec  = get_vector("fixes_24hr", 0)
            long_vec = get_vector("long_fixes", 0)
            load_vec = get_vector("hangar_load_pct", 0)

            # 5. COMMIT MATH (Daily Demand vs Daily Supply)
            daily_req_sorties = np.divide(sched, days_vec, out=np.zeros_like(sched), where=days_vec > 0)
            commit_pct = np.divide(daily_req_sorties * 100.0, flyable, 
                                   out=np.zeros_like(flyable), 
                                   where=flyable > 0)

            valid_commit = commit_pct[np.isfinite(commit_pct)]
            over_risk = float(np.mean(valid_commit > commit_thresh) * 100.0) if len(valid_commit) > 0 else 0.0
            
            # Debugging Output
            print(f"Month {m} - Avg Flyable: {np.mean(flyable):.1f} | Avg Commit: {np.mean(valid_commit) if len(valid_commit)>0 else 0:.1f}%")

            # 6. BUILD SUMMARY
            summary.append({
                "month": m,
                "month_name": calendar.month_abbr[m] if isinstance(m, int) else str(m),
                "scheduled_mean": float(np.nanmean(sched)),
                "flown_mean": float(np.nanmean(flown)),
                "mc_sim": float(np.nanmean(mc_sim_vec)) if not np.all(np.isnan(mc_sim_vec)) else 0.0,
                "mc_hist": float(np.nanmean(mc_hist_vec)) if not np.all(np.isnan(mc_hist_vec)) else 0.0,
                "flyable_mean": float(np.nanmean(flyable)),
                "commit_pct_mean": float(np.mean(valid_commit)) if len(valid_commit) > 0 else 0.0,
                "overcommit_risk": over_risk,
                "asd_mean": float(np.nanmean(asd_vec)) if not np.all(np.isnan(asd_vec)) else 2.0,
                "nmc_sched_mean": float(np.nanmean(nmc_s)),
                "nmc_unsch_mean": float(np.nanmean(nmc_u)),

                # Map to keys the Sand Chart expects
                "depot_mean": float(np.nanmean(depot_vec)),
                "deployment_mean": float(np.nanmean(deploy_vec)),
                "tdy_mean": float(np.nanmean(tdy_vec)),

                "fixes_8hr_mean": float(np.nanmean(f8_vec)),
                "fixes_12hr_mean": float(np.nanmean(f12_vec)),
                "fixes_24hr_mean": float(np.nanmean(f24_vec)),
                "long_fixes_mean": float(np.nanmean(long_vec)),
                "hangar_load_pct_mean": float(np.nanmean(load_vec))
            })

        return summary

    def run(self):
        trials = int(self.params.get("trials", 500))
        return self.simulate(trials=trials)
