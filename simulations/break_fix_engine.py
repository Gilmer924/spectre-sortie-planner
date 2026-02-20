# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 19:28:12 2026

@author: scwri
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import date, timedelta

# ----------------------------
# Helpers
# ----------------------------

MONTH_ORDER = ["Oct","Nov","Dec","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep"]
MONTH_DAYS  = {"Oct":31,"Nov":30,"Dec":31,"Jan":31,"Feb":28,"Mar":31,"Apr":30,"May":31,"Jun":30,"Jul":31,"Aug":31,"Sep":30}

def fy26_month_to_calendar(month_key):
    """
    Accepts:
      - "Oct", "Nov", ... (FY style)
      - 10, 11, 12, 1..9 (int)
      - "10", "11", ... (numeric str)
      - "October", ... (full month name)
    Returns (year, month_num). Year is FY26 assumption as currently designed.
    """
    # FY26 mapping assumption (Oct=2025, Jan=2026)
    fy_map = {"Oct":10,"Nov":11,"Dec":12,"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,"Jul":7,"Aug":8,"Sep":9}

    # normalize
    if month_key is None:
        raise KeyError("month_key is None")

    # ints
    if isinstance(month_key, (int, np.integer)):
        mnum = int(month_key)
    else:
        s = str(month_key).strip()

        # numeric strings like "10"
        if s.isdigit():
            mnum = int(s)
        else:
            # try FY abbrev first
            s3 = s[:3].title()
            if s3 in fy_map:
                mnum = fy_map[s3]
            else:
                # try full month names -> month number
                import calendar
                name_to_num = {name: i for i, name in enumerate(calendar.month_name) if name}
                mnum = name_to_num.get(s.title())
                if mnum is None:
                    raise KeyError(s)

    year = 2025 if mnum in (10, 11, 12) else 2026
    return year, mnum

def build_fly_calendar(year: int, month: int, add_one_weekend: bool = True,
                       holidays: set[date] | None = None,
                       family_days: set[date] | None = None) -> list[date]:
    holidays = holidays or set()
    family_days = family_days or set()

    # all dates in the month
    d = date(year, month, 1)
    dates = []
    while d.month == month:
        dates.append(d)
        d += timedelta(days=1)

    # baseline: Mon–Fri
    fly = [x for x in dates if x.weekday() < 5]

    # remove holidays / family days
    fly = [x for x in fly if x not in holidays and x not in family_days]

    # add 1 weekend (Sat+Sun) closest to mid-month
    if add_one_weekend:
        weekends = [x for x in dates if x.weekday() in (5, 6)]
        if weekends:
            mid = dates[len(dates)//2]
            candidate = min(weekends, key=lambda x: abs((x - mid).days))
            partner = candidate + timedelta(days=1) if candidate.weekday() == 5 else candidate - timedelta(days=1)
            fly.extend([candidate])
            if partner.month == month:
                fly.append(partner)
            fly = sorted(set(fly))

    return fly

# year, month = 2025, 10
# fly_dates = build_fly_calendar(year, month)

# print([d.strftime("%a %d-%b") for d in fly_dates])
# print("Fly days:", len(fly_dates))

def build_attempt_plan_from_calendar(
    days_in_month: int,
    fly_dates: list[date],
    attempted_sorties_month: int
) -> tuple[list[int], list[bool]]:
    plan = [0] * days_in_month
    fly_mask = [False] * days_in_month

    if attempted_sorties_month <= 0 or not fly_dates:
        return plan, fly_mask

    # mark fly days
    for dt in fly_dates:
        fly_mask[dt.day - 1] = True

    # spread monthly attempts across fly days
    n = len(fly_dates)
    base = attempted_sorties_month // n
    rem  = attempted_sorties_month % n

    for i, dt in enumerate(sorted(fly_dates)):
        idx = dt.day - 1
        plan[idx] = base + (1 if i < rem else 0)

    return plan, fly_mask

@dataclass(frozen=True)
class YearParams:
    paa: int
    commit_rate: float  # e.g. 0.65
    seed: int = 42

# ----------------------------
# Core day-step engine (break/fix + ledger)
# ----------------------------

# 1) Print semantics fix: show UnsDown_used_for_avail explicitly
# 2) Attempt capacity clamp based on flyers_today
# 3) GA spared cap based on spares_reserved_today
# 4) Spare policy: hybrid (min 1 spare + GA-driven pad, capped at 20% of PAA)

def run_month(
    rng: np.random.Generator,
    month_key: str,
    days: int,
    paa: int,
    commit_rate: float,
    scheduled_overhead_tails: int,
    om_days: int,
    attempted_sorties_month: int,
    attrition_rate: float,
    break_rate_per_executed: float,
    ga_rate_per_attempt: float,
    p_ga_spared: float,
    fix_p1: float, fix_p2: float, fix_p3: float,
    # carryover pipeline from prior month
    r1: int, r2: int, r3: int, rL: int,
    # long-repair behavior knobs
    p_go_long: float = 0.15,
    p_long_return: float = 0.25,
    # spare policy knobs (POC defaults; can move to planning inputs later)
    spare_cap_pct: float = 0.20,     # doctrine: don't exceed 20% of PAA as spares
    min_spares: int = 1,             # typical default
    spare_pad_factor: float = 2.0,  # >1.0 pads spares above expected GA volume
    sorties_per_jet_per_day: float = 1.5,  # capacity clamp; POC proxy for turn pattern
    print_days: bool = False,
):
    # Commit cap against PAA (constant planning restraint)
    commit_cap = int(np.floor(commit_rate * paa))

    # Daily attempt plan (calendar-driven)
    year, month_num = fy26_month_to_calendar(month_key)

    fly_dates = build_fly_calendar(
        year=year,
        month=month_num,
        add_one_weekend=True,
        holidays=set(),
        family_days=set(),
    )

    attempt_plan, fly_mask = build_attempt_plan_from_calendar(
        days_in_month=days,
        fly_dates=fly_dates,
        attempted_sorties_month=attempted_sorties_month
    )

    # Ledger totals
    ledger = {
        "breaks_in": 0,
        "returns_out": 0,
        "gate_1d": 0,
        "gate_2d": 0,
        "gate_3d": 0,
        "gate_long": 0,

        "ga_total": 0,
        "ga_spared": 0,
        "ga_not_spared": 0,

        "spares_reserved": 0,
        "policy_spares_demand": 0,
        "spares_capped_by_pct_days": 0,
        "spares_capped_by_commit_days": 0,
        "spare_shortfall_events": 0,  # times GA_spared_raw > spares_reserved_today

        "ledger_fail_days": 0,
        "executed_sorties": 0,
        "attempted_sorties": 0,
        "shortfall_days": 0,
        "peak_unsched_down": 0,
    }

    if print_days:
        print(f"\n{month_key}: days={days}  OH_sched={scheduled_overhead_tails}  OM={om_days}  AttemptMo={attempted_sorties_month}")
        print("Day | WIPstart | RetOut | UnsDown_used | Avail | Commit | SpRes | Flyers | Attempt(plan->cap) | GA(T/S/NS) | Exec | BreakIn | Gate(1/2/3/L) | WIPend | OK?")
        print("-" * 160)

    # Pre-clamp key probabilities
    ga_rate = max(0.0, min(1.0, ga_rate_per_attempt))
    p_spared = max(0.0, min(1.0, p_ga_spared))
    brk_rate = max(0.0, min(1.0, break_rate_per_executed))
    fx1 = max(0.0, min(1.0, fix_p1))
    fx2 = max(0.0, min(1.0, fix_p2))
    fx3 = max(0.0, min(1.0, fix_p3))

    for day in range(1, days + 1):
        is_fly_day = fly_mask[day - 1]

        # WIP at start (includes long repairs)
        wip_start = r1 + r2 + r3 + rL

        # ----------------------------
        # Returns / progression
        # only progresses on fly days
        # ----------------------------
        returns_out = 0
        if is_fly_day:
            # normal bins: r1 returns; r2->r1; r3->r2
            returns_out += r1
            r1, r2, r3 = r2, r3, 0

            # long bin returns only on fly days
            if rL > 0:
                long_returns = rng.binomial(rL, max(0.0, min(1.0, p_long_return)))
                rL -= int(long_returns)
                returns_out += int(long_returns)

        # ----- PRINT SEMANTICS FIX -----
        # This is the EXACT unsched_down used to compute avail/commit for today.
        unsched_down_used = r1 + r2 + r3 + rL
        ledger["peak_unsched_down"] = max(ledger["peak_unsched_down"], unsched_down_used)

        # Available tails (structural)
        avail = max(paa - scheduled_overhead_tails - unsched_down_used, 0)

        # Commit cap vs PAA, can't exceed avail
        commit_today = min(commit_cap, avail)

        # Planned attempts from calendar plan
        planned_attempt_today = attempt_plan[day - 1]
        attempted_today = planned_attempt_today

        # Non-fly day: no flying, no attempt
        if not is_fly_day:
            attempted_today = 0

        # If nothing committed, no attempts
        if commit_today <= 0:
            attempted_today = 0

        # ----------------------------
        # NEW: SPARE POLICY (hybrid)
        # ----------------------------
        #   • min 1 spare
        #   • GA-driven pad based on planned attempt load
        #   • capped at 20% of PAA (or whatever spare_cap_pct is)
        #   • never exceed today’s committed aircraft
        # ----------------------------
        spare_cap = int(np.floor(spare_cap_pct * paa))

        # planned_attempt_today should be the pre-cap, fly-day-gated value used for planning
        # (If you don't already have it, set it earlier as: planned_attempt_today = attempt_plan[day - 1])
        expected_ga = planned_attempt_today * ga_rate_per_attempt

        # pad factor lets you be conservative (e.g., 1.0 = 1 spare per expected GA)
        ga_driven_spares = int(np.ceil(spare_pad_factor * expected_ga))

        # Hybrid policy: always at least 1, plus GA-driven pad
        policy_spares = max(min_spares, ga_driven_spares)

        # Reserve spares but never exceed cap or today's committed tails
        spares_reserved_today = min(policy_spares, spare_cap, commit_today)

        flyers_today = max(commit_today - spares_reserved_today, 0)

        # Ledger tracking
        ledger["spares_reserved"] += int(spares_reserved_today)
        ledger["policy_spares_demand"] += int(policy_spares)                 # NEW (optional but useful)
        ledger["spares_capped_by_pct_days"] += int(policy_spares > spare_cap) # NEW
        ledger["spares_capped_by_commit_days"] += int(policy_spares > commit_today) # NEW

        # ----------------------------
        # NEW: ATTEMPT CAPACITY CLAMP
        # ----------------------------
        max_attempt_cap = int(np.floor(flyers_today * sorties_per_jet_per_day))
        if attempted_today > max_attempt_cap:
            attempted_today = max_attempt_cap

        # Track shortfall only against what we ACTUALLY attempted (post clamp)
        # (planned shortfall tracking can be done separately if desired)

        # ----------------------------
        # GA + EXECUTION + BREAKS
        # ----------------------------
        ga_total = 0
        ga_spared_raw = 0
        ga_spared = 0
        ga_not_spared = 0

        if attempted_today > 0:
            ga_total = rng.binomial(attempted_today, ga_rate)

            # "Raw" spared based on historical share
            ga_spared_raw = rng.binomial(ga_total, p_spared)

            # NEW: GA spared cap based on reserved spares
            ga_spared = min(ga_spared_raw, spares_reserved_today)
            if ga_spared_raw > spares_reserved_today:
                ledger["spare_shortfall_events"] += 1

            ga_not_spared = ga_total - ga_spared

        # Launches are attempts that don't become not-spared GA
        launches = max(attempted_today - ga_not_spared, 0)

        # Attrition applies to launches (airborne sortie loss)
        executed_today = rng.binomial(launches, max(0.0, 1.0 - attrition_rate))

        # Maintenance burden:
        #  • ALL GA create a break (something broke on the ground)
        #  • executed sorties can also break post-flight
        breaks_from_ga = ga_total
        breaks_from_land = rng.binomial(executed_today, brk_rate)
        breaks_in = int(breaks_from_ga + breaks_from_land)

        # Assign breaks into normal fix bins vs long bin
        gate1 = gate2 = gate3 = 0
        to_long = 0

        if breaks_in > 0:
            to_long = rng.binomial(breaks_in, max(0.0, min(1.0, p_go_long)))
            remaining = breaks_in - to_long

            if remaining > 0:
                # multinomial requires probs sum to 1; caller normalized monthly
                gate1, gate2, gate3 = rng.multinomial(remaining, [fx1, fx2, fx3])
                r1 += int(gate1)
                r2 += int(gate2)
                r3 += int(gate3)

            rL += int(to_long)
            ledger["gate_long"] += int(to_long)

        # WIP end
        wip_end = r1 + r2 + r3 + rL

        # Ledger conservation check
        expected_wip_end = wip_start - returns_out + breaks_in
        ok = (wip_end == expected_wip_end)
        if not ok:
            ledger["ledger_fail_days"] += 1

        # If executed < attempted (post clamp), count shortfall days
        if executed_today < attempted_today:
            ledger["shortfall_days"] += 1

        # Totals
        ledger["returns_out"] += int(returns_out)

        ledger["ga_total"] += int(ga_total)
        ledger["ga_spared"] += int(ga_spared)
        ledger["ga_not_spared"] += int(ga_not_spared)

        ledger["breaks_in"] += int(breaks_in)
        ledger["gate_1d"] += int(gate1)
        ledger["gate_2d"] += int(gate2)
        ledger["gate_3d"] += int(gate3)

        ledger["attempted_sorties"] += int(attempted_today)
        ledger["executed_sorties"] += int(executed_today)

        if print_days:
            print(
                f"{day:>3} | {wip_start:>7} | {returns_out:>6} | {unsched_down_used:>11} | {avail:>5} | {commit_today:>6} |"
                f" {spares_reserved_today:>5} | {flyers_today:>6} |"
                f" {planned_attempt_today:>6}->{attempted_today:<6} |"
                f" {ga_total:>2}/{ga_spared:>2}/{ga_not_spared:>2}     | {executed_today:>4} | {breaks_in:>7} |"
                f" {int(gate1):>2}/{int(gate2):>2}/{int(gate3):>2} L{int(to_long):>2} | {wip_end:>5} | {'OK' if ok else 'FAIL'}"
            )

    end_wip = r1 + r2 + r3 + rL
    return ledger, (r1, r2, r3, rL), end_wip

def run_fy_once(plan: pd.DataFrame, hist: pd.DataFrame, yp: YearParams, seed: int, print_days: bool = False) -> tuple[pd.DataFrame, dict]:
    """
    Runs Oct–Sep once and returns:
      (1) month_results_df: 12-row dataframe (Oct..Sep)
      (2) year_summary: dict of key annual metrics
    """
    rng = np.random.default_rng(seed)

    # Carryover pipeline across months
    r1 = r2 = r3 = rL = 0

    rows = []
    for m in MONTH_ORDER:
        p = plan.loc[plan["Month"] == m].iloc[0]
        om_days = int(p["O&M Days"])

        scheduled_overhead = int(
            p["Depot/UPNR Tails"] +
            p["Scheduled MX Tails"] +
            p["Deployment Tails"] +
            p["TDY Tails"]
        )

        h = hist.loc[hist["month_key"] == m].iloc[0]

        attempted_sorties_month = int(round(h["sorties_scheduled"]))
        attrition_rate = float(h["Attrition rate"])
        break_rate = float(h["break_rate_per_executed"])
        fix_p1, fix_p2, fix_p3 = float(h["fix_p1"]), float(h["fix_p2"]), float(h["fix_p3"])

        ga_rate = float(h["ga_rate_per_attempt"])
        p_spared = float(h["p_ga_spared"])

        ledger, (r1, r2, r3, rL), end_wip = run_month(
            rng=rng,
            month_key=m,
            days=MONTH_DAYS[m],
            paa=yp.paa,
            commit_rate=yp.commit_rate,
            scheduled_overhead_tails=scheduled_overhead,
            om_days=om_days,
            attempted_sorties_month=attempted_sorties_month,
            attrition_rate=attrition_rate,
            break_rate_per_executed=break_rate,
            ga_rate_per_attempt=ga_rate,
            p_ga_spared=p_spared,
            fix_p1=fix_p1, fix_p2=fix_p2, fix_p3=fix_p3,
            r1=r1, r2=r2, r3=r3, rL=rL,
            print_days=print_days
        )

        rows.append({
            "Month": m,
            "OM_Days": om_days,
            "Sched_OH_Tails": scheduled_overhead,

            "Attempted_Sorties": ledger["attempted_sorties"],
            "Executed_Sorties": ledger["executed_sorties"],

            "Breaks_In": ledger["breaks_in"],
            "Returns_Out": ledger["returns_out"],

            "Gates_1d": ledger["gate_1d"],
            "Gates_2d": ledger["gate_2d"],
            "Gates_3d": ledger["gate_3d"],
            "Gates_Long": ledger["gate_long"],

            # ---- SPARES ----
            "Spares_Reserved": ledger["spares_reserved"],
            "Spares_Policy_Demand": ledger["policy_spares_demand"],
            "Spare_CapPct_Days": ledger["spares_capped_by_pct_days"],
            "Spare_CapCommit_Days": ledger["spares_capped_by_commit_days"],

            "GA_Total": ledger["ga_total"],
            "GA_Spared": ledger["ga_spared"],
            "GA_NotSpared": ledger["ga_not_spared"],

            "End_WIP": end_wip,
            "Peak_Unsched_Down": ledger["peak_unsched_down"],
            "Ledger_Fail_Days": ledger["ledger_fail_days"],
            "Shortfall_Days": ledger["shortfall_days"],
        })

    month_df = pd.DataFrame(rows)
    # print("MONTH_DF COLS:", month_df.columns.tolist())

    # --- Annual rollups (robust to column name drift) ---
    def colsum(df: pd.DataFrame, col: str) -> int:
        return int(df[col].sum()) if col in df.columns else 0

    def colmax(df: pd.DataFrame, col: str) -> int:
        return int(df[col].max()) if col in df.columns else 0

    year_summary = {
        "seed": seed,
        "Executed_Sorties_YR": colsum(month_df, "Executed_Sorties"),
        "Attempted_Sorties_YR": colsum(month_df, "Attempted_Sorties"),
        "Breaks_In_YR": colsum(month_df, "Breaks_In"),
        "GA_Total_YR": colsum(month_df, "GA_Total"),
        "GA_Spared_YR": colsum(month_df, "GA_Spared"),
        "GA_NotSpared_YR": colsum(month_df, "GA_NotSpared"),
        "Spares_Reserved_YR": colsum(month_df, "Spares_Reserved"),
        "Spare_Shortfall_Events_YR": colsum(month_df, "Spare_Shortfall_Events"),
        "Ledger_Fail_Days_YR": colsum(month_df, "Ledger_Fail_Days"),
        "Shortfall_Days_YR": colsum(month_df, "Shortfall_Days"),
        "Peak_Unsched_Down_YR": colmax(month_df, "Peak_Unsched_Down"),
    }

    return month_df, year_summary

def run_monte_carlo(plan: pd.DataFrame, hist: pd.DataFrame, yp: YearParams, n_sims: int = 500, base_seed: int = 1000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs FY n_sims times.
    Returns:
      - year_df: one row per sim with annual totals
      - month_long_df: stacked month results with sim_id for per-month distributions
    """
    year_rows = []
    month_frames = []

    for i in range(n_sims):
        seed = base_seed + i
        # DEBUG: confirm seeds
        if i < 5:
            print("sim", i, "seed", seed)
        month_df, year_summary = run_fy_once(plan, hist, yp, seed=seed, print_days=False)
        year_rows.append(year_summary)

        month_df = month_df.copy()
        month_df["sim_id"] = i
        month_frames.append(month_df)

    year_df = pd.DataFrame(year_rows)
    month_long_df = pd.concat(month_frames, ignore_index=True)

    return year_df, month_long_df

# ----------------------------
# Load your inputs + run Oct–Sep
# ----------------------------

def main():
    planning_csv = "Planning Inputs Test.csv"
    hist_xlsx = "Historical Data Input.xlsx"

    plan = pd.read_csv(planning_csv)
    hist = pd.read_excel(hist_xlsx, sheet_name="Historical Data Input")
    # print("HIST COLUMNS:", list(hist.columns))

    # Normalize month keys to Oct/Nov/...
    # planning CSV already uses Oct/Nov...
    plan["Month"] = plan["Month"].astype(str).str.strip()

    # historical xlsx uses "October", "November", ...
    month_map = {
        "October":"Oct","November":"Nov","December":"Dec","January":"Jan","February":"Feb","March":"Mar",
        "April":"Apr","May":"May","June":"Jun","July":"Jul","August":"Aug","September":"Sep"
    }
    hist["month_key"] = hist["month"].map(month_map)

    # ----------------------------
    # Derive rates from historical sheet (POC)
    # ----------------------------

    # GA-derived rates (must be created BEFORE the month loop)
    hist["ga_rate_per_attempt"] = (hist["ground_aborts"] / hist["sorties_scheduled"]).fillna(0.0).clip(0.0, 1.0)

    hist["p_ga_spared"] = 1.0
    mask = hist["ground_aborts"] > 0
    hist.loc[mask, "p_ga_spared"] = (
        1.0 - (hist.loc[mask, "not_spared_ground_aborts"] / hist.loc[mask, "ground_aborts"])
    ).fillna(1.0).clip(0.0, 1.0)

    # Break/fix rates
    hist["break_rate_per_executed"] = (hist["breaks"] / hist["sorties_flown"]).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)
    hist["fix_p1"] = (hist["fixes_8hr"]  / hist["breaks"]).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)
    hist["fix_p2"] = (hist["fixes_12hr"] / hist["breaks"]).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)
    hist["fix_p3"] = (hist["fixes_24hr"] / hist["breaks"]).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)

    # Normalize fix probabilities so they sum to 1 when possible.
    # If there are zero breaks, default to all fixes in 1-day to avoid NaNs.
    s = (hist["fix_p1"] + hist["fix_p2"] + hist["fix_p3"])
    nonzero = s > 0
    hist.loc[nonzero, "fix_p1"] = hist.loc[nonzero, "fix_p1"] / s[nonzero]
    hist.loc[nonzero, "fix_p2"] = hist.loc[nonzero, "fix_p2"] / s[nonzero]
    hist.loc[nonzero, "fix_p3"] = hist.loc[nonzero, "fix_p3"] / s[nonzero]
    hist.loc[~nonzero, ["fix_p1", "fix_p2", "fix_p3"]] = [1.0, 0.0, 0.0]

    # ----------------------------
    # Sim constants
    # ----------------------------
    yp = YearParams(paa=24, commit_rate=0.65, seed=42)
    rng = np.random.default_rng(yp.seed)

    # Carryover pipeline across months
    r1 = r2 = r3 = rL = 0

    rows = []

    for m in MONTH_ORDER:
        # Planning inputs
        p = plan.loc[plan["Month"] == m].iloc[0]
        om_days = int(p["O&M Days"])

        scheduled_overhead = int(
            p["Depot/UPNR Tails"] +
            p["Scheduled MX Tails"] +
            p["Deployment Tails"] +
            p["TDY Tails"]
        )

        # Historical inputs (this is where 'h' exists!)
        h = hist.loc[hist["month_key"] == m].iloc[0]

        attempted_sorties_month = int(round(h["sorties_scheduled"]))
        attrition_rate = float(h["Attrition rate"])
        break_rate = float(h["break_rate_per_executed"])
        fix_p1, fix_p2, fix_p3 = float(h["fix_p1"]), float(h["fix_p2"]), float(h["fix_p3"])

        ga_rate = float(h["ga_rate_per_attempt"])
        p_spared = float(h["p_ga_spared"])

        ledger, (r1, r2, r3, rL), end_wip = run_month(
            rng=rng,
            month_key=m,
            days=MONTH_DAYS[m],
            paa=yp.paa,
            commit_rate=yp.commit_rate,
            scheduled_overhead_tails=scheduled_overhead,
            om_days=om_days,
            attempted_sorties_month=attempted_sorties_month,
            attrition_rate=attrition_rate,
            break_rate_per_executed=break_rate,
            ga_rate_per_attempt=ga_rate,
            p_ga_spared=p_spared,
            fix_p1=fix_p1, fix_p2=fix_p2, fix_p3=fix_p3,
            r1=r1, r2=r2, r3=r3, rL=rL,
            print_days=False,
        )

        rows.append({
            "Month": m,
            "OM_Days": om_days,
            "Sched_OH_Tails": scheduled_overhead,
            "Attempted_Sorties": ledger["attempted_sorties"],
            "GA_Total": ledger.get("ga_total", 0),
            "GA_Spared": ledger.get("ga_spared", 0),
            "GA_NotSpared": ledger.get("ga_not_spared", 0),
            "Executed_Sorties": ledger["executed_sorties"],
            "Breaks_In": ledger["breaks_in"],
            "Returns_Out": ledger["returns_out"],
            "Gates_1d": ledger["gate_1d"],
            "Gates_2d": ledger["gate_2d"],
            "Gates_3d": ledger["gate_3d"],
            "Gates_Long": ledger["gate_long"],
            "End_WIP": end_wip,
            "Peak_Unsched_Down": ledger["peak_unsched_down"],
            "Ledger_Fail_Days": ledger["ledger_fail_days"],
            "Shortfall_Days": ledger["shortfall_days"],

        })

        # --- Monte Carlo ---
    yp = YearParams(paa=28, commit_rate=0.55, seed=84)

    year_df, month_long_df = run_monte_carlo(
        plan=plan,
        hist=hist,
        yp=yp,
        n_sims=500,      # start with 200-500; bump to 2000 later
        base_seed=9000
    )

    # Commander-friendly percentiles (annual)
    def pct(s, p): 
        return float(np.percentile(s, p))

    print("\n=== MONTE CARLO (Annual) ===")
    for col in ["Executed_Sorties_YR", "Breaks_In_YR", "GA_NotSpared_YR", "Peak_Unsched_Down_YR", "Spares_Reserved_YR", "Spare_Shortfall_Events_YR"]:
        p10 = pct(year_df[col], 10)
        p50 = pct(year_df[col], 50)
        p90 = pct(year_df[col], 90)
        print(f"{col:>24}: P10={p10:7.1f}  P50={p50:7.1f}  P90={p90:7.1f}")

    # Probability-of-meeting style metric (example: executed sorties >= planned attempted)
    year_df["Exec_Rate_YR"] = year_df["Executed_Sorties_YR"] / year_df["Attempted_Sorties_YR"]
    print("\nExecution Rate (Annual) Percentiles:")
    print("Executed samples:", year_df["Executed_Sorties_YR"].head(10).tolist())
    print("Breaks samples:  ", year_df["Breaks_In_YR"].head(10).tolist())
    print("GA_NS samples:   ", year_df["GA_NotSpared_YR"].head(10).tolist())

    for p in [10, 50, 90]:
        val = np.percentile(year_df["Exec_Rate_YR"], p)
        print(f"  P{p}: {val:.3f}")

    # print(f"\nP(Executed >= Attempted) = {prob_meet:.3f}")

    # Save outputs for Excel / quick inspection
    year_df.to_csv("mc_year_summary.csv", index=False)
    month_long_df.to_csv("mc_month_detail.csv", index=False)
    print("\nWrote: mc_year_summary.csv and mc_month_detail.csv")

    # out = pd.DataFrame(rows)
    # print("\n=== POC Oct–Sep Summary ===")
    # print(out.to_string(index=False))

    # bad = out[out["Ledger_Fail_Days"] > 0]
    # if len(bad) > 0:
    #     print("\n!!! Ledger failures detected in months:", bad["Month"].tolist())
    # else:
    #     print("\nLedger check: PASS (no FAIL days). Breaks are cycling through fix gates correctly.")

if __name__ == "__main__":
    main()
