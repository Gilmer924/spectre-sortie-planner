# -*- coding: utf-8 -*-
"""
Streamlit Frontend for SPECTRE Sortie Simulation
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import calendar
from calendar import month_name
import datetime
import math
import re
from scipy import stats
from sim_engine import run_weekly_simulation, run_personnel_simulation, run_historical_simulation
from io import BytesIO
from PIL import Image
# RIM / North Star projection engine
from simulations.rim_projection import RIMProjectionInputs, project_rim_as_dicts
from simulations.rim_requirements import compute_crew_aircraft_requirement


# ---------- Turn Factor Helpers ----------

def compute_turn_factor_from_patterns(patterns_for_week):
    """
    Compute Turn Factor and Turn Load from a list of daily pattern strings, e.g.:
      ["10x8", "8x6", "0x0", ...].

    Definition:
      ΣFirst  = sum of all first-go aircraft across the week
      ΣAll    = sum of all sorties across all goes (first + second + ...)

      Turn Factor (TF) = ΣAll / ΣFirst             (sorties per first-go jet)
      Turn Load        = TF - 1                    (extra sorties per first-go jet)

    Returns:
      (turn_factor, turn_load, total_first, total_all)
    """
    first_total = 0
    all_total   = 0

    for pat in patterns_for_week:
        # Extract all integer groups in the pattern string (handles NxM or NxMxK)
        nums = list(map(int, re.findall(r"(\d+)", str(pat))))
        if not nums:
            continue

        first_go = nums[0]
        other_go = nums[1:]  # all additional goes, if present

        first_total += first_go
        all_total   += first_go + sum(other_go)

    if first_total <= 0:
        return float("nan"), float("nan"), 0, all_total

    tf = all_total / first_total
    load = tf - 1.0
    return tf, load, first_total, all_total

# ---------- Page Setup ----------
st.set_page_config(page_title="SPECTRE Sortie Planner", layout="wide")

st.markdown(
    "<h1 style='text-align: center;'>✈️ SPECTRE: Sortie Planner & Analysis Suite</h1>",
    unsafe_allow_html=True
)

# ---------- Load Logo ----------
try:
    logo = Image.open("spectre_logo.png")
    st.image(logo, width=200)
except FileNotFoundError:
    st.warning("Logo not found. Add 'spectre_logo.png' to app folder.")


# #------ RIM Toggle -------- REMOVED
# st.sidebar.markdown("### Experimental Modules")
# enable_rim_ns = st.sidebar.checkbox("Enable RIM / North Star (beta)", value=False)

#          Sidebar section label           
st.sidebar.markdown(" Simulation Modules ")

# ---------- Simulation Selector ----------
sim_options = [
    "Weekly Simulation",
    "Personnel Simulation",
    "Quick Probability Analysis",
    "Annual Historical Simulation",
    "RIM / North Star Analysis (beta)",
]
sim_choice = st.sidebar.selectbox("Choose Simulation Module", sim_options)

# ---------- WEEKLY SIMULATION ----------
if sim_choice == "Weekly Simulation":
    st.header("🗓️ Weekly Sortie Simulation")

    # --- Sidebar Inputs: Core structure (no inline RIM here) ---
    TAI = st.sidebar.number_input(
        "Total AC Inventory (TAI)",
        value=12,
        step=1,
        key="weekly_tai",
        help="Total aircraft on the books for this unit (training + backup)."
    )

    Overhead = st.sidebar.number_input(
        "Overhead AC (NMC total)",
        value=2,
        step=1,
        key="weekly_overhead",
        help="Total NMC aircraft that reduce effective possessed (EP) in this week."
    )

    ASD       = st.sidebar.number_input("Avg Sortie Duration (hrs)", value=1.9)
    Break_pct = st.sidebar.slider("Break Rate (%)", 0.0, 100.0, 16.6)
    GA_pct    = st.sidebar.slider("Ground Abort Rate (%)", 0.0, 100.0, 7.5)
    Fix8      = st.sidebar.slider("Fix Rate 8 Hr (%)", 0.0, 100.0, 25.0)
    Fix12     = st.sidebar.slider("Fix Rate 12 Hr (%)", 0.0, 100.0, 35.0)
    Fix24     = st.sidebar.slider("Fix Rate 24 Hr (%)", 0.0, 100.0, 45.0)
    WxAttr    = st.sidebar.slider("Weather Attrition (%)", 0.0, 100.0, 0.0)
    SortieAttr= st.sidebar.slider("Sortie Attrition (%)", 0.0, 100.0, 20.0)

    Trials    = st.sidebar.slider(
        "MC Trials",
        min_value=100,
        max_value=1000,
        value=500,
        step=100,
    )

    commit_thresh = st.sidebar.slider(
        "Commitment Rate Threshold (%)",
        min_value=0.0,
        max_value=100.0,
        value=65.0,
        help="Warn when daily commit rate exceeds this % of available aircraft"
    )

    # ⬇️ leave all your existing turn-pattern inputs, run button, and
    #     results / charts code as-is below this point

    # ── RIM / NS structure (does NOT change sim_engine yet) ───────────────
    # if enable_rim_ns:
    #     st.sidebar.markdown("**RIM Structure (Experimental)**")

    #     PAI = st.sidebar.number_input(
    #         "PAI (Primary Aircraft Inventory)",
    #         value=TAI,
    #         step=1,
    #         key="rim_pai",
    #         help="Primary aircraft normally used for training/ops."
    #     )
    #     # BAI derived from TAI - PAI
    #     BAI = max(int(TAI) - int(PAI), 0)
    #     st.sidebar.caption(f"BAI (Backup Aircraft Inventory) assumed: {BAI} = TAI − PAI")

    #     st.sidebar.markdown("**Degrader Bins (Overhead Detail)**")
    #     rim_nmcm = st.sidebar.number_input(
    #         "NMCM (NMC Maintenance)",
    #         value=0,
    #         step=1,
    #         key="rim_nmcm",
    #     )
    #     rim_nmcs = st.sidebar.number_input(
    #         "NMCS (NMC Supply)",
    #         value=0,
    #         step=1,
    #         key="rim_nmcs",
    #     )
    #     rim_nmcb = st.sidebar.number_input(
    #         "NMCB (Both)",
    #         value=0,
    #         step=1,
    #         key="rim_nmcb",
    #     )
    #     rim_nmc_fly = st.sidebar.number_input(
    #         "NMC Flyable tails",
    #         value=0,
    #         step=1,
    #         key="rim_nmc_fly",
    #         help="Flyable but still coded NMC (headroom for scheduling, not EP)."
    #     )
    #     rim_depot = st.sidebar.number_input(
    #         "Depot / Programmed (PDM, etc.)",
    #         value=0,
    #         step=1,
    #         key="rim_depot",
    #     )
    #     rim_upnr = st.sidebar.number_input(
    #         "UPNR",
    #         value=0,
    #         step=1,
    #         key="rim_upnr",
    #     )

    #     # Store richer structure for future visualizations / RIM math
    #     st.session_state["rim_structure_weekly"] = {
    #         "TAI": int(TAI),
    #         "PAI": int(PAI),
    #         "BAI": int(BAI),
    #         "NMCM": int(rim_nmcm),
    #         "NMCS": int(rim_nmcs),
    #         "NMCB": int(rim_nmcb),
    #         "NMC_flyable": int(rim_nmc_fly),
    #         "Depot": int(rim_depot),
    #         "UPNR": int(rim_upnr),
    #         "Overhead_input": int(Overhead),
    #     }

    # --- Sidebar Inputs: Reliability / Attrition ---
    # ASD = st.sidebar.number_input(
    #     "Avg Sortie Duration (hrs)",
    #     value=2.5,
    #     step=0.1,
    #     key="weekly_asd",
    # )
    # Break_pct = st.sidebar.slider(
    #     "Break Rate (%)",
    #     0.0, 100.0, 16.6,
    #     key="weekly_break",
    # )
    # GA_pct = st.sidebar.slider(
    #     "Ground Abort Rate (%)",
    #     0.0, 100.0, 7.5,
    #     key="weekly_ga",
    # )
    # Fix8 = st.sidebar.slider(
    #     "Fix Rate 8 Hr (%)",
    #     0.0, 100.0, 20.0,
    #     key="weekly_fix8",
    # )
    # Fix12 = st.sidebar.slider(
    #     "Fix Rate 12 Hr (%)",
    #     0.0, 100.0, 30.0,
    #     key="weekly_fix12",
    # )
    # Fix24 = st.sidebar.slider(
    #     "Fix Rate 24 Hr (%)",
    #     0.0, 100.0, 40.0,
    #     key="weekly_fix24",
    # )
    # WxAttr = st.sidebar.slider(
    #     "Weather Attrition (%)",
    #     0.0, 100.0, 0.0,
    #     key="weekly_wx",
    # )
    # SortieAttr = st.sidebar.slider(
    #     "Sortie Attrition (%)",
    #     0.0, 100.0,
    #     20.0,
    #     help="Percent of scheduled sorties lost to non-weather causes (not GA)."
    # )

    # Trials = st.sidebar.slider(
    #     "MC Trials",
    #     100, 2000, 500,
    #     key="weekly_trials",
    # )
    # commit_thresh = st.sidebar.slider(
    #     "Commitment Rate Threshold (%)",
    #     min_value=0.0,
    #     max_value=100.0,
    #     value=65.0,
    #     key="weekly_commit_thresh",
    #     help="Warn when daily commit rate exceeds this % of available aircraft"
    # )

    # --- Turn Pattern Input ---
    num_weeks = st.number_input(
        "Number of Weeks",
        min_value=1, max_value=12, value=1, step=1,
        key="weekly_num_weeks",
    )
    all_weeks_patterns = []
    weekend_flags = []
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    for w in range(num_weeks):
        st.subheader(f"Week {w+1}")
        cols = st.columns(len(days))
        pattern = []
        for i, d in enumerate(days):
            default = "8x6" if i < 5 else "0x0"
            pattern.append(
                cols[i].text_input(
                    f"W{w+1}_{d}",
                    value=default,
                    key=f"weekly_turn_w{w+1}_{d}",
                )
            )
        all_weeks_patterns.append(pattern)
        weekend_flags.append(
            st.checkbox(
                f"Weekend duty W{w+1}",
                key=f"weekly_weekend_{w}"
            )
        )

    # --- Run button: store full multi-week trials in session state ---
    run_clicked = st.button("Run Weekly Simulation")

    if run_clicked:
        params = {
            "TAI": TAI,
            "Overhead": Overhead,
            "Break %": Break_pct,
            "GA %": GA_pct,
            "Fix-8 Hr": Fix8,
            "Fix-12 Hr": Fix12,
            "Fix-24 Hr": Fix24,
            "Wx Attr %": WxAttr,
            "Sortie Attr %": SortieAttr,
            "ASD (hrs)": ASD,
            "weeks": num_weeks,
            "turn_patterns": all_weeks_patterns,
            "weekend_duty": weekend_flags,
        }

        trials = run_weekly_simulation(params, Trials)

        # Store everything needed to re-slice without re-running
        st.session_state["weekly_trials_data"] = trials              # list of trials, each = list of weeks
        st.session_state["weekly_num_weeks_data"] = num_weeks
        st.session_state["weekly_patterns_data"] = all_weeks_patterns
        st.session_state["weekly_weekend_flags_data"] = weekend_flags

        # Default view to Week 1 on a fresh run
        st.session_state["weekly_selected_week"] = 1

    # --- Week selector: use stored num_weeks if available ---
    effective_weeks = int(st.session_state.get("weekly_num_weeks_data", num_weeks))
    selected_week = st.selectbox(
        "View Results for Week",
        list(range(1, effective_weeks + 1)),
        key="weekly_selected_week",
    )

    # --- Display Results if Available ---
    if "weekly_trials_data" in st.session_state:
        trials_data = st.session_state["weekly_trials_data"]

        # Guard: if something weird got stored, bail cleanly
        if isinstance(trials_data, int):
            st.error("Weekly trials data is malformed. Please rerun the Weekly Simulation.")
            st.stop()

        sel_w = int(selected_week)  # 1-based index

        # Use patterns / weekend flags from the run, not current sidebar
        patterns_run = st.session_state.get("weekly_patterns_data", all_weeks_patterns)
        weekend_flags_run = st.session_state.get("weekly_weekend_flags_data", weekend_flags)

        # Safety: clamp selected week to available weeks
        max_weeks_from_data = len(trials_data[0]) if trials_data and isinstance(trials_data[0], list) else effective_weeks
        sel_w = max(1, min(sel_w, max_weeks_from_data))

        # Slice out the selected week from all trials
        week_results = [trial[sel_w - 1] for trial in trials_data]
        df = pd.DataFrame(week_results)

        days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

        # ── Compute daily scheduled sorties from *stored* patterns ──────
        raw_patterns = patterns_run[sel_w - 1]  # e.g. ["10x8","10x8","8x6","8x6","8x6","0x0","0x0"]
        daily_sched_matrix = np.array([
            [sum(map(int, re.findall(r"\d+", pat))) for pat in raw_patterns]
            for _ in week_results
        ])
        avg_sched = daily_sched_matrix.mean(axis=0)

        # ── Compute avg flown per day by subtracting losses ─────────────
        daily_losses = [r["daily_losses"] for r in week_results]
        avg_flown = np.mean([
            [
                sched
                - loss["weather"]
                - loss["ground_aborts"]
                - loss["sortie_attr"]
                - loss["mnd"]
                for sched, loss in zip(sched_row, loss_row)
            ]
            for sched_row, loss_row in zip(daily_sched_matrix, daily_losses)
        ], axis=0)

        # ── Availability (start / end) ───────────────────────────────────
        avg_start = np.stack(df["daily_available_start"].values).mean(axis=0)
        avg_end   = np.stack(df["daily_available_end"].values).mean(axis=0)

        # ── Parse turn patterns into first/second-go lists ───────────────
        go_lists = [
            list(map(int, re.findall(r"(\d+)", pat)))
            for pat in raw_patterns
        ]
        first_go_vals  = [gl[0] if len(gl) >= 1 else 0 for gl in go_lists]
        second_go_vals = [gl[1] if len(gl) >= 2 else 0 for gl in go_lists]

        # ── Enhanced Over-commitment Alerts ─────────────────────────────
        with st.expander("⚠️ Alerts & Warnings (click to collapse)", expanded=True):
            for idx, day in enumerate(days):
                first_go = int(first_go_vals[idx]) if idx < len(first_go_vals) else 0
                avail = avg_start[idx]
                commit_pct = (first_go / avail * 100) if avail > 0 else 0
                if first_go > 0 and commit_pct > commit_thresh:
                    st.warning(
                        f"High commitment on {day}: "
                        f"{first_go} sorties ÷ {avail:.1f} AC → {commit_pct:.0f}% "
                        f"(threshold={commit_thresh:.0f}%)"
                    )

        # ── RISK METRICS & SUGGESTIONS (Revised) ────────────────────────
        mnd_events   = [r.get("mnd", 0) for r in week_results]
        abort_events = [r.get("ground_aborts", 0) for r in week_results]

        avg_mnd    = float(np.mean(mnd_events)) if mnd_events else 0.0
        avg_aborts = float(np.mean(abort_events)) if abort_events else 0.0

        # Probability of at least one event in a week
        p_any_mnd   = 100.0 * sum(1 for x in mnd_events   if x > 0) / len(mnd_events)   if mnd_events   else 0.0
        p_any_abort = 100.0 * sum(1 for x in abort_events if x > 0) / len(abort_events) if abort_events else 0.0

        risk_msg = (
            f"Avg MNDs/week: {avg_mnd:.2f} | Avg Aborts/week: {avg_aborts:.2f}  \n"
            f"P(any MND): {p_any_mnd:.1f}% | P(any abort): {p_any_abort:.1f}%"
        )

        if p_any_mnd > 20 or avg_mnd >= 1.0:
            st.error(risk_msg + " → HIGH MND risk.")
        elif p_any_mnd > 5 or avg_mnd > 0.25:
            st.warning(risk_msg + " → Moderate MND risk.")
        else:
            st.success(risk_msg + " → Low MND risk 👍")

        if (p_any_mnd > 5 or avg_mnd > 0.25) and not weekend_flags_run[sel_w - 1]:
            st.info("💡 Suggestion: enable Weekend Duty for repairs.")
        if any(s > 0.9 * a for s, a in zip(avg_sched, avg_end)):
            st.info("💡 Suggestion: reduce daily pattern to build buffer.")
        if df["fix_completed"].mean() < df["breaks"].mean():
            st.info("💡 Suggestion: increase short-turn fix rates.")

        # ── Turn-Factor Calculation (fleet-level) ──────────────────────
        sum_first = sum(first_go_vals)
        sum_second = sum(second_go_vals)
        if sum_first > 0:
            turn_factor = (sum_first + sum_second) / sum_first
        else:
            turn_factor = float("nan")

        st.info(
            f"🔄 Turn Factor (TF): "
            f"({sum_first} + {sum_second}) / {sum_first or 1} = "
            f"{turn_factor:.2f}  \n"
            f"→ Sum of First-Go={sum_first}, Second-Go={sum_second}"
        )

        # Share TF with the RIM / NS module
        try:
            st.session_state["weekly_turn_factor"] = float(turn_factor)
        except Exception:
            pass

        # # ============================
        # # ⭐ RIM / NS Crew Requirement (Experimental)
        # # ============================
        # if enable_rim_ns:
        #     from simulations.rim_requirements import compute_crew_aircraft_requirement

        #     st.subheader("RIM / North Star (Experimental) — Crew Requirement")

        #     st.caption(
        #         "This calculation is read-only and does not change the weekly sim. "
        #         "It shows how many aircraft are required from a crew perspective, "
        #         "given your current Turn Factor and a monthly crew requirement."
        #     )

        #     # Inputs specific to RIM requirement
        #     rim_col1, rim_col2, rim_col3 = st.columns(3)
        #     with rim_col1:
        #         crew_ratio = st.number_input(
        #             "Crew Ratio (crews per aircraft)",
        #             min_value=0.0,
        #             value=1.2,
        #             step=0.1,
        #             key="rim_crew_ratio",
        #             help="Example: 1.2 means 1.2 crews per PAI."
        #         )
        #     with rim_col2:
        #         spcm = st.number_input(
        #             "Sorties per Crew per Month (SPCM)",
        #             min_value=0.0,
        #             value=4.0,
        #             step=0.5,
        #             key="rim_spcm",
        #             help="How many sorties each crew must fly per month."
        #         )
        #     with rim_col3:
        #         om_days_rim = st.number_input(
        #             "O&M Days in Planning Month (for RIM)",
        #             min_value=1,
        #             value=20,
        #             step=1,
        #             key="rim_om_days",
        #             help="Use your monthly fly days here (not just this week)."
        #         )

        #     # Use your 'Sortie Attrition (%)' as the attrition input for now
        #     attrition_rate_rim = SortieAttr / 100.0

        #     # Crew-based requirement (RIM)
        #     rim_req = compute_crew_aircraft_requirement(
        #         pai=int(TAI),  # later we can refine PAI vs BAI/Overhead
        #         crew_ratio=float(crew_ratio),
        #         sorties_per_crew_month=float(spcm),
        #         om_days=int(om_days_rim),
        #         turn_factor=float(turn_factor),
        #         attrition_rate=float(attrition_rate_rim),
        #     )

        #     st.markdown(
        #         f"**Crew-based aircraft required:** **{rim_req['aircraft_required']}**  \n"
        #         f"- Net sorties per month (crews): {rim_req['net_sorties_month']:.1f}  \n"
        #         f"- Daily gross lines (after attrition): {rim_req['daily_gross_lines']:.1f}  \n"
        #         f"- Turn Factor used: {rim_req['turn_factor']:.2f} sorties/jet/day  \n"
        #         f"- Attrition used: {rim_req['attrition_rate']:.0%}"
        #     )

        #     # --- Compare crew-based requirement to simulated availability ---
        #     st.subheader("RIM vs Simulated Weekly Capacity")

        #     try:
        #         # avg_start / avg_end are already computed above for the selected week
        #         avg_start_cap = float(np.mean(avg_start))   # start-of-day capacity
        #         avg_end_cap   = float(np.mean(avg_end))     # end-of-day capacity

        #         rim_ac = rim_req["aircraft_required"]
        #         delta_start = avg_start_cap - rim_ac
        #         delta_end   = avg_end_cap - rim_ac

        #         st.markdown(
        #             f"""
        #             **Average Start-of-Day Capacity:** {avg_start_cap:.1f} aircraft  
        #             **Average End-of-Day Capacity:** {avg_end_cap:.1f} aircraft  
        #             **Crew-Based Requirement (RIM):** {rim_ac} aircraft  

        #             • Start-of-day margin: **{delta_start:+.1f}** aircraft  
        #             • End-of-day margin: **{delta_end:+.1f}** aircraft  
        #             """
        #         )

        #         # --- Store a simple summary for future dashboards/modules ---
        #         st.session_state["weekly_rim_summary"] = {
        #             "turn_factor": float(turn_factor),
        #             "crew_aircraft_required": int(rim_ac),
        #             "avg_start_capacity": avg_start_cap,
        #             "avg_end_capacity": avg_end_cap,
        #             "tai": int(TAI),
        #             "overhead": int(Overhead),
        #             "sortie_attrition": float(SortieAttr / 100.0),
        #         }

        #         # Simple status indicator
        #         if delta_start >= 0 and delta_end >= 0:
        #             st.success("RIM requirement is within simulated capacity for this pattern.")
        #         elif delta_start >= 0 and delta_end < 0:
        #             st.warning("You start the day within RIM, but end-of-day capacity drops below requirement.")
        #         else:
        #             st.error("Simulated capacity is below RIM requirement for most of the day.")

        #         # ===============================
        #         # ⭐ STEP 3: RIM 30-DAY PROJECTION
        #         # ===============================
        #         st.markdown("---")
        #         st.subheader("📈 30-Day RIM Projection (Experimental)")

        #         degrade_1 = st.number_input(
        #             "Known upcoming degraders in next 30 days",
        #             min_value=0, value=0, step=1,
        #             key="rim_deg_known",
        #             help="E.g., scheduled phase, HSC, ISO, long-term NMCM, etc."
        #         )

        #         degrade_2 = st.number_input(
        #             "Unscheduled/predictive degraders",
        #             min_value=0, value=0, step=1,
        #             key="rim_deg_unsched",
        #             help="E.g., expected breaks from trend, cann requirements, supply delays."
        #         )

        #         total_new_degraders = degrade_1 + degrade_2

        #         projected_ep = max(avg_start_cap - total_new_degraders, 0)
        #         projected_margin = projected_ep - rim_ac

        #         st.info(
        #             f"**Projected EP in 30 days:** {projected_ep:.1f} aircraft  \n"
        #             f"Degrader impact: –{total_new_degraders} aircraft  \n"
        #             f"Projected RIM Margin: **{projected_margin:+.1f} aircraft**"
        #         )

        #         if projected_margin >= 0:
        #             st.success("Projection: You remain inside RIM requirement.")
        #         else:
        #             st.error(
        #                 "Projection: You fall BELOW RIM requirement — risk of under-supporting the flying hour program."
        #             )

        #     except Exception as e:
        #         # If sim hasn't run yet or arrays aren't available
        #         st.warning(f"RIM comparison to availability not available: {e}")

        # # ─────────────────────────────────────────────────────────────
        # # RIM / North Star Projection (beta)
        # # Uses weekly patterns + break/fix settings to project EP_home & RIM
        # # ─────────────────────────────────────────────────────────────
        # if enable_rim_ns:
        #     with st.expander("📉 RIM / North Star Projection (beta)", expanded=False):
        #         st.caption(
        #             "Projects EP_home, NMC pool, and RIM over a short horizon using your "
        #             "weekly pattern, break rate, and an approximate fix rate."
        #         )

        #         # --- Structure inputs for RIM (PAI/BAI & non-possessed bins) ---
        #         c1, c2, c3 = st.columns(3)
        #         with c1:
        #             rim_pai = st.number_input(
        #                 "PAI (Primary Aircraft Inventory)",
        #                 min_value=0,
        #                 value=int(TAI),
        #                 step=1,
        #                 help="Core aircraft you intend to fly (RIM requirement basis)."
        #             )
        #         with c2:
        #             rim_bai = st.number_input(
        #                 "BAI (Backup Aircraft Inventory)",
        #                 min_value=0,
        #                 value=0,
        #                 step=1,
        #                 help="Backup aircraft that still count in TAI but are not primary flyers."
        #             )
        #         with c3:
        #             rim_horizon = st.number_input(
        #                 "RIM Horizon (days)",
        #                 min_value=3,
        #                 max_value=30,
        #                 value=7,
        #                 step=1,
        #                 help="How many days ahead to project readiness."
        #             )

        #         st.markdown("**Overhead / Degraders breakout (optional, defaults from Overhead):**")
        #         c4, c5, c6, c7 = st.columns(4)
        #         with c4:
        #             rim_nmcm = st.number_input(
        #                 "NMCM",
        #                 min_value=0,
        #                 value=int(Overhead),
        #                 step=1,
        #                 help="Maintenance NMC aircraft."
        #             )
        #         with c5:
        #             rim_nmcs = st.number_input("NMCS", min_value=0, value=0, step=1)
        #         with c6:
        #             rim_nmcb = st.number_input("NMCB", min_value=0, value=0, step=1)
        #         with c7:
        #             rim_nmc_fly = st.number_input(
        #                 "NMC Flyable",
        #                 min_value=0,
        #                 value=0,
        #                 step=1,
        #                 help="Flyable but NMC (adds headroom, not EP)."
        #             )

        #         st.markdown("**Depot / UPNR / Deployed / Alert / Spares:**")
        #         c8, c9, c10, c11, c12 = st.columns(5)
        #         with c8:
        #             rim_depot = st.number_input("Depot", min_value=0, value=0, step=1)
        #         with c9:
        #             rim_upnr = st.number_input("UPNR", min_value=0, value=0, step=1)
        #         with c10:
        #             rim_deployed = st.number_input(
        #                 "Deployed",
        #                 min_value=0,
        #                 value=0,
        #                 step=1,
        #                 help="Aircraft away from home station."
        #             )
        #         with c11:
        #             op_alert = st.number_input(
        #                 "Alert Requirement",
        #                 min_value=0,
        #                 value=0,
        #                 step=1,
        #                 help="Alert tails that must be preserved first."
        #             )
        #         with c12:
        #             rim_spares_bin = st.number_input(
        #                 "Reserved Spares (bin)",
        #                 min_value=0,
        #                 value=0,
        #                 step=1,
        #                 help="Spares treated as committed in RIM."
        #             )

        #         # --- Build daily training AM & sorties from your patterns ---
        #         # First-go (AM) tails already parsed: first_go_vals (length 7)
                
        #         # --- Recompute first-go values for RIM block ---
        #         rim_patterns = all_weeks_patterns[sel_w-1]  # weekly pattern strings (Mon–Sun)
        #         rim_go_lists = [list(map(int, re.findall(r"(\d+)", pat))) for pat in rim_patterns]

        #         first_go_vals = [gl[0] if len(gl) >= 1 else 0 for gl in rim_go_lists]
        #         second_go_vals = [gl[1] if len(gl) >= 2 else 0 for gl in rim_go_lists]

        #         # Daily first-go (AM) training lines
        #         daily_training_am = [int(v) for v in first_go_vals]

        #         # Use avg_flown as the daily sortie demand seen in the sim
        #         daily_sorties = [int(round(v)) for v in avg_flown]

        #         # --- Translate your fix rates into an approximate daily fix fraction ---
        #         # Simple approximation: average of 8/12/24-hr fix rates as a daily fraction
        #         fix_rate_day = min(
        #             1.0,
        #             max(0.0, (Fix8 + Fix12 + Fix24) / 300.0)  # e.g. (25+35+56)/300 ≈ 0.39
        #         )

        #         rim_inputs = RIMProjectionInputs(
        #             tai=int(TAI),
        #             pai=int(rim_pai),
        #             bai=int(rim_bai),
        #             depot=int(rim_depot),
        #             upnr=int(rim_upnr),
        #             other_non_possessed=0,
        #             nmcm=int(rim_nmcm),
        #             nmcs=int(rim_nmcs),
        #             nmcb=int(rim_nmcb),
        #             nmc_flyable=int(rim_nmc_fly),
        #             deployed=int(rim_deployed),
        #             alert=int(op_alert),
        #             spares_bin=int(rim_spares_bin),
        #             horizon_days=int(rim_horizon),
        #             daily_training_am=daily_training_am,
        #             daily_sorties=daily_sorties,
        #             break_rate_per_sortie=float(Break_pct) / 100.0,
        #             fix_rate_per_day=fix_rate_day,
        #             commit_cap=commit_thresh / 100.0,
        #         )

        #         try:
        #             rim_results = project_rim_as_dicts(rim_inputs)
        #             rim_df = pd.DataFrame(rim_results)

        #             st.markdown("**RIM Projection Table (per day)**")
        #             st.dataframe(
        #                 rim_df[["day", "ep_home", "training_am", "alert", "spares_bin",
        #                         "rim", "nmc_total", "breaks", "fixes"]],
        #                 use_container_width=True
        #             )

        #             c_r1, c_r2 = st.columns(2)

        #             # RIM vs Day
        #             with c_r1:
        #                 fig_rim = go.Figure()
        #                 fig_rim.add_trace(go.Scatter(
        #                     x=rim_df["day"],
        #                     y=rim_df["rim"],
        #                     mode="lines+markers",
        #                     name="RIM"
        #                 ))
        #                 fig_rim.update_layout(
        #                     title="RIM vs Day",
        #                     xaxis_title="Day",
        #                     yaxis_title="RIM (EP_home − Requirement)"
        #                 )
        #                 st.plotly_chart(fig_rim, use_container_width=True)

        #             # EP_home vs Requirement & NMC
        #             with c_r2:
        #                 req_series = (
        #                     rim_df["training_am"] + rim_df["alert"] + rim_df["spares_bin"]
        #                 )
        #                 fig_ep = go.Figure()
        #                 fig_ep.add_trace(go.Scatter(
        #                     x=rim_df["day"],
        #                     y=rim_df["ep_home"],
        #                     mode="lines+markers",
        #                     name="EP_home"
        #                 ))
        #                 fig_ep.add_trace(go.Scatter(
        #                     x=rim_df["day"],
        #                     y=req_series,
        #                     mode="lines+markers",
        #                     name="Requirement (Alert+AM+Spares)"
        #                 ))
        #                 fig_ep.add_trace(go.Bar(
        #                     x=rim_df["day"],
        #                     y=rim_df["nmc_total"],
        #                     name="NMC pool",
        #                     opacity=0.4
        #                 ))
        #                 fig_ep.update_layout(
        #                     title="EP_home, Requirement & NMC vs Day",
        #                     xaxis_title="Day",
        #                     yaxis_title="Tails"
        #                 )
        #                 st.plotly_chart(fig_ep, use_container_width=True)

        #             st.caption(
        #                 "RIM < 0 indicates a shortfall: not enough EP_home remaining after Alert, "
        #                 "Training AM, and Reserved Spares. Adjust degraders, fix rates, or requirements "
        #                 "to explore solutions."
        #             )
        #         except Exception as e:
        #             st.error(f"RIM projection failed: {e}")

        # --- Display Results & Charts ---
        st.subheader(f"📊 Weekly Results — Week {sel_w}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # 1) Sorties Flown Distribution
        with col1:
            flown = df["flown"]
            fig1 = go.Figure()
            fig1.add_trace(go.Histogram(x=flown, nbinsx=15, name="Flown"))
            fig1.update_layout(
                title="Sorties Flown Distribution",
                xaxis_title="Sorties Flown",
                yaxis_title="Frequency"
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        # 2) Scheduled vs Avg Flown
        with col2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=days, y=avg_sched, mode="lines+markers", name="Scheduled"))
            fig2.add_trace(go.Scatter(x=days, y=avg_flown, mode="lines+markers", name="Avg Flown"))
            fig2.update_layout(
                title="Scheduled vs Avg Flown",
                xaxis_title="Day",
                yaxis_title="Sorties"
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # 3) Daily Aircraft Availability
        with col3:
            # Build availability band between start and end of day
            fig3 = go.Figure()
        
            # Upper bound (End‑of‑day availability)
            fig3.add_trace(go.Scatter(
                x=days,
                y=avg_end,
                mode="lines",
                name="End‑of‑Day Avail",
                line=dict(width=2)
            ))
        
            # Lower bound (Start‑of‑day availability)
            fig3.add_trace(go.Scatter(
                x=days,
                y=avg_start,
                mode="lines",
                name="Start‑of‑Day Avail",
                line=dict(width=2)
            ))
        
            # Fill area between start and end
            fig3.add_trace(go.Scatter(
                x=days + days[::-1],
                y=list(avg_end) + list(avg_start[::-1]),
                fill='toself',
                fillcolor='rgba(0,100,200,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False
            ))
        
            fig3.update_layout(
                title="Avg Daily Availability (Start → End)",
                xaxis_title="Day of Week",
                yaxis_title="Aircraft Available",
                legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98)
            )
            st.plotly_chart(fig3, use_container_width=True)

        # 4) Breaks vs Fix Completions
        with col4:
            breaks_mean = df["breaks"].mean()
            fixes_mean  = df["fix_completed"].mean()
            breaks_std  = df["breaks"].std()
            fixes_std   = df["fix_completed"].std()
        
            fig4 = go.Figure()
            fig4.add_trace(go.Bar(
                x=["Breaks","Fixes"],
                y=[breaks_mean, fixes_mean],
                error_y=dict(type="data", array=[breaks_std, fixes_std]),
                marker_color=["salmon","skyblue"]
            ))
            fig4.update_layout(
                title="Avg Breaks vs Fix Completions",
                yaxis_title="Count"
            )
            st.plotly_chart(fig4, use_container_width=True)

        # --- Compute Weekly Hours Flown ---
        df["hours_flown"] = df["flown"] * ASD

        # --- Final Summary Table ---
        st.dataframe(df.describe().transpose())

# ---------- PERSONNEL SIMULATION ----------
elif sim_choice == "Personnel Simulation":
    import calendar, pandas as pd, numpy as np
    from collections import Counter
    st.header("👥 Personnel Capacity Simulation")

    # — Sidebar Basic Params —
    TAI    = st.sidebar.number_input("TAI (# Aircraft)", value=28, step=1)
    shifts = st.sidebar.number_input("Shifts", min_value=1, max_value=10, value=1, step=1)
    lps    = st.sidebar.number_input("Labor Hrs/Sortie", value=10.0)
    TrialsP= st.sidebar.slider("Monte Carlo Trials", 1, 5000, 500)

    # — Timeframe + Goals —
    mode = st.radio("Timeframe", ["Full FY", "Single Month"])
    
    if mode == "Full FY":
        months = list(range(1, 13))
        month_names = [calendar.month_abbr[m] for m in months]
        default_om = [20] * 12
        default_goals = [0] * 12
    
        planning_df = pd.DataFrame({
            "Month": month_names,
            "O&M Days": default_om,
            "Goal Sorties": default_goals
        })
        with st.expander("🗓️ O&M Days & Goals Planning Matrix", expanded=True):
            st.caption("Edit O&M Days or monthly sortie goals as needed. One row per month.")
            edited_plan = st.data_editor(
                planning_df,
                use_container_width=True,
                hide_index=True,
                num_rows="fixed",
                key="personnel_om_matrix"
            )
        om_days = {i+1: int(edited_plan.loc[i, "O&M Days"]) for i in range(12)}
        month_goals = {i+1: int(edited_plan.loc[i, "Goal Sorties"]) for i in range(12)}
        months = list(range(1, 13))
    
    else:
        st.subheader("O&M Days & Goal — Single Month")
        month = st.selectbox("Month", list(range(1,13)))
        om_days     = {month: st.number_input("O&M days", value=20, min_value=0)}
        month_goals = {month: st.number_input("Sortie goal", value=0, min_value=0)}
        months = [month]

    # — Absence & UTE Rates —
    st.subheader("Absence & Skill Factors")
    c1,c2,c3 = st.columns(3)
    with c1:
        leave = st.number_input("Leave rate", value=0.10, min_value=0.0, max_value=1.0)
        tdy   = st.number_input("TDY rate",   value=0.05, min_value=0.0, max_value=1.0)
    with c2:
        deploy= st.number_input("Deploy rate", value=0.02, min_value=0.0, max_value=1.0)
    with c3:
        st.caption("👩‍💻 UTE (hrs/day) per skill level")
        ute3  = st.number_input("3-lvl UTE", value=0.25)
        ute5  = st.number_input("5-lvl UTE", value=0.50)
        ute7  = st.number_input("7-lvl UTE", value=1.00)

    # — Workcenters Manual Editor —
    st.subheader("Workcenters (Authorized vs Assigned)")
    n_wcs = st.number_input("How many workcenters?", min_value=1, max_value=20, value=1, step=1)
    wc_list = []
    for i in range(n_wcs):
        st.markdown(f"**Workcenter #{i+1}**")
        cols = st.columns([2,1,1,1,1,1,1])
        shop  = cols[0].text_input(f"Shop name #{i+1}", key=f"wc_shop_{i}")
        a3    = cols[1].number_input(f"Auth 3-lvl", key=f"wc_auth3_{i}", min_value=0.0, value=0.0)
        a5    = cols[2].number_input(f"Auth 5-lvl", key=f"wc_auth5_{i}", min_value=0.0, value=0.0)
        a7    = cols[3].number_input(f"Auth 7-lvl", key=f"wc_auth7_{i}", min_value=0.0, value=0.0)
        s3    = cols[4].number_input(f"Asgn 3-lvl", key=f"wc_asn3_{i}", min_value=0.0, value=0.0)
        s5    = cols[5].number_input(f"Asgn 5-lvl", key=f"wc_asn5_{i}", min_value=0.0, value=0.0)
        s7    = cols[6].number_input(f"Asgn 7-lvl", key=f"wc_asn7_{i}", min_value=0.0, value=0.0)
        wc_list.append({
            "shop": shop.strip(),
            "asn":  {"3": s3, "5": s5, "7": s7},
            "auth": {"3": a3, "5": a5, "7": a7},
        })

    # build the final dict for the backend
    workcenters = {
        wc["shop"]: wc["asn"]
        for wc in wc_list
        if wc["shop"] and sum(wc["asn"].values()) > 0
    }

    # — Run Simulation —
    if st.button("Run Personnel Simulation"):
        params = {
            "TAI":                    TAI,
            "months":                 months,
            "om_days":                om_days,
            "month_goals":            month_goals,
            "leave_rate":             leave,
            "tdy_rate":               tdy,
            "deploy_rate":            deploy,
            "ute_rates":              {"3": ute3, "5": ute5, "7": ute7},
            "labor_hours_per_sortie": lps,
            "workcenters":            workcenters,
        }
        per_results = run_personnel_simulation(params, trials=TrialsP)

        # 1) Main summary DataFrame (first trial)
        df_per = pd.DataFrame(per_results[0]).set_index("month")

        skill_levels = ["3", "5", "7"]
        for lvl in skill_levels:
            ratio_col = f"ppl_per_ac_{lvl}"
            df_per[ratio_col] = [
                sum(wc.get(lvl, 0) for wc in workcenters.values()) / TAI
                for _ in df_per.index
            ]
        show_cols = [
            "present_people", "available_hours", "sorties_supported",
            "ppl_per_ac", "shifts_supported", "shortfall",
            "limiting_shop", "limiting_shop_sorties",
            "ppl_per_ac_3", "ppl_per_ac_5", "ppl_per_ac_7"
        ]
        st.success("✅ Simulation complete!")
        st.write("### Monthly Summary (including bottlenecks & shortfalls)")
        st.dataframe(df_per[show_cols])

        # 2) Alert for each bottleneck/shortfall (per month)
        st.subheader("⚠️ Bottleneck Alerts")
        for m, row in df_per.iterrows():
            if row.get("shortfall", False):
                st.warning(
                    f"{calendar.month_abbr[m]} shortfall: "
                    f"Bottleneck = {row['limiting_shop'] or 'N/A'} "
                    f"(can only support {row['limiting_shop_sorties']:.1f} sorties)"
                )

        # 3) Frequency of bottleneck shop across all trials (first month as example)
        if TrialsP > 1:
            st.subheader("🔍 Bottleneck Frequency (Monte Carlo)")
            all_bottlenecks = Counter(
                trial[0].get("limiting_shop", "") for trial in per_results if trial[0].get("shortfall", False)
            )
            if all_bottlenecks:
                most_common = all_bottlenecks.most_common(1)[0]
                st.info(
                    f"Most common bottleneck for {calendar.month_abbr[months[0]]}: "
                    f"{most_common[0]} ({most_common[1]} out of {TrialsP} trials)"
                )

        # 4) Supported vs Goal plot
        months_list = df_per.index.tolist()
        supported = df_per["sorties_supported"].tolist()
        goals = [month_goals.get(m,0) for m in months_list]
        import plotly.graph_objects as go
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=months_list, y=supported, mode="lines+markers", name="Supported"))
        fig1.add_trace(go.Scatter(x=months_list, y=goals, mode="lines+markers", name="Goal"))
        # 5) Annotate shortfall months
        shortfall_months = [m for m, row in df_per.iterrows() if row["shortfall"]]
        fig1.add_trace(go.Scatter(
            x=shortfall_months,
            y=[df_per.loc[m, "sorties_supported"] for m in shortfall_months],
            mode="markers",
            marker=dict(color="red", size=10),
            name="Shortfall"
        ))
        fig1.update_layout(
            title="Monthly Sorties Supported vs Goal (Shortfalls in Red)",
            xaxis_title="Month",
            yaxis_title="Sorties",
            xaxis=dict(tickmode="array", tickvals=months_list)
        )
        st.plotly_chart(fig1, use_container_width=True)

        # 6) Histogram for MC >1
        if TrialsP > 1:
            all_months = list(zip(*[
                [trial[m]["sorties_supported"] for m in range(len(months_list))]
                for trial in per_results
            ]))
            st.subheader("📊 Monte Carlo Distribution (First Month)")
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(x=all_months[0], nbinsx=20, name=f"Month {months_list[0]}"))
            fig2.update_layout(
                title=f"Histogram: Sorties Supported (Month {months_list[0]})",
                xaxis_title="Sorties Supported",
                yaxis_title="Frequency"
            )
            st.plotly_chart(fig2, use_container_width=True)

        # 7) Heatmap: % Manned by Shop × Month
        shops  = list(workcenters.keys())
        matrix = []
        for shop in shops:
            row = []
            auth = sum(wc["auth"].get(lvl,0) for wc in wc_list if wc["shop"]==shop for lvl in ("3","5","7"))
            for idx,m in enumerate(months_list):
                pres = df_per.loc[m, "present_people"]
                row.append((pres/auth*100) if auth>0 else 0)
            matrix.append(row)
        fig3 = go.Figure(data=go.Heatmap(
            z=matrix,
            x=[calendar.month_abbr[m] for m in months_list],
            y=shops,
            colorscale="RdYlGn",
            zmin=0, zmax=100
        ))
        fig3.update_layout(title="% Manned by Shop × Month")
        st.plotly_chart(fig3, use_container_width=True)

        # 8) Downloadable Results Table (all columns)
        st.subheader("📥 Download Results")
        csv_buf = df_per[show_cols].reset_index().to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Personnel Sim Results CSV",
            csv_buf,
            file_name="personnel_sim_results.csv",
            mime="text/csv"
        )

# ---------- QUICK PROBABILITY ANALYSIS ----------
elif sim_choice == "Quick Probability Analysis":
    st.header("⚙️ Quick Probability Calculator")

    # — Inputs —
    n  = st.number_input("Planned sorties (n)", min_value=1, value=50)
    p  = st.number_input("Success rate p", min_value=0.0, max_value=1.0, value=0.80)
    x  = st.number_input("Target successes (x)", min_value=0, max_value=n, value=40)
    ci = st.number_input("Confidence level", min_value=0.0, max_value=1.0, value=0.95)
    ss = st.number_input("Sample size (for CI on p)", min_value=1, value=365)

    if st.button("Calculate"):
        # — Compute distribution & CI —
        from scipy import stats
        import numpy as np
        import plotly.graph_objects as go

        z  = stats.norm.ppf((1+ci)/2)
        me = z * np.sqrt((p*(1-p))/ss)
        p_lo, p_hi = max(0,p-me), min(1,p+me)
        xs  = np.arange(0, n+1)
        pmf = stats.binom.pmf(xs, n, p)

        # — Draw chart —
        fig = go.Figure()
        fig.add_trace(go.Bar(x=xs, y=pmf, name="P(X=k)"))
        fig.add_vline(x=x,     line_color="red",   line_dash="dash", name="Target (x)")
        fig.add_vline(x=n*p,   line_color="green", line_dash="dash", name="Mean (n·p)")
        fig.update_layout(
            title=f"Binomial(n={n}, p={p:.2f}) Distribution",
            xaxis_title="Number of Successes (k)",
            yaxis_title="Probability P(X=k)",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig, use_container_width=True)

        # — Numeric results —
        prob = 1 - stats.binom.cdf(x-1, n, p)
        st.write(f"• **P(X ≥ {x})** = {prob:.3f}")
        st.write(f"• **{ci*100:.0f}% CI for p**: [{p_lo:.2%}, {p_hi:.2%}]")

        # — Explanatory legend —
        st.markdown("---")
        st.markdown("#### 📖 Chart Explanation")
        st.markdown("""
        - **Bars**: Probability of observing exactly _k_ successes in _n_ trials (Binomial PMF).  
        - **Red dashed line**: Your **target** number of successes (_x_).  
        - **Green dashed line**: The **mean** (_n·p_) of the distribution.  
        - **P(X ≥ x)**: Shown above, is the probability of achieving at least _x_ successes.  
        - **CI on p**: Your confidence interval around the success rate _p_, shown numerically.  
        """)

# ─── ANNUAL HISTORICAL SIMULATION ───────────────────────────────────
elif sim_choice == "Annual Historical Simulation":
    import calendar
    import pandas as pd
    import numpy as np
    from simulations.utils import calculate_monthly_rates
    from simulations.historical_annual import HistoricalAnnualSimulation

    st.header("📅 Annual Historical Simulation")

    # --- 1) Upload historical data ---
    uploaded = st.file_uploader("Upload 4-yr roll-up (sheet 'Historical Data Input')", type=["xlsx","xls"])
    if not uploaded:
        st.info("Please upload your Excel file to proceed."); st.stop()
    df_hist = pd.read_excel(uploaded, sheet_name="Historical Data Input")
    df_hist.columns = (
        df_hist.columns.str.strip().str.lower().str.replace(" ", "_")
    )
    name_to_num = {name: idx for idx, name in enumerate(calendar.month_name) if name}
    df_hist["month_num"] = (
        df_hist["month"].astype(str).str.strip().str.title().map(name_to_num)
    )

# --- 2) Display & Edit Calculated Monthly Rates from Import ---
    rates_df = calculate_monthly_rates(df_hist)
    
    with st.expander("📊 Historical Monthly Rates (edit as needed)", expanded=True):
        st.caption("⬆️ You may edit the rates below for each month before running the simulation. These edited values will be used for all downstream calculations. The original imported/calculated rates are shown for reference below.")
        # Always-editable grid (Option 1)
        editable_rates = rates_df.copy()
        edited = st.data_editor(
            editable_rates,
            hide_index=True,
            use_container_width=True,
            num_rows="fixed",  # disables adding/removing months
            key="edit_grid_all_rates"
        )
        # Show a reminder/info
        st.info("You may change any monthly rate in the table above. These edits will override the imported historical values for the simulation.")
    
    with st.expander("See Original Calculated Rates", expanded=False):
        st.dataframe(rates_df, use_container_width=True)
    
    # Use 'edited' as rates_df for all downstream calculations
    rates_df = edited

    # --- 3) Editable Planning Matrix ---
    st.subheader("📝 FY Planning Inputs (Oct–Sep)")
    months = list(range(10, 13)) + list(range(1, 10))  # Oct–Sep
    month_names = [calendar.month_abbr[m] for m in months]
    default_matrix = pd.DataFrame({
        "Month": month_names,
        "O&M Days": [20]*12,
        "Degraders": [0]*12,
        "Turn Pattern": ["8x6"]*12,
        "Commit %": [65]*12,
        "ASD": [2.0]*12
    })

    with st.expander("📝 FY Planning Inputs (Oct–Sep)", expanded=True):
        plan_df = st.data_editor(
            default_matrix,
            num_rows="fixed",
            use_container_width=True,
            hide_index=True,
            key="annual_planning_grid"
        )

    # --- 4) Top-level planning goals ---
    with st.expander("📝 FY Planning Inputs (Oct–Sep)", expanded=True):
        TAI         = st.number_input("TAI (Aircraft Inventory)", value=12, step=1)
        FY_goal     = st.number_input("FY Flying-Hour Goal (hrs)", value=0.0, step=100.0)
        ASD         = st.number_input("Avg Sortie Duration (hrs)", value=2.0, step=0.1)
        MC_target   = st.slider("MC-Rate Target (%)", 0, 100, 80, help="Desired minimum MC-rate")
        mc_delta    = st.slider("Adjust MC Rate (%)", -10, 10, 0, step=1, help="Test impact of improved/degraded MC-rate")

        # --- Uncertainty slider for Monte Carlo noise ---
        uncertainty = st.slider(
            "Monthly uncertainty (±%)",
            min_value=0, max_value=20, value=5, step=1,
            help="Amount of random variation in MC and Execution rates per month"
        ) / 100.0

    # --- 5) Extract planning inputs as dicts keyed by month_num ---
    om_days = {m: int(plan_df.loc[i, "O&M Days"]) for i, m in enumerate(months)}
    degraders = {m: int(plan_df.loc[i, "Degraders"]) for i, m in enumerate(months)}
    turn_patterns = {m: plan_df.loc[i, "Turn Pattern"] for i, m in enumerate(months)}
    commit_rates = {m: float(plan_df.loc[i, "Commit %"])/100.0 for i, m in enumerate(months)}
    asd_dict = {m: float(plan_df.loc[i, "ASD"]) for i, m in enumerate(months)}  # Not used yet, for future

    # --- 6) Run Simulation Button ---
    if st.button("Run Annual Analysis"):
        # Apply MC-delta (if any) to rates
        rates_df_adj = rates_df.copy()
        if mc_delta != 0:
            rates_df_adj["mc_rate"] = (rates_df["mc_rate"] * (1 + mc_delta/100)).clip(upper=1.0)

        sim_params = {
            "rates_df": rates_df_adj,
            "TAI": TAI,
            "om_days": om_days,
            "planned_degraders": degraders,
            "turn_patterns": turn_patterns,
            "commit_rates": commit_rates,
            "uncertainty": uncertainty
        }
        sim = HistoricalAnnualSimulation()
        sim.params = sim_params

        # --- Run the Monte Carlo sim just once ---
        all_trials, summary = sim.simulate(trials=500)

        # Handle if summary is a list of lists
        if isinstance(summary, list) and len(summary) > 0 and isinstance(summary[0], list):
            main_results = summary[0]
        else:
            main_results = summary

        # Now main_results should be a list of dicts, safe for DataFrame
        results_df = pd.DataFrame(main_results)

        # PATCH: Ensure "month" column is present and correct
        months = list(range(10, 13)) + list(range(1, 10))
        if "month" not in results_df.columns:
            results_df["month"] = months[:len(results_df)]

        # Assign month_name for charts/tables
        results_df["month_name"] = [calendar.month_abbr[m] for m in results_df["month"]]

        # Optional: Print for debug
        # print("DEBUG: results_df columns:", results_df.columns.tolist())
        # print("DEBUG: results_df head:\n", results_df.head())

        # --- 7) Show Alerts for Over-commitment ---
        with st.expander("⚠️ Alerts & Warnings", expanded=False):
            for idx, m in enumerate(months):
                r = rates_df_adj[rates_df_adj["month_num"]==m].iloc[0]
                flyable_ac = max(0, math.floor(TAI - degraders[m]) * r["mc_rate"])
                first_go = int(turn_patterns[m].split("x")[0])
                if flyable_ac > 0:
                    commit_pct = first_go / flyable_ac * 100
                    if commit_pct > 80:
                        st.warning(f"{calendar.month_abbr[m]}: Over-committed! "
                                   f"{first_go} committed / {flyable_ac:.1f} flyable ({commit_pct:.1f}%)")
                    else:
                        st.info(f"{calendar.month_abbr[m]}: Commit rate {commit_pct:.1f}% (OK)")
                else:
                    st.error(f"{calendar.month_abbr[m]}: No flyable AC available!")

        # --- 7b) Probability of Meeting Flying Hour Goal ---
        st.subheader("🎯 Probability of Meeting Flying Hour Goal")
        # Calculate hours flown for each trial (as sum over months)
        all_hours_flown = []
        for trial in all_trials:
            trial_hours = 0.0
            for m_result in trial:
                # Use month-specific ASD
                asd = asd_dict[m_result["month"]]
                # Use mean/actual "flown" key as present in your trial dict
                flown_val = (
                    m_result["flown"] if "flown" in m_result else m_result.get("flown_mean", 0)
                )
                trial_hours += flown_val * asd
            all_hours_flown.append(trial_hours)
        n_success = sum(h >= FY_goal for h in all_hours_flown)
        prob_success = n_success / len(all_hours_flown) if len(all_hours_flown) else 0.0
        st.info(
            f"Estimated **probability of meeting your goal**: "
            f"**{prob_success:.1%}** ({n_success}/{len(all_hours_flown)} trials)"
        )

        import plotly.graph_objects as go
        fig_prob = go.Figure()
        fig_prob.add_trace(go.Histogram(x=all_hours_flown, nbinsx=30))
        fig_prob.add_vline(x=FY_goal, line_color="red", line_dash="dash", annotation_text="Goal")
        fig_prob.update_layout(
            xaxis_title="Simulated FY Hours Flown",
            yaxis_title="Frequency",
            title="Distribution of Simulated Total Flying Hours"
        )
        st.plotly_chart(fig_prob, use_container_width=True)
        st.write(
            f"**Mean simulated hours flown:** {np.mean(all_hours_flown):,.1f}  \n"
            f"**Std. deviation:** {np.std(all_hours_flown):,.1f}"
        )

        # --- 8) Visuals for Scheduled/Flown Sorties and Hours ---
        st.subheader("📈 Sorties Scheduled vs Flown")
        sched_col = "scheduled_mean" if "scheduled_mean" in results_df.columns else "scheduled"
        flown_col = "flown_mean" if "flown_mean" in results_df.columns else "flown"

        fig1 = go.Figure()
        fig1.add_trace(go.Bar(x=results_df["month_name"], y=results_df[sched_col], name="Scheduled"))
        fig1.add_trace(go.Bar(x=results_df["month_name"], y=results_df[flown_col], name="Flown"))
        fig1.update_layout(barmode="group", yaxis_title="Sorties")
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("📈 Hours Scheduled vs Flown")
        hours_sched = results_df[sched_col] * [asd_dict[m] for m in results_df["month"]]
        hours_flown = results_df[flown_col] * [asd_dict[m] for m in results_df["month"]]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=results_df["month_name"], y=hours_sched, name="Hours Scheduled"))
        fig2.add_trace(go.Bar(x=results_df["month_name"], y=hours_flown, name="Hours Flown"))
        fig2.update_layout(barmode="group", yaxis_title="Hours")
        st.plotly_chart(fig2, use_container_width=True)

        # Calculate hours_flown for each month
        results_df["hours_flown"] = results_df["flown_mean"] * results_df["asd_mean"]

        # --- 9) Downloadable Table ---
        st.subheader("📋 Detailed Results Table")
        
        # Specify your desired column order
        csv_columns = [
            "month", "month_name",
            "scheduled_mean",
            # "scheduled_ci_lo",
            # "scheduled_ci_hi",
            "flown_mean",
            "hours_flown",
            "asd_mean",
            "flown_ci_lo",
            "flown_ci_hi",
            "mc_rate_mean",
            "execution_rate_mean",
            "avg_flyable",
            "overcommit_risk",
            "break_rate_mean",
            "gab_rate_mean",
            "spared_gab_rate_mean"
        ]

        # Ensure the DataFrame has all these columns (add empty if needed for robustness)
        for col in csv_columns:
            if col not in results_df.columns:
                results_df[col] = ""

        # Reorder columns
        results_df_totals = results_df[csv_columns].copy()

        # Build the totals row
        total_row = {
            "month": np.nan,
            "month_name": "Total",
            "scheduled_mean": results_df["scheduled_mean"].sum(),
            # "scheduled_ci_lo": "",
            # "scheduled_ci_hi": "",
            "flown_mean": results_df["flown_mean"].sum(),
            "hours_flown": results_df["hours_flown"].sum(),
            "asd_mean": np.nan,
            "flown_ci_lo": np.nan,
            "flown_ci_hi": np.nan,
            "mc_rate_mean": np.nan,
            "execution_rate_mean": np.nan,
            "avg_flyable": np.nan,
            "overcommit_risk": np.nan,
            "break_rate_mean": np.nan,
            "gab_rate_mean": np.nan,
            "spared_gab_rate_mean": np.nan
        }

        # Append the total row
        results_df_totals = pd.concat([
            results_df_totals,
            pd.DataFrame([total_row])
        ], ignore_index=True)

        # Show table in app
        st.dataframe(results_df_totals, use_container_width=True)

        # Download as CSV
        csv_buf = results_df_totals.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Detailed Results CSV",
            csv_buf,
            file_name="annual_sim_results.csv",
            mime="text/csv"
        )

# ─── RIM / NORTH STAR ANALYSIS (BETA) ────────────────────────────────
elif sim_choice == "RIM / North Star Analysis (beta)":
    from simulations.rim_requirements import compute_crew_aircraft_requirement

    st.header("⭐ RIM / North Star Analysis (beta)")

    st.markdown(
        "This mode looks at your fleet structure, degraders, and crew demand to "
        "estimate the aircraft required to meet your crew flying requirements."
    )

    # ==========================
    # 1) Inventory & Structure
    # ==========================
    st.subheader("Fleet Structure")

    inv_col1, inv_col2 = st.columns(2)
    with inv_col1:
        rim_tai = st.number_input(
            "TAI (Total Aircraft Inventory)",
            value=18,
            step=1,
            key="rim_tai",
        )
        rim_pai = st.number_input(
            "PAI (Primary Aircraft Inventory)",
            value=16,
            step=1,
            key="rim_pai",
            help="Primary aircraft resourced for crews and flying hours."
        )
    with inv_col2:
        rim_depot = st.number_input(
            "Depot / Programmed (PDM, mods, etc.)",
            value=1,
            step=1,
            key="rim_depot",
        )
        rim_upnr = st.number_input(
            "UPNR (Unprogrammed / Reserve)",
            value=1,
            step=1,
            key="rim_upnr",
            help="Aircraft held in non-flying reserve / awaiting disposition."
        )

    # Derived structure
    rim_pai = min(int(rim_pai), int(rim_tai))
    rim_bai = max(int(rim_tai) - int(rim_pai), 0)
    nonposs = max(int(rim_depot) + int(rim_upnr), 0)
    possessed = max(int(rim_tai) - nonposs, 0)

    st.markdown(
        f"""
        **Structure snapshot**  
        • TAI: **{rim_tai}**  
        • PAI: **{rim_pai}**  
        • BAI: **{rim_bai}** (TAI − PAI)  
        • Depot + UPNR (non-possessed): **{nonposs}**  
        • Possessed: **{possessed}** (TAI − Depot − UPNR)
        """
    )

    # ==========================
    # 2a) Operational Requirements (non-crew)
    # ==========================
    st.subheader("Operational Requirements (non-crew)")

    # Initialize to zero so they're always defined even if something above changes
    op_spares = 0
    op_trainers = 0
    op_fleet = 0
    op_contg = 0
    op_alert = 0

    or_c1, or_c2, or_c3 = st.columns(3)

    with or_c1:
        op_spares = st.number_input(
            "Spares (tails held aside)",
            min_value=0,
            value=0,
            step=1,
            key="rim_op_spares",
            help="Pure spares that normally don't generate planned line sorties."
        )
        op_trainers = st.number_input(
            "Ground Trainers / static / SIM tails",
            min_value=0,
            value=0,
            step=1,
            key="rim_op_trainers",
        )

    with or_c2:
        op_fleet = st.number_input(
            "Fleet Management / Test / FCF tails",
            min_value=0,
            value=0,
            step=1,
            key="rim_op_fleet",
        )
        op_contg = st.number_input(
            "Contingency / Exercise package",
            min_value=0,
            value=0,
            step=1,
            key="rim_op_contg",
        )

    with or_c3:
        op_alert = st.number_input(
            "Alert Requirement (tails/day)",
            min_value=0,
            value=0,
            step=1,
            key="rim_op_alert",
            help="Alert / DCA / Homeland Defence tails held on status."
        )
        
        # Total non-crew operational tails (these are reserved before FHP)
    op_noncrew_total = int(op_spares + op_trainers + op_fleet + op_contg + op_alert)

    st.markdown(
        f"""
        **Non-crew operational requirement snapshot**  
        • Spares: **{op_spares}**  
        • Ground trainers / static: **{op_trainers}**  
        • Fleet mgmt / test / FCF: **{op_fleet}**  
        • Contingency / exercise: **{op_contg}**  
        • Alert requirement: **{op_alert}**  
        • **Total non-crew requirement:** **{op_noncrew_total}** tails/day
        """
    )

    # ---- Compute non-crew operational requirement total ----
    op_noncrew_total = (
        int(op_spares or 0)
        + int(op_trainers or 0)
        + int(op_fleet or 0)
        + int(op_contg or 0)
        + int(op_alert or 0)
    )

    # ==========================
    # 2) Degraders & NMC Bins
    # ==========================
    st.subheader("Degraders & NMC Bins")

    degr_c1, degr_c2, degr_c3 = st.columns(3)
    with degr_c1:
        rim_nmcm = st.number_input(
            "NMCM (NMC Maintenance)",
            value=0,
            step=1,
            key="rim_nmcm_beta",
        )
        rim_nmcs = st.number_input(
            "NMCS (NMC Supply)",
            value=0,
            step=1,
            key="rim_nmcs_beta",
        )
    with degr_c2:
        rim_nmcb = st.number_input(
            "NMCB (Both)",
            value=0,
            step=1,
            key="rim_nmcb_beta",
        )
        rim_nmc_fly = st.number_input(
            "NMC Flyable tails",
            value=0,
            step=1,
            key="rim_nmc_fly_beta",
            help="Flyable but coded NMC; they add scheduling headroom but do NOT increase EP."
        )
    with degr_c3:
        rim_deployed = st.number_input(
            "Deployed (EP away from home)",
            value=0,
            step=1,
            key="rim_deployed_beta",
        )
        # 🔁 IMPORTANT: no Alert here anymore – Alert is now in the
        # "Operational Requirements (non-crew)" section as `op_alert`.

    # --- NMC / EP math ---
    nmc_total = max(int(rim_nmcm) + int(rim_nmcs) + int(rim_nmcb), 0)
    ep = max(possessed - nmc_total, 0)              # effective possessed (not NMC)
    ep_home = max(ep - int(rim_deployed), 0)        # EP at home station

    # --- Reserve non-crew operational tails from EP_home ---
    # These are spares, trainers, fleet mgmt, contingency, and alert tails.
    # They are still possessed, but not available to cover crew-driven FHP.
    op_reserved = min(ep_home, op_noncrew_total)

    avail_legacy = max(ep_home - op_reserved, 0)    # legacy FHP tails after all non-crew requirements
    avail_eff = avail_legacy + max(int(rim_nmc_fly), 0)  # add NMC-flyable headroom

    st.markdown(
        f"""
        **Availability math (home station)**  
        • NMC total (bins): **{nmc_total}**  
        • EP: **{ep}**  
        • EP_home (−Deployed): **{ep_home}**  

        **Non-crew reservations (from EP_home)**  
        • Total non-crew requirement: **{op_noncrew_total}** tails/day  
        • Non-crew tails actually reserved (capped by EP_home): **{op_reserved}**

        **FHP availability**  
        • Avail for FHP (legacy, after non-crew): **{avail_legacy}**  
        • Avail FHP (+NMC Flyable headroom): **{avail_eff}**
        """
    )


    # ==========================
    # 3) Crew-Based RIM Requirement
    # ==========================
    st.subheader("Crew-Based Requirement (RIM)")

    # --- Base crew ratio input ---
    cfg_c1, cfg_c2, cfg_c3 = st.columns(3)
    with cfg_c1:
        crew_ratio = st.number_input(
            "Crew Ratio (crews per PAI)",
            min_value=0.0,
            value=1.2,
            step=0.1,
            key="rim_crew_ratio_beta",
            help="Example: 1.2 means 1.2 crews per PAI aircraft."
        )
    with cfg_c2:
        spcm = st.number_input(
            "Sorties per Crew per Month (SPCM)",
            min_value=0.0,
            value=4.0,
            step=0.5,
            key="rim_spcm_beta",
            help="How many sorties each base crew must fly per month."
        )
    with cfg_c3:
        om_days = st.number_input(
            "O&M Days in Month",
            min_value=1,
            value=20,
            step=1,
            key="rim_om_days_beta",
            help="Use your monthly fly days (e.g., Mon–Fri x 4 weeks)."
        )

    # --- Crew demand mode: ratio+overhead vs overhead-only ---
    demand_mode = st.radio(
        "Crew demand mode",
        ["Crew Ratio + Crew Overhead", "Crew Overhead Only"],
        key="rim_demand_mode_beta",
        help=(
            "• Crew Ratio + Crew Overhead: use PAI-based crew ratio and add overhead pools.\n"
            "• Crew Overhead Only: ignore crew ratio and use only the CMR/BMC crew inputs below."
        ),
    )

    # --- Turn Factor selection ---
    st.markdown("**Turn Factor (TF) — sorties per jet per fly day**")
    tf_source = st.radio(
        "Turn Factor source",
        ["Use from Weekly Simulation (if available)", "Enter manually"],
        key="rim_tf_source_beta",
    )

    tf_default = st.session_state.get("weekly_turn_factor", float("nan"))
    if tf_source == "Use from Weekly Simulation (if available)" and not math.isnan(tf_default):
        turn_factor = float(tf_default)
        st.info(f"Using TF from Weekly Simulation: **{turn_factor:.2f}** sorties/jet/day.")
    else:
        base_tf = 1.0 if math.isnan(tf_default) else round(float(tf_default), 2)
        turn_factor = st.number_input(
            "Turn Factor (TF)",
            min_value=0.1,
            value=base_tf,
            step=0.05,
            key="rim_tf_manual_beta",
        )

    attr_rate = st.slider(
        "Sortie attrition (for RIM calc) %",
        min_value=0.0,
        max_value=50.0,
        value=20.0,
        step=1.0,
        key="rim_attr_rate_beta",
    ) / 100.0

    # --- Crew Overhead (CMR/BMC mix) ---
    extra_crews = []

    with st.expander("Crew Overhead (CMR/BMC mix)", expanded=False):
        st.caption(
            "Use this to add specific crew pools with their own sortie requirements, "
            "e.g. CMR/BMC Wingmen and Flight Leads."
        )

        group_defs = [
            ("CMR Wingmen", "cmr_wg"),
            ("CMR Flight Leads", "cmr_fl"),
            ("BMC Wingmen", "bmc_wg"),
            ("BMC Flight Leads", "bmc_fl"),
        ]

        for label, key_suffix in group_defs:
            c1, c2 = st.columns(2)
            with c1:
                count = st.number_input(
                    f"{label} – # crews",
                    min_value=0,
                    value=0,
                    step=1,
                    key=f"rim_{key_suffix}_count",
                )
            with c2:
                spcm_extra = st.number_input(
                    f"{label} – sorties/crew/month",
                    min_value=0.0,
                    value=0.0,
                    step=0.5,
                    key=f"rim_{key_suffix}_spcm",
                )

            if count > 0 and spcm_extra > 0:
                extra_crews.append(
                    {
                        "name": label,
                        "count": float(int(count)),  # enforce whole crews
                        "spcm": float(spcm_extra),
                    }
                )

    # --- Build inputs for the requirement engine ---
    if demand_mode == "Crew Overhead Only":
        # Ignore crew ratio; requirement comes only from overhead pools
        eff_crew_ratio = 0.0
        eff_spcm = 0.0
    else:
        eff_crew_ratio = float(crew_ratio)
        eff_spcm = float(spcm)

    # Compute requirement
    rim_req = compute_crew_aircraft_requirement(
        pai=int(rim_pai),
        crew_ratio=eff_crew_ratio,
        sorties_per_crew_month=eff_spcm,
        om_days=int(om_days),
        turn_factor=float(turn_factor),
        attrition_rate=float(attr_rate),
        extra_crews=extra_crews,
    )

    # --- Core RIM aircraft requirement (crew-only) ---
    rim_ac = int(rim_req["aircraft_required"])
    crew_ac_req = rim_ac

    # We'll fold in non-crew overhead (spares/trainers/etc.) in a moment
    total_req = crew_ac_req + int(op_noncrew_total)

    # Margins for crew-only and total requirement
    margin_ep        = ep        - crew_ac_req
    margin_avail     = avail_eff - crew_ac_req
    margin_total_req = avail_eff - total_req

    st.markdown("### RIM Summary")

    c_struct, c_avail, c_req = st.columns(3)

    with c_struct:
        st.markdown(
            f"""
            **Inventory / Structure**  
            • TAI: **{rim_tai}**  
            • PAI: **{rim_pai}**  
            • BAI: **{rim_bai}**  
            • Depot+UPNR: **{nonposs}**  
            • Possessed: **{possessed}**
            """
        )

    with c_avail:
        st.markdown(
            f"""
            **Availability (Home)**  
            • NMC total: **{nmc_total}**  
            • EP: **{ep}**  
            • EP_home: **{ep_home}**  
            • Avail FHP (legacy): **{avail_legacy}**  
            • Avail FHP (+NMC Flyable): **{avail_eff}**
            """
        )

    # Turn pattern logic: what TF is required vs what TF can be supported
    lines_required = float(rim_req["daily_gross_lines"])  # scheduled lines/day (post-attrition)
    tf_used        = float(rim_req["turn_factor"])
    lines_supportable = avail_eff * tf_used if avail_eff > 0 else 0.0
    tf_required = (lines_required / avail_eff) if avail_eff > 0 else float("nan")

    with c_req:
        st.markdown(
            f"""
            **Crew-Only RIM Requirement**  
            • Aircraft required (RIM): **{crew_ac_req}**  

            **Non-crew overhead (daily tails)**  
            • Spares/Trainers/Fleet/Contingency/Alert: **{op_noncrew_total}**  

            **Total operational requirement**  
            • **{total_req}** tails/day  

            **Margins (vs Avail FHP + NMC-Flyable)**  
            • Crew-only margin: **{margin_avail:+.1f}** tails  
            • Total requirement margin: **{margin_total_req:+.1f}** tails  

            **Turn Pattern View**  
            • Daily lines required (crew demand): {lines_required:.2f}  
            • Lines supportable at current TF ({tf_used:.2f}): {lines_supportable:.2f}  
            • TF required using Avail FHP: {tf_required:.2f} sorties/jet/day
            """
        )

    # --- Crew & Sortie Breakdown (flattened 4-panel layout) ---
    st.markdown("### Crew & Sortie Breakdown")

    bc1, bc2, bc3, bc4 = st.columns(4)

    # Safely pull extra fields (with defaults if your rim_requirements.py is older)
    base_crews          = float(rim_req.get("base_crews",  crew_ratio * rim_pai))
    extra_crews_total   = float(rim_req.get("extra_crews_total", 0.0))
    total_crews         = float(rim_req.get("total_crews", base_crews + extra_crews_total))

    base_month          = float(rim_req.get("base_net_sorties_month", 0.0))
    extra_month         = float(rim_req.get("extra_net_sorties_month", 0.0))
    total_month         = float(rim_req.get("net_sorties_month", base_month + extra_month))

    daily_net           = float(rim_req.get("daily_net_sorties", total_month / max(1, om_days)))
    daily_gross         = float(rim_req.get("daily_gross_lines", daily_net / max(1e-6, 1 - attr_rate)))

    tf_val              = float(rim_req.get("turn_factor", turn_factor))
    attr_val            = float(rim_req.get("attrition_rate", attr_rate))
    om_days_val         = int(rim_req.get("om_days", om_days))

    # Enforce whole numbers for crews and sorties (no fractional crews/sorties)
    base_crews_i        = math.ceil(base_crews)
    extra_crews_i       = math.ceil(extra_crews_total)
    total_crews_i       = math.ceil(total_crews)

    daily_net_i         = math.ceil(daily_net)
    daily_gross_i       = math.ceil(daily_gross)
    total_month_i       = math.ceil(total_month)

    with bc1:
        st.markdown(
            f"""
            **Crews**  
            • Base crews (ratio × PAI): {base_crews_i}  
            • Extra crews (overhead pools): {extra_crews_i}  
            • **Total crews:** {total_crews_i}
            """
        )

    with bc2:
        st.markdown(
            f"""
            **Sorties – Monthly**  
            • Base crews: {math.ceil(base_month)}  
            • Extra crews: {math.ceil(extra_month)}  
            • **Total (executed requirement):** {total_month_i}
            """
        )

    with bc3:
        st.markdown(
            f"""
            **Sorties – Daily**  
            • **Crew requirement (executed):** {daily_net_i} sorties/day  
            • **Scheduled lines (after attrition):** {daily_gross_i} lines/day  
            """
        )

    with bc4:
        st.markdown(
            f"""
            **Assumptions**  
            • Turn Factor (TF): {tf_val:.2f} sorties/jet/day  
            • Attrition used: {attr_val:.0%}  
            • O&M days in month: {om_days_val}  

            _Crew requirement (executed) is fixed by crew ratio and SPCM._  
            _Attrition increases the **scheduled lines** needed to achieve that requirement._
            """
        )

    # --- Overall status based on total requirement ---
    if margin_total_req >= 0:
        st.success("RIM requirement (crew + overhead) is within structure and NMC-flyable headroom.")
    elif margin_avail >= 0:
        st.warning("Crew-only RIM is covered, but adding overhead pushes you close to the edge.")
    else:
        st.error(
            "Fleet structure is below crew-based RIM requirement — something has to give "
            "(alert, trainers, fleet mgmt, contingency, or commit rate)."
        )

    # ==========================
    # 4) Turn Pattern Recommendation (Crew vs Fleet)
    # ==========================
    st.markdown("---")
    st.subheader("🧮 Turn Pattern Recommendation (Crew vs Fleet)")

    # Crew-demand gross daily lines (after attrition)
    req_gross = max(float(rim_req.get("daily_gross_lines", 0.0)), 0.0)

    # Turn Factor actually used
    tf_used = max(float(rim_req.get("turn_factor", turn_factor)), 1e-6)

    # --- Required pattern from crew demand (2-Go baseline) ---
    req_am_2go = math.ceil(req_gross / tf_used) if req_gross > 0 else 0
    # raw PM if you literally try to enforce TF with 2 goes
    raw_pm_2go = math.ceil(req_am_2go * (tf_used - 1.0)) if tf_used > 1 else 0
    # cap PM so nights are never heavier than days
    req_pm_2go = min(raw_pm_2go, req_am_2go)

    # --- Supportable pattern based on available fleet (2-Go baseline) ---
    commit_cap_pct = st.slider(
        "Commitment Cap for Turn Pattern (%)",
        min_value=40.0, max_value=100.0,
        value=65.0, step=1.0,
        key="rim_commit_cap_turn",
        help="Upper limit on how much of the available fleet may be committed."
    )
    commit_cap = commit_cap_pct / 100.0

    avail_for_tf = max(float(avail_eff), 0.0)

    sup_am_2go = math.floor(avail_for_tf * commit_cap)
    sup_pm_2go = math.floor(sup_am_2go * (tf_used - 1.0)) if tf_used > 1 else 0
    sup_pm_2go = min(sup_pm_2go, sup_am_2go)  # same 'night < day' guard

    actual_commit_pct = (
        sup_am_2go / avail_for_tf * 100.0
        if avail_for_tf > 0 else 0.0
    )

    col_req, col_sup, col_assess = st.columns(3)

    # ----------------------------
    # REQUIRED PATTERN (CREW DEMAND) — 2-Go baseline
    # ----------------------------
    with col_req:
        st.markdown(
            f"""
            **Required (Crew Demand — 2-Go)**  
            • Daily gross sorties (crew-driven): **{req_gross:.1f}**  
            • Required AM tails: **{req_am_2go}**  
            • Required PM turns (capped ≤ AM): **{req_pm_2go}**  
            • **Required pattern (2-Go):** **{req_am_2go} × {req_pm_2go}**  
            • **TF used:** {tf_used:.2f}
            """
        )
        if abs(tf_used - 1.0) < 1e-6:
            st.info("TF = 1.0 eliminates PM turns. Increase TF to reduce AM demand.")
    
        # ================================
        # 🟦 Surge Mode Toggle (3rd-Go On/Off)
        # ================================
        surge_on = st.checkbox(
            "Enable 3rd-Go Surge Recommendation",
            value=False,
            key="rim_enable_3go",
            help="Allows a 3rd go to spread sortie demand and reduce PM burden."
        )

        if surge_on and req_am_2go > 0 and tf_used > 1.0:

            # total sorties required under TF
            total_sorties = math.ceil(tf_used * req_am_2go)
            extra_sorties = max(total_sorties - req_am_2go, 0)

            # split extra sorties (approx 70/30) between PM and Go3
            pm_3go = math.ceil(extra_sorties * 0.70)
            go3_3go = extra_sorties - pm_3go

            # clamp to avoid impossible patterns
            pm_3go = max(0, min(pm_3go, req_am_2go))
            go3_3go = max(0, min(go3_3go, req_am_2go))

            # ensure PM >= Go3
            if go3_3go > pm_3go:
                pm_3go, go3_3go = go3_3go, pm_3go

            # recompute TF achieved by this discrete pattern
            sorties_3go = req_am_2go + pm_3go + go3_3go
            tf_3go = sorties_3go / req_am_2go if req_am_2go else tf_used

            st.markdown(
                f"""
                ### Surge (3-Go) Pattern  
                • AM: **{req_am_2go}**  
                • PM: **{pm_3go}**  
                • 3rd Go: **{go3_3go}**  
                • **Surge pattern:** **{req_am_2go} × {pm_3go} × {go3_3go}**  
                • Achieved TF: **{tf_3go:.2f}**
                """
            )

    # ----------------------------
    # SUPPORTABLE PATTERN (FLEET) — 2-Go baseline
    # ----------------------------
    with col_sup:
        st.markdown(
            f"""
            **Supportable (Fleet & Cap — 2-Go)**  
            • Avail for FHP (+NMC Flyable): **{avail_for_tf:.1f}**  
            • Commit cap: **{commit_cap_pct:.0f}%**  
            • Supportable AM tails: **{sup_am_2go}**  
            • Supportable PM turns: **{sup_pm_2go}**  
            • **Supportable pattern (2-Go):** **{sup_am_2go} × {sup_pm_2go}**  
            • **Actual Commit % Used:** {actual_commit_pct:.0f}%  
            • **TF used:** {tf_used:.2f}
            """
        )

    # ----------------------------
    # ASSESSMENT + RECOMMENDATIONS
    # ----------------------------
    with col_assess:
        # Min commit cap required to cover required AM (2-Go)
        rec_commit_cap = (
            req_am_2go / avail_for_tf * 100.0
            if avail_for_tf > 0 else float("inf")
        )

        # Min TF to hit crew sorties if you limit PM to AM (2-Go assumption)
        if req_am_2go > 0:
            rec_tf = req_gross / req_am_2go
        else:
            rec_tf = tf_used

        if sup_am_2go >= req_am_2go and sup_pm_2go >= req_pm_2go:
            st.success("Fleet can support the required crew-driven 2-Go pattern at this commit level.")
        else:
            st.error("Fleet cannot support the required crew-driven 2-Go pattern at this commit level.")

        st.markdown(
            f"""
            **Recommendations**  
            • **Min commit rate required (2-Go):** {rec_commit_cap:.0f}%  
            • **Min TF suggested (holding AM at {req_am_2go}):** {rec_tf:.2f}  
            """
        )



    # ==========================
    # 4b) 30-Day RIM Projection (What-if)
    # ==========================
    st.markdown("---")
    st.subheader("📈 30-Day RIM Projection (What-if Degraders)")

    proj_c1, proj_c2 = st.columns(2)
    with proj_c1:
        degr_sched = st.number_input(
            "Scheduled degraders in next 30 days",
            min_value=0,
            value=0,
            step=1,
            key="rim_proj_sched_beta",
            help="E.g., phase, ISO, PDM, major inspections scheduled."
        )
    with proj_c2:
        degr_unsched = st.number_input(
            "Expected unscheduled degraders (trend-based)",
            min_value=0,
            value=0,
            step=1,
            key="rim_proj_unsched_beta",
            help="E.g., trend-based NMCM, new supply issues, cann birds."
        )

    total_proj_degr = degr_sched + degr_unsched
    projected_ep = max(ep - total_proj_degr, 0)
    projected_margin = projected_ep - rim_ac

    st.info(
        f"**Projected EP in 30 days:** {projected_ep} tails  \n"
        f"Total new degraders: **{total_proj_degr}**  \n"
        f"Projected margin vs RIM: **{projected_margin:+.1f}** aircraft"
    )

    if projected_margin >= 0:
        st.success("Projection: you remain inside RIM requirement under this degrader forecast.")
    else:
        st.error("Projection: under this degrader forecast, you will fall below RIM requirement.")
