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


#          Sidebar section label           
st.sidebar.markdown(" Simulation Modules ")

# ---------- Simulation Selector ----------
sim_options = [
    "SPECTRE Flight Manual",  # ELIF Placed under North Star Analysis
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

    # --- 0) First Look metadata (for export & filing) ---
    with st.expander("📌 First Look Metadata", expanded=True):
        unit_name = st.text_input(
            "Unit / Wing / MDS (for export)",
            value="",
            key="annual_firstlook_unit",
            help="Example: 94 AW / C-130H or 301 FW / F-35A"
        )
        run_date = st.date_input(
            "Run date",
            value=None,
            key="annual_firstlook_date",
            help="Date this First Look was generated (used in the export)."
        )

    # --- 1) Upload historical data ---
    uploaded = st.file_uploader("Upload 4-yr roll-up (sheet 'Historical Data Input')", type=["xlsx","xls"])
    if not uploaded:
        st.info("Please upload your Excel file to proceed."); st.stop()
    df_hist = pd.read_excel(uploaded, sheet_name="Historical Data Input") # Excel must be titled "Historical Data Input"
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
        "O&M Days": [20] * 12,

        # Planned non-possessed tails by month (Depot / Mods / UPNR)
        "Depot/UPNR Tails": [0] * 12,

        # Degrader bucket (we’ll later split into sched/unsch if desired)
        "Scheduled MX Tails": [0] * 12,

        # NEW: Off-station bins (force-flow)
        "Deployment Tails": [0] * 12,          # off-station, no home training
        "TDY Tails": [0] * 12,                  # off-station, can still produce hours
        "TDY Hours per Tail (mo)": [0.0] * 12,   # avg monthly hours produced per TDY tail

        "Turn Pattern": ["8x6"] * 12,
        "Commit %": [65] * 12,
        "ASD": [2.0] * 12,
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
    with st.expander("🎯 FY Goals & Assumptions", expanded=True):
        TAI         = st.number_input("TAI (Aircraft Inventory)", value=12, step=1)
        FY_goal     = st.number_input("FY Flying-Hour Goal (hrs)", value=0.0, step=100.0)
        ASD         = st.number_input("Avg Sortie Duration (hrs)", value=2.0, step=0.1)

        MC_target   = st.slider(
            "MC-Rate Target (%)", 0, 100, 80,
            help="Goal setter only. Used for comparison vs simulated MC mean by month."
        )

        mc_delta = st.slider(
            "Adjust MC Rate (%)", -10, 10, 0, step=1,
            help=(
                "Applies a proportional adjustment to the imported/calculated MC rate for testing.\n"
                "Example: +5 increases MC by 5% (0.80 → 0.84). This does NOT change the upload; "
                "it only modifies the simulation inputs."
            )
        )

        uncertainty = st.slider(
            "Monthly uncertainty (±%)",
            min_value=0, max_value=20, value=5, step=1,
            help="Random variation applied to AA/MC and execution rates each trial/month."
        ) / 100.0

    # --- 5) Extract planning inputs as dicts keyed by month_num ---
    om_days = {m: int(plan_df.loc[i, "O&M Days"]) for i, m in enumerate(months)}
    depot_upnr_tails = {m: int(plan_df.loc[i, "Depot/UPNR Tails"]) for i, m in enumerate(months)}

    # FIX: Map the UI column "Scheduled MX Tails" to the variable "planned_degraders"
    planned_degraders = {m: int(plan_df.loc[i, "Scheduled MX Tails"]) for i, m in enumerate(months)}

    deployment_tails = {m: int(plan_df.loc[i, "Deployment Tails"]) for i, m in enumerate(months)}
    tdy_tails = {m: int(plan_df.loc[i, "TDY Tails"]) for i, m in enumerate(months)}
    tdy_hours_per_tail = {m: float(plan_df.loc[i, "TDY Hours per Tail (mo)"]) for i, m in enumerate(months)}

    turn_patterns = {m: str(plan_df.loc[i, "Turn Pattern"]) for i, m in enumerate(months)}
    commit_rates = {m: float(plan_df.loc[i, "Commit %"]) / 100.0 for i, m in enumerate(months)}
    asd_dict = {m: float(plan_df.loc[i, "ASD"]) for i, m in enumerate(months)}

    # --- 6) Run Simulation Button ---
    if st.button("Run Annual Analysis"):
        rates_df_adj = rates_df.copy()
        if mc_delta != 0:
            rates_df_adj["mc_rate"] = (rates_df["mc_rate"] * (1 + mc_delta/100)).clip(upper=1.0)

        sim_params = {
            "rates_df": rates_df_adj,
            "TAI": TAI,
            "om_days": om_days,
            # NOW THESE MATCH: Variable created in Step 5 is used here in Step 6
            "planned_degraders": planned_degraders, 
            "planned_depot_upnr": depot_upnr_tails,
            "planned_deploy_tails": deployment_tails,
            "planned_tdy_tails": tdy_tails,
            "planned_tdy_hours_per_tail": tdy_hours_per_tail,
            "turn_patterns": turn_patterns,
            "commit_rates": commit_rates,
            "uncertainty": uncertainty,
        }

        sim = HistoricalAnnualSimulation()
        sim.params = sim_params

        # Ensure the simulation knows 'Degraders' are 'Scheduled Maintenance'
        if "degraders" in rates_df.columns:
            rates_df["nmc_sched"] = rates_df["degraders"]

        # Execute simulation
        all_trials, summary = sim.simulate(trials=500)

        # --- CORRECTED METRIC CAPTURE ---

        # 1. Calculate Total Hours for EVERY trial (500 totals)
        # We multiply sorties (flown) by ASD to get actual hours
        trial_hour_totals = []
        for trial in all_trials:
            trial_total = 0.0
            for m_result in trial:
                sorties = m_result.get('flown', 0)
                # Use the month-specific ASD we planned earlier
                asd_val = asd_dict.get(m_result['month'], ASD)
                trial_total += (sorties * asd_val)
            trial_hour_totals.append(trial_total)

        # 2. Mean Flown = Average of those 500 totals
        mean_flown = float(np.mean(trial_hour_totals))

        # 3. Probability of Success = % of trials hitting the hour goal
        if len(trial_hour_totals) > 0:
            prob_success = float(np.sum(np.array(trial_hour_totals) >= FY_goal) / len(trial_hour_totals))
        else:
            prob_success = 0.0

        # 4. Status Delta (Shows the gap to the commander)
        gap = mean_flown - FY_goal
        if prob_success >= 0.80:
            status_delta = f"Executable ({gap:+,.0f} hrs)"
        elif prob_success >= 0.50:
            status_delta = f"Risk ({gap:+,.0f} hrs)"
        else:
            status_delta = f"High Risk ({gap:+,.0f} hrs)"

        # Handle summary structure
        # 1. Flatten the summary results into a DataFrame
        main_results = summary[0] if isinstance(summary, list) and isinstance(summary[0], list) else summary
        results_df = pd.DataFrame(main_results)

        # 2. Extract specific Trial 0 for the Maintenance Chart
        if all_trials and len(all_trials) > 0:
            df_maintenance = pd.DataFrame(all_trials[0])
            df_maintenance["month_name"] = [calendar.month_abbr[int(m)] for m in df_maintenance["month"]]
        else:
            df_maintenance = pd.DataFrame()

        # 3. THE BRIDGE: Map Simulation keys to Frontend keys
        # We use a dictionary to check if the column actually exists before renaming
        rename_map = {
            "mc_sim": "mc_rate_mean",
            "flyable_ac": "avg_flyable",
            "mc_hist": "mc_hist",
            "scheduled": "scheduled_mean",
            "flown": "flown_mean"
        }

        # Apply the rename
        results_df.rename(columns=rename_map, inplace=True)
        
        # --- SAFETY FALLBACK ---
        # If 'mc_rate_mean' is STILL missing, the sim might be returning 
        # mc_sim_mean instead. This catch-all prevents the UI error.
        if "mc_rate_mean" not in results_df.columns:
            if "mc_sim_mean" in results_df.columns:
                results_df.rename(columns={"mc_sim_mean": "mc_rate_mean"}, inplace=True)
            elif "mc_sim" in results_df.columns:
                results_df["mc_rate_mean"] = results_df["mc_sim"]
        
        # Calculate derived metrics for the frontend
        if "hours_flown" not in results_df.columns and "flown_mean" in results_df.columns:
            # Match ASD back to month for accurate hour calculation
            results_df["hours_flown"] = results_df.apply(
                lambda x: x["flown_mean"] * asd_dict.get(x["month"], ASD), axis=1
            )

        # Patch: Ensure months and month names are set
        months_seq = list(range(10, 13)) + list(range(1, 10))
        if "month" not in results_df.columns:
            results_df["month"] = months_seq[:len(results_df)]
        results_df["month_name"] = [calendar.month_abbr[int(m)] for m in results_df["month"]]

        # --- STEP 1: CALCULATE STATUS & BUILD RISK BUBBLES ---

        # A. Calculate the thresholds for the "Commander's Logic"
        worst_idx = results_df['mc_rate_mean'].idxmin()
        worst_month_name = results_df.loc[worst_idx, 'month_name']
        worst_mc_val = results_df.loc[worst_idx, 'mc_rate_mean']
        
        if prob_success >= 0.80:
            status_label, status_col, status_icon = "HEALTHY", "green", "✅"
            status_msg = f"Plan is **fully executable**. Fleet health remains stable."
        elif prob_success >= 0.50:
            status_label, status_col, status_icon = "CAUTION", "orange", "⚠️"
            status_msg = f"**Executable with risk.** Critical constraints in {worst_month_name} ({worst_mc_val:.1%} MC)."
        else:
            status_label, status_col, status_icon = "CRITICAL", "red", "🚨"
            status_msg = f"**Plan unsupportable.** High risk of fleet grounding in {worst_month_name}."
        
        # B. Build the Bubble Chart with matching colors
        risk_colors = []
        for mc in results_df["mc_rate_mean"]:
            if mc >= 0.80: risk_colors.append("#2ecc71") # Green
            elif mc >= 0.50: risk_colors.append("#f39c12") # Orange
            else: risk_colors.append("#e74c3c") # Red
        
        fig_risk_bubbles = go.Figure()
        fig_risk_bubbles.add_trace(go.Scatter(
            x=results_df["month_name"],
            y=results_df["mc_rate_mean"],
            mode='markers+text',
            text=[f"{val:.0%}" for val in results_df["mc_rate_mean"]],
            textposition="top center",
            marker=dict(size=40, color=risk_colors, opacity=0.9, line=dict(color='black', width=2))
        ))
        
        fig_risk_bubbles.update_layout(
            title="<b>Monthly Fleet Risk Matrix</b>",
            yaxis=dict(title="MC Rate", tickformat=".0%", range=[0.4, 1.0]),
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            showlegend=False
        )
         # Card patch 1 ends
         
         # --- BUILD THE MAINTENANCE SATURATION CHART (fig_mx) ---
        import plotly.graph_objects as go

        fig_mx = go.Figure()

        # Scheduled Maintenance (The predictable stuff)
        fig_mx.add_trace(go.Bar(
            x=results_df["month_name"],
            y=results_df["nmc_sched_mean"],
            name="Scheduled (Avg)",
            marker_color='#3498db',
            hovertemplate='%{y:.1f} AC'
        ))
        
        # Unscheduled Maintenance (The "breaks")
        fig_mx.add_trace(go.Bar(
            x=results_df["month_name"],
            y=results_df["nmc_unsch_mean"],
            name="Unscheduled (Avg)",
            marker_color='#e67e22',
            hovertemplate='%{y:.1f} AC'
        ))
        
        fig_mx.update_layout(
            # Explicitly label as Fleet Average
            title="<b>Fleet Maintenance Saturation</b> (Avg of 500 Trials)",
            barmode='stack',
            xaxis_title="Month",
            yaxis_title="Avg Aircraft Offline",
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            # Add a hover compare mode so they see the total at a glance
            hovermode="x unified"
        )

         # Card Patch 2 begin
         
         # --- STEP 2: DISPLAY THE BRIEFING CARD ---
        st.markdown("---")
        st.header("⚡ Commander's Briefing Card")
        
        # Use a container to box the summary so it stands out
        with st.container(border=True):
            # Header showing the Stoplight Status
            st.markdown(f"### Current Fleet Readiness: :{status_col}[{status_icon} {status_label}]")
            
            # 4-Column Metric Row
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Prob. of Success", f"{prob_success:.1%}", 
                          delta=status_delta, 
                          delta_color="normal" if status_col == "green" else "inverse")
            with c2:
                st.metric("Mean Hours", f"{mean_flown:,.0f}", delta=f"{mean_flown - FY_goal:,.0f} vs Goal")
            with c3:
                st.metric("Avg Fleet MC", f"{results_df['mc_rate_mean'].mean():.1%}")
            with c4:
                st.metric("Worst Month", worst_month_name, help=f"Lowest MC recorded: {worst_mc_val:.1%}")
        
            # Actionable Insight Box
            st.info(f"**COMMANDER'S ACTION:** {status_msg}")
        
            # Visuals Row: Risk Matrix & Maintenance Saturation
            v1, v2 = st.columns(2)
            with v1:
                # Displays the Bubble Chart we prepped in Step 1
                st.plotly_chart(fig_risk_bubbles, use_container_width=True)
            with v2:
                # Displays the Maintenance Saturation Chart
                if 'fig_mx' in locals():
                    st.plotly_chart(fig_mx, use_container_width=True)
                else:
                    st.warning("Maintenance Detail Chart not found in memory.")
        
        st.markdown("---")
         
        # --- MC Target vs Sim (display-only) ---
        st.subheader("🎯 MC Target vs Simulated MC")
        
        import plotly.graph_objects as go
        mc_target = float(MC_target) / 100.0
        
        if "mc_rate_mean" in results_df.columns:
            # Use the already calculated mc_hist from bridge or rebuild
            results_df["mc_gap_to_target"] = results_df["mc_rate_mean"] - mc_target
        
            months_below = int((results_df["mc_gap_to_target"] < 0).sum())
            worst_gap = float(results_df["mc_gap_to_target"].min())
        
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("MC Target", f"{mc_target:.0%}")
            with c2:
                st.metric("Months below target", f"{months_below}/12")
            with c3:
                st.metric("Worst gap", f"{worst_gap:+.1%}")
        
            fig_mc = go.Figure()
            fig_mc.add_trace(go.Scatter(
                x=results_df["month_name"], y=results_df["mc_rate_mean"],
                mode="lines+markers", name="Simulated MC (mean)"
            ))
        
            if "mc_hist" in results_df.columns and results_df["mc_hist"].notna().any():
                fig_mc.add_trace(go.Scatter(
                    x=results_df["month_name"], y=results_df["mc_hist"],
                    mode="lines+markers", name="Historical MC (from upload)"
                ))
        
            fig_mc.add_hline(
                y=mc_target, line_dash="dash",
                annotation_text="MC Target", annotation_position="top left"
            )
            fig_mc.update_layout(
                yaxis_tickformat=".0%", yaxis_title="MC Rate",
                xaxis_title="Month", title="Simulated MC vs Historical MC"
            )
            st.plotly_chart(fig_mc, use_container_width=True)
        else:
            st.error("Error: Simulation did not return 'mc_rate_mean'. Check class mapping.")

        with st.expander("⚠️ Alerts & Warnings", expanded=False):
            st.caption(
                "This panel separates **structure risk** (tails removed from home) from **performance risk** (MC/execution). "
                "Supply here follows the sim engine: **flyable_home ≈ floor(tai_home × MC)**."
            )
        
            for m in months:
                # Use adjusted rates to match the sim logic
                month_match = rates_df_adj[rates_df_adj["month_num"] == m]
                if month_match.empty:
                    st.info(f"{calendar.month_abbr[m]}: No historical rate data available.")
                    continue
                
                r = month_match.iloc[0]
                mc_used = float(r.get("mc_rate", 0.0))
        
                # Structure inputs
                degr = int(planned_degraders.get(m, 0))
                depot = int(depot_upnr_tails.get(m, 0))
                deploy = deployment_tails.get(m, 0)
                tdy = tdy_tails.get(m, 0)
        
                tai_home = max(0, int(TAI) - degr - depot - deploy - tdy)
        
                # Plan requirement
                # Safely parse the first-go from turn pattern (e.g., "8x6" -> 8)
                pattern_str = str(turn_patterns.get(m, "0"))
                try:
                    first_go = int(pattern_str.split("x")[0]) if "x" in pattern_str else int(pattern_str)
                    sorties_per_day = sum(map(int, [x for x in pattern_str.split("x") if x.isdigit()]))
                except ValueError:
                    first_go = 0
                    sorties_per_day = 0

                days = int(om_days.get(m, 0))
                scheduled = sorties_per_day * days
        
                # Performance-driven supply
                flyable = max(0, math.floor(tai_home * mc_used))
        
                # Commit percent calculation (Safe from ZeroDivision/RuntimeWarning)
                if flyable > 0:
                    commit_pct = (first_go / flyable * 100.0)
                else:
                    commit_pct = 0.0 # Set to 0 or handle as 100% depending on preference
        
                # Limiting factor logic
                structure_limited = (tai_home < first_go) if first_go > 0 else False
                rate_limited = (tai_home >= first_go and flyable < first_go) if first_go > 0 else False
        
                # Header for the specific month
                status_text = f"**{calendar.month_abbr[m]}** — {tai_home} home | MC {mc_used:.1%} → {flyable} flyable | First-go {first_go}"
                
                # Logic Branch for Alerts
                if first_go == 0 or days == 0:
                    st.info(f"{status_text} | No flying scheduled.")
                    continue
        
                if tai_home <= 0:
                    st.error(f"❌ {calendar.month_abbr[m]}: STRUCTURE-LIMITED — All tails are in maintenance/off-station.")
                elif structure_limited:
                    st.error(
                        f"❌ {calendar.month_abbr[m]}: STRUCTURE-LIMITED — Total home aircraft ({tai_home}) cannot support first-go ({first_go}) even at 100% MC."
                    )
                elif rate_limited:
                    st.warning(
                        f"⚠️ {calendar.month_abbr[m]}: RATE-LIMITED — {tai_home} tails available, but {mc_used:.1%} MC only yields {flyable} flyable (Need {first_go})."
                    )
                elif commit_pct > 80:
                    st.warning(
                        f"⚠️ {calendar.month_abbr[m]}: HIGH COMMIT — Utilizing {commit_pct:.1f}% of flyable assets. High risk of ground aborts."
                    )
                else:
                    st.success(f"✅ {calendar.month_abbr[m]}: Healthy — {commit_pct:.1f}% commit.")

        # 7b) Probability of Meeting Flying Hour Goal ---
        st.subheader("🎯 Probability of Meeting Flying Hour Goal")
        
        all_hours_flown = []
        
        # We need the tail counts and hours per tail to calculate the total off-station contribution
        # tdy_tails: {month_num: count}, tdy_hours_per_tail: {month_num: hours_per_tail}
        
        for trial in all_trials:
            trial_total_hours = 0.0
            for m_result in trial:
                m_num = m_result["month"]
                
                # 1. Home Station Hours: (Simulated Flown Sorties * ASD)
                # Use the trial-specific 'flown' if available, else fall back to mean
                flown_sorties = m_result.get("flown", m_result.get("flown_mean", 0))
                asd_val = asd_dict.get(m_num, ASD)
                home_hours = flown_sorties * asd_val
                
                # 2. TDY/Off-Station Hours: (Number of Tails * Hours per Tail)
                num_tdy_tails = tdy_tails.get(m_num, 0)
                hrs_per_tdy_tail = tdy_hours_per_tail.get(m_num, 0.0)
                offstation_hours = num_tdy_tails * hrs_per_tdy_tail
                
                trial_total_hours += (home_hours + offstation_hours)

            all_hours_flown.append(trial_total_hours)

        # Statistics
        mean_flown = np.mean(all_hours_flown)
        std_flown = np.std(all_hours_flown)
        n_success = sum(h >= FY_goal for h in all_hours_flown)
        prob_success = n_success / len(all_hours_flown) if all_hours_flown else 0.0

        # Metrics display
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Probability of Goal", f"{prob_success:.1%}")
        with c2:
            st.metric("Mean Annual Hours", f"{mean_flown:,.0f}")
        with c3:
            st.metric("Goal Shortfall", f"{min(0, mean_flown - FY_goal):,.0f}")

        if prob_success < 0.5:
            st.warning(f"⚠️ High Risk: There is only a {prob_success:.1%} chance of hitting the {FY_goal:,.0f} hour goal based on current historical rates and planning bins.")
        else:
            st.success(f"✅ Goal Attainable: Simulation suggests a {prob_success:.1%} confidence level.")

        # Distribution Plot
        import plotly.graph_objects as go
        fig_prob = go.Figure()
        fig_prob.add_trace(go.Histogram(
            x=all_hours_flown, 
            nbinsx=40, 
            name="Simulated Trials",
            marker_color='#636EFA',
            opacity=0.75
        ))
        
        # Add Goal Line
        fig_prob.add_vline(x=FY_goal, line_color="red", line_dash="dash", 
                          annotation_text=f"FY Goal: {FY_goal:,.0f}", 
                          annotation_position="top right")
        
        fig_prob.update_layout(
            xaxis_title="Total FY Flying Hours",
            yaxis_title="Trial Frequency",
            title="Monte Carlo Distribution of Annual Flying Hours",
            bargap=0.05
        )
        st.plotly_chart(fig_prob, use_container_width=True)

        # --- 8) Visuals for Scheduled/Flown Sorties and Hours ---
        st.subheader("📈 Sorties Scheduled vs Flown")
        
        # Ensure we use the bridged column names
        sched_col = "scheduled_mean"
        flown_col = "flown_mean"

        fig1 = go.Figure()
        fig1.add_trace(go.Bar(x=results_df["month_name"], y=results_df[sched_col], name="Scheduled"))
        fig1.add_trace(go.Bar(x=results_df["month_name"], y=results_df[flown_col], name="Flown"))
        fig1.update_layout(barmode="group", yaxis_title="Sorties", hovermode="x unified")
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("📈 Total Hours (Home + Off-station)")
        
        # We use the hours_flown we calculated in the Probability section which includes TDY
        # If you want to show Scheduled Hours vs Actual Flown:
        hours_sched = results_df[sched_col] * [asd_dict.get(m, ASD) for m in results_df["month"]]

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=results_df["month_name"], y=hours_sched, name="Home Hours Scheduled"))
        fig2.add_trace(go.Bar(x=results_df["month_name"], y=results_df["hours_flown"], name="Total Hours Flown (Inc. TDY)"))
        fig2.update_layout(barmode="group", yaxis_title="Hours", hovermode="x unified")
        st.plotly_chart(fig2, use_container_width=True)

        # --- NEW SECTION: Maintenance Recovery Visualization (TRUE AVERAGES) ---
        st.subheader("🛠️ Maintenance Recovery & Fix Cycle")
        
        # We use results_df because it contains the means we just added to _summarize
        if "fixes_8hr_mean" in results_df.columns:
            fig_mx = go.Figure()
            
            fig_mx.add_trace(go.Bar(name='8hr Fix', x=results_df['month_name'], y=results_df['fixes_8hr_mean'], marker_color='#2ecc71'))
            fig_mx.add_trace(go.Bar(name='12hr Fix', x=results_df['month_name'], y=results_df['fixes_12hr_mean'], marker_color='#f1c40f'))
            fig_mx.add_trace(go.Bar(name='24hr Fix', x=results_df['month_name'], y=results_df['fixes_24hr_mean'], marker_color='#e67e22'))
            fig_mx.add_trace(go.Bar(name='Long Fix', x=results_df['month_name'], y=results_df['long_fixes_mean'], marker_color='#e74c3c'))

            fig_mx.update_layout(
                barmode='stack', 
                title="Monthly Aircraft Fix Distribution (Avg of 500 Trials)",
                yaxis_title="Avg Number of Aircraft",
                hovermode="x unified"
            )
            st.plotly_chart(fig_mx, use_container_width=True)
            
            # Use the mean load pct we added
            avg_load = results_df['hangar_load_pct_mean'].mean()
            st.info(f"💡 **Insight:** Across all simulations, your hangar load averages **{avg_load:.1f}%**.")
        else:
            st.warning("Maintenance fix data (8/12/24hr) not found in simulation output. Check historical_annual.py exports.")

        # --- NEW SECTION: FLEET INVENTORY SAND CHART ---
        st.subheader("📊 Fleet Inventory & Commit Risk")
        
        # --- 1. PRE-CALCULATE (Individual Categories) ---
        flyable_ac = results_df['flyable_mean']
        nmc_sched  = results_df['nmc_sched_mean']
        
        # Pull the specific means we added to _summarize
        depot_tails   = results_df.get('depot_mean', pd.Series([0.0]*len(results_df)))
        deploy_tails  = results_df.get('deployment_mean', pd.Series([0.0]*len(results_df)))
        tdy_tails     = results_df.get('tdy_mean', pd.Series([0.0]*len(results_df)))

        # DYNAMIC REMAINDER
        # This math now explicitly subtracts every single known category from TAI
        nmc_unsch = TAI - (flyable_ac + nmc_sched + depot_tails + deploy_tails + tdy_tails)
        nmc_unsch = nmc_unsch.clip(lower=0)

        # --- 2. BUILD THE FIGURE (Explicit Layers) ---
        fig_sand = go.Figure()

        # Foundation: Flyable
        fig_sand.add_trace(go.Bar(name='Flyable (MC)', x=results_df['month_name'], y=list(flyable_ac), marker_color='#2ecc71'))
        
        # The 'Broken' Remainder
        fig_sand.add_trace(go.Bar(name='Unscheduled MX', x=results_df['month_name'], y=list(nmc_unsch), marker_color='#e67e22'))
        
        # Planned Maintenance
        fig_sand.add_trace(go.Bar(name='Scheduled MX', x=results_df['month_name'], y=list(nmc_sched), marker_color='#3498db'))
        
        # --- THE OFF-STATION SPLIT (Ensures Depot is visible) ---
        fig_sand.add_trace(go.Bar(name='Deployment', x=results_df['month_name'], y=list(deploy_tails), marker_color='#9b59b6'))
        fig_sand.add_trace(go.Bar(name='TDY', x=results_df['month_name'], y=list(tdy_tails), marker_color='#8e44ad'))
        fig_sand.add_trace(go.Bar(name='Depot / UPNR', x=results_df['month_name'], y=list(depot_tails), marker_color='#95a5a6'))

        # 3. OVERLAY COMMIT LINE (Secondary Axis)
        fig_sand.add_trace(go.Scatter(
            name='Commit Rate (%)', 
            x=results_df['month_name'], 
            y=results_df['commit_pct_mean'],
            yaxis='y2', 
            line=dict(color='#f1c40f', width=4, dash='dot'),
            hovertemplate='%{y:.1f}%'
        ))
        
        # 4. LAYOUT & THRESHOLD
        fig_sand.update_layout(
            barmode='stack',
            title=f"Total Fleet Inventory Allocation (TAI: {TAI})",
            yaxis=dict(title="Number of Aircraft", range=[0, TAI], fixedrange=True),
            yaxis2=dict(
                title="Commit Rate (%)",
                overlaying='y',
                side='right',
                range=[0, 105], # Room for the label
                showgrid=False
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified"
        )

        # Add Risk Threshold Line (Connected to the secondary % axis)
        fig_sand.add_hline(
            y=MC_target, 
            line_dash="dot",
            line_color="red",
            annotation_text=f"Risk Threshold ({MC_target}%)",
            annotation_position="top right",
            yref="y2" 
        )

        st.plotly_chart(fig_sand, use_container_width=True)
        # --- 9) Downloadable Table ---
        st.subheader("📋 Detailed Results Table")
        
        # Specified column order (Matches simulation output + Bridge)
        csv_columns = [
            "month", "month_name",
            "scheduled_mean",
            "flown_mean",
            "hours_flown",
            "asd_mean",
            "mc_rate_mean",
            "mc_gap_to_target",
            "execution_rate_mean",
            "avg_flyable",
            "overcommit_risk",
            "break_rate_mean",
            "gab_rate_mean"
        ]

        # Final check for mc_gap (redundancy for safety)
        if "mc_rate_mean" in results_df.columns:
            results_df["mc_gap_to_target"] = results_df["mc_rate_mean"] - mc_target

        # Create the display copy
        results_df_totals = results_df[[c for c in csv_columns if c in results_df.columns]].copy()

        # Build the totals row safely
        numeric_cols = ["scheduled_mean", "flown_mean", "hours_flown"]
        total_row = {col: results_df[col].sum() for col in numeric_cols if col in results_df.columns}
        total_row["month_name"] = "TOTAL FY"
        total_row["month"] = 99 # Sort helper

        # Append the total row
        results_df_totals = pd.concat([
            results_df_totals,
            pd.DataFrame([total_row])
        ], ignore_index=True)

        # Add Metadata
        unit_safe = (unit_name or "Unknown_Unit").strip()
        run_date_safe = str(run_date) if run_date else "No_Date"

        results_df_totals.insert(0, "unit", unit_safe)
        results_df_totals.insert(1, "run_date", run_date_safe)
        results_df_totals.insert(2, "FY_goal_hours", float(FY_goal))

        # Show table
        st.dataframe(results_df_totals.style.format({
            "mc_rate_mean": "{:.1%}",
            "mc_gap_to_target": "{:+.1%}",
            "execution_rate_mean": "{:.1%}",
            "hours_flown": "{:.1f}",
            "avg_flyable": "{:.1f}"
        }, na_rep="-"), use_container_width=True)

        # Download as CSV
        csv_buf = results_df_totals.to_csv(index=False).encode("utf-8")
        safe_unit = unit_safe.replace("/", "-").replace(" ", "_")
        file_name = f"FirstLook_{safe_unit}_{run_date_safe}.csv"
        
        st.download_button(
            "📥 Download Detailed Results CSV",
            csv_buf,
            file_name=file_name,
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
    # 2b) Degraders & NMC Bins
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

    # --- Crew demand mode: ratio+overhead vs overhead-only vs Training (NS Trg) ---
    demand_mode = st.radio(
        "Crew demand mode",
        [
            "Crew Ratio + Crew Overhead",
            "Crew Overhead Only",
            "NS Trg (Training Flight)",
        ],
        key="rim_demand_mode_beta",
        help=(
            "• Crew Ratio + Crew Overhead: use PAI-based crew ratio and add overhead pools.\n"
            "• Crew Overhead Only: ignore crew ratio and use only the CMR/BMC crew inputs below.\n"
            "• NS Trg: training production model using Student crews + IP sorties + refly rate."
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
    
    # Attrition affects how many *scheduled* lines are needed to achieve the executed requirement
    attr_rate = st.slider(
        "Sortie attrition (for RIM calc) %",
        min_value=0.0,
        max_value=50.0,
        value=20.0,
        step=1.0,
        key="rim_attr_rate_beta",
    ) / 100.0
    
    # ==========================
    # NS Trg (Training Flight) inputs + compute (no crew ratio, no CMR/BMC overhead)
    # ==========================
    if demand_mode == "NS Trg (Training Flight)":
        st.markdown("**NS Trg Inputs (Training Flight)**")
    
        tr_c1, tr_c2, tr_c3 = st.columns(3)
        with tr_c1:
            ns_trg_student_crews = st.number_input(
                "Student crews (#)",
                min_value=0,
                value=0,
                step=1,
                key="ns_trg_student_crews",
                help="Number of student crews being trained in the month.",
            )
        with tr_c2:
            ns_trg_spcm_student = st.number_input(
                "Sorties per student crew per month",
                min_value=0.0,
                value=0.0,
                step=0.5,
                key="ns_trg_spcm_student",
                help="Monthly sortie requirement per student crew (effective/training credit).",
            )
        with tr_c3:
            ns_trg_refly_rate = st.slider(
                "Refly rate (%)",
                min_value=0.0,
                max_value=50.0,
                value=10.5,
                step=0.25,
                key="ns_trg_refly_rate",
                help="Fraction of sorties that must be re-flown; increases workload.",
            ) / 100.0
    
        ns_trg_ip_sorties_month = st.number_input(
            "Instructor Pilot sorties (monthly total)",
            min_value=0.0,
            value=0.0,
            step=5.0,
            key="ns_trg_ip_sorties_month",
            help="Total IP sorties required in the month (workload, not per crew).",
        )
    
        # -------------------------
        # Training math
        # -------------------------
        # Effective student requirement (credit needed)
        effective_student_month = float(ns_trg_student_crews) * float(ns_trg_spcm_student)
    
        # Refly increases workload: need to fly more sorties to achieve effective credit
        # if refly=0.10 -> only 90% effective -> divide by 0.90
        student_workload_month = effective_student_month / max(1e-6, (1.0 - float(ns_trg_refly_rate)))
    
        # Total executed workload/month = student workload + IP workload
        executed_month = student_workload_month + float(ns_trg_ip_sorties_month)
    
        # Whole sorties
        executed_month_i = int(math.ceil(executed_month))
    
        # Executed per day (whole sorties/day)
        executed_day = executed_month_i / max(1, int(om_days))
        executed_day_i = int(math.ceil(executed_day))
    
        # Scheduled lines/day needed after attrition (whole lines/day)
        scheduled_lines = executed_day_i / max(1e-6, (1.0 - float(attr_rate)))
        scheduled_lines_i = int(math.ceil(scheduled_lines))
    
        # Aircraft required given TF (whole aircraft)
        tf_val = max(1e-6, float(turn_factor))
        aircraft_required = int(math.ceil(scheduled_lines_i / tf_val))
    
        # Build a rim_req-like dict so downstream code keeps working
        # IMPORTANT: In training mode, students are the "base crews" and extra crews = 0
        rim_req = {
            "pai": int(rim_pai),
            "crew_ratio": 0.0,
            "sorties_per_crew_month": 0.0,
            "om_days": int(om_days),
            "turn_factor": float(tf_val),
            "attrition_rate": float(attr_rate),
    
            "base_crews": float(int(ns_trg_student_crews)),
            "extra_crews_total": 0.0,
            "total_crews": float(int(ns_trg_student_crews)),
    
            # Monthly
            "base_net_sorties_month": float(effective_student_month),   # effective credit requirement
            "extra_net_sorties_month": 0.0,
            "net_sorties_month": float(executed_month_i),               # workload requirement (students+IP)
    
            # Daily
            "daily_net_sorties": float(executed_day_i),                 # executed workload/day
            "daily_gross_lines": float(scheduled_lines_i),              # scheduled lines/day after attrition
    
            "aircraft_required": aircraft_required,
    
            # Training audit fields
            "ns_trg_student_crews": int(ns_trg_student_crews),
            "ns_trg_spcm_student": float(ns_trg_spcm_student),
            "ns_trg_refly_rate": float(ns_trg_refly_rate),
            "ns_trg_ip_sorties_month": float(ns_trg_ip_sorties_month),
            "ns_trg_effective_student_month": float(effective_student_month),
            "ns_trg_student_workload_month": float(student_workload_month),
        }
    
    # ==========================
    # Operational modes (crew ratio / overhead pools)
    # ==========================
    else:
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
                            "count": float(int(count)),  # whole crews
                            "spcm": float(spcm_extra),
                        }
                    )
    
        # --- Build inputs for the requirement engine ---
        if demand_mode == "Crew Overhead Only":
            eff_crew_ratio = 0.0
            eff_spcm = 0.0
        else:
            eff_crew_ratio = float(crew_ratio)
            eff_spcm = float(spcm)
    
        rim_req = compute_crew_aircraft_requirement(
            pai=int(rim_pai),
            crew_ratio=eff_crew_ratio,
            sorties_per_crew_month=eff_spcm,
            om_days=int(om_days),
            turn_factor=float(turn_factor),
            attrition_rate=float(attr_rate),
            extra_crews=extra_crews,
        )
    
    # Convenience values (prevents undefined errors downstream)
    executed_day_i = int(math.ceil(float(rim_req.get("daily_net_sorties", 0.0))))
    scheduled_day_i = int(math.ceil(float(rim_req.get("daily_gross_lines", 0.0))))


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

    # --- Crew & Sortie Breakdown (mode-aware) ---
    st.markdown("### Crew & Sortie Breakdown")
    
    bc1, bc2, bc3, bc4 = st.columns(4)
    
    # Pull common values (work for both modes)
    tf_val      = float(rim_req.get("turn_factor", turn_factor))
    attr_val    = float(rim_req.get("attrition_rate", attr_rate))
    om_days_val = int(rim_req.get("om_days", om_days))
    
    # Daily executed + daily scheduled (whole numbers)
    executed_day_i  = int(math.ceil(float(rim_req.get("daily_net_sorties", 0.0))))
    scheduled_day_i = int(math.ceil(float(rim_req.get("daily_gross_lines", 0.0))))
    
    # Monthly totals inferred from daily (whole numbers, consistent)
    executed_month_i  = int(math.ceil(executed_day_i * max(1, om_days_val)))
    scheduled_month_i = int(math.ceil(scheduled_day_i * max(1, om_days_val)))
    
    is_training = (demand_mode == "NS Trg (Training Flight)")
    
    if is_training:
        # ---- Training-specific fields ----
        student_crews = int(rim_req.get("ns_trg_student_crews", 0))
        spcm_student  = float(rim_req.get("ns_trg_spcm_student", 0.0))
        refly_rate    = float(rim_req.get("ns_trg_refly_rate", 0.0))
        ip_month      = float(rim_req.get("ns_trg_ip_sorties_month", 0.0))
    
        # Effective training requirement (progress credit) and workload after refly
        effective_need_month = student_crews * spcm_student
        workload_student_month = (
            effective_need_month / max(1e-6, (1.0 - refly_rate))
            if effective_need_month > 0 else 0.0
        )
    
        # Total executed workload (students workload + IP)
        executed_month_calc = workload_student_month + ip_month
    
        # Whole-number display
        effective_need_month_i    = int(math.ceil(effective_need_month))
        workload_student_month_i  = int(math.ceil(workload_student_month))
        ip_month_i                = int(math.ceil(ip_month))
        executed_month_calc_i     = int(math.ceil(executed_month_calc))
    
        with bc1:
            st.markdown(
                f"""
                **Crews (Training)**  
                • **Student crews:** {student_crews}  
                • Base crews (ratio × PAI): 0  
                • Overhead pools: 0  
                """
            )
    
        with bc2:
            st.markdown(
                f"""
                **Sorties – Monthly (Training)**  
                • Effective student requirement: **{effective_need_month_i}**  
                • Student workload (after refly): **{workload_student_month_i}**  
                • IP sorties (monthly): **{ip_month_i}**  
                • **Total executed workload:** **{executed_month_calc_i}**  
                """
            )
    
        with bc3:
            st.markdown(
                f"""
                **Sorties – Daily (Training)**  
                • **Executed workload/day:** **{executed_day_i}**  
                • **Scheduled lines/day (after attrition):** **{scheduled_day_i}**  
                """
            )
    
        with bc4:
            st.markdown(
                f"""
                **Assumptions (Training)**  
                • Turn Factor (TF): {tf_val:.2f} sorties/jet/day  
                • Attrition used: {attr_val:.0%}  
                • Refly rate: {refly_rate:.0%}  
                • O&M days in month: {om_days_val}  
    
                _Refly increases workload: workload = effective ÷ (1 − refly)._  
                _Attrition increases scheduled lines: scheduled = executed ÷ (1 − attrition)._  
                """
            )
    
    else:
        # ---- Operational (existing logic) ----
        base_crews        = float(rim_req.get("base_crews",  crew_ratio * rim_pai))
        extra_crews_total = float(rim_req.get("extra_crews_total", 0.0))
        total_crews       = float(rim_req.get("total_crews", base_crews + extra_crews_total))
    
        base_month  = float(rim_req.get("base_net_sorties_month", 0.0))
        extra_month = float(rim_req.get("extra_net_sorties_month", 0.0))
        total_month = float(rim_req.get("net_sorties_month", base_month + extra_month))
    
        # Whole numbers
        base_crews_i  = int(math.ceil(base_crews))
        extra_crews_i = int(math.ceil(extra_crews_total))
        total_crews_i = int(math.ceil(total_crews))
    
        base_month_i  = int(math.ceil(base_month))
        extra_month_i = int(math.ceil(extra_month))
        total_month_i = int(math.ceil(total_month))
    
        with bc1:
            st.markdown(
                f"""
                **Crews (Operational)**  
                • Base crews (ratio × PAI): {base_crews_i}  
                • Extra crews (overhead pools): {extra_crews_i}  
                • **Total crews:** {total_crews_i}
                """
            )
    
        with bc2:
            st.markdown(
                f"""
                **Sorties – Monthly (Operational)**  
                • Base crews: {base_month_i}  
                • Extra crews: {extra_month_i}  
                • **Total (executed requirement):** {total_month_i}
                """
            )
    
        with bc3:
            st.markdown(
                f"""
                **Sorties – Daily (Operational)**  
                • **Executed requirement/day:** {executed_day_i}  
                • **Scheduled lines/day (after attrition):** {scheduled_day_i}  
                """
            )
    
        with bc4:
            st.markdown(
                f"""
                **Assumptions (Operational)**  
                • Turn Factor (TF): {tf_val:.2f} sorties/jet/day  
                • Attrition used: {attr_val:.0%}  
                • O&M days in month: {om_days_val}  
    
                _Executed requirement is driven by crews (ratio/SPCM + overhead pools)._  
                _Attrition increases the **scheduled lines** needed to achieve that executed requirement._  
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
    # 4) North Star – Reverse-Engineer Annual FHP
    # ==========================
    st.markdown("---")
    st.subheader("🎯 North Star – Reverse Engineer Annual FHP Target")

    ns_c1, ns_c2, ns_c3 = st.columns(3)
    with ns_c1:
        target_hours = st.number_input(
            "Annual FHP Target (flying hours)",
            min_value=0.0,
            value=100000.0,
            step=1000.0,
            key="rim_target_hours",
            help="Total flying hours you need to produce in the year."
        )
    with ns_c2:
        asd_ns = st.number_input(
            "Average Sortie Duration (hrs)",
            min_value=0.1,
            value=2.0,
            step=0.1,
            key="rim_target_asd",
            help="Planned average sortie duration for the FHP."
        )
    with ns_c3:
        om_days_ns = st.number_input(
            "O&M Days per Year (North Star)",
            min_value=1,
            value=260,
            step=1,
            key="rim_target_om_days",
            help="Training days per year (e.g., 5 days/week × 52 weeks ≈ 260)."
        )

    # Only compute if there is a non-zero target
    if target_hours > 0 and asd_ns > 0 and om_days_ns > 0 and turn_factor > 0:
        # 1) Total sorties required
        total_sorties_req = target_hours / asd_ns

        # 2) Daily sorties required
        daily_sorties_req = total_sorties_req / om_days_ns

        # 3) Daily gross sorties (undo attrition)
        safe_attr = min(max(attr_rate, 0.0), 0.95)  # clamp 0–95% just for safety
        daily_gross_sorties = daily_sorties_req / (1.0 - safe_attr)

        # 4) Aircraft required (North Star)
        ac_required_ns = math.ceil(daily_gross_sorties / turn_factor)

        st.markdown("### North Star FHP Requirement Summary")

        col_ns1, col_ns2 = st.columns(2)
        with col_ns1:
            st.markdown(
                f"""
                **Targets & Demand**  
                • Target hours: **{target_hours:,.0f} hrs**  
                • ASD: **{asd_ns:.2f} hr/sortie**  
                • Total sorties required: **{total_sorties_req:,.0f}**  
                • O&M days: **{om_days_ns} days**  
                • Daily sortie requirement: **{daily_sorties_req:,.1f} /day**
                """
            )
        with col_ns2:
            margin_ns = avail_eff - ac_required_ns
            st.markdown(
                f"""
                **Capacity & Aircraft Requirement**  
                • Attrition (used): **{safe_attr*100:.0f}%**  
                • Daily gross sorties (post-attrition): **{daily_gross_sorties:,.1f} /day**  
                • Turn Factor (TF): **{turn_factor:.2f} sorties/jet/day**  
                • **Aircraft required (North Star): {ac_required_ns} tails**  
                • Avail FHP (+NMC Flyable): **{avail_eff} tails**  
                • Margin vs availability: **{margin_ns:+.0f} tails**
                """
            )

        # --- Math transparency: plain-language formulas + numbers ---
        with st.expander("Show North Star math (how this was calculated)", expanded=False):
            st.markdown(
                f"""
                **1. Total sorties required**

                Formula:  
                `Total sorties required = Target hours ÷ ASD`

                With your inputs:  
                `Total sorties required = {target_hours:,.0f} ÷ {asd_ns:.2f} = {total_sorties_req:,.1f}`


                **2. Daily sorties required**

                Formula:  
                `Daily sorties required = Total sorties required ÷ O&M days`

                With your inputs:  
                `Daily sorties required = {total_sorties_req:,.1f} ÷ {om_days_ns} = {daily_sorties_req:,.2f}`


                **3. Daily gross sorties (undo attrition)**

                Formula:  
                `Daily gross sorties = Daily sorties required ÷ (1 − attrition)`

                With your inputs:  
                `Daily gross sorties = {daily_sorties_req:,.2f} ÷ (1 − {safe_attr:.2f}) = {daily_gross_sorties:,.2f}`


                **4. Aircraft required (North Star)**

                Formula:  
                `Aircraft required = ceil(Daily gross sorties ÷ Turn Factor)`

                With your inputs:  
                `Aircraft required = ceil({daily_gross_sorties:,.2f} ÷ {turn_factor:.2f}) = {ac_required_ns}`
                """
            )

        # Optional: share in session_state for future modules/dashboard cards
        st.session_state["northstar_fhp_summary"] = {
            "target_hours": float(target_hours),
            "asd": float(asd_ns),
            "om_days": int(om_days_ns),
            "attrition": float(safe_attr),
            "turn_factor": float(turn_factor),
            "total_sorties": float(total_sorties_req),
            "daily_sorties": float(daily_sorties_req),
            "daily_gross": float(daily_gross_sorties),
            "ac_required_ns": int(ac_required_ns),
            "avail_fhp": int(avail_eff),
            "margin_vs_avail": float(margin_ns),
        }

    else:
        st.info(
            "Set a non-zero target hours, ASD, sensible O&M days, Turn Factor, and "
            "commit cap to see the North Star reverse-engineering."
        )

    # ==========================
    # 5) 30-Day RIM Projection (What-if)
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

# ─── SPECTRE Introduction ───────────────────────────────────
elif sim_choice == "SPECTRE Flight Manual":
    st.header("📘 SPECTRE — How-To Guide")

    st.markdown("""
    **SPECTRE** helps you translate **fleet structure + maintenance reality + crew demand**
    into **what you can actually schedule** (Weekly) and **what you must have to meet requirements** (RIM / North Star).
    """)

    with st.expander("3) Key definitions ", expanded=False):
        st.markdown("""
        - **TAI**: Total Aircraft Inventory. Total Tails Assigned
        - **PAI**: Primary Assigned Inventory. Tails resourced for crews/flying hours (your “programmed” fleet)
        - **BAI**: Tails that preserve the program when Depot/UPNR hits (TAI − PAI)
        - **Depot/UPNR**: non-possessed / not available to schedule
        - **EP**: effective possessed after NMC bins (what’s truly available)
        - **NMC Flyable**: coded NMC but can still be scheduled (headroom, not EP)
        - **Turn Factor (TF)**: sorties per jet per fly day (how hard you’re turning)
        - **Attrition**: sorties that won’t happen (weather/abort/cancel), driving extra scheduled lines
        """)

    with st.expander(" MC vs AA (How SPECTRE Uses Them)", expanded=False):
        st.markdown("""
    ### Definitions
    - **MC Rate** = **MC Hours / Possessed Hours**  
      *What I did with the equipment I possessed.*  
      This reflects **maintenance performance** on aircraft that were physically present.
    
    - **AA Rate** = **MC Hours / TAI Hours**  
      *What I did with the equipment I own (assigned).*  
      This reflects **structural availability** (sensitive to Depot/UPNR, deployments, off-station time).
    
    ### Annual First Look Modeling Rule
    SPECTRE separates **Structure** from **Performance**:
    
    1) **Structure (Planning bins remove tails at home):**  
       TAI_home = TAI − Degraders − Depot/UPNR − Deployments − TDY − Scheduled Mx − Unscheduled Mx reserve
    
    2) **Performance (MC applies to what remains at home):**  
       Flyable_home ≈ TAI_home × MC_rate
    
    ### Why AA is still shown
    AA is reported for **context and validation**:
    - If simulated availability is far below historical AA, you likely over-binned structure.
    - If MC is healthy but AA is poor, the limiting factor is usually **possession/structure**, not maintenance performance.
    """)

    with st.expander("1) What each mode does", expanded=False):
        st.markdown("""
        **Weekly Simulation**
        - Simulates week-by-week flying execution using break/fix/attrition assumptions.
        - Outputs distributions (risk), availability bands, and “can we execute this schedule?”

        **RIM / North Star Analysis (beta)**
        - Converts crew demand and FHP targets into aircraft required.
        - Applies structure: Depot/UPNR, NMC bins, deployed, non-crew reservations (spares/trainers/alert/etc.).
        - Produces margins and turn-pattern recommendations.
        """)

    with st.expander("2) 90-second workflow", expanded=False):
        st.markdown("""
        **Step A — Weekly (execution reality)**
        1. Enter TAI + Overhead (or your normal EP drivers)
        2. Set break rate + fix rates + ground abort + weather + sortie attrition
        3. Build the week’s turn patterns (e.g., 10x8 M–Th, 8x0 Fri)
        4. Run → review availability and risk

        **Step B — RIM / North Star (requirement reality)**
        1. Enter structure: TAI / PAI / Depot / UPNR
        2. Enter operational requirements: spares, trainers, fleet mgmt, contingency, alert
        3. Enter NMC bins and NMC-flyable headroom
        4. Enter crews + SPCM, confirm Turn Factor (TF), set attrition
        5. Read margins + pattern recommendations + reverse FHP math
        """)

    with st.expander("4) Interpreting the output", expanded=False):
        st.markdown("""
        - **Crew requirement (executed)**: the sorties that must happen to satisfy crews/FHP
        - **Scheduled lines (after attrition)**: what you must schedule to achieve that executed requirement
        - **Margins**: how much aircraft capacity you have left (or how far short you are)
        - **Recommendations**: what TF / commit rate / pattern shape is implied to close the gap
        """)

    with st.expander("📐 Formulas (North Star / RIM)", expanded=False):
    
        show_example = st.toggle("Show example (numbers applied)", value=False, key="ns_formula_example")
    
        st.markdown("""
        ### Definitions (inputs)
        - **PAI** = Primary Aircraft Inventory (tails resourced for crews/FHP)
        - **CR** = Crew Ratio (crews per PAI)
        - **SPCM** = Sorties per Crew per Month
        - **OM** = O&M Days in Month (fly days)
        - **TF** = Turn Factor (sorties per jet per fly day)
        - **α** = Attrition rate (0–1). Example: 20% → α = 0.20
        - **Extra crews** = additional crew pools (CMR/BMC Wingmen/FL, etc.)
    
        **Training (NS Trg) additions**
        - **Student crews** = number of student crews in training
        - **SPCM_student** = sorties per student crew per month
        - **Refly** = fraction of student sorties that must be repeated (0–1)
        - **IP sorties** = instructor pilot sorties required per month (monthly total)
    
        ---
    
        ## 1) Crew-only North Star (base crews via crew ratio)
    
        **Step 1 — Base crews**
        - **Crews_base = ceil(CR × PAI)**
    
        **Step 2 — Monthly executed sorties requirement**
        - **S_exec_month = Crews_base × SPCM**
    
        **Step 3 — Daily executed requirement (what must actually happen)**
        - **S_exec_day = ceil(S_exec_month ÷ OM)**
    
        **Step 4 — Daily scheduled lines required (undo attrition)**
        - **S_sched_day = ceil(S_exec_day ÷ (1 − α))**
    
        **Step 5 — Aircraft required for the schedule**
        - **A_req = ceil(S_sched_day ÷ TF)**
    
        > Where TF, Attrition, and O&M sit:
        > - **OM** divides monthly sorties into daily sorties
        > - **α** increases scheduled lines above executed requirement
        > - **TF** converts scheduled lines into aircraft required
    
        ---
    
        ## 2) Crew + Overhead (base crew ratio + extra pools)
    
        **Step 1 — Extra crews monthly sorties**
        For each overhead pool *i*:
        - **S_i = ceil(Crews_i × SPCM_i)**
    
        Then:
        - **S_extra_month = Σ S_i**
    
        **Step 2 — Total executed monthly sorties**
        - **S_exec_month = (Crews_base × SPCM) + S_extra_month**
    
        Then same as above:
        - **S_exec_day = ceil(S_exec_month ÷ OM)**
        - **S_sched_day = ceil(S_exec_day ÷ (1 − α))**
        - **A_req = ceil(S_sched_day ÷ TF)**
    
        ---
    
        ## 3) Overhead-only mode (ignore crew ratio, use overhead pools only)
    
        **Executed monthly sorties**
        - **S_exec_month = Σ ceil(Crews_i × SPCM_i)**
    
        Then:
        - **S_exec_day = ceil(S_exec_month ÷ OM)**
        - **S_sched_day = ceil(S_exec_day ÷ (1 − α))**
        - **A_req = ceil(S_sched_day ÷ TF)**
    
        ---
    
        ## 4) NS Trg (Training Flight) — student production + IP sorties + refly
    
        **Step 1 — Student effective requirement (monthly)**
        - **S_student_effective = Student_crews × SPCM_student**
    
        **Step 2 — Refly inflates workload (monthly)**
        - **S_student_gross = S_student_effective ÷ (1 − Refly)**
    
        **Step 3 — Add instructor workload**
        - **S_exec_month = S_student_gross + IP_sorties_month**
    
        Then same daily/attrition/TF pipeline:
        - **S_exec_day = ceil(S_exec_month ÷ OM)**
        - **S_sched_day = ceil(S_exec_day ÷ (1 − α))**
        - **A_req = ceil(S_sched_day ÷ TF)**
    
        Notes:
        - **Refly is not attrition.** Refly creates additional workload to achieve the *effective* training requirement.
        - **Attrition** still inflates scheduled lines above executed requirement.
    
        ---
    
        ## 5) Operational Requirements (non-crew) effects on availability (structure side)
    
        These are **reserved tails** (they reduce FHP availability, not crew demand):
        - **Op_total = Spares + Trainers + FleetMgmt + Contingency + Alert**
        - **Op_reserved = min(EP_home, Op_total)**
        - **Avail_FHP = max(EP_home − Op_reserved, 0)**
        - **Avail_eff = Avail_FHP + NMC_flyable**
    
        ---
    
        ## 6) Turn Pattern implication (2-Go baseline)
    
        Given TF and an AM go count:
        - **PM_turns ≈ floor(AM × (TF − 1))**
        - **Total sorties/day ≈ AM + PM**
    
        This is why raising TF can reduce AM demand but can increase PM burden.
    
        ---
        """)

    if show_example:
        st.markdown("""
        ### Example (numbers applied)

        **Operational (Crew-only)**
        - PAI=16, CR=1.2 ⇒ Crews_base=ceil(1.2×16)=ceil(19.2)=20  
        - SPCM=4 ⇒ S_exec_month=20×4=80  
        - OM=20 ⇒ S_exec_day=ceil(80/20)=4  
        - α=0.20 ⇒ S_sched_day=ceil(4/0.8)=ceil(5)=5  
        - TF=1.5 ⇒ A_req=ceil(5/1.5)=ceil(3.33)=4  

        **Training (NS Trg)**
        - Student_crews=12, SPCM_student=6 ⇒ S_student_effective=72  
        - Refly=10% ⇒ S_student_gross=72/0.9=80  
        - IP_sorties_month=20 ⇒ S_exec_month=100  
        - OM=20 ⇒ S_exec_day=ceil(100/20)=5  
        - α=0.20 ⇒ S_sched_day=ceil(5/0.8)=ceil(6.25)=7  
        - TF=1.5 ⇒ A_req=ceil(7/1.5)=ceil(4.67)=5
        """)


    # ---- Example Toggle -------------------------------------------------
    show_example = st.checkbox("Show worked example (formulas + numbers)", value=False, key="howto_show_example")
    
    if show_example:
        st.markdown("### Worked Example")
    
        # Pull current values if present; otherwise use defaults
        # (These keys are from our RIM tab inputs; adjust if your keys differ.)
        pai = int(st.session_state.get("rim_pai", st.session_state.get("rim_pai_beta", 16)))
        cr = float(st.session_state.get("rim_crew_ratio_beta", 1.2))
        spcm = float(st.session_state.get("rim_spcm_beta", 4.0))
        om = int(st.session_state.get("rim_om_days_beta", 20))
        tf = float(st.session_state.get("weekly_turn_factor", st.session_state.get("rim_tf_manual_beta", 1.5)))
    
        # Attrition slider is stored as percent in our UI; convert to 0..1
        attr_pct = float(st.session_state.get("rim_attr_rate_beta", 20.0))
        alpha = max(0.0, min(0.95, attr_pct / 100.0))
    
        # Optional overhead pools (if user already set them); otherwise example 0
        # If you store overhead pools in a list, you can replace this with that logic.
        # We'll try common keys we used for the Crew Overhead expander:
        def get_int(key: str, default: int = 0) -> int:
            try:
                return int(st.session_state.get(key, default))
            except Exception:
                return default
    
        def get_float(key: str, default: float = 0.0) -> float:
            try:
                return float(st.session_state.get(key, default))
            except Exception:
                return default
    
        overhead_groups = [
            ("CMR Wingmen",      get_int("rim_cmr_wg_count", 0), get_float("rim_cmr_wg_spcm", 0.0)),
            ("CMR Flight Leads", get_int("rim_cmr_fl_count", 0), get_float("rim_cmr_fl_spcm", 0.0)),
            ("BMC Wingmen",      get_int("rim_bmc_wg_count", 0), get_float("rim_bmc_wg_spcm", 0.0)),
            ("BMC Flight Leads", get_int("rim_bmc_fl_count", 0), get_float("rim_bmc_fl_spcm", 0.0)),
        ]
    
        # --- Crew-only math ---
        crews_base = math.ceil(cr * pai)
        s_exec_month = crews_base * spcm
        s_exec_day = math.ceil(s_exec_month / max(1, om))
        s_sched_day = math.ceil(s_exec_day / max(1e-6, (1.0 - alpha)))
        a_req = math.ceil(s_sched_day / max(1e-6, tf))
    
        st.markdown(
            f"""
    **Inputs used**
    - PAI = **{pai}**
    - Crew Ratio (CR) = **{cr:.2f}**
    - SPCM = **{spcm:.2f}**
    - OM days = **{om}**
    - TF = **{tf:.2f}**
    - Attrition α = **{alpha:.0%}**
    
    ---
    
    ### A) Crew-only (base crews via ratio)
    
    **1) Base crews**
    - Crews_base = ceil(CR × PAI)  
    - Crews_base = ceil({cr:.2f} × {pai}) = ceil({cr*pai:.2f}) = **{crews_base}**
    
    **2) Monthly executed sorties**
    - S_exec_month = Crews_base × SPCM  
    - S_exec_month = {crews_base} × {spcm:.2f} = **{s_exec_month:.0f}**
    
    **3) Daily executed requirement**
    - S_exec_day = ceil(S_exec_month ÷ OM)  
    - S_exec_day = ceil({s_exec_month:.0f} ÷ {om}) = ceil({s_exec_month/om:.2f}) = **{s_exec_day}**
    
    **4) Daily scheduled lines required (undo attrition)**
    - S_sched_day = ceil(S_exec_day ÷ (1 − α))  
    - S_sched_day = ceil({s_exec_day} ÷ (1 − {alpha:.2f})) = ceil({s_exec_day/(1-alpha):.2f}) = **{s_sched_day}**
    
    **5) Aircraft required**
    - A_req = ceil(S_sched_day ÷ TF)  
    - A_req = ceil({s_sched_day} ÷ {tf:.2f}) = ceil({s_sched_day/tf:.2f}) = **{a_req}**
    """
        )
    
        # --- Crew + Overhead math ---
        s_extra_month = 0
        used_groups = []
        for name, cnt, grp_spcm in overhead_groups:
            if cnt > 0 and grp_spcm > 0:
                s_i = math.ceil(cnt * grp_spcm)
                s_extra_month += s_i
                used_groups.append((name, cnt, grp_spcm, s_i))
    
        if used_groups:
            s_exec_month_total = s_exec_month + s_extra_month
            s_exec_day_total = math.ceil(s_exec_month_total / max(1, om))
            s_sched_day_total = math.ceil(s_exec_day_total / max(1e-6, (1.0 - alpha)))
            a_req_total = math.ceil(s_sched_day_total / max(1e-6, tf))
    
            lines = []
            for (name, cnt, grp_spcm, s_i) in used_groups:
                lines.append(f"- {name}: ceil({cnt} × {grp_spcm:.2f}) = **{s_i}** sorties/month")
    
            st.markdown(
                f"""
    ---
    
    ### B) Crew + Overhead (ratio crews + extra pools)
    
    **Overhead pools included**
    {chr(10).join(lines)}
    
    **Extra monthly executed sorties**
    - S_extra_month = Σ ceil(Crews_i × SPCM_i) = **{s_extra_month}**
    
    **Total executed monthly sorties**
    - S_exec_month_total = S_exec_month + S_extra_month  
    - = {s_exec_month:.0f} + {s_extra_month} = **{s_exec_month_total:.0f}**
    
    **Daily executed**
    - S_exec_day_total = ceil({s_exec_month_total:.0f} ÷ {om}) = **{s_exec_day_total}**
    
    **Daily scheduled (after attrition)**
    - S_sched_day_total = ceil({s_exec_day_total} ÷ (1 − {alpha:.2f})) = **{s_sched_day_total}**
    
    **Aircraft required**
    - A_req_total = ceil({s_sched_day_total} ÷ {tf:.2f}) = **{a_req_total}**
    """
            )
        else:
            st.info("No overhead pools are currently set (>0 crews and >0 SPCM). Add values in Crew Overhead to see Crew+Overhead and Overhead-only examples.")
    
        # --- Overhead-only math (if overhead groups exist) ---
        if used_groups:
            s_exec_month_oh = s_extra_month
            s_exec_day_oh = math.ceil(s_exec_month_oh / max(1, om))
            s_sched_day_oh = math.ceil(s_exec_day_oh / max(1e-6, (1.0 - alpha)))
            a_req_oh = math.ceil(s_sched_day_oh / max(1e-6, tf))
    
            st.markdown(
                f"""
    ---
    
    ### C) Overhead-only (ignore crew ratio; overhead pools only)
    
    **Executed monthly sorties**
    - S_exec_month = **{s_exec_month_oh}**
    
    **Daily executed**
    - S_exec_day = ceil({s_exec_month_oh} ÷ {om}) = **{s_exec_day_oh}**
    
    **Daily scheduled (after attrition)**
    - S_sched_day = ceil({s_exec_day_oh} ÷ (1 − {alpha:.2f})) = **{s_sched_day_oh}**
    
    **Aircraft required**
    - A_req = ceil({s_sched_day_oh} ÷ {tf:.2f}) = **{a_req_oh}**
    """
            )

    st.info("Tip: Use Weekly to understand execution risk; use RIM/North Star to explain *why* you can’t meet everything at once.")
