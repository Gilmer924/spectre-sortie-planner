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


#------ RIM Toggle --------
st.sidebar.markdown("### Experimental Modules")
enable_rim_ns = st.sidebar.checkbox("Enable RIM / North Star (beta)", value=False)

# ---------- Simulation Selector ----------
sim_options = [
    "Weekly Simulation",
    "Personnel Simulation",
    "Quick Probability Analysis",
    "Annual Historical Simulation"
]
sim_choice = st.sidebar.selectbox("Choose Simulation Module", sim_options)

# ---------- WEEKLY SIMULATION ----------
if sim_choice == "Weekly Simulation":
    st.header("🗓️ Weekly Sortie Simulation")
    # --- Sidebar Inputs ---
    TAI       = st.sidebar.number_input("Total AC Inventory", value=12)
    Overhead  = st.sidebar.number_input("Overhead AC", value=2)

    # ── RIM / NS structure (does NOT change sim_engine yet) ───────────────
    if enable_rim_ns:
        st.sidebar.markdown("**RIM Structure (Experimental)**")

        PAI = st.sidebar.number_input(
            "PAI (Primary Aircraft Inventory)",
            value=TAI,
            step=1,
            help="Primary aircraft normally used for training/ops."
        )
        BAI = max(TAI - PAI, 0)
        st.sidebar.caption(f"BAI (Backup Aircraft Inventory) assumed: {BAI} = TAI − PAI")

        st.sidebar.markdown("**Degrader Bins (Overhead)**")
        rim_nmcm = st.sidebar.number_input("NMCM (NMC Maintenance)", value=0, step=1)
        rim_nmcs = st.sidebar.number_input("NMCS (NMC Supply)", value=0, step=1)
        rim_nmcb = st.sidebar.number_input("NMCB (Both)", value=0, step=1)
        rim_nmc_fly = st.sidebar.number_input(
            "NMC Flyable tails",
            value=0,
            step=1,
            help="Flyable but still coded NMC (headroom for scheduling, not EP)."
        )
        rim_depot = st.sidebar.number_input("Depot / Programmed (PDM, etc.)", value=0, step=1)
        rim_upnr  = st.sidebar.number_input("UPNR", value=0, step=1)

        # Store this richer structure for RIM/NS math & future visualizations
        st.session_state["rim_structure_weekly"] = {
            "TAI": int(TAI),
            "PAI": int(PAI),
            "BAI": int(BAI),
            "NMCM": int(rim_nmcm),
            "NMCS": int(rim_nmcs),
            "NMCB": int(rim_nmcb),
            "NMC_flyable": int(rim_nmc_fly),
            "Depot": int(rim_depot),
            "UPNR": int(rim_upnr),
        }

    ASD       = st.sidebar.number_input("Avg Sortie Duration (hrs)", value=1.9)
    Break_pct = st.sidebar.slider("Break Rate (%)", 0.0, 100.0, 16.6)
    GA_pct    = st.sidebar.slider("Ground Abort Rate (%)", 0.0, 100.0, 7.5)
    Fix8      = st.sidebar.slider("Fix Rate 8 Hr (%)", 0.0, 100.0, 25.0)
    Fix12     = st.sidebar.slider("Fix Rate 12 Hr (%)", 0.0, 100.0, 35.0)
    Fix24     = st.sidebar.slider("Fix Rate 24 Hr (%)", 0.0, 100.0, 56.0)
    WxAttr    = st.sidebar.slider("Weather Attrition (%)", 0.0, 100.0, 2.5)
    SortieAttr= st.sidebar.slider("Sortie Attrition (%)", 0.0, 100.0, 0.0)
    Trials    = st.sidebar.slider("MC Trials", 100, 2000, 500)
    commit_thresh = st.sidebar.slider(
        "Commitment Rate Threshold (%)",
        min_value=0.0,
        max_value=100.0,
        value=65.0,
        help="Warn when daily commit rate exceeds this % of available aircraft"
    )
    # --- Turn Pattern Input ---
    num_weeks = st.number_input("Number of Weeks", min_value=1, max_value=12, value=1, step=1)
    all_weeks_patterns = []
    weekend_flags = []
    days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    for w in range(num_weeks):
        st.subheader(f"Week {w+1}")
        cols = st.columns(len(days))
        pattern = []
        for i, d in enumerate(days):
            default = "8x6" if i<5 else "0x0"
            pattern.append(cols[i].text_input(f"W{w+1}_{d}", value=default))
        all_weeks_patterns.append(pattern)
        weekend_flags.append(st.checkbox(f"Weekend duty W{w+1}", key=f"wknd_{w}"))

    selected_week = st.selectbox("View Results for Week", list(range(1, num_weeks+1)))
    if st.button("Run Weekly Simulation"):
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
            "weekend_duty": weekend_flags
        }
        st.session_state.weekly_data = (run_weekly_simulation(params, Trials), selected_week)

    if "weekly_data" in st.session_state:
        trials_data, sel_w = st.session_state["weekly_data"]
        week_results = [trial[sel_w-1] for trial in trials_data]
        df = pd.DataFrame(week_results)
#       st.write("Columns:", df.columns.tolist())
        days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

        # ── Compute daily scheduled sorties from patterns ───────────────
        daily_sched_matrix = np.array([
            [sum(map(int, re.findall(r"\d+", pat)))
             for pat in all_weeks_patterns[sel_w-1]]
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
              for sched, loss in zip(sched_row, loss_row)]
            for sched_row, loss_row in zip(daily_sched_matrix, daily_losses)
        ], axis=0)

        # 3) Availability
        avg_start = np.stack(df["daily_available_start"].values).mean(axis=0)
        avg_end   = np.stack(df["daily_available_end"].values).mean(axis=0)

        # ── Compute avg start/end availability ──────────────────────────
        # avg_start = np.array([r["daily_available_start"] for r in week_results]).mean(axis=0)
        # avg_end   = np.array([r["daily_available_end"]   for r in week_results]).mean(axis=0)
        
        # ── Enhanced Over-commitment Alerts ──────────────────────────
        # use the average end-of-day availability as your capacity
        # avg_sched = np.array([r["daily_schedule"]         for r in week_results]).mean(axis=0)
        avg_avail = avg_end
        
        # parse “Go” counts from your turn-pattern strings
        # go_counts = [
        #     len(re.findall(r"\d+x\d+", pat)) or 1 
        #     for pat in all_weeks_patterns[sel_w-1]
        # ]
        
        with st.expander("⚠️ Alerts & Warnings (click to collapse)", expanded=True):
            for idx, day in enumerate(days):
                # 1) Parse the “first‐Go” sorties from your pattern, e.g. “10x8” → 10
                pat = all_weeks_patterns[sel_w-1][idx]
                m = re.match(r"(\d+)", pat)
                first_go = int(m.group(1)) if m else 0
        
                # 2) Use start-of-day availability, not end-of-day
                avail = avg_start[idx]
        
                # 3) Commit rate (%)
                commit_pct = (first_go / avail * 100) if avail > 0 else 0
        
                # 4) Fire warning only if that exceeds your threshold
                if first_go > 0 and commit_pct > commit_thresh:
                    st.warning(
                        f"High commitment on {day}: "
                        f"{first_go} sorties ÷ {avail:.1f} AC → {commit_pct:.0f}% "
                        f"(threshold={commit_thresh:.0f}%)"
                    )

        # ── RISK METRICS & SUGGESTIONS (per-sortie loss rate) ───────────
        total_flown = sum(r["flown"] for r in week_results)
        total_mnd   = sum(r["mnd"] for r in week_results)
        total_ga    = sum(r["ground_aborts"] for r in week_results)

        if total_flown > 0:
            mnd_risk   = (total_mnd / total_flown) * 100.0
            abort_risk = (total_ga / total_flown) * 100.0
        else:
            mnd_risk = abort_risk = 0.0

        risk_msg = f"MND-Risk: {mnd_risk:.1f}% | Abort-Risk: {abort_risk:.1f}%"

        if mnd_risk > 10 or abort_risk > 10:
            st.error(risk_msg + " → HIGH risk of failure!")
        elif mnd_risk > 5 or abort_risk > 5:
            st.warning(risk_msg + " → Medium risk; consider adjustments.")
        else:
            st.success(risk_msg + " → Low risk 👍")


        # ── Turn-Factor Calculation ────────────────────────────────────
        # Grab the raw pattern strings for the selected week
        raw_patterns = all_weeks_patterns[sel_w-1]

        # 1) Parse each “NxM” token into ints
        go_lists = [list(map(int, re.findall(r"(\d+)", pat))) for pat in raw_patterns]
        
        # 2) Extract first and second Go values (0 if missing)
        first_go_vals  = [gl[0] if len(gl) >= 1 else 0 for gl in go_lists]
        second_go_vals = [gl[1] if len(gl) >= 2 else 0 for gl in go_lists]

        # 3) Compute means and round up
        mean_first  = np.mean(first_go_vals)
        mean_second = np.mean(second_go_vals)
        avg_first   = math.ceil(mean_first)
        avg_second  = math.ceil(mean_second)

        # 4) Turn Factor = (First + Second) / First
        if avg_first > 0:
            turn_factor = (avg_first + avg_second) / avg_first
        else:
            turn_factor = float("nan")

        # 5) Display
        st.info(
            f"🔄 Turn Factor: "
            f"({avg_first} + {avg_second}) / {avg_first} = {turn_factor:.2f}  \n"
            f"→ Rounded averages: First Go={avg_first}, Second Go={avg_second}"
        )

        # --- RIM / North Star (Experimental) — Crew-Based Requirement ---
        if enable_rim_ns:
            from simulations.rim_requirements import compute_crew_aircraft_requirement
        
            st.subheader("RIM / North Star (Experimental) — Crew Requirement")
        
            st.caption(
                "This calculation is read-only and does not change the weekly sim. "
                "It shows how many aircraft are required from a crew perspective, "
                "given your current Turn Factor and a monthly crew requirement."
            )
        
            # Inputs specific to RIM requirement
            rim_col1, rim_col2, rim_col3 = st.columns(3)
            with rim_col1:
                crew_ratio = st.number_input(
                    "Crew Ratio (crews per aircraft)",
                    min_value=0.0,
                    value=1.2,
                    step=0.1,
                    key="rim_crew_ratio",
                    help="Example: 1.2 means 1.2 crews per PAI."
                )
            with rim_col2:
                spcm = st.number_input(
                    "Sorties per Crew per Month (SPCM)",
                    min_value=0.0,
                    value=4.0,
                    step=0.5,
                    key="rim_spcm",
                    help="How many sorties each crew must fly per month."
                )
            with rim_col3:
                om_days_rim = st.number_input(
                    "O&M Days in Planning Month (for RIM)",
                    min_value=1,
                    value=20,
                    step=1,
                    key="rim_om_days",
                    help="Use your monthly fly days here (not just this week)."
                )
        
            # Use your 'Sortie Attrition (%)' as the attrition input for now
            attrition_rate_rim = SortieAttr / 100.0
        
            rim_req = compute_crew_aircraft_requirement(
                pai=int(TAI),  # later we can refine PAI vs BAI/Overhead
                crew_ratio=float(crew_ratio),
                sorties_per_crew_month=float(spcm),
                om_days=int(om_days_rim),
                turn_factor=float(turn_factor),
                attrition_rate=float(attrition_rate_rim),
            )
        
            st.markdown(
                f"**Crew-based aircraft required:** **{rim_req['aircraft_required']}**  \n"
                f"- Net sorties per month (crews): {rim_req['net_sorties_month']:.1f}  \n"
                f"- Daily gross lines (after attrition): {rim_req['daily_gross_lines']:.1f}  \n"
                f"- Turn Factor used: {rim_req['turn_factor']:.2f} sorties/jet/day  \n"
                f"- Attrition used: {rim_req['attrition_rate']:.0%}"
            )
            
            # --- Compare crew-based requirement to simulated availability ---
            st.subheader("RIM vs Simulated Weekly Capacity")
            
            try:
                # avg_start / avg_end are already computed above for the selected week
                avg_start_cap = float(np.mean(avg_start))   # start-of-day capacity
                avg_end_cap   = float(np.mean(avg_end))     # end-of-day capacity
            
                rim_ac = rim_req["aircraft_required"]
                delta_start = avg_start_cap - rim_ac
                delta_end   = avg_end_cap - rim_ac
            
                st.markdown(
                    f"""
                    **Average Start-of-Day Capacity:** {avg_start_cap:.1f} aircraft  
                    **Average End-of-Day Capacity:** {avg_end_cap:.1f} aircraft  
                    **Crew-Based Requirement (RIM):** {rim_ac} aircraft  
            
                    • Start-of-day margin: **{delta_start:+.1f}** aircraft  
                    • End-of-day margin: **{delta_end:+.1f}** aircraft  
                    """
                )

                # --- Store a simple summary for future dashboards/modules ---
                st.session_state["weekly_rim_summary"] = {
                    "turn_factor": float(turn_factor),
                    "crew_aircraft_required": int(rim_req["aircraft_required"]),
                    "avg_start_capacity": float(np.mean(avg_start)),
                    "avg_end_capacity": float(np.mean(avg_end)),
                    "tai": int(TAI),
                    "overhead": int(Overhead),
                    "sortie_attrition": float(SortieAttr / 100.0),
                }

                # Simple status indicator
                if delta_start >= 0 and delta_end >= 0:
                    st.success("RIM requirement is within simulated capacity for this pattern.")
                elif delta_start >= 0 and delta_end < 0:
                    st.warning("You start the day within RIM, but end-of-day capacity drops below requirement.")
                else:
                    st.error("Simulated capacity is below RIM requirement for most of the day.")
            except Exception as e:
                st.warning(f"RIM comparison to availability not available: {e}")

                # ─────────────────────────────────────────────────────────────
        # RIM / North Star Projection (beta)
        # Uses weekly patterns + break/fix settings to project EP_home & RIM
        # ─────────────────────────────────────────────────────────────
        if enable_rim_ns:
            with st.expander("📉 RIM / North Star Projection (beta)", expanded=False):
                st.caption(
                    "Projects EP_home, NMC pool, and RIM over a short horizon using your "
                    "weekly pattern, break rate, and an approximate fix rate."
                )

                # --- Structure inputs for RIM (PAI/BAI & non-possessed bins) ---
                c1, c2, c3 = st.columns(3)
                with c1:
                    rim_pai = st.number_input(
                        "PAI (Primary Aircraft Inventory)",
                        min_value=0,
                        value=int(TAI),
                        step=1,
                        help="Core aircraft you intend to fly (RIM requirement basis)."
                    )
                with c2:
                    rim_bai = st.number_input(
                        "BAI (Backup Aircraft Inventory)",
                        min_value=0,
                        value=0,
                        step=1,
                        help="Backup aircraft that still count in TAI but are not primary flyers."
                    )
                with c3:
                    rim_horizon = st.number_input(
                        "RIM Horizon (days)",
                        min_value=3,
                        max_value=30,
                        value=7,
                        step=1,
                        help="How many days ahead to project readiness."
                    )

                st.markdown("**Overhead / Degraders breakout (optional, defaults from Overhead):**")
                c4, c5, c6, c7 = st.columns(4)
                with c4:
                    rim_nmcm = st.number_input(
                        "NMCM",
                        min_value=0,
                        value=int(Overhead),
                        step=1,
                        help="Maintenance NMC aircraft."
                    )
                with c5:
                    rim_nmcs = st.number_input("NMCS", min_value=0, value=0, step=1)
                with c6:
                    rim_nmcb = st.number_input("NMCB", min_value=0, value=0, step=1)
                with c7:
                    rim_nmc_fly = st.number_input(
                        "NMC Flyable",
                        min_value=0,
                        value=0,
                        step=1,
                        help="Flyable but NMC (adds headroom, not EP)."
                    )

                st.markdown("**Depot / UPNR / Deployed / Alert / Spares:**")
                c8, c9, c10, c11, c12 = st.columns(5)
                with c8:
                    rim_depot = st.number_input("Depot", min_value=0, value=0, step=1)
                with c9:
                    rim_upnr = st.number_input("UPNR", min_value=0, value=0, step=1)
                with c10:
                    rim_deployed = st.number_input(
                        "Deployed",
                        min_value=0,
                        value=0,
                        step=1,
                        help="Aircraft away from home station."
                    )
                with c11:
                    rim_alert = st.number_input(
                        "Alert Requirement",
                        min_value=0,
                        value=0,
                        step=1,
                        help="Alert tails that must be preserved first."
                    )
                with c12:
                    rim_spares_bin = st.number_input(
                        "Reserved Spares (bin)",
                        min_value=0,
                        value=0,
                        step=1,
                        help="Spares treated as committed in RIM."
                    )

                # --- Build daily training AM & sorties from your patterns ---
                # First-go (AM) tails already parsed: first_go_vals (length 7)
                daily_training_am = [int(v) for v in first_go_vals]

                # Use avg_flown as the daily sortie demand seen in the sim
                daily_sorties = [int(round(v)) for v in avg_flown]

                # --- Translate your fix rates into an approximate daily fix fraction ---
                # Simple approximation: average of 8/12/24-hr fix rates as a daily fraction
                fix_rate_day = min(
                    1.0,
                    max(0.0, (Fix8 + Fix12 + Fix24) / 300.0)  # e.g. (25+35+56)/300 ≈ 0.39
                )

                rim_inputs = RIMProjectionInputs(
                    tai=int(TAI),
                    pai=int(rim_pai),
                    bai=int(rim_bai),
                    depot=int(rim_depot),
                    upnr=int(rim_upnr),
                    other_non_possessed=0,
                    nmcm=int(rim_nmcm),
                    nmcs=int(rim_nmcs),
                    nmcb=int(rim_nmcb),
                    nmc_flyable=int(rim_nmc_fly),
                    deployed=int(rim_deployed),
                    alert=int(rim_alert),
                    spares_bin=int(rim_spares_bin),
                    horizon_days=int(rim_horizon),
                    daily_training_am=daily_training_am,
                    daily_sorties=daily_sorties,
                    break_rate_per_sortie=float(Break_pct) / 100.0,
                    fix_rate_per_day=fix_rate_day,
                    commit_cap=commit_thresh / 100.0,
                )

                try:
                    rim_results = project_rim_as_dicts(rim_inputs)
                    rim_df = pd.DataFrame(rim_results)

                    st.markdown("**RIM Projection Table (per day)**")
                    st.dataframe(
                        rim_df[["day", "ep_home", "training_am", "alert", "spares_bin",
                                "rim", "nmc_total", "breaks", "fixes"]],
                        use_container_width=True
                    )

                    c_r1, c_r2 = st.columns(2)

                    # RIM vs Day
                    with c_r1:
                        fig_rim = go.Figure()
                        fig_rim.add_trace(go.Scatter(
                            x=rim_df["day"],
                            y=rim_df["rim"],
                            mode="lines+markers",
                            name="RIM"
                        ))
                        fig_rim.update_layout(
                            title="RIM vs Day",
                            xaxis_title="Day",
                            yaxis_title="RIM (EP_home − Requirement)"
                        )
                        st.plotly_chart(fig_rim, use_container_width=True)

                    # EP_home vs Requirement & NMC
                    with c_r2:
                        req_series = (
                            rim_df["training_am"] + rim_df["alert"] + rim_df["spares_bin"]
                        )
                        fig_ep = go.Figure()
                        fig_ep.add_trace(go.Scatter(
                            x=rim_df["day"],
                            y=rim_df["ep_home"],
                            mode="lines+markers",
                            name="EP_home"
                        ))
                        fig_ep.add_trace(go.Scatter(
                            x=rim_df["day"],
                            y=req_series,
                            mode="lines+markers",
                            name="Requirement (Alert+AM+Spares)"
                        ))
                        fig_ep.add_trace(go.Bar(
                            x=rim_df["day"],
                            y=rim_df["nmc_total"],
                            name="NMC pool",
                            opacity=0.4
                        ))
                        fig_ep.update_layout(
                            title="EP_home, Requirement & NMC vs Day",
                            xaxis_title="Day",
                            yaxis_title="Tails"
                        )
                        st.plotly_chart(fig_ep, use_container_width=True)

                    st.caption(
                        "RIM < 0 indicates a shortfall: not enough EP_home remaining after Alert, "
                        "Training AM, and Reserved Spares. Adjust degraders, fix rates, or requirements "
                        "to explore solutions."
                    )
                except Exception as e:
                    st.error(f"RIM projection failed: {e}")

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
