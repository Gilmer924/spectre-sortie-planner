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

        # ── RISK METRICS & SUGGESTIONS ────────────────────────
        mnd_risk   = sum(1 for r in week_results if r["mnd"]>0)/len(week_results)*100
        abort_risk = sum(1 for r in week_results if r["ground_aborts"]>0)/len(week_results)*100
        risk_msg = f"MND-Risk: {mnd_risk:.1f}% | Abort-Risk: {abort_risk:.1f}%"

        if mnd_risk > 10 or abort_risk > 10:
            st.error(risk_msg + " → HIGH risk of failure!")
        elif mnd_risk > 5 or abort_risk > 5:
            st.warning(risk_msg + " → Medium risk; consider adjustments.")
        else:
            st.success(risk_msg + " → Low risk 👍")

        if mnd_risk > 5 and not weekend_flags[sel_w-1]:
            st.info("💡 Suggestion: enable Weekend Duty for repairs.")
        if any(s > 0.9 * a for s, a in zip(avg_sched, avg_avail)):
            st.info("💡 Suggestion: reduce daily pattern to build buffer.")
        if df["fix_completed"].mean() < df["breaks"].mean():
            st.info("💡 Suggestion: increase short-turn fix rates.")

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
    import calendar, pandas as pd
    st.header("👥 Personnel Capacity Simulation")

    # — Sidebar Basic Params —
    TAI    = st.sidebar.number_input("TAI (# Aircraft)", value=28, step=1)
    shifts = st.sidebar.number_input("Shifts", min_value=1, max_value=10, value=1, step=1)
    lps    = st.sidebar.number_input("Labor Hrs/Sortie", value=10.0)
    TrialsP= st.sidebar.slider("Monte Carlo Trials", 1, 5000, 500)

    # — Timeframe + Goals —
    mode = st.radio("Timeframe", ["Full FY", "Single Month"])
    if mode == "Full FY":
        st.subheader("O&M Days & Goals — Full FY")
        cols = st.columns(2)
        with cols[0]:
            om_days = {m: st.number_input(f"O&M days {m}", value=20, min_value=0, key=f"om{m}") for m in range(1,13)}
        with cols[1]:
            month_goals = {m: st.number_input(f"Goal sorties {m}", value=0, min_value=0, key=f"goal{m}") for m in range(1,13)}
    else:
        st.subheader("O&M Days & Goal — Single Month")
        month = st.selectbox("month", list(range(1,13)))
        om_days     = {month: st.number_input("O&M days", value=20, min_value=0)}
        month_goals = {month: st.number_input("Sortie goal", value=0, min_value=0)}

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
            "months":                 list(om_days.keys()),
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

        # turn the **first** trial into a DataFrame
        df_per = pd.DataFrame(per_results[0]).set_index("month")
        st.success("✅ Simulation complete!")
        st.write("### Monthly Summary")
        st.dataframe(df_per)

        # — Chart: Supported vs Goal —
        months    = df_per.index.tolist()
        supported = df_per["sorties_supported"].tolist()
        goals     = [month_goals.get(m,0) for m in months]
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=months, y=supported, mode="lines+markers", name="Supported"))
        fig1.add_trace(go.Scatter(x=months, y=goals,     mode="lines+markers", name="Goal"))
        fig1.update_layout(
            title="Monthly Sorties Supported vs Goal",
            xaxis_title="Month",
            yaxis_title="Sorties",
            xaxis=dict(tickmode="array", tickvals=months)
        )
        st.plotly_chart(fig1, use_container_width=True)

        # — Optionally: distribution histogram if MC >1 —
        if TrialsP > 1:
            all_months = list(zip(*[
                [trial[m]["sorties_supported"] for m in range(len(months))]
                for trial in per_results
            ]))
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(x=all_months[0], nbinsx=20, name=f"Month {months[0]}"))
            fig2.update_layout(
                title=f"Histogram: Sorties Supported (Month {months[0]})",
                xaxis_title="Sorties Supported",
                yaxis_title="Frequency"
            )
            st.plotly_chart(fig2, use_container_width=True)

        # — Heatmap: % Manned by Shop × Month —
        shops  = list(workcenters.keys())
        matrix = []
        for shop in shops:
            row = []
            auth = sum(wc["auth"].get(lvl,0) for wc in wc_list if wc["shop"]==shop for lvl in ("3","5","7"))
            for idx,m in enumerate(months):
                pres = per_results[0][idx]["present_people"]
                row.append((pres/auth*100) if auth>0 else 0)
            matrix.append(row)

        fig3 = go.Figure(data=go.Heatmap(
            z=matrix,
            x=[calendar.month_abbr[m] for m in months],
            y=shops,
            colorscale="RdYlGn",
            zmin=0, zmax=100
        ))
        fig3.update_layout(title="% Manned by Shop × Month")
        st.plotly_chart(fig3, use_container_width=True)


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
    st.header("📅 Annual Historical Simulation")

    # 1) Upload & preview raw data
    uploaded = st.file_uploader(
        "Upload 4-yr roll-up (sheet “Historical Data Input”)", 
        type=["xlsx","xls"]
    )
    if not uploaded:
        st.info("Please upload your Excel file to proceed.")
        st.stop()

    # 2) Read in the sheet
    try:
        df_hist = pd.read_excel(uploaded, sheet_name="Historical Data Input")
    except Exception as e:
        st.error(f"Could not read sheet: {e}")
        st.stop()

        # ── normalize column names ────────────────────────────────────
    df_hist.columns = (
        df_hist.columns
               .str.strip()
               .str.lower()
               .str.replace(" ", "_")
    )
    # sanity check
    st.write("🔍 Columns after normalization:", df_hist.columns.tolist())
    
    # 3) quick prod-rate debug
    prod = (
        df_hist.set_index("month")["sorties_flown"]
               / df_hist.set_index("month")["hours_flown"]
    )
    st.write("📊 Historical prod rates:", prod.to_dict())
    
    # ── Normalize month names to numbers ─────────────────────────
        # ── Normalize month names to numbers ─────────────────────────
    import calendar
    name_to_num = {name: idx for idx, name in enumerate(calendar.month_name) if name}
    df_hist["month_num"] = (
        df_hist["month"]
        .astype(str)
        .str.strip()
        .str.title()
        .map(name_to_num)
    )
        # quick sanity check
    st.write("🔍 month_num values:", df_hist["month_num"].unique())

    # 2) Core inputs & targets
    with st.expander("▶️ Core & Targets", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            TAI         = st.number_input("TAI (Aircraft Inventory)", value=12, step=1)
            MC_target   = st.slider("MC-Rate Target (%)", 0, 100, 80, 
                                    help="Your desired minimum MC-rate")
            FY_goal     = st.number_input("FY Flying-Hour Goal (hrs)", value=0.0, step=100.0)
            spares_pct  = st.slider("Spares Buffer (%)", 0.0, 100.0, 20.0) / 100.0
        with col2:
            st.markdown("**Turn Pattern by month**")
            patterns = {
                m: st.text_input(f" Mo{m} pattern", "10x8", key=f"tp{m}")
                for m in range(1,13)
            }
            st.markdown("**Commit Rate by M=month (%)**")
            commits = {
                m: st.slider(f" Mo{m} commit", 0, 100, 65, key=f"cr{m}") / 100.0
                for m in range(1,13)
            }

    # 3) O&M days & planned degraders
    with st.expander("▶️ O&M Days & Planned Degraders", expanded=False):
        om_days   = {}
        degraders = {}
        for m in range(1,13):
            max_d = calendar.monthrange(2025, m)[1]
            om_days[m]   = st.number_input(f"O&M days Mo{m}", 
                                           min_value=0, max_value=max_d,
                                           value=max_d, key=f"om{m}")
            degraders[m] = st.number_input(f"Degraders Mo{m}", 
                                           min_value=0, max_value=TAI,
                                           value=0, key=f"deg{m}")

    # 4) Run the Monte Carlo analysis
    if st.button("Run Annual Analysis"):
        params = {
            "TAI":                TAI,
            "om_days":            om_days,
            "planned_degraders":  degraders,
            "turn_patterns":      patterns,
            "commit_rates":       commits,
            "spares_pct":         spares_pct,
            "historical_df":      df_hist
        }
        try:
            out = run_historical_simulation(params, trials=500)[0]
        except Exception as e:
            st.error(f"Annual sim failed: {e}")
            st.stop()

        # 5) Build output DataFrame
        df_out = pd.DataFrame(out)
        df_out["month_name"] = df_out["month"].map({i: calendar.month_abbr[i] for i in df_out["month"]})

        # 6) Monthly Sortie Capacity (mean ± 95% CI)
        st.subheader("📈 Monthly Sortie Capacity (mean ± 95% CI)")
        fig = go.Figure([
            go.Scatter(
                x=df_out["month_name"],
                y=df_out["cap_mean"],
                mode="lines+markers",
                name="Mean Capacity"
            ),
            go.Scatter(
                x=df_out["month_name"].tolist() + df_out["month_name"][::-1].tolist(),
                y=df_out["cap_ci_hi"].tolist()  + df_out["cap_ci_lo"][::-1].tolist(),
                fill="toself", fillcolor="rgba(0,100,200,0.2)",
                line=dict(width=0), hoverinfo="skip", showlegend=False
            )
        ])
        fig.update_layout(xaxis_title="month", yaxis_title="Sorties")
        st.plotly_chart(fig, use_container_width=True)

        # 7) Over-commitment risk
        st.subheader("⚠️ Over-commitment Risk by month")
        risk_fig = go.Figure([go.Bar(
            x=df_out["month_name"],
            y=df_out["overcommit_risk"],
            name="Risk %"
        )])
        risk_fig.update_layout(
            xaxis_title="month",
            yaxis_title="Risk % (capacity < attempted)",
            yaxis=dict(ticksuffix="%")
        )
        st.plotly_chart(risk_fig, use_container_width=True)

        # 8) Pass/Fail summary
        avg_mc = df_out["cap_mean"].sum()  # or any MC metric you prefer
        # (you could compute average MC-rate here if your simulate returns it)
        cap_hours = df_out["cap_mean"].sum()  # sorties ≈ hours if ASD≈1
        st.markdown("---")
        if cap_hours < FY_goal:
            st.warning(f"❌ Total capacity {cap_hours:.0f} sorties < FY goal {FY_goal:.0f} hrs")
        else:
            st.success(f"✅ Total capacity {cap_hours:.0f} sorties ≥ FY goal {FY_goal:.0f} hrs")

        # 9) Detailed table + CSV download
        st.subheader("📋 Detailed Monthly Results")
        st.dataframe(df_out.set_index("month_name"))
        csv_buf = df_out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download CSV", csv_buf,
            file_name="annual_capacity_results.csv",
            mime="text/csv"
        )
