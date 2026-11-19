"""Convenient and Fair: volleyball pool-play scheduler + format fairness.

Run:  streamlit run app/streamlit_app.py
The app only calls vsched; all logic lives in the package.
"""
from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from vsched.cost import COMPONENTS, check_feasible, load_weights
from vsched.generator import make_instance
from vsched.methods import solve
from vsched.model import Instance, Schedule
from vsched.present import grid_frame, parse_team_list, schedule_csv, timeline_frame
from vsched.roundrobin import instance_matches, instance_pools
from vsched.sim.durations import simulate_delays
from vsched.sim.fairness import simulate_spread
from vsched.sim.tournament import Format

st.set_page_config(page_title="Convenient and Fair", page_icon="🏐", layout="wide")

METHOD_LABELS = {
    "annealing": "Simulated annealing (recommended)",
    "greedy": "Greedy best-first (fast)",
    "wastar2": "Weighted A* (w = 2)",
    "astar": "A* (optimal; small inputs)",
    "ucs": "Uniform-cost search (optimal; tiny inputs)",
    "template": "Standard pool-sheet template",
    "random": "Random baseline",
}
EXACT = {"astar", "ucs", "wastar2", "wastar1.5"}
TINY_MATCHES = 12  # exact search is offered freely only up to this many pool matches
LIMITS = {"node_limit": 200_000, "time_limit_s": 15}
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


@st.cache_data(show_spinner=False)
def run_method(inst_d: dict, weights: dict, method: str, seed: int = 0) -> dict:
    inst = Instance.from_dict(inst_d)
    kw = dict(LIMITS)
    r = solve(method, inst, weights, seed=seed, fallback=True, **kw)
    return {"schedule": r.schedule.to_dict(), "cost": r.cost, "feasible": r.feasible,
            "nodes": r.nodes_expanded, "runtime": r.runtime_s, "hit_limit": r.hit_limit,
            "optimal": r.optimal_proven, "violations": check_feasible(inst, r.schedule)}


@st.cache_data(show_spinner=False)
def run_fairness(n_teams: int, pool_size: int, cut: int, seeding: str, sd: float, runs: int, seed: int) -> dict:
    rest = [p for p in range(1, pool_size + 1) if p > cut]
    fmt = Format(n_teams, pool_size, [("gold", list(range(1, cut + 1))), ("silver", rest)], "best_of_3", seeding)
    r = simulate_spread(fmt, sd, runs, seed)
    r.pop("champion_counts", None)
    return r


@st.cache_data(show_spinner=False)
def run_delays(inst_d: dict, sched_d: dict, cv: float, reps: int) -> dict:
    d = simulate_delays(Instance.from_dict(inst_d), Schedule.from_dict(sched_d), n_reps=reps, seed=0, cv=cv)
    return {k: v.tolist() for k, v in d.items()}


def cost_table(results: dict[str, dict]) -> pd.DataFrame:
    rows = {}
    for name, r in results.items():
        rows[name] = {k: (r["cost"][k] if r["feasible"] else None) for k in COMPONENTS + ("total",)}
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🏐 Convenient and Fair")
st.sidebar.caption("Pool-play scheduling by search, and format fairness by simulation.")
with st.sidebar:
    st.subheader("Tournament")
    mode = st.radio("Teams", ["Generate N teams", "Paste a team list"], horizontal=True)
    if mode == "Generate N teams":
        n_teams = st.number_input("Number of teams", 3, 48, 12)
        coach_frac = st.slider("Share of teams sharing a coach", 0.0, 0.4, 0.15, 0.05)
        team_text = None
    else:
        team_text = st.text_area("One team per line: name, seed, coach (seed and coach optional)",
                                 "Eagles, 1, K1\nSharks, 2\nBears, 3\nWolves, 4, K1\nHawks, 5\nLions, 6\nOwls, 7\nFoxes, 8",
                                 height=170)
    pool_size = st.select_slider("Pool size", [3, 4, 5, 6], 4)
    courts = st.number_input("Courts", 1, 16, 3)
    c1, c2 = st.columns(2)
    start = c1.text_input("Start", "08:00")
    end = c2.text_input("End", "16:00")
    match_minutes = st.select_slider("Match length (minutes)", [30, 40, 45, 50, 60, 75, 90], 60)
    with st.expander("Advanced: cost weights"):
        base_w = load_weights()
        weights = {k: float(st.number_input(k.replace("_", " "), 0.0, 100.0, float(base_w[k]), 1.0)) for k in COMPONENTS}

try:
    if team_text is not None:
        teams = parse_team_list(team_text)
        inst = Instance("app", tuple(teams), int(pool_size), int(courts), start, end, int(match_minutes), 0)
    else:
        inst = make_instance(int(n_teams), int(pool_size), int(courts), seed=0, instance_id="app", start=start,
                             end=end, match_minutes=int(match_minutes), coach_frac=coach_frac)
    pools = instance_pools(inst)
except Exception as e:  # bad input: show it, don't crash
    st.error(f"Input problem: {e}")
    st.stop()

n_matches = len(instance_matches(inst))
inst_d = inst.to_dict()
tiny = n_matches <= TINY_MATCHES

tab1, tab2, tab3, tab4 = st.tabs(["Build a schedule", "Compare methods", "Is this format fair?", "Delays"])

# ------------------------------------------------------------------ tab 1
with tab1:
    st.markdown(f"**{len(inst.teams)} teams** in **{len(pools)} pools** · **{n_matches} pool matches** · "
                f"**{inst.courts} courts × {inst.n_slots} slots** ({inst.start}–{inst.end}, {inst.match_minutes} min)")
    method = st.selectbox("Method", list(METHOD_LABELS), format_func=METHOD_LABELS.get)
    if method in EXACT and not tiny:
        st.warning(f"{METHOD_LABELS[method]} is exact search; on {n_matches} matches it may not finish. "
                   f"It runs with a limit ({LIMITS['time_limit_s']} s / {LIMITS['node_limit']:,} nodes) "
                   "and falls back to greedy if it hits it.")
    with st.spinner("Building schedule…"):
        res = run_method(inst_d, weights, method)
        tpl = run_method(inst_d, weights, "template")
    st.session_state["current"] = res

    if not res["feasible"]:
        st.error("No legal schedule found. Try more courts, a longer day, or shorter matches.")
    else:
        if "fallback" in res["schedule"]["method"]:
            st.warning("Search hit its limit, so this is the greedy schedule instead.")
        elif res["optimal"]:
            st.success("Proven optimal for these weights.")
        sched = Schedule.from_dict(res["schedule"])
        left, right = st.columns([3, 1.3])
        with left:
            st.subheader("Court × time grid")
            text, color = grid_frame(inst, sched)
            st.dataframe(text.style.apply(lambda _: color, axis=None), width="stretch",
                         height=min(40 * (len(text) + 1), 640))
            st.caption("Labels are pool + seed within the pool (A1 = top seed in pool A).")
        with right:
            st.subheader("Cost vs template")
            ct = cost_table({"This schedule": res, "Template": tpl})
            st.dataframe(ct.style.format("{:.0f}", na_rep="n/a"), width="stretch")
            if res["feasible"] and tpl["feasible"]:
                diff = tpl["cost"]["total"] - res["cost"]["total"]
                st.metric("Total cost", f"{res['cost']['total']:.0f}", f"{-diff:+.0f} vs template",
                          delta_color="inverse")
            elif not tpl["feasible"]:
                st.caption("The template does not fit this day.")
            st.caption(f"{res['runtime']:.2f} s · {res['nodes']:,} nodes expanded")
        st.subheader("Team timelines")
        tt, tc = timeline_frame(inst, sched)
        st.dataframe(tt.style.apply(lambda _: tc, axis=None), width="stretch")
        st.caption("Blue = play, orange = ref, pink dot = idle slot between a team's first and last duty.")
        d1, d2 = st.columns(2)
        d1.download_button("Download CSV", schedule_csv(inst, sched), "schedule.csv", "text/csv")
        d2.download_button("Download JSON", json.dumps({"instance": inst_d, "schedule": res["schedule"]}, indent=2),
                           "schedule.json", "application/json")

# ------------------------------------------------------------------ tab 2
with tab2:
    st.write("Every method on the same input. Exact searches run with a limit and fall back to greedy.")
    chosen = st.multiselect("Methods", list(METHOD_LABELS),
                            default=["template", "random", "greedy", "annealing"] + (["astar"] if tiny else []))
    if st.button("Run comparison", type="primary"):
        rows = []
        bar = st.progress(0.0)
        for i, m in enumerate(chosen):
            r = run_method(inst_d, weights, m)
            rows.append({"method": METHOD_LABELS[m], **{k: r["cost"][k] for k in COMPONENTS + ("total",)},
                         "runtime_s": round(r["runtime"], 2), "nodes_expanded": r["nodes"],
                         "proven_optimal": r["optimal"], "note": "fell back to greedy" if "fallback" in r["schedule"]["method"] else ""})
            bar.progress((i + 1) / len(chosen))
        df = pd.DataFrame(rows).set_index("method")
        st.dataframe(df.style.format(precision=0, subset=list(COMPONENTS) + ["total"]), width="stretch")
        fig = go.Figure(go.Bar(x=df.index, y=df["total"], marker_color=BLUE,
                               hovertemplate="%{x}<br>total cost %{y:.0f}<extra></extra>"))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="total cost (lower is better)")
        st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ tab 3
with tab3:
    st.write("How often does the truly best team win Gold? Ratings are redrawn every run from Normal(1500, SD); "
             "each set is won with Elo probability 1 / (1 + 10^(−Δ/400)).")
    c1, c2, c3 = st.columns(3)
    f_teams = c1.number_input("Teams", 8, 48, 24, key="f_teams")
    f_sizes = c1.multiselect("Pool sizes", [3, 4, 5, 6], [3, 4, 6])
    f_cut = c2.radio("Gold takes", [1, 2], index=1, format_func=lambda c: f"top {c} per pool", horizontal=True)
    f_seed = c2.radio("Pools", ["snake", "random"], horizontal=True)
    f_sd = c3.select_slider("Rating spread (SD)", [0, 50, 100, 150, 200, 250, 400], 150)
    f_runs = c3.select_slider("Runs", [500, 1000, 2000, 5000, 10000], 2000)
    rows = []
    for k in f_sizes:
        if f_teams / -(-f_teams // k) < 3:
            continue
        r = run_fairness(int(f_teams), k, f_cut, f_seed, float(f_sd), int(f_runs), 1)
        rows.append({"pool size": k, **r})
    if rows:
        df = pd.DataFrame(rows)
        fig = go.Figure(go.Bar(x=[f"pools of {k}" for k in df["pool size"]], y=df.p_best_wins_gold, marker_color=BLUE,
                               error_y=dict(type="data", symmetric=False, array=df.ci_high - df.p_best_wins_gold,
                                            arrayminus=df.p_best_wins_gold - df.ci_low, color="#52514e"),
                               hovertemplate="%{x}<br>P(best wins Gold) %{y:.1%}<extra></extra>"))
        fig.add_hline(y=1 / f_teams, line_dash="dash", line_color="#8a8984",
                      annotation_text=f"pure luck = 1/{f_teams}", annotation_position="top left")
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), yaxis_tickformat=".0%",
                          yaxis_title="P(best team wins Gold), 95% CI")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(df[["pool size", "p_best_wins_gold", "ci_low", "ci_high", "kendall_tau_mean",
                         "matches_per_team_mean"]].rename(columns={
                             "p_best_wins_gold": "P(best wins Gold)", "ci_low": "95% low", "ci_high": "95% high",
                             "kendall_tau_mean": "rank agreement (Kendall τ)", "matches_per_team_mean": "matches / team"})
                     .style.format(precision=3), width="stretch", hide_index=True)

# ------------------------------------------------------------------ tab 4
with tab4:
    cur = st.session_state.get("current")
    if not cur or not cur["feasible"]:
        st.info("Build a schedule in the first tab, then replay it here.")
    else:
        st.write("Replays the current schedule with lognormal match lengths (mean = planned length). "
                 "A match starts at its planned time or when its court, teams and ref are free.")
        cv = st.slider("Match-length variability (coefficient of variation)", 0.05, 0.5, 0.15, 0.05)
        d = run_delays(inst_d, cur["schedule"], cv, 1000)
        c1, c2, c3 = st.columns(3)
        ov = pd.Series(d["overrun_min"])
        c1.metric("Median overrun", f"{ov.median():.0f} min")
        c2.metric("90th percentile overrun", f"{ov.quantile(0.9):.0f} min")
        c3.metric("Median worst match delay", f"{pd.Series(d['max_delay_min']).median():.0f} min")
        fig = go.Figure(go.Histogram(x=ov, marker_color=ORANGE, nbinsx=40,
                                     hovertemplate="%{x:.0f} min: %{y} replays<extra></extra>"))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="minutes past planned end",
                          yaxis_title="replays (of 1,000)", bargap=0.05)
        st.plotly_chart(fig, width="stretch")
