"""Confident AI-emulating Streamlit Evaluation Dashboard for DeepEval."""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from remediation_engine.evals.db import EvalDatabase
from remediation_engine.evals.runner import SUITE_PATHS, create_sample_run, run_eval_subprocess

# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="DeepEval Local Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Confident AI styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-pass {
        background-color: #d1e7dd;
        color: #0f5132;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-fail {
        background-color: #f8d7da;
        color: #842029;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-skip {
        background-color: #fff3cd;
        color: #664d03;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .text-box-card {
        margin-bottom: 12px;
    }
    .text-box-title {
        font-weight: 600;
        font-size: 0.88rem;
        margin-bottom: 6px;
        color: #ffffff !important;
    }
    .text-box-content {
        background-color: #f8fafc;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        padding: 12px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        font-size: 0.85rem;
        line-height: 1.5;
        color: #000000 !important;
        white-space: pre-wrap;
        word-break: break-word;
        overflow-wrap: break-word;
        max-height: none;
    }
    .reason-box {
        background-color: #f1f5f9;
        border-left: 4px solid #0284c7;
        padding: 12px 16px;
        border-radius: 0 6px 6px 0;
        color: #000000 !important;
        font-size: 0.92rem;
        line-height: 1.5;
        margin-top: 8px;
        margin-bottom: 8px;
    }
    .reason-box b, .reason-box strong, .reason-box span, .reason-box p {
        color: #000000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_database() -> EvalDatabase:
    """Initialize and cache the EvalDatabase instance."""
    return EvalDatabase()


db = get_database()

# ---------------------------------------------------------------------------
# Sidebar Navigation & Filters
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🛡️ Confident AI")
    st.caption("DeepEval Local Evaluation Platform")
    st.divider()

    runs = db.get_runs(limit=100)

    st.subheader("Run Selector")
    if runs:
        run_options = {
            f"{r['timestamp'][:19].replace('T', ' ')} | {r['suite_name']} ({r['pass_rate']}% pass)": r[
                "run_id"
            ]
            for r in runs
        }
        selected_label = st.selectbox(
            "Select Evaluation Run",
            options=list(run_options.keys()),
            index=0,
        )
        selected_run_id = run_options[selected_label]
    else:
        st.info("No evaluation runs recorded yet.")
        selected_run_id = None

    st.divider()
    st.subheader("Filter Test Cases")
    status_filter = st.selectbox("Status", ["All", "PASSED", "FAILED", "SKIPPED"], index=0)
    suite_filter = st.selectbox(
        "Component Suite",
        ["All", "report", "triage", "fix_planner", "qa_critic", "subagent", "general"],
        index=0,
    )
    search_query = st.text_input("Search (text, ID, prompt)", placeholder="e.g. CVE, narrative...")

    st.divider()
    st.subheader("Quick Actions")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🌱 Demo Run", help="Seed a sample evaluation run for immediate preview"):
            create_sample_run(db=db)
            st.success("Sample run created!")
            st.rerun()
    with col_b:
        if st.button("🔄 Refresh", help="Reload database runs"):
            st.rerun()

    with st.expander("Database Info"):
        st.write(f"**DB Path:** `{db.db_path}`")
        st.write(f"**Total Runs:** {len(runs)}")
        if st.button("⚠️ Clear DB", type="secondary"):
            db.clear_all()
            st.warning("Database cleared.")
            st.rerun()

# ---------------------------------------------------------------------------
# Main Tabs Layout
# ---------------------------------------------------------------------------

tab_explorer, tab_trends, tab_compare, tab_runner = st.tabs(
    [
        "📊 Evaluation Explorer",
        "📈 Historical Trends",
        "⚖️ Run Comparison",
        "🚀 Run Evaluations",
    ]
)

# ---------------------------------------------------------------------------
# TAB 1: Evaluation Explorer (Confident AI Core View)
# ---------------------------------------------------------------------------

with tab_explorer:
    if not selected_run_id:
        st.info(
            "👋 Welcome to the DeepEval Local Dashboard! "
            "No evaluation runs are stored yet. Click **'🌱 Demo Run'** in the sidebar to populate sample data "
            "or run your test suite via **'🚀 Run Evaluations'** or `pytest tests/evals`."
        )
    else:
        run_data = db.get_run(selected_run_id)
        if run_data:
            # Top KPI Metric Row
            kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
            kpi1.metric("Total Test Cases", run_data["total_tests"])
            kpi2.metric(
                "Pass Rate",
                f"{run_data['pass_rate']}%",
                f"{run_data['passed_tests']} passed / {run_data['failed_tests']} failed",
            )
            kpi3.metric("Duration", f"{run_data['duration_seconds']:.2f}s")
            kpi4.metric("Total Cost", f"${run_data['total_cost']:.4f}")
            kpi5.metric("Judge Model", run_data["judge_model"])

            st.caption(
                f"**Run ID:** `{run_data['run_id']}` | **Timestamp:** `{run_data['timestamp']}` | "
                f"**Suite:** `{run_data['suite_name']}` | **Live LLM:** `{'Yes' if run_data['is_live'] else 'No'}`"
            )

            # Test Cases List
            test_cases = db.get_test_cases(
                run_id=selected_run_id,
                status_filter=status_filter,
                suite_filter=suite_filter,
                search_query=search_query,
            )

            st.markdown(f"### 🧪 Test Cases ({len(test_cases)})")

            if not test_cases:
                st.warning("No test cases match the selected filters.")
            else:
                for idx, tc in enumerate(test_cases, start=1):
                    status_badge = (
                        "🟢 PASS"
                        if tc["status"] == "PASSED"
                        else "🔴 FAIL"
                        if tc["status"] == "FAILED"
                        else "🟡 SKIP"
                    )

                    # Summary metric scores for expander label
                    metric_summaries = [
                        f"{m['metric_name']}: {m['score']:.2f} ({'PASS' if m['success'] else 'FAIL'})"
                        for m in tc.get("metrics", [])
                    ]
                    metric_header = (
                        " | ".join(metric_summaries) if metric_summaries else "No metrics recorded"
                    )

                    expander_title = f"{status_badge} | #{idx} [{tc['suite']}] {tc['test_name']} — {metric_header}"

                    with st.expander(
                        expander_title, expanded=(tc["status"] == "FAILED" and idx <= 3)
                    ):
                        # Top metadata row
                        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                        m_col1.write(f"**Suite:** `{tc['suite']}`")
                        m_col2.write(f"**Status:** `{tc['status']}`")
                        m_col3.write(f"**Latency:** `{tc['latency_seconds']:.2f}s`")
                        m_col4.write(f"**Cost:** `${tc['cost']:.4f}`")

                        # Side-by-side Input / Actual Output / Expected Output (fully expanded, no scroll clipping)
                        c_in, c_act, c_exp = st.columns(3)
                        with c_in:
                            escaped_input = html.escape(tc["input_text"] or "(empty)")
                            st.markdown(
                                f'<div class="text-box-card">'
                                f'<div class="text-box-title">📥 Input (Prompt / Evidence):</div>'
                                f'<div class="text-box-content">{escaped_input}</div>'
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                        with c_act:
                            escaped_actual = html.escape(tc["actual_output"] or "(empty)")
                            st.markdown(
                                f'<div class="text-box-card">'
                                f'<div class="text-box-title">📤 Actual Output:</div>'
                                f'<div class="text-box-content">{escaped_actual}</div>'
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                        with c_exp:
                            escaped_expected = html.escape(
                                tc["expected_output"] or "(none specified)"
                            )
                            st.markdown(
                                f'<div class="text-box-card">'
                                f'<div class="text-box-title">🎯 Expected Output / Ground Truth:</div>'
                                f'<div class="text-box-content">{escaped_expected}</div>'
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                        # Context documents if present
                        if tc.get("context_text"):
                            with st.expander("📄 Evaluation Ground Truth Context", expanded=False):
                                escaped_context = html.escape(tc["context_text"])
                                st.markdown(
                                    f'<div class="text-box-content">{escaped_context}</div>',
                                    unsafe_allow_html=True,
                                )

                        # Metrics breakdown
                        st.markdown("#### 📏 Metric Breakdown")
                        metrics = tc.get("metrics", [])
                        if not metrics:
                            st.write("No granular metric logs recorded for this test.")
                        else:
                            for m in metrics:
                                m_pass = m["success"]
                                st.markdown(
                                    f"- **{m['metric_name']}** — Score: `{m['score']:.2f}` / Threshold: `{m['threshold']:.2f}` "
                                    f"({'✅ PASSED' if m_pass else '❌ FAILED'})"
                                )
                                st.progress(min(max(float(m["score"]), 0.0), 1.0))
                                if m.get("reason"):
                                    escaped_reason = html.escape(m["reason"])
                                    st.markdown(
                                        f"<div class='reason-box'><b>LLM Judge Reason:</b> {escaped_reason}</div>",
                                        unsafe_allow_html=True,
                                    )
                                if m.get("verbose_logs"):
                                    with st.expander("Verbose Metric Logs", expanded=False):
                                        escaped_verbose = html.escape(m["verbose_logs"])
                                        st.markdown(
                                            f'<div class="text-box-content">{escaped_verbose}</div>',
                                            unsafe_allow_html=True,
                                        )

# ---------------------------------------------------------------------------
# TAB 2: Historical Trends & Analytics
# ---------------------------------------------------------------------------

with tab_trends:
    st.markdown("### 📈 Evaluation Trends & Historical Performance")
    all_runs = db.get_runs(limit=100)

    if len(all_runs) < 2:
        st.info(
            "Record at least 2 evaluation runs to visualize historical trends and metric curves."
        )
    else:
        df_runs = pd.DataFrame(all_runs)
        df_runs["timestamp_dt"] = pd.to_datetime(df_runs["timestamp"])
        df_runs = df_runs.sort_values("timestamp_dt")

        # Pass Rate Trendline
        st.subheader("Pass Rate Trajectory (%)")
        st.line_chart(
            data=df_runs.set_index("timestamp_dt")[["pass_rate"]],
            y_label="Pass Rate %",
        )

        # Test Counts Stacked
        st.subheader("Test Execution Volume")
        st.bar_chart(
            data=df_runs.set_index("timestamp_dt")[
                ["passed_tests", "failed_tests", "skipped_tests"]
            ],
            stack=True,
        )

        # Granular Metric Trends
        st.subheader("Metric Score Trajectories")
        trends = db.get_metric_trends()
        if trends:
            df_trends = pd.DataFrame(trends)
            df_trends["timestamp_dt"] = pd.to_datetime(df_trends["timestamp"])

            metrics_list = sorted(df_trends["metric_name"].unique())
            selected_metric = st.selectbox("Select Metric for Deep Dive", options=metrics_list)

            filtered_trends = df_trends[df_trends["metric_name"] == selected_metric]
            st.line_chart(
                data=filtered_trends.set_index("timestamp_dt")[["avg_score"]],
                y_label=f"Average {selected_metric} Score",
            )
        else:
            st.write("No granular metric score records available yet.")

# ---------------------------------------------------------------------------
# TAB 3: Run Comparison (Regression Analysis)
# ---------------------------------------------------------------------------

with tab_compare:
    st.markdown("### ⚖️ Side-by-Side Run Comparison & Regression Analysis")
    all_runs = db.get_runs(limit=100)

    if len(all_runs) < 2:
        st.info("At least 2 runs are required to perform regression comparison.")
    else:
        col_r1, col_r2 = st.columns(2)
        run_dict = {
            f"{r['timestamp'][:19]} ({r['suite_name']}) - {r['run_id']}": r["run_id"]
            for r in all_runs
        }
        keys = list(run_dict.keys())

        with col_r1:
            label_a = st.selectbox("Baseline Run (A)", options=keys, index=min(1, len(keys) - 1))
            run_id_a = run_dict[label_a]

        with col_r2:
            label_b = st.selectbox("Candidate Run (B)", options=keys, index=0)
            run_id_b = run_dict[label_b]

        comparison = db.get_run_comparison(run_id_a, run_id_b)

        if "error" in comparison:
            st.error(comparison["error"])
        else:
            # Comparison KPI cards
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Baseline Pass Rate", f"{comparison['run_a']['pass_rate']}%")
            c2.metric(
                "Candidate Pass Rate",
                f"{comparison['run_b']['pass_rate']}%",
                f"{comparison['pass_rate_delta']:+.1f}%",
            )
            c3.metric(
                "Regressions",
                comparison["total_regressions"],
                delta=-comparison["total_regressions"],
                delta_color="inverse",
            )
            c4.metric(
                "Fixes",
                comparison["total_fixes"],
                delta=comparison["total_fixes"],
                delta_color="normal",
            )

            # Regressions table
            if comparison["regressions"]:
                st.error("🚨 Detected Regressions (Passed in Baseline, Failed in Candidate)")
                for reg in comparison["regressions"]:
                    st.write(
                        f"- **{reg['test_name']}** (`{reg['suite']}`): Status changed from `{reg['status_a']}` to `{reg['status_b']}`"
                    )

            # Fixes table
            if comparison["fixes"]:
                st.success("🎉 Resolved Fixes (Failed in Baseline, Passed in Candidate)")
                for fix in comparison["fixes"]:
                    st.write(
                        f"- **{fix['test_name']}** (`{fix['suite']}`): Status changed from `{fix['status_a']}` to `{fix['status_b']}`"
                    )

            # Full test cases comparison table
            st.subheader("All Test Cases Diff")
            comp_rows = [
                {
                    "Test Case": item["test_name"],
                    "Suite": item["suite"],
                    "Status (A)": item["status_a"],
                    "Status (B)": item["status_b"],
                    "Score (A)": item["score_a"],
                    "Score (B)": item["score_b"],
                    "Delta": item["score_delta"],
                }
                for item in comparison["comparisons"]
            ]
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 4: Evaluation Runner
# ---------------------------------------------------------------------------

with tab_runner:
    st.markdown("### 🚀 Launch Evaluation Suite")
    st.caption(
        "Trigger DeepEval test suites directly from the dashboard. Results auto-populate upon completion."
    )

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        suite_choice = st.selectbox(
            "Evaluation Suite",
            options=list(SUITE_PATHS.keys()),
            format_func=lambda s: f"{s.upper()} ({', '.join(SUITE_PATHS[s])})",
        )
    with col_s2:
        judge_choice = st.text_input("Judge Model", value="gpt-4o")
    with col_s3:
        live_eval = st.checkbox("Enable Live LLM Judge (--run-eval-live)", value=False)

    if st.button("▶ Start Evaluation Run", type="primary"):
        output_placeholder = st.empty()
        log_lines: list[str] = []

        with st.spinner("Executing evaluation suite..."):
            runner_gen = run_eval_subprocess(
                suite_key=suite_choice,
                is_live=live_eval,
                judge_model=judge_choice,
            )
            for line in runner_gen:
                log_lines.append(line)
                output_placeholder.code("".join(log_lines[-40:]), language="bash")

        st.success("Evaluation run completed and persisted to database!")
        st.rerun()
