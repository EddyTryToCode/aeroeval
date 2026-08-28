"""
Page 1: Evaluation Overview & Recent Runs.
"""

import streamlit as st

from aeroeval.dashboard.utils import apply_custom_css, load_all_evaluation_runs

st.set_page_config(page_title="Overview | AeroEval", page_icon="📈", layout="wide")
apply_custom_css()

st.title("📈 Platform Overview & Evaluation Runs")
st.markdown("Inspect latest automated evaluation runs, detection metrics, and real-time efficiency metrics.")

runs = load_all_evaluation_runs()

if not runs:
    st.info("No recorded evaluation runs found in `reports/`. Run `python -m aeroeval evaluate` to generate evaluation artifacts.")
else:
    selected_run = st.selectbox("Select Evaluation Run:", options=list(runs.keys()))
    run_data = runs[selected_run]

    det = run_data.get("detection", {})
    eff = run_data.get("efficiency", {})
    rec = run_data.get("recommendation", {})

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Name", run_data.get("model_name", "N/A"))
        st.metric("mAP @ 0.50", f"{det.get('mAP50', 0.0) * 100:.1f}%")
    with col2:
        st.metric("Timestamp", run_data.get("timestamp", "N/A"))
        st.metric("mAP @ 0.50:0.95", f"{det.get('mAP50_95', 0.0) * 100:.1f}%")
    with col3:
        st.metric("Streaming E2E FPS", f"{eff.get('fps_e2e', 0.0)} FPS")
        st.metric("Mean Latency", f"{eff.get('e2e_latency_mean_ms', 0.0)} ms")

    st.markdown("### 📋 Per-Class Detection Breakdown")
    per_class = det.get("per_class_metrics", {})
    if per_class:
        cls_df = [{"Class": k, "AP50-95": v.get("ap50_95", 0.0)} for k, v in per_class.items()]
        st.bar_chart(data=cls_df, x="Class", y="AP50-95")

    st.markdown("### 📄 Full Summary JSON")
    st.json(run_data)
