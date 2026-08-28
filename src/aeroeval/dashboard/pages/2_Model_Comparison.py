"""
Page 2: Model Comparison & Trade-off Analytics.
"""

import streamlit as st

from aeroeval.dashboard.utils import ROOT_DIR, apply_custom_css, load_experiment_matrix

st.set_page_config(page_title="Model Comparison | AeroEval", page_icon="⚖️", layout="wide")
apply_custom_css()

st.title("⚖️ Cross-Model Trade-off & Pareto Analytics")
st.markdown("Compare accuracy, latency, model size, and hardware trade-offs across evaluated models.")

df = load_experiment_matrix()

st.subheader("📊 Comparative Metrics Table")
st.dataframe(df, use_container_width=True)

# Scatter plot: mAP vs Latency
st.subheader("🎯 Accuracy vs Latency Pareto Frontier")
chart_path = ROOT_DIR / "reports" / "pareto_frontier_accuracy_vs_latency.png"
if chart_path.exists():
    st.image(str(chart_path), caption="Pareto Frontier: Accuracy (mAP) vs Latency (ms)", use_container_width=True)
else:
    st.scatter_chart(df, x="Latency_ms", y="mAP50", color="Model")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Per-Class AP Comparison")
    per_class_img = ROOT_DIR / "reports" / "per_class_ap50_comparison.png"
    if per_class_img.exists():
        st.image(str(per_class_img), use_container_width=True)

with col2:
    st.subheader("🕸️ Multi-Dimensional Radar Comparison")
    radar_img = ROOT_DIR / "reports" / "deployment_profile_radar_comparison.png"
    if radar_img.exists():
        st.image(str(radar_img), use_container_width=True)
