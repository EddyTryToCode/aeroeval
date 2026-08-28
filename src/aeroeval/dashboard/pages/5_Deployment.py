"""
Page 5: Edge Deployment & Model Recommendation Engine.
"""

import streamlit as st

from aeroeval.dashboard.utils import ROOT_DIR, apply_custom_css, load_onnx_comparison

st.set_page_config(page_title="Deployment Recommender | AeroEval", page_icon="🚀", layout="wide")
apply_custom_css()

st.title("🚀 Edge Deployment & Model Recommendation Engine")
st.markdown("Compare runtime inference backends (PyTorch vs ONNX) and select optimal models based on operational UAV profiles.")

# 1. ONNX Comparison
st.subheader("⚡ Runtime Optimization: PyTorch vs ONNXRuntime")
df_onnx = load_onnx_comparison()
st.dataframe(df_onnx, use_container_width=True)

chart_onnx = ROOT_DIR / "reports" / "benchmark" / "onnx_benchmark_comparison.png"
if chart_onnx.exists():
    st.image(str(chart_onnx), use_container_width=True)

st.markdown("---")

# 2. Automated Profile Recommender
st.subheader("🎯 Deployment Profile Recommender")

profile_choice = st.selectbox(
    "Select Target UAV Mission Profile:",
    options=["real_time_uav", "high_accuracy", "edge_device"],
    format_func=lambda x: {
        "real_time_uav": "🛸 Real-Time UAV Perception (30% Acc, 30% Latency, 25% Robustness, 15% Memory)",
        "high_accuracy": "🎯 High-Altitude Reconnaissance (50% Acc, 15% Latency, 25% Robustness, 10% Memory)",
        "edge_device": "🔋 Edge Microcontroller / Jetson Nano (20% Acc, 25% Latency, 15% Robustness, 40% Memory)"
    }[x]
)

rec_md = ROOT_DIR / "reports" / "recommendation_summary.md"
if rec_md.exists():
    with open(rec_md, "r", encoding="utf-8") as f:
        st.markdown(f.read())
else:
    st.info("Run `python -m aeroeval recommend` to generate comprehensive recommendation rankings.")
