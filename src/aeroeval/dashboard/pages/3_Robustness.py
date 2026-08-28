"""
Page 3: Robustness & Environmental Corruption Evaluation.
"""

import pandas as pd
import streamlit as st

from aeroeval.dashboard.utils import ROOT_DIR, apply_custom_css

st.set_page_config(page_title="Robustness Benchmark | AeroEval", page_icon="🌪️", layout="wide")
apply_custom_css()

st.title("🌪️ Robustness & Environmental Corruption Benchmark")
st.markdown("Evaluate UAV model resilience against blur, weather, sensor noise, downscaling, and occlusion.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🔥 Robustness Degradation Heatmap")
    heatmap_img = ROOT_DIR / "reports" / "robustness_heatmap.png"
    if heatmap_img.exists():
        st.image(str(heatmap_img), use_container_width=True)

with col2:
    st.subheader("📉 Corruption Severity Degradation Curves")
    curves_img = ROOT_DIR / "reports" / "robustness_degradation_curves.png"
    if curves_img.exists():
        st.image(str(curves_img), use_container_width=True)

st.markdown("---")

st.subheader("📋 Robustness Performance Summary Table")
robustness_csv = ROOT_DIR / "reports" / "robustness_benchmark_metrics.csv"
if robustness_csv.exists():
    df_rob = pd.read_csv(robustness_csv)
    st.dataframe(df_rob, use_container_width=True)
else:
    st.info("Robustness metrics table available after running full robustness benchmarks.")
