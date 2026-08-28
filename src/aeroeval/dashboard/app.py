"""
AeroEval — Streamlit Main Dashboard Application.
"""

import streamlit as st

from aeroeval.dashboard.utils import apply_custom_css, load_experiment_matrix

st.set_page_config(
    page_title="AeroEval — UAV Vision Evaluation Platform",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()

# Header
st.markdown("""
<div style="padding: 10px 0 25px 0;">
    <span class="badge-tag">Platform v1.0 • Real-Time UAV AI Benchmark</span>
    <h1 style="font-size: 2.8rem; margin-top: 10px; font-weight: 800; background: linear-gradient(90deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        AeroEval Evaluation Dashboard
    </h1>
    <p style="color: #94a3b8; font-size: 1.1rem;">
        Comprehensive multi-modal benchmark platform for drone perception, small-object detection, robustness, and edge deployment efficiency.
    </p>
</div>
""", unsafe_allow_html=True)

# Overview KPIs
col1, col2, col3, col4 = st.columns(4)

matrix_df = load_experiment_matrix()
best_acc = matrix_df["mAP50"].max() if "mAP50" in matrix_df.columns else 0.472
best_fps = matrix_df["FPS"].max() if "FPS" in matrix_df.columns else 74.6
best_rob = matrix_df["Robustness_%"].max() if "Robustness_%" in matrix_df.columns else 89.1
min_size = matrix_df["Size_MB"].min() if "Size_MB" in matrix_df.columns else 5.2

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Highest Accuracy (mAP@0.5)</div>
        <div class="metric-val">{best_acc * 100:.1f}%</div>
        <div style="color: #10b981; font-size: 0.85rem;">Exp B3 (YOLO11m-960)</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Max Inference Throughput</div>
        <div class="metric-val">{best_fps:.1f} <span style="font-size: 1rem; color: #94a3b8;">FPS</span></div>
        <div style="color: #38bdf8; font-size: 0.85rem;">Exp A (YOLO11n-640)</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Max Robustness Retention</div>
        <div class="metric-val">{best_rob:.1f}%</div>
        <div style="color: #f59e0b; font-size: 0.85rem;">Under Corruptions</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Smallest Model Size</div>
        <div class="metric-val">{min_size:.1f} <span style="font-size: 1rem; color: #94a3b8;">MB</span></div>
        <div style="color: #6366f1; font-size: 0.85rem;">Edge Deployable</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.subheader("📊 Evaluated Model Matrix")
st.dataframe(matrix_df, use_container_width=True)

st.markdown("""
### 🧭 Navigation Guide
Use the sidebar navigation to explore detailed evaluation analytics:
1. **Overview**: Key findings, latest evaluation runs, and platform metrics.
2. **Model Comparison**: Interactive Pareto frontier (Accuracy vs Latency), per-class AP, and radar charts.
3. **Robustness**: Optical & environmental corruption degradation analysis and heatmap.
4. **Error Analysis**: Failure taxonomy classification and root-cause breakdowns.
5. **Deployment**: PyTorch vs ONNX benchmarks and automated profile recommendation engine.
""")
