"""
Page 4: Failure Taxonomy & Error Root Cause Analysis.
"""

import streamlit as st

from aeroeval.dashboard.utils import ROOT_DIR, apply_custom_css

st.set_page_config(page_title="Error Analysis | AeroEval", page_icon="🔍", layout="wide")
apply_custom_css()

st.title("🔍 Failure Taxonomy & Error Root Cause Analysis")
st.markdown("Diagnose model false alarms, small-object misses, class confusions, and localization errors.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🥧 Error Distribution Breakdown")
    err_dist_img = ROOT_DIR / "reports" / "error_analysis" / "error_distribution_comparison.png"
    if err_dist_img.exists():
        st.image(str(err_dist_img), use_container_width=True)

with col2:
    st.subheader("🔄 Top Confused Object Classes")
    confused_img = ROOT_DIR / "reports" / "error_analysis" / "top_confused_classes.png"
    if confused_img.exists():
        st.image(str(confused_img), use_container_width=True)

st.markdown("---")

st.subheader("🖼️ Sample Failure Cases")
sample_img = ROOT_DIR / "reports" / "error_analysis" / "sample_failure_cases.png"
if sample_img.exists():
    st.image(str(sample_img), caption="Qualitative Failure Mode Samples on VisDrone Validation Imagery", use_container_width=True)
