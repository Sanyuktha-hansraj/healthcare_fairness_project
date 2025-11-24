"""Streamlit dashboard for Healthcare Fairness ML Project."""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="Healthcare Fairness Dashboard",
    page_icon="🏥",
    layout="wide"
)

# Main page
st.title("🏥 Healthcare Fairness ML Dashboard")
st.markdown("### DDI Dermatology AI - Fairness Analysis")

st.markdown("""
This dashboard presents fairness analysis for AI-based skin lesion classification.

**Project Goals:**
- Detect algorithmic bias across skin tone groups
- Measure fairness using multiple metrics
- Develop mitigation strategies for equitable healthcare AI

**Navigation:**
Use the sidebar to navigate between different analysis sections.
""")

st.sidebar.success("Select a page above.")

# Quick stats
st.markdown("---")
st.markdown("### Quick Stats")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Images", "656", help="Total images in DDI dataset")

with col2:
    st.metric("Skin Tone Groups", "3", help="Fitzpatrick I-II, III-IV, V-VI")

with col3:
    st.metric("Fairness Metrics", "4", help="Demographic Parity, Equal Opportunity, etc.")

st.markdown("---")
st.info("👈 Select **EDA & Cohort** from the sidebar to view exploratory data analysis")
