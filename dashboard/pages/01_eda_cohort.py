"""EDA & Cohort Analysis page for dashboard."""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_loader import DDI_DataLoader

st.title("📊 Exploratory Data Analysis & Cohort")

# Load data
@st.cache_data
def load_data():
    loader = DDI_DataLoader(root_dir="./DDI")
    return loader, loader.get_metadata()

try:
    loader, metadata = load_data()
    
    # Overview
    st.header("1. Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Images", len(metadata))
    
    with col2:
        st.metric("Malignant Cases", int(metadata['malignant'].sum()))
    
    with col3:
        st.metric("Benign Cases", int((~metadata['malignant']).sum()))
    
    with col4:
        malignancy_rate = metadata['malignant'].mean() * 100
        st.metric("Malignancy Rate", f"{malignancy_rate:.1f}%")
    
    # Skin tone distribution
    st.header("2. Skin Tone Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        skin_counts = metadata['skin_tone_label'].value_counts()
        skin_counts.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
        ax.set_title('Distribution by Skin Tone')
        ax.set_xlabel('Fitzpatrick Scale')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col2:
        st.markdown("### Skin Tone Counts")
        st.dataframe(loader.get_subgroup_counts(), use_container_width=True)
    
    # Malignancy by group
    st.header("3. Outcome Prevalence by Demographic Group")
    
    outcome_rates = loader.get_outcome_by_group()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(outcome_rates['skin_tone_label'], outcome_rates['malignancy_rate'], 
               color='coral', edgecolor='black')
        ax.set_title('Malignancy Rate by Skin Tone')
        ax.set_xlabel('Skin Tone')
        ax.set_ylabel('Malignancy Rate (%)')
        plt.xticks(rotation=45)
        
        # Add overall rate line
        overall_rate = metadata['malignant'].mean() * 100
        ax.axhline(y=overall_rate, color='red', linestyle='--', 
                   label=f'Overall: {overall_rate:.1f}%')
        ax.legend()
        
        st.pyplot(fig)
    
    with col2:
        st.markdown("### Malignancy Rates")
        st.dataframe(outcome_rates, use_container_width=True)
    
    # Fairness observations
    st.header("4. Initial Fairness Observations")
    
    max_rate = outcome_rates['malignancy_rate'].max()
    min_rate = outcome_rates['malignancy_rate'].min()
    disparity = max_rate - min_rate
    
    st.markdown(f"""
    **Outcome Disparity Analysis:**
    - Highest malignancy rate: **{max_rate:.2f}%**
    - Lowest malignancy rate: **{min_rate:.2f}%**
    - Absolute disparity: **{disparity:.2f} percentage points**
    """)
    
    if disparity > 10:
        st.warning(f"⚠️ Substantial disparity detected ({disparity:.1f}pp). This may indicate biological differences, data collection bias, or potential fairness concerns.")
    else:
        st.success(f"✓ Disparity is moderate ({disparity:.1f}pp)")
    
    # Raw data
    with st.expander("📋 View Raw Data"):
        st.dataframe(metadata, use_container_width=True)
    
except FileNotFoundError as e:
    st.error(f"❌ Dataset not found: {e}")
    st.info("Please ensure DDI dataset is extracted in the project root directory.")
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
