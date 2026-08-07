import os
import json
import time
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# Professional page configuration
st.set_page_config(
    page_title="VoltBrain | ML Project", 
    page_icon="⚡", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource(show_spinner=False)
def load_assets():
    model = joblib.load(os.path.join(BASE_DIR, 'models', 'voltbrain_pipeline.pkl'))
    with open(os.path.join(BASE_DIR, 'models', 'training_report.json'), 'r') as f:
        report = json.load(f)
    df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'EV_Specs_Dataset.csv'))
    return model, report, df

try:
    pipeline, report, df = load_assets()
except Exception:
    st.error("⚠️ Error: Could not load model or dataset files. Please check the 'models' and 'data' directories.")
    st.stop()

# --- SIDEBAR ---
st.sidebar.title("⚡ VoltBrain")
st.sidebar.markdown("**Electric Vehicle Range Prediction**")
st.sidebar.divider()

if 'page' not in st.session_state:
    st.session_state.page = "Prediction"

st.sidebar.markdown("### Menu")
if st.sidebar.button("🔋 Prediction", use_container_width=True):
    st.session_state.page = "Prediction"
if st.sidebar.button("📊 Dashboard", use_container_width=True):
    st.session_state.page = "Dashboard"
if st.sidebar.button("📈 Analytics", use_container_width=True):
    st.session_state.page = "Analytics"
if st.sidebar.button("🗄️ Dataset", use_container_width=True):
    st.session_state.page = "Dataset"

page = st.session_state.page

st.sidebar.divider()
st.sidebar.markdown("**Project Details**")
st.sidebar.caption("Course: Machine Learning Fundamentals")
st.sidebar.caption("Topic: EV Range Estimation Model")
st.sidebar.caption("Year: 2026")

# --- PAGE: PREDICTION ---
if page == "Prediction":
    st.title("🔋 EV Range Predictor")
    st.write("Enter the vehicle specifications below to estimate its real-world driving range.")
    st.divider()
    
    col1, col2 = st.columns([1.2, 1], gap="large")
    
    with col1:
        with st.form("vehicle_form", border=True):
            st.subheader("Vehicle Parameters")
            
            c_batt, c_eff = st.columns(2)
            batt = c_batt.slider("Battery Capacity (kWh)", 20.0, 150.0, 75.0, 0.5)
            eff = c_eff.slider("Energy Efficiency (Wh/km)", 100.0, 300.0, 160.0, 1.0)
            
            spd = st.slider("Top Speed (km/h)", 100.0, 300.0, 200.0, 5.0)
            
            c_drv, c_brd = st.columns(2)
            drv = c_drv.selectbox("Drivetrain Type", ['AWD', 'RWD', 'FWD'])
            brd = c_brd.selectbox("Brand/Manufacturer", ['Tesla', 'Nissan', 'Porsche', 'Audi', 'Ford', 'Chevrolet', 'Hyundai', 'Kia', 'BMW', 'Mercedes', 'Other'])
            
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Calculate Range", type="primary", use_container_width=True)
            
    with col2:
        with st.container(height=390, border=True):
            st.subheader("Prediction Results")
            if submitted:
                # Prepare input
                df_in = pd.DataFrame({
                    'Battery Capacity (kWh)': [batt], 
                    'Efficiency (Wh/km)': [eff], 
                    'Top Speed (km/h)': [spd], 
                    'Brand': [brd], 
                    'Drive Type': [drv]
                })
                
                # Inference
                pred = pipeline.predict(df_in)[0]
                
                st.success("Calculation complete.")
                
                st.metric(label="Estimated Real-World Range", value=f"{pred:.0f} km")
                
                st.divider()
                st.write("**Scenario Estimates**")
                e1, e2, e3 = st.columns(3)
                e1.metric("City", f"{pred * 1.08:.0f} km")
                e2.metric("Highway", f"{pred * 0.88:.0f} km")
                e3.metric("Winter", f"{pred * 0.81:.0f} km")
            else:
                st.info("Please configure the vehicle parameters and calculate to view the real-world range estimate.")

# --- PAGE: DASHBOARD ---
elif page == "Dashboard":
    st.title("📊 Model Dashboard")
    st.write("Overview of the Random Forest model performance and training metrics.")
    st.divider()
    
    st.subheader("Performance Metrics")
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R² Score", f"{report['R2']:.3f}", help="Coefficient of determination (higher is better).")
        c2.metric("Mean Absolute Error", f"{report['MAE']:.1f} km", help="Average error in kilometers.")
        c3.metric("Root Mean Sq Error", f"{report['RMSE']:.1f} km")
        c4.metric("Training Dataset", f"{report['Dataset Rows']} rows")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        with st.container(border=True):
            st.markdown("#### 🧠 About the Architecture")
            st.info(
                "For this project, I used an ensemble **Random Forest Regressor**. "
                "The data pipeline automatically handles standard scaling for numerical features "
                "(like Battery Capacity and Efficiency) and one-hot encoding for categorical variables "
                "(like Brand and Drivetrain)."
            )
    with col_b:
        with st.container(border=True):
            st.markdown("#### 🎯 Project Goal")
            st.success(
                "The goal is to accurately predict the usable, real-world WLTP range of an electric vehicle "
                "based purely on its physical and mechanical specifications, eliminating range anxiety through data."
            )

# --- PAGE: ANALYTICS ---
elif page == "Analytics":
    st.title("📈 Exploratory Data Analytics")
    st.write("Visual explorations of the EV dataset to understand physical correlations and distributions.")
    st.divider()
    
    mapping = {'Battery_Capacity_kWh': 'Battery Capacity (kWh)', 'Range_Km': 'Range (km)'}
    tdf = df.rename(columns=mapping)
    
    # Advanced UI/UX: Interactive Quick Filters and Metrics
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        st.metric("Total Vehicles Analyzed", f"{len(tdf)}")
    with col_f2:
        if 'Range (km)' in tdf.columns:
            st.metric("Average Range", f"{tdf['Range (km)'].mean():.0f} km")
    with col_f3:
        if 'Efficiency_Whkm' in tdf.columns:
            st.metric("Avg. Efficiency", f"{tdf['Efficiency_Whkm'].mean():.0f} Wh/km")
            
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Advanced UI/UX: Tabs for different analytical views
    tab1, tab2 = st.tabs(["📊 Correlations", "📉 Distributions"])
    
    with tab1:
        st.subheader("Battery vs Range Correlation")
        with st.container(border=True):
            if 'Battery Capacity (kWh)' in tdf.columns and 'Range (km)' in tdf.columns:
                fig1 = px.scatter(
                    tdf, 
                    x='Battery Capacity (kWh)', 
                    y='Range (km)', 
                    color='Brand' if 'Brand' in tdf.columns else None,
                    hover_data=tdf.columns
                )
                fig1.update_layout(
                    plot_bgcolor='white', paper_bgcolor='white',
                    font=dict(color='black'),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                fig1.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E5E5E5', zeroline=False)
                fig1.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E5E5E5', zeroline=False)
                st.plotly_chart(fig1, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': False}, theme=None)
            
            st.info("💡 **Observation:** We observe a strong positive linear correlation between battery capacity and range, though efficiency variations between brands cause some spread.")
            
    with tab2:
        st.subheader("Range Frequency Distribution")
        with st.container(border=True):
            if 'Range (km)' in tdf.columns:
                fig2 = px.histogram(
                    tdf, 
                    x='Range (km)', 
                    nbins=24, 
                    color_discrete_sequence=['#000000'],
                    marginal='box', 
                    opacity=0.85
                )
                fig2.update_traces(marker_line_color='#FFFFFF', marker_line_width=1.5)
                fig2.update_layout(
                    plot_bgcolor='white', paper_bgcolor='white',
                    font=dict(color='black'),
                    bargap=0.05
                )
                fig2.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E5E5E5', zeroline=False)
                fig2.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E5E5E5', zeroline=False)
                st.plotly_chart(fig2, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': False}, theme=None)
            
            st.info("💡 **Observation:** The majority of modern EVs cluster around the 350-450 km range mark, with a right-skewed tail representing premium long-range models.")

# --- PAGE: DATASET ---
elif page == "Dataset":
    st.title("🗄️ Dataset Reference")
    st.markdown("Below is a sample of the raw data used to train the machine learning model.")
    st.divider()
    
    # Use native dataframe styling which looks very clean
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )
    
    with st.expander("View Data Dictionary"):
        st.markdown("""
        - **Brand**: The manufacturer of the vehicle.
        - **Battery_Capacity_kWh**: Gross energy storage capacity of the battery pack.
        - **Top_Speed_kmh**: Maximum rated speed.
        - **Range_Km**: Real-world tested driving range.
        - **Efficiency_Whkm**: Energy consumed per kilometer driven.
        """)
