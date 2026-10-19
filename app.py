# Import required libraries for the Streamlit dashboard
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px

# 1. Page Configuration for Professional Theme
st.set_page_config(
    page_title="Walmart Retail Intelligence Engine",
    page_icon="🛒",
    layout="wide"
)

# 2. Institutional Title & Branding
st.title("🛒 Walmart Retail Analytics & Predictive Sales Forecasting System")
st.markdown("---")


# 3. Securely load the optimized production XGBoost model from the artifacts folder
@st.cache_resource
def load_xgboost_model():
    # Updated path pointing directly to your artifacts directory
    model_path = r"C:\Users\lenovo\OneDrive\Desktop\Walmart_Sales_Forecasting_System\artifacts\walmart_xgboost_model.pkl"
    return joblib.load(model_path)


try:
    model = load_xgboost_model()
except Exception as e:
    st.error(f"Error loading the production model from artifacts. Details: {e}")


# 4. Load the clean dataset for rendering live interactive plots
@st.cache_data
def load_clean_data():
    data_path = r"C:\Users\lenovo\OneDrive\Desktop\Walmart_Sales_Forecasting_System\data\clean\walmart_clean.csv"
    return pd.read_csv(data_path)


try:
    df_clean = load_clean_data()
except:
    df_clean = None

# 5. Create structural Tabs for clean dashboard navigation
tab1, tab2 = st.tabs(["📊 Business Intelligence Dashboard", "🔮 Predictive Forecast Engine"])

# --- TAB 1: BUSINESS ANALYTICS & INSIGHTS (Project B & C Study) ---
with tab1:
    st.header("Retail Performance Analytics & Macroeconomic Study")
    st.markdown(
        "This section analyzes the historical impact of core economic indicators on Walmart's sales performance.")

    if df_clean is not None:
        # Create a two-column grid layout for side-by-side plots
        col_plot1, col_meta2 = st.columns(2)

        with col_plot1:
            st.subheader("Unemployment Rate Impact Matrix")
            fig_unemp = px.scatter(df_clean, x='Unemployment', y='Weekly_Sales', trendline="ols",
                                   title="Weekly Sales Trend Across Unemployment Levels",
                                   color_discrete_sequence=['#1f77b4'], template="plotly_white")
            st.plotly_chart(fig_unemp, use_container_width=True)
            st.info(
                "Analytical Insight: A negative correlation is visible. As regional unemployment metrics escalate, consumer spending limits squeeze down peak store revenues.")

        with col_meta2:
            st.subheader("Consumer Price Index (CPI) Inflation Era Study")
            fig_cpi = px.scatter(df_clean, x='CPI', y='Weekly_Sales', trendline="ols",
                                 title="Weekly Sales Distribution Across CPI Clusters",
                                 color_discrete_sequence=['#2ca02c'], template="plotly_white")
            st.plotly_chart(fig_cpi, use_container_width=True)
            st.info(
                "Analytical Insight: Strict horizontal clustering reveals that Walmart's everyday low-price retail strategy effectively insulates sales performance across distinct macroeconomic inflation eras.")
    else:
        st.warning("Clean dataset baseline file not detected at the data/clean directory.")

# --- TAB 2: INTERACTIVE PRODUCTION FORECAST ENGINE (Project A) ---
with tab2:
    st.header("XGBoost Predictive Forecast Engine")
    st.write(
        "Configure the operational metrics below to simulate real-time demand and generate an institutional sales forecast:")
    st.markdown("<br>", unsafe_allow_html=True)

    # Dual-column responsive input form layout
    col1, col2 = st.columns(2)

    with col1:
        store = st.number_input("Target Store Identification ID", min_value=1, max_value=45, value=1, step=1)
        holiday = st.selectbox("Is it a Scheduled Holiday Week?", options=[0, 1], format_func=lambda
            x: "Yes (Holiday Promotion Sales Active)" if x == 1 else "No (Regular Operational Week)")
        temp = st.slider("Local Environmental Temperature (Fahrenheit)", min_value=-10.0, max_value=110.0, value=60.0,
                         step=0.1)
        fuel = st.slider("Regional Fuel Index Price (per Gallon USD)", min_value=2.0, max_value=5.0, value=3.4,
                         step=0.01)

    with col2:
        cpi = st.slider("Consumer Price Index Value (CPI Metrics)", min_value=120.0, max_value=230.0, value=171.5,
                        step=0.1)
        unemployment = st.slider("Local Area Unemployment Rate (%)", min_value=3.0, max_value=15.0, value=8.0, step=0.1)
        year = st.selectbox("Target Forecasting Calendar Year", options=[2010, 2011, 2012], index=2)
        month = st.selectbox("Target Forecasting Calendar Month", options=list(range(1, 13)), index=5,
                             format_func=lambda x: pd.to_datetime(str(x), format='%m').strftime('%B'))
        week = st.slider("Target Scheduling Week of the Year", min_value=1, max_value=52, value=26)

    st.markdown("---")

    # Center aligned forecasting execution button
    _, btn_col, _ = st.columns([2, 1, 2])
    with btn_col:
        execute_forecast = st.button("🔮 Generate Revenue Forecast", type="primary", use_container_width=True)

    if execute_forecast:
        # Structure the precise features in the chronological order required by the trained XGBoost model
        structured_payload = np.array([[store, holiday, temp, fuel, cpi, unemployment, year, month, week]])

        # Calculate inference output using the cached production model
        sales_prediction = model.predict(structured_payload)

        # Beautifully display the predicted monetary value on screen
        st.markdown("<br>", unsafe_allow_html=True)
        st.success(
            f"### 📈 Projected Revenue Output for Store {store} (Week {week}): **${float(sales_prediction[0]):,.2f} USD**")
