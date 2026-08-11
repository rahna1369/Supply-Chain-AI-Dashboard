import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# 1. Page Layout Setup
st.set_page_config(page_title="Supply Chain AI Dashboard", page_icon="📦", layout="wide")
st.title("📦 Supply Chain AI Predictive Dashboard")

# 2. Data Loader
@st.cache_data
def load_data():
    return pd.read_csv('tuned_deep_learning_supply_chain_output.csv')

try:
    df = load_data()

    # 3. KPI Cards Section
    st.subheader("Key Metrics")
    k1, k2, k3, k4 = st.columns(4)

    avg_act = df['Actual_Lead_Time'].mean()
    avg_pred = df['Predicted_Lead_Time'].mean()
    rmse = np.sqrt(((df['Actual_Lead_Time'] - df['Predicted_Lead_Time']) ** 2).mean())
    disruptions = int(df['Actual_Disruption'].sum())

    k1.metric("Avg Actual Lead Time", f"{avg_act:.2f} Days")
    k2.metric("Avg Predicted Lead Time", f"{avg_pred:.2f} Days", delta=f"{avg_pred - avg_act:.2f}")
    k3.metric("Model Error (RMSE)", f"{rmse:.2f} Days")
    k4.metric("Total Disruptions", f"{disruptions}")

    st.markdown("---")

    # 4. Interactive Visualizations
    st.subheader("Visual Analytics")
    fig = px.scatter(
        df,
        x='Actual_Lead_Time',
        y='Predicted_Lead_Time',
        trendline="ols",
        title="Actual vs Predicted Lead Time (Days)"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # 5. Raw Data Table
    st.subheader("Raw Predictions")
    st.dataframe(df, use_container_width=True)

except FileNotFoundError:
    st.error("⚠️ 'tuned_deep_learning_supply_chain_output.csv' not found. Make sure the CSV file is in the exact same folder as app.py.")
