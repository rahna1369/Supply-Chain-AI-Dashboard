import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
import torch.nn as nn

# ==========================================
# 1. PyTorch Neural Network Architectures
# ==========================================
class TunableRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super(TunableRegressor, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.layer3 = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

class TunableClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super(TunableClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.layer3 = nn.Linear(hidden_dim // 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x

# ==========================================
# 2. Page Configuration & Styling
# ==========================================
st.set_page_config(page_title="Supply Chain AI Dashboard", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #e6edf3; }
    div[data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 16px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. Load Trained Pipeline safely
# ==========================================
MODEL_PATH = 'supply_chain_pipeline.joblib'

@st.cache_resource
def load_pipeline():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading pipeline: {e}")
            return None
    else:
        st.error(f"File '{MODEL_PATH}' not found in: {os.getcwd()}")
        return None

pipeline = load_pipeline()

# ==========================================
# 4. Live Predictor Sidebar
# ==========================================
st.sidebar.header("🔮 Live Shipment Predictor")

if pipeline is not None:
    reg_model = pipeline['reg_model']
    cls_model = pipeline['cls_model']
    preprocessor = pipeline['preprocessor']

    st.sidebar.subheader("Shipment Attributes")

    # Input controls set with exact bounds matching dataset distribution
    carrier_score = st.sidebar.slider("Carrier Reliability Score", 0.50, 1.00, 0.75, step=0.01)
    distance_km = st.sidebar.number_input("Distance (km)", min_value=500.0, max_value=15000.0, value=7750.0, step=50.0)
    weight_mt = st.sidebar.number_input("Weight (MT)", min_value=1.0, max_value=500.0, value=243.5, step=1.0)
    fuel_index = st.sidebar.number_input("Fuel Price Index", min_value=1.00, max_value=5.00, value=2.84, step=0.05)

    weather = st.sidebar.selectbox("Weather Condition", ["Clear", "Rainy", "Stormy", "Foggy"])
    transport_mode = st.sidebar.selectbox("Transport Mode", ["Road", "Rail", "Air", "Sea"])
    product_cat = st.sidebar.selectbox("Product Category", ["Electronics", "Perishables", "Industrial", "Apparel"])

    if st.sidebar.button("Run Predictions"):
        raw_input = {
            'Carrier_Reliability_Score': [carrier_score],
            'Distance_km': [distance_km],
            'Weight_MT': [weight_mt],
            'Fuel_Price_Index': [fuel_index],
            'Weather_Condition': [weather],
            'Transport_Mode': [transport_mode],
            'Product_Category': [product_cat]
        }
        input_df = pd.DataFrame(raw_input)

        # Preprocess features
        processed = preprocessor.transform(input_df)
        if hasattr(processed, "toarray"):
            processed = processed.toarray()

        tensor_in = torch.tensor(processed, dtype=torch.float32)

        reg_model.eval()
        cls_model.eval()

        with torch.no_grad():
            pred_days_float = reg_model(tensor_in).item()
            disrupt_prob = cls_model(tensor_in).item()

        # Format Lead Time into Days & Hours
        days = int(pred_days_float)
        hours = int((pred_days_float - days) * 24)

        st.sidebar.markdown("---")
        st.sidebar.subheader("🎯 Prediction Output")
        st.sidebar.metric("⏱️ Estimated Lead Time", f"{days}d {hours}h ({pred_days_float:.1f} Days)")

        if disrupt_prob >= 0.5:
            st.sidebar.error(f"⚠️ High Disruption Risk ({disrupt_prob * 100:.1f}%)")
        else:
            st.sidebar.success(f"✅ Low Risk / On-Time ({disrupt_prob * 100:.1f}%)")
else:
    st.sidebar.warning("⚠️ Pipeline file not loaded.")

# ==========================================
# 5. Main Dashboard Screen
# ==========================================
st.title("📦 Supply Chain AI Predictive Dashboard")

df = pipeline.get('output_df') if pipeline is not None else None

if df is not None:
    # Key Performance Metrics
    k1, k2, k3, k4 = st.columns(4)
    avg_act = df['Actual_Lead_Time'].mean()
    avg_pred = df['Predicted_Lead_Time'].mean()
    rmse = np.sqrt(((df['Actual_Lead_Time'] - df['Predicted_Lead_Time']) ** 2).mean())
    disruptions = int(df['Actual_Disruption'].sum())

    k1.metric("⏱️ Avg Actual Lead Time", f"{avg_act:.1f} Days")
    k2.metric("🎯 Avg Predicted Lead Time", f"{avg_pred:.1f} Days", delta=f"{avg_pred - avg_act:+.1f}")
    k3.metric("📉 Model RMSE", f"{rmse:.2f} Days")
    k4.metric("🔔 Historical Disruptions", f"{disruptions}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Visualization Scatter Plot
    fig = px.scatter(
        df,
        x='Actual_Lead_Time',
        y='Predicted_Lead_Time',
        color='Predicted_Disruption',
        trendline="ols",
        title="Actual vs Predicted Delivery Lead Time",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Output Prediction Dataset")
    st.dataframe(df, use_container_width=True)
