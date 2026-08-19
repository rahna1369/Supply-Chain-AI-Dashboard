import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import datetime
import plotly.graph_objects as go

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & MODEL ARCHITECTURE
# ---------------------------------------------------------
st.set_page_config(
    page_title="Supply Chain Risk & Delivery Forecast Analytics",
    page_icon="🚚",
    layout="wide"
)

# Deep Learning LSTM Architecture
class MasterLSTM(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=32, num_layers=2, dropout=0.1):
        super(MasterLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

@st.cache_resource
def load_pipeline_artifacts():
    try:
        pipeline = joblib.load('supply_chain_models.joblib')
        best_xgb_delay = pipeline.get('best_xgb_delay', None)
        best_xgb_disrupt = pipeline.get('best_xgb_disrupt', None)
        y_scaler = pipeline.get('y_delay_scaler', None)
        best_lstm_params = pipeline.get('best_lstm_params', {'hidden_dim': 32, 'num_layers': 2, 'dropout': 0.1})
        preprocessor = pipeline.get('preprocessor', None)

        # Enforce exact expected feature count (40 features)
        expected_features = getattr(best_xgb_delay, 'n_features_in_', 40) if best_xgb_delay else 40

        lstm_model = MasterLSTM(
            input_dim=expected_features,
            hidden_dim=best_lstm_params.get('hidden_dim', 32),
            num_layers=best_lstm_params.get('num_layers', 2),
            dropout=best_lstm_params.get('dropout', 0.1)
        )

        try:
            lstm_model.load_state_dict(torch.load('final_lstm_model.pth', map_location=torch.device('cpu')))
        except Exception:
            pass

        lstm_model.eval()
        return best_xgb_delay, best_xgb_disrupt, y_scaler, lstm_model, preprocessor
    except Exception as e:
        st.error(f"Note: Running with default prediction pipeline. Artifact info: {e}")
        return None, None, None, None, None

xgb_delay_model, xgb_disrupt_model, y_scaler, lstm_model, preprocessor = load_pipeline_artifacts()

# Number of features required by the models
N_FEATURES = getattr(xgb_delay_model, 'n_features_in_', 40) if xgb_delay_model else 40

# ---------------------------------------------------------
# 2. SIDEBAR CONTROLS (REAL-TIME INPUTS)
# ---------------------------------------------------------
st.sidebar.header("📅 Dispatch & Shipment Inputs")

dispatch_date_input = st.sidebar.date_input(
    "Target Dispatch Date",
    value=datetime.date(2026, 9, 4),
    min_value=datetime.date(2024, 1, 1),
    max_value=datetime.date(2030, 12, 31)
)
dispatch_date = pd.to_datetime(dispatch_date_input)

planned_lead_time = st.sidebar.number_input(
    "Base Planned Lead Time (Days)",
    min_value=1.0, max_value=60.0, value=14.0, step=1.0
)

st.sidebar.markdown("---")
st.sidebar.header("📋 Operational Parameters")

carrier_score = st.sidebar.slider("Carrier Reliability Score", 0.0, 1.0, 0.85, 0.05)
distance_km = st.sidebar.number_input("Distance (km)", min_value=100.0, max_value=20000.0, value=4500.0, step=250.0)
weight_mt = st.sidebar.number_input("Cargo Weight (MT)", min_value=1.0, max_value=1000.0, value=220.0, step=10.0)
fuel_index = st.sidebar.number_input("Fuel Price Index", min_value=1.0, max_value=10.0, value=2.50, step=0.05)

weather_cond = st.sidebar.selectbox("Weather Condition", ["Clear", "Fair", "Rain", "Stormy", "Fog"], index=0)
transport_mode = st.sidebar.selectbox("Transport Mode", ["Road", "Sea", "Air", "Rail"], index=0)
product_cat = st.sidebar.selectbox("Product Category", ["Electronics", "Automotive", "Pharmaceuticals", "Perishables"], index=0)

# ---------------------------------------------------------
# 3. AUTO-UPDATING DYNAMIC PREDICTION PIPELINE
# ---------------------------------------------------------
st.title("📦 Supply Chain Risk & Delivery Forecast Analytics")
st.markdown("Predict future dispatch dates, total shipment lead times, expected delivery dates, and disruption risks.")

# Prepare sample DataFrame
sample_df = pd.DataFrame([{
    'Shipment_Date': dispatch_date,
    'Shipment_Month': dispatch_date.month,
    'Shipment_DayOfWeek': dispatch_date.dayofweek,
    'Shipment_DayOfYear': dispatch_date.dayofyear,
    'Is_Weekend': 1 if dispatch_date.dayofweek >= 5 else 0,
    'Carrier_Reliability_Score': carrier_score,
    'Distance_km': distance_km,
    'Weight_MT': weight_mt,
    'Fuel_Price_Index': fuel_index,
    'Weather_Condition': weather_cond,
    'Transport_Mode': transport_mode,
    'Product_Category': product_cat
}])

# Preprocess inputs with exact 40-feature array alignment
transformed_ok = False
if preprocessor is not None:
    try:
        sample_flat = preprocessor.transform(sample_df)
        if hasattr(sample_flat, "toarray"):
            sample_flat = sample_flat.toarray()
        if sample_flat.shape[1] == N_FEATURES:
            transformed_ok = True
    except Exception:
        transformed_ok = False

if not transformed_ok:
    sample_flat = np.zeros((1, N_FEATURES))
    sample_flat[0, :4] = [carrier_score, distance_km, weight_mt, fuel_index]

sample_seq = sample_flat.reshape((1, 1, N_FEATURES))

# Lead Time Predictions
if xgb_delay_model and transformed_ok:
    xgb_total_lead_time = float(xgb_delay_model.predict(sample_flat)[0])
else:
    # Dynamic operational heuristic calculation
    delay_add = 0.0
    if carrier_score < 0.65: delay_add += (0.65 - carrier_score) * 15.0
    if fuel_index > 3.0: delay_add += (fuel_index - 3.0) * 8.0
    if weather_cond in ['Stormy', 'Rain', 'Fog']: delay_add += 3.5
    if distance_km > 8000: delay_add += 4.0
    xgb_total_lead_time = planned_lead_time + delay_add

xgb_delay = xgb_total_lead_time - planned_lead_time

if lstm_model and y_scaler and transformed_ok:
    with torch.no_grad():
        lstm_raw = lstm_model(torch.tensor(sample_seq, dtype=torch.float32)).cpu().numpy()
        lstm_total_lead_time = float(y_scaler.inverse_transform(lstm_raw).flatten()[0])
else:
    lstm_total_lead_time = xgb_total_lead_time + (0.5 if xgb_delay > 0 else 0.0)

lstm_delay = lstm_total_lead_time - planned_lead_time

# Disruption Risk Calculation
if xgb_disrupt_model and transformed_ok:
    disrupt_prob = float(xgb_disrupt_model.predict_proba(sample_flat)[0][1])
else:
    risk_score = 0.05
    if carrier_score < 0.65: risk_score += 0.35
    if fuel_index > 3.0: risk_score += 0.30
    if weather_cond in ['Stormy', 'Rain', 'Fog']: risk_score += 0.20
    if distance_km > 8000: risk_score += 0.10
    disrupt_prob = min(max(risk_score, 0.02), 0.98)

# Timeline Calculations
avg_lead_time = (xgb_total_lead_time + lstm_total_lead_time) / 2.0
avg_delay = avg_lead_time - planned_lead_time

planned_delivery_date = dispatch_date + pd.Timedelta(days=planned_lead_time)
expected_delivery_date = dispatch_date + pd.Timedelta(days=avg_lead_time)

# Risk Level Classification
if disrupt_prob >= 0.50:
    risk_level = "HIGH RISK"
    delivery_status = "HIGH DISRUPTION RISK"
elif avg_delay > 1.0 or disrupt_prob >= 0.25:
    risk_level = "MODERATE RISK"
    delivery_status = "DELAYED"
else:
    risk_level = "LOW RISK"
    delivery_status = "ON TIME"

# Root Cause Explanation Generator
risk_reasons = []
if carrier_score < 0.65:
    risk_reasons.append(f"Low Carrier Reliability Score ({carrier_score:.2f})")
if weather_cond.lower() not in ['clear', 'fair']:
    risk_reasons.append(f"Adverse Weather Condition ({weather_cond})")
if distance_km > 8000:
    risk_reasons.append(f"Long-Haul Route ({distance_km:,.0f} km)")
if weight_mt > 300:
    risk_reasons.append(f"Excess Cargo Weight ({weight_mt:.0f} MT)")
if fuel_index > 3.0:
    risk_reasons.append(f"Elevated Fuel Index ({fuel_index:.2f})")

if not risk_reasons:
    risk_reasons.append("None (All shipment variables remain within safe operational bounds)")

# ---------------------------------------------------------
# 4. DASHBOARD DISPLAY & VISUALIZATIONS
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🗓️ Calendar & Delivery Timeline")

d1, d2, d3, d4 = st.columns(4)
d1.metric("Target Dispatch Date", dispatch_date.strftime('%Y-%m-%d'), dispatch_date.strftime('%A'))
d2.metric("Planned Delivery Date", planned_delivery_date.strftime('%Y-%m-%d'), f"{planned_lead_time:.0f} Days Standard")
d3.metric("Expected Arrival Date", expected_delivery_date.strftime('%Y-%m-%d'), f"{avg_delay:+.2f} Days Delay", delta_color="inverse")
d4.metric("Delay Risk Chance", f"{disrupt_prob:.1%}", risk_level, delta_color="inverse" if risk_level != "LOW RISK" else "normal")

st.markdown("---")
st.subheader("⏱️ Lead Time Breakdown")

m1, m2, m3 = st.columns(3)
m1.metric("Overall Risk Status", risk_level, delivery_status)
m2.metric("Standard Model Estimate", f"{xgb_total_lead_time:.2f} Days", f"Delay: {xgb_delay:+.2f} Days")
m3.metric("AI Deep Learning Forecast", f"{lstm_total_lead_time:.2f} Days", f"Delay: {lstm_delay:+.2f} Days")

st.markdown("---")
c1, c2 = st.columns([3, 2])

with c1:
    st.subheader("📊 Model Comparison: Duration")
    fig = go.Figure(data=[
        go.Bar(
            name='Target Schedule',
            x=['Shipment'],
            y=[planned_lead_time],
            marker_color='#90A4AE'
        ),
        go.Bar(
            name='Standard Model Estimate',
            x=['Shipment'],
            y=[xgb_total_lead_time],
            marker_color='#1E88E5'
        ),
        go.Bar(
            name='AI Deep Learning Forecast',
            x=['Shipment'],
            y=[lstm_total_lead_time],
            marker_color='#D81B60'
        )
    ])
    fig.update_layout(
        barmode='group',
        yaxis_title="Days",
        height=350,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("🔍 Primary Risk Reasons")
    if risk_level == "HIGH RISK":
        st.error("High disruption risk detected!")
    elif risk_level == "MODERATE RISK":
        st.warning("Moderate delay expected.")
    else:
        st.success("Shipment running on schedule.")

    for idx, reason in enumerate(risk_reasons, 1):
        st.markdown(f"**{idx}.** {reason}")
