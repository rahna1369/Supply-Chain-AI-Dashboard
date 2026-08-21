# Supply Chain Risk & Delivery Forecast Analytics

## 📖 Introduction
Global supply chains frequently face bottlenecks, unpredictable port delays, and unexpected operational disruptions. This project provides an end-to-end predictive analytics solution designed to bring proactive visibility to logistics operations. By combining hyperparameter-tuned machine learning models (XGBoost) with deep learning sequence forecasting (PyTorch LSTM), this system predicts expected delivery dates, quantifies disruption risks, and forecasts shipment lead times. The entire intelligence pipeline is embedded within an interactive Streamlit web dashboard to enable logistics operators and planners to identify high-risk shipment routes and intervene before costly bottlenecks occur.

---

## 📌 Key Features

* **Disruption Risk Classification:** Uses tuned XGBoost classification models to predict high-risk shipments with calibrated probability scores.
* **Lead Time & Delay Forecasting:** Combines PyTorch LSTM neural networks and XGBoost regression models to forecast precise delivery delays in days.
* **Interactive Web Dashboard:** Delivers dynamic visual insights built with Streamlit and Plotly, including calendar timelines, fleet risk breakdowns, and port-to-port vulnerability analytics.
* **Lightweight Model Pipeline:** Pre-trained weights and scalers serialized into `.joblib` and `.pth` artifacts for fast cloud execution.

---

## 🛠️ Tech Stack

* **Machine Learning & AI:** XGBoost, PyTorch (LSTM sequence models), Scikit-learn, Optuna
* **Frontend & Dashboards:** Streamlit, Plotly Express
* **Data Pipelines:** Pandas, NumPy
* **Deployment & Control:** GitHub, Streamlit Community Cloud, Joblib

  ---
## 📂 Repository Structure

```text
Supply-Chain-Delay-Prediction/
├── app.py                      # Main Streamlit web application file
├── requirements.txt            # Python environment dependencies
├── supply_chain_models.joblib  # Serialized XGBoost model, scalers, and encoders
├── final_lstm_model.pth        # Saved PyTorch LSTM model weights
├── all_models_predictions.csv  # Evaluated prediction dataset
└── README.md                   # Full project documentation
```
---

## 📖 Conclusion

This project delivers an end-to-end predictive analytics solution for modern logistics management. By combining hyperparameter-tuned XGBoost with deep learning PyTorch LSTM models, the system transforms supply chain management from reactive problem-solving to proactive disruption prevention. The interactive Streamlit dashboard enables logistics planners to identify high-risk shipment routes early, optimize delivery timelines, and eliminate costly operational bottlenecks.






