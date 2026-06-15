import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ESG & Financial Performance",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/esg_processed.csv")
    except FileNotFoundError:
        df = pd.read_csv("data/raw_data.csv")
    if "ESG_Category" not in df.columns:
        def categorize(score):
            if score > 75: return "Excellent"
            elif score >= 50: return "Good"
            elif score >= 25: return "Average"
            else: return "Poor"
        df["ESG_Category"] = df["ESG_Overall"].apply(categorize)
    return df

@st.cache_resource
def load_models():
    models = {}
    paths = {
        "clf_model":   "models/best_classification_model.pkl",
        "clf_scaler":  "models/classification_scaler.pkl",
        "label_enc":   "models/label_encoder.pkl",
        "reg_model":   "models/best_regression_model.pkl",
        "reg_scaler":  "models/regression_scaler.pkl",
    }
    for key, path in paths.items():
        if os.path.exists(path):
            models[key] = joblib.load(path)
        else:
            models[key] = None
    return models

df = load_data()
models = load_models()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.image("https://img.shields.io/badge/ESG-Dashboard-2ecc71?style=for-the-badge", use_column_width=True)
st.sidebar.title("🌍 ESG Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "🔍 EDA Explorer", "🤖 ML Predictions", "📈 Forecasts", "📥 Export Data"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔎 Filters")

year_min, year_max = int(df["Year"].min()), int(df["Year"].max())
year_range = st.sidebar.slider("Year Range", year_min, year_max, (year_min, year_max))

industries = ["All"] + sorted(df["Industry"].unique().tolist())
selected_industry = st.sidebar.selectbox("Industry", industries)

regions = ["All"] + sorted(df["Region"].unique().tolist())
selected_region = st.sidebar.selectbox("Region", regions)

# Apply filters
filtered = df[(df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])]
if selected_industry != "All":
    filtered = filtered[filtered["Industry"] == selected_industry]
if selected_region != "All":
    filtered = filtered[filtered["Region"] == selected_region]

st.sidebar.markdown("---")
st.sidebar.caption(f"Showing **{len(filtered):,}** records")
st.sidebar.caption("Made with ❤️ by [Yonathan Hary](https://github.com/YonathanHH)")

# ─────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────
if page == "📊 Overview":
    st.title("🌍 ESG & Financial Performance Dashboard")
    st.markdown("Analyzing the relationship between ESG factors and financial performance across **1,000 global companies (2015–2025)**.")
    st.markdown("---")

    # KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Records", f"{len(filtered):,}")
    col2.metric("Avg ESG Score", f"{filtered['ESG_Overall'].mean():.1f}")
    col3.metric("Avg Profit Margin", f"{filtered['ProfitMargin'].mean():.2f}%")
    col4.metric("Avg Revenue ($M)", f"{filtered['Revenue'].mean():,.0f}")
    col5.metric("Avg Growth Rate", f"{filtered['GrowthRate'].mean():.2f}%")

    st.markdown("---")

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("ESG Score Distribution")
        fig = px.histogram(filtered, x="ESG_Overall", nbins=40,
                           color_discrete_sequence=["#2ecc71"])
        fig.update_layout(height=350, margin=dict(t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("ESG Category Breakdown")
        cat_counts = filtered["ESG_Category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        fig = px.pie(cat_counts, names="Category", values="Count",
                     color="Category",
                     color_discrete_map={
                         "Excellent": "#2ecc71", "Good": "#3498db",
                         "Average": "#f39c12", "Poor": "#e74c3c"
                     },
                     hole=0.4)
        fig.update_layout(height=350, margin=dict(t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col_l2, col_r2 = st.columns(2)

    with col_l2:
        st.subheader("ESG vs Profit Margin")
        fig = px.scatter(filtered, x="ESG_Overall", y="ProfitMargin",
                         color="Industry", opacity=0.5,
                         trendline="ols",
                         color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=380, margin=dict(t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_r2:
        st.subheader("Avg ESG Score by Industry")
        ind_avg = filtered.groupby("Industry")["ESG_Overall"].mean().sort_values(ascending=True).reset_index()
        fig = px.bar(ind_avg, x="ESG_Overall", y="Industry", orientation="h",
                     color="ESG_Overall", color_continuous_scale="Greens")
        fig.update_layout(height=380, margin=dict(t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: EDA EXPLORER
# ─────────────────────────────────────────────
elif page == "🔍 EDA Explorer":
    st.title("🔍 EDA Explorer")
    st.markdown("Interactively explore distributions, correlations, and temporal trends.")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📊 Distributions", "🔗 Correlations", "📅 Trends Over Time"])

    numeric_cols = ["ESG_Overall", "ESG_Environmental", "ESG_Social", "ESG_Governance",
                    "Revenue", "ProfitMargin", "MarketCap", "GrowthRate",
                    "CarbonEmissions", "WaterUsage", "EnergyConsumption"]
    numeric_cols = [c for c in numeric_cols if c in filtered.columns]

    with tab1:
        col_sel = st.selectbox("Select Column", numeric_cols, key="dist_col")
        group_by = st.selectbox("Color By", ["None", "Industry", "Region", "ESG_Category"], key="dist_group")
        color = None if group_by == "None" else group_by
        fig = px.histogram(filtered, x=col_sel, color=color, nbins=50,
                           marginal="box", opacity=0.75,
                           color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        selected_cols = st.multiselect("Select Columns for Correlation", numeric_cols, default=numeric_cols[:6])
        if len(selected_cols) >= 2:
            corr = filtered[selected_cols].corr()
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdYlGn",
                            zmin=-1, zmax=1, title="Correlation Heatmap")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select at least 2 columns.")

    with tab3:
        y_metric = st.selectbox("Metric to Track", numeric_cols, key="trend_metric")
        group_trend = st.selectbox("Group By", ["Overall", "Industry", "Region"], key="trend_group")
        if group_trend == "Overall":
            ts = filtered.groupby("Year")[y_metric].mean().reset_index()
            fig = px.line(ts, x="Year", y=y_metric, markers=True,
                          title=f"Avg {y_metric} Over Time",
                          color_discrete_sequence=["#3498db"])
        else:
            ts = filtered.groupby(["Year", group_trend])[y_metric].mean().reset_index()
            fig = px.line(ts, x="Year", y=y_metric, color=group_trend,
                          markers=True, title=f"Avg {y_metric} Over Time by {group_trend}",
                          color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: ML PREDICTIONS
# ─────────────────────────────────────────────
elif page == "🤖 ML Predictions":
    st.title("🤖 ML Predictions")
    st.markdown("Use trained models to make real-time predictions.")
    st.markdown("---")

    tab_clf, tab_reg = st.tabs(["🏷️ ESG Classifier", "💰 Profit Margin Regressor"])

    with tab_clf:
        st.subheader("Predict ESG Performance Category")
        st.caption("Model: Tuned Random Forest | Features: CarbonEmissions, WaterUsage, EnergyConsumption")

        col1, col2, col3 = st.columns(3)
        carbon = col1.number_input("Carbon Emissions", min_value=0.0, value=float(df["CarbonEmissions"].mean()), step=100.0)
        water  = col2.number_input("Water Usage",      min_value=0.0, value=float(df["WaterUsage"].mean()),      step=100.0)
        energy = col3.number_input("Energy Consumption", min_value=0.0, value=float(df["EnergyConsumption"].mean()), step=100.0)

        if st.button("🔮 Predict ESG Category", type="primary"):
            if models["clf_model"] and models["clf_scaler"] and models["label_enc"]:
                X_input = np.array([[carbon, water, energy]])
                X_scaled = models["clf_scaler"].transform(X_input)
                pred = models["clf_model"].predict(X_scaled)[0]
                proba = models["clf_model"].predict_proba(X_scaled)[0]
                label = models["label_enc"].inverse_transform([pred])[0]
                color_map = {"Excellent": "🟢", "Good": "🔵", "Average": "🟡", "Poor": "🔴"}
                st.success(f"**Predicted ESG Category: {color_map.get(label, '')} {label}**")
                proba_df = pd.DataFrame({
                    "Category": models["label_enc"].classes_,
                    "Probability": proba
                }).sort_values("Probability", ascending=False)
                fig = px.bar(proba_df, x="Category", y="Probability",
                             color="Category",
                             color_discrete_map={"Excellent":"#2ecc71","Good":"#3498db","Average":"#f39c12","Poor":"#e74c3c"},
                             title="Prediction Probabilities")
                fig.update_layout(yaxis_range=[0, 1], height=320)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Classification model not found. Run `02_classification.ipynb` first to train and save the model.")

    with tab_reg:
        st.subheader("Predict Profit Margin")
        st.caption("Model: Tuned XGBoost | Features: ESG_Overall, CarbonEmissions, WaterUsage, EnergyConsumption")

        col1, col2, col3, col4 = st.columns(4)
        esg_score = col1.number_input("ESG Overall Score", 0.0, 100.0, 0.0, key="input_esg_overall")
        carbon_r  = col2.number_input("Carbon Emissions", min_value=0.0, value=0.0, key="input_carbon_emissions")
        water_r   = col3.number_input("Water Usage", min_value=0.0, value=0.0, key="input_water_usage")
        energy_r  = col4.number_input("Energy Consumption", min_value=0.0, value=0.0, key="input_energy_consumption")

        if st.button("💡 Predict Profit Margin", type="primary"):
            if models["reg_model"] and models["reg_scaler"]:
                X_input = np.array([[esg_score, carbon_r, water_r, energy_r]])
                X_scaled = models["reg_scaler"].transform(X_input)
                pred_margin = models["reg_model"].predict(X_scaled)[0]
                avg_margin = df["ProfitMargin"].mean()
                delta = pred_margin - avg_margin
                col_a, col_b = st.columns(2)
                col_a.metric("Predicted Profit Margin", f"{pred_margin:.2f}%", f"{delta:+.2f}% vs avg")
                col_b.metric("Dataset Average",         f"{avg_margin:.2f}%")
            else:
                st.warning("⚠️ Regression model not found. Run `03_regression.ipynb` first to train and save the model.")

# ─────────────────────────────────────────────
# PAGE: FORECASTS
# ─────────────────────────────────────────────
elif page == "📈 Forecasts":
    st.title("📈 ESG & Financial Forecasts (2026–2030)")
    st.markdown("---")

    forecast_path = "data/forecast.csv"
    comparison_path = "outputs/model_comparison_timeseries.csv"

    if os.path.exists(forecast_path):
        forecast_df = pd.read_csv(forecast_path)
        ts_hist = df.groupby("Year")[["GrowthRate","Revenue","ESG_Overall"]].mean().reset_index()

        METRICS = ["GrowthRate", "Revenue", "ESG_Overall"]
        selected_metric = st.selectbox("Select Metric", METRICS)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ts_hist["Year"], y=ts_hist[selected_metric],
            mode="lines+markers", name="Historical",
            line=dict(color="#2c3e50", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df["Year"], y=forecast_df[f"{selected_metric}_SARIMA"],
            mode="lines+markers", name="SARIMA Forecast",
            line=dict(color="#e74c3c", dash="dash", width=2)
        ))
        fig.add_vline(x=2025.5, line_dash="dash", line_color="gray",
                      annotation_text="Forecast Start")
        fig.update_layout(
            title=f"{selected_metric} — Historical & Forecast 2026–2030",
            xaxis_title="Year", yaxis_title=selected_metric, height=480
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Forecast Table (2026–2030)")
        st.dataframe(forecast_df.set_index("Year"), use_container_width=True)

        if os.path.exists(comparison_path):
            st.markdown("---")
            st.subheader("Model Comparison (MAPE & RMSE)")
            comp_df = pd.read_csv(comparison_path)
            st.dataframe(comp_df, use_container_width=True)
    else:
        st.warning("⚠️ Forecast data not found. Run `04_timeseries.ipynb` first to generate forecasts.")
        st.info("Once you run the notebook, `outputs/forecast_results_2026_2030.csv` will appear here automatically.")

# ─────────────────────────────────────────────
# PAGE: EXPORT DATA
# ─────────────────────────────────────────────
elif page == "📥 Export Data":
    st.title("📥 Export Data")
    st.markdown("Download the filtered dataset or model results.")
    st.markdown("---")

    st.subheader("Filtered Dataset")
    st.dataframe(filtered.head(200), use_container_width=True)
    st.caption(f"Previewing first 200 of {len(filtered):,} rows")

    col1, col2, col3 = st.columns(3)
    with col1:
        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", csv, "esg_filtered.csv", "text/csv")
    with col2:
        json_data = filtered.to_json(orient="records").encode("utf-8")
        st.download_button("⬇️ Download JSON", json_data, "esg_filtered.json", "application/json")
    with col3:
        try:
            import io
            buf = io.BytesIO()
            filtered.to_excel(buf, index=False, engine="openpyxl")
            st.download_button("⬇️ Download Excel", buf.getvalue(), "esg_filtered.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception:
            st.info("Install openpyxl for Excel export.")

    st.markdown("---")
    st.subheader("Summary Statistics")
    st.dataframe(filtered.describe().T.style.format("{:.2f}"), use_container_width=True)