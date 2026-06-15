# 🌍 ESG & Financial Performance Analysis
<img width="776" height="595" alt="esg" src="https://github.com/user-attachments/assets/15ab4196-6939-4931-921e-ced2741ea762" />

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

**Comprehensive Data Science Portfolio Project**

Analyzing the relationship between Environmental, Social, and Governance (ESG) factors and financial performance across 1,000 global companies from 2015-2025.

[Dataset](https://www.kaggle.com/datasets/shriyashjagtap/esg-and-financial-performance-dataset) | [Streamlit Dashboard](http://esgfin-relation.streamlit.app/)

</div>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Dataset Description](#-dataset-description)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Methodology](#-methodology)
- [Results](#-results)
- [Technologies Used](#-technologies-used)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 Project Overview

This project explores the critical relationship between sustainability practices and financial success in the modern business landscape. By analyzing ESG (Environmental, Social, and Governance) metrics alongside financial performance indicators, we uncover insights that help organizations balance profitability with environmental responsibility.

### Business Questions Addressed

1. **Do higher ESG scores correlate with better financial performance?**
2. **Can we predict a company's ESG category based on environmental metrics?**
3. **How do sustainability factors influence profit margins?**
4. **What are the future trends in ESG and financial growth?**

---

## ✨ Key Features

### 🔍 Exploratory Data Analysis (EDA)
- Comprehensive statistical analysis of ESG and financial metrics
- Distribution analysis and outlier detection
- Correlation heatmaps and relationship visualization
- Industry and regional performance comparison
- Temporal trend analysis (2015-2025)
- Statistical significance testing (t-tests, ANOVA, correlation tests)

### 🤖 Machine Learning Models

#### Multi-Class Classification
- **Objective**: Predict ESG performance category (Excellent, Good, Average, Poor)
- **Features**: Carbon Emissions, Water Usage, Energy Consumption
- **Models Tested**: Logistic Regression, Random Forest, Gradient Boosting, SVM, KNN
- **Best Model**: Random Forest (with GridSearchCV tuning)

#### Regression Analysis
- **Objective**: Predict profit margin based on ESG and environmental metrics
- **Features**: ESG Overall Score, Carbon Emissions, Water Usage, Energy Consumption
- **Models Compared**:
  - Linear Regression (Simple, Polynomial)
  - Regularized Models (Ridge, Lasso, ElasticNet)
  - Ensemble Methods (Random Forest, Gradient Boosting, XGBoost)
- **Best Model**: Random Forest Regressor

### 📊 Time Series Forecasting
- **Metrics Forecasted**: Growth Rate, Revenue, ESG Overall Score
- **Forecast Horizon**: 5 years (2026-2030)
- **Method**: SARIMA (statsmodels)
- **Evaluation**: MAPE and RMSE

### 🎨 Interactive Dashboard
- Real-time data filtering by year, industry, and region
- Interactive visualizations with Plotly
- ML model predictions interface
- Downloadable reports (CSV, Excel, JSON)
- Responsive design for all devices

---

## 📊 Dataset Description

**Source**: [Kaggle - ESG and Financial Performance Dataset](https://www.kaggle.com/datasets/shriyashjagtap/esg-and-financial-performance-dataset)

**Overview**: Simulated financial and ESG performance data for 1,000 global companies across 9 industries and 7 regions from 2015-2025.

### Key Features

| Category | Features |
|----------|----------|
| **Identifiers** | CompanyID, CompanyName, Industry, Region, Year |
| **Financial Metrics** | Revenue, ProfitMargin, MarketCap, GrowthRate |
| **ESG Scores** | ESG_Overall, ESG_Environmental, ESG_Social, ESG_Governance |
| **Environmental** | CarbonEmissions, WaterUsage, EnergyConsumption |

### Industries Covered
- Technology, Finance, Energy, Healthcare, Manufacturing
- Retail, Transportation, Telecommunications, Consumer Goods

### Geographic Regions
- North America, Europe, Asia, South America, Africa, Oceania, Middle East

---

## 📁 Project Structure

```
ESG_and_Financial_Relationship/
│
├── 01_EDA.ipynb                         # EDA and statistical analysis
├── 02_classification.ipynb              # ESG classification modeling
├── 03_regression.ipynb                  # Profit margin prediction
├── 04_timeseries.ipynb                  # SARIMA forecasting
│
├── app.py                               # Streamlit dashboard
├── requirements.txt                     # Python dependencies
├── README.md                            # This file
│
├── data/
│   ├── raw_data.csv                     # Raw dataset
│   └── esg_processed.csv               # Processed dataset with ESG categories
│
├── models/
│   ├── best_classification_model.pkl    # Trained Random Forest classifier
│   ├── classification_scaler.pkl        # Feature scaler for classification
│   ├── label_encoder.pkl                # ESG category encoder
│   ├── best_regression_model.pkl        # Trained Random Forest regressor
│   └── regression_scaler.pkl            # Feature scaler for regression
│
└── outputs/
    ├── forecast_results_2026_2030.csv   # SARIMA forecast results
    ├── model_comparison_timeseries.csv  # Time series model metrics
    ├── regression_results.csv           # Regression model comparison
    ├── classification_results.csv       # Classification model comparison
    └── *.png / *.html                   # Plots and interactive charts
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone Repository

```bash
git clone https://github.com/YonathanHH/ESG_and_Financial_Relationship.git
cd ESG_and_Financial_Relationship
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset

1. Visit [Kaggle Dataset Page](https://www.kaggle.com/datasets/shriyashjagtap/esg-and-financial-performance-dataset)
2. Download `esg_financial_data.csv`
3. Place in the `data/` folder

---

## 💻 Usage

### Run Notebooks in Sequence

```bash
# 1. Data Exploration
jupyter notebook 01_EDA.ipynb

# 2. Classification Modeling
jupyter notebook 02_classification.ipynb

# 3. Regression Modeling
jupyter notebook 03_regression.ipynb

# 4. Time Series Forecasting
jupyter notebook 04_timeseries.ipynb
```

### Launch Interactive Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

---

## 🔬 Methodology

### 1. Data Exploration & Preprocessing

**Steps:**
- Load and inspect dataset structure
- Handle missing values (GrowthRate NaN for 2015 is expected)
- Create ESG performance categories based on overall scores:
  - Excellent: > 75
  - Good: 50–75
  - Average: 25–49
  - Poor: < 25
- Analyze distributions, correlations, and temporal trends
- Statistical hypothesis testing

**Key Findings:**
- Moderate positive correlation between ESG scores and profit margins
- Technology and Finance sectors lead in ESG performance
- Upward trend in ESG scores from 2015 to 2025
- Regional variations in sustainability practices

### 2. Classification Modeling

**Problem**: Predict ESG category from environmental footprint

**Process:**
1. Feature selection: CarbonEmissions, WaterUsage, EnergyConsumption
2. Data splitting: 80% train, 20% test (stratified)
3. Feature scaling: StandardScaler
4. Model training: 5 algorithms tested
5. Hyperparameter tuning: GridSearchCV on best model
6. Evaluation: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

**Best Model: Random Forest** (tuned via GridSearchCV)

### 3. Regression Modeling

**Problem**: Predict profit margin from ESG and environmental factors

**Process:**
1. Feature selection: ESG_Overall, CarbonEmissions, WaterUsage, EnergyConsumption
2. Data splitting: 80% train, 20% test
3. Feature scaling: StandardScaler
4. Model training: 8 algorithms compared
5. Evaluation: RMSE, MAE, R²

**Best Model: Random Forest Regressor**

### 4. Time Series Forecasting

**Problem**: Forecast 5-year trends in GrowthRate, Revenue, and ESG_Overall

**Method: SARIMA**
- Annual data aggregated (mean) across all companies per year
- Stationarity tested with ADF test
- ACF/PACF plots used to determine SARIMA orders
- Trained on 2015–2022, evaluated on 2023–2025, forecast to 2030
- Evaluated with MAPE and RMSE

---

## 📈 Results

### Classification Performance

**Best Model: Tuned Random Forest**

**Key Insights:**
- Carbon emissions is the strongest predictor of ESG category
- Water usage and energy consumption are also significant
- Model successfully distinguishes between performance tiers

### Regression Performance

**Best Model: Random Forest Regressor**

| Metric | Score |
|--------|-------|
| R² Score | 0.1475 |
| RMSE | 8.2383 |
| MAE | 6.2218 |

**Key Insights:**
- ESG_Overall shows a positive direction on profit margins
- Higher carbon emissions tend to correlate with lower profitability
- Weak R² suggests profit margin is influenced by many factors beyond ESG metrics alone

### Forecasting Results (2026–2030)

**Model: SARIMA**

| Metric | MAPE |
|--------|------|
| GrowthRate | 4.35% |
| Revenue | 1.00% |
| ESG_Overall | 0.16% |

**Key Insights:**
- Revenue and ESG_Overall forecasts are highly accurate (low MAPE)
- GrowthRate is more volatile but still within acceptable error margins
- ESG scores show a continued upward trend through 2030

---

## 🛠️ Technologies Used

### Programming & Libraries

**Core:**
- Python 3.8+
- NumPy, Pandas — Data manipulation
- Scikit-learn — Machine learning
- XGBoost — Gradient boosting

**Visualization:**
- Plotly — Interactive charts
- Matplotlib, Seaborn — Static plots

**Time Series:**
- Statsmodels — SARIMA

**Web Dashboard:**
- Streamlit — Interactive web app
- Joblib — Model serialization

### Development Tools
- Git & GitHub — Version control
- Jupyter Notebook — Analysis and exploration
- VS Code — IDE

---

## 🔮 Future Improvements

### Model Enhancements
- [ ] Deep learning models (LSTM, Transformer for time series)
- [ ] SHAP values for model interpretability
- [ ] Ensemble stacking for improved predictions
- [ ] Automated hyperparameter tuning with Optuna

### Data Additions
- [ ] Real-world ESG data from MSCI or Refinitiv
- [ ] News sentiment analysis integration
- [ ] Regulatory compliance scores
- [ ] Supply chain sustainability metrics

### Dashboard Features
- [ ] Real-time data updates
- [ ] Custom report generation (PDF)
- [ ] Geospatial visualization with Folium

### Deployment
- [ ] Dockerization
- [ ] CI/CD pipeline
- [ ] Cloud deployment (AWS / Streamlit Cloud)
- [ ] API endpoints for model serving (FastAPI)

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add docstrings to functions
- Include unit tests for new features
- Update documentation as needed

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Contact

**Yonathan Hary**
- LinkedIn: [Yonathan Hary](https://www.linkedin.com/in/yonathanhary/)
- Email: yonathan.hary@outlook.com
- GitHub: [@YonathanHH](https://github.com/YonathanHH)

### Project Link
[https://github.com/YonathanHH/ESG_and_Financial_Relationship](https://github.com/YonathanHH/ESG_and_Financial_Relationship)

---

## 🙏 Acknowledgments

- **Dataset**: [Shriyash Jagtap on Kaggle](https://www.kaggle.com/datasets/shriyashjagtap/esg-and-financial-performance-dataset)
- **Inspiration**: Growing importance of ESG in investment decisions
- **Learning Resources**: Scikit-learn Docs, Statsmodels Docs, Streamlit Gallery, Kaggle Community

---

## 📚 References

1. Friede, G., Busch, T., & Bassen, A. (2015). ESG and financial performance: aggregated evidence from more than 2000 empirical studies. *Journal of Sustainable Finance & Investment*, 5(4), 210-233.

2. Serafeim, G., & Yoon, A. (2022). Stock price reactions to ESG news: The role of ESG ratings and disagreement. *Review of Accounting Studies*.

3. Giese, G., Lee, L. E., Melas, D., Nagy, Z., & Nishikawa, L. (2019). Foundations of ESG investing: How ESG affects equity valuation, risk, and performance. *The Journal of Portfolio Management*, 45(5), 69-83.

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

Made with ❤️ for Data Science and Sustainability

</div>
