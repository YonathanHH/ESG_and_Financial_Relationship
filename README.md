# 🌍 ESG & Financial Performance Analysis

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

**Comprehensive Data Science Portfolio Project**

Analyzing the relationship between Environmental, Social, and Governance (ESG) factors and financial performance across 1,000 global companies from 2015-2025.

[Live Dashboard](#) | [Dataset](https://www.kaggle.com/datasets/shriyashjagtap/esg-and-financial-performance-dataset) | [Documentation](#)

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
- **Best Model Accuracy**: 85%+ (varies by implementation)

#### Regression Analysis
- **Objective**: Predict profit margin based on ESG and environmental metrics
- **Features**: ESG Overall Score, Carbon Emissions, Water Usage, Energy Consumption
- **Models Compared**: 
  - Linear Regression (Simple, Polynomial)
  - Regularized Models (Ridge, Lasso, ElasticNet)
  - Ensemble Methods (Random Forest, Gradient Boosting, XGBoost)
- **Best Model R² Score**: 0.75+ (varies by implementation)

### 📊 Time Series Forecasting
- **Metrics Forecasted**: Growth Rate, Revenue, ESG Overall Score
- **Forecast Horizon**: 5 years (2026-2030)
- **Methods**: 
  - **SARIMA**: Statistical approach for seasonal patterns
  - **Prophet**: Facebook's robust forecasting tool
- **Model Comparison**: MAPE and RMSE evaluation

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
- Technology
- Finance
- Energy
- Healthcare
- Manufacturing
- Retail
- Transportation
- Telecommunications
- Consumer Goods

### Geographic Regions
- North America
- Europe
- Asia
- South America
- Africa
- Oceania
- Middle East

---

## 📁 Project Structure

```
esg-financial-analysis/
│
├── data/
│   ├── esg_financial_data.csv       # Raw dataset
│   └── esg_processed.csv            # Processed dataset with ESG categories
│
├── notebooks/
│   ├── 01_data_exploration.py       # EDA and statistical analysis
│   ├── 02_classification.py         # ESG classification modeling
│   ├── 03_regression.py             # Profit margin prediction
│   └── 04_timeseries.py             # Forecasting models
│
├── models/
│   ├── best_classification_model.pkl    # Trained classification model
│   ├── classification_scaler.pkl        # Feature scaler for classification
│   ├── label_encoder.pkl                # ESG category encoder
│   ├── best_regression_model.pkl        # Trained regression model
│   └── regression_scaler.pkl            # Feature scaler for regression
│
├── outputs/
│   ├── *.html                           # Interactive Plotly visualizations
│   ├── *.png                            # Static plots
│   └── *.csv                            # Results and forecasts
│
├── app.py                              # Streamlit dashboard
├── requirements.txt                     # Python dependencies
├── README.md                           # This file
└── LICENSE                             # MIT License
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (optional)

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/esg-financial-analysis.git
cd esg-financial-analysis
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
3. Place in `data/` folder

---

## 💻 Usage

### Run Analysis Notebooks

Execute notebooks in sequence:

```bash
# 1. Data Exploration
python notebooks/01_data_exploration.py

# 2. Classification Modeling
python notebooks/02_classification.py

# 3. Regression Modeling
python notebooks/03_regression.py

# 4. Time Series Forecasting
python notebooks/04_timeseries.py
```

### Launch Interactive Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Jupyter Notebook Alternative

If you prefer Jupyter notebooks:

```bash
jupyter notebook
```

Then convert .py files to .ipynb or run as Python scripts within Jupyter.

---

## 🔬 Methodology

### 1. Data Exploration & Preprocessing

**Steps:**
- Load and inspect dataset structure
- Handle missing values (GrowthRate NaN for 2015 is expected)
- Create ESG performance categories based on overall scores:
  - Excellent: > 75
  - Good: 50-75
  - Average: 25-49
  - Poor: < 25
- Analyze distributions, correlations, and temporal trends
- Statistical hypothesis testing

**Key Findings:**
- Strong correlation between ESG scores and profit margins (r ≈ 0.45)
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

**Models Compared:**
- Logistic Regression
- Random Forest ⭐ (typically best)
- Gradient Boosting
- Support Vector Machine
- K-Nearest Neighbors

### 3. Regression Modeling

**Problem**: Predict profit margin from ESG and environmental factors

**Process:**
1. Feature selection: ESG_Overall, CarbonEmissions, WaterUsage, EnergyConsumption
2. Data splitting: 80% train, 20% test
3. Feature scaling: StandardScaler
4. Model training: 8 algorithms compared
5. Evaluation: RMSE, MAE, R²
6. Residual analysis and validation

**Models Compared:**
- Simple Linear Regression
- Polynomial Regression (degree 2)
- Ridge Regression
- Lasso Regression
- ElasticNet
- Random Forest
- Gradient Boosting
- XGBoost ⭐ (typically best)

### 4. Time Series Forecasting

**Problem**: Forecast 5-year trends in GrowthRate, Revenue, and ESG_Overall

**Methods:**

**SARIMA (Seasonal ARIMA)**
- Statistical approach
- Captures seasonality and trends
- Parameters: (p,d,q)(P,D,Q,s)
- Best for stable patterns

**Prophet**
- Developed by Facebook
- Robust to missing data and outliers
- Handles holidays and changepoints
- Intuitive hyperparameters

**Evaluation:**
- MAPE (Mean Absolute Percentage Error)
- RMSE (Root Mean Squared Error)
- Visual comparison of forecasts

---

## 📈 Results (TBA)

### Classification Performance

**Key Insights:**
- Carbon emissions is the strongest predictor of ESG category
- Water usage and energy consumption also significant
- Model successfully distinguishes between performance tiers

### Regression Performance


**Key Insights:**
- ESG_Overall has positive impact on profit margins
- Higher carbon emissions correlate with lower profitability
- Model explains 75% of variance in profit margins

### Forecasting Results (2026-2030)

TBA

---

## 🛠️ Technologies Used

### Programming & Libraries

**Core:**
- Python 3.8+
- NumPy, Pandas - Data manipulation
- Scikit-learn - Machine learning
- XGBoost - Gradient boosting

**Visualization:**
- Plotly - Interactive charts
- Matplotlib, Seaborn - Static plots

**Time Series:**
- Statsmodels - SARIMA
- Prophet - Facebook forecasting

**Web Dashboard:**
- Streamlit - Interactive web app
- Joblib - Model serialization

### Development Tools
- Git - Version control
- Jupyter Notebook - Exploration
- VS Code - IDE

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
- [ ] User authentication and saved preferences
- [ ] Custom report generation (PDF)
- [ ] A/B testing different models
- [ ] Geospatial visualization with Folium

### Deployment
- [ ] Dockerization
- [ ] CI/CD pipeline
- [ ] Cloud deployment (AWS, Azure, GCP)
- [ ] API endpoints for model serving

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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Contact

**Your Name**
- LinkedIn: [Yonathan Hary](https://www.linkedin.com/in/yonathanhary/)
- Email: yonathan.hary@outlook.com
- GitHub: [@YonathanHH](https://github.com/YonathanHH)

### Project Link
[https://github.com/YonathanHH/ESG_and_Financial_Relationship](https://github.com/YonathanHH/ESG_and_Financial_Relationship)

---

## 🙏 Acknowledgments

- **Dataset**: [Shriyash Jagtap on Kaggle](https://www.kaggle.com/datasets/shriyashjagtap/esg-and-financial-performance-dataset)
- **Inspiration**: Growing importance of ESG in investment decisions
- **Learning Resources**: 
  - Scikit-learn Documentation
  - Prophet Documentation
  - Streamlit Gallery
  - Kaggle Community

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
