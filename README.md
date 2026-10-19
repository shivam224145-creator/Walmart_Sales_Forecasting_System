# Walmart Retail Analytics & Predictive Sales Forecasting System

An end-to-end retail intelligence data product that analyzes the impact of seasonal holidays and macroeconomic indicators (Unemployment Rate, Consumer Price Index) on Walmart's retail performance. The system deploys a high-accuracy predictive engine to forecast future weekly store sales.

https://walmartsalesforecastingsystem-rgihxe4ze5cw9iexj7x3rw.streamlit.app/

## 📊 Project Overview & Architecture
This project consolidates three major analytical tracks into a single unified business solution:
1. **Weekly Sales Prediction Engine:** Forecasts precise monetary revenue metrics per store.
2. **Holiday Sales Impact Analysis:** Visualizes and measures the percentage revenue spikes during promotional seasonal holidays.
3. **Economic Indicator Impact Study:** Evaluates how macro-level shifts in inflation (CPI) and employment metrics alter consumer purchasing behavior.

### 🏗️ Data Pipeline Architecture
```text
[Raw Data: Walmart_Sales.csv] 
              │
              ▼
[Data Cleaning & Datetime Parsing] ──> Saved to /data/processed/
              │
              ▼
[Feature Engineering: Extraction of Year, Month, Week] ──> Saved to /data/clean/
              │
              ▼
[Advanced Model Training & Hyperparameter Tuning] ──> Saved to /artifacts/
              │
              └─► Approach 1: Live Interactive Streamlit Dashboard UI
              │
              └─► Approach 2: High-Performance FastAPI Backend Engine (Swagger UI)
```

---

## 🛠️ Repository & File Structure
```text
Walmart_Sales_Forecasting_System/
│
├── artifacts/
│   ├── walmart_xgboost_model.pkl       # Optimized Production Model (97.79% R²)
│   └── walmart_linear_model.pkl        # Baseline Exploratory Model (15.55% R²)
│
├── data/
│   ├── raw/
│   │   └── Walmart_Sales.csv           # Original unaltered benchmark dataset
│   ├── processed/
│   │   └── walmart_processed.csv       # Dataset with sanitized datetime indexing
│   └── clean/
│       └── walmart_clean.csv           # Final dataset with newly engineered features
│
├── walmart_analysis.ipynb              # Step-by-step Jupyter Notebook lifecycle file
├── app.py                              # Approach 1: Streamlit Dashboard Web UI Script
├── main.py                             # Approach 2: FastAPI Backend Engine Script
└── requirements.txt                    # Project environment dependencies matrix
```

---

## 🔬 Model Performance & Selection Evaluation

The system was evaluated using baseline algorithms before transitioning to advanced tree-based boosting methods to secure optimal industrial precision.

| Machine Learning Model | Model Accuracy (R² Score) | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) |
| :--- | :---: | :---: | :---: |
| **Linear Regression** *(Baseline)* | 15.55% | \$432,594.98 | \$521,583.50 |
| **Random Forest Regressor** | 95.94% | \$62,032.85 | \$114,300.25 |
| **Optimized XGBoost Regressor** *(Tuned)* | **97.79%** | **\$49,635.36** | **\$84,313.03** |

### 💡 Core Analytical Insights (Why XGBoost Won?)
* **Non-Linear Dynamics:** Linear Regression struggled because relationships between sales and external metrics like CPI and Temperature are highly complex and non-linear.
* **Sequential Learning Optimization:** By tuning tree depth and scale parameters (`n_estimators=300`), the XGBoost engine isolated minor mathematical residuals left by individual decision trees, dropping the average forecast error down below **\$50K**.

---

## 🚀 Dual-Deployment Delivery Approaches

### 🎈 Approach 1: Interactive Streamlit Dashboard UI
An intuitive business intelligence dashboard built to present live visual analytics graphs alongside an input interface for corporate stakeholders.
* **Local Run Command:**
  ```bash
  pip install -r requirements.txt
  streamlit run app.py
  ```

### ⚡ Approach 2: High-Performance FastAPI Backend
An enterprise-grade, lightweight REST API engine utilizing Pydantic schemas for automated payload data validation. It comes out-of-the-box with an interactive **Swagger UI** testing environment.
* **Local Run Command:**
  ```bash
  uvicorn main:app --reload
  ```
* **Interactive Testing URL:** Open `http://127.0.0` in your browser to access the live API execution panel.

---

## 👨‍💻 Developer Profile

**SHIVAM SHUKLA**  
*Data Science & Retail Analytics Specialist*

I build data-driven systems designed to bridge the gap between complex machine learning architectures and core business decision-making. Specialized in structuring highly optimized, end-to-end machine learning pipelines from raw data parsing to robust multi-tier cloud deployments.

* **GitHub:** https://github.com/shivam224145-creator/
* **LinkedIn:** https://www.linkedin.com/in/shivam-shukla-a50006161/
* **Streamlit Portfolio Website:** https://share.streamlit.io/user/shivam224145-creator

---
### 🛠️ Tech Stack & Toolkit Matrix
* **Languages:** Python (Pandas, NumPy, Scikit-Learn, XGBoost, Joblib)
* **Web Frameworks:** FastAPI, Streamlit, Pydantic, Uvicorn
* **Data Visualization:** Plotly Express, Seaborn, Matplotlib
