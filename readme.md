# 🚀 Swiggy Delivery Time Prediction

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/scikit--learn-1.3+-orange.svg" alt="sklearn">
  <img src="https://img.shields.io/badge/MLflow-2.8+-purple.svg" alt="MLflow">
  <img src="https://img.shields.io/badge/Pipeline-Architecture-green.svg" alt="Pipeline">
</p>

> **Production-grade ML pipeline** for predicting food delivery times using **method chaining**, **type-hinted functional programming**, and **experiment tracking**. Built with modern MLOps practices.

---

## ✨ What Makes This Different

| Feature | Implementation | Impact |
|---------|---------------|--------|
| **🔧 Method Chaining** | `df.pipe(clean).pipe(feature_engineer).pipe(model)` | Readable, testable, no intermediate variables |
| **📝 Type Hints** | `def clean_data(data: pd.DataFrame) -> pd.DataFrame` | Self-documenting, IDE-friendly, fewer bugs |
| **⚡ Functional Pipelines** | Pure functions, no side effects | Reproducible, unit-testable components |
| **🎯 Missing Value Intelligence** | `MissingIndicator` + KNN imputation | Captures missingness patterns as features |
| **📊 MLflow Integration** | Experiment tracking & model registry | Production MLOps ready |

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Raw Data      │────▶│  Data Cleaning   │────▶│  Feature Eng  │
│  (45,593 rows)  │     │  (Method Chain)  │     │  (Haversine +  │
└─────────────────┘     └──────────────────┘     │   Time Features)│
                                                 └─────────────────┘
                                                          │
                              ┌───────────────────────────┘
                              ▼
                    ┌──────────────────┐
                    │  Preprocessing   │
                    │  Pipeline        │
                    │                  │
                    │  ┌────────────┐  │
                    │  │ Simple     │  │  ◄── Mode/Missing imputation
                    │  │ Imputer    │  │
                    │  └────────────┘  │
                    │        │          │
                    │  ┌────────────┐  │
                    │  │ Column     │  │  ◄── OneHot + Ordinal encoding
                    │  │ Transformer│  │
                    │  └────────────┘  │
                    │        │          │
                    │  ┌────────────┐  │
                    │  │ KNN        │  │  ◄── Distance-based imputation
                    │  │ Imputer    │  │
                    │  └────────────┘  │
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Model Training  │
                    │  (RF/XGB/Light)  │
                    └──────────────────┘
```

---

## 🧹 Data Cleaning Pipeline (Method Chaining)

```python
# No messy intermediate variables. Pure flow.
cleaned_df = (
    df
    .pipe(drop_anomalies)           # Remove 38 minors + 53 six-star ratings
    .pipe(handle_hidden_nans)        # "NaN " → np.nan (8,515 values!)
    .pipe(clean_coordinates)        # Absolute values + threshold filtering
    .pipe(extract_datetime_features) # order_date → day/month/weekend/time_of_day
    .pipe(calculate_haversine)       # Restaurant ↔ Delivery distance
    .pipe(rename_columns)           # snake_case, descriptive names
)
```

**Key Insight:** Discovered `"NaN "` (with trailing space) as hidden missing values — a classic real-world data quality issue.

---

## 🔬 Exploratory Data Analysis

### Statistical Rigor
- **Chi-squared tests** for categorical associations (`festival` ↔ `traffic`: p < 0.001)
- **ANOVA** for numerical-categorical relationships
- **Jarque-Bera** normality testing on target variable

### Feature Engineering
| Feature | Method | Business Logic |
|---------|--------|----------------|
| `distance` | Haversine formula | Actual delivery distance in km |
| `pickup_time_minutes` | `order_picked - order_time` | Restaurant preparation time |
| `distance_type` | `pd.cut()` bins | Ordinal: short → very_long |
| `is_weekend` | `dt.day_name().isin([Sat,Sun])` | Weekend demand patterns |
| `order_time_of_day` | Custom `np.select()` | Morning/Afternoon/Evening/Night |

---

## ⚙️ Preprocessing Pipeline

```python
processing_pipeline = Pipeline([
    ("simple_imputer", ColumnTransformer([
        ("mode_imputer", SimpleImputer(strategy="most_frequent", add_indicator=True), 
         ['multiple_deliveries', 'festival', 'city_type']),
        ("missing_imputer", SimpleImputer(strategy="constant", fill_value="missing", add_indicator=True),
         ['weather', 'type_of_order', 'type_of_vehicle', 'is_weekend', 'order_time_of_day'])
    ], remainder="passthrough")),
    
    ("preprocess", ColumnTransformer([
        ("scale", MinMaxScaler(), num_cols),
        ("nominal_encode", OneHotEncoder(drop="first", sparse_output=False), nominal_cat_cols),
        ("ordinal_encode", OrdinalEncoder(categories=[traffic_order, distance_type_order],
                                          encoded_missing_value=-999), ordinal_cat_cols)
    ], remainder="passthrough")),
    
    ("knn_imputer", KNNImputer(n_neighbors=5))  # Final polish on remaining NaNs
])
```

**Innovation:** `add_indicator=True` captures *which* values were missing — often predictive!

---

## 📈 Model Performance

| Model | Train MAE | Test MAE | R² (Test) | CV R² (5-fold) |
|-------|-----------|----------|-----------|----------------|
| Linear Regression | 4.83 min | 4.86 min | 0.58 | - |
| **Random Forest** | **1.22 min** | **3.29 min** | **0.80** | **0.784 ± 0.003** |

> **Target Transformation:** Yeo-Johnson PowerTransformer on `time_taken` for normality.

---

## 🧪 Experiment Tracking (MLflow)

```python
with mlflow.start_run(run_name="Missing Indicator + KNN"):
    mlflow.log_param("experiment_type", "Advanced Imputation")
    mlflow.log_params(model.get_params())
    mlflow.log_metric("test_mae", 3.29)
    mlflow.log_metric("cv_r2", 0.784)
    # Full reproducibility: params, metrics, artifacts, model version
```

---

## 🚀 Quick Start

```bash
# Clone & setup
git clone https://github.com/yourusername/swiggy-delivery-prediction.git
cd swiggy-delivery-prediction
pip install -r requirements.txt

# Start MLflow tracking server
mlflow ui --port 5000

# Run pipeline
python src/train.py --model random_forest --track-experiments
```

---

## 📁 Project Structure

```
swiggy-delivery-prediction/
├── 📂 data/
│   ├── raw/swiggy.csv                    # Original 45K records
│   └── processed/swiggy_cleaned.csv      # Post-method-chain
├── 📂 src/
│   ├── data_clean_utils.py               # 🔧 Method chaining core
│   ├── features.py                       # Haversine + time features
│   ├── pipeline.py                       # sklearn Pipeline definitions
│   └── train.py                          # Entry point with MLflow
├── 📂 notebooks/
│   ├── 01_data_cleaning.ipynb            # EDA + anomaly detection
│   ├── 02_feature_engineering.ipynb      # Method chaining demo
│   └── 03_model_training.ipynb           # Pipeline + tuning
├── 📂 tests/
│   └── test_pipelines.py                 # Unit tests for pure functions
├── README.md
└── requirements.txt
```

---

## 🎓 Key Learnings

| Challenge | Solution | Takeaway |
|-----------|----------|----------|
| Hidden string NaNs | Regex detection + `replace("NaN ", np.nan)` | Always inspect `df.sample(50)` |
| 17% missing data | Missing indicators + KNN imputation | Missingness is information |
| 4,071 invalid coordinates | Absolute values + threshold to NaN | Domain knowledge > statistics |
| Target bimodality | Yeo-Johnson transformation | Check distributions before modeling |

---

## 🔮 Next Steps

- [ ] **Hyperparameter tuning:** `Optuna` for RF/XGB/LightGBM
- [ ] **Feature selection:** `SelectKBest` + `RFE` on 50+ features
- [ ] **Model stacking:** Ensemble of tree-based models
- [ ] **SHAP interpretability:** Explain delivery time drivers
- [ ] **API deployment:** FastAPI + Docker containerization

---

## 🛠️ Tech Stack

<p align="left">
  <img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white" />
  <img src="https://img.shields.io/badge/seaborn-3793EF?style=for-the-badge&logoColor=white" />
</p>

---

<p align="center">
  <b>Built with method chaining, type safety, and MLOps best practices.</b><br>
  <i>⭐ Star if you find the pipeline architecture useful!</i>
</p>
```