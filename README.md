# 🏠 House Price Prediction — California Housing Dataset

A complete, beginner-friendly **regression ML project** using the California Housing dataset.
Covers the full ML pipeline: EDA → Feature Engineering → Model Comparison → Evaluation.

---

## 📌 Project Overview

| Item | Detail |
|------|--------|
| **Problem** | Predict median house prices across California districts |
| **Dataset** | `sklearn.datasets.fetch_california_housing` (20,640 samples, 8 features) |
| **Type** | Supervised Learning — Regression |
| **Best Model** | Gradient Boosting (~0.84 R²) |

---

## 🗂️ Project Structure

```
house-price-prediction/
│
├── house_price_prediction.py   # Main script (EDA + models + plots)
├── requirements.txt            # Dependencies
├── README.md                   # This file
│
└── outputs/                    # Auto-generated plots
    ├── price_distribution.png
    ├── correlation_heatmap.png
    ├── feature_scatter.png
    ├── model_comparison.png
    ├── actual_vs_predicted.png
    └── feature_importance.png
```

---

## 🔍 What's Inside

### 1. Exploratory Data Analysis (EDA)
- Target distribution (raw + log-transformed)
- Correlation heatmap
- Scatter plots of top features vs house price

### 2. Feature Engineering
- Rooms per person, Bedrooms per room
- Population per household
- Income × Rooms interaction
- Log transforms on skewed features

### 3. Models Compared
| Model | CV R² (approx) |
|-------|---------------|
| Linear Regression | ~0.61 |
| Ridge Regression | ~0.61 |
| Random Forest | ~0.81 |
| **Gradient Boosting** | **~0.84** |

### 4. Evaluation Metrics
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score + 5-Fold Cross-Validation R²

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run the project
```bash
python house_price_prediction.py
```

Output plots will be saved in the current directory.

---

## 📦 Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
```

---

## 📊 Sample Results

**Gradient Boosting — Best Model**
- MAE  ≈ $30,000
- RMSE ≈ $45,000
- R²   ≈ 0.84

---

## 💡 Key Learnings

- Log-transforming a skewed target can significantly improve model residuals
- Tree-based ensemble models (RF, GBM) outperform linear models on this dataset
- Feature engineering (interaction terms, ratios) adds predictive signal
- Always use cross-validation — a single train/test split can be misleading

---

## 🛠️ Future Improvements

- [ ] Hyperparameter tuning with `GridSearchCV` / `Optuna`
- [ ] XGBoost / LightGBM comparison
- [ ] Geo-clustering features using Latitude & Longitude
- [ ] Streamlit web app for interactive predictions

---

## 📄 License
MIT
