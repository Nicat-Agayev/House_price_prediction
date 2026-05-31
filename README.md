# Housing Price Prediction

A regression project that predicts house prices from features such as area, number of rooms, furnishing status and location-related attributes. It covers the full workflow: data cleaning, exploratory data analysis, feature engineering, model training and evaluation.

## Notebook

GitHub may not render the notebook preview properly because it contains several visualizations.  
For the full rendered notebook with plots, use nbviewer:

[View Notebook on nbviewer](https://nbviewer.org/github/Nicat-Agayev/House_Price_prediction/blob/main/notebooks/house_price_prediction.ipynb)

The dataset is small but non-trivial: the features have **strong multicollinearity** and the target (price) is **right-skewed**. The project handles both and compares four regression models.

**Dataset:** [Housing Prices Dataset (Kaggle)](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)

## Steps

1. **Data cleaning** — checked for missing values (none) and inspected data types.
2. **EDA** — examined price/area distributions (confirmed the skew), built a correlation matrix to expose multicollinearity, and explored the categorical features.
3. **Outlier handling** — removed extreme `area` outliers only. Price (the target) outliers were kept on purpose, since removing rows based on the target would bias the model.
4. **Feature engineering** — encoded binary features as 0/1, ordinal-encoded `furnishingstatus`, and added a `total_rooms` feature.
5. **Modelling** — log-transformed the skewed target, standardized the features, and compared four models with shuffled 5-fold cross-validation.

### Handling the two challenges
- **Skewed target** → trained on `log(price)` and converted predictions back to real prices.
- **Multicollinearity** → included **Ridge** and **Lasso** (regularized linear models) alongside Linear Regression and Random Forest.

## Results

5-fold cross-validated R² (on `log(price)`):

| Model              | CV R² |
|--------------------|:-----:|
| Linear Regression  | 0.66  |
| Ridge              | 0.66  |
| Lasso              | 0.66  |
| Random Forest      | 0.63  |

The linear models perform best and almost identically (the regularization confirms multicollinearity is under control). **Ridge** was selected as the final model. On the held-out test set:

| Metric | Value |
|--------|-------|
| R²     | 0.62  |
| MAE    | ~926,000 (~20% of the mean price) |
| RMSE   | ~1,341,000 |

`area` is the most influential feature.

**Why not higher?** The dataset is small (~500 rows) and has **no location information** — the single biggest driver of real house prices — which caps the achievable accuracy. Recognizing this limitation is part of the analysis.

## Repository structure

```
house-price-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── Housing.csv
├── notebooks/
│   └── house_price_prediction.ipynb
└── models/
    └── house_price_model.pkl   # created when the notebook runs
```

## How to run

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/house_price_prediction.ipynb
```

Run the cells top to bottom. With `random_state=42` set throughout, the results are reproducible.

## Tools

Python · NumPy · pandas · scikit-learn · Matplotlib · seaborn
