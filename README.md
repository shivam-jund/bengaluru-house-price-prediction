# Bengaluru House Price Prediction

Predict the price of a residential property in Bengaluru from its location, size (BHK), total area (sqft) and number of bathrooms, then serve the model through a small website.

## Goal

A real-estate business (think Zillow's Zestimate, or MagicBricks in India) wants an automatic price estimate for a listing. This project builds that estimator end to end:

1. **Model**: clean the data, engineer features, remove outliers, reduce dimensionality and fit a regression model (pandas, Matplotlib, scikit-learn).
2. **Export**: save the final model as a pickle file.
3. **API**: a FastAPI server that loads the pickle and exposes HTTP endpoints for price prediction.
4. **Website**: an HTML/CSS/JavaScript page that calls those endpoints.

The target variable is `price`, in **lakhs of INR** (1 lakh = 100,000 INR). Prices stay in lakhs everywhere in the project so the website output matches the training data.

## Data

Kaggle, [Bengaluru House Price Data](https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data) (file `bengaluru_house_prices.csv`). Terms of use are those on the dataset page.

Columns: `area_type`, `availability`, `location`, `size`, `society`, `total_sqft`, `bath`, `balcony`, `price`.

Download the CSV from Kaggle and save it as `data/bengaluru_house_prices.csv`.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook model/price_prediction.ipynb
```

## Repository layout

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── bengaluru_house_prices.csv    # downloaded from Kaggle (see Data)
└── model/
    └── price_prediction.ipynb      # cleaning, feature engineering, model building
```

The FastAPI server and the website will be added in later sessions.

## Known limitations

- **`area_type` is excluded from the baseline.** "Super built-up Area" and "Carpet Area" change what `total_sqft` actually measures, so the column may carry real signal. It was dropped in Session 1 as a judgment call, not because it is unusable. To test in Session 5: does adding it improve the cross-validation score?
- **`availability`, `society` and `balcony` were dropped on data-quality grounds** (possession dates mixed with a label, a large share missing with very many distinct names, and a minor amenity count with missing values). Their predictive value was not tested.
