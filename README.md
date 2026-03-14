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

Kaggle, [Bengaluru House Price Data](https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data) (file `Bengaluru_House_Data.csv`). Terms of use are those on the dataset page.

Columns: `area_type`, `availability`, `location`, `size`, `society`, `total_sqft`, `bath`, `balcony`, `price`.

Save the CSV in `data/`. The notebook accepts `bengaluru_house_prices.csv` (the copy used in Session 1) or `Bengaluru_House_Data.csv` (the file name on Kaggle).

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
│   └── bengaluru_house_prices.csv  # the dataset (see Data)
└── model/
    └── price_prediction.ipynb      # Sessions 1-2: cleaning and feature engineering (model building comes next)
```

The FastAPI server and the website will be added in later sessions.

## Known limitations

- **`area_type` is excluded from the baseline.** "Super built-up Area" and "Carpet Area" change what `total_sqft` actually measures, so the column may carry real signal. It was dropped in Session 1 as a judgment call, not because it is unusable. To test in Session 5: does adding it improve the cross-validation score?
- **`availability`, `society` and `balcony` were dropped on data-quality grounds** (possession dates mixed with a label, a large share missing with very many distinct names, and a minor amenity count with missing values). Their predictive value was not tested.
- **The 74 rows dropped for a missing `location`, `size` or `bath` (0.56% of the data) are not a random sample.** Most have a ranged `total_sqft` and no `balcony` value, and 16 are plot listings with no bedroom count, so the baseline under-represents those listing types. Details are in section 4 of the notebook.
- **"RK" listings are counted as 1 BHK.** 13 rows (0.10%) are written "1 RK" (one room plus a kitchen, no separate bedroom). They got `bhk = 1`, so the model cannot tell them from 1 BHK flats. Their median price is lower (25.25 lakh against 32.00 lakh for 1 BHK).
- **`total_sqft` ranges are replaced by their midpoint.** This affects 144 rows. The midpoint hides how wide a range is: the median range is 160 sqft wide, the widest is "3067 - 8156" (5,089 sqft).
- **Unit-string handling removed 14 rows.** 32 rows quoted in square metres (17) or square yards (15) were converted to square feet (× 10.7639 and × 9). 14 rows quoted in acres (8), cents (2), guntha (2), perch (1) or grounds (1) were dropped (0.11% of the rows): these are land-parcel units, some defined differently from region to region, so the model never sees listings written in them.
- **Extreme values are still in the data.** Some `total_sqft` entries are implausibly small (1 to 60 sqft for homes with 3 to 7 bedrooms), some areas are plot-sized (up to 52,272 sqft), and 26 rows have 10 or more bedrooms (the largest is 43). They are reported in the notebook and left in until Session 4.
- **`price_per_sqft` is built from the target (`price`).** It exists only to spot outliers in Session 4 and must never be used as a model input.

## Open items

- **529 repeated rows**: decide in Session 4, before the train/test split, using all nine raw columns (after the column drops, legitimate flats of the same size at the same price would look like duplicates).
- **Location names**: normalize them (strip, collapse inner whitespace, case-fold) first thing in Session 3. A plain `strip()` will not fix "Sarjapur  Road" (two spaces).
