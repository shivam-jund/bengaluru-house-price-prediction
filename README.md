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

The results in the notebook were produced with Python 3.13, pandas 3.1.0, NumPy 2.5.4, Matplotlib 3.11.2 and scikit-learn 1.9.1. `requirements.txt` pins `scikit-learn` (a pickle loads only with the version that wrote it; see Model export) and leaves the other libraries unpinned until Session 9, so a newer or older version of those may print slightly different numbers (`LinearRegression` on a sparse matrix is solved iteratively, and a fully grown tree depends on how ties between equal splits are visited, so the last decimals can move). The notebook takes about 7 minutes to run, almost all of it in the random-forest grid search of Session 5 and the cross-validation runs of Session 6.

## Repository layout

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── bengaluru_house_prices.csv  # the dataset (see Data)
└── model/
    ├── price_prediction.ipynb      # Sessions 1-6: cleaning, features, location grouping, duplicate and outlier removal, model building and selection, robustness checks, export
    ├── price_prediction_2026-11-04_session4b.ipynb   # dated copy: the notebook as it stood at the end of Session 4b, before any model code
    ├── price_prediction_2026-11-06_session5.ipynb    # dated copy: the notebook as it stood at the end of Session 5, before the robustness checks
    ├── pricing.py                  # normalize_location, load_artifacts, predict_price (used by the notebook and, in Session 7, by the API)
    └── artifacts/                  # written by the notebook at the end of Session 6
        ├── house_price_model.pkl   # the fitted Pipeline (joblib, compress=3), 16.1 MB
        ├── locations.json          # the 240 location names plus other, and the lowercase-name lookup
        └── metadata.json           # library versions, seed, rows, features, validation limits, metrics, notes
```

The FastAPI server comes in Session 7 and the website in Session 8.

## Location handling (Session 3)

`location` starts with 1,299 distinct raw names and ends with 241 categories:

1. **Clean the names.** Strip and collapse whitespace (1,299 names become 1,288), merge case variants (1,276), then remove trailing commas and periods (1,266). If a place is written in several ways, its most frequent spelling is used, so the website can later show names like "JP Nagar".
2. **Group rare locations.** A location with 10 or fewer listings is replaced by `other`. That leaves 240 named locations plus `other`, 241 categories in all. At the end of Session 3, `other` held 2,868 of the 13,232 rows (21.7%) and was the largest category (the largest named location, Whitefield, has 537).
3. **No hand edits.** Nine names with a digit glued to a letter (for example "1Hanuman Nagar") are left as they are. Each has one listing, so all of them fall into `other`.

The threshold of 10 is a convention. On those 13,232 rows the notebook compares it with 5 (386 named locations, 13.3% of the rows in `other`) and 20 (143 named locations, 32.3%); Session 6 then compared them by cross-validation, counting listings on these 13,232 rows, and kept 10 (see Robustness checks).

## Duplicate and outlier removal (Session 4, revised as 4b)

Four rules, applied in this order. The notebook counts each one, prints 10 random removed rows per rule for a check by eye, and ends with this table:

| Rule | What it removes | Rows before | Removed | Rows after |
|:--|:--|--:|--:|--:|
| 1. Repeats | rows that repeat an earlier row on all nine raw columns (the first copy stays) | 13,232 | 529 | 12,703 |
| 2a. Area | fewer than 300 sqft per bedroom (`total_sqft / bhk`) | 12,703 | 737 | 11,966 |
| 2b. Bathrooms | more than `bhk + 2` bathrooms | 11,966 | 9 | 11,957 |
| 3. Location fences | `log(price_per_sqft)` outside `Q1 − 1.5 × IQR` to `Q3 + 1.5 × IQR` of its own location (230 locations with 10 or more rows; the fences are turned back into INR per sqft) | 11,957 | 447 | 11,510 |
| 4. Pooled fence | the same fence on `log(price_per_sqft)` for all rows of places with fewer than 10 rows (2,589 rows from 956 places; fences 1,625 to 22,224 INR per sqft) | 11,510 | 80 | 11,430 |
| **Total** | | 13,232 | **1,802 (13.6%)** | **11,430** |

Counted from the 13,320 raw rows, 1,890 rows (14.2%) are gone: 88 in Sessions 1 and 2 and 1,802 in Session 4.

**Why this order.** Quartiles come from the rows present when a rule runs: a repeated row counts twice, and an impossible row (120 lakh for 1 sqft) widens the fences of its location. So the repeats go first, then the domain rules, then the fences.

**Choices.**

- **Repeats are tested on all nine raw columns.** On the five columns the frame kept the count would be 888, and 359 of those rows differ in `society`, `area_type`, `availability` or `balcony`, so they can be different flats.
- **300 sqft per bedroom is a convention.** The notebook shows what 250 and 350 would do: 250 keeps 132 of the 737 rows, 350 removes 349 more.
- **Rules 3 and 4 work on the logarithm of `price_per_sqft`.** The first version of Session 4 used the raw values. `price_per_sqft` is right-skewed and an error usually scales a value, but fences on the raw values are symmetric in rupees: `Q1 − 1.5 × IQR` is below zero in 28 of the 230 locations and in the pooled group (−1,667), so there the rules could only cut the top of the market. On the log scale the fences are positive and can flag cheap rows too (118 of the 447 location flags are below the lower fence). The notebook keeps the raw results next to the log ones, on the same 11,957 rows:

  | Version | Rows flagged (Rule 3 + Rule 4) | Rows left | Listings of 300 lakh or more removed (of 779) | `price_per_sqft`, min to max | SD | Rows under 1,000 left |
  |:--|--:|--:|--:|--:|--:|--:|
  | Raw fences (first version) | 575 + 235 | 11,147 | 373 (47.9%) | 267.8 to 29,166.7 | 2,507.2 | 11 |
  | Raw Rule 3, log Rule 4 | 575 + 80 | 11,302 | 285 (36.6%) | 405.4 to 29,166.7 | 2,858.8 | 2 |
  | **Log fences (used)** | 447 + 80 | 11,430 | 216 (27.7%) | 1,666.7 to 34,050.2 | 3,006.1 | 0 |

- **IQR fences, not mean ± 1 SD.** On the same rows mean ± 1 SD would flag 1,805 rows (15.1%), about four times the 447 (3.7%) that the location fences flag.
- **Fences use the cleaned location names, not the grouped `location`.** Its `other` bucket mixes 1,026 different places, so a fence over it would mean nothing.
- **Report only, nothing removed:** 1,570 of the 3,974 comparable 3 BHK rows (39.5%) cost less per sqft than the 2 BHK average of their location. That is too common to be a sign of error.

**Result.** 11,430 rows × 6 columns (`location`, `total_sqft`, `bath`, `price`, `bhk`, `price_per_sqft`), no nulls. `price_per_sqft` has minimum 1,666.7, maximum 34,050.2 and standard deviation 3,006.1 (before: maximum 12,000,000, standard deviation 106,601.3). The smallest `total_sqft` is 300 (before: 1).

## Model building and selection (Session 5)

**Setup.** `price_per_sqft` is dropped first (it is built from the target) and the notebook asserts that it is gone. Features: `location` (241 categories), `total_sqft`, `bath`, `bhk`. Target: `price`, in lakh. The 11,430 rows are split with `train_test_split(test_size=0.2, random_state=10)` into 9,144 training and 2,286 test rows. The same seed (10) is used for the cross-validation folds and the forests. The split is random, not stratified. The test set holds 500 `other` rows (21.9%) and 123 rows priced 300 lakh or more (5.4%); the training set holds 1,942 (21.2%) and 440 (4.8%).

**Preprocessing.** Every model is a scikit-learn `Pipeline`: a `ColumnTransformer` that one-hot encodes `location` (`OneHotEncoder(handle_unknown="ignore")`) and passes `total_sqft`, `bath` and `bhk` through, then the model. Cross-validation fits the whole Pipeline on each training fold, so the encoder never sees the validation fold or the test set. A location the encoder has not seen is encoded as zeros instead of raising an error. The numbers are not scaled.

**Metrics.** MAE in lakh (the average size of the error); median absolute percentage error, "median % error" (the typical error relative to the price of the home); R² for reference. R² is mostly a score on the dearest homes: the listings priced 300 lakh or more are 4.8% of the training rows and hold 79.8% of the variation around the mean. Models are therefore chosen by MAE, with the median % error beside it.

**Baselines, scored on the test set** (both fitted on the training rows only):

| Baseline | MAE (lakh) | Median % error | R² |
|:--|--:|--:|--:|
| `DummyRegressor(strategy="median")` (always 69.1 lakh) | 54.17 | 38.2 | −0.103 |
| Median INR per sqft × `total_sqft` (5,272 INR per sqft) | 35.43 | 22.9 | 0.454 |

**Cross-validation, training rows only.** 5-fold shuffled `KFold`, the same folds for every model and for the baselines, `GridSearchCV` choosing by `neg_mean_absolute_error`. Mean ± standard deviation over the five folds:

| Model (grid) | Best hyperparameters | CV MAE (lakh) | CV median % error | CV R² |
|:--|:--|--:|--:|--:|
| Median baseline | | 55.29 ± 1.04 | 39.5 ± 0.8 | −0.080 ± 0.011 |
| Area baseline | | 35.85 ± 0.71 | 23.4 ± 0.6 | 0.464 ± 0.045 |
| `LinearRegression` (none) | | 31.34 ± 1.38 | 22.5 ± 0.4 | 0.703 ± 0.037 |
| `Lasso` (`alpha` 0.01, 0.1, 1) | `alpha=0.01` | 31.09 ± 1.35 | 22.4 ± 0.5 | 0.703 ± 0.036 |
| `DecisionTreeRegressor` (`max_depth` 5, 10, 20 × `min_samples_leaf` 1, 5, 20) | `max_depth=20`, `min_samples_leaf=1` | 29.97 ± 0.62 | 18.9 ± 0.6 | 0.626 ± 0.057 |
| **`RandomForestRegressor`** (100 trees; `max_depth` 10, 20, None × `min_samples_leaf` 1, 5) | `max_depth=None`, `min_samples_leaf=1` | **26.39 ± 1.22** | **15.2 ± 0.7** | 0.710 ± 0.065 |

The forest has the lowest MAE on all five folds, by at least 2.6 lakh over the best other model on that fold. R² does not separate the models (the forest and linear regression differ by 0.007, and the decision tree beats both linear models on MAE and loses to them on R²), because a few very expensive homes dominate it. R² near 0.7 is what this data supports; it is not a reason to remove more outliers.

**A `log(price)` target** (`TransformedTargetRegressor` around the whole Pipeline, so the metrics stay in lakh), on the same folds:

| Model | CV MAE (lakh) | CV median % error | CV R² |
|:--|--:|--:|--:|
| `LinearRegression` | 31.34 ± 1.38 | 22.5 ± 0.4 | 0.703 ± 0.037 |
| `LinearRegression`, log target | 360.42 ± 433.95 | 20.1 ± 0.4 | −35,334.842 ± 53,811.871 |
| `Lasso` | 31.09 ± 1.35 | 22.4 ± 0.5 | 0.703 ± 0.036 |
| `Lasso`, log target (`alpha=0.0001`) | 399.92 ± 486.84 | 20.2 ± 0.4 | −44,485.996 ± 67,932.791 |
| `RandomForestRegressor` | 26.39 ± 1.22 | 15.2 ± 0.7 | 0.710 ± 0.065 |
| `RandomForestRegressor`, log target | 26.62 ± 0.92 | 15.3 ± 0.6 | 0.697 ± 0.039 |

R² fell for a real reason. A model of `log(price)` multiplies the price by 1.34 for every extra 1,000 sqft, so it explodes on the ten training listings above 10,000 sqft: the 30,400 sqft listing, which sold for 1,824 lakh, is predicted (out of fold) at 1,954,432 lakh, and one row holds 79.6% of the squared error (the exact digits depend on the solver; an exact least-squares fit gives the same pattern). The log target does improve the linear models' median % error (22.5% to 20.1%) and the small homes (up to 1,500 sqft), but R² and MAE are measured in lakh, where those few rows decide. The forest averages training prices, so it cannot leave their range and does not change. The target stays `price` in lakh.

**Final model.** `RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_leaf=1, random_state=10)` inside the Pipeline, fitted on all 9,144 training rows and evaluated once on the 2,286 test rows, in a single notebook cell:

| | MAE (lakh) | Median % error | R² |
|:--|--:|--:|--:|
| Median baseline | 54.17 | 38.2 | −0.103 |
| Area baseline | 35.43 | 22.9 | 0.454 |
| **Final model** | **25.92** | **14.9** | **0.695** |

Cross-validation expected an MAE of 26.39 ± 1.22 lakh, so the test score agrees with it. The typical home is predicted within about 15% of its price.

**Where it fails** (test set; mean signed error = predicted − actual, so negative means under-prediction):

| Group | Rows | Mean actual price (lakh) | MAE (lakh) | Median % error | Mean signed error (lakh) |
|:--|--:|--:|--:|--:|--:|
| all test rows | 2,286 | 103.3 | 25.92 | 14.9 | −1.15 |
| named locations | 1,786 | 96.7 | 19.31 | 12.7 | −2.99 |
| `other` | 500 | 126.9 | 49.53 | 26.9 | +5.45 |
| actual price under 75 lakh | 1,242 | 51.0 | 10.65 | 13.9 | +5.26 |
| 75 to 150 | 685 | 101.2 | 21.93 | 13.1 | +0.22 |
| 150 to 300 | 236 | 202.7 | 56.42 | 20.8 | −6.43 |
| 300 and above | 123 | 453.2 | 143.80 | 26.3 | −63.35 |

`other` has about twice the error of the named locations, and the 300+ lakh band is under-predicted by 63 lakh on average, with only 123 rows behind it. The notebook also draws predicted against actual price on log axes with the line y = x.

**Sanity predictions** (hand-made inputs; the comparison listings are training rows, not test rows):

| Listing | Predicted (lakh) | Median of similar training listings (same location and BHK, area within 15%) | Middle half | At exactly this area (median, mean; listings) |
|:--|--:|:--|:--|:--|
| Whitefield, 2 BHK, 1,200 sqft, 2 bathrooms | 48.0 | 58.0 (156 listings) | 45 to 73 | 45.8, 48.1 (10) |
| Sarjapur Road, 3 BHK, 1,600 sqft, 3 bathrooms | 103.2 | 95.0 (69 listings) | 65 to 110 | 102.5, 102.5 (2) |
| `other`, 2 BHK, 1,200 sqft, 2 bathrooms | 71.9 | 55.0 (510 listings) | 46 to 68 | 61.5, 71.6 (66) |

Two of the three are inside the middle half of their comparable listings. Whitefield is at the low end (17% under the median): 1,200 sqft is a heaped area (5.1% of the training rows) and Whitefield prices climb steeply around it, so the window of ±15% mixes cheaper and dearer listings, and the ten listings at exactly 1,200 sqft have a mean of 48.1. The `other` case is 31% above the median and above the middle half; the forest follows the 66 `other` listings at exactly 1,200 sqft (mean 71.6), and its prediction moves between 60.1 and 84.5 for area changes of 50 sqft or less. Nothing was changed because of these checks.

## Robustness checks and export (Session 6)

### Robustness checks

Cross-validation as in Session 5 (training rows only, the same 5 folds and seed, the forest fixed to the Session 5 winner, no test set). The rule, fixed before the runs: change a setting only if the mean MAE improves by at least 1 lakh **and** the variant wins on at least 4 of the 5 folds, compared with the current setup on the same folds.

| Variant | Rows | Categories | Rows in `other` | CV MAE ± std (lakh) | Lower than current by | Folds won | Decision |
|:--|--:|--:|--:|--:|--:|--:|:--|
| Current (threshold 10) | 11,430 | 241 | 21.4% | 26.39 ± 1.22 | | | baseline |
| Threshold 5 | 11,430 | 387 | 13.1% | 25.62 ± 1.03 | 0.77 | 4 of 5 | keep the current setup |
| Threshold 20 | 11,430 | 144 | 31.5% | 27.65 ± 1.29 | −1.27 | 0 of 5 | keep the current setup |
| `area_type` added (one-hot) | 11,430 | 241 | 21.4% | 25.96 ± 1.37 | 0.43 | 5 of 5 | keep the current setup |
| Repeats left in (diagnostic) | 11,878 | 241 | 20.7% | 25.30 ± 1.32 | 1.09 | | diagnostic only: repeats stay removed |

- **No setting changes.** Threshold 5 and `area_type` each help a little, consistently, but by less than the rule's 1 lakh, which is also under the fold standard deviation of about 1.2. They are recorded as candidates, not findings. Threshold 20 is worse.
- **The repeats are a leak.** With rule 1 switched off the CV MAE looks 1.1 lakh better, because copies of one listing sit on both sides of a fold and the model is scored on homes it has seen. The rows differ between the two runs, so this is a diagnostic, not a paired comparison.
- **"Rows"** is the size of the frame the variant is built on; the cross-validation itself runs on the 9,144 training rows of Session 5 (for the repeats run, on the training part of its own 80/20 split).
- **Thresholds** count listings per cleaned name on the 13,232 rows of Session 3, so threshold 10 reproduces the current frame exactly. **`area_type`** is reloaded from the raw CSV by row index and one-hot encoded next to `location`.

### The predict function

`model/pricing.py` holds everything the API needs, so the logic is not copied into the server:

```python
from pricing import predict_price
predict_price("Whitefield", 1200, 2, 2)
# {'price_lakh': 48.0, 'location': 'Whitefield', 'location_type': 'named'}
```

Run it from `model/` or put `model/` on `sys.path`.

- `normalize_location(name)` collapses whitespace and removes trailing commas and periods, keeping the capitalisation. The notebook (Sessions 3 and 4) imports this same function.
- `load_artifacts()` reads the three artifacts once and warns if the installed scikit-learn is not the one in `metadata.json`.
- `predict_price(location, total_sqft, bath, bhk)` returns the price in lakh (one decimal), the location used and whether it is `named` or `other`. A name is matched after `normalize_location` and `.casefold()`; a name that is not in the list is treated as `other` (so "Whitefield " is Whitefield and "Nowhere Layout" is `other`).
- **Validation.** It raises `ValueError` with a readable message for: fewer than 300 sqft per bedroom, more than `bhk + 2` bathrooms, `bhk` or `bath` below 1 or not whole, `total_sqft` outside 300 to 30,400 (the range of the training data), values that are not numbers, and an empty location. These are the Session 4 rules, applied to the input: the model only saw homes that passed them. Session 7 turns the errors into HTTP 422.

### Model export

The final configuration (Session 5, nothing changed by the checks above) is **refitted on all 11,430 rows** and written to `model/artifacts/`:

| File | Contents | Size |
|:--|:--|--:|
| `house_price_model.pkl` | the fitted Pipeline, `joblib.dump(..., compress=3)`; load with `joblib.load` | 16.1 MB (75.1 MB without compression) |
| `locations.json` | `display_names` (240 names plus `other`) and `key_to_name` (lowercase name to display name) | 13 KB |
| `metadata.json` | library versions, date, seed, rows fitted, feature names, validation limits, Session 5 metrics, notes | 1 KB |

The three files total 16.2 MB, under the 50 MB limit. Compression changes the file, not the model: a test reloads it and compares predictions.

- **The deployed model has no held-out score of its own.** It has seen the 20% that Session 5 held out. `metadata.json` labels its metrics as "from the model fit on 80% of the rows" (test MAE 25.92 lakh, median % error 14.9%, CV MAE 26.39 ± 1.22; typical error 12.7% for named locations and 26.9% for `other`).
- **Both models on the three sanity inputs:** Whitefield 48.0 and 48.0 lakh, Sarjapur Road 103.2 and 97.4, an unlisted place 71.9 and 73.0 (the Session 5 model and the refit).
- **Tests** (plain `assert`s, in the notebook): the reloaded model predicts identically to the in-memory one; the three sanity inputs fall inside the 10th to 90th percentile of similar listings; 500 sqft with 3 BHK and 1 BHK with 4 bathrooms are rejected; "Whitefield " becomes Whitefield and "Nowhere Layout" becomes `other`. A fresh Python process, with only `from pricing import predict_price`, returns a prediction.
- **Security and versions.** A pickle runs code when it is loaded: load `house_price_model.pkl` only from a source you trust. It loads reliably only with the scikit-learn version that wrote it (1.9.1, pinned in `requirements.txt` and recorded in `metadata.json`).
- **Predictions for `other` are about twice as uncertain** as for named locations; `metadata.json` says so, and Session 7 surfaces it in the API.

## Known limitations

- **`area_type` is excluded from the baseline.** "Super built-up Area" and "Carpet Area" change what `total_sqft` actually measures, so the column may carry real signal. It was dropped in Session 1 as a judgment call, not because it is unusable. Session 6 tested it (one-hot, same folds): CV MAE 25.96 against 26.39 and 5 of 5 folds won, but the gain of 0.43 lakh is under the 1 lakh rule, so it was not added. Adding it later would mean one more input in the API and on the website.
- **`availability`, `society` and `balcony` were dropped on data-quality grounds** (possession dates mixed with a label, a large share missing with very many distinct names, and a minor amenity count with missing values). Their predictive value was not tested.
- **The 74 rows dropped for a missing `location`, `size` or `bath` (0.56% of the data) are not a random sample.** Most have a ranged `total_sqft` and no `balcony` value, and 16 are plot listings with no bedroom count, so the baseline under-represents those listing types. Details are in section 4 of the notebook.
- **"RK" listings are counted as 1 BHK.** 13 rows (0.10%) are written "1 RK" (one room plus a kitchen, no separate bedroom). They got `bhk = 1`, so the model cannot tell them from 1 BHK flats. Their median price is lower (25.25 lakh against 32.00 lakh for 1 BHK).
- **`total_sqft` ranges are replaced by their midpoint.** This affects 144 rows. The midpoint hides how wide a range is: the median range is 160 sqft wide, the widest is "3067 - 8156" (5,089 sqft).
- **Unit-string handling removed 14 rows.** 32 rows quoted in square metres (17) or square yards (15) were converted to square feet (× 10.7639 and × 9). 14 rows quoted in acres (8), cents (2), guntha (2), perch (1) or grounds (1) were dropped (0.11% of the rows): these are land-parcel units, some defined differently from region to region, so the model never sees listings written in them.
- **The model under-predicts expensive homes.** A share of the top of the market was removed as outliers: 216 of the 779 listings priced 300 lakh or more (27.7%) are gone, against 13.6% of all rows (116 by the location fences, 63 by the pooled fence, 27 by the area rule, 7 by the bathroom rule, 3 as repeats). Some of them are errors and some are real homes, but the frame the model learns from holds fewer expensive homes than the market, so its estimates for them will probably be too low. The first version of the rules, on the raw values, removed 47.9% of them (373 of 779). Session 5 measured it: on the test set the 300+ lakh band (123 rows) is under-predicted by 63.35 lakh on average (MAE 143.80) and carries about 30% of all the absolute error. The forest's averaging of training prices and the grouping of the bands by actual price add to the effect, and the notebook does not separate the three.
- **Some extreme values are still in the data.** The area rule removed the impossible areas (seven rows of 1 to 60 sqft). No row below 1,000 INR per sqft is left (there were 17, among them two ordinary-size homes with prices that look too low: 1,798 sqft at 10 lakh and 2,400 sqft at 12 lakh) and the minimum is 1,666.7. Two rows above 30,000 INR per sqft stay (34,050 in Rajaji Nagar and 30,242 in Indira Nagar, both in locations with their own fences), and 8 rows have 10 or more bedrooms (the most is 16).
- **The pooled fence is still blunt.** One pair of limits (1,625 to 22,224 INR per sqft) is applied to 956 different places. The 80 rows it removes come from places with a median of 12,234 INR per sqft against 5,456 for the survivors (the median, over the rows, of the median price per sqft of each row's own location; the removed rows themselves have a median of 26,200 against 5,275), so it still takes mostly high-priced places. It takes 80 rows, not the 235 of the raw version.
- **The area and bathroom rules can remove real homes.** 600 of the 737 area-rule rows have 4 or more bedrooms, and 173 of the 737 (23%, against 6.4% of all rows) are exactly 1,200 sqft, the size of a 30 × 40 ft site. For these houses `total_sqft` is most likely the site size and not the floor area, which is why the area rule removes them (1,200 sqft for 5 or more bedrooms is under 300 sqft each). They are probably real homes, so the model sees fewer large houses. Most of the 9 bathroom-rule rows are large, expensive homes (median 775 lakh; 7 cost 300 lakh or more).
- **The repeat test cannot tell a repeat from an identical flat.** The data has no listing id, so one flat posted twice and two identical flats in one project at one price look the same. One copy of each always stays.
- **`price_per_sqft` is built from the target (`price`).** It was made to spot outliers, and the Session 4 fences use it, so it stays in the Session 4 frame. Session 5 drops it before any model (the notebook asserts it is gone), and it must never be a model input.
- **The grouping and the Session 4 rules use all rows.** Which locations stay named is decided from listing counts over all 13,232 rows, and the repeats, domain rules and fences are computed on all rows, all before the train/test split, so they use information from rows that will later be test data. Because the fences use `price_per_sqft`, a row is also kept or removed according to its own price, test rows included: a model is scored on listings that pass the rules, while the website can receive a listing the rules would have removed. The Session 5 scores describe listings that passed these rules. A stricter version would compute everything on the training rows only.
- **The location grouping was not redone after the removals.** `other` now holds 2,442 of the 11,430 rows (21.4%, against 21.7% before the rules), and 36 of the 240 named locations have 10 or fewer rows left (the smallest has 5).
- **`other` is a catch-all, not a place.** It mixes many rare locations: in Session 3 its median price per sqft was 5,908 INR against 5,333 for the named locations, and its middle half was much wider (4,444 to 10,000 against 4,221 to 6,978). Knowing that a listing is in `other` says little about its price. Session 5 measured about twice the error there (test MAE 49.53 lakh against 19.31 for the named locations, median % error 26.9% against 12.7%), and the forest's predictions for an `other` listing move a lot with small changes of area. Session 6 tested thresholds 5 and 20 and kept 10, and the website session decides how `other` is treated.
- **Names are merged only by whitespace, case and trailing punctuation.** Misspellings ("Whietfield"), other spellings of what looks like the same road ("Kanakapura Road" next to "Kanakpura Road") and compound names with an inner comma ("Judicial Layout, Kanakapura Road") stay separate. The rare ones fall into `other`.
- **The test numbers come from one random split with one seed.** The final model's fold MAEs range from 24.8 to 28.3 lakh, so another split would give somewhat different test numbers, and a difference of a lakh or so between two models should not be read as real. The 300+ lakh band has only 123 test rows and the 150 to 300 band 236, so their numbers are rough.
- **The final forest is large.** The plain pickle of the model fitted on all rows is 75.1 MB (100 trees with no depth limit and one row per leaf); `joblib.dump(..., compress=3)` brings the file to 16.1 MB with identical predictions, under GitHub's 50 MB warning. The forest itself was not made smaller.
- **Some ideas were left out on purpose.** A `log(total_sqft)` feature (the brief fixes four features), scaling for Lasso, more trees or `max_features` for the forest, and a split stratified on price band were not tried. The forest's best setting is at the edge of its grid (no depth limit, one row per leaf), so a wider grid might find a slightly better one; its lead over the next model is 3.6 lakh of MAE.
- **A location the encoder has not seen is encoded as zeros.** That is not the same as `other`; `predict_price` avoids it by mapping every name that is not in the list to `other` before the model sees it.
- **`predict_price` has no upper limit on `bhk` or `bath`.** A 40 BHK with enough area passes the checks, and the forest answers like the nearest homes it has seen.
- **The refit model is not scored.** The metrics in `metadata.json` come from the model fitted on 80% of the rows, so the deployed model's own error is expected to be similar, not measured.
- **The `other` swing is mostly data.** Many listings are at heaped areas (1,200 sqft and similar) and `other` mixes many places, so its predictions move a lot with small changes of area. This is documented, not fixed.

## Open items

- **Session 7**: the FastAPI server around `predict_price` (HTTP 422 for the `ValueError`s, the `other` uncertainty in the response).
- **Session 8**: the website. It decides how a typed place that is not in the list is shown (`other`).
- **Candidates not adopted in Session 6**: location threshold 5 and `area_type`, both under the 1 lakh rule. The dearest homes remain under-predicted.
- **Session 9**: pin the other libraries in `requirements.txt`, an end-to-end test, and Git if you choose.
