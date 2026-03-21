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
    └── price_prediction.ipynb      # Sessions 1-4: cleaning, feature engineering, location grouping, duplicate and outlier removal (model building comes next)
```

The FastAPI server and the website will be added in later sessions.

## Location handling (Session 3)

`location` starts with 1,299 distinct raw names and ends with 241 categories:

1. **Clean the names.** Strip and collapse whitespace (1,299 names become 1,288), merge case variants (1,276), then remove trailing commas and periods (1,266). If a place is written in several ways, its most frequent spelling is used, so the website can later show names like "JP Nagar".
2. **Group rare locations.** A location with 10 or fewer listings is replaced by `other`. That leaves 240 named locations plus `other`, 241 categories in all. At the end of Session 3, `other` held 2,868 of the 13,232 rows (21.7%) and was the largest category (the largest named location, Whitefield, has 537).
3. **No hand edits.** Nine names with a digit glued to a letter (for example "1Hanuman Nagar") are left as they are. Each has one listing, so all of them fall into `other`.

The threshold of 10 is a convention. On those 13,232 rows the notebook compares it with 5 (386 named locations, 13.3% of the rows in `other`) and 20 (143 named locations, 32.3%); Session 5 tests whether a different one changes the cross-validation score.

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

## Known limitations

- **`area_type` is excluded from the baseline.** "Super built-up Area" and "Carpet Area" change what `total_sqft` actually measures, so the column may carry real signal. It was dropped in Session 1 as a judgment call, not because it is unusable. To test in Session 5: does adding it improve the cross-validation score?
- **`availability`, `society` and `balcony` were dropped on data-quality grounds** (possession dates mixed with a label, a large share missing with very many distinct names, and a minor amenity count with missing values). Their predictive value was not tested.
- **The 74 rows dropped for a missing `location`, `size` or `bath` (0.56% of the data) are not a random sample.** Most have a ranged `total_sqft` and no `balcony` value, and 16 are plot listings with no bedroom count, so the baseline under-represents those listing types. Details are in section 4 of the notebook.
- **"RK" listings are counted as 1 BHK.** 13 rows (0.10%) are written "1 RK" (one room plus a kitchen, no separate bedroom). They got `bhk = 1`, so the model cannot tell them from 1 BHK flats. Their median price is lower (25.25 lakh against 32.00 lakh for 1 BHK).
- **`total_sqft` ranges are replaced by their midpoint.** This affects 144 rows. The midpoint hides how wide a range is: the median range is 160 sqft wide, the widest is "3067 - 8156" (5,089 sqft).
- **Unit-string handling removed 14 rows.** 32 rows quoted in square metres (17) or square yards (15) were converted to square feet (× 10.7639 and × 9). 14 rows quoted in acres (8), cents (2), guntha (2), perch (1) or grounds (1) were dropped (0.11% of the rows): these are land-parcel units, some defined differently from region to region, so the model never sees listings written in them.
- **The model will probably under-predict expensive homes.** A share of the top of the market was removed as outliers: 216 of the 779 listings priced 300 lakh or more (27.7%) are gone, against 13.6% of all rows (116 by the location fences, 63 by the pooled fence, 27 by the area rule, 7 by the bathroom rule, 3 as repeats). Some of them are errors and some are real homes, but the frame the model learns from holds fewer expensive homes than the market, so its estimates for them will probably be too low. The first version of the rules, on the raw values, removed 47.9% of them (373 of 779). Session 5 reports the errors by price band.
- **Some extreme values are still in the data.** The area rule removed the impossible areas (seven rows of 1 to 60 sqft). No row below 1,000 INR per sqft is left (there were 17, among them two ordinary-size homes with prices that look too low: 1,798 sqft at 10 lakh and 2,400 sqft at 12 lakh) and the minimum is 1,666.7. Two rows above 30,000 INR per sqft stay (34,050 in Rajaji Nagar and 30,242 in Indira Nagar, both in locations with their own fences), and 8 rows have 10 or more bedrooms (the most is 16).
- **The pooled fence is still blunt.** One pair of limits (1,625 to 22,224 INR per sqft) is applied to 956 different places. The 80 rows it removes come from places with a median of 12,234 INR per sqft against 5,456 for the survivors (the median, over the rows, of the median price per sqft of each row's own location; the removed rows themselves have a median of 26,200 against 5,275), so it still takes mostly high-priced places. It takes 80 rows, not the 235 of the raw version.
- **The area and bathroom rules can remove real homes.** 600 of the 737 area-rule rows have 4 or more bedrooms, and 173 of the 737 (23%, against 6.4% of all rows) are exactly 1,200 sqft, the size of a 30 × 40 ft site. For these houses `total_sqft` is most likely the site size and not the floor area, which is why the area rule removes them (1,200 sqft for 5 or more bedrooms is under 300 sqft each). They are probably real homes, so the model sees fewer large houses. Most of the 9 bathroom-rule rows are large, expensive homes (median 775 lakh; 7 cost 300 lakh or more).
- **The repeat test cannot tell a repeat from an identical flat.** The data has no listing id, so one flat posted twice and two identical flats in one project at one price look the same. One copy of each always stays.
- **`price_per_sqft` is built from the target (`price`).** It was made to spot outliers, and the Session 4 fences use it, so it is still in the frame. It goes first in Session 5 and must never be a model input.
- **The grouping and the Session 4 rules use all rows.** Which locations stay named is decided from listing counts over all 13,232 rows, and the repeats, domain rules and fences are computed on all rows, all before the train/test split, so they use information from rows that will later be test data. Because the fences use `price_per_sqft`, a row is also kept or removed according to its own price, test rows included: a model is scored on listings that pass the rules, while the website can receive a listing the rules would have removed. A stricter version would compute everything on the training rows only.
- **The location grouping was not redone after the removals.** `other` now holds 2,442 of the 11,430 rows (21.4%, against 21.7% before the rules), and 36 of the 240 named locations have 10 or fewer rows left (the smallest has 5).
- **`other` is a catch-all, not a place.** It mixes many rare locations: in Session 3 its median price per sqft was 5,908 INR against 5,333 for the named locations, and its middle half was much wider (4,444 to 10,000 against 4,221 to 6,978). Knowing that a listing is in `other` says little about its price. Session 6 decides how the website treats it.
- **Names are merged only by whitespace, case and trailing punctuation.** Misspellings ("Whietfield"), other spellings of what looks like the same road ("Kanakapura Road" next to "Kanakpura Road") and compound names with an inner comma ("Judicial Layout, Kanakapura Road") stay separate. The rare ones fall into `other`.

## Open items

- **Session 5, first step: drop `price_per_sqft`.** It is built from the target.
- **Session 5 tests**: thresholds 5 and 20 on the cleaned frame (36 named locations now have 10 or fewer rows); scores with and without the repeats; errors reported separately for named locations and `other`, and by price band (27.7% of the listings priced 300 lakh or more were removed).
- **`other` on the website**: Session 6 decides how it is treated.
