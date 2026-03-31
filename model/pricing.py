"""Price prediction for a Bengaluru listing (Session 6).

Three things that the notebook and the API (Session 7) share:

    normalize_location(name)                          clean a raw or typed location name
    load_artifacts()                                  load the model, the location list and the metadata
    predict_price(location, total_sqft, bath, bhk)    check the inputs, return the price in lakh INR

The files are read from model/artifacts/, which the notebook writes at the end of Session 6.
A pickle runs code when it is loaded: load house_price_model.pkl only from a source you trust,
and only with the scikit-learn version recorded in metadata.json.

    from pricing import predict_price
    predict_price("Whitefield", 1200, 2, 2)
"""
import json
import math
import warnings
from pathlib import Path

import joblib
import pandas as pd

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
OTHER = "other"
_loaded = {}                      # artifacts already read from disk, by folder


def normalize_location(name):
    """Collapse runs of whitespace to one space and remove trailing commas and periods (and the spaces in front of them).
    The capitalisation is kept; compare names with .casefold()."""
    if not isinstance(name, str):
        raise ValueError(f"location must be text, got {name!r}")
    return " ".join(name.split()).rstrip(" ,.")


def load_artifacts(folder=ARTIFACTS):
    """Read the three artifacts once and return {"model", "locations", "metadata"}."""
    folder = Path(folder)
    if folder not in _loaded:
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        locations = json.loads((folder / "locations.json").read_text(encoding="utf-8"))
        import sklearn
        fitted_with = metadata["libraries"]["scikit-learn"]
        if sklearn.__version__ != fitted_with:
            warnings.warn(f"the model was fitted with scikit-learn {fitted_with} but {sklearn.__version__} is installed; "
                          f"install the pinned version from requirements.txt", stacklevel=2)
        model = joblib.load(folder / "house_price_model.pkl")
        _loaded[folder] = {"model": model, "locations": locations, "metadata": metadata}
    return _loaded[folder]


def _number(value, name):
    """A finite number, or a ValueError that says what was wrong."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}") from None
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return number


def predict_price(location, total_sqft, bath, bhk):
    """Price in lakh INR (one decimal) and whether the location is a named one or `other`.

    Raises ValueError, with a readable message, for inputs the cleaning rules of Session 4 would have removed
    (fewer than 300 sqft per bedroom, more than bhk + 2 bathrooms), for a bhk below 1 and for a total_sqft
    outside the range of the training data. A location that is not in the list is treated as `other`."""
    artifacts = load_artifacts()
    limits = artifacts["metadata"]["validation"]

    name = normalize_location(location)
    if not name:
        raise ValueError("location must not be empty")
    total_sqft, bath, bhk = _number(total_sqft, "total_sqft"), _number(bath, "bath"), _number(bhk, "bhk")

    if bhk != int(bhk) or bhk < limits["min_bhk"]:
        raise ValueError(f"bhk must be a whole number of at least {limits['min_bhk']}, got {bhk:g}")
    if bath != int(bath) or bath < limits["min_bath"]:
        raise ValueError(f"bath must be a whole number of at least {limits['min_bath']}, got {bath:g}")
    if not limits["min_total_sqft"] <= total_sqft <= limits["max_total_sqft"]:
        raise ValueError(f"total_sqft must be between {limits['min_total_sqft']:,.0f} and {limits['max_total_sqft']:,.0f} sqft, "
                         f"the range of the training data, got {total_sqft:,.0f}")
    needed = limits["min_sqft_per_bedroom"] * bhk
    if total_sqft < needed:
        raise ValueError(f"a {bhk:g} BHK needs at least {needed:,.0f} sqft ({limits['min_sqft_per_bedroom']} sqft per bedroom), got {total_sqft:,.0f}")
    if bath > bhk + limits["max_baths_over_bhk"]:
        raise ValueError(f"bath can be at most bhk + {limits['max_baths_over_bhk']}, which is {bhk + limits['max_baths_over_bhk']:g} "
                         f"for a {bhk:g} BHK, got {bath:g}")

    display = artifacts["locations"]["key_to_name"].get(name.casefold())          # the named location, or None
    row = pd.DataFrame([{"location": display or OTHER, "total_sqft": total_sqft, "bath": bath, "bhk": int(bhk)}])
    price = float(artifacts["model"].predict(row)[0])
    return {"price_lakh": round(price, 1), "location": display or OTHER, "location_type": "named" if display else OTHER}
