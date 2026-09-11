import os
import numpy as np
import pandas as pd

# Paths for input (cleaned CSV) and output (feature-engineered CSV)
INPUT_FILENAME = "Property_Price_Register_Ireland-28-05-2021_cleaned.csv"
OUTPUT_FILENAME = "Property_Price_Register_Ireland-28-05-2021_features.csv"


def col(df, options):
    # Return the first matching column name from a list of possible names.
    # Matching is case-insensitive.
    lut = {c.lower(): c for c in df.columns}
    for opt in options:
        if opt.lower() in lut:
            return lut[opt.lower()]
    raise KeyError(f"None of {options} found in dataset columns.")


def parse_condition(desc: str) -> str:
    # Identify whether a property is new or second-hand
    if not isinstance(desc, str):
        return "unknown"

    s = desc.strip().lower()

    # Detect second-hand keywords (including Irish translations)
    if (
        "second-hand" in s
        or "second hand" in s
        or "athláimhe" in s
        or "ath laimhe" in s
        or "second" in s
    ):
        return "second_hand"

    # Detect new-build keywords
    if "new" in s or "nua" in s:
        return "new"

    return "unknown"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Create a copy so we don't modify the original data
    out = df.copy()

    # Find the correct column names (case-insensitive)
    c_desc = col(out, ["property_desc", "PROPERTY_DESC"])
    c_date = col(out, ["sale_date", "SALE_DATE"])

    # Convert date column to real datetime format
    out[c_date] = pd.to_datetime(out[c_date], errors="coerce", utc=False)

    # ----------------------- Feature Engineering -----------------------

    # Create new feature: 1 = new build, 0 = second-hand/unknown
    condition = out[c_desc].apply(parse_condition)
    out["is_new_build"] = (condition == "new").astype("Int64")

    # Extract year and month from sale_date
    out["year_sold"] = out[c_date].dt.year.astype("Int64")
    out["month_sold"] = out[c_date].dt.month.astype("Int64")

    # Drop any previously encoded columns (leftover from older versions)
    to_drop = [
        name
        for name in out.columns
        if name.startswith(("PROPERTY_CONDITION", "PROPERTY_CATEGORY", "COND_", "CAT_"))
    ]
    out = out.drop(columns=to_drop, errors="ignore")

    return out


def main():
    # Load cleaned CSV file
    print("Loading dataset:", INPUT_FILENAME)
    df = pd.read_csv(INPUT_FILENAME)

    # Apply feature engineering
    print("Engineering features (is_new_build, year_sold, month_sold, datetime conversion)...")
    df_out = engineer_features(df)

    # Save engineered CSV
    df_out.to_csv(OUTPUT_FILENAME, index=False)

    # Identify new columns added
    added = sorted(set(df_out.columns) - set(df.columns))

    print("\nFeature engineering complete")
    print(f"Output file saved as: {OUTPUT_FILENAME}")
    print(f"Added columns: {', '.join(added)}")


if __name__ == "__main__":
    main()
