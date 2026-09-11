# -------------------------------------------------------------------------
# Ireland Property Price Register Dataset Profiler
# -------------------------------------------------------------------------
# This script performs a structured read-only inspection of the dataset:
#   1. Loads the CSV (defaults to the file in the same folder)
#   2. Parses potential date columns
#   3. Prints dataset size, memory use, column names, data types
#   4. Reports missing values and "placeholder" missing entries like N/A
#   5. Summarizes datetime columns and duplicate rows
#   6. Displays key distributions (COUNTY, PRICE, PROPERTY_DESC, etc.)
# -------------------------------------------------------------------------

import argparse      # For reading command-line arguments
import os            # For file path handling
import sys           # For exiting gracefully on failure
from typing import List  # For type hints

import numpy as np   # For numeric type checking
import pandas as pd  # For data loading and analysis


# Helper Function: Convert string-like date columns into datetime (safe conversion)
def try_parse_dates(df: pd.DataFrame, date_like_cols: List[str]) -> pd.DataFrame:
    # Loop through the given column names and try to convert each to datetime
    for c in date_like_cols:
        if c in df.columns and not np.issubdtype(df[c].dtype, np.datetime64):
            try:
                # Convert to datetime safely, coercing bad entries to NaT
                df[c] = pd.to_datetime(df[c], errors="coerce", utc=False)
            except Exception:
                # Ignore conversion errors; keep the column unchanged
                pass
    return df


# Helper Function: Print a clear divider for console readability
def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# Main script logic
def main() -> None:
    # ---- Set up argument parser ----
    default_path = os.path.join(os.path.dirname(__file__), "Property_Price_Register_Ireland-28-05-2021.csv")
    ap = argparse.ArgumentParser(description="Profile an Ireland PPR CSV (read-only).")
    ap.add_argument("--path", default=default_path, help="Path to the CSV file (default: same folder)")
    ap.add_argument("--sep", default=",", help="CSV separator (default ,)")
    ap.add_argument("--encoding", default=None, help="CSV encoding if needed")
    ap.add_argument("--rows", type=int, default=None, help="Limit rows to read (for large files)")
    args = ap.parse_args()

    # ---- Configure Pandas display for wide console ----
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 160)

    # ---- LOAD CSV ----
    print_section("LOAD")
    try:
        df = pd.read_csv(
            args.path,
            sep=args.sep,
            encoding=args.encoding,
            nrows=args.rows,
            low_memory=False
        )
    except Exception as e:
        # Exit if the file cannot be read
        print(f"Failed to read CSV: {e}", file=sys.stderr)
        sys.exit(1)

    # Try to parse date columns automatically
    df = try_parse_dates(df, ["SALE_DATE", "SaleDate", "date", "Date"])

    # ---- FILE INFO ----
    print_section("DATASET FILE INFO")
    print(f"File: {args.path}")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    mem_bytes = df.memory_usage(deep=True).sum()
    print(f"Approx. memory usage: {mem_bytes/1024**2:.2f} MB")

    # ---- COLUMN NAMES ----
    print_section("COLUMN NAMES")
    print(list(df.columns))

    # ---- DATA TYPES ----
    print_section("DATA TYPES")
    print(df.dtypes)

    # ---- MISSING VALUES ----
    print_section("MISSING VALUES PER COLUMN")
    miss = df.isna().sum().sort_values(ascending=False)         # Count missing values
    miss_pct = (miss / len(df) * 100).round(2)                  # Calculate missing percentages
    mv = pd.DataFrame({"missing_count": miss, "missing_%": miss_pct})
    print(mv)

    # ---- Scan for explicit "missing" placeholders ----
    placeholders = {"", " ", "NA", "N/A", "na", "n/a", "None", "NULL", "null", "-", "--", "nan"}
    check_cols = ["PROPERTY_SIZE_DESC", "POSTAL_CODE"]  # Common columns with text placeholders

    for c in check_cols:
        if c not in df.columns:
            print(f"\nColumn {c} not found in dataset.")
            continue

        # Only check columns that are strings or categorical (warning-free version)
        if pd.api.types.is_string_dtype(df[c]) or isinstance(df[c].dtype, pd.CategoricalDtype):
            s = df[c].astype(str).str.strip()
            potential_missing = s[s.isin(placeholders)]
            unique_missing = potential_missing.unique()

            if len(unique_missing) > 0:
                print(f"\nColumn: {c}")
                print("Unique 'missing-like' values found:")
                for val in unique_missing:
                    display_val = "(empty string)" if val == "" else repr(val)
                    count_val = (s == val).sum()
                    print(f"  {display_val}: {count_val}")
            else:
                print(f"\nColumn: {c} — no explicit placeholder missing values found.")

    # ---- UNIQUE VALUE COUNTS ----
    print_section("UNIQUE VALUES PER COLUMN")
    nunique = df.nunique(dropna=True).sort_values(ascending=False)
    print(nunique)

    # ---- DATE COLUMN SUMMARIES ----
    dt_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    if dt_cols:
        print_section("DATE COLUMNS SUMMARY")
        for c in dt_cols:
            s = df[c].dropna()
            if s.empty:
                print(f"{c}: all missing")
                continue
            print(f"{c}: min={s.min()}, max={s.max()}, non-null count={s.size:,}")
            try:
                by_year = s.dt.year.value_counts().sort_index()
                print(f"\n{c} — counts by year:\n{by_year}")
            except Exception:
                pass

    # ---- DUPLICATES CHECK ----
    print_section("DUPLICATES")
    dup_all = df.duplicated().sum()  # Count identical rows
    print(f"Exact duplicate rows: {dup_all}")

    # ---- DOMAIN CHECKS ----
    print_section("DOMAIN CHECKS")

    # Inspect SALE_PRICE column if present
    if "SALE_PRICE" in df.columns:
        sp = df["SALE_PRICE"].dropna()
        if not sp.empty:
            print(
                f"SALE_PRICE: min={sp.min():,.2f}, max={sp.max():,.2f}, "
                f"median={sp.median():,.2f}, mean={sp.mean():,.2f}"
            )
            print("\nSALE_PRICE quantiles:")
            print(sp.quantile([0, .01, .05, .25, .5, .75, .95, .99, 1])
                    .apply(lambda x: f"{x:,.2f}"))

    # Display value counts for important categorical fields
    for col in ["COUNTY", "POSTAL_CODE", "PROPERTY_DESC", "PROPERTY_SIZE_DESC"]:
        if col in df.columns:
            print("\n")
            print(df[col].value_counts(dropna=False))

            # Also show percentage share by COUNTY
            if col == "COUNTY":
                try:
                    county_pct = (
                        df["COUNTY"]
                        .value_counts(dropna=False, normalize=True)
                        .mul(100)
                        .round(2)
                    )
                    print("\nCOUNTY share of dataset (%):")
                    print(county_pct)
                except Exception as e:
                    print(f"(Could not compute COUNTY percentage share: {e})")

    # ---- FINISH ----
    print_section("FINISH")


# Entry point: ensures main() only runs when executed directly
if __name__ == "__main__":
    main()
