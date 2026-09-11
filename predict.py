import os
import re
import argparse
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Optional interactivity for Matplotlib bars
try:
    import mplcursors
    HAS_MPLCURSORS = True
except Exception:
    HAS_MPLCURSORS = False


# ----------------------------- helpers ----------------------------- #
def coerce_price(s: pd.Series) -> pd.Series:
    """Convert price-like strings (with €, commas) to numeric."""
    if np.issubdtype(s.dtype, np.number):
        return s.astype(float)
    s = (
        s.astype(str)
         .str.replace(r"[€,]", "", regex=True)
         .str.replace(r"[^\d.]", "", regex=True)
    )
    return pd.to_numeric(s, errors="coerce")


def find_date_col(df: pd.DataFrame) -> Optional[str]:
    for cand in ["sale_date", "date_of_sale", "date", "sale date", "date of sale"]:
        for c in df.columns:
            if c.lower().replace(" ", "_") == cand.replace(" ", "_"):
                return c
    # heuristic fallback
    for c in df.columns:
        ser = df[c].astype(str)
        if ser.str.contains(r"\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}", regex=True, na=False).mean() > 0.3:
            return c
    return None


def find_county_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if re.search(r"county", c, flags=re.I):
            return c
    return None


def normalize_county_name(s: pd.Series) -> pd.Series:
    """Normalize county names for joining to a GeoJSON/Shapefile attribute."""
    ser = s.astype(str).str.strip().str.lower()
    ser = ser.str.replace(r"^county\s+", "", regex=True)
    ser = ser.str.replace(r"[’'`]", "", regex=True)  # drop apostrophes
    ser = ser.str.replace(r"[^a-z\s-]", " ", regex=True)
    ser = ser.str.replace(r"\s+", " ", regex=True).str.strip()
    return ser


def add_size_numeric_if_available(df: pd.DataFrame, price_col: str, numeric_features: List[str]) -> List[str]:
    """If a size/area column exists and is mostly numeric, add it as an extra numeric feature."""
    for c in df.columns:
        if c == price_col:
            continue
        if re.search(r"(area|size|sqm|m2|m\u00b2)", c, flags=re.I):
            maybe = pd.to_numeric(df[c], errors="coerce")
            if maybe.notna().sum() > len(df) * 0.2:
                df[c + "_num"] = maybe
                numeric_features.append(c + "_num")
            break
    return numeric_features


def autodetect_features_csv(script_dir: Path) -> Path:
    """Prefer the ENGINEERED features file; else error with a clear message."""
    preferred = script_dir / "Property_Price_Register_Ireland-28-05-2021_features.csv"
    if preferred.exists():
        return preferred
    raise SystemExit(
        "Could not find 'Property_Price_Register_Ireland-28-05-2021_features.csv' in the script folder.\n"
        "Make sure you ran your feature_engineer step and saved the features CSV next to this script."
    )


def autodetect_geo(script_dir: Path) -> Optional[Path]:
    """Try to find a likely counties boundary file."""
    candidates = list(script_dir.glob("*.geojson")) + \
                 list(script_dir.glob("*.json")) + \
                 list(script_dir.glob("*.shp"))
    if not candidates:
        return None
    ranked = sorted(
        candidates,
        key=lambda p: (
            not re.search(r"(county|counties|admin|boundary|ireland)", p.name, re.I),
            p.suffix.lower() != ".geojson",
            p.name.lower()
        )
    )
    return ranked[0] if ranked else None


# ---------------------- hover helper (bars only) ---------------------- #
def _format_value(v: float, integer: bool) -> str:
    if integer:
        return f"{int(round(v)):,}"
    # decide decimals based on magnitude
    return f"{v:,.2f}" if abs(v) < 1e6 else f"{v:,.0f}"


def add_bar_hover(ax: plt.Axes, integer: bool = False, label_prefix: Optional[str] = None):
    """
    Attach hover tooltips to all Rectangle patches (bars) in the given axes.
    Requires mplcursors; otherwise no-op.
    """
    if not HAS_MPLCURSORS:
        print("[INFO] Install 'mplcursors' to enable bar hover tooltips: pip install mplcursors")
        return

    # Pre-cache tick positions and labels for mapping x -> label
    xticks = np.array(ax.get_xticks())
    xticklabels = [t.get_text() for t in ax.get_xticklabels()]

    def on_add(sel):
        bar = sel.artist  # Rectangle
        try:
            h = bar.get_height()
            xc = bar.get_x() + bar.get_width() / 2.0
            # map-x-to-nearest-tick-label (works for categorical bars too)
            lbl = ""
            if len(xticks) > 0 and len(xticklabels) > 0:
                idx = int(np.argmin(np.abs(xticks - xc)))
                if 0 <= idx < len(xticklabels):
                    lbl = xticklabels[idx]
            val_txt = _format_value(h, integer)
            if label_prefix and lbl:
                text = f"{label_prefix} {lbl}\nValue: {val_txt}"
            elif lbl:
                text = f"{lbl}: {val_txt}"
            else:
                text = f"{val_txt}"
            sel.annotation.set_text(text)
            sel.annotation.get_bbox_patch().set(alpha=0.9)
        except Exception:
            # fallback: just show height
            sel.annotation.set_text(_format_value(bar.get_height(), integer))
            sel.annotation.get_bbox_patch().set(alpha=0.9)

    patches = [p for p in ax.patches if hasattr(p, "get_height")]
    if patches:
        cursor = mplcursors.cursor(patches, hover=True)
        cursor.connect("add", on_add)


# ---------------------- choropleth (robust + interactive) ---------------------- #
def plot_choropleth_avg_price_by_county(
    df: pd.DataFrame,
    county_col: str,
    price_col: str,
    geo_path: str,
    map_out_path: Optional[str] = None,
    html_out_path: Optional[str] = None,
    geo_name_candidates: Tuple[str, ...] = (
        "name", "county", "countyname", "county_name",
        "NAME_1", "NAME_EN", "shapeName", "adm1_en", "adm1_name"
    )
):
    """
    Plot average sale price by county with:
      - robust name matching + aliasing
      - polygons with missing data filled with Cork average AND labeled as 'Cork'
      - static choropleth (matplotlib) and an interactive (Folium) map with pan/zoom
    """
    try:
        import geopandas as gpd
    except Exception:
        print("[INFO] GeoPandas not installed; skipping choropleth.")
        return

    if not geo_path:
        print("[INFO] No --geo provided; skipping choropleth.")
        return

    # --- ensure numeric price
    df = df.copy()
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

    # --- normalize CSV county names (force NA/blank to Cork)
    df_tmp = df[[county_col, price_col]].dropna(subset=[price_col]).copy()
    df_tmp[county_col] = df_tmp[county_col].fillna("Cork").replace(r"^\s*$", "Cork", regex=True)
    df_tmp["_county_norm"] = normalize_county_name(df_tmp[county_col])

    # --- load geo and pick county label column
    gdf = gpd.read_file(geo_path)
    geo_name_col = None
    lower_map = {c.lower(): c for c in gdf.columns}
    for cand in geo_name_candidates:
        if cand.lower() in lower_map:
            geo_name_col = lower_map[cand.lower()]
            break
    if geo_name_col is None:
        for c in gdf.columns:
            if gdf[c].dtype == object and c != gdf.geometry.name:
                geo_name_col = c
                break
    if geo_name_col is None:
        raise SystemExit("Could not identify a county name column in the geo file.")

    gdf = gdf.copy()
    gdf["_county_raw"] = gdf[geo_name_col].astype(str)
    gdf["_county_norm"] = normalize_county_name(gdf["_county_raw"])

    # --- alias map to improve matches
    alias_map = {
        # ROI punctuation/variants
        "dun laoghaire–rathdown": "dun laoghaire rathdown",
        "dun laoghaire - rathdown": "dun laoghaire rathdown",
        "dun laoghaire rathdown": "dun laoghaire rathdown",
        "cork city and county": "cork",
        "limerick city and county": "limerick",
        "waterford city and county": "waterford",
        # NI examples (only if your geo includes NI)
        "londonderry": "derry",
        "derry and strabane": "derry",
        "armagh city banbridge and craigavon": "armagh",
        "causeway coast and glens": "londonderry",
    }
    gdf["_county_norm"] = gdf["_county_norm"].map(lambda x: alias_map.get(x, x))
    df_tmp["_county_norm"] = df_tmp["_county_norm"].map(lambda x: alias_map.get(x, x))

    # --- compute averages
    avg_price = df_tmp.groupby("_county_norm", dropna=True)[price_col].mean().reset_index()

    # --- merge
    merged = gdf.merge(avg_price, how="left", on="_county_norm")

    # Track polygons with no data before fill
    missing_before_fill = merged[price_col].isna()

    # --- fill with Cork avg
    cork_avg = float(
        avg_price.loc[avg_price["_county_norm"] == "cork", price_col].mean()
    ) if "cork" in set(avg_price["_county_norm"]) else np.nan
    if np.isfinite(cork_avg):
        merged[price_col] = merged[price_col].fillna(cork_avg)
        fill_msg = f"Filled NA counties with Cork average: €{cork_avg:,.0f}"
    else:
        fill_msg = "Cork average not found in data; NA counties left as NA."

    # --- ensure display label says 'Cork' where we filled
    merged["_county_label"] = merged["_county_raw"]
    merged.loc[missing_before_fill, "_county_label"] = "Cork"

    # --- diagnostics
    csv_counties = set(avg_price["_county_norm"])
    geo_counties = set(gdf["_county_norm"])
    matched = sorted(csv_counties & geo_counties)
    only_in_csv = sorted(csv_counties - geo_counties)
    only_in_geo = sorted(geo_counties - csv_counties)

    print("\n[Choropleth diagnostics]")
    print(f"- Geo name column used: {geo_name_col}")
    print(f"- Counties with data that matched geometry: {len(matched)}")
    if only_in_csv:
        print(f"- In CSV but not in Geo (check naming): {only_in_csv}")
    if only_in_geo:
        print(f"- In Geo but not in CSV (labeled 'Cork' and filled with Cork avg if available): {only_in_geo}")
    print(f"- {fill_msg}")

    # --- STATIC map (matplotlib)
    fig, ax = plt.subplots()
    merged.plot(
        column=price_col,
        legend=True,
        ax=ax,
        cmap="viridis",
        edgecolor="white",
        linewidth=0.4,
        missing_kwds={"hatch": "///", "edgecolor": "lightgray", "label": "No data"},
    )
    ax.set_title("Average Sale Price by County")
    ax.set_axis_off()
    try:
        reps = merged.geometry.representative_point()
        for xy, name, val in zip(reps, merged["_county_label"], merged[price_col]):
            label = f"{name}\n€{val:,.0f}" if pd.notna(val) else f"{name}\n(n/a)"
            ax.annotate(text=label, xy=(xy.x, xy.y), ha="center", va="center", fontsize=6)
    except Exception as e:
        print(f"[INFO] Skipping labels due to geometry issue: {e}")
    plt.tight_layout()
    if map_out_path:
        plt.savefig(map_out_path, dpi=240, bbox_inches="tight")
        print(f"[INFO] Choropleth (static) saved to: {map_out_path}")
    plt.show()

    # --- INTERACTIVE map (Folium)
    if html_out_path:
        try:
            import folium
            from folium.features import GeoJson, GeoJsonTooltip
            from folium.plugins import Fullscreen, MousePosition, MiniMap

            m = folium.Map(
                location=[53.5, -8.0],
                zoom_start=6,
                tiles="CartoDB positron",
                control_scale=True,
                zoom_control=True,
                prefer_canvas=True,
                scrollWheelZoom=True,
                dragging=True,
            )

            def style_fn(feature):
                val = feature["properties"].get(price_col)
                return {
                    "fillOpacity": 0.75 if val is not None else 0.2,
                    "weight": 0.7,
                    "color": "#ffffff",
                }

            def highlight_fn(feature):
                return {"weight": 2.5, "color": "#333333"}

            gj = GeoJson(
                merged.to_json(),
                style_function=style_fn,
                highlight_function=highlight_fn,
                name="Average price by county",
                tooltip=GeoJsonTooltip(
                    fields=["_county_label", price_col],
                    aliases=["County", "Avg price (€)"],
                    sticky=True,
                    localize=True,
                ),
                zoom_on_click=True,
                smooth_factor=0.5,
            )
            gj.add_to(m)

            Fullscreen(position="topright").add_to(m)
            MiniMap(toggle_display=True).add_to(m)
            MousePosition(position="bottomleft", separator=" , ").add_to(m)

            try:
                bounds = merged.total_bounds  # minx, miny, maxx, maxy
                m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
            except Exception:
                pass

            m.save(html_out_path)
            print(f"[INFO] Interactive map saved to: {html_out_path}")
        except Exception as e:
            print(f"[WARN] Interactive map failed ({e}). Make sure 'folium' is installed.")


# ----------------------------- EDA & ML visuals (with hover + forecasts) ----------------------------- #
def get_property_type_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if re.search(r"(property.*type|property.*desc|type|house.*type)", c, flags=re.I):
            return c
    return None


def safe_month_year_quarter(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    out["_year"] = out[date_col].dt.year
    out["_month"] = out[date_col].dt.month
    out["_quarter"] = out[date_col].dt.quarter
    return out


def plot_all_requested_graphs(
    df: pd.DataFrame,
    price_col: str,
    date_col: Optional[str],
    df_full: Optional[pd.DataFrame] = None,
):
    """Build the requested charts, including multiple forecast models."""

    # ------------------------------------------------------------------
    # 1) Forecasts to 2030 (GBR, RF, Linear, Polynomial, ARIMA, Prophet)
    # ------------------------------------------------------------------
    if date_col:
        work = df.dropna(subset=[price_col]).copy()
        work["_year"] = work[date_col].dt.year

        # Use only years with enough data to be meaningful
        enough = work.groupby("_year")[price_col].size()
        valid_years = set(enough[enough >= 20].index)
        work = work[work["_year"].isin(valid_years)]

        if len(work) >= 100 and len(valid_years) >= 3:
            # Annual medians (time series base)
            med = work.groupby("_year")[price_col].median()
            years = med.index.values.astype(int)
            med_values = med.values

            start_year = int(years.min())
            end_year_train = int(years.max())
            end_year = max(2030, end_year_train)

            # For models that were fitted with feature names, use a DataFrame
            year_grid = np.arange(start_year, end_year + 1)
            year_grid_df = pd.DataFrame({"_year": year_grid})

            # ---------------- GBR + RF (tuned) ----------------
            Xy = work[["_year"]]
            yy = work[price_col]

            # Tuned Gradient Boosting
            gbr = GradientBoostingRegressor(
                random_state=42,
                n_estimators=500,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
            )
            gbr.fit(Xy, yy)

            # Tuned Random Forest
            rf = RandomForestRegressor(
                n_estimators=600,
                random_state=42,
                n_jobs=-1,
                max_depth=25,
                min_samples_split=4,
                min_samples_leaf=2,
                max_features="sqrt",
            )
            rf.fit(Xy, yy)

            # Use DataFrame with column "_year" to avoid feature-name warning
            y_pred_gbr = gbr.predict(year_grid_df)
            y_pred_rf = rf.predict(year_grid_df)

            plt.figure()
            plt.scatter(
                med.index,
                med.values,
                s=30,
                alpha=0.7,
                label="Median actual"
            )
            plt.plot(
                year_grid,
                y_pred_gbr,
                linewidth=2,
                color="#003f5c",
                label="Predicted (Gradient Boosting)"
            )
            plt.plot(
                year_grid,
                y_pred_rf,
                linewidth=2,
                linestyle="--",
                color="#ffa600",
                label="Predicted (Random Forest)"
            )
            plt.axvline(
                x=end_year_train,
                color="grey",
                linestyle=":",
                linewidth=1,
                label=f"Last observed year ({end_year_train})"
            )
            plt.title("Predicted Sale Price Trend by Year (GBR & RF, extended to 2030)")
            plt.xlabel("Year sold")
            plt.ylabel("Sale price (€)")
            plt.legend()
            plt.tight_layout()
            plt.show()

            # ---------------- Linear vs Polynomial Regression ----------
            X_lr = years.reshape(-1, 1)
            y_lr = med_values

            lin_reg = LinearRegression()
            lin_reg.fit(X_lr, y_lr)

            # For linear regression we can safely use ndarray
            year_grid_2d = year_grid.reshape(-1, 1)
            y_lin = lin_reg.predict(year_grid_2d)

            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_poly = poly.fit_transform(X_lr)
            lin_poly = LinearRegression()
            lin_poly.fit(X_poly, y_lr)
            y_poly = lin_poly.predict(poly.transform(year_grid_2d))

            plt.figure()
            plt.scatter(
                years,
                med_values,
                s=30,
                alpha=0.7,
                label="Median actual"
            )
            plt.plot(
                year_grid,
                y_lin,
                linewidth=2,
                label="Linear Regression"
            )
            plt.plot(
                year_grid,
                y_poly,
                linewidth=2,
                linestyle="--",
                label="Polynomial Regression (deg=2)"
            )
            plt.axvline(
                x=end_year_train,
                color="grey",
                linestyle=":",
                linewidth=1,
                label=f"Last observed year ({end_year_train})"
            )
            plt.title("Predicted Sale Price Trend by Year (Linear & Polynomial, to 2030)")
            plt.xlabel("Year sold")
            plt.ylabel("Sale price (€)")
            plt.legend()
            plt.tight_layout()
            plt.show()

            # ---------------- ARIMA forecast ---------------------------
            h = end_year - end_year_train
            if h > 0:
                try:
                    from statsmodels.tsa.arima.model import ARIMA

                    # Use a simple RangeIndex to avoid index warnings
                    y_series = pd.Series(med_values)
                    arima_model = ARIMA(y_series, order=(1, 1, 1))
                    arima_res = arima_model.fit()
                    arima_forecast = arima_res.forecast(steps=h)
                    future_years = np.arange(end_year_train + 1, end_year + 1)

                    plt.figure()
                    plt.scatter(years, med_values, s=30, alpha=0.7, label="Median actual")
                    plt.plot(years, med_values, linewidth=1.5, label="Historical median")
                    plt.plot(
                        future_years,
                        arima_forecast.values,
                        linewidth=2,
                        linestyle="--",
                        label="ARIMA forecast"
                    )
                    plt.axvline(
                        x=end_year_train,
                        color="grey",
                        linestyle=":",
                        linewidth=1,
                        label=f"Last observed year ({end_year_train})"
                    )
                    plt.title("Predicted Sale Price Trend by Year (ARIMA, to 2030)")
                    plt.xlabel("Year sold")
                    plt.ylabel("Sale price (€)")
                    plt.legend()
                    plt.tight_layout()
                    plt.show()
                except Exception as e:
                    print(f"[INFO] ARIMA forecast skipped: {e}")

            # ---------------- Prophet forecast -------------------------
            if h > 0:
                try:
                    try:
                        from prophet import Prophet
                    except ImportError:
                        from fbprophet import Prophet  # older package name

                    df_prophet = pd.DataFrame({
                        "ds": pd.to_datetime(years.astype(str) + "-01-01"),
                        "y": med_values,
                    })
                    m_prophet = Prophet(yearly_seasonality=False, weekly_seasonality=False,
                                        daily_seasonality=False)
                    m_prophet.fit(df_prophet)

                    # Use "YE" instead of deprecated "Y"
                    future = m_prophet.make_future_dataframe(periods=h, freq="YE")
                    forecast = m_prophet.predict(future)

                    # Extract only up to our desired end_year
                    forecast["year"] = forecast["ds"].dt.year
                    forecast_trim = forecast[forecast["year"] <= end_year]

                    plt.figure()
                    plt.scatter(years, med_values, s=30, alpha=0.7, label="Median actual")
                    plt.plot(
                        forecast_trim["year"],
                        forecast_trim["yhat"],
                        linewidth=2,
                        label="Prophet forecast"
                    )
                    plt.axvline(
                        x=end_year_train,
                        color="grey",
                        linestyle=":",
                        linewidth=1,
                        label=f"Last observed year ({end_year_train})"
                    )
                    plt.title("Predicted Sale Price Trend by Year (Prophet, to 2030)")
                    plt.xlabel("Year sold")
                    plt.ylabel("Sale price (€)")
                    plt.legend()
                    plt.tight_layout()
                    plt.show()
                except Exception as e:
                    print(f"[INFO] Prophet forecast skipped: {e}")
        else:
            print("[INFO] Skipping year prediction graphs (not enough data or year variety).")
    else:
        print("[INFO] Skipping year prediction graphs (no date column).")

    # ------------------------------------------------------------------
    # 2) Correlation heatmap (numeric columns)
    # ------------------------------------------------------------------
    num_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    if num_df.shape[1] >= 2:
        corr = num_df.corr(numeric_only=True)
        plt.figure()
        im = plt.imshow(corr.values, interpolation="nearest")
        plt.title("Correlation Heatmap (numeric variables)")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(range(corr.shape[1]), corr.columns, rotation=90)
        plt.yticks(range(corr.shape[0]), corr.index)
        plt.tight_layout()
        plt.show()
    else:
        print("[INFO] Skipping heatmap: not enough numeric columns.")

    # ------------------------------------------------------------------
    # 3) Time-based bar charts with hover
    # ------------------------------------------------------------------
    if date_col:
        dd = safe_month_year_quarter(df.dropna(subset=[date_col]).copy(), date_col)

        # Mean Sale Price by Month (with month names)
        month_stats = dd.groupby("_month")[price_col].mean()
        if len(month_stats) > 0:
            fig, ax = plt.subplots()

            xs = list(range(1, 13))
            vals = [month_stats.get(m, np.nan) for m in xs]

            month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

            ax.bar(xs, vals)
            ax.set_title("Mean Sale Price by Month")
            ax.set_xlabel("Month")
            ax.set_ylabel("Mean sale price (€)")

            ax.set_xticks(xs)
            ax.set_xticklabels(month_names, rotation=45)
            add_bar_hover(ax, integer=False)
            plt.tight_layout()
            plt.show()

        # Mean Sale Price by Year
        year_stats = dd.groupby("_year")[price_col].mean()
        if len(year_stats) > 0:
            fig, ax = plt.subplots()
            years_vals = list(map(int, year_stats.index))
            ax.bar(years_vals, year_stats.values)
            ax.set_title("Mean Sale Price by Year")
            ax.set_xlabel("Year")
            ax.set_ylabel("Mean sale price (€)")
            ax.set_xticks(years_vals)
            add_bar_hover(ax, integer=False)
            plt.tight_layout()
            plt.show()

        # Average purchases per quarter across years
        counts = dd.groupby(["_year", "_quarter"]).size().reset_index(name="n")
        avg_per_q = counts.groupby("_quarter")["n"].mean().reindex([1, 2, 3, 4])
        fig, ax = plt.subplots()
        ax.bar(avg_per_q.index.astype(int), avg_per_q.values)
        ax.set_title("Average Number of Purchases per Quarter (across years)")
        ax.set_xlabel("Quarter")
        ax.set_ylabel("Average transactions per year")
        ax.set_xticks([1, 2, 3, 4])
        add_bar_hover(ax, integer=False)
        plt.tight_layout()
        plt.show()
    else:
        print("[INFO] Skipping month/year/quarter charts: no date column.")

    # ------------------------------------------------------------------
    # 4) Property type distribution (ALL types, using FULL dataset)
    #     + merge "New Dwelling house /Apartment" and
    #       "New dwelling (House/Apartment)"
    # ------------------------------------------------------------------
    source_for_prop = df_full if df_full is not None else df
    prop_col = get_property_type_col(source_for_prop)
    if prop_col:
        ser = (
            source_for_prop[prop_col]
            .astype(str)
            .str.strip()
            .str.title()
        )

        # Normalise blanks to NaN
        ser = ser.replace({"": np.nan})

        # Merge the two variants into one label
        merge_map = {
            "New Dwelling House /Apartment": "New Dwelling (House/Apartment)",
            "New Dwelling (House/Apartment)": "New Dwelling (House/Apartment)",
        }
        ser = ser.replace(merge_map)

        vc = ser.dropna().value_counts()   # all types, not just top 20

        if len(vc) > 0:
            fig, ax = plt.subplots()
            # Use numeric x positions to avoid set_ticklabels warning
            x = np.arange(len(vc))
            ax.bar(x, vc.values)
            ax.set_title("Property Type Distribution (full dataset)")
            ax.set_xlabel("Property type")
            ax.set_ylabel("Count")
            ax.set_xticks(x)
            ax.set_xticklabels(vc.index, rotation=45, ha="right")
            add_bar_hover(ax, integer=True)
            plt.tight_layout()
            plt.show()
    else:
        print("[INFO] Skipping property type chart: no suitable column found.")


# ----------------------------- main ----------------------------- #
def main():
    script_dir = Path(__file__).resolve().parent

    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=False, help="Path to features CSV (optional if in same folder)")
    ap.add_argument("--n-rows", type=int, default=40000, help="Rows to sample for ML")
    ap.add_argument("--geo", dest="geo_path", default=None, help="Path to counties GeoJSON/Shapefile for choropleth")
    ap.add_argument("--map-out", dest="map_out", default=None, help="Optional path to save STATIC choropleth PNG")
    ap.add_argument("--html-out", dest="html_out", default=None, help="Optional path to save INTERACTIVE HTML map")
    args = ap.parse_args()

    # Auto-detect features dataset if --in not provided
    if args.in_path:
        csv_path = Path(args.in_path)
    else:
        csv_path = autodetect_features_csv(script_dir)
        print(f"[INFO] Using dataset: {csv_path.name}")

    # Auto-detect geo if --geo not provided
    if args.geo_path:
        geo_path = Path(args.geo_path)
    else:
        geo_auto = autodetect_geo(script_dir)
        geo_path = geo_auto if geo_auto is not None else None
        if geo_path:
            print(f"[INFO] Using geo file: {geo_path.name}")
        else:
            print("[INFO] No geo file found in script folder; choropleth will be skipped unless --geo is provided.")

    # Load data (features file)
    df_raw = pd.read_csv(csv_path, low_memory=True)

    # Identify key columns in the features dataset
    price_col = None
    for cand in ["sale_price", "price", "saleprice"]:
        for c in df_raw.columns:
            if c.lower() == cand:
                price_col = c
                break
        if price_col:
            break
    if price_col is None:
        price_col = next((c for c in df_raw.columns if re.search(r"price", c, flags=re.I)), None)
    if price_col is None:
        raise SystemExit("No 'sale_price' (or *price*) column found in the features file.")

    date_col = find_date_col(df_raw)
    county_col = find_county_col(df_raw)

    # Coerce price and parse date if present (on the full df)
    df_raw[price_col] = coerce_price(df_raw[price_col])

    if county_col:
        df_raw[county_col] = df_raw[county_col].fillna("Cork")
        df_raw[county_col] = df_raw[county_col].replace(r"^\s*$", "Cork", regex=True)

    if date_col:
        df_raw[date_col] = pd.to_datetime(df_raw[date_col], errors="coerce", utc=False)
        if "year_sold" not in df_raw:
            df_raw["year_sold"] = df_raw[date_col].dt.year
        if "month_sold" not in df_raw:
            df_raw["month_sold"] = df_raw[date_col].dt.month

    # ---------------------------------------------------------
    # df_full: all rows with price present (for property desc)
    # + outlier cleaning / normalisation for ML and charts
    # ---------------------------------------------------------
    df_full = df_raw.dropna(subset=[price_col]).copy()

    # === Outlier cleaning & normalisation ===
    # 1) Drop obviously invalid / tiny transactions (e.g. transfers)
    df_full = df_full[df_full[price_col] >= 20_000]

    # 2) Clip extreme high/low prices to the 1st and 99th percentiles
    q_low = df_full[price_col].quantile(0.01)
    q_high = df_full[price_col].quantile(0.99)
    df_full[price_col] = df_full[price_col].clip(lower=q_low, upper=q_high)

    # df: possibly sampled version for ML + other plots
    df = df_full.copy()
    if len(df) > args.n_rows:
        df = df.sample(n=args.n_rows, random_state=42)

    # Baseline ML feature setup
    numeric_features = [c for c in ["year_sold", "month_sold"] if c in df.columns]
    numeric_features = add_size_numeric_if_available(df, price_col, numeric_features)

    categorical_features: List[str] = []
    if county_col:
        categorical_features.append(county_col)
    for c in df.columns:
        if re.search(r"(property.*desc|property.*type|description)", c, flags=re.I):
            categorical_features.append(c)
            break

    X = df[numeric_features + categorical_features].copy()
    y = df[price_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    transformers = [("num", "passthrough", numeric_features)]
    if len(categorical_features) > 0:
        transformers.append(("cat", ohe, categorical_features))

    preprocess = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    # ---------------------------------------------------------
    # Compare performance of multiple models on same data
    # (MAE, RMSE, R^2 for each) + R^2 bar chart
    # ---------------------------------------------------------
    models = {
        "Random Forest (base)": RandomForestRegressor(
            n_estimators=600,
            random_state=42,
            n_jobs=-1,
            max_depth=25,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            random_state=42,
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
        ),
        "Linear Regression": LinearRegression(),
        # Fine-tuned Gradient Boosting as final/best model
        "Gradient Boosting (tuned final)": GradientBoostingRegressor(
            random_state=42,
            n_estimators=800,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.9,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=None,
        ),
    }

    model_names = []
    r2_scores = []

    print("\n=== Model comparison (same train/test split) ===")
    print(f"{'Model':25s}  {'MAE':>12s}  {'RMSE':>12s}  {'R^2':>8s}")
    print("-" * 65)

    for name, mdl in models.items():
        m_pipe = Pipeline([("prep", preprocess), ("model", mdl)])
        m_pipe.fit(X_train, y_train)
        preds_m = m_pipe.predict(X_test)

        mae_m = mean_absolute_error(y_test, preds_m)
        mse_m = mean_squared_error(y_test, preds_m)
        rmse_m = float(np.sqrt(mse_m))
        r2_m = r2_score(y_test, preds_m)

        model_names.append(name)
        r2_scores.append(r2_m)

        print(f"{name:25s}  {mae_m:12,.2f}  {rmse_m:12,.2f}  {r2_m:8.4f}")

    # Bar chart of R² scores
    fig, ax = plt.subplots()
    x = np.arange(len(model_names))
    bars = ax.bar(x, r2_scores)

    # Highlight base model (index 0) and final tuned model (last index)
    if len(bars) > 0:
        # Base model highlight
        bars[0].set_edgecolor("red")
        bars[0].set_linewidth(3)
        # Final tuned GB model highlight
        bars[-1].set_edgecolor("black")
        bars[-1].set_linewidth(3)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.set_ylabel("R² score (higher is better)")
    ax.set_title("Model Accuracy Comparison (R²)\nBase: Random Forest, Final: Tuned Gradient Boosting")

    add_bar_hover(ax, integer=False)
    plt.tight_layout()
    plt.show()

    # Build graphs (these now also use the cleaned df)
    plot_all_requested_graphs(df, price_col=price_col, date_col=date_col, df_full=df_full)

    # Choropleth maps (optional)
    if county_col and (geo_path := (Path(args.geo_path) if args.geo_path else autodetect_geo(script_dir))):
        try:
            plot_choropleth_avg_price_by_county(
                df=df,
                county_col=county_col,
                price_col=price_col,
                geo_path=str(geo_path),
                map_out_path=args.map_out,
                html_out_path=args.html_out
            )
        except Exception as e:
            print(f"[WARN] Choropleth failed: {e}")
    else:
        if not county_col:
            print("[INFO] No county column found; skipping choropleth.")
        if not (args.geo_path or autodetect_geo(script_dir)):
            print("[INFO] No geo file provided/found; skipping choropleth.")


if __name__ == "__main__":
    main()
