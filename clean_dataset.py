import argparse
import os
import re
import unicodedata
import pandas as pd
import numpy as np


# ------------------------------- helpers ------------------------------------ #
def normalize_unicode(s: str) -> str:
    # Basic Unicode normalisation and whitespace cleanup.
    # - Normalises characters (e.g. accented forms)
    # - Collapses multiple spaces into a single space
    if not isinstance(s, str):
        return s
    s_norm = unicodedata.normalize("NFKC", s)
    s_norm = re.sub(r"\s+", " ", s_norm.strip())
    return s_norm


def apply_replacements(s: str, replacements: dict) -> str:
    # Replace exact substrings based on a mapping dictionary.
    # Matching is case-insensitive but the rest of the string is preserved.
    if not isinstance(s, str):
        return s
    s_work = s
    for bad, good in replacements.items():
        s_work = re.sub(re.escape(bad), good, s_work, flags=re.IGNORECASE)
    return s_work


def translate_gaelic_phrases(s: str) -> str:
    # Translate known Irish phrases and mojibake variants to English.
    # This is mainly used for:
    #   - 'Ní Bhaineann' -> 'Not applicable'
    #   - 'Baile Átha Cliath <num>' -> 'Dublin <num>'
    #   - Dwelling-related phrases (Teach/Árasán Cónaithe ...)
    # Note: size/area translations were removed; this focuses on dwelling/type text.
    if not isinstance(s, str):
        return s

    original = s
    s = normalize_unicode(s)

    # Fix common mojibake fragments for Irish diacritics BEFORE mapping phrases.
    mojibake_fixes = {
        "�": "",        # drop unknown replacement char
        "?": "",        # drop stray '?' from bad decodes
        "n�": "ní",
        "Ni ": "Ní ",
        "N� ": "Ní ",
        "c?naithe": "cónaithe",
        "�ras�n": "árasán",
        "?ras?n": "árasán",
        "Teach/": "Teach/",  # keep as-is but ensures it's present in the string
    }
    s = apply_replacements(s, mojibake_fixes)

    # --- Dublin & 'Ní Bhaineann' handling (regex) ---

    # 1) Any form of "Ní Bhaineann" -> "Not applicable"
    s = re.sub(r"\bN[ií]\s+Bhaineann\b", "Not applicable", s, flags=re.IGNORECASE)

    # 2) "Baile Átha/Atha/?tha/�tha Cliath <num>" -> "Dublin <num>"
    s = re.sub(
        r"\bBaile\s*[ÁA?�]tha\s*Cliath\s*(\d+)\b",
        r"Dublin \1",
        s,
        flags=re.IGNORECASE
    )

    # 3) Base "Baile Átha/Atha/?tha/�tha Cliath" (no number) -> "Dublin"
    s = re.sub(
        r"\bBaile\s*[ÁA?�]tha\s*Cliath\b",
        "Dublin",
        s,
        flags=re.IGNORECASE
    )

    # Phrase-level translations for dwelling descriptions
    phrase_map = {
        "Teach/Árasán Cónaithe Nua": "New dwelling (House/Apartment)",
        "Teach/Arasán Cónaithe Nua": "New dwelling (House/Apartment)",
        "Teach/Arasan Conaithe Nua": "New dwelling (House/Apartment)",
        "Teach/Árasán Cónaithe Athúsáide": "Second-hand dwelling (House/Apartment)",
        "Teach/Árasán Cónaithe Athchóirithe": "Refurbished dwelling (House/Apartment)",
    }

    # Raw variants of phrases that appear in the original data (safety net)
    raw_variants = {
        "Teach/�ras�n C�naithe Ath�imhe": "Existing/second-hand dwelling (House/Apartment)",
        "Teach/�ras�n C�naithe Nua": "New dwelling (House/Apartment)",
        "Teach/?ras?n C?naithe Nua": "New dwelling (House/Apartment)",
        "N� Bhaineann": "Not applicable",
        "Baile �tha Cliath 3": "Dublin 3",
        "Baile �tha Cliath 18": "Dublin 18",
        "Baile ?tha Cliath 17": "Dublin 17",
        "Baile �tha Cliath 14": "Dublin 14",
        "Baile �tha Cliath 5": "Dublin 5",
        "Baile �tha Cliath 15": "Dublin 15",
        "Baile �tha Cliath 4": "Dublin 4",
        "Baile �tha Cliath 9": "Dublin 9",
    }

    # If the whole (normalised) string exactly matches a known phrase, map it.
    for ga, en in phrase_map.items():
        if s.lower() == ga.lower():
            return en

    # If the original raw text exactly matches a raw variant, map it.
    for ga, en in raw_variants.items():
        if original.strip().lower() == ga.strip().lower():
            return en

    # Word-level substitutions for dwellings (non-size tokens only)
    token_map = {
        "Teach": "House",
        "Árasán": "Apartment",
        "Arasán": "Apartment",
        "Arasan": "Apartment",
        "Cónaithe": "Residential",
        "Conaithe": "Residential",
        "Nua": "New",
        "Athúsáide": "Second-hand",
        "Athchóirithe": "Refurbished",
    }

    if token_map:
        def replace_tokens(text: str) -> str:
            # Replace any token found in token_map, preserving other words.
            def sub_one(m):
                token = m.group(0)
                return token_map.get(token, token)

            # Build regex to match any of the tokens as whole words
            pattern = r"\b(" + "|".join(
                map(re.escape, sorted(token_map.keys(), key=len, reverse=True))
            ) + r")\b"
            return re.sub(pattern, sub_one, text)

        s = replace_tokens(s)

    # Final whitespace tidy
    s = re.sub(r"\s+", " ", s).strip()
    return s


def to_snake_case(name: str) -> str:
    # Convert a column name to snake_case:
    # - Normalise Unicode
    # - Replace non-alphanumeric characters with underscores
    # - Collapse multiple underscores and lower-case the result
    if not isinstance(name, str):
        return name
    name = unicodedata.normalize("NFKC", name)
    name = re.sub(r"[^\w]+", "_", name.strip())
    name = re.sub(r"_+", "_", name).strip("_").lower()
    return name


def try_parse_dates(df: pd.DataFrame, date_like_cols=None) -> pd.DataFrame:
    # Attempt to parse any columns that look like dates into datetime.
    if date_like_cols is None:
        date_like_cols = [c for c in df.columns if "date" in c.lower()]
    for c in date_like_cols:
        try:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=False)
        except Exception:
            # If parsing fails, leave column unchanged
            pass
    return df


def coerce_numeric(series: pd.Series):
    # Convert currency/number strings (e.g. '€123,456.78') to numeric values.
    # If the series is already numeric, just coerce directly.
    if not pd.api.types.is_string_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    s = series.astype(str)
    # Remove currency symbols and commas
    s = s.str.replace(r"[€$,]", "", regex=True)
    # Remove whitespace
    s = s.str.replace(r"\s+", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


# ------------------------------- main --------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    # Default path: same folder as this script, using the original raw CSV
    default_path = os.path.join(
        os.path.dirname(__file__),
        "Property_Price_Register_Ireland-28-05-2021.csv"
    )
    ap.add_argument(
        "--path",
        default=default_path,
        help="Path to the CSV file (default: same folder)"
    )
    ap.add_argument(
        "--sep",
        default=",",
        help="CSV separator (default ,)"
    )
    args = ap.parse_args()

    in_path = args.path
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"File not found: {in_path}")

    # Load the raw CSV file
    df = pd.read_csv(
        in_path,
        sep=args.sep,
        encoding="utf-8",
        engine="python",
        on_bad_lines="skip"
    )

    # Normalise column names to snake_case
    df.columns = [to_snake_case(c) for c in df.columns]

    # --- Remove unwanted columns if present ---
    # We drop property_size_desc, postal_code and address-related columns
    # because they are not needed (or too detailed) for the analysis.
    drop_exact = {
        "property_size_desc",
        "postal_code",
        "address",           # requested column to drop
        "property_address",  # common variant
        "full_address"       # common variant
    }
    drop_cols = [c for c in df.columns if c in drop_exact]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"Dropped columns: {', '.join(drop_cols)}")

    # Detect which columns are text/object type
    text_cols = [c for c in df.columns if df[c].dtype == object]

    # 1) Normalise all text columns (Unicode and whitespace)
    for c in text_cols:
        df[c] = df[c].apply(normalize_unicode)

    # 2) Translate Irish phrases in all text columns
    #    (dwelling types, 'Baile Átha Cliath', 'Ní Bhaineann', etc.)
    for c in text_cols:
        df[c] = df[c].apply(translate_gaelic_phrases)

    # 2.5) Convert 'Not applicable' → NaN in any postal_code-like columns (if any remain).
    postal_candidates = [
        c for c in df.columns
        if "postal" in c.lower() or "postcode" in c.lower()
    ]
    if postal_candidates:
        postal_col = postal_candidates[0]
        df[postal_col] = df[postal_col].replace(
            to_replace=r"(?i)^\s*not\s+applicable\s*$",
            value=np.nan,
            regex=True
        )

    # 3) Parse date-like columns into datetime
    df = try_parse_dates(df)

    # 4) Convert likely numeric columns (price, amount, eur, etc.) to numeric types
    numeric_name_hints = ["price", "amount", "eur", "euro", "value", "cost", "sq", "sqm", "m2", "area"]
    for c in df.columns:
        if any(h in c for h in numeric_name_hints):
            df[c] = coerce_numeric(df[c])

    # 5) Final text cleanup: strip stray spaces again
    for c in text_cols:
        df[c] = df[c].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # Save the cleaned CSV next to the original, with "_cleaned" suffix
    root, ext = os.path.splitext(in_path)
    out_path = f"{root}_cleaned.csv"
    df.to_csv(out_path, index=False)
    print(f"Cleaned file written to: {out_path}")


if __name__ == "__main__":
    main()
