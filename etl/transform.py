"""
Transforms raw CSVs into a single, model‑ready table.

Inputs (data/extracted/):
  - weather_us_county_2018_2023.csv   (daily county weather from nClimGrid)
  - FAF5.7.1_State.csv                (state-to-state freight flows from FAF)

Output (data/processed/):
  - model_ready_state_year.csv         (one row per state-year with weather + freight)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import sys

sys.stdout.reconfigure(encoding="utf-8")


#swap state FIPS codes to USPS abbreviations for readability
STATE_FIPS_TO_ABBR = {
    "01":"AL","02":"AK","04":"AZ","05":"AR","06":"CA","08":"CO","09":"CT","10":"DE","11":"DC",
    "12":"FL","13":"GA","15":"HI","16":"ID","17":"IL","18":"IN","19":"IA","20":"KS","21":"KY",
    "22":"LA","23":"ME","24":"MD","25":"MA","26":"MI","27":"MN","28":"MS","29":"MO","30":"MT",
    "31":"NE","32":"NV","33":"NH","34":"NJ","35":"NM","36":"NY","37":"NC","38":"ND","39":"OH",
    "40":"OK","41":"OR","42":"PA","44":"RI","45":"SC","46":"SD","47":"TN","48":"TX","49":"UT",
    "50":"VT","51":"VA","53":"WA","54":"WV","55":"WI","56":"WY","72":"PR"
}

ROOT = Path(__file__).resolve().parents[1]
EXTRACTED = ROOT / "data" / "extracted"
PROCESSED = ROOT / "data" / "processed"

WEATHER_IN = EXTRACTED / "weather_us_county_2018_2023.csv"
FAF_IN = EXTRACTED / "FAF5.7.1_State.csv"
OUT = PROCESSED / "model_ready_state_year.csv"


# -----------------------------
# Weather: county‑daily -> state‑year
# -----------------------------
def build_weather_state_year(weather_csv: Path) -> pd.DataFrame:
    """
    Aggregate county‑daily weather to state‑year features.

    Returns one row per (state_fips, year) with engineered features like:
      - total/mean precipitation, 95th percentile precip
      - mean/std temperature
      - counts of hot/cold days, heavy/very‑heavy rain days
      - n_days coverage (for data completeness)
    """
    w = pd.read_csv(weather_csv, dtype={"county_fips": str})
    # minimal column validation
    need_cols = {"county_fips", "date", "prcp_mm", "tavg_c"}
    missing = need_cols - set(w.columns)
    if missing:
        raise ValueError(f"[WEATHER] Missing expected columns: {missing}")

    # Build join keys and ensure proper types
    w["date"] = pd.to_datetime(w["date"])
    w["state_fips"] = w["county_fips"].str.zfill(5).str[:2]
    w["year"] = w["date"].dt.year

    # Numeric conversions (coerce bad values to NaN)
    w["prcp_mm"] = pd.to_numeric(w["prcp_mm"], errors="coerce")
    w["tavg_c"] = pd.to_numeric(w["tavg_c"], errors="coerce")

    # Daily indicator features that can matter for logistics
    w["heavy_rain_day"] = (w["prcp_mm"] >= 10).astype(int)
    w["very_heavy_rain_day"] = (w["prcp_mm"] >= 25).astype(int)
    w["hot_day"] = (w["tavg_c"] >= 32).astype(int)
    w["cold_day"] = (w["tavg_c"] <= 0).astype(int)

    # State‑year aggregation: robust summary features
    agg = w.groupby(["state_fips","year"]).agg(
        prcp_total_mm = ("prcp_mm","sum"),
        prcp_mean_mm  = ("prcp_mm","mean"),
        prcp95_mm     = ("prcp_mm", lambda s: np.nanpercentile(s.dropna(),95) if s.notna().any() else np.nan),
        tavg_mean_c   = ("tavg_c","mean"),
        tavg_std_c    = ("tavg_c","std"),
        hot_days      = ("hot_day","sum"),
        cold_days     = ("cold_day","sum"),
        heavy_rain_days = ("heavy_rain_day","sum"),
        very_heavy_rain_days = ("very_heavy_rain_day","sum"),
        n_days        = ("tavg_c","count")  # how many daily rows exist (coverage)
    ).reset_index()

    # Human‑readable state abbreviation
    agg["state_abbr"] = agg["state_fips"].map(STATE_FIPS_TO_ABBR)
    print(f"[WEATHER] Built state‑year table: {agg.shape}")
    return agg


# -----------------------------
# FAF: memory‑safe state‑year aggregation
# -----------------------------
def build_faf_state_year(faf_csv: Path, years=tuple(range(2018, 2025))) -> pd.DataFrame:
    """
    Build state‑year freight metrics from FAF flows WITHOUT creating huge long tables.

    Steps:
      1) Choose origin/destination state code pair (prefer the more complete: dms_* vs fr_*).
      2) For each requested year, sum per origin (OUTBOUND) and per destination (INBOUND)
         across the year‑specific columns: tons_<year>, value_<year>, tmiles_<year>.
      3) Concatenate tiny per‑year tables and groupby to collapse duplicates.
      4) Merge inbound/outbound at the state‑year level.

    Result columns (present only if available in input):
      ['state_fips','year',
       'tons_out','value_out','tmiles_out',
       'tons_in','value_in','tmiles_in',
       'state_abbr']
    """
    f = pd.read_csv(faf_csv, low_memory=False)

    # --- Pick most complete origin/destination pair (case-insensitive) ---
    cols_lower = {c.lower(): c for c in f.columns}
    fr_orig = cols_lower.get("fr_orig")
    fr_dest = cols_lower.get("fr_dest")
    dms_orig = cols_lower.get("dms_origst")
    dms_dest = cols_lower.get("dms_destst")

    def nn(c): return f[c].notna().sum() if c in f.columns else 0
    fr_score  = (nn(fr_orig)  + nn(fr_dest)) if fr_orig and fr_dest else 0
    dms_score = (nn(dms_orig) + nn(dms_dest)) if dms_orig and dms_dest else 0

    if dms_score >= fr_score and dms_orig and dms_dest:
        orig_col, dest_col, pair = dms_orig, dms_dest, "dms_*"
    elif fr_orig and fr_dest:
        orig_col, dest_col, pair = fr_orig, fr_dest, "fr_*"
    else:
        raise ValueError("[FAF] No usable origin/destination columns (fr_* or dms_*).")

    print(f"[FAF] Using {pair}: {orig_col}, {dest_col}")

    # Coerce to numeric and drop rows with missing state codes; convert to 2‑digit FIPS strings
    f["orig_state_fips"] = pd.to_numeric(f[orig_col], errors="coerce")
    f["dest_state_fips"] = pd.to_numeric(f[dest_col], errors="coerce")
    before = len(f)
    f = f.dropna(subset=["orig_state_fips","dest_state_fips"])
    after = len(f)
    print(f"[FAF] Dropped {before - after} rows with missing state codes; remaining: {after}")

    f["orig_state_fips"] = f["orig_state_fips"].astype(int).astype(str).str.zfill(2)
    f["dest_state_fips"] = f["dest_state_fips"].astype(int).astype(str).str.zfill(2)

    # --- Discover available year columns for each metric; intersect with requested years ---
    def year_col_map(prefix: str) -> dict[int, str]:
        # match case-insensitively and ignore spaces
        out = {}
        for c in f.columns:
            cname = c.strip().lower()
            if cname.startswith(prefix):
                # expect 'prefixYYYY'
                try:
                    year = int(c.strip().split("_", 1)[1])
                    out[year] = c
                except Exception:
                    # if not strictly 'prefix_YYYY', try regex or skip
                    pass
        return out

    tons_map   = year_col_map("tons_")
    value_map  = year_col_map("value_")
    tmiles_map = year_col_map("tmiles_")

    avail_years = sorted(set(tons_map) | set(value_map) | set(tmiles_map))
    use_years = [y for y in years if y in avail_years]
    if not use_years:
        raise ValueError(f"[FAF] No FAF years found in requested {years}. Available: {avail_years}")

    print(f"[FAF] Years -> using {use_years}, available {avail_years[:10]}...")

    out_tables, in_tables = [], []

    def add_metric(direction: str, metric_name: str, y: int, series: pd.Series | None):
        """
        direction: 'out' or 'in'
        metric_name: 'tons'|'value'|'tmiles'
        y: year int
        series: pandas Series indexed by state_fips with summed values
        """
        if series is None:
            return
        dfm = series.reset_index()
        dfm.columns = ["state_fips", f"{metric_name}_{direction}"]
        dfm["year"] = y
        if direction == "out":
            out_tables.append(dfm)
        else:
            in_tables.append(dfm)

    # Iterate over years and aggregate by origin/destination
    for y in use_years:
        tcol = tons_map.get(y)
        vcol = value_map.get(y)
        mcol = tmiles_map.get(y)

        # OUTBOUND (by origin)
        tons_out   = f.groupby("orig_state_fips")[tcol].sum() if tcol else None
        value_out  = f.groupby("orig_state_fips")[vcol].sum() if vcol else None
        tmiles_out = f.groupby("orig_state_fips")[mcol].sum() if mcol else None
        add_metric("out", "tons",   y, tons_out)
        add_metric("out", "value",  y, value_out)
        add_metric("out", "tmiles", y, tmiles_out)

        # INBOUND (by destination)
        tons_in   = f.groupby("dest_state_fips")[tcol].sum() if tcol else None
        value_in  = f.groupby("dest_state_fips")[vcol].sum() if vcol else None
        tmiles_in = f.groupby("dest_state_fips")[mcol].sum() if mcol else None
        add_metric("in", "tons",   y, tons_in)
        add_metric("in", "value",  y, value_in)
        add_metric("in", "tmiles", y, tmiles_in)

    # Collapse tables by concatenation + groupby (avoids duplicate-column merge issues)
    def collapse_tables(tables: list[pd.DataFrame]) -> pd.DataFrame:
        if not tables:
            return pd.DataFrame(columns=["state_fips","year"])
        cat = pd.concat(tables, ignore_index=True)
        # Multiple rows per (state_fips, year) across metrics → sum safely
        agg = cat.groupby(["state_fips","year"], as_index=False).sum(numeric_only=True)
        return agg

    out_all = collapse_tables(out_tables)  # -> may contain tons_out/value_out/tmiles_out
    in_all  = collapse_tables(in_tables)   # -> may contain tons_in/value_in/tmiles_in

    # Join inbound & outbound → final FAF per state-year
    faf_state_year = out_all.merge(in_all, on=["state_fips","year"], how="outer")

    # Add USPS abbreviations
    faf_state_year["state_abbr"] = faf_state_year["state_fips"].map(STATE_FIPS_TO_ABBR)

    # Order columns consistently if present
    order = [
        "state_fips","year",
        "tons_out","value_out","tmiles_out",
        "tons_in","value_in","tmiles_in",
        "state_abbr",
    ]
    faf_state_year = faf_state_year[[c for c in order if c in faf_state_year.columns]] \
                        .sort_values(["state_fips","year"]).reset_index(drop=True)

    print(f"[FAF] Final FAF state‑year shape: {faf_state_year.shape}")
    return faf_state_year


# -----------------------------
# Orchestrator
# -----------------------------
def run():
    """
    Orchestration for the Transform step:
      - builds weather features by state‑year
      - builds freight metrics by state‑year
      - left‑joins them on (state_fips, year)
      - writes the final model‑ready CSV
    """
    PROCESSED.mkdir(parents=True, exist_ok=True)

    # Build both sides
    weather = build_weather_state_year(WEATHER_IN)
    faf = build_faf_state_year(FAF_IN)  # years=range(2018, 2025) by default

    # Left join: keep all freight rows; attach weather when available
    merged = faf.merge(weather, on=["state_fips","year"], how="left", suffixes=("_faf","_wx"))

    # Flag rows missing weather (can happen if year ranges don't overlap)
    merged["missing_weather"] = merged["n_days"].isna().astype(int)

    # Tidy state_abbr column (weather had one too)
    if "state_abbr_wx" in merged.columns and "state_abbr" in merged.columns:
        # Prefer the FAF one; fill with weather one if needed
        merged["state_abbr"] = merged["state_abbr"].fillna(merged["state_abbr_wx"])
        merged.drop(columns=["state_abbr_wx"], inplace=True)

    merged.to_csv(OUT, index=False)
    print(f"[OK] wrote {OUT} with shape {merged.shape}")


if __name__ == "__main__":
    run()