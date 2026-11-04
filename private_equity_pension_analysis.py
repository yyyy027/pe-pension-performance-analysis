from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

API_URL = "https://publicplansdata.org/api/"

START_YEAR = 2001
END_YEAR = 2025

BASE_DIR = Path.home() / ".spyder-py3"
OUTPUT_DIR = BASE_DIR / "private_equity_research"

TIMEOUT = 120

VARIABLES = [
    "PlanName",
    "fy",
    "PETotal_Rtrn",
    "PETotal_Actl",
    "EQTotal_Rtrn",
    "EQTotal_Actl",
    "RETotal_Rtrn",
    "HFTotal_Rtrn",
]

RETURN_COLUMNS = [
    "PETotal_Rtrn",
    "EQTotal_Rtrn",
    "RETotal_Rtrn",
    "HFTotal_Rtrn",
]

ALLOCATION_COLUMNS = [
    "PETotal_Actl",
    "EQTotal_Actl",
]

NUMERIC_COLUMNS = RETURN_COLUMNS + ALLOCATION_COLUMNS


# ============================================================
# 2. RESET OUTPUT DIRECTORY
# ============================================================

def reset_output_directory() -> None:
    """Delete and recreate the dedicated research output folder."""

    expected_parent = BASE_DIR.resolve()
    target = OUTPUT_DIR.resolve()

    if target.parent != expected_parent:
        raise RuntimeError("Unsafe deletion path.")

    if target.name != "private_equity_research":
        raise RuntimeError("Unexpected output directory.")

    if target.exists():
        shutil.rmtree(target)
        print(f"Deleted previous folder: {target}")

    target.mkdir(parents=True, exist_ok=True)

    print(f"Created project folder: {target}")


# ============================================================
# 3. API REQUEST
# ============================================================

def api_request(
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    """Execute an API request and validate the JSON response."""

    response = requests.get(
        API_URL,
        params=params,
        timeout=TIMEOUT,
    )

    response.raise_for_status()

    try:
        result = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Invalid API JSON response: {response.text[:500]}"
        ) from exc

    if not isinstance(result, list) or not result:
        raise RuntimeError(
            f"Unexpected API response: {str(result)[:500]}"
        )

    return result


# ============================================================
# 4. VALIDATE VARIABLES
# ============================================================

def validate_api_variables() -> list[str]:
    """Check that all required fields exist in the API."""

    print("\nChecking available API variables...")

    response = api_request({
        "q": "ListVariables",
        "format": "json",
    })

    available_fields = {
        str(row.get("FieldName", "")).strip()
        for row in response[1:]
        if isinstance(row, dict)
    }

    missing = [
        column
        for column in VARIABLES
        if column not in available_fields
    ]

    if missing:
        raise ValueError(
            f"Missing API variables: {missing}"
        )

    print(f"Validated {len(VARIABLES)} required variables.")

    # Identify an optional pension-plan identifier.
    lookup = {
        field.lower(): field
        for field in available_fields
    }

    candidates = ["ppd_id", "PPDID", "PPD_ID"]

    plan_id = next(
        (
            lookup[candidate.lower()]
            for candidate in candidates
            if candidate.lower() in lookup
        ),
        None,
    )

    if plan_id is None:
        raise ValueError(
            "No pension plan identifier found in API metadata."
        )

    requested = VARIABLES + [plan_id]

    print(f"Using plan identifier: {plan_id}")

    return requested


# ============================================================
# 5. DOWNLOAD DATA
# ============================================================

def download_data() -> pd.DataFrame:
    """Download pension investment data for the requested period."""

    requested_variables = validate_api_variables()

    print("\nDownloading Public Plans Database data...")

    response = api_request({
        "q": "QVariables",
        "variables": ",".join(requested_variables),
        "filterfystart": START_YEAR,
        "filterfyend": END_YEAR,
        "format": "json",
    })

    records = response[1:]

    if not records:
        raise RuntimeError("API returned no observations.")

    df = pd.DataFrame(records)

    missing = set(requested_variables) - set(df.columns)

    if missing:
        raise ValueError(
            f"API response missing columns: {missing}"
        )

    print("Downloaded raw data shape:", df.shape)

    return df


# ============================================================
# 6. CLEAN AND VALIDATE DATA
# ============================================================

def clean_data(
    raw_df: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    """
    Remove empty records and combine complementary plan-year rows.

    Conflicting non-null values are flagged, never silently averaged.
    """

    df = raw_df.copy()

    required = [
        "ppd_id",
        "PlanName",
        "fy",
        *NUMERIC_COLUMNS,
    ]

    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # Validate identifiers.
    df["PlanName"] = (
        df["PlanName"]
        .astype("string")
        .str.strip()
    )

    df["ppd_id"] = pd.to_numeric(
        df["ppd_id"],
        errors="coerce",
    )

    df["fy"] = pd.to_numeric(
        df["fy"],
        errors="coerce",
    )

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    df = df.dropna(
        subset=["ppd_id", "PlanName", "fy"]
    )

    df["fy"] = df["fy"].astype(int)

    df = df[
        df["fy"].between(START_YEAR, END_YEAR)
    ].copy()

    # Remove records with no investment information.
    before_empty_removal = len(df)

    df = df.dropna(
        subset=NUMERIC_COLUMNS,
        how="all",
    )

    empty_rows_removed = (
        before_empty_removal - len(df)
    )

    # Check whether a plan ID maps to multiple plan names.
    name_counts = (
        df.groupby("ppd_id")["PlanName"]
        .nunique()
    )

    invalid_ids = name_counts[name_counts > 1]

    if not invalid_ids.empty:
        raise ValueError(
            "Some pension IDs map to multiple plan names. "
            f"IDs: {invalid_ids.index.tolist()[:20]}"
        )

    # Identify duplicate plan-year groups.
    group_keys = ["ppd_id", "fy"]

    duplicate_mask = df.duplicated(
        subset=group_keys,
        keep=False,
    )

    duplicate_groups = (
        df.loc[duplicate_mask, group_keys]
        .drop_duplicates()
        .shape[0]
    )

    print("\nDATA CLEANING")
    print("Empty investment rows removed:", empty_rows_removed)
    print("Duplicate plan-year groups:", duplicate_groups)

    # Detect conflicts in non-null numeric observations.
    conflicts = (
        df.groupby(group_keys)[NUMERIC_COLUMNS]
        .nunique(dropna=True)
    )

    conflict_mask = conflicts.gt(1).any(axis=1)

    conflict_count = int(conflict_mask.sum())

    print("Conflicting plan-year groups:", conflict_count)

    if conflict_count > 0:

        conflict_keys = (
            conflicts.loc[conflict_mask]
            .reset_index()[group_keys]
        )

        conflict_details = df.merge(
            conflict_keys,
            on=group_keys,
            how="inner",
        )

        conflict_path = (
            OUTPUT_DIR / "conflicting_plan_year_records.csv"
        )

        conflict_details.to_csv(
            conflict_path,
            index=False,
        )

        raise ValueError(
            f"{conflict_count} conflicting plan-year groups found. "
            f"Review: {conflict_path}"
        )

    # Combine complementary records.
    # GroupBy.first selects the first non-null value in each column.
    result = (
        df.sort_values(group_keys)
        .groupby(
            group_keys,
            as_index=False,
            sort=True,
        )
        .agg({
            "PlanName": "first",
            **{
                column: "first"
                for column in NUMERIC_COLUMNS
            },
        })
    )

    # Final integrity checks.
    if result.duplicated(group_keys).any():
        raise RuntimeError(
            "Duplicate plan-year records remain."
        )

    if result[NUMERIC_COLUMNS].isna().all(axis=1).any():
        raise RuntimeError(
            "Empty investment observations remain."
        )

    if result["ppd_id"].nunique() == 0:
        raise RuntimeError("No pension plans remain.")

    result = result.sort_values(
        group_keys
    ).reset_index(drop=True)

    print("\nCLEANED DATA")
    print("Shape:", result.shape)
    print("Unique plans:", result["ppd_id"].nunique())
    print(
        "Year range:",
        result["fy"].min(),
        "-",
        result["fy"].max(),
    )

    print("\nReturn validation:")
    print(result[RETURN_COLUMNS].describe())

    print("\nAllocation validation:")
    print(result[ALLOCATION_COLUMNS].describe())

    return result, "ppd_id"


# ============================================================
# 7. ANNUAL PERFORMANCE SUMMARY
# ============================================================

def create_annual_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate annual asset-class return and allocation statistics."""

    annual = (
        df.groupby("fy")
        .agg(
            PE_Median=("PETotal_Rtrn", "median"),
            PE_Mean=("PETotal_Rtrn", "mean"),
            PE_Observations=("PETotal_Rtrn", "count"),
            Public_Equity_Median=("EQTotal_Rtrn", "median"),
            Real_Estate_Median=("RETotal_Rtrn", "median"),
            Hedge_Fund_Median=("HFTotal_Rtrn", "median"),
            Median_PE_Allocation=("PETotal_Actl", "median"),
        )
        .reset_index()
    )

    return annual


# ============================================================
# 8. PAIRED PE VS PUBLIC EQUITY
# ============================================================

def create_paired_comparison(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare PE and public equity for the same plan and fiscal year.
    """

    paired = df.dropna(
        subset=[
            "PETotal_Rtrn",
            "EQTotal_Rtrn",
        ]
    ).copy()

    paired["Excess_Return"] = (
        paired["PETotal_Rtrn"]
        - paired["EQTotal_Rtrn"]
    )

    annual_paired = (
        paired.groupby("fy")
        .agg(
            Median_PE_Return=("PETotal_Rtrn", "median"),
            Median_Public_Return=("EQTotal_Rtrn", "median"),
            Median_Excess_Return=("Excess_Return", "median"),
            Mean_Excess_Return=("Excess_Return", "mean"),
            Number_of_Plan_Observations=("Excess_Return", "count"),
        )
        .reset_index()
    )

    return paired, annual_paired


# ============================================================
# 9. PLAN-LEVEL STATISTICS
# ============================================================

def create_plan_statistics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate descriptive PE statistics for each pension plan."""

    stats = (
        df.groupby("ppd_id")
        .agg(
            PlanName=("PlanName", "first"),
            Observations=("PETotal_Rtrn", "count"),
            Average_PE_Return=("PETotal_Rtrn", "mean"),
            Median_PE_Return=("PETotal_Rtrn", "median"),
            PE_Return_Std=("PETotal_Rtrn", "std"),
            Average_PE_Allocation=("PETotal_Actl", "mean"),
        )
        .reset_index()
    )

    return stats


# ============================================================
# 10. ROLLING FIVE-YEAR ANNUALIZED RETURNS
# ============================================================

def calculate_rolling_returns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate five-year annualized geometric returns.

    Requires five consecutive fiscal years and five non-null returns.
    """

    results = []

    assets = {
        "PE": "PETotal_Rtrn",
        "Public_Equity": "EQTotal_Rtrn",
    }

    for plan_id, group in df.groupby(
        "ppd_id",
        sort=False,
    ):

        group = group.sort_values("fy").copy()

        years = group["fy"].to_numpy()

        for asset_name, column in assets.items():

            returns = group[column].to_numpy(
                dtype=float
            )

            # A valid five-year window requires:
            # - five consecutive fiscal years
            # - five non-null returns
            # - positive cumulative wealth factor

            valid = np.isfinite(returns)

            growth = np.where(
                valid,
                1.0 + returns,
                np.nan,
            )

            rolling_product = (
                pd.Series(growth)
                .rolling(5, min_periods=5)
                .apply(np.prod, raw=True)
                .to_numpy()
            )

            consecutive = np.zeros(
                len(years),
                dtype=bool,
            )

            for i in range(4, len(years)):
                consecutive[i] = (
                    np.all(
                        np.diff(years[i - 4:i + 1]) == 1
                    )
                )

            valid_window = (
                consecutive
                & np.isfinite(rolling_product)
                & (rolling_product > 0)
            )

            positions = np.flatnonzero(valid_window)

            if len(positions) == 0:
                continue

            annualized = (
                rolling_product[positions] ** 0.2
            ) - 1.0

            results.append(
                pd.DataFrame({
                    "ppd_id": plan_id,
                    "PlanName": group["PlanName"].iloc[0],
                    "End_Year": years[positions],
                    "Asset_Class": asset_name,
                    "Annualized_5Y_Return": annualized,
                })
            )

    if not results:
        return pd.DataFrame(
            columns=[
                "ppd_id",
                "PlanName",
                "End_Year",
                "Asset_Class",
                "Annualized_5Y_Return",
            ]
        )

    return pd.concat(
        results,
        ignore_index=True,
    )


# ============================================================
# 11. ALLOCATION ANALYSIS
# ============================================================

def create_allocation_analysis(
    df: pd.DataFrame,
) -> tuple[float, pd.DataFrame]:
    """Examine the association between PE allocation and PE returns."""

    data = df.dropna(
        subset=[
            "PETotal_Actl",
            "PETotal_Rtrn",
        ]
    ).copy()

    correlation = data[
        ["PETotal_Actl", "PETotal_Rtrn"]
    ].corr().iloc[0, 1]

    allocation_bins = [
        -np.inf,
        0.05,
        0.10,
        0.15,
        0.20,
        np.inf,
    ]

    allocation_labels = [
        "0–5%",
        "5–10%",
        "10–15%",
        "15–20%",
        "Above 20%",
    ]

    data["Allocation_Band"] = pd.cut(
        data["PETotal_Actl"],
        bins=allocation_bins,
        labels=allocation_labels,
        include_lowest=True,
    )

    summary = (
        data.groupby(
            "Allocation_Band",
            observed=False,
        )
        .agg(
            Median_PE_Return=("PETotal_Rtrn", "median"),
            Mean_PE_Return=("PETotal_Rtrn", "mean"),
            Observations=("PETotal_Rtrn", "count"),
        )
        .reset_index()
    )

    return float(correlation), summary


# ============================================================
# 12. CORRELATION ANALYSIS
# ============================================================

def create_correlation_analysis(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate asset-class return correlations."""

    return df[
        RETURN_COLUMNS
    ].corr(method="pearson")


# ============================================================
# 13. CHART GENERATION
# ============================================================

def create_charts(
    df: pd.DataFrame,
    annual: pd.DataFrame,
    rolling: pd.DataFrame,
) -> None:

    # --------------------------------------------------------
    # Chart 1: Annual PE vs public equity
    # --------------------------------------------------------

    plt.figure(figsize=(12, 6))

    plt.plot(
        annual["fy"],
        annual["PE_Median"] * 100,
        marker="o",
        label="Private Equity",
    )

    plt.plot(
        annual["fy"],
        annual["Public_Equity_Median"] * 100,
        marker="o",
        label="Public Equity",
    )

    plt.title("Annual Median Returns: PE vs Public Equity")
    plt.xlabel("Fiscal Year")
    plt.ylabel("Return (%)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "annual_return_comparison.png",
        dpi=200,
    )

    plt.close()

    # --------------------------------------------------------
    # Chart 2: PE return distribution
    # --------------------------------------------------------

    pe = df["PETotal_Rtrn"].dropna() * 100

    if not pe.empty:

        plt.figure(figsize=(10, 6))

        plt.hist(
            pe,
            bins=35,
            edgecolor="black",
        )

        plt.title("Distribution of Annual PE Returns")
        plt.xlabel("Annual Return (%)")
        plt.ylabel("Frequency")
        plt.grid(alpha=0.2)
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "pe_return_distribution.png",
            dpi=200,
        )

        plt.close()

    # --------------------------------------------------------
    # Chart 3: PE allocation vs PE return
    # --------------------------------------------------------

    scatter = df.dropna(
        subset=[
            "PETotal_Actl",
            "PETotal_Rtrn",
        ]
    )

    if not scatter.empty:

        plt.figure(figsize=(10, 6))

        plt.scatter(
            scatter["PETotal_Actl"] * 100,
            scatter["PETotal_Rtrn"] * 100,
            alpha=0.35,
        )

        plt.title("PE Allocation vs PE Return")
        plt.xlabel("PE Allocation (%)")
        plt.ylabel("PE Annual Return (%)")
        plt.grid(alpha=0.2)
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "allocation_vs_return.png",
            dpi=200,
        )

        plt.close()

    # --------------------------------------------------------
    # Chart 4: Rolling five-year returns
    # --------------------------------------------------------

    if not rolling.empty:

        plt.figure(figsize=(10, 6))

        for asset in ["PE", "Public_Equity"]:

            values = (
                rolling.loc[
                    rolling["Asset_Class"] == asset,
                    "Annualized_5Y_Return",
                ].dropna() * 100
            )

            if not values.empty:

                plt.hist(
                    values,
                    bins=30,
                    alpha=0.5,
                    label=asset,
                )

        plt.title("Rolling Five-Year Annualized Returns")
        plt.xlabel("Annualized Return (%)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(alpha=0.2)
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "five_year_return_distribution.png",
            dpi=200,
        )

        plt.close()

    # --------------------------------------------------------
    # Chart 5: Annual PE allocation trend
    # --------------------------------------------------------

    allocation_trend = (
        df.groupby("fy")["PETotal_Actl"]
        .median()
        .dropna()
    )

    if not allocation_trend.empty:

        plt.figure(figsize=(12, 6))

        plt.plot(
            allocation_trend.index,
            allocation_trend.values * 100,
            marker="o",
        )

        plt.title("Median PE Allocation Across Pension Plans")
        plt.xlabel("Fiscal Year")
        plt.ylabel("PE Allocation (%)")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "pe_allocation_trend.png",
            dpi=200,
        )

        plt.close()


# ============================================================
# 14. RESEARCH SUMMARY
# ============================================================

def create_research_summary(
    df: pd.DataFrame,
    paired: pd.DataFrame,
    allocation_correlation: float,
    raw_count: int,
) -> dict[str, Any]:

    paired_correlation = (
        paired["PETotal_Rtrn"].corr(
            paired["EQTotal_Rtrn"]
        )
        if len(paired) >= 2
        else np.nan
    )

    summary = {
        "Data_Source": "Public Plans Database",
        "API": API_URL,
        "Coverage_Start": int(df["fy"].min()),
        "Coverage_End": int(df["fy"].max()),
        "Unique_Plans": int(df["ppd_id"].nunique()),
        "Raw_API_Observations": int(raw_count),
        "Cleaned_Plan_Year_Observations": int(len(df)),
        "PE_Return_Observations": int(
            df["PETotal_Rtrn"].count()
        ),
        "Public_Equity_Return_Observations": int(
            df["EQTotal_Rtrn"].count()
        ),
        "Median_PE_Annual_Return": float(
            df["PETotal_Rtrn"].median()
        ),
        "Mean_PE_Annual_Return": float(
            df["PETotal_Rtrn"].mean()
        ),
        "Median_Public_Equity_Annual_Return": float(
            df["EQTotal_Rtrn"].median()
        ),
        "Median_PE_Allocation": float(
            df["PETotal_Actl"].median()
        ),
        "PE_Public_Equity_Correlation_Paired": (
            float(paired_correlation)
            if pd.notna(paired_correlation)
            else None
        ),
        "PE_Allocation_Return_Correlation": (
            float(allocation_correlation)
            if pd.notna(allocation_correlation)
            else None
        ),
        "Return_Units": "Decimal; 0.10 = 10%",
        "Research_Type": (
            "Pension-level asset-class performance analysis"
        ),
        "Fund_Level_IRR_TVPI_DPI_RVPI": False,
    }

    return summary


# ============================================================
# 15. README
# ============================================================

def create_readme(
    summary: dict[str, Any],
) -> None:

    text = f"""
# Private Equity Performance Analysis Using US Public Pension Data

## Research Objective

Analyze reported private-equity asset-class returns and allocations
across US public pension plans and compare them with public equity.

## Data Source

Public Plans Database:

https://publicplansdata.org/public-plans-database/

API:

https://publicplansdata.org/api/

## Dataset Coverage

- Fiscal years: {summary['Coverage_Start']}–{summary['Coverage_End']}
- Unique pension plans: {summary['Unique_Plans']}
- Raw API observations: {summary['Raw_API_Observations']}
- Cleaned plan-year observations: {summary['Cleaned_Plan_Year_Observations']}
- PE return observations: {summary['PE_Return_Observations']}

## Variables

- PETotal_Rtrn: Private equity annual return
- PETotal_Actl: Actual private equity allocation
- EQTotal_Rtrn: Public equity annual return
- EQTotal_Actl: Actual public equity allocation
- RETotal_Rtrn: Real estate annual return
- HFTotal_Rtrn: Hedge fund annual return

## Methodology

1. API data extraction using Python Requests.
2. Data validation and cleaning using Pandas and NumPy.
3. Removal of empty investment records.
4. Duplicate plan-year reconciliation using non-null values.
5. Conflict detection before record aggregation.
6. Annual asset-class performance analysis.
7. Same-plan, same-year PE versus public equity comparison.
8. Plan-level descriptive statistics.
9. Rolling five-year annualized geometric returns.
10. PE allocation and return correlation analysis.

## Return Units

Returns and allocations are treated as decimal values.

For example:
0.10 = 10%

No division by 100 is applied.

## Research Summary

- Median PE annual return: {summary['Median_PE_Annual_Return']:.2%}
- Mean PE annual return: {summary['Mean_PE_Annual_Return']:.2%}
- Median public equity return: {summary['Median_Public_Equity_Annual_Return']:.2%}
- Median PE allocation: {summary['Median_PE_Allocation']:.2%}
- Paired PE/public equity correlation: {summary['PE_Public_Equity_Correlation_Paired']}
- PE allocation/return correlation: {summary['PE_Allocation_Return_Correlation']}

## Limitations

This dataset contains pension-plan asset-class returns.
It does not provide individual private-equity fund cash flows.

Therefore, this analysis does not calculate individual fund-level:

- IRR
- TVPI
- DPI
- RVPI

Private-market returns can be affected by appraisal-based valuation,
reporting lags, valuation smoothing and differences in reporting methods.

The analysis is descriptive and does not establish causal relationships.
"""

    (OUTPUT_DIR / "README.md").write_text(
        text.strip(),
        encoding="utf-8",
    )


# ============================================================
# 16. MAIN EXECUTION
# ============================================================

def main() -> None:

    print("=" * 65)
    print("PRIVATE EQUITY PERFORMANCE RESEARCH")
    print("=" * 65)

    # Delete previous outputs.
    reset_output_directory()

    # Download raw API data.
    raw_df = download_data()

    raw_df.to_csv(
        OUTPUT_DIR / "raw_api_data.csv",
        index=False,
    )

    # Clean and validate.
    df, plan_key = clean_data(raw_df)

    df.to_csv(
        OUTPUT_DIR / "raw_cleaned_panel.csv",
        index=False,
    )

    # Annual performance.
    annual = create_annual_summary(df)

    annual.to_csv(
        OUTPUT_DIR / "annual_performance_summary.csv",
        index=False,
    )

    # Paired comparison.
    paired, annual_paired = create_paired_comparison(df)

    paired.to_csv(
        OUTPUT_DIR / "paired_pe_public_equity.csv",
        index=False,
    )

    annual_paired.to_csv(
        OUTPUT_DIR / "paired_comparison.csv",
        index=False,
    )

    # Plan-level statistics.
    plan_stats = create_plan_statistics(df)

    plan_stats.to_csv(
        OUTPUT_DIR / "plan_level_statistics.csv",
        index=False,
    )

    # Rolling returns.
    rolling = calculate_rolling_returns(df)

    rolling.to_csv(
        OUTPUT_DIR / "rolling_five_year_returns.csv",
        index=False,
    )

    # Allocation analysis.
    allocation_corr, allocation_summary = (
        create_allocation_analysis(df)
    )

    allocation_summary.to_csv(
        OUTPUT_DIR / "allocation_return_analysis.csv",
        index=False,
    )

    # Correlation matrix.
    correlation_matrix = create_correlation_analysis(df)

    correlation_matrix.to_csv(
        OUTPUT_DIR / "asset_class_correlation.csv"
    )

    # Charts.
    create_charts(
        df,
        annual,
        rolling,
    )

    # Summary.
    summary = create_research_summary(
        df,
        paired,
        allocation_corr,
        len(raw_df),
    )

    with open(
        OUTPUT_DIR / "research_summary.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    # README.
    create_readme(summary)

    # --------------------------------------------------------
    # PRINT FINAL RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("ANNUAL PERFORMANCE SUMMARY")
    print("=" * 65)

    display_annual = annual.copy()

    percentage_columns = [
        "PE_Median",
        "PE_Mean",
        "Public_Equity_Median",
        "Real_Estate_Median",
        "Hedge_Fund_Median",
        "Median_PE_Allocation",
    ]

    display_annual[percentage_columns] *= 100

    print(
        display_annual.tail(10).to_string(index=False)
    )

    print("\n" + "=" * 65)
    print("PAIRED PE VS PUBLIC EQUITY")
    print("=" * 65)

    display_paired = annual_paired.copy()

    paired_percentage_columns = [
        "Median_PE_Return",
        "Median_Public_Return",
        "Median_Excess_Return",
        "Mean_Excess_Return",
    ]

    display_paired[paired_percentage_columns] *= 100

    print(
        display_paired.tail(10).to_string(index=False)
    )

    print("\n" + "=" * 65)
    print("RESEARCH SUMMARY")
    print("=" * 65)

    print(
        f"Unique pension plans: "
        f"{summary['Unique_Plans']}"
    )

    print(
        f"Raw API observations: "
        f"{summary['Raw_API_Observations']}"
    )

    print(
        f"Cleaned observations: "
        f"{summary['Cleaned_Plan_Year_Observations']}"
    )

    print(
        f"Median PE return: "
        f"{summary['Median_PE_Annual_Return']:.2%}"
    )

    print(
        f"Mean PE return: "
        f"{summary['Mean_PE_Annual_Return']:.2%}"
    )

    print(
        f"Median public equity return: "
        f"{summary['Median_Public_Equity_Annual_Return']:.2%}"
    )

    print(
        f"Median PE allocation: "
        f"{summary['Median_PE_Allocation']:.2%}"
    )

    print(
        f"Paired PE/public equity correlation: "
        f"{summary['PE_Public_Equity_Correlation_Paired']}"
    )

    print(
        f"PE allocation/return correlation: "
        f"{summary['PE_Allocation_Return_Correlation']}"
    )

    print("\n" + "=" * 65)
    print("PROJECT COMPLETED")
    print("=" * 65)

    print(f"\nOutput directory:\n{OUTPUT_DIR}")

    print("\nGenerated files:")

    for path in sorted(OUTPUT_DIR.iterdir()):
        print(path.name)


if __name__ == "__main__":
    main()