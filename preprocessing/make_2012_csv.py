from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "불면증_경혈데이터.xlsx"
OUTPUT_FILE = BASE_DIR / "preprocessing" / "2012.csv"
SHEET_NAME = "2012_acupuncture_SR"
EXPECTED_HEADERS = ["No", "First author", "Year", "Treatment", "Acupoint"]


def find_header_row(df: pd.DataFrame) -> int:
    for idx in range(len(df)):
        row_values = [str(value).strip() for value in df.iloc[idx].tolist()]
        if all(header in row_values for header in EXPECTED_HEADERS):
            return idx
    raise ValueError(f"Could not find header row in sheet: {SHEET_NAME}")


def split_acupoints(value: object) -> list[str]:
    if pd.isna(value):
        return []

    normalized = str(value).replace("\n", " ").replace(";", ",")
    parts = [part.strip() for part in normalized.split(",")]
    return [part for part in parts if part]


def build_2012_dataframe() -> pd.DataFrame:
    raw_df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME, header=None)
    header_row = find_header_row(raw_df)

    df = raw_df.iloc[header_row + 1 :].copy()
    df.columns = [str(value).strip() for value in raw_df.iloc[header_row].tolist()]

    required_columns = EXPECTED_HEADERS
    df = df[required_columns]
    df = df.dropna(subset=["No", "Acupoint"], how="any")

    df = df.rename(
        columns={
            "No": "no",
            "First author": "first_author",
            "Year": "year",
            "Treatment": "treatment",
            "Acupoint": "acupoints_raw",
        }
    )

    df["sheet"] = SHEET_NAME
    df["no"] = pd.to_numeric(df["no"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["no", "year"])
    df["no"] = df["no"].astype(int)
    df["year"] = df["year"].astype(int)
    df["first_author"] = df["first_author"].astype(str).str.strip()
    df["treatment"] = df["treatment"].astype(str).str.replace("\n", " ", regex=False).str.strip()
    df["acupoints_raw"] = df["acupoints_raw"].astype(str).str.strip()
    df["acupoints_list"] = df["acupoints_raw"].apply(split_acupoints).apply(
        lambda items: json.dumps(items, ensure_ascii=False)
    )

    return df[
        ["sheet", "no", "first_author", "year", "treatment", "acupoints_raw", "acupoints_list"]
    ].reset_index(drop=True)


def main() -> None:
    df = build_2012_dataframe()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
