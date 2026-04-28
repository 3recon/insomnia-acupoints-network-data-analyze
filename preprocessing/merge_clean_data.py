from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

CLEAN_FILES = [
    DATA_DIR / "2012_clean.csv",
    DATA_DIR / "2016_clean.csv",
    DATA_DIR / "2020_clean.csv",
]

MAPPING_LOG_FILES = [
    DATA_DIR / "2012_mapping_log.csv",
    DATA_DIR / "2016_mapping_log.csv",
    DATA_DIR / "2020_mapping_log.csv",
]

ALL_PAPERS_OUTPUT = DATA_DIR / "all_papers_clean.csv"
ALL_MAPPING_LOG_OUTPUT = DATA_DIR / "all_mapping_log.csv"
EXCLUDED_PAPERS_OUTPUT = DATA_DIR / "all_excluded_papers.csv"


def read_csvs(paths: list[Path]) -> pd.DataFrame:
    missing = [path for path in paths if not path.exists()]
    if missing:
        missing_names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required files: {missing_names}")

    return pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)


def parse_list(value: str) -> list[str]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected list string, got: {value}")
    return parsed


def validate_outputs(mapping_log_df: pd.DataFrame) -> None:
    review_needed_count = int(mapping_log_df["action"].eq("review_needed").sum())
    if review_needed_count:
        raise ValueError(f"Found {review_needed_count} review_needed rows in mapping log")


def main() -> None:
    papers_df = read_csvs(CLEAN_FILES)
    mapping_log_df = read_csvs(MAPPING_LOG_FILES)

    validate_outputs(mapping_log_df)

    empty_standard = papers_df["standard_acupoints_list"].apply(
        lambda value: len(parse_list(value)) == 0
    )
    excluded_papers_df = papers_df.loc[empty_standard].copy()
    papers_df = papers_df.loc[~empty_standard].copy()

    papers_df.to_csv(ALL_PAPERS_OUTPUT, index=False, encoding="utf-8-sig")
    mapping_log_df.to_csv(ALL_MAPPING_LOG_OUTPUT, index=False, encoding="utf-8-sig")
    excluded_papers_df.to_csv(EXCLUDED_PAPERS_OUTPUT, index=False, encoding="utf-8-sig")

    print(f"Saved {len(papers_df)} rows to {ALL_PAPERS_OUTPUT}")
    print(f"Saved {len(mapping_log_df)} rows to {ALL_MAPPING_LOG_OUTPUT}")
    print(f"Saved {len(excluded_papers_df)} rows to {EXCLUDED_PAPERS_OUTPUT}")
    print("Mapping actions:")
    print(mapping_log_df["action"].value_counts().to_string())


if __name__ == "__main__":
    main()
