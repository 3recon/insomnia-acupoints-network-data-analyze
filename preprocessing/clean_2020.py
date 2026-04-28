from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from preprocessing.clean_2012 import (
    log_item,
    normalize_body_code,
    normalize_key,
    parse_acupoints_list,
    unique_preserve_order,
)
from preprocessing.clean_2016 import EAR_MAPPING_2016, normalize_acupoint as normalize_acupoint_2016


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "2020.csv"
CLEAN_OUTPUT_FILE = BASE_DIR / "data" / "2020_clean.csv"
MAPPING_LOG_FILE = BASE_DIR / "data" / "2020_mapping_log.csv"

EAR_MAPPING_2020 = {
    **EAR_MAPPING_2016,
    "sympathetic autonomic": "EAR_SYMPATHETIC",
}


def remove_parenthetical_notes(value: str) -> str:
    return re.sub(r"\([^)]*\)", "", value).strip()


def normalize_acupoint(original: str) -> list[dict[str, str]]:
    value = str(original).strip()
    key = normalize_key(value)

    if key.startswith("ex-hn1("):
        return [log_item(value, "EX-HN1", "body", "map", "논문 오타 주석 제거")]

    body_code = normalize_body_code(value)
    if body_code:
        action = "keep" if body_code == value else "map"
        reason = "표준 체침 경혈 코드" if action == "keep" else "체침 경혈 코드 표준화"
        return [log_item(value, body_code, "body", action, reason)]

    ear_candidate = remove_parenthetical_notes(value)
    ear_key = normalize_key(ear_candidate)
    if ear_key in EAR_MAPPING_2020:
        return [log_item(value, EAR_MAPPING_2020[ear_key], "ear", "map", "이침 경혈 표준화")]

    return normalize_acupoint_2016(value)


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean_rows = []
    log_rows = []

    for _, row in df.iterrows():
        standard_items: list[str] = []
        dropped_items: list[str] = []

        for original in parse_acupoints_list(row["acupoints_list"]):
            normalized_items = normalize_acupoint(original)
            for item in normalized_items:
                log_rows.append(
                    {
                        "paper_no": row["no"],
                        "first_author": row["first_author"],
                        "year": row["year"],
                        **item,
                    }
                )

                if item["action"] in {"keep", "map", "split"}:
                    standard_items.append(item["standard_acupoint"])
                else:
                    dropped_items.append(item["original_acupoint"])

        clean_row = row.to_dict()
        clean_row["standard_acupoints_list"] = json.dumps(
            unique_preserve_order(standard_items),
            ensure_ascii=False,
        )
        clean_row["dropped_acupoints_list"] = json.dumps(
            unique_preserve_order(dropped_items),
            ensure_ascii=False,
        )
        clean_rows.append(clean_row)

    return pd.DataFrame(clean_rows), pd.DataFrame(log_rows)


def main() -> None:
    df = pd.read_csv(INPUT_FILE)
    clean_df, mapping_log_df = clean_dataframe(df)

    clean_df.to_csv(CLEAN_OUTPUT_FILE, index=False, encoding="utf-8-sig")
    mapping_log_df.to_csv(MAPPING_LOG_FILE, index=False, encoding="utf-8-sig")

    print(f"Saved {len(clean_df)} rows to {CLEAN_OUTPUT_FILE}")
    print(f"Saved {len(mapping_log_df)} rows to {MAPPING_LOG_FILE}")


if __name__ == "__main__":
    main()
