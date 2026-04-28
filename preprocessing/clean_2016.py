from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from preprocessing.clean_2012 import (
    EAR_MAPPING,
    log_item,
    normalize_acupoint as normalize_acupoint_2012,
    normalize_body_code,
    normalize_key,
    parse_acupoints_list,
    unique_preserve_order,
)


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "2016.csv"
CLEAN_OUTPUT_FILE = BASE_DIR / "data" / "2016_clean.csv"
MAPPING_LOG_FILE = BASE_DIR / "data" / "2016_mapping_log.csv"

NAMED_BODY_POINTS = {
    "sishencong": "SISHENCONG",
    "anmian": "ANMIAN",
    "yintang": "YINTANG",
    "taiyang": "TAIYANG",
}

EAR_MAPPING_2016 = {
    **EAR_MAPPING,
    "shenmen": "EAR_SHENMEN",
    "brain": "EAR_BRAIN",
    "hypothysis": "EAR_HYPOPHYSIS",
}

DROP_VALUES_2016 = {
    "bl20 or bl15",
    "pc7 or ki1",
}


def normalize_acupoint(original: str) -> list[dict[str, str]]:
    value = str(original).strip()
    key = normalize_key(value)

    if key == "lr3. yintang":
        return [
            log_item(value, "LR3", "body", "split", "마침표로 연결된 체침 경혈 분리"),
            log_item(value, "YINTANG", "body", "split", "마침표로 연결된 체침 경혈 분리"),
        ]

    if key.startswith("ear points:"):
        key = normalize_key(key.replace("ear points:", "", 1))

    if key.startswith("moxa:"):
        candidate = value.split(":", 1)[1].strip()
        body_code = normalize_body_code(candidate)
        if body_code:
            return [log_item(value, body_code, "body", "map", "치료 방식 접두사 제거")]

    if key in DROP_VALUES_2016:
        return [log_item(value, "", "unknown", "drop", "or 표현으로 특정 경혈 확정 불가")]

    if key in NAMED_BODY_POINTS:
        return [log_item(value, NAMED_BODY_POINTS[key], "body", "map", "명명된 체침/기혈 표준화")]

    if key in EAR_MAPPING_2016:
        return [log_item(value, EAR_MAPPING_2016[key], "ear", "map", "이침 경혈 표준화")]

    return normalize_acupoint_2012(value)


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
