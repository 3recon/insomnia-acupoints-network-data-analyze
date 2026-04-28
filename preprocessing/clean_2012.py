from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "2012.csv"
CLEAN_OUTPUT_FILE = BASE_DIR / "data" / "2012_clean.csv"
MAPPING_LOG_FILE = BASE_DIR / "data" / "2012_mapping_log.csv"

BODY_CODE_RE = re.compile(r"^(?P<prefix>[A-Z]{1,3}(?:-[A-Z]{2})?)(?P<number>\d+)$")

EAR_MAPPING = {
    "(ear) shenmen": "EAR_SHENMEN",
    "ear shenmen": "EAR_SHENMEN",
    "shenmen(ear)": "EAR_SHENMEN",
    "shenmen(tf4)": "EAR_SHENMEN",
    "(ear) zhenjing": "EAR_ZHENJING",
    "heart": "EAR_HEART",
    "kidney": "EAR_KIDNEY",
    "liver": "EAR_LIVER",
    "spleen": "EAR_SPLEEN",
    "lung": "EAR_LUNG",
    "occiput": "EAR_OCCIPUT",
    "subcortex": "EAR_SUBCORTEX",
    "gallbladder": "EAR_GALLBLADDER",
    "stomach": "EAR_STOMACH",
    "endocrine": "EAR_ENDOCRINE",
    "sympathetic": "EAR_SYMPATHETIC",
    "cortical areas": "EAR_CORTICAL_AREAS",
    "small intestine": "EAR_SMALL_INTESTINE",
    "large intestine": "EAR_LARGE_INTESTINE",
    "sanjiao": "EAR_SANJIAO",
}

DROP_VALUES = {
    "hands(unspecified)",
    "multiple acupoints(경혈 명시x)",
    "경혈 명시x",
    "1 inch above medial malleolus",
    "bailing",
    "족태양방광경(bl)",
    "독맥(gv)의 모든 경혈: bl1",
}


def normalize_key(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).lower()


def normalize_body_code(value: str) -> str | None:
    value = value.strip()
    value = re.sub(r"\((bilateral|bilaterally|unilateral|unilaterally)\)", "", value, flags=re.I)
    value = re.sub(r"\(논문에는 h7이라고 되어있음\)", "", value, flags=re.I)
    value = value.strip().upper()

    match = BODY_CODE_RE.match(value)
    if not match:
        return None

    prefix = match.group("prefix")
    number = match.group("number")
    if prefix == "DU":
        prefix = "GV"
    elif prefix == "RN":
        prefix = "CV"

    return f"{prefix}{number}"


def log_item(
    original: str,
    standard: str,
    system: str,
    action: str,
    reason: str,
) -> dict[str, str]:
    return {
        "original_acupoint": original,
        "standard_acupoint": standard,
        "system": system,
        "action": action,
        "reason": reason,
    }


def normalize_acupoint(original: str) -> list[dict[str, str]]:
    value = str(original).strip()
    key = normalize_key(value)

    if key == "occiput and subcortex":
        return [
            log_item(value, "EAR_OCCIPUT", "ear", "split", "합쳐진 이침 경혈 분리"),
            log_item(value, "EAR_SUBCORTEX", "ear", "split", "합쳐진 이침 경혈 분리"),
        ]

    if key in DROP_VALUES:
        return [log_item(value, "", "unknown", "drop", "특정 경혈로 식별 불가")]

    if key in EAR_MAPPING:
        return [log_item(value, EAR_MAPPING[key], "ear", "map", "이침 경혈 표준화")]

    body_code = normalize_body_code(value)
    if body_code:
        action = "keep" if body_code == value else "map"
        reason = "표준 체침 경혈 코드" if action == "keep" else "체침 경혈 코드 표준화"
        return [log_item(value, body_code, "body", action, reason)]

    return [log_item(value, "", "unknown", "review_needed", "규칙으로 분류되지 않아 검토 필요")]


def parse_acupoints_list(value: str) -> list[str]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list):
        raise ValueError("acupoints_list must contain a list representation")
    return [str(item) for item in parsed]


def unique_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    unique_items = []
    for item in items:
        if item and item not in seen:
            unique_items.append(item)
            seen.add(item)
    return unique_items


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
