from pathlib import Path

import pandas as pd

from src.config import DEFAULT_TRANSCRIPT_PATH


def read_table(file_or_path):
    """CSV / Excel を拡張子に応じて DataFrame として読み込む。"""
    name = getattr(file_or_path, "name", str(file_or_path))
    suffix = Path(name).suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(file_or_path)
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(file_or_path)

    raise ValueError(f"対応していないファイル形式です: {suffix}")


def read_default_transcript():
    return read_table(DEFAULT_TRANSCRIPT_PATH)
