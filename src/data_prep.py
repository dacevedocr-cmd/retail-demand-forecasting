"""Shared data preparation for the Valle Central demand-forecasting project.

Every notebook imports from here, so cleaning, feature engineering and the
temporal split are defined once and can't drift between stages.
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
FULL_DATA = REPO_ROOT / "data" / "raw" / "valle_central_ventas_semanales.csv"
SAMPLE_DATA = REPO_ROOT / "data" / "sample" / "valle_central_sample.csv"

TARGET = "unidades"
GROUP_KEYS = ["id_sucursal", "id_producto"]

# Columns that are only known after (or are decided by) the planner -> leakage.
LEAKAGE_COLUMNS = ["stock_inicial", "stock_final"]

CATEGORICAL = ["categoria", "tipo_sucursal"]
BOOLEAN = [
    "promocion", "promocion_sin_registro", "es_feriado_temporada",
    "es_quincena", "es_diciembre", "perecedero",
]
# All engineered numeric candidates; the final subset is chosen in notebook 02.
NUMERIC_CANDIDATES = [
    "precio", "lag_1", "lag_2", "lag_4",
    "media_movil_4", "media_movil_8", "std_movil_4", "semana_transcurrida",
]

TEST_WEEKS = 13  # last quarter of 2025
VAL_WEEKS = 13   # the quarter before it


def resolve_data_path():
    """Use the full dataset when present, otherwise the published sample."""
    if FULL_DATA.exists():
        return FULL_DATA, "full"
    return SAMPLE_DATA, "sample"


def load_raw(path=None):
    if path is None:
        path, _ = resolve_data_path()
    return pd.read_csv(path, parse_dates=["fecha"], low_memory=False)


def clean(df):
    """Apply the data-audit decisions from stage 1."""
    df = df.copy()
    df = df.drop(columns=[c for c in LEAKAGE_COLUMNS if c in df.columns])
    df = df.drop_duplicates()

    # 'promocion' was not recorded at all in 2023. Keep an explicit
    # "no record" flag instead of silently pretending there was no promotion.
    df["promocion_sin_registro"] = df["promocion"].isna()
    df["promocion"] = (
        df["promocion"].map({True: True, False: False, "True": True, "False": False})
        .fillna(False).astype(bool)
    )

    # Price: carry the last known price of the same product-store forward;
    # only the first weeks of a series (no earlier price) are back-filled.
    df = df.sort_values(GROUP_KEYS + ["fecha"])
    df["precio"] = df.groupby(GROUP_KEYS)["precio"].transform(lambda s: s.ffill().bfill())
    return df.reset_index(drop=True)


def add_features(df):
    """Lags and rolling statistics per product-store, using past weeks only."""
    df = df.sort_values(GROUP_KEYS + ["fecha"]).reset_index(drop=True)
    grp = df.groupby(GROUP_KEYS)[TARGET]

    df["lag_1"] = grp.shift(1)
    df["lag_2"] = grp.shift(2)
    df["lag_4"] = grp.shift(4)
    # shift(1) BEFORE rolling so the current week never enters its own feature
    df["media_movil_4"] = grp.transform(lambda s: s.shift(1).rolling(4).mean())
    df["media_movil_8"] = grp.transform(lambda s: s.shift(1).rolling(8).mean())
    df["std_movil_4"] = grp.transform(lambda s: s.shift(1).rolling(4).std())
    df["semana_transcurrida"] = df.groupby(GROUP_KEYS).cumcount()

    # The first 8 weeks of each series lack enough history for every feature.
    return df.dropna(subset=["lag_4", "media_movil_8", "std_movil_4"]).reset_index(drop=True)


def temporal_split(df):
    """Split by calendar position: train | validation (13 wks) | test (13 wks)."""
    weeks = np.sort(df["fecha"].unique())
    val_weeks = weeks[-(TEST_WEEKS + VAL_WEEKS):-TEST_WEEKS]
    test_weeks = weeks[-TEST_WEEKS:]
    part = np.where(df["fecha"].isin(test_weeks), "test",
                    np.where(df["fecha"].isin(val_weeks), "val", "train"))
    df = df.assign(particion=part)
    return (df[df.particion == "train"].copy(),
            df[df.particion == "val"].copy(),
            df[df.particion == "test"].copy())


def prepare(path=None):
    """Load -> clean -> features -> split. Returns (train, val, test)."""
    return temporal_split(add_features(clean(load_raw(path))))


# ---------- metrics ----------
def wape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.abs(y_true - y_pred).sum() / y_true.sum()


def mape_nonzero(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
