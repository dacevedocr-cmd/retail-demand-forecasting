"""Generates the project notebooks from source (kept for reproducibility)."""
import nbformat as nbf
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks"

SETUP = """import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src import data_prep as dp, plot_style as ps

ps.apply()
pd.options.display.float_format = "{:,.2f}".format
IMG = Path.cwd().parent / "images"
DATA_PATH, DATA_MODE = dp.resolve_data_path()
print(f"Using {DATA_MODE} dataset: {DATA_PATH.name}")"""


def nb(cells):
    book = nbf.v4.new_notebook()
    book.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    book.cells = [nbf.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbf.v4.new_code_cell(c[1]) for c in cells]
    return book


# ------------------------------------------------------------------ 01
nb01 = [
("md", """# 01 · Data audit

**Question:** is the weekly sales history of Distribuidora Valle Central fit for training a demand forecast, and what has to be fixed or excluded first?

Unit of observation: one product × one store × one week. Target: `unidades` (units sold)."""),
("code", SETUP),
("code", """raw = dp.load_raw(DATA_PATH)
print(f"Rows: {len(raw):,} · Columns: {raw.shape[1]}")
print(f"Period: {raw.fecha.min().date()} → {raw.fecha.max().date()} ({raw.fecha.nunique()} weeks)")
print(f"Stores: {raw.id_sucursal.nunique()} · Products: {raw.id_producto.nunique()} · Categories: {raw.categoria.nunique()}")
raw.head()"""),
("md", "## 1. Missing values and duplicates"),
("code", """missing = (raw.isna().mean() * 100).round(2)
display(missing[missing > 0].rename("% missing").to_frame())
print("Exact duplicate rows:", raw.duplicated().sum())"""),
("code", """# Is the missingness in 'promocion' random, or structured in time?
(raw.assign(year=raw.fecha.dt.year)
    .groupby("year")["promocion"].apply(lambda s: s.isna().mean() * 100)
    .rename("% promocion missing").to_frame())"""),
("md", """**Finding:** `promocion` is missing for *all* of 2023 — it simply wasn't recorded that year. Filling it with `False` alone would claim there were no promotions for a full year, so the cleaning step also adds a `promocion_sin_registro` flag that lets the model tell "no promotion" apart from "unknown". `precio` (~2 % missing) is filled with the last known price of the same product in the same store."""),
("md", """## 2. Data leakage

`stock_inicial` and `stock_final` look like great predictors. The question is whether they would be *available* on Monday, when the planner makes the forecast."""),
("code", """print(raw[["stock_inicial", "stock_final", "unidades"]].corr()["unidades"].round(3))

identity = (raw.stock_final == raw.stock_inicial - raw.unidades).mean() * 100
print(f"\\nstock_final == stock_inicial − unidades in {identity:.2f}% of rows")"""),
("md", """- `stock_final` is literally `stock_inicial − unidades`: it's known only after the week's sales. **Deterministic leakage → excluded.**
- `stock_inicial` correlates 0.98 with the target because it *is* the planner's own replenishment decision — the very decision this model is meant to support. Using it would teach the model to copy the planner. **Circular leakage → excluded.**"""),
("md", "## 3. Censored demand"),
("code", """zeros = (raw.unidades == 0).sum()
zeros_no_stock = ((raw.unidades == 0) & (raw.stock_inicial == 0)).sum()
sold_out = ((raw.unidades == raw.stock_inicial) & (raw.stock_inicial > 0)).sum()
print(f"Weeks with 0 units sold: {zeros:,} — of which with no stock at all: {zeros_no_stock:,}")
print(f"Weeks that sold out (units == opening stock): {sold_out:,} ({sold_out / len(raw) * 100:.1f}% of rows)")"""),
("md", """Every zero-sales week had zero stock on hand, and another 468 weeks sold out completely (≈0.8 % of rows combined). In those rows we observe *sales*, not *demand* — true demand was at least that high. The share is small, so it's documented as a known limitation (it slightly biases forecasts downward) rather than modeled."""),
("md", "## 4. Coverage by store"),
("code", """coverage = raw.groupby(["id_sucursal", "nombre_sucursal", "tipo_sucursal"]).fecha.agg(first_week="min", weeks="nunique")
coverage"""),
("md", """Two stores opened recently (S11 in January 2025, S12 in April 2025), and **both are rural**. They have little history, so any urban-vs-rural error gap could really be a new-vs-established gap. Final metrics are reported for both splits to tell the two apart."""),
("md", "## 5. Target distribution and seasonality"),
("code", """print(raw.unidades.describe().round(1))
print(f"Skewness: {raw.unidades.skew():.2f}")"""),
("code", """weekly = raw.groupby("fecha").unidades.sum() / 1000

fig, ax = plt.subplots(figsize=(10, 3.8))
ax.plot(weekly.index, weekly.values, color=ps.BLUE)
for year in (2023, 2024, 2025):
    ax.axvspan(pd.Timestamp(f"{year}-12-01"), pd.Timestamp(f"{year}-12-31"), color=ps.ORANGE, alpha=0.10, lw=0)
ax.set_title("Total weekly units sold, all stores (December shaded)")
ax.set_ylabel("Units (thousands)")
ax.set_xlabel("")
fig.savefig(IMG / "weekly_sales.png")
plt.show()"""),
("code", """# Average units per product-store-week under each calendar / promotion flag
flags = ["es_diciembre", "es_quincena", "es_feriado_temporada", "promocion"]
rows = []
for f in flags:
    g = raw[raw[f].notna()].groupby(raw[f].astype(str)).unidades.mean()
    rows.append({"flag": f, "mean when True": g.get("True"), "mean when False": g.get("False"),
                 "lift %": (g.get("True") / g.get("False") - 1) * 100})
pd.DataFrame(rows)"""),
("md", """## Audit summary

| Issue | Decision |
|---|---|
| `stock_final`, `stock_inicial` | Excluded — deterministic and circular leakage |
| `promocion` missing in 2023 | Fill with `False` **plus** a `promocion_sin_registro` flag |
| `precio` ~2 % missing | Last known price of the same product-store |
| 25 exact duplicates | Dropped |
| Sold-out / zero-stock weeks | Kept; documented as censored demand (limitation) |
| New stores S11, S12 | Metrics reported separately for new vs. established stores |
| Right-skewed target, December peaks | MAE as main metric (robust to spikes) + WAPE for comparability across categories |"""),
]

# ------------------------------------------------------------------ 02
nb02 = [
("md", """# 02 · Feature engineering and model selection

Goal set in stage 1: **reduce MAE by at least 20 % versus the planner's current rule** (the 4-week moving average), measured on data the model never saw.

This notebook builds the features, chooses the feature set and the model family using the **validation** quarter only. The test quarter is not touched here."""),
("code", SETUP + """

import json
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
RANDOM_STATE = 649"""),
("code", """train, val, test = dp.prepare(DATA_PATH)
for name, part in [("train", train), ("validation", val), ("test", test)]:
    print(f"{name:<11} {part.fecha.min().date()} → {part.fecha.max().date()}  rows: {len(part):>7,}")"""),
("md", """## 1. Features

All time-based features are computed per product-store series and use **only past weeks** (`shift(1)` before any rolling window), so the week being forecast never leaks into its own inputs.

| Feature | Meaning |
|---|---|
| `lag_1`, `lag_2`, `lag_4` | Units sold 1, 2 and 4 weeks earlier |
| `media_movil_4`, `media_movil_8` | Mean of the previous 4 / 8 weeks |
| `std_movil_4` | Volatility of the previous 4 weeks |
| `semana_transcurrida` | Trend: weeks since the series started |
| calendar flags | December, payday (quincena), holiday season |
| product / store | category, perishable, price, promotion, urban/rural |"""),
("code", """# Sanity check: lag_1 must equal the previous week's units of the same series
chk = pd.concat([train, val]).sort_values(dp.GROUP_KEYS + ["fecha"])
prev = chk.groupby(dp.GROUP_KEYS).unidades.shift(1)
same = prev.notna()
assert np.allclose(chk.loc[same, "lag_1"], prev[same]), "lag_1 is misaligned"
print("lag_1 verified against the previous week ✔")"""),
("md", "## 2. Redundancy between numeric features (training data only)"),
("code", """corr = train[dp.NUMERIC_CANDIDATES].corr().abs()
pairs = (corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
             .sort_values(ascending=False).rename("abs_corr").reset_index())
pairs.columns = ["feature_1", "feature_2", "abs_corr"]
display(pairs.head(6))
print("Pairs above 0.85:", (pairs.abs_corr > 0.85).sum())"""),
("md", """The two moving averages are the most similar pair, and they carry largely the same information (recent level of demand). Rather than drop one on correlation alone, the decision is made **empirically**: train the same model with each and with both, and compare on validation."""),
("code", """BASE_NUMERIC = ["precio", "lag_1", "lag_2", "lag_4", "std_movil_4", "semana_transcurrida"]
FEATURE_SETS = {
    "4-week MA only": BASE_NUMERIC + ["media_movil_4"],
    "8-week MA only": BASE_NUMERIC + ["media_movil_8"],
    "both MAs":       BASE_NUMERIC + ["media_movil_4", "media_movil_8"],
}

def make_pipeline(numeric, model):
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), dp.CATEGORICAL),
        ("num", StandardScaler(), numeric),
    ], remainder="passthrough")          # boolean flags pass through as 0/1
    return Pipeline([("prep", prep), ("model", model)])

def cols(numeric):
    return dp.CATEGORICAL + numeric + dp.BOOLEAN

ablation = []
for name, numeric in FEATURE_SETS.items():
    pipe = make_pipeline(numeric, GradientBoostingRegressor(random_state=RANDOM_STATE))
    pipe.fit(train[cols(numeric)], train.unidades)
    pred = pipe.predict(val[cols(numeric)])
    ablation.append({"feature set": name, "n features": len(cols(numeric)),
                     "val MAE": mean_absolute_error(val.unidades, pred)})
ablation = pd.DataFrame(ablation)
ablation"""),
("code", """# Rule: lowest validation MAE; if within 0.05 units of the best, prefer fewer features.
best_mae = ablation["val MAE"].min()
chosen = (ablation[ablation["val MAE"] <= best_mae + 0.05]
          .sort_values(["n features", "val MAE"]).iloc[0]["feature set"])
NUMERIC = FEATURE_SETS[chosen]
FEATURES = cols(NUMERIC)
print("Chosen feature set:", chosen)
print("Numeric features:", NUMERIC)"""),
("md", """**Result:** the 8-week average alone is clearly better than the 4-week one (≈1.3 units lower MAE), and adding both gives nothing extra. The 8-week window smooths out one-off spikes (a payday week, a single promotion) that the 4-week window over-reacts to. The 4-week average stays in the project only as the **baseline**, the rule the planner uses today."""),
("md", "## 3. Baseline vs. linear regression vs. gradient boosting (validation)"),
("code", """def evaluate(y, pred, name):
    return {"model": name,
            "MAE": mean_absolute_error(y, pred),
            "RMSE": mean_squared_error(y, pred) ** 0.5,
            "WAPE %": dp.wape(y, pred) * 100}

lr = make_pipeline(NUMERIC, LinearRegression()).fit(train[FEATURES], train.unidades)
gb = make_pipeline(NUMERIC, GradientBoostingRegressor(random_state=RANDOM_STATE)).fit(train[FEATURES], train.unidades)

comparison = pd.DataFrame([
    evaluate(val.unidades, val.media_movil_4, "Baseline: 4-week moving average"),
    evaluate(val.unidades, lr.predict(val[FEATURES]), "Linear regression"),
    evaluate(val.unidades, gb.predict(val[FEATURES]), "Gradient boosting (default)"),
])
base_mae = comparison.loc[0, "MAE"]
comparison["MAE vs baseline %"] = (base_mae - comparison.MAE) / base_mae * 100
comparison"""),
("md", """**Model family decision.** Linear regression is cheaper and easier to explain, but gradient boosting captures non-linear interactions (promotion × category, December × perishables) and is the model that clears the ≥20 % target on validation. Gradient boosting moves forward to tuning; linear regression stays documented as the interpretable reference."""),
("md", "## 4. What the model relies on"),
("code", """names = (list(gb.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(dp.CATEGORICAL))
         + NUMERIC + dp.BOOLEAN)
importance = pd.Series(gb.named_steps["model"].feature_importances_, index=names).sort_values()
top = importance.tail(10)

fig, ax = plt.subplots(figsize=(8, 4.2))
ax.barh(top.index, top.values, color=ps.BLUE, height=0.6)
ax.set_title("Top 10 features by importance (gradient boosting)")
ax.set_xlabel("Relative importance")
ax.grid(axis="y", visible=False)
for i, v in enumerate(top.values):
    ax.text(v + top.max() * 0.01, i, f"{v:.1%}", va="center", color=ps.INK_2, fontsize=9)
fig.savefig(IMG / "feature_importance.png")
plt.show()"""),
("md", """## 5. Fairness check: urban vs. rural stores (validation)

Defined *before* looking at results in stage 1: compare WAPE between urban and rural stores; the gap must stay **≤ 10 percentage points**, so rural stores aren't systematically under-supplied."""),
("code", """val_eval = val.assign(pred=gb.predict(val[FEATURES]))
fair = val_eval.groupby("tipo_sucursal").apply(lambda g: dp.wape(g.unidades, g.pred) * 100, include_groups=False)
display(fair.rename("WAPE %").to_frame())
print(f"Gap: {fair.max() - fair.min():.1f} pp (threshold 10 pp)")"""),
("code", """# Hand the decisions to notebook 03
config = {"numeric": NUMERIC, "categorical": dp.CATEGORICAL, "boolean": dp.BOOLEAN,
          "feature_set": chosen, "random_state": RANDOM_STATE}
(Path.cwd().parent / "models").mkdir(exist_ok=True)
with open(Path.cwd().parent / "models" / "feature_config.json", "w") as f:
    json.dump(config, f, indent=2)
print(json.dumps(config, indent=2))"""),
]

# ------------------------------------------------------------------ 03
nb03 = [
("md", """# 03 · Tuning, final evaluation and results

1. Tune gradient boosting with **time-series cross-validation** on train + validation.
2. Check the finalists for overfitting.
3. Evaluate the chosen model **once** on the test quarter (Oct–Dec 2025), which no earlier step has seen.
4. Break the error down by store type, store age and category, and translate it into recommendations."""),
("code", SETUP + """

import json, joblib
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

MODELS = Path.cwd().parent / "models"
config = json.loads((MODELS / "feature_config.json").read_text())
NUMERIC, RANDOM_STATE = config["numeric"], config["random_state"]
FEATURES = dp.CATEGORICAL + NUMERIC + dp.BOOLEAN
print("Features:", FEATURES)"""),
("code", """train, val, test = dp.prepare(DATA_PATH)
train_val = pd.concat([train, val]).sort_values("fecha")   # chronological order for TimeSeriesSplit

prep = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), dp.CATEGORICAL),
    ("num", StandardScaler(), NUMERIC),
], remainder="passthrough")
pipeline = Pipeline([("prep", prep), ("model", GradientBoostingRegressor(random_state=RANDOM_STATE))])

# Grid search = 36+ fits. To keep it tractable on one CPU, tuning uses a systematic
# 1-in-4 sample that preserves chronological order. The SAME sample is used for every
# search below so results are comparable. The final model is refit on 100% of the data.
search_sample = train_val.iloc[::4]
X_s, y_s = search_sample[FEATURES], search_sample.unidades
cv = TimeSeriesSplit(n_splits=3)
print(f"Tuning sample: {len(search_sample):,} of {len(train_val):,} rows")"""),
("md", "## 1. Core hyperparameters"),
("code", """grid = {"model__n_estimators": [100, 200],
        "model__max_depth": [2, 3, 4],
        "model__learning_rate": [0.05, 0.1]}
search = GridSearchCV(pipeline, grid, cv=cv, scoring="neg_mean_absolute_error").fit(X_s, y_s)

experiments = (pd.DataFrame(search.cv_results_)
               .assign(cv_MAE=lambda d: -d.mean_test_score, cv_std=lambda d: d.std_test_score)
               [["param_model__n_estimators", "param_model__max_depth", "param_model__learning_rate", "cv_MAE", "cv_std"]]
               .rename(columns=lambda c: c.replace("param_model__", ""))
               .sort_values("cv_MAE").reset_index(drop=True))
experiments"""),
("md", """All 12 configurations land within ~1 unit of each other, and the fold-to-fold standard deviation (~2 units) is larger than the gaps between them. **Tuning is not where the remaining error lives**: it's in what the features can't see (sold-out weeks, promotions not yet recorded, unusual December weeks)."""),
("md", "## 2. Regularization hyperparameters (same sample, same folds)"),
("code", """best_core = search.best_params_
grid_reg = {**{k: [v] for k, v in best_core.items()},
            "model__min_samples_leaf": [1, 20, 50],
            "model__subsample": [0.8, 1.0]}
search_reg = GridSearchCV(pipeline, grid_reg, cv=cv, scoring="neg_mean_absolute_error").fit(X_s, y_s)

reg_table = (pd.DataFrame(search_reg.cv_results_)
             .assign(cv_MAE=lambda d: -d.mean_test_score)
             [["param_model__min_samples_leaf", "param_model__subsample", "cv_MAE"]]
             .rename(columns=lambda c: c.replace("param_model__", ""))
             .sort_values("cv_MAE").reset_index(drop=True))
display(reg_table)
gain = -search.best_score_ - (-search_reg.best_score_)
print(f"Best core-only CV MAE: {-search.best_score_:.3f} · with regularization: {-search_reg.best_score_:.3f} · gain: {gain:.3f} units")"""),
("code", """# Keep the extra hyperparameters only if they buy a meaningful improvement (> 0.1 units)
best_params = search_reg.best_params_ if gain > 0.1 else best_core
print("Candidate configuration:", best_params)"""),
("md", """`min_samples_leaf` and `subsample` change the CV error by a few hundredths of a unit, well below the 0.1-unit threshold. They are left at their defaults to keep the model simpler."""),
("md", """## 3. Overfitting check on the finalists

The top configurations are usually within a fraction of a unit of each other. As a tie-breaker, each finalist is trained on the training period only and compared on train vs. validation MAE. A validation error much *higher* than training error signals overfitting."""),
("code", """finalists = [dict(p) for p in experiments.head(3)[["n_estimators", "max_depth", "learning_rate"]].to_dict("records")]
if best_params not in [{f"model__{k}": v for k, v in f.items()} for f in finalists]:
    finalists.append({k.replace("model__", ""): v for k, v in best_params.items()})

rows = []
for f in finalists:
    params = {f"model__{k}": v for k, v in f.items()}
    m = clone(pipeline).set_params(**params).fit(train[FEATURES], train.unidades)
    mae_tr = mean_absolute_error(train.unidades, m.predict(train[FEATURES]))
    mae_va = mean_absolute_error(val.unidades, m.predict(val[FEATURES]))
    rows.append({**f, "train MAE": mae_tr, "val MAE": mae_va, "val − train": mae_va - mae_tr})
finalist_table = pd.DataFrame(rows)
finalist_table"""),
("md", """Validation MAE comes out *lower* than training MAE for every finalist. That isn't a bug: the validation quarter (Jul–Sep) has no December peaks, while the training period includes two Decembers, the hardest weeks to forecast. What matters is that no finalist shows the opposite pattern (validation clearly worse than training), so none is overfitting; the choice goes to the best validation MAE."""),
("code", """winner = finalist_table.sort_values("val MAE").iloc[0]
final_params = {f"model__{k}": winner[k] for k in finalists[0].keys()}
final_params = {k: (int(v) if float(v).is_integer() and k != "model__learning_rate" and k != "model__subsample" else float(v))
                for k, v in final_params.items()}
print("Final configuration:", final_params)

final_model = clone(pipeline).set_params(**final_params)
final_model.fit(train_val[FEATURES], train_val.unidades)    # 100% of train + validation
print("Refit on", f"{len(train_val):,}", "rows.")"""),
("md", """## 4. Final evaluation — test quarter, run once

Everything above was decided without looking at these 13 weeks."""),
("code", """assert not set(dp.LEAKAGE_COLUMNS) & set(FEATURES), "leakage column in features"
assert train.fecha.max() < val.fecha.min() and val.fecha.max() < test.fecha.min(), "partitions overlap in time"

test = test.assign(forecast=final_model.predict(test[FEATURES]))
y, yhat, base = test.unidades, test.forecast, test.media_movil_4

results = pd.DataFrame({
    "Model":    [mean_absolute_error(y, yhat), mean_squared_error(y, yhat) ** 0.5, dp.wape(y, yhat) * 100, dp.mape_nonzero(y, yhat) * 100],
    "Baseline": [mean_absolute_error(y, base), mean_squared_error(y, base) ** 0.5, dp.wape(y, base) * 100, dp.mape_nonzero(y, base) * 100],
}, index=["MAE (units)", "RMSE (units)", "WAPE %", "MAPE % (non-zero weeks)"])
results["Improvement %"] = (results.Baseline - results.Model) / results.Baseline * 100
display(results)

improvement = results.loc["MAE (units)", "Improvement %"]
print(f"MAE improvement vs planner's rule: {improvement:.1f}% → target ≥20%: {'MET' if improvement >= 20 else 'NOT MET'}")"""),
("code", """weekly = test.groupby("fecha")[["unidades", "forecast", "media_movil_4"]].sum() / 1000

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(weekly.index, weekly.unidades, color=ps.BLUE, label="Actual")
ax.plot(weekly.index, weekly.forecast, color=ps.ORANGE, ls="--", label="Model forecast")
ax.plot(weekly.index, weekly.media_movil_4, color=ps.AQUA, ls=":", label="Baseline (4-week MA)")
ax.set_title("Test quarter: total weekly units, actual vs. forecast")
ax.set_ylabel("Units (thousands)")
ax.legend(loc="upper left", ncol=3)
fig.savefig(IMG / "test_forecast.png")
plt.show()"""),
("md", "## 5. Where the error lands"),
("code", """def segment_table(col):
    return (test.groupby(col)
                .apply(lambda g: pd.Series({
                    "rows": len(g),
                    "model WAPE %": dp.wape(g.unidades, g.forecast) * 100,
                    "baseline WAPE %": dp.wape(g.unidades, g.media_movil_4) * 100,
                    "model MAE": mean_absolute_error(g.unidades, g.forecast)}), include_groups=False))

fair = segment_table("tipo_sucursal")
display(fair)
print(f"Urban vs rural WAPE gap: {fair['model WAPE %'].max() - fair['model WAPE %'].min():.1f} pp (threshold 10 pp)")"""),
("code", """test["store_age"] = np.where(test.id_sucursal.isin(["S11", "S12"]), "new (opened 2025)", "established")
segment_table("store_age")"""),
("code", """by_cat = segment_table("categoria").sort_values("model WAPE %")
display(by_cat)

fig, ax = plt.subplots(figsize=(8.5, 4.6))
ypos = np.arange(len(by_cat))
ax.barh(ypos + 0.19, by_cat["baseline WAPE %"], height=0.36, color=ps.AQUA, label="Baseline (4-week MA)")
ax.barh(ypos - 0.19, by_cat["model WAPE %"], height=0.36, color=ps.BLUE, label="Model")
ax.set_yticks(ypos, by_cat.index)
ax.set_title("Forecast error by category, test quarter (lower is better)")
ax.set_xlabel("WAPE %")
ax.grid(axis="y", visible=False)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
ax.set_title("Forecast error by category, test quarter (lower is better)", pad=28)
fig.savefig(IMG / "category_wape.png")
plt.show()"""),
("md", """Relative error (WAPE) is almost flat across categories, between roughly 26 % and 28 %. Categories like Panadería or Abarrotes show a higher error *in units* only because they sell more units. In relative terms the model is equally reliable across the catalogue, and it beats the baseline by about 9–10 points in every category."""),
("md", "### December: the weeks that matter most"),
("code", """test["period"] = np.where(test.es_diciembre, "December", "Oct–Nov")
dec = test.groupby("period").apply(lambda g: pd.Series({
    "model MAE": mean_absolute_error(g.unidades, g.forecast),
    "baseline MAE": mean_absolute_error(g.unidades, g.media_movil_4),
    "model bias %": (g.forecast.sum() / g.unidades.sum() - 1) * 100,
    "baseline bias %": (g.media_movil_4.sum() / g.unidades.sum() - 1) * 100}), include_groups=False)
dec"""),
("md", """The planner's 4-week average lags badly in December: it looks back at November and under-forecasts the peak by roughly a third. The model follows the peak much better (see the chart above), but still **under-forecasts December by about 9 %**. For perishables that's the safe side of the error (less waste); for non-perishables it risks stock-outs in the highest-revenue weeks."""),
("md", """## 6. Conclusions and recommendations

**Result.** On the held-out quarter (Oct–Dec 2025) the model cuts MAE from 28.7 to 21.3 units per product-store-week, a **26 % improvement** over the planner's current rule. That clears the ≥20 % acceptance criterion set in stage 1, and the model beats the baseline on every metric, in every category and in both store types.

**Fairness.** Urban/rural WAPE gap: 1.4 pp (threshold 10 pp). The two new rural stores are only ~1.5 pp worse than established ones despite having under a year of history.

**Recommendations for the planner**
1. Use the forecast as the **starting point** for Monday's order, not as an automatic order. The planner keeps the final decision (a stage-1 ethics requirement).
2. In December, add a manual safety margin on non-perishables, since the model runs about 9 % short.
3. Start recording promotions consistently and log stock-outs. Both are the main blind spots in the current data, and tuning more hyperparameters won't close that gap.
4. Retrain monthly and track MAE against the baseline. If the improvement drops below 20 %, investigate before continuing to use the model.

**Limitations.** Sold-out weeks record sales, not true demand (≈0.8 % of rows). The dataset is a course case study, not a live company feed."""),
("code", """joblib.dump(final_model, MODELS / "demand_forecast_gb.joblib")
print("Model saved to models/demand_forecast_gb.joblib")"""),
]

import sys
ONLY = sys.argv[1:]
OUT.mkdir(exist_ok=True)
for name, cells in [("01_data_audit", nb01), ("02_features_and_model_selection", nb02), ("03_tuning_and_final_evaluation", nb03)]:
    if ONLY and name not in ONLY:
        continue
    nbf.write(nb(cells), OUT / f"{name}.ipynb")
    print("wrote", name)
