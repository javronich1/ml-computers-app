# ──────────────────────────────────────────────────────────────────────
# 0️⃣  Imports
# ──────────────────────────────────────────────────────────────────────
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor

# ──────────────────────────────────────────────────────────────────────
# 1️⃣  Load data  ➜  drop price-range cols that leak information
# ──────────────────────────────────────────────────────────────────────
DF_PATH = Path("../data/processed/featurized_selected.csv")     # <— adjust if needed
df      = pd.read_csv(DF_PATH, low_memory=False)

# *NEVER* leave rows without a target
df = df.dropna(subset=["Precio_avg"])

# target
df["LogPrice"] = np.log1p(df["Precio_avg"])
y  = df["LogPrice"]

# numeric matrix only
X  = (
    df.select_dtypes(include="number")
      .replace([np.inf, -np.inf], np.nan)
      .drop(columns=["Precio_avg", "Precio_min", "Precio_max", "LogPrice"],
            errors="ignore")
)

# ──────────────────────────────────────────────────────────────────────
# 2️⃣  Train / test split
# ──────────────────────────────────────────────────────────────────────
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=None
)

# ──────────────────────────────────────────────────────────────────────
# 3️⃣  Build a *single* pipeline  (imputer → HistGB)
#     ⚠️ HistGB in <-sklearn-> 1.1 does **not** accept “subsample=”
# ──────────────────────────────────────────────────────────────────────
hgb_params = dict(
    learning_rate    = 0.05,
    max_iter         = 1_000,
    min_samples_leaf = 20,
    max_depth        = None,
    random_state     = 42
)

pipe = Pipeline([
    ("imp",  SimpleImputer(strategy="median")),
    ("hgb",  HistGradientBoostingRegressor(**hgb_params))
])

pipe.fit(X_tr, y_tr)

# ──────────────────────────────────────────────────────────────────────
# 4️⃣  Metrics
# ──────────────────────────────────────────────────────────────────────
pred_log  = pipe.predict(X_te)
rmse_log  = np.sqrt(mean_squared_error(y_te, pred_log))
rmse_eur  = np.sqrt(
                mean_squared_error(
                    np.expm1(y_te),
                    np.expm1(pred_log)
                )
            )

print(f"✅  Test RMSE_log   : {rmse_log:.3f}")
print(f"✅  Test RMSE_price : €{rmse_eur:,.2f}")

# ──────────────────────────────────────────────────────────────────────
# 5️⃣  Persist for Streamlit
# ──────────────────────────────────────────────────────────────────────
MODEL_PATH = Path("../models/hgb_v1.pkl")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(pipe, MODEL_PATH)
print(f"📦  Saved full pipeline ➜ {MODEL_PATH.relative_to(Path.cwd())}")