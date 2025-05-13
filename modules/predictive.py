# modules/predictive.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def app():
    st.header("🤖 Predictive Pricing")

    # 1. Load model & data
    rf = joblib.load("models/best_random_forest.pkl")
    df = pd.read_csv("data/processed/featurized_final.csv", low_memory=False)

    # 2. Define engineered‐spec columns
    spec_cols = ["CPU_perf", "log_RAM_GB", "ppi", "log_SSD_GB"]

    # 3. Recompute cluster mapping on full catalog
    X_specs = df[spec_cols].fillna(0)
    scaler  = StandardScaler().fit(X_specs)
    kmeans  = KMeans(n_clusters=4, random_state=42)\
                  .fit(scaler.transform(X_specs))
    df["cluster"] = kmeans.labels_

    # 4. Determine all numeric features RF expects
    numeric_cols = (
    df.select_dtypes(include=[np.number])
      .drop(columns=["Precio_avg","LogPrice","Precio_min","Precio_max"], errors="ignore")
      .columns
      .tolist()
    )

    # 5. Build median defaults for each numeric feature
    defaults = df[numeric_cols].median().to_dict()

    # 6. Sidebar sliders for raw inputs
    st.sidebar.subheader("Configure your machine")

    cores = st.sidebar.slider(
        "CPU cores",
        int(df["Procesador_Cores"].min()),
        int(df["Procesador_Cores"].max()),
        int(df["Procesador_Cores"].median()),
        key="cores"
    )
    ghz = st.sidebar.slider(
        "Clock speed (GHz)",
        1.0, 5.0,
        float(df["CPU_GHz"].median()),
        step=0.1,
        key="ghz"
    )
    ram = st.sidebar.slider(
        "RAM (GB)",
        int(df["RAM_GB"].min()),
        int(df["RAM_GB"].max()),
        int(df["RAM_GB"].median()),
        key="ram"
    )
    ssd99 = int(df["Disco duro_Capacidad de memoria SSD"].quantile(0.99))
    ssd = st.sidebar.slider(
        "SSD (GB)", 0, ssd99,
        int(df["Disco duro_Capacidad de memoria SSD"].median()),
        key="ssd"
    )
    ppi = st.sidebar.slider(
        "Screen PPI",
        100.0, 350.0,
        float((np.sqrt(df["Res_Horiz_px"]**2 + df["Res_Vert_px"]**2)
               / df["Pantalla_Tamaño_pulg"]).median()),
        step=1.0,
        key="ppi"
    )

    # 7. Build query vector from defaults
    Xq = pd.DataFrame([defaults])

    # 8. Override raw inputs
    Xq["Procesador_Cores"]                    = cores
    Xq["CPU_GHz"]                              = ghz
    Xq["RAM_GB"]                               = ram
    Xq["Disco duro_Capacidad de memoria SSD"]  = ssd

    # 9. Override engineered specs
    Xq["CPU_perf"]   = cores * ghz
    Xq["log_RAM_GB"] = np.log1p(ram)
    Xq["ppi"]        = ppi
    Xq["log_SSD_GB"] = np.log1p(ssd)

    # 10. Recompute cluster for this query
    Xq["cluster"] = kmeans.predict(scaler.transform(Xq[spec_cols]))[0]

    # Debug: show the final query vector
    st.sidebar.write("🔧 Slider values", {
        "cores": cores, "ghz": ghz, "ram": ram, "ssd": ssd, "ppi": ppi
    })
    st.write("🔧 Query vector Xq:", Xq.iloc[0])

    # 11. Predict & display price
    Xq = Xq[numeric_cols]
    logp  = rf.predict(Xq)[0]
    price = np.expm1(logp)
    
    # ─── 12. Display with a placeholder to avoid caching ───────────
    price_placeholder = st.empty()
    price_placeholder.metric("💰 Predicted Price (€)", f"{price:,.0f}")

    # ─── 13. SHAP explanation ───────────────────────────────────────
    explainer = shap.TreeExplainer(rf)
    shap_vals = explainer.shap_values(Xq)
    st.subheader("🔎 Feature Contributions (SHAP)")
    shap.initjs()
    fp = shap.force_plot(
        explainer.expected_value,
        shap_vals[0],
        Xq.iloc[0],
        feature_names=Xq.columns.tolist(),
        matplotlib=False
    )
    st.components.v1.html(fp.html(), height=400)