# pages/descriptive.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def app():
    st.header("🔍 Descriptive Analytics")

    # 1. Load featurized data
    df = pd.read_csv("../data/processed/featurized_selected.csv", low_memory=False)

    # 2. Recompute segments via clustering on key specs
    features = ['CPU_perf', 'log_RAM_GB', 'ppi', 'log_SSD_GB']
    X = df[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=4, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    # Map numeric clusters to named segments
    seg_map = {0: 'Budget', 1: 'Mid-Range', 2: 'High-End', 3: 'Ultra-Premium'}
    df['segment'] = [seg_map[i] for i in labels]

    # 3. Display segment profiles
    seg_df = pd.read_csv("../data/processed/featurized_selected.csv", index_col=0)
    st.subheader("Segment Profiles")
    st.dataframe(seg_df.style.format("{:.2f}"))

    # 4. Reconstruct Product_Type from one-hot columns
    prod_cols = [c for c in df.columns if c.startswith("Tipo_")]
    df['Product_Type'] = (
        df[prod_cols]
          .idxmax(axis=1)
          .str.replace("Tipo_", "", regex=False)
    )

    # 5. Top 5 Product Types bar chart
    st.subheader("Top 5 Product Types")
    top5 = df['Product_Type'].value_counts().nlargest(5)
    fig1, ax1 = plt.subplots()
    sns.barplot(x=top5.values, y=top5.index, palette='magma', ax=ax1)
    ax1.set_xlabel("Count")
    ax1.set_ylabel("Product Type")
    st.pyplot(fig1)

    # 6. Price distribution by segment
    st.subheader("Price Distribution by Segment")
    order = ['Budget', 'Mid-Range', 'High-End', 'Ultra-Premium']
    fig2, ax2 = plt.subplots()
    sns.boxplot(
        data=df,
        x='segment',
        y='Precio_avg',
        order=order,
        ax=ax2
    )
    ax2.set_xlabel("Segment")
    ax2.set_ylabel("Average Price (€)")
    st.pyplot(fig2)
