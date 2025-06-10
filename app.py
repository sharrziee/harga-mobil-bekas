
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import gdown
import os
import joblib

model_path = "model_rf.pkl"
if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?id=1uqQ80WNa3JhFdhH-wNoW9aaflhz4SMe-"
    gdown.download(url, model_path, quiet=False)

model = joblib.load(model_path)

st.set_page_config(page_title="Dashboard Harga Mobil Bekas", layout="wide")

# Load model dan data
model = joblib.load("model_rf.pkl")
columns = joblib.load("columns.pkl")
df_encoded = pd.read_csv("data_encoded.csv")
eval_df = pd.read_csv("model_eval_result.csv")

# Sidebar navigasi
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", [
    "📊 Eksplorasi Data", 
    "🧪 Evaluasi Model", 
    "📉 Distribusi Residual", 
    "🔥 Feature Importance", 
    "🔮 Prediksi Harga"
])

if page == "📊 Eksplorasi Data":
    st.title("📊 Eksplorasi Data")

    st.subheader("Heatmap Korelasi")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_encoded.corr(numeric_only=True), cmap="coolwarm", ax=ax1)
    st.pyplot(fig1)

    st.subheader("Distribusi Harga Mobil")
    fig2, ax2 = plt.subplots()
    sns.histplot(df_encoded["price_usd"], bins=50, kde=True, ax=ax2)
    st.pyplot(fig2)

    st.subheader("Tahun Produksi vs Harga")
    fig3, ax3 = plt.subplots()
    sns.scatterplot(data=df_encoded, x="year_produced", y="price_usd", ax=ax3, alpha=0.3)
    st.pyplot(fig3)

    st.subheader("Boxplot Harga berdasarkan Jenis Bahan Bakar")
    fuel_cols = [col for col in df_encoded.columns if col.startswith("engine_fuel_")]
    melted = df_encoded.melt(id_vars="price_usd", value_vars=fuel_cols, var_name="fuel", value_name="value")
    melted = melted[melted["value"] == 1]
    melted["fuel"] = melted["fuel"].str.replace("engine_fuel_", "")
    fig4, ax4 = plt.subplots()
    sns.boxplot(x="fuel", y="price_usd", data=melted, ax=ax4)
    st.pyplot(fig4)

elif page == "🧪 Evaluasi Model":
    st.title("🧪 Evaluasi Model")

    st.subheader("Perbandingan Metrik Evaluasi")
    mae_rf = np.mean(np.abs(eval_df["Actual"] - eval_df["RF_Pred"]))
    mse_rf = np.mean((eval_df["Actual"] - eval_df["RF_Pred"])**2)
    rmse_rf = np.sqrt(mse_rf)
    r2_rf = 1 - mse_rf / np.var(eval_df["Actual"])

    mae_dt = np.mean(np.abs(eval_df["Actual"] - eval_df["DT_Pred"]))
    mse_dt = np.mean((eval_df["Actual"] - eval_df["DT_Pred"])**2)
    rmse_dt = np.sqrt(mse_dt)
    r2_dt = 1 - mse_dt / np.var(eval_df["Actual"])

    st.write(pd.DataFrame({
        "Model": ["Random Forest", "Decision Tree"],
        "MAE": [mae_rf, mae_dt],
        "MSE": [mse_rf, mse_dt],
        "RMSE": [rmse_rf, rmse_dt],
        "R2 Score": [r2_rf, r2_dt]
    }))

    st.subheader("Visualisasi Aktual vs Prediksi")
    fig5, ax5 = plt.subplots()
    ax5.scatter(eval_df["Actual"], eval_df["RF_Pred"], alpha=0.3, label="Random Forest")
    ax5.scatter(eval_df["Actual"], eval_df["DT_Pred"], alpha=0.3, label="Decision Tree", color="orange")
    ax5.plot([eval_df["Actual"].min(), eval_df["Actual"].max()],
             [eval_df["Actual"].min(), eval_df["Actual"].max()],
             "r--")
    ax5.set_xlabel("Harga Aktual")
    ax5.set_ylabel("Prediksi")
    ax5.legend()
    st.pyplot(fig5)

elif page == "📉 Distribusi Residual":
    st.title("📉 Distribusi Residual")

    st.subheader("Random Forest Residual")
    fig6, ax6 = plt.subplots()
    sns.histplot(eval_df["RF_Residual"], bins=50, kde=True, ax=ax6)
    st.pyplot(fig6)

    st.subheader("Decision Tree Residual")
    fig7, ax7 = plt.subplots()
    sns.histplot(eval_df["DT_Residual"], bins=50, kde=True, ax=ax7, color="orange")
    st.pyplot(fig7)

elif page == "🔥 Feature Importance":
    st.title("🔥 Feature Importance")

    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    sorted_features = np.array(columns)[sorted_idx]
    sorted_importances = importances[sorted_idx]

    fig8, ax8 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=sorted_importances[:15], y=sorted_features[:15], ax=ax8)
    st.pyplot(fig8)

elif page == "🔮 Prediksi Harga":
    st.title("🔮 Prediksi Harga Mobil Bekas")

    odometer = st.number_input("Jarak Tempuh (km)", 0, 1000000, 50000)
    year = st.number_input("Tahun Produksi", 1900, 2025, 2015)
    engine_capacity = st.number_input("Kapasitas Mesin (L)", 0.5, 10.0, 2.0)
    has_warranty = st.selectbox("Garansi?", ["Ya", "Tidak"])
    drivetrain = st.selectbox("Penggerak Roda", ["front", "rear", "all"])
    transmission = st.selectbox("Transmisi", ["automatic", "mechanical"])
    engine_fuel = st.selectbox("Bahan Bakar", ["gasoline", "diesel", "electric", "hybrid", "other"])
    num_photos = st.slider("Jumlah Foto Iklan", 0, 50, 10)
    duration_listed = st.slider("Durasi Iklan (hari)", 0, 500, 30)
    up_counter = st.slider("Jumlah Update Iklan", 0, 100, 5)

    input_data = {
        'odometer_value': [odometer],
        'year_produced': [year],
        'engine_capacity': [engine_capacity],
        'has_warranty': [1 if has_warranty == "Ya" else 0],
        'number_of_photos': [num_photos],
        'duration_listed': [duration_listed],
        'up_counter': [up_counter]
    }

    for col in columns:
        if col not in input_data:
            input_data[col] = [0]

    # One-hot
    if f"drivetrain_{drivetrain}" in columns:
        input_data[f"drivetrain_{drivetrain}"] = [1]
    if f"transmission_{transmission}" in columns:
        input_data[f"transmission_{transmission}"] = [1]
    if f"engine_fuel_{engine_fuel}" in columns:
        input_data[f"engine_fuel_{engine_fuel}"] = [1]

    input_df = pd.DataFrame(input_data)[columns]

    if st.button("Prediksi"):
        pred = model.predict(input_df)[0]
        st.success(f"Estimasi Harga Mobil: ${pred:,.2f}")
