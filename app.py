import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Laptop Price Predictor",
    layout="centered",
)

# ─────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model        = joblib.load("model.pkl")
    imputer      = joblib.load("imputer.pkl")
    encoders     = joblib.load("encoders.pkl")
    num_cols     = joblib.load("num_cols.pkl")
    cat_cols     = joblib.load("cat_cols.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    return model, imputer, encoders, num_cols, cat_cols, feature_cols

model, imputer, encoders, num_cols, cat_cols, feature_cols = load_artifacts()

# ─────────────────────────────────────────────
# PREPROCESSING (must exactly match train.py)
# ─────────────────────────────────────────────
def preprocess(raw: dict) -> pd.DataFrame:
    df = pd.DataFrame([raw])

    # Derived columns
    df["Resolution_Pixels"] = df["resolution_width"] * df["resolution_height"]
    df["is_touch_screen"]   = df["is_touch_screen"].astype(int)
    df.drop(columns=["resolution_width", "resolution_height"], inplace=True)

    # Impute numerics
    df[num_cols] = imputer.transform(df[num_cols])

    # Encode categoricals
    for col in cat_cols:
        le = encoders[col]
        df[col] = df[col].astype(str).map(
            lambda x, le=le: le.transform([x])[0] if x in le.classes_ else -1
        )

    # Align to exact training column order
    df = df.reindex(columns=feature_cols, fill_value=0)
    return df

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.title("Laptop Price Predictor")
st.markdown("Estimate a laptop's market price using **Gradient Boosting** (R² = 0.93).")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Brand & OS")
    brand         = st.selectbox("Brand", ["Dell", "HP", "Lenovo", "Apple", "Asus", "Acer", "MSI", "Samsung", "Microsoft", "Other"])
    os            = st.selectbox("Operating System", ["Windows", "macOS", "Linux", "Chrome OS", "Other"])
    year_warranty = st.selectbox("Warranty (years)", [1, 2, 3])

with col2:
    st.subheader("Processor")
    processor_brand = st.selectbox("Processor Brand", ["Intel", "AMD", "Apple", "Qualcomm", "Other"])
    processor_tier  = st.selectbox("Processor Tier", ["Core i3", "Core i5", "Core i7", "Core i9", "Ryzen 3", "Ryzen 5", "Ryzen 7", "Ryzen 9", "M1", "M2", "M3", "Other"])
    num_cores       = st.slider("Number of Cores",   2, 24, 8)
    num_threads     = st.slider("Number of Threads", 2, 32, 16)

st.divider()
col3, col4 = st.columns(2)

with col3:
    st.subheader("Memory & Storage")
    ram_memory                 = st.selectbox("RAM (GB)", [4, 8, 16, 32, 64, 128])
    primary_storage_type       = st.selectbox("Primary Storage Type", ["SSD", "HDD", "eMMC", "NVMe SSD"])
    primary_storage_capacity   = st.selectbox("Primary Storage (GB)", [128, 256, 512, 1024, 2048])
    secondary_storage_type     = st.selectbox("Secondary Storage Type", ["None", "HDD", "SSD"])
    secondary_storage_capacity = st.selectbox("Secondary Storage (GB)", [0, 512, 1024, 2048])

with col4:
    st.subheader("Display & GPU")
    display_size      = st.selectbox("Display Size (inches)", [11.6, 13.3, 14.0, 15.6, 16.0, 17.3])
    resolution_width  = st.selectbox("Resolution Width",  [1366, 1920, 2560, 3840])
    resolution_height = st.selectbox("Resolution Height", [768,  1080, 1440, 2160])
    gpu_brand         = st.selectbox("GPU Brand", ["Intel", "NVIDIA", "AMD", "Apple", "Other"])
    gpu_type          = st.selectbox("GPU Type",  ["Integrated", "Dedicated"])
    is_touch_screen   = st.checkbox("Touchscreen")

st.divider()

# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────
if st.button("🔍 Predict Price", use_container_width=True, type="primary"):
    raw = {
        "brand":                      brand,
        "Rating":                     4.0,
        "processor_brand":            processor_brand,
        "processor_tier":             processor_tier,
        "num_cores":                  num_cores,
        "num_threads":                num_threads,
        "ram_memory":                 ram_memory,
        "primary_storage_type":       primary_storage_type,
        "primary_storage_capacity":   primary_storage_capacity,
        "secondary_storage_type":     secondary_storage_type,
        "secondary_storage_capacity": secondary_storage_capacity,
        "gpu_brand":                  gpu_brand,
        "gpu_type":                   gpu_type,
        "is_touch_screen":            int(is_touch_screen),
        "display_size":               display_size,
        "resolution_width":           resolution_width,
        "resolution_height":          resolution_height,
        "OS":                         os,
        "year_of_warranty":           year_warranty,
    }

    try:
        X_input         = preprocess(raw)
        log_price       = model.predict(X_input)[0]
        predicted_price = np.expm1(log_price)

        predicted_price_rupiah = predicted_price * 187.87

        st.success(f"### Estimated Price: Rp {predicted_price_rupiah:,.2f}")
        st.caption("Based on Gradient Boosting trained on 991 laptops · R² = 0.93")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.divider()
st.caption("Classical ML Project · Gradient Boosting Regressor · No feature engineering needed")