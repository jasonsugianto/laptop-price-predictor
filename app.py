import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Spec2Price",
    layout="centered",
)

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

def preprocess(raw: dict) -> pd.DataFrame:
    df = pd.DataFrame([raw])

    df["Resolution_Pixels"] = df["resolution_width"] * df["resolution_height"]
    df["is_touch_screen"]   = df["is_touch_screen"].astype(int)
    df.drop(columns=["resolution_width", "resolution_height"], inplace=True)

    df["Rating"] = df["Rating"] / 20

    df[num_cols] = imputer.transform(df[num_cols])

    lowercase_cols = ["brand", "processor_brand", "processor_tier", 
                  "gpu_brand", "gpu_type", "OS"]
    
    for col in lowercase_cols:
        df[col] = df[col].str.lower()

    for col in cat_cols:
        le = encoders[col]
        df[col] = df[col].astype(str).map(
            lambda x, le=le: le.transform([x])[0] if x in le.classes_ else -1
        )

    df = df.reindex(columns=feature_cols, fill_value=0)
    return df

st.title("Spec2Price")
st.markdown("Laptop Price Predictor")
st.markdown("Estimate a laptop's market price using **Gradient Boosting** (R² = 0.93).")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Brand & OS")
    brand = st.selectbox("Brand", [
    "tecno", "hp", "acer", "lenovo", "apple", "infinix", "asus", "dell",
    "samsung", "msi", "wings", "ultimus", "primebook", "iball", "zebronics",
    "chuwi", "gigabyte", "jio", "honor", "realme", "avita", "microsoft",
    "fujitsu", "lg", "walker", "axl"
    ])
    os = st.selectbox("Operating System", ["windows", "mac", "dos", "android", "chrome", "ubuntu", "other"])
    year_warranty = st.selectbox("Warranty (years)", [1, 2, 3])

with col2:
    st.subheader("Processor")
    processor_brand = st.selectbox("Processor Brand", [
    "intel", "amd", "apple", "other"
    ])
    processor_tier = st.selectbox("Processor Tier", [
    "core i3", "core i5", "core i7", "core i9", "core ultra 7",
    "ryzen 3", "ryzen 5", "ryzen 7", "ryzen 9",
    "m1", "m2", "m3",
    "celeron", "pentium", "other"
    ])
    num_cores = st.slider("Number of Cores",   2, 24, 6)
    num_threads = st.slider("Number of Threads", 0, 32, 12)

st.divider()
col3, col4 = st.columns(2)

with col3:
    st.subheader("Memory & Storage")
    ram_memory = st.selectbox("RAM (GB)", [2, 4, 8, 12, 16, 24, 32, 36, 64])
    primary_storage_type = st.selectbox("Primary Storage Type", ["SSD", "HDD"])
    primary_storage_capacity = st.selectbox("Primary Storage (GB)", [32, 64, 128, 256, 512, 1024, 2048])
    secondary_storage_type = st.selectbox("Secondary Storage Type", ["No secondary storage", "SSD"])
    secondary_storage_capacity = st.selectbox("Secondary Storage (GB)", [0, 128, 256, 512])

with col4:
    st.subheader("Display & GPU")
    display_size = st.selectbox("Display Size (inches)", [10.1, 11.6, 12.4, 13.0, 13.3, 13.4, 13.5, 13.6, 14.0, 14.1, 14.2, 14.5, 15.0, 15.3, 15.6, 16.0, 16.1, 16.2, 17.3, 18.0])
    resolution_width = st.selectbox("Resolution Width",  [1080, 1200, 1280, 1366, 1440, 1536, 1600, 1920, 2048, 2160, 2240, 2256, 2560, 2880, 3000, 3024, 3072, 3200, 3456, 3840])
    resolution_height = st.selectbox("Resolution Height", [768, 800, 1024, 1080, 1200, 1400, 1440, 1504, 1536, 1600, 1620, 1660, 1664, 1800, 1864, 1920, 1964, 2000, 2160, 2234, 2400, 2560])
    gpu_brand = st.selectbox("GPU Brand", ["intel", "amd", "apple", "nvidia", "arm"])
    gpu_type = st.selectbox("GPU Type",  ["integrated", "dedicated", "apple"])
    is_touch_screen = st.checkbox("Touchscreen")

st.divider()

if st.button("Predict Price", use_container_width=True, type="primary"):
    raw = {
        "brand": brand,
        "Rating": 4.0,
        "processor_brand": processor_brand,
        "processor_tier": processor_tier,
        "num_cores": num_cores,
        "num_threads": num_threads,
        "ram_memory": ram_memory,
        "primary_storage_type": primary_storage_type,
        "primary_storage_capacity": primary_storage_capacity,
        "secondary_storage_type": secondary_storage_type,
        "secondary_storage_capacity": secondary_storage_capacity,
        "gpu_brand": gpu_brand,
        "gpu_type": gpu_type,
        "is_touch_screen": int(is_touch_screen),
        "display_size": display_size,
        "resolution_width": resolution_width,
        "resolution_height": resolution_height,
        "OS": os,
        "year_of_warranty": year_warranty,
    }

    try:
        X_input = preprocess(raw)
        log_price = model.predict(X_input)[0]
        predicted_price = np.expm1(log_price)

        predicted_price_rupiah = predicted_price * 187.87

        st.success(f"### Estimated Price: Rp {predicted_price_rupiah:,.2f}")
        st.caption("INR/IDR rate: 187.87 · May vary with market")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.divider()
st.caption("Classical ML Project · Gradient Boosting Regressor · R² = 0.93 on 991 laptops")