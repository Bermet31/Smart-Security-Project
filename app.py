import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# ----- Streamlit Config -----
st.set_page_config(page_title="Smart Security", layout="wide")
st.title("Smart Security for Student & Faculty Logins")

# ----- Load CSV -----
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

try:
    raw_df = load_data("cleaned_logins.csv")
except FileNotFoundError:
    st.error("cleaned_logins.csv missing.")
    st.stop()

# st.subheader("Columns")
# st.write(list(raw_df.columns))

# ----- Rename common columns -----
rename_dict = {
    "StudentID": "student_id",
    "CourseID": "major",
    "LoginAttempts": "login_attempts",
    "IPAddress": "ip_address",
    "LoginType": "login_success",
    "PreviousLoginTime": "last_login_time",
}
df = raw_df.rename(columns=rename_dict)

# Preserve original anomaly flags if exist
if "anomaly" in df.columns:
    df = df.rename(columns={"anomaly": "anomaly_original"})
if "threshold_flag" in df.columns:
    df = df.rename(columns={"threshold_flag": "threshold_original"})

# ----- Keep only relevant columns -----
cols = ["student_id", "major", "login_attempts", "ip_address",
        "login_success", "last_login_time"]
df = df[[c for c in cols if c in df.columns]].copy()

# ----- Data cleaning -----
if "login_attempts" in df.columns:
    df["login_attempts"] = pd.to_numeric(df["login_attempts"], errors="coerce").fillna(0)

if "last_login_time" in df.columns:
    df["last_login_time"] = pd.to_datetime(df["last_login_time"], errors="coerce")
    df["login_date"] = df["last_login_time"].dt.date

# ----- Sidebar controls -----
st.sidebar.header("Settings")
contamination = st.sidebar.slider("IsolationForest contamination", 0.01, 0.20, 0.05)
n_clusters = st.sidebar.slider("KMeans clusters", 2, 6, 3)
threshold_quantile = st.sidebar.slider("Threshold quantile", 0.80, 0.99, 0.95)

# ----- Encode categories for ML -----
ml_df = df.copy()
for col in ml_df.select_dtypes(include="object"):
    ml_df[col] = LabelEncoder().fit_transform(ml_df[col].astype(str))

# Remove IDs / datetime
exclude = {"student_id", "last_login_time", "login_date"}
X = ml_df[[c for c in ml_df.columns if c not in exclude]]

if X.shape[1] < 2:
    st.error("Need more numeric features.")
    st.stop()

# ----- Isolation Forest -----
iso = IsolationForest(contamination=contamination, random_state=42)
df["anomaly_iforest"] = (iso.fit_predict(X) == -1).astype(int)

# ----- KMeans distance anomaly -----
km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
clusters = km.fit_predict(X)
dist = km.transform(X).min(axis=1)
km_thr = np.quantile(dist, threshold_quantile)

df["cluster"] = clusters
df["anomaly_kmeans"] = (dist > km_thr).astype(int)

# ----- Threshold rule -----
if "login_attempts" in df.columns:
    thr_val = df["login_attempts"].quantile(threshold_quantile)
    df["threshold_flag"] = (df["login_attempts"] >= thr_val).astype(int)
else:
    thr_val = None
    df["threshold_flag"] = 0

# ----- Combined flag -----
df["suspicious_overall"] = (
    df["anomaly_iforest"] | df["anomaly_kmeans"] | df["threshold_flag"]
).astype(int)

# ----- Summary section -----
st.subheader("Dataset Preview")
st.dataframe(df.head(100))

st.subheader("Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total", len(df))
col2.metric("Max Attempts", int(df["login_attempts"].max()))
col3.metric("IF Anomalies", int(df["anomaly_iforest"].sum()))
col4.metric("KM Anomalies", int(df["anomaly_kmeans"].sum()))

# ----- Stats -----
st.subheader("Stats")

if "login_success" in df.columns:
    st.bar_chart(df["login_success"].value_counts())

if "ip_address" in df.columns:
    st.bar_chart(df["ip_address"].value_counts().head(10))

# ----- Time charts -----
if "login_date" in df.columns:
    st.subheader("Daily Activity")

    daily = df.groupby("login_date").size()
    daily_anom = df.groupby("login_date")["suspicious_overall"].sum()

    fig1, ax1 = plt.subplots()
    ax1.plot(daily.index, daily.values)
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.bar(daily_anom.index, daily_anom.values)
    st.pyplot(fig2)

# ----- Cluster anomalies -----
st.subheader("Cluster Summary")
summ = df.groupby("cluster")[["anomaly_kmeans", "anomaly_iforest"]].sum()
st.dataframe(summ)

fig3, ax3 = plt.subplots()
summ["anomaly_kmeans"].plot(kind="bar", ax=ax3)
st.pyplot(fig3)

# ----- Suspicious events -----
st.subheader("Suspicious Logins")

susp = df[df["suspicious_overall"] == 1]

if susp.empty:
    st.success("No suspicious logins.")
else:
    st.warning(f"{len(susp)} suspicious events.")
    st.dataframe(susp.head(200))

    st.download_button(
        "Download CSV",
        susp.to_csv(index=False),
        "suspicious_logins.csv",
        "text/csv"
    )
