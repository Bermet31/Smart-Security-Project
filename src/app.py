import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

import bcrypt
import streamlit as st
import pandas as pd

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed.encode())


# -----------------------------------------------------------
# Load CSS
# -----------------------------------------------------------
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("src/styles/nau_style.css")

# Load users database
users_df = pd.read_csv("src/users.csv")


# -----------------------------------------------------------
# Authentication State
# -----------------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "full_name" not in st.session_state:
    st.session_state.full_name = None


# -----------------------------------------------------------
# Show login screen if NOT logged in
# -----------------------------------------------------------
if not st.session_state.logged_in:
    st.header("Login Required")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_btn = st.button("Login")

    if login_btn:
        if username in users_df["username"].values:
            stored_hash = users_df.loc[
                users_df["username"] == username, "password_hash"
            ].values[0]
            full_name = users_df.loc[
                users_df["username"] == username, "full_name"
            ].values[0]

            if check_password(password, stored_hash):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.full_name = full_name

                st.success("✔ Login successful")
                st.rerun()
            else:
                st.error("❌ Incorrect password")
        else:
            st.error("❌ User does not exist")

    st.stop()


# -----------------------------------------------------------
# Sidebar — show user & logout
# -----------------------------------------------------------
st.sidebar.markdown(f"👤 {st.session_state.full_name}")
logout_btn = st.sidebar.button("Logout")

if logout_btn:
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.full_name = None
    st.success("✔ Logged out")
    st.rerun()

# -----------------------------------------------------------
#   Streamlit Config
# -----------------------------------------------------------
st.set_page_config(page_title="Smart Security", layout="wide")

# NAU header bar
st.markdown(
    """
    <div class="nau-header">
        <div>
            <p class="nau-header-title">Smart Security for Student &amp; Faculty Logins</p>
            <p class="nau-header-subtitle">
                North American University · Anomaly Detection Dashboard
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar logo
try:
    st.sidebar.image("assets/NAU-Logo.png", use_container_width=True)
except:
    pass

st.sidebar.markdown("### Navigation")

page = st.sidebar.selectbox(
    "",
    ["Overview", "Daily Activity", "Cluster Analysis", "Suspicious Logins"]
)

st.sidebar.markdown("---")


# -----------------------------------------------------------
#   Load CSV
# -----------------------------------------------------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

try:
    raw_df = load_data("data/cleaned_logins.csv")
except FileNotFoundError:
    st.error("cleaned_logins.csv missing.")
    st.stop()

# -----------------------------------------------------------
#   Rename columns
# -----------------------------------------------------------
rename_dict = {
    "StudentID": "student_id",
    "CourseID": "major",
    "LoginAttempts": "login_attempts",
    "IPAddress": "ip_address",
    "LoginType": "login_success",
    "PreviousLoginTime": "last_login_time",
}
df = raw_df.rename(columns=rename_dict)

# Normalize login_success values
if "login_success" in df.columns:
    df["login_success"] = df["login_success"].astype(str).str.strip().str.lower()
    df["login_success"] = df["login_success"].replace({
        "credit": "success",
        "debit": "failed"
    })

# Keep relevant columns
cols = ["student_id", "major", "login_attempts", "ip_address",
        "login_success", "last_login_time"]
df = df[[c for c in cols if c in df.columns]].copy()

# -----------------------------------------------------------
#   Data cleaning
# -----------------------------------------------------------
if "login_attempts" in df.columns:
    df["login_attempts"] = pd.to_numeric(df["login_attempts"], errors="coerce").fillna(0)

if "last_login_time" in df.columns:
    df["last_login_time"] = pd.to_datetime(df["last_login_time"], errors="coerce")
    df["login_date"] = df["last_login_time"].dt.date

# -----------------------------------------------------------
#   Sidebar controls
# -----------------------------------------------------------
st.sidebar.header("Settings")
contamination = st.sidebar.slider("IsolationForest contamination", 0.01, 0.20, 0.05)
n_clusters = st.sidebar.slider("KMeans clusters", 2, 6, 3)
threshold_quantile = st.sidebar.slider("Threshold quantile", 0.80, 0.99, 0.95)

st.sidebar.header("Filters")
student_filter = st.sidebar.text_input("Student ID contains")
ip_filter = st.sidebar.text_input("IP address contains")
only_suspicious = st.sidebar.checkbox("Show only suspicious", value=False)

# -----------------------------------------------------------
#   ML Encoding
# -----------------------------------------------------------
ml_df = df.copy()
for col in ml_df.select_dtypes(include="object"):
    ml_df[col] = LabelEncoder().fit_transform(ml_df[col].astype(str))

exclude = {"student_id", "last_login_time", "login_date"}
X = ml_df[[c for c in ml_df.columns if c not in exclude]]

if X.shape[1] < 2:
    st.error("Need more numeric features.")
    st.stop()

# -----------------------------------------------------------
#   Anomaly Detection
# -----------------------------------------------------------
iso = IsolationForest(contamination=contamination, random_state=42)
df["anomaly_iforest"] = (iso.fit_predict(X) == -1).astype(int)

km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
clusters = km.fit_predict(X)
dist = km.transform(X).min(axis=1)
km_thr = np.quantile(dist, threshold_quantile)

df["cluster"] = clusters
df["anomaly_kmeans"] = (dist > km_thr).astype(int)

if "login_attempts" in df.columns:
    thr_val = df["login_attempts"].quantile(threshold_quantile)
    df["threshold_flag"] = (df["login_attempts"] >= thr_val).astype(int)
else:
    thr_val = None
    df["threshold_flag"] = 0

df["suspicious_overall"] = (
    df["anomaly_iforest"] | df["anomaly_kmeans"] | df["threshold_flag"]
).astype(int)

# -----------------------------------------------------------
#   Risk Score
# -----------------------------------------------------------
risk = np.zeros(len(df), dtype=float)

if "login_attempts" in df.columns and df["login_attempts"].max() > 0:
    risk += (df["login_attempts"] / df["login_attempts"].max()) * 40

risk += df["anomaly_iforest"] * 30
risk += df["anomaly_kmeans"] * 20
risk += df["threshold_flag"] * 10

df["risk_score"] = np.round(np.clip(risk, 0, 100), 1)


# -----------------------------------------------------------
#   Filters
# -----------------------------------------------------------
filtered_df = df.copy()

if student_filter:
    filtered_df = filtered_df[
        filtered_df["student_id"].astype(str).str.contains(student_filter, case=False, na=False)
    ]

if ip_filter and "ip_address" in filtered_df.columns:
    filtered_df = filtered_df[
        filtered_df["ip_address"].astype(str).str.contains(ip_filter, case=False, na=False)
    ]

if only_suspicious:
    filtered_df = filtered_df[filtered_df["suspicious_overall"] == 1]

columns_to_show = [
    "student_id",
    "login_attempts",
    "ip_address",
    "login_success",
    "last_login_time",
    "threshold_flag",
    "anomaly_iforest",
    "anomaly_kmeans",
    "suspicious_overall",
    "risk_score"
]
columns_existing = [c for c in columns_to_show if c in filtered_df.columns]
df_display = filtered_df[columns_existing].copy()

# -----------------------------------------------------------
#   UI Sections
# -----------------------------------------------------------

# -----------------------------------------------------------
# Color formatting for login_success column
# -----------------------------------------------------------
def color_login_status(val):
    if val == "success":
        return "color: green; font-weight: bold;"
    elif val == "failed":
        return "color: red; font-weight: bold;"
    else:
        return ""


# OVERVIEW
if page == "Overview":
    st.subheader("Dataset Preview")
    st.dataframe(
    df_display.style.applymap(color_login_status, subset=["login_success"])
)


    st.subheader("Summary (Filtered)")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Rows", len(filtered_df))
    col2.metric("Max Attempts", int(filtered_df["login_attempts"].max()))
    col3.metric("IF Anomalies", int(filtered_df["anomaly_iforest"].sum()))
    col4.metric("KM Anomalies", int(filtered_df["anomaly_kmeans"].sum()))
    col5.metric("Average Risk", float(filtered_df["risk_score"].mean()))

    if "login_success" in filtered_df.columns:
        st.bar_chart(filtered_df["login_success"].value_counts())

# DAILY ACTIVITY
elif page == "Daily Activity":
    st.subheader("Daily Login Activity")

    if "login_date" in filtered_df.columns:
        daily = filtered_df.groupby("login_date").size()
        st.bar_chart(daily)

# CLUSTER ANALYSIS
elif page == "Cluster Analysis":
    st.subheader("Cluster Summary")

    if "cluster" in filtered_df.columns:
        summ = filtered_df.groupby("cluster")[["anomaly_kmeans", "anomaly_iforest"]].sum()
        st.dataframe(summ)

# SUSPICIOUS LOGINS
elif page == "Suspicious Logins":
    st.subheader("Suspicious Logins")

    susp = filtered_df[filtered_df["suspicious_overall"] == 1]

    if susp.empty:
        st.info("No suspicious login activity detected.")
    else:
        st.dataframe(
    susp.style.applymap(color_login_status, subset=["login_success"])
)


        if "ip_address" in susp.columns:
            st.subheader("Top Suspicious IP Addresses")
            st.bar_chart(susp["ip_address"].value_counts().head(10))

        st.download_button(
            "Download Suspicious Records",
            susp.to_csv(index=False),
            "suspicious_filtered.csv",
            "text/csv"
        )
