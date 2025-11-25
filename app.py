import streamlit as st
import pandas as pd

# ---------------- Page Config ----------------
st.set_page_config(page_title="🧠 Smart Security for Student Logins", layout="wide")
st.title("🧠 Smart Security for Student Logins")

# ---------------- Load Dataset ----------------
df = pd.read_csv("cleaned_logins.csv")
df.columns = df.columns.str.strip()  # Remove extra spaces

# ---------------- Show Actual Columns ----------------
st.subheader("🔍 Columns in CSV")
st.write(df.columns.tolist())

# ---------------- Expected Columns ----------------
expected_columns = [
    "student_id",
    "major",
    "login_attempts",
    "ip_address",
    "login_success",
    "last_login_time",
    "anomaly",
    "threshold_flag"
]

# ---------------- Rename Columns ----------------
rename_dict = {
    "StudentID": "student_id",
    "CourseID": "major",
    "LoginAttempts": "login_attempts",
    "IPAddress": "ip_address",
    "LoginType": "login_success",        # or "LoginMethod" depending on logic
    "PreviousLoginTime": "last_login_time",
    "anomaly": "anomaly",
    "threshold_flag": "threshold_flag"
}

df = df.rename(columns=rename_dict)

# ---------------- Keep Existing Columns ----------------
columns_existing = [col for col in expected_columns if col in df.columns]
missing_columns = [col for col in expected_columns if col not in df.columns]

if missing_columns:
    st.warning(f"The following expected columns are missing: {missing_columns}")

if not columns_existing:
    st.error("No expected columns found. Please check your CSV file.")
else:
    df = df[columns_existing]

# ---------------- Display Unified Table ----------------
st.subheader("📊 Student Logins Data")
st.dataframe(df)

# ---------------- Quick Summary ----------------
st.subheader("📈 Quick Summary")
st.write("Number of rows:", len(df))

if "login_attempts" in df.columns:
    st.write("Max login attempts:", df["login_attempts"].max())
if "anomaly" in df.columns:
    st.write("Number of anomalies detected:", df["anomaly"].sum())
if "threshold_flag" in df.columns:
    st.write("Number of threshold flags:", df["threshold_flag"].sum())
