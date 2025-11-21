import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="NCS Job Application Agent Dashboard", page_icon="🧠", layout="wide")
st.title("🧠 NCS Job Application Agent Dashboard (v2)")

CSV_FILE = "ncs_job_results.csv"

# 🧩 Check file existence
if not os.path.exists(CSV_FILE):
    st.warning("⚠️ No ncs_job_results.csv found. Run app_ncs_v2.py first.")
    st.stop()

# 📂 Load CSV
df = pd.read_csv(CSV_FILE)

# 🧹 Clean columns (normalize)
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
expected_cols = [
    "no", "title", "company", "url", "similarity", "apply_decision",
    "last_date", "short_desc"
]
for col in expected_cols:
    if col not in df.columns:
        df[col] = ""

# 🧮 Summary stats
total_jobs = len(df)
applied = len(df[df["apply_decision"].astype(str).str.lower() == "true"])
skipped = total_jobs - applied

col1, col2, col3 = st.columns(3)
col1.metric("Total Jobs Processed", total_jobs)
col2.metric("Applied", applied)
col3.metric("Skipped", skipped)

# 🎚️ Filters
st.sidebar.header("🔍 Filters")
min_similarity = st.sidebar.slider("Minimum Similarity", 0.0, 1.0, 0.0, 0.01)
decision_filter = st.sidebar.selectbox("Filter by Apply Decision", ["All", "Applied", "Skipped"])

# Apply filters
filtered_df = df[df["similarity"] >= min_similarity]

if decision_filter == "Applied":
    filtered_df = filtered_df[df["apply_decision"].astype(str).str.lower() == "true"]
elif decision_filter == "Skipped":
    filtered_df = filtered_df[df["apply_decision"].astype(str).str.lower() == "false"]

# 📈 Charts
st.subheader("📈 Similarity Distribution")
if not filtered_df.empty:
    st.bar_chart(filtered_df.set_index("company")["similarity"])
else:
    st.info("No data available for selected filters.")

# 🔥 Top 5 Matches
st.subheader("🔥 Top 5 Job Matches by Similarity")
if not filtered_df.empty:
    top_df = filtered_df.sort_values(by="similarity", ascending=False).head(5)
    st.dataframe(top_df[["title", "company", "similarity", "apply_decision", "url"]], use_container_width=True)
else:
    st.info("No top matches to show.")

# 🧭 Explore All Jobs
st.subheader("🧭 Explore All Jobs")
if not filtered_df.empty:
    st.dataframe(
        filtered_df[["no", "title", "company", "similarity", "apply_decision", "last_date", "short_desc", "url"]],
        use_container_width=True
    )
else:
    st.warning("No jobs to display for the selected filters.")

# 📬 Job Detail Viewer
st.subheader("📬 Job Detail Viewer")
if not filtered_df.empty:
    job_titles = filtered_df["title"].fillna("Untitled").tolist()
    selected_title = st.selectbox("Select a job to view details", job_titles)
    selected = filtered_df[filtered_df["title"] == selected_title].iloc[0]

    st.markdown(f"**🏢 Company:** {selected['company']}")
    st.markdown(f"**🔗 URL:** [{selected['url']}]({selected['url']})")
    st.markdown(f"**📈 Similarity:** {selected['similarity']}")
    st.markdown(f"**📄 Apply Decision:** {selected['apply_decision']}")
    st.markdown(f"**📅 Last Date:** {selected['last_date']}")
    st.markdown(f"**🧾 Description:** {selected['short_desc']}")
else:
    st.info("Select filters that show at least one job.")
