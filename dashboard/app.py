import streamlit as st
import pandas as pd
import os
from PIL import Image

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Wildlife Alert System",
    page_icon="🦁",
    layout="wide"
)

# -----------------------------
# Header
# -----------------------------
st.title("🦁 Wildlife Detection & Alert System")
st.write("Real-Time Wildlife Monitoring Dashboard")

# -----------------------------
# Refresh button
# -----------------------------
if st.button("🔄 Refresh Dashboard"):
    st.rerun()

st.divider()

# -----------------------------
# File paths
# -----------------------------
csv_file = "outputs/alerts.csv"
alert_folder = "outputs/alerts"

# -----------------------------
# Check CSV
# -----------------------------
if not os.path.exists(csv_file):
    st.error("Alert log file not found!")
    st.stop()

# -----------------------------
# Read CSV
# -----------------------------
df = pd.read_csv(csv_file)

# -----------------------------
# Statistics
# -----------------------------
total_alerts = len(df)

unique_animals = (
    df["Animal"].nunique()
    if not df.empty
    else 0
)

unique_ids = (
    df["Track_ID"].nunique()
    if not df.empty
    else 0
)

# -----------------------------
# Dashboard metrics
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "🚨 Total Alerts",
        total_alerts
    )

with col2:
    st.metric(
        "🐾 Animal Classes",
        unique_animals
    )

with col3:
    st.metric(
        "🎯 Tracked Objects",
        unique_ids
    )

st.divider()

# -----------------------------
# Latest Alert
# -----------------------------
st.subheader("🚨 Latest Alert")

if not df.empty:

    latest = df.iloc[-1]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Animal",
            latest["Animal"]
        )

    with col2:
        st.metric(
            "Track ID",
            latest["Track_ID"]
        )

    with col3:
        st.metric(
            "Confidence",
            f"{latest['Confidence']:.2f}"
        )

    with col4:
        st.metric(
            "Event",
            latest["Event"]
        )

    st.write(
        f"**Time:** {latest['Time']}"
    )

st.divider()

# -----------------------------
# Latest Alert Screenshot
# -----------------------------
st.subheader("📸 Latest Alert Evidence")

if os.path.exists(alert_folder):

    images = [
        file
        for file in os.listdir(alert_folder)
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    if images:

        # Sort by filename so newest timestamp is last
        images.sort()

        latest_image = images[-1]

        image_path = os.path.join(
            alert_folder,
            latest_image
        )

        image = Image.open(image_path)

        st.image(
            image,
            caption=latest_image,
            use_container_width=True
        )

    else:
        st.info("No alert screenshots available.")

else:
    st.info("Alert folder not found.")

st.divider()

# -----------------------------
# Alert Events
# -----------------------------
st.subheader("📋 Alert Events")

if not df.empty:

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

else:
    st.info("No alerts recorded yet.")

st.divider()

# -----------------------------
# Animal Statistics
# -----------------------------
st.subheader("🐾 Animals Detected")

if not df.empty:

    animal_counts = (
        df["Animal"]
        .value_counts()
        .reset_index()
    )

    animal_counts.columns = [
        "Animal",
        "Alerts"
    ]

    st.bar_chart(
        animal_counts.set_index("Animal")
    )

st.divider()

# -----------------------------
# Recent Alert Screenshots
# -----------------------------
st.subheader("🖼️ Recent Alert Evidence")

if os.path.exists(alert_folder):

    images = [
        file
        for file in os.listdir(alert_folder)
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    images.sort(reverse=True)

    if images:

        cols = st.columns(3)

        for i, image_name in enumerate(images[:6]):

            image_path = os.path.join(
                alert_folder,
                image_name
            )

            with cols[i % 3]:

                image = Image.open(image_path)

                st.image(
                    image,
                    caption=image_name,
                    use_container_width=True
                )

    else:
        st.info("No alert screenshots available.")

# -----------------------------
# Footer
# -----------------------------
st.divider()

st.caption(
    "Deep Learning-Based Real-Time Wildlife "
    "Detection, Tracking, and Alert System"
)