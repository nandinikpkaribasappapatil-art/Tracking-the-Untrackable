import os
import time
import pandas as pd
import streamlit as st
from engine import process_video

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="ARGUS-X | Intelligent Tracking",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM DASHBOARD DESIGN
# --------------------------------------------------

st.markdown("""
<style>
.stApp {
    background-color: #101827;
    color: #F1F5F9;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}
.hero {
    background: linear-gradient(120deg, #172B46, #263D60);
    padding: 28px;
    border-radius: 16px;
    border: 1px solid #385273;
    margin-bottom: 20px;
}
.hero h1 {
    color: white;
    font-size: 42px;
    margin-bottom: 5px;
}
.hero p {
    color: #C8D8EA;
    font-size: 16px;
}
.metric-card {
    background: #1C293B;
    border: 1px solid #34465D;
    padding: 20px;
    border-radius: 13px;
    min-height: 115px;
}
.metric-label {
    color: #AFC0D4;
    font-size: 14px;
}
.metric-value {
    color: #FFFFFF;
    font-size: 30px;
    font-weight: bold;
    margin-top: 8px;
}
.section-heading {
    color: #EAF2FF;
    font-size: 23px;
    font-weight: 700;
    margin-top: 25px;
    margin-bottom: 12px;
}
.flow-box {
    background: #1C293B;
    border: 1px solid #45617F;
    border-radius: 10px;
    padding: 14px 8px;
    text-align: center;
    color: white;
    font-weight: 600;
}
.flow-arrow {
    color: #64B5F6;
    font-size: 24px;
    text-align: center;
    padding-top: 8px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <h1>👁️ ARGUS-X</h1>
    <h3 style="color:#D9E8F8;">Track the Untrackable</h3>
    <p>
    Long-Horizon Multi-Object Identity Recovery
    Under Camera, Occlusion & Appearance Drift
    </p>
</div>
""", unsafe_allow_html=True)

st.caption(
    "AI-powered person detection, multi-object tracking "
    "and appearance-based global identity memory."
)

st.warning(
    "Research/demo prototype only. The current identity descriptor "
    "uses colour-based appearance features, not a trained person-ReID "
    "model. Similar outfits, lighting changes and camera changes can "
    "cause errors. Do not use for real-world security decisions."
)

# --------------------------------------------------
# SIDEBAR SETTINGS
# --------------------------------------------------

with st.sidebar:
    st.header("⚙️ Processing Settings")

    conf = st.slider(
        "Detection confidence",
        0.10, 0.80, 0.30, 0.05
    )

    threshold = st.slider(
        "Identity match threshold",
        0.55, 0.95, 0.76, 0.01
    )

    memory = st.slider(
        "Memory duration (seconds)",
        5, 120, 30, 5
    )

    stride = st.select_slider(
        "Frame stride (higher = faster, less detail)",
        options=[1, 2, 3],
        value=1
    )

    max_width = st.select_slider(
        "Maximum video width",
        options=[640, 800, 960],
        value=800
    )

    st.divider()

    st.subheader("About ARGUS-X")
    st.write("Version: Research Demo")
    st.write("Detection: YOLO11n")
    st.write("Tracking: BoT-SORT")
    st.write("Identity: Global Memory")

# --------------------------------------------------
# SYSTEM ARCHITECTURE
# --------------------------------------------------

with st.expander("🧠 View System Architecture", expanded=False):

    st.markdown("### How ARGUS-X works")

    stages = [
        ("📹", "Video Input"),
        ("🎯", "YOLO Detection"),
        ("🔄", "BoT-SORT Tracking"),
        ("🧠", "Global Identity Memory"),
        ("📊", "Results & Analytics")
    ]

    columns = st.columns(len(stages))

    for col, stage in zip(columns, stages):
        with col:
            st.markdown(
                f"""
                <div class="flow-box">
                    <div style="font-size:25px">{stage[0]}</div>
                    <div>{stage[1]}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("""
    **Processing flow:**

    1. Upload a CCTV/video clip.
    2. Detect people using YOLO11n.
    3. Track people using BoT-SORT.
    4. Compare appearance descriptors with stored identities.
    5. Generate global IDs and tracking records.
    """)

# --------------------------------------------------
# VIDEO UPLOAD
# --------------------------------------------------

st.markdown(
    '<div class="section-heading">📤 Upload Video</div>',
    unsafe_allow_html=True
)

video = st.file_uploader(
    "Upload a permitted CCTV/video clip",
    type=["mp4", "avi", "mov", "mkv"]
)

if video is not None:
    st.success(f"Selected video: {video.name}")

# --------------------------------------------------
# PROCESS VIDEO
# --------------------------------------------------

if video is not None:

    if st.button(
        "🚀 Run ARGUS-X",
        type="primary",
        use_container_width=True
    ):

        os.makedirs("output", exist_ok=True)

        input_path = os.path.join(
            "output",
            "uploaded_video" + os.path.splitext(video.name)[1]
        )

        with open(input_path, "wb") as f:
            f.write(video.getbuffer())

        status = st.empty()

        try:
            status.info(
                "⏳ Processing video on CPU. "
                "Please wait..."
            )

            start_time = time.perf_counter()

            result = process_video(
                input_path,
                "output/argus_output.mp4",
                "output/tracks.csv",
                conf=conf,
                imgsz=640,
                max_width=max_width,
                memory_seconds=memory,
                reid_threshold=threshold,
                frame_stride=stride
            )

            processing_time = time.perf_counter() - start_time

            # Save results so they remain visible
            # when dashboard filters are changed.

            st.session_state["argus_result"] = result
            st.session_state["argus_time"] = processing_time
            st.session_state["argus_filename"] = video.name

            status.success("✅ Processing complete!")

        except Exception as e:
            status.error(f"Processing failed: {e}")
            st.info(
                "Try a shorter MP4, lower resolution, "
                "frame stride 2, and close other applications."
            )

# --------------------------------------------------
# DISPLAY SAVED RESULTS
# --------------------------------------------------

if "argus_result" in st.session_state:

    result = st.session_state["argus_result"]
    processing_time = st.session_state["argus_time"]

    df = pd.read_csv(result["csv"])

    # --------------------------------------------------
    # SUMMARY CARDS
    # --------------------------------------------------

    st.markdown(
        '<div class="section-heading">📊 Processing Overview</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    metrics = [
        (c1, "🎞️ Frames Processed", result["frames_processed"]),
        (c2, "🎯 Total Detections", result["detections"]),
        (c3, "👥 Unique Global IDs", result["unique_global_ids"]),
        (c4, "⏱️ Processing Time", f"{processing_time:.2f} sec")
    ]

    for col, label, value in metrics:
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # --------------------------------------------------
    # VIDEO PREVIEW
    # --------------------------------------------------

    st.markdown(
        '<div class="section-heading">🎬 Annotated Video Preview</div>',
        unsafe_allow_html=True
    )

    if os.path.exists(result["video"]):

        st.video(result["video"])

        with open(result["video"], "rb") as f:
            video_bytes = f.read()

        st.download_button(
            "⬇️ Download Annotated MP4",
            data=video_bytes,
            file_name="argus_output.mp4",
            mime="video/mp4",
            use_container_width=True
        )

    # --------------------------------------------------
    # IDENTITY EVENT ANALYSIS
    # --------------------------------------------------

    st.markdown(
        '<div class="section-heading">🧠 Identity Event Analysis</div>',
        unsafe_allow_html=True
    )

    if not df.empty and "identity_event" in df.columns:

        event_counts = (
            df["identity_event"]
            .value_counts()
            .reindex(
                ["NEW ID", "TRACKING", "RE-IDENTIFIED"],
                fill_value=0
            )
        )

        e1, e2, e3 = st.columns(3)

        e1.metric("🆕 New IDs", int(event_counts["NEW ID"]))
        e2.metric("🔄 Tracking Events", int(event_counts["TRACKING"]))
        e3.metric(
            "♻️ Re-identification Events",
            int(event_counts["RE-IDENTIFIED"])
        )

        chart_df = event_counts.rename_axis(
            "Identity Event"
        ).reset_index(name="Count")

        st.bar_chart(
            chart_df.set_index("Identity Event"),
            y="Count",
            use_container_width=True
        )

        st.caption(
            "These are event-record counts, not counts of "
            "verified real-world people."
        )

    # --------------------------------------------------
    # GLOBAL ID ANALYSIS
    # --------------------------------------------------

    st.markdown(
        '<div class="section-heading">👥 Global Identity Analysis</div>',
        unsafe_allow_html=True
    )

    if not df.empty and "global_id" in df.columns:

        identity_counts = (
            df.groupby("global_id")
            .size()
            .rename("Detection Records")
            .sort_values(ascending=False)
        )

        st.bar_chart(identity_counts)

        st.caption(
            "The chart shows the number of detection records "
            "associated with each global ID."
        )

    # --------------------------------------------------
    # TRACKING RECORDS
    # --------------------------------------------------

    st.markdown(
        '<div class="section-heading">📋 Tracking Records</div>',
        unsafe_allow_html=True
    )

    if not df.empty:

        f1, f2, f3 = st.columns(3)

        with f1:
            selected_id = st.selectbox(
                "Filter by Global ID",
                ["All"] + sorted(
                    df["global_id"].dropna().unique().tolist()
                )
            )

        with f2:
            event_options = ["All"] + sorted(
                df["identity_event"].dropna().unique().tolist()
            )

            selected_event = st.selectbox(
                "Filter by Identity Event",
                event_options
            )

        with f3:
            search_text = st.text_input(
                "Search records",
                placeholder="Enter tracker ID or value"
            )

        filtered_df = df.copy()

        if selected_id != "All":
            filtered_df = filtered_df[
                filtered_df["global_id"] == selected_id
            ]

        if selected_event != "All":
            filtered_df = filtered_df[
                filtered_df["identity_event"] == selected_event
            ]

        if search_text:
            matching_rows = filtered_df.astype(str).apply(
                lambda column: column.str.contains(
                    search_text,
                    case=False,
                    regex=False
                )
            ).any(axis=1)

            filtered_df = filtered_df[matching_rows]

        st.caption(
            f"Showing {len(filtered_df)} of {len(df)} records"
        )

        st.dataframe(
            filtered_df.tail(500),
            use_container_width=True,
            hide_index=True
        )

        if len(filtered_df) > 500:
            st.info(
                "Showing the latest 500 matching records. "
                "Download the CSV for the complete dataset."
            )

    else:
        st.info("No tracking records were generated.")

    # --------------------------------------------------
    # DOWNLOAD SECTION
    # --------------------------------------------------

    st.markdown(
        '<div class="section-heading">📥 Download Reports</div>',
        unsafe_allow_html=True
    )

    d1, d2 = st.columns(2)

    with d1:
        with open(result["csv"], "rb") as f:
            csv_bytes = f.read()

        st.download_button(
            "📄 Download Tracking CSV",
            data=csv_bytes,
            file_name="argus_tracks.csv",
            mime="text/csv",
            use_container_width=True
        )

    with d2:
        if os.path.exists(result["video"]):
            with open(result["video"], "rb") as f:
                mp4_bytes = f.read()

            st.download_button(
                "🎬 Download Annotated Video",
                data=mp4_bytes,
                file_name="argus_output.mp4",
                mime="video/mp4",
                use_container_width=True
            )

    # --------------------------------------------------
    # FOOTER
    # --------------------------------------------------

    st.divider()

    st.caption(
        "ARGUS-X | Intelligent Multi-Object Tracking | "
        "Research & Demonstration Prototype"
    )

else:

    # --------------------------------------------------
    # INITIAL DASHBOARD MESSAGE
    # --------------------------------------------------

    st.info(
        "👋 Welcome to ARGUS-X. Upload a permitted video "
        "and click Run ARGUS-X to begin processing."
    )

    st.markdown("### 🎯 What ARGUS-X can do")

    a, b, c = st.columns(3)

    a.info("**Person Detection**\n\nDetect people in video frames.")

    b.info("**Multi-Object Tracking**\n\nMaintain temporary tracking IDs.")

    c.info("**Identity Recovery**\n\nAttempt to reconnect new tracks with stored identities.")

    st.caption(
        "Recommended first test: a 15–30 second 720p video "
        "with 2–5 people walking and briefly occluding one another."
    )