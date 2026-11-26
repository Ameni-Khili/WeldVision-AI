
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import cv2
import os
import glob
from PIL import Image
from datetime import datetime
from ultralytics import YOLO

# ==========================================
# 1. CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="WeldVision AI",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "/content/runs/segment/train-5/weights/best.pt"
RUN_DIR = "/content/runs/segment/train-5"

CLASS_NAMES = [
    "Bad Welding",
    "Crack",
    "Good Welding",
    "Porosity",
    "Reinforcement",
    "Spatters"
]

# ==========================================
# 2. CUSTOM DESIGN
# ==========================================

st.markdown("""
<style>
.stApp {
    background-color: #101827;
    color: #F1F5F9;
}
[data-testid="stSidebar"] {
    background-color: #172235;
}
h1, h2, h3 {
    color: #F8FAFC;
}
p, label {
    color: #CBD5E1;
}
div[data-testid="stMetric"] {
    background-color: #1E2B40;
    border: 1px solid #34445C;
    padding: 18px;
    border-radius: 12px;
}
div[data-testid="stMetricValue"] {
    color: #38BDF8;
}
.stButton button {
    background-color: #0284C7;
    color: white;
    border-radius: 8px;
    border: none;
    font-weight: bold;
}
hr {
    border-color: #334155;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return YOLO(MODEL_PATH)

model = load_model()

# ==========================================
# 4. SESSION STATE
# ==========================================

if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ==========================================
# 5. SIDEBAR NAVIGATION
# ==========================================

st.sidebar.markdown("# 🏭 WeldVision AI")
st.sidebar.caption("Intelligent Welding Inspection")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "🏠 Overview",
        "🔍 AI Detection",
        "📊 Defect Analytics",
        "🧠 Model Performance",
        "📁 Inspection History"
    ]
)

st.sidebar.divider()
st.sidebar.markdown("### ⚙️ Model Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    0.05, 0.95, 0.25, 0.05
)

st.sidebar.caption("YOLOv8n-Seg")
st.sidebar.caption("6 Welding Classes")

if model is None:
    st.error("Model not found. Please check MODEL_PATH.")
    st.stop()

# ==========================================
# 6. HEADER
# ==========================================

st.markdown(
    "<h1>WeldVision AI</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "AI-Based Welding Defect Detection & Quality Inspection"
)

st.divider()

# ==========================================
# PAGE 1 - OVERVIEW
# ==========================================

if page == "🏠 Overview":

    st.subheader("🏠 Dashboard Overview")

    total_inspections = len(st.session_state.history)

    total_detections = sum(
        item["detections"]
        for item in st.session_state.history
    )

    defect_inspections = sum(
        1 for item in st.session_state.history
        if item["defects"] > 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Inspections",
        total_inspections
    )

    col2.metric(
        "Detected Instances",
        total_detections
    )

    col3.metric(
        "Inspections with Defects",
        defect_inspections
    )

    col4.metric(
        "Welding Classes",
        6
    )

    st.divider()

    st.subheader("System Information")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 🤖 AI Model")
        st.write("Model: YOLOv8n-Seg")
        st.write("Task: Instance Segmentation")
        st.write("Framework: Ultralytics")
        st.write("Model Status: Loaded")

    with c2:
        st.markdown("### 🔧 Inspection Classes")

        for name in CLASS_NAMES:
            st.markdown(f"• {name}")

    st.divider()

    st.subheader("Recent Inspections")

    if st.session_state.history:
        history_df = pd.DataFrame(
            st.session_state.history
        )
        st.dataframe(
            history_df.tail(10),
            use_container_width=True
        )
    else:
        st.info("No inspections yet. Go to AI Detection to start.")

# ==========================================
# PAGE 2 - AI DETECTION
# ==========================================

elif page == "🔍 AI Detection":

    st.subheader("🔍 AI Welding Defect Detection")

    st.write(
        "Upload a welding image to detect and segment welding defects."
    )

    uploaded_file = st.file_uploader(
        "Upload Welding Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")
        image_array = np.array(image)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Original Image")
            st.image(
                image_array,
                use_container_width=True
            )

        if st.button(
            "🚀 Analyze Image",
            type="primary",
            use_container_width=True
        ):

            with st.spinner("AI is analyzing the image..."):

                results = model.predict(
                    source=image_array,
                    conf=confidence,
                    verbose=False
                )

                result = results[0]

                annotated = result.plot()
                annotated = cv2.cvtColor(
                    annotated,
                    cv2.COLOR_BGR2RGB
                )

                boxes = result.boxes

                if boxes is not None and len(boxes) > 0:

                    class_ids = (
                        boxes.cls.cpu().numpy().astype(int)
                    )

                    scores = boxes.conf.cpu().numpy()

                    names = [
                        result.names[int(i)]
                        for i in class_ids
                    ]

                    counts = pd.Series(names).value_counts()

                    avg_conf = float(scores.mean())

                    # Defect classes exclude Good Welding
                    defect_count = sum(
                        count for name, count in counts.items()
                        if name.lower() != "good welding"
                    )

                    st.session_state.last_result = {
                        "image": annotated,
                        "names": names,
                        "scores": scores.tolist(),
                        "counts": counts.to_dict(),
                        "detections": len(class_ids),
                        "defects": defect_count,
                        "average_confidence": avg_conf
                    }

                    st.session_state.history.append({
                        "Date": datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "Image": uploaded_file.name,
                        "Detections": len(class_ids),
                        "Defects": defect_count,
                        "Average Confidence": round(
                            avg_conf * 100, 2
                        )
                    })

                else:

                    st.session_state.last_result = {
                        "image": annotated,
                        "names": [],
                        "scores": [],
                        "counts": {},
                        "detections": 0,
                        "defects": 0,
                        "average_confidence": 0
                    }

                    st.session_state.history.append({
                        "Date": datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "Image": uploaded_file.name,
                        "Detections": 0,
                        "Defects": 0,
                        "Average Confidence": 0
                    })

        # Display latest result
        data = st.session_state.last_result

        if data is not None:

            with col2:
                st.markdown("### Segmentation Result")
                st.image(
                    data["image"],
                    use_container_width=True
                )

            st.divider()
            st.markdown("### 📊 Inspection Results")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Total Detections",
                data["detections"]
            )

            c2.metric(
                "Detected Defects",
                data["defects"]
            )

            c3.metric(
                "Average Confidence",
                f'{data["average_confidence"] * 100:.2f}%'
            )

            if data["detections"] > 0:

                st.markdown("### Defect Distribution")

                df = pd.DataFrame(
                    list(data["counts"].items()),
                    columns=["Class", "Count"]
                )

                fig = px.bar(
                    df,
                    x="Class",
                    y="Count",
                    color="Class",
                    text="Count",
                    title="Detected Welding Classes"
                )

                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#101827",
                    plot_bgcolor="#101827"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                details = pd.DataFrame({
                    "Class": data["names"],
                    "Confidence (%)": [
                        round(s * 100, 2)
                        for s in data["scores"]
                    ]
                })

                st.markdown("### Detection Details")

                st.dataframe(
                    details,
                    use_container_width=True
                )

                csv = details.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "⬇️ Download Detection Results CSV",
                    csv,
                    "welding_detection.csv",
                    "text/csv"
                )

            else:
                st.success(
                    "No objects detected at the selected confidence threshold."
                )

# ==========================================
# PAGE 3 - DEFECT ANALYTICS
# ==========================================

elif page == "📊 Defect Analytics":

    st.subheader("📊 Welding Defect Analytics")

    if not st.session_state.history:
        st.info(
            "Run some image inspections first to generate analytics."
        )

    else:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.metric(
            "Total Inspected Images",
            len(history_df)
        )

        st.metric(
            "Total Detected Defects",
            int(history_df["Defects"].sum())
        )

        fig = px.bar(
            history_df,
            x="Image",
            y="Detections",
            color="Defects",
            title="Detections by Image"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#101827",
            plot_bgcolor="#101827"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        fig2 = px.line(
            history_df,
            x="Date",
            y="Average Confidence",
            markers=True,
            title="Average Confidence by Inspection"
        )

        fig2.update_layout(
            template="plotly_dark",
            paper_bgcolor="#101827",
            plot_bgcolor="#101827"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

# ==========================================
# PAGE 4 - MODEL PERFORMANCE
# ==========================================

elif page == "🧠 Model Performance":

    st.subheader("🧠 Model Performance")

    st.write("YOLOv8n-Seg Training Evaluation")

    results_file = os.path.join(
        RUN_DIR,
        "results.csv"
    )

    if os.path.exists(results_file):

        df = pd.read_csv(results_file)
        df.columns = df.columns.str.strip()

        last = df.iloc[-1]

        def find_metric(keyword):
            matches = [
                c for c in df.columns
                if keyword.lower() in c.lower()
            ]
            return matches[0] if matches else None

        metrics = [
            (
                "Box Precision",
                find_metric("metrics/precision(B)")
            ),
            (
                "Box Recall",
                find_metric("metrics/recall(B)")
            ),
            (
                "Box mAP50",
                find_metric("metrics/mAP50(B)")
            ),
            (
                "Box mAP50-95",
                find_metric("metrics/mAP50-95(B)")
            ),
            (
                "Mask Precision",
                find_metric("metrics/precision(M)")
            ),
            (
                "Mask Recall",
                find_metric("metrics/recall(M)")
            ),
            (
                "Mask mAP50",
                find_metric("metrics/mAP50(M)")
            ),
            (
                "Mask mAP50-95",
                find_metric("metrics/mAP50-95(M)")
            )
        ]

        cols = st.columns(4)

        for i, (label, column) in enumerate(metrics):

            if column is not None:
                value = float(last[column])

                cols[i % 4].metric(
                    label,
                    f"{value * 100:.2f}%"
                )

        st.divider()

        st.markdown("### Training Evolution")

        epoch_column = find_metric("epoch")

        if epoch_column is not None:

            metric_columns = [
                c for c in df.columns
                if "metrics/" in c
            ]

            selected = st.selectbox(
                "Select Metric",
                metric_columns
            )

            fig = px.line(
                df,
                x=epoch_column,
                y=selected,
                markers=True,
                title=selected
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#101827",
                plot_bgcolor="#101827"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.markdown("### Training Curves")

        plot_files = [
            "results.png",
            "confusion_matrix.png",
            "MaskPR_curve.png",
            "BoxPR_curve.png",
            "MaskF1_curve.png",
            "BoxF1_curve.png"
        ]

        for file in plot_files:

            path = os.path.join(
                RUN_DIR,
                file
            )

            if os.path.exists(path):
                st.markdown(f"**{file}**")
                st.image(
                    path,
                    use_container_width=True
                )

    else:
        st.warning(
            "Training results.csv not found in the selected training folder."
        )

# ==========================================
# PAGE 5 - INSPECTION HISTORY
# ==========================================

elif page == "📁 Inspection History":

    st.subheader("📁 Inspection History")

    if st.session_state.history:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True
        )

        csv = history_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Export History CSV",
            csv,
            "inspection_history.csv",
            "text/csv"
        )

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.session_state.last_result = None
            st.rerun()

    else:
        st.info("No inspection history available yet.")

# ==========================================
# FOOTER
# ==========================================

st.divider()

st.markdown(
    "<center>WeldVision AI | YOLOv8-Seg | "
    "Intelligent Industrial Quality Inspection</center>",
    unsafe_allow_html=True
)


