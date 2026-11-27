import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import cv2
import os
from PIL import Image
from datetime import datetime
from ultralytics import YOLO
from io import BytesIO

# PDF report generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage
)

# ==========================================
# 1. CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="WeldVision AI",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "best.pt"
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
# PDF REPORT GENERATOR
# ==========================================

def generate_pdf_report(data):
    """Create a downloadable PDF inspection report from the latest YOLO result."""
    pdf_buffer = BytesIO()

    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "WeldVisionTitle",
        parent=styles["Title"],
        textColor=colors.HexColor("#12345A"),
        alignment=TA_CENTER,
        fontSize=22,
        spaceAfter=8
    )

    story = [
        Paragraph("WeldVision AI", title_style),
        Paragraph(
            "Intelligent Welding Inspection Report",
            styles["Heading2"]
        ),
        Spacer(1, 16)
    ]

    # Inspection information
    story.append(Paragraph("1. Inspection Information", styles["Heading2"]))
    info_rows = [
        ["Inspection Date", data.get("inspection_date", "Not available")],
        ["Image Name", data.get("filename", "Not available")],
        ["AI Model", "YOLOv8n-Seg"],
        ["Total Detections", str(data.get("detections", 0))],
        ["Detected Defects", str(data.get("defects", 0))],
        [
            "Average Confidence",
            f'{data.get("average_confidence", 0) * 100:.2f}%'
        ]
    ]

    info_table = Table(info_rows, colWidths=[170, 310])
    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#DCE6F1")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
    ]))
    story.extend([info_table, Spacer(1, 18)])

    # Original image
    story.append(Paragraph("2. Original Welding Image", styles["Heading2"]))
    original_image = data.get("original_image")
    if original_image is not None:
        original_buffer = BytesIO()
        Image.fromarray(original_image).save(original_buffer, format="PNG")
        original_buffer.seek(0)
        story.append(RLImage(
            original_buffer, width=450, height=270, kind="proportional"
        ))
        story.append(Spacer(1, 15))

    # Segmentation result
    story.append(Paragraph("3. AI Segmentation Result", styles["Heading2"]))
    annotated_image = data.get("image")
    if annotated_image is not None:
        annotated_buffer = BytesIO()
        Image.fromarray(annotated_image).save(annotated_buffer, format="PNG")
        annotated_buffer.seek(0)
        story.append(RLImage(
            annotated_buffer, width=450, height=270, kind="proportional"
        ))
        story.append(Spacer(1, 18))

    # Detection details
    story.append(Paragraph("4. Detection Details", styles["Heading2"]))
    table_rows = [["Welding Class", "Confidence (%)"]]
    for name, score in zip(data.get("names", []), data.get("scores", [])):
        table_rows.append([str(name), f"{float(score) * 100:.2f}%"])

    if not data.get("names"):
        table_rows.append(["No objects detected", "-"])

    detection_table = Table(table_rows, colWidths=[280, 200])
    detection_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12345A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
    ]))
    story.extend([detection_table, Spacer(1, 18)])

    # Conclusion
    story.append(Paragraph("5. Inspection Conclusion", styles["Heading2"]))
    defects = int(data.get("defects", 0))
    detections = int(data.get("detections", 0))

    if defects > 0:
        conclusion = (
            f"{defects} welding defect instance(s) were detected. "
            "Further quality inspection is recommended."
        )
    elif detections > 0:
        conclusion = (
            "Only Good Welding detections were found at the selected "
            "confidence threshold. This AI result should still be checked "
            "by a qualified inspector."
        )
    else:
        conclusion = (
            "No objects were detected at the selected confidence threshold. "
            "This does not prove that the weld is defect-free."
        )

    story.append(Paragraph(conclusion, styles["BodyText"]))
    story.append(Spacer(1, 20))
    story.append(Paragraph(
        "Generated automatically by WeldVision AI",
        styles["Italic"]
    ))

    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


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

# PAGE 2 - AI DETECTION
# ==========================================

elif page == "🔍 AI Detection":

    st.subheader("🔍 AI Welding Defect Detection")
    st.write(
        "Upload a welding image to detect and segment welding defects."
    )

    uploaded_file = st.file_uploader(
        "Upload Welding Image",
        type=["jpg", "jpeg", "png"],
        key="welding_image_uploader"
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")
        image_array = np.array(image)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Original Image")
            st.image(image_array, use_container_width=True)

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
                inspection_time = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                if boxes is not None and len(boxes) > 0:

                    class_ids = boxes.cls.cpu().numpy().astype(int)
                    scores = boxes.conf.cpu().numpy()

                    names = [
                        result.names[int(class_id)]
                        for class_id in class_ids
                    ]

                    counts = pd.Series(names).value_counts()
                    avg_conf = float(scores.mean())

                    # Good Welding is not counted as a defect.
                    defect_count = sum(
                        int(count)
                        for name, count in counts.items()
                        if name.lower() != "good welding"
                    )

                    st.session_state.last_result = {
                        "image": annotated,
                        "original_image": image_array.copy(),
                        "filename": uploaded_file.name,
                        "inspection_date": inspection_time,
                        "names": names,
                        "scores": scores.tolist(),
                        "counts": counts.to_dict(),
                        "detections": len(class_ids),
                        "defects": defect_count,
                        "average_confidence": avg_conf
                    }

                else:

                    st.session_state.last_result = {
                        "image": annotated,
                        "original_image": image_array.copy(),
                        "filename": uploaded_file.name,
                        "inspection_date": inspection_time,
                        "names": [],
                        "scores": [],
                        "counts": {},
                        "detections": 0,
                        "defects": 0,
                        "average_confidence": 0.0
                    }


                # Store history using consistent lowercase keys.
                st.session_state.history.append({
                    "date": inspection_time,
                    "image": uploaded_file.name,
                    "detections": st.session_state.last_result["detections"],
                    "defects": st.session_state.last_result["defects"],
                    "average_confidence": round(
                        st.session_state.last_result["average_confidence"] * 100,
                        2
                    )
                })

    # Display latest saved result (outside uploaded_file)
    data = st.session_state.get("last_result")

    if data is not None:

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Original Image")
            st.image(
                data["original_image"],
                use_container_width=True
            )

        with col2:
            st.markdown("### Segmentation Result")
            st.image(
                data["image"],
                use_container_width=True
            )

        st.divider()
        st.markdown("### 📊 Inspection Results")

        c1, c2, c3 = st.columns(3)

        c1.metric("Total Detections", data["detections"])
        c2.metric("Detected Defects", data["defects"])
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

            st.plotly_chart(fig, use_container_width=True)

            details = pd.DataFrame({
                "Class": data["names"],
                "Confidence (%)": [
                    round(score * 100, 2)
                    for score in data["scores"]
                ]
            })

            st.markdown("### Detection Details")
            st.dataframe(details, use_container_width=True)

            csv_data = details.to_csv(index=False).encode("utf-8")

            st.download_button(
                "⬇️ Download Detection Results CSV",
                csv_data,
                "welding_detection.csv",
                "text/csv"
            )

        else:
            st.success(
                "No objects detected at the selected confidence threshold."
            )

        st.divider()
        st.markdown("### 📄 Inspection Report")

        pdf_data = generate_pdf_report(data)

        st.download_button(
            label="📥 Download Inspection Report (PDF)",
            data=pdf_data,
            file_name="WeldVision_Inspection_Report.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True,
            key="download_inspection_pdf"
        )

st.plotly_chart(fig, use_container_width=True)

            details = pd.DataFrame({  
                    "Class": data["names"],
                    "Confidence (%)": [
                        round(score * 100, 2)
                        for score in data["scores"]
                    ]
                })

                st.markdown("### Detection Details")
                st.dataframe(details, use_container_width=True)

                csv_data = details.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "⬇️ Download Detection Results CSV",
                    csv_data,
                    "welding_detection.csv",
                    "text/csv"
                )

            else:
                st.success(
                    "No objects detected at the selected confidence threshold."
                )

            # PDF inspection report
            st.divider()
            st.markdown("### 📄 Inspection Report")

            pdf_data = generate_pdf_report(data)

            st.download_button(
                label="📥 Download Inspection Report (PDF)",
                data=pdf_data,
                file_name="WeldVision_Inspection_Report.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
                key="download_inspection_pdf"
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
            int(history_df["defects"].sum())
        )

        fig = px.bar(
            history_df,
            x="image",
            y="detections",
            color="defects",
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
            x="date",
            y="average_confidence",
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

# ==========================================
# PAGE 4 - MODEL PERFORMANCE
# ==========================================

elif page == "🧠 Model Performance":

    st.subheader("🧠 Model Performance")

    st.write("YOLOv8n-Seg Training Evaluation")
    st.success("Displaying saved evaluation results - Train 5")

    # Overall Model Metrics
    st.markdown("### Overall Performance")

    metrics = [
        ("Box Precision", 66.6),
        ("Box Recall", 56.1),
        ("Box mAP50", 61.4),
        ("Box mAP50-95", 34.9),
        ("Mask Precision", 62.5),
        ("Mask Recall", 53.3),
        ("Mask mAP50", 55.3),
        ("Mask mAP50-95", 25.0)
    ]

    cols = st.columns(4)

    for i, (label, value) in enumerate(metrics):
        cols[i % 4].metric(
            label,
            f"{value:.1f}%"
        )

    st.divider()

    # Mask Performance by Class
    st.markdown("### Mask mAP50 by Welding Class")

    class_results = {
        "Welding Class": [
            "Bad Welding",
            "Crack",
            "Good Welding",
            "Porosity",
            "Reinforcement",
            "Spatters"
        ],
        "mAP50 (%)": [
            67.4,
            50.0,
            71.9,
            45.8,
            53.6,
            43.3
        ]
    }

    class_df = pd.DataFrame(class_results)

    fig = px.bar(
        class_df,
        x="Welding Class",
        y="mAP50 (%)",
        color="mAP50 (%)",
        text="mAP50 (%)",
        title="Segmentation Performance by Class",
        range_y=[0, 100]
    )

    fig.update_traces(texttemplate="%{text:.1f}%")

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#101827",
        plot_bgcolor="#101827"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # Model Information
    st.markdown("### Model Information")

    col1, col2, col3 = st.columns(3)

    col1.metric("Model", "YOLOv8n-Seg")
    col2.metric("Number of Classes", "6")
    col3.metric("Training Run", "Train 5")

    st.info(
        "These values are the previously recorded evaluation "
        "results of Train 5. Original training curves are not "
        "available because results.csv is missing."
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


