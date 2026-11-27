from pathlib import Path
p=Path('/mnt/data/Pasted text.txt')
s=p.read_text()
# clean imports
start=s.index('import streamlit as st')
end=s.index('# ==========================================\n# 1. CONFIGURATION')
imports='''import streamlit as st
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

'''
s=imports+s[end:]
# Insert PDF helper before PAGE 1
marker='# ==========================================\n# PAGE 1 - OVERVIEW'
helper='''# ==========================================
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


'''
s=s.replace(marker,helper+marker)
# replace PAGE2 region
start=s.index('# PAGE 2 - AI DETECTION')
start=s.rfind('# ==========================================',0,start)
end=s.index('# PAGE 3 - DEFECT ANALYTICS')
end=s.rfind('# ==========================================',0,end)
page2='''# PAGE 2 - AI DETECTION
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


'''
s=s[:start]+page2+s[end:]
# Fix overview history fields
s=s.replace('item["detections"]','item["detections"]')
s=s.replace('item["defects"]','item["defects"]')
# Replace history page references capitalized
s=s.replace('history_df["Defects"]','history_df["defects"]')
s=s.replace('x="Image",\n            y="Detections",\n            color="Defects",','x="image",\n            y="detections",\n            color="defects",')
s=s.replace('x="Date",\n            y="Average Confidence",','x="date",\n            y="average_confidence",')
# table page uses lowercase data, fine.
out=Path('/mnt/data/app.py')
out.write_text(s)
print("written",len(s.splitlines()),"lines")
