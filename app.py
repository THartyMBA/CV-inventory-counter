# inventory_counter.py
"""
Object-Detection Inventory Counter  📦🚗
────────────────────────────────────────────────────────────────────────────
Upload one or more images of inventory (shelves, parking lots, etc.).
This POC:

1. Loads a pretrained YOLOv8n model (ultralytics) cached on startup.  
2. Runs object detection on each uploaded image.  
3. Displays the annotated image with bounding boxes.  
4. Shows a per-class count of detections.  
5. Lets you download the counts as a CSV.

*Demo only*—no batch processing or auth.  
For enterprise-grade computer-vision pipelines, [contact me](https://drtomharty.com/bio).
"""

import io
import os
import streamlit as st
from ultralytics import YOLO
import numpy as np
import pandas as pd
from PIL import Image

# ───────────────────────────────────────────────────────
# Model loading (cached)
# ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    # Uses the small YOLOv8n model for speed on CPU
    return YOLO("yolov8n.pt")

model = load_model()

# ───────────────────────────────────────────────────────
# Streamlit UI
# ───────────────────────────────────────────────────────
st.set_page_config(page_title="Inventory Counter", layout="wide")
st.title("📦 Object-Detection Inventory Counter")

st.info(
    "🔔 **Demo Notice**  \n"
    "This is a proof-of-concept using YOLOv8n on CPU.  \n"
    "For production-grade CV pipelines, [contact me](https://drtomharty.com/bio).",
    icon="💡"
)

uploaded_files = st.file_uploader(
    "Upload one or more images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.stop()

results_list = []
annotated_images = []

for file in uploaded_files:
    # Read image
    img = Image.open(file).convert("RGB")
    img_array = np.array(img)

    # Run detection
    results = model(img_array)

    # Annotate image
    annotated = results[0].plot()  # returns np.ndarray with boxes + labels
    annotated_img = Image.fromarray(annotated)
    annotated_images.append((file.name, annotated_img))

    # Count classes
    counts = {}
    for box in results[0].boxes:
        cls_id = int(box.cls.cpu().numpy())
        cls_name = model.names[cls_id]
        counts[cls_name] = counts.get(cls_name, 0) + 1

    # Prepare for CSV
    for cls_name, count in counts.items():
        results_list.append({
            "image": file.name,
            "class": cls_name,
            "count": count
        })
    if not counts:
        results_list.append({
            "image": file.name,
            "class": "None detected",
            "count": 0
        })

# ───────────────────────────────────────────────────────
# Display results
# ───────────────────────────────────────────────────────
st.subheader("Annotated Images")
for name, img in annotated_images:
    st.image(img, caption=name, use_column_width=True)

st.subheader("Detection Counts")
df_counts = pd.DataFrame(results_list)
st.dataframe(df_counts)

# Download counts as CSV
csv_bytes = df_counts.to_csv(index=False).encode()
st.download_button(
    label="⬇️ Download counts CSV",
    data=csv_bytes,
    file_name="detection_counts.csv",
    mime="text/csv"
)
