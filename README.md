# CV-inventory-counter

📦 Object-Detection Inventory Counter
A Streamlit proof-of-concept that uses a pretrained YOLOv8 model to count items in your inventory images—shelves, parking lots, warehouse racks, and more.

Demo only—no batch processing, authentication, or production scaling.
For enterprise-grade computer-vision pipelines, contact me.

🔍 What it does
Upload one or more images (.jpg, .jpeg, .png).

Detect objects in each image using YOLOv8n (small model) on CPU.

Annotate images with bounding boxes and labels.

Count occurrences of each detected class per image.

Display annotated images and a table of per-class counts.

Download the counts as detection_counts.csv.

✨ Key Features
Ultra-lightweight: YOLOv8n model (~27 MB) runs on CPU.

Single-file app: no backend—everything in inventory_counter.py.

Interactive UI: upload, view, and download in one place.

Reusable: swap in custom models by changing the YOLO("...") path.

Zero secrets: no API keys or external services needed.

🚀 Quick Start (Local)
bash
Copy
Edit
git clone https://github.com/THartyMBA/inventory-counter.git
cd inventory-counter
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run inventory_counter.py
Browse to http://localhost:8501.

Upload your images.

View annotated images and download counts.

☁️ Deploy on Streamlit Community Cloud
Push this repo (public or private) under THartyMBA to GitHub.

Go to streamlit.io/cloud → New app → select your repo & branch → Deploy.

Share your live URL.

🛠️ Requirements
txt
Copy
Edit
streamlit>=1.32
ultralytics
opencv-python
pillow
pandas
🗂️ Repo Structure
vbnet
Copy
Edit
inventory-counter/
├─ inventory_counter.py   ← single-file Streamlit app  
├─ requirements.txt  
└─ README.md              ← you’re reading it  
📜 License
CC0 1.0 – public-domain dedication. Attribution appreciated but not required.

🙏 Acknowledgements
Streamlit – rapid web UIs for Python

Ultralytics YOLO – object-detection framework

OpenCV & Pillow – image processing

Pandas – tabular data handling

Count your inventory in seconds—enjoy! 🎉
