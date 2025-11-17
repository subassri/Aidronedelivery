# Aidronedelivery
Aero AI - Drone Delivery System with ETA Prediction, Drone Detection, and Multi-Agent Chatbot
Aidrone is an end-to-end Streamlit application combining:

✔ 1. Drone Delivery ETA Prediction

Predict delivery time using a trained machine-learning model.

✔ 2. Drone Object Detection (YOLO / Custom Model)

Upload an image → detect drones → display annotated results.

✔ 3. AI Chatbot Assistant

Ask any queries related to drone operations or troubleshooting.

📦 Project Modules
🔢 1. ETA Prediction

Uses ML model (eta_model.pt or .h5)

Uses drone.csv for label encoders

Uses drone_geo.csv for source/destination coordinates

Displays ETA metrics + interactive geo map

🎯 2. Drone Detection

Loads your detection model (YOLOv7, YOLOv8, Faster-RCNN, etc.)

Detects drones in uploaded images

Displays bounding boxes and detection confidence

🤖 3. Chatbot Assistant

Offline / lightweight LLM or rule-based support

Helps with drone troubleshooting, ETA logic, system usage

No external API calls

No keys required

📁 Folder Structure
Aidrone/
│
├── cloud.env                   # All path configs (models, CSVs)
├── requirements.txt
├── README.md
│
├── app/
│   ├── ui.py                   # Main Streamlit app (3 pages)
│   ├── eta_module.py           # ETA logic
│   ├── detect_module.py        # Drone detection logic
│   ├── chatbot_module.py       # Chatbot logic
│   └── utils.py
│
├── models/
│   ├── eta_model.pt
│   ├── scaler.pkl
│   ├── detection_model.pt
│   └── tokenizer.pkl           # (optional for chatbot)
│
└── data/
    ├── drone.csv
    ├── drone_geo.csv
    └── labels.txt              # (optional YOLO labels)
⚙️ cloud.env

Your environment file (required!) should look like this:
ETA_MODEL_PATH=models/eta_model.pt
SCALER_PATH=models/scaler.pkl
DRONE_CSV=data/drone.csv
DRONE_GEO_CSV=data/drone_geo.csv

DETECTION_MODEL_PATH=models/detection_model.pt
DETECTION_LABELS=data/labels.txt

CHATBOT_MODEL_PATH=models/chatbot/
CHATBOT_TOKENIZER=models/tokenizer.pkl

DEBUG_MODE=true

🛠 Installation
pip install -r requirements.txt

▶ Run the Application
streamlit run app/ui.py

📌 Page Details
🧭 1. ETA Prediction Page

User selects drone attributes

Encoders generated from drone.csv

Passes through scaler + ML model

Predicts delivery ETA

Looks up matching drone route in drone_geo.csv

Renders a smooth, flicker-free Folium map

Outputs:

Estimated Time

Distance

Speed

Route Map (Source → Destination)

🛰 2. Drone Detection Page

Upload image/video

Model detects drones

Annotated output displayed

Confidence thresholds adjustable

💬 3. Chatbot Page

Lightweight offline chatbot

Answers:

Drone troubleshooting

ETA explanations

Climate/speed factors

Error messages

UI help

Supports conversational memory per session
📦 Requirements
streamlit
pandas
python-dotenv
folium
streamlit-folium
opencv-python
torch
tensorflow
scikit-learn
numpy
Pillow

🧪 Testing Checklist
ETA Prediction

✔ Model loads
✔ Encoders loaded
✔ GEO map appears without flicker
✔ Missing-lat/lon rows skipped safely

Drone Detection

✔ Model loads
✔ Image processing works
✔ Bounding boxes drawn correctly

Chatbot

✔ Loads local model
✔ Provides basic assistance


