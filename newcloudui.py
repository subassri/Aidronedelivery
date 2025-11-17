# newcloudui.py — Unified Aero AI Streamlit app 
import os
import io
import logging
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import streamlit.components.v1 as components
from sklearn.preprocessing import LabelEncoder
from fpdf import FPDF
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import folium
from streamlit_folium import st_folium

# ------------------- Logging -------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aero_ai")

# ------------------- CONFIG -------------------
st.set_page_config(page_title="Aero AI", layout="wide", initial_sidebar_state="expanded")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# File names (place any model/csv files here)
DRONE_CSV = os.path.join(BASE_DIR, "drone.csv")
DRONE_GEO_CSV = os.path.join(BASE_DIR, "drone_geo.csv")
ETA_MODEL_PATH = os.path.join(BASE_DIR, "best_model.keras")     # put your keras model here
SCALER_PATH = os.path.join(BASE_DIR, "scaler.joblib")           # put your scaler here
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "drone_detection_model.pt")  # put your yolov8/yolov5 weights here
BANNER_PATH = os.path.join(BASE_DIR, "banner.jpg")
AITOP_PATH = os.path.join(BASE_DIR, "aitop.jpg")

# Load environment variables from cloud.env (DO NOT commit real secrets)
ENV_PATH = os.path.join(BASE_DIR, "cloud.env")
if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)
    logger.info("Loaded environment from cloud.env")
else:
    logger.info("cloud.env not found — continuing with env vars / defaults")

# TiDB / other env (placeholders only)
TIDB_HOST = os.getenv("TIDB_HOST")
TIDB_PORT = int(os.getenv("TIDB_PORT") or 0)
TIDB_USER = os.getenv("TIDB_USER")
TIDB_PASS = os.getenv("TIDB_PASS")
TIDB_DB = os.getenv("TIDB_DB")
SSL_CA_PATH = os.getenv("SSL_CA_PATH")  # path to CA PEM if required

# ------------------- TiDB CONNECTION (optional & safe) -------------------
engine = None
if TIDB_HOST and TIDB_USER and TIDB_PASS and TIDB_DB and TIDB_PORT:
    try:
        connect_args = {"ssl": {"ca": SSL_CA_PATH}} if SSL_CA_PATH else {}
        engine = create_engine(
            f"mysql+pymysql://{TIDB_USER}:{TIDB_PASS}@{TIDB_HOST}:{TIDB_PORT}/{TIDB_DB}",
            connect_args=connect_args,
            pool_pre_ping=True,
            future=True,
        )
        # quick smoke test
        with engine.begin() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Connected to TiDB (from env)")
    except Exception as e:
        logger.warning(f"Failed to connect to TiDB: {e}")
        engine = None
else:
    logger.info("TiDB not configured — running in offline mode")

def safe_read(sql, params=None):
    """Read from TiDB if configured. Returns DataFrame or empty DataFrame."""
    if engine is None:
        return pd.DataFrame()
    try:
        return pd.read_sql(sql, con=engine, params=params)
    except Exception as e:
        logger.warning(f"DB read failed: {e}")
        return pd.DataFrame()

# ------------------- CSS / Sidebar -------------------
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #D4AF37 0%, #FFD700 100%);
    color: #000000;
    padding-top: 20px;
    border-radius: 0 15px 15px 0;
}
[data-testid="stSidebar"] h2 { color: #000000; text-align: center; font-weight: 700; font-size: 22px; }
[data-testid="stSidebar"] img { display:block; margin: 8px auto; border-radius: 8px; }
.chat-bubble-user {background:#DCF8C6; padding:10px 14px; margin:6px; border-radius:12px; max-width:72%; float:right;}
.chat-bubble-bot {background:#F1F0F0; padding:10px 14px; margin:6px; border-radius:12px; max-width:72%; float:left;}
.clearfix {clear:both;}
</style>
""", unsafe_allow_html=True)

# ------------------- Load resources -------------------
@st.cache_resource
def load_drone_df_and_encoders(csv_path=DRONE_CSV):
    """Load drone CSV and fit LabelEncoders for categorical fields (if present)."""
    if not os.path.exists(csv_path):
        logger.warning(f"{csv_path} not found")
        return None, {}
    df = pd.read_csv(csv_path)
    encs = {}
    for c in ["drone_type", "climate_condition", "source_area", "destination_area", "traffic_condition", "drone_id"]:
        if c in df.columns:
            try:
                encs[c] = LabelEncoder().fit(df[c].astype(str))
            except Exception as e:
                logger.warning(f"Encoder fit failed for {c}: {e}")
                encs[c] = None
        else:
            encs[c] = None
    return df, encs

@st.cache_resource
def load_eta_model_cached(path=ETA_MODEL_PATH):
    if os.path.exists(path):
        try:
            model = load_model(path)
            logger.info("Loaded ETA model")
            return model
        except Exception as e:
            logger.warning(f"Failed to load ETA model: {e}")
            return None
    logger.info("ETA model not found")
    return None

@st.cache_resource
def load_scaler_cached(path=SCALER_PATH):
    if os.path.exists(path):
        try:
            scal = joblib.load(path)
            logger.info("Loaded scaler")
            return scal
        except Exception as e:
            logger.warning(f"Failed to load scaler: {e}")
            return None
    logger.info("Scaler not found")
    return None

@st.cache_resource
def load_yolo_cached(path=YOLO_MODEL_PATH):
    if os.path.exists(path):
        try:
            y = YOLO(path)
            logger.info("Loaded YOLO model")
            return y
        except Exception as e:
            logger.warning(f"Failed to load YOLO model: {e}")
            return None
    logger.info("YOLO model not found")
    return None

drone_df, encs = load_drone_df_and_encoders()
eta_model = load_eta_model_cached()
scaler = load_scaler_cached()
detection_model = load_yolo_cached()

# ------------------- Sidebar Navigation -------------------
with st.sidebar:
    if os.path.exists(AITOP_PATH):
        st.image(AITOP_PATH, use_container_width=True)
    st.markdown("<h2>Aero AI</h2>", unsafe_allow_html=True)
    st.markdown("---")
    pages = ["Home", "ETA Prediction", "Drone Detection", "Chatbot"]
    selected = st.radio("Navigation", pages, index=0)

# ------------------- HOME PAGE -------------------
if selected == "Home":
    if os.path.exists(BANNER_PATH):
        st.image(BANNER_PATH, use_container_width=True)
    st.markdown("<h1 style='text-align:center;'>🚁 Aero AI - Drone Delivery System</h1>", unsafe_allow_html=True)
    st.markdown("AI-powered ETA prediction, drone detection, and a TiDB-backed chatbot.")
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.markdown("### ⏱ ETA Prediction\nAccurate delivery time estimation using ML models")
    col2.markdown("### 🎯 Drone Detection\nDetect drones / damage using YOLO")
    col3.markdown("### 💬 Chatbot\nConversational interface for ETA, status, PDF")

# ------------------- ETA PREDICTION PAGE -------------------
elif selected == "ETA Prediction":
    st.header("⏱️ Drone Delivery ETA Prediction")

    # Require resources
    if eta_model is None or scaler is None or drone_df is None or not encs:
        st.warning("ETA model / scaler / drone.csv encoders not loaded. Check files in project directory.")
        st.stop()

    st.markdown("Fill the fields below (encoders loaded from `drone.csv`).")
    with st.form("eta_form"):
        st.subheader("Order / Drone Details")
        c1, c2 = st.columns(2)
        with c1:
            order_id_txt = st.text_input("Order ID", value=f"ORD{np.random.randint(1000,9999)}")
            drone_id_sel = st.selectbox("Drone ID", list(encs["drone_id"].classes_))
            distance_km   = st.number_input("Distance (km)", value=10.0, step=0.1, format="%.2f")
            payload_weight_kg = st.number_input("Payload Weight (kg)", value=2.0, step=0.1, format="%.2f")
        with c2:
            drone_type_sel = st.selectbox("Drone Type", list(encs["drone_type"].classes_))
            drone_speed_kmph = st.number_input("Drone Speed (km/h)", value=30.0, step=0.1, format="%.2f")
            battery_efficiency_pct = st.number_input("Battery Efficiency (%)", value=90.0,
                                                     min_value=0.0, max_value=100.0, step=1.0)
            wind_speed_kmph = st.number_input("Wind Speed (km/h)", value=5.0, step=0.1, format="%.2f")

        st.subheader("Environment")
        climate_condition_sel = st.selectbox(
            "Climate Condition",
            list(encs["climate_condition"].classes_) if encs.get("climate_condition") else ["Sunny"]
        )
        temperature_c = st.number_input("Temperature (°C)", value=25.0, step=0.1, format="%.2f")
        humidity_percent = st.number_input("Humidity (%)", value=60.0, step=1.0)

        st.subheader("Location")
        source_area_sel = st.selectbox("Source Area", list(encs["source_area"].classes_))
        destination_area_sel = st.selectbox("Destination Area", list(encs["destination_area"].classes_))
        traffic_condition_sel = st.selectbox("Traffic Condition", list(encs["traffic_condition"].classes_))

        predict_btn = st.form_submit_button("🚀 Predict ETA")

    # Single-predict behavior: map shown after prediction; no session map-cache required
    if predict_btn:
        try:
            input_df = pd.DataFrame({
                'drone_id':          [encs["drone_id"].transform([drone_id_sel])[0]],
                'drone_type':        [encs["drone_type"].transform([drone_type_sel])[0]],
                'drone_speed_kmph':  [drone_speed_kmph],
                'payload_weight_kg': [payload_weight_kg],
                'distance_km':       [distance_km],
                'battery_efficiency':[battery_efficiency_pct],
                'climate_condition': [encs["climate_condition"].transform([climate_condition_sel])[0]]
                                                    if encs.get("climate_condition") else [0],
                'wind_speed_kmph':   [wind_speed_kmph],
                'source_area':       [encs["source_area"].transform([source_area_sel])[0]],
                'destination_area':  [encs["destination_area"].transform([destination_area_sel])[0]],
                'traffic_condition': [encs["traffic_condition"].transform([traffic_condition_sel])[0]]
            })

            X_scaled = scaler.transform(input_df)
            pred = eta_model.predict(X_scaled)
            eta_val = float(pred[0][0]) if hasattr(pred, "shape") else float(pred)

            col1, col2, col3 = st.columns(3)
            col1.metric("Estimated Time", f"{eta_val:.0f} min")
            col2.metric("Distance",    f"{distance_km} km")
            col3.metric("Speed",       f"{drone_speed_kmph} km/h")
            st.success("✅ ETA Prediction Complete!")

            # Show route map for this drone/order using drone_geo.csv
            if os.path.exists(DRONE_GEO_CSV):
                geo_df = pd.read_csv(DRONE_GEO_CSV)
                geo_df_valid = geo_df.dropna(subset=['source_lat','source_lon','destination_lat','destination_lon'])
                geo_df_valid = geo_df_valid[
                    (geo_df_valid['source_lat'] != '') & (geo_df_valid['source_lon'] != '') &
                    (geo_df_valid['destination_lat'] != '') & (geo_df_valid['destination_lon'] != '')
                ]
                filtered = geo_df_valid[geo_df_valid['drone_id'] == drone_id_sel]

                if not filtered.empty:
                    last_order = filtered.iloc[-1]
                    s_lat, s_lon = float(last_order['source_lat']), float(last_order['source_lon'])
                    d_lat, d_lon = float(last_order['destination_lat']), float(last_order['destination_lon'])
                    center_lat, center_lon = (s_lat + d_lat) / 2, (s_lon + d_lon) / 2

                    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
                    folium.Marker([s_lat, s_lon], popup=f"Source: {last_order['source_area']}", icon=folium.Icon(color="green")).add_to(m)
                    folium.Marker([d_lat, d_lon], popup=f"Destination: {last_order['destination_area']}", icon=folium.Icon(color="red")).add_to(m)
                    folium.PolyLine([[s_lat, s_lon], [d_lat, d_lon]], color="blue", weight=3, opacity=0.7).add_to(m)

                    st.subheader("📍 Drone Delivery Route Map")
                    st_folium(m, width=700, height=400)
                else:
                    st.warning("⚠ No route coordinates for the selected drone.")
            else:
                st.info("drone_geo.csv not found — skipping route map (place a drone_geo.csv file in project root).")

        except Exception as e:
            logger.error(f"ETA prediction error: {e}")
            st.error(f"Prediction failed: {e}")

# ------------------- DRONE DETECTION PAGE -------------------
elif selected == "Drone Detection":
    st.header("🎯 Drone Detection (Manual Upload)")

    # Use relative YOLO path
    st.info(f"Using YOLO model: {YOLO_MODEL_PATH}")
    try:
        detection_model = load_yolo_cached(YOLO_MODEL_PATH) or detection_model
        if detection_model:
            st.success(f"✅ Loaded YOLO model")
        else:
            st.warning("YOLO model not loaded — place your model at the path above if you want detection.")
    except Exception as e:
        st.error(f"Failed to initialize YOLO model: {e}")
        detection_model = None

    if detection_model is not None:
        uploaded_file = st.file_uploader("Upload a drone image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if img is not None:
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)
                try:
                    primary_conf = 0.25
                    fallback_conf = 0.1

                    results = detection_model(img, conf=primary_conf)
                    r0 = results[0]
                    boxes = getattr(r0, "boxes", None)

                    if boxes is None or len(boxes) == 0:
                        st.info(f"No detections at primary confidence — trying fallback threshold {fallback_conf} ...")
                        results = detection_model(img, conf=fallback_conf)
                        r0 = results[0]
                        boxes = getattr(r0, "boxes", None)

                    annotated = r0.plot()
                    st.image(annotated[:, :, ::-1], caption="Detection Result", use_container_width=True)

                    if boxes is not None and len(boxes) > 0:
                        class_names = [detection_model.names[int(b.cls)] for b in boxes]
                        confidences = [float(b.conf.cpu().numpy()) if hasattr(b.conf, "cpu") else float(b.conf) for b in boxes]

                        drones = [(n, c) for n, c in zip(class_names, confidences) if n != "damage"]
                        damages = [(n, c) for n, c in zip(class_names, confidences) if n == "damage"]

                        unique_drones = {}
                        for n, c in drones:
                            if n not in unique_drones or c > unique_drones[n]:
                                unique_drones[n] = c

                        st.subheader("Detected Drone Types")
                        if unique_drones:
                            for t, conf in unique_drones.items():
                                st.markdown(f"- **{t}** (Confidence: {conf:.2f})")
                        else:
                            st.info("No drones detected.")

                        st.subheader("Detected Damages")
                        if damages:
                            for t, conf in damages:
                                st.markdown(f"- **{t}** (Confidence: {conf:.2f})")
                        else:
                            st.info("No damages detected.")

                        col1, col2 = st.columns(2)
                        col1.metric("Total Drones", len(unique_drones))
                        col2.metric("Total Damages", len(damages))
                        st.info(f"Model used: {os.path.basename(YOLO_MODEL_PATH)}")

                        # store metadata in TiDB if available
                        if engine:
                            try:
                                import json, base64
                                drone_meta = json.dumps(unique_drones)
                                damage_meta = json.dumps(damages)
                                filename = uploaded_file.name
                                _, buffer = cv2.imencode('.jpg', img)
                                img_base64 = base64.b64encode(buffer).decode('utf-8')
                                damage_flag = 1 if damages else 0

                                stmt = text("""
                                    INSERT INTO drone_detection_records 
                                    (filename, drone_types, drone_confidences, damage_detected, image_base64, created_at)
                                    VALUES (:filename, :drone_types, :drone_confidences, :damage_detected, :image_base64, NOW())
                                """)
                                with engine.begin() as conn:
                                    conn.execute(stmt, {
                                        "filename": filename,
                                        "drone_types": ",".join(unique_drones.keys()) if unique_drones else None,
                                        "drone_confidences": ",".join([f"{c:.2f}" for c in unique_drones.values()]) if unique_drones else None,
                                        "damage_detected": damage_flag,
                                        "image_base64": img_base64
                                    })
                                st.success("✅ Detection metadata saved to TiDB")
                            except Exception as e:
                                logger.warning(f"Failed to save detection metadata: {e}")
                                st.error(f"Failed to save to TiDB: {e}")
                    else:
                        st.info("No detections — try a different image or check confidence threshold.")
                except Exception as e:
                    logger.error(f"Detection error: {e}")
                    st.error(f"Detection error: {e}")
            else:
                st.error("Failed to read uploaded image.")

# ------------------- CHATBOT PAGE -------------------
elif selected == "Chatbot":
    st.header("💬 Aero AI Chatbot")
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    def generate_pdf_bytes(order_id, record_summary_text):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, f"Drone Delivery Report: {order_id}", ln=True, align="C")
        pdf.ln(6)
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 8, record_summary_text)
        return io.BytesIO(pdf.output(dest="S").encode("latin-1"))

    # Note: Email feature removed for GitHub-safe version
    def send_email_placeholder(order_id, to_email, body_html):
        return False, "Email feature disabled in public repo. Configure SendGrid in cloud.env to enable."

    def process_command(user_text):
        tokens = user_text.strip().split()
        if not tokens: return "⚠ Please type a command. Example: ETA ORD123"
        cmd = tokens[0].lower()
        if cmd in ("hi","hello","hey"):
            return "Hello! Commands: ETA <ORDER_ID>, Status <ORDER_ID>, Damage <ORDER_ID>, Generate PDF <ORDER_ID>"
        if cmd == "eta":
            if len(tokens)<2: return "⚠ Provide order ID. ETA ORD123"
            order_id = tokens[1].strip()
            if engine:
                df = safe_read("SELECT ETA_actual_min FROM drone_delivery_eta WHERE TRIM(order_id)=%s LIMIT 1", params=(order_id,))
                if not df.empty: return f"⏱ ETA for {order_id}: {df.iloc[0,0]} min"
                else: return f"No ETA record for {order_id} in DB."
            return f"⏱ ETA for {order_id}: 14.2 minutes (mock)"
        if cmd == "status":
            if len(tokens)<2: return "⚠ Provide order ID. Status ORD123"
            order_id = tokens[1].strip()
            if engine:
                df = safe_read("SELECT drone_id, ETA_actual_min FROM drone_delivery_eta WHERE TRIM(order_id)=%s LIMIT 1", params=(order_id,))
                if not df.empty:
                    return f"📦 Order {order_id} — Drone {df.iloc[0]['drone_id']} — ETA {df.iloc[0]['ETA_actual_min']} min"
                else: return f"No status record for {order_id} in DB."
            return f"📦 Status for {order_id}: In Transit (mock)"
        if cmd == "damage":
            if len(tokens)<2: return "⚠ Provide order ID. Damage ORD123"
            order_id = tokens[1].strip()
            if engine:
                df = safe_read("SELECT filename, has_damage FROM drone_dataset WHERE filename LIKE %s LIMIT 1", params=(f"%{order_id}%",))
                if not df.empty: return "Damage info:\n" + df.to_string(index=False)
                else: return f"No damage info for {order_id} in DB."
            return f"🛠 Damage {order_id}: No damage detected (mock)"
        if cmd == "send" and len(tokens)>=3 and tokens[1].lower()=="email":
            return "❌ Email feature disabled in public repo. Configure SendGrid in cloud.env to enable."
        if cmd == "generate" and len(tokens)>=3 and tokens[1].lower()=="pdf":
            order_id = tokens[2].strip()
            summary_text = f"Order report for {order_id}\n"
            if engine:
                df = safe_read("SELECT * FROM drone_delivery_eta WHERE TRIM(order_id)=%s LIMIT 1", params=(order_id,))
                summary_text += df.to_string(index=False) if not df.empty else "No record; mock report.\n"
            pdf_bytes = generate_pdf_bytes(order_id, summary_text)
            return {"pdf_bytes": pdf_bytes, "order_id": order_id}
        return "Unknown command. Try: ETA ORD0001, Status ORD0001, Generate PDF ORD0001"

    # Render chat history
    for msg in st.session_state.chat_messages:
        role, content = msg.get("role"), msg.get("content")
        if role == "user":
            st.markdown(f"<div class='chat-bubble-user'>{content}</div><div class='clearfix'></div>", unsafe_allow_html=True)
        else:
            if isinstance(content, dict) and content.get("pdf_bytes"):
                st.markdown(f"<div class='chat-bubble-bot'>📄 PDF ready: {content.get('order_id')}</div><div class='clearfix'></div>", unsafe_allow_html=True)
                st.download_button(label=f"📥 Download {content.get('order_id')}.pdf",
                                   data=content.get("pdf_bytes"),
                                   file_name=f"{content.get('order_id')}.pdf",
                                   mime="application/pdf")
            else:
                st.markdown(f"<div class='chat-bubble-bot'>{content}</div><div class='clearfix'></div>", unsafe_allow_html=True)

    # Chat input
    user_text = st.chat_input("Type your message (e.g. ETA ORD435504)")
    if user_text:
        st.session_state.chat_messages.append({"role":"user","content":user_text})
        st.markdown(f"<div class='chat-bubble-user'>{user_text}</div><div class='clearfix'></div>", unsafe_allow_html=True)
        result = process_command(user_text)
        if isinstance(result, dict) and result.get("pdf_bytes"):
            st.session_state.chat_messages.append({"role":"assistant","content":result})
            st.markdown(f"<div class='chat-bubble-bot'>📄 PDF generated for {result.get('order_id')}</div><div class='clearfix'></div>", unsafe_allow_html=True)
            st.download_button(label=f"📥 Download {result.get('order_id')}.pdf",
                               data=result.get("pdf_bytes"),
                               file_name=f"{result.get('order_id')}.pdf",
                               mime="application/pdf")
        else:
            st.session_state.chat_messages.append({"role":"assistant","content":str(result)})
            st.markdown(f"<div class='chat-bubble-bot'>{result}</div><div class='clearfix'></div>", unsafe_allow_html=True)
