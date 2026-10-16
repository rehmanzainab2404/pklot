import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os
import tempfile
import sqlite3
import pandas as pd
from datetime import datetime
import time

st.set_page_config(
    page_title="SmartPark Intelligence Core",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
    }
    .metric-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 10px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    .metric-label {
        font-size: 11px;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-num {
        font-size: 24px;
        font-weight: 700;
        margin-top: 2px;
    }
    .c-total { color: #0F172A; }
    .c-vacant { color: #059669; }
    .c-occupied { color: #E11D48; }
    .c-rate { color: #2563EB; }
    .status-pill {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        background-color: #DCFCE7;
        color: #166534;
        margin-top: 4px;
    }
    .db-pill {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        background-color: #E0E7FF;
        color: #3730A3;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

DB_FILE = "smartpark.db"

def init_database():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS occupancy_telemetry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            source_type TEXT,
            total_bays INTEGER,
            occupied_bays INTEGER,
            vacant_bays INTEGER,
            occupancy_rate REAL
        )
    """)
    conn.commit()
    conn.close()

init_database()

def insert_telemetry(source_type, total, occupied, vacant, rate):
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT INTO occupancy_telemetry (timestamp, source_type, total_bays, occupied_bays, vacant_bays, occupancy_rate)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (current_time, source_type, total, occupied, vacant, rate))
        conn.commit()
        conn.close()
    except Exception:
        pass

def fetch_telemetry_records(limit=8):
    try:
        conn = sqlite3.connect(DB_FILE)
        df = pd.read_sql_query(f"""
            SELECT timestamp AS Time, source_type AS Source, occupied_bays AS Occupied, vacant_bays AS Available, occupancy_rate || '%' AS Rate
            FROM occupancy_telemetry
            ORDER BY id DESC LIMIT {limit}
        """, conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

def get_all_records_csv():
    try:
        conn = sqlite3.connect(DB_FILE)
        df = pd.read_sql_query("SELECT * FROM occupancy_telemetry ORDER BY id DESC", conn)
        conn.close()
        return df.to_csv(index=False).encode('utf-8')
    except Exception:
        return "".encode('utf-8')

@st.cache_resource
def load_yolo():
    path = "smartpark_yolo.pt"
    if os.path.exists(path):
        return YOLO(path)
    return None

detector = load_yolo()

def get_slots(width, height):
    slots = []
    rows = 2
    cols = 8
    slot_w = int(width * 0.08)
    slot_h = int(height * 0.22)
    start_x = int(width * 0.12)
    spacing_x = int(width * 0.10)
    
    for r in range(rows):
        start_y = int(height * 0.16) if r == 0 else int(height * 0.56)
        for c in range(cols):
            x1 = start_x + (c * spacing_x)
            y1 = start_y
            x2 = min(x1 + slot_w, width - 10)
            y2 = min(y1 + slot_h, height - 10)
            if x2 > x1 and y2 > y1:
                slots.append((x1, y1, x2, y2))
    return slots

def process_frame(frame, model):
    h, w, _ = frame.shape
    slots = get_slots(w, h)
    
    occupied = 0
    vacant = 0
    
    for (x1, y1, x2, y2) in slots:
        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            continue
            
        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        patch_resized = cv2.resize(patch_rgb, (128, 128))
        
        preds = model(patch_resized, verbose=False)[0]
        label = preds.names[preds.probs.top1]
        
        if label.lower() == "occupied":
            occupied += 1
            color = (72, 29, 225)
            text = "OCCUPIED"
        else:
            vacant += 1
            color = (105, 150, 5)
            text = "VACANT"
            
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, text, (x1 + 4, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        
    return frame, len(slots), occupied, vacant

st.markdown("<h3 style='margin-bottom: 2px; font-weight: 700;'>SmartPark Surveillance Core</h3>", unsafe_allow_html=True)
st.markdown("<p style='color: #64748B; font-size: 13px; margin-bottom: 18px;'>Autonomous Edge Telemetry & SQLite Integrated Storage Engine</p>", unsafe_allow_html=True)

if detector is None:
    st.error("Model weights file 'smartpark_yolo.pt' daryaft nahi hui.")
    st.stop()

col_left, col_right = st.columns([1.1, 2.3], gap="large")

with col_left:
    st.markdown("<h5 style='color: #334155;'>Occupancy Summary</h5>", unsafe_allow_html=True)
    m_total = st.empty()
    m_vacant = st.empty()
    m_occupied = st.empty()
    m_rate = st.empty()
    
    st.markdown("""
    <div class="metric-box">
        <div class="metric-label">System Architecture</div>
        <div class="status-pill">YOLOV8-NANO • ACTIVE</div>
        <div class="db-pill">SQLITE3 • CONNECTED</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h5 style='color: #334155; margin-top: 14px;'>Recent Audit Logs</h5>", unsafe_allow_html=True)
    table_placeholder = st.empty()
    
    csv_bytes = get_all_records_csv()
    st.download_button(
        label="Download Full Audit Log (CSV)",
        data=csv_bytes,
        file_name="smartpark_audit_log.csv",
        mime="text/csv",
        use_container_width=True
    )

def update_ui_metrics(total, vacant, occupied, rate):
    m_total.markdown(f'<div class="metric-box"><div class="metric-label">Total Bays</div><div class="metric-num c-total">{total}</div></div>', unsafe_allow_html=True)
    m_vacant.markdown(f'<div class="metric-box"><div class="metric-label">Available</div><div class="metric-num c-vacant">{vacant}</div></div>', unsafe_allow_html=True)
    m_occupied.markdown(f'<div class="metric-box"><div class="metric-label">Occupied</div><div class="metric-num c-occupied">{occupied}</div></div>', unsafe_allow_html=True)
    m_rate.markdown(f'<div class="metric-box"><div class="metric-label">Occupancy Rate</div><div class="metric-num c-rate">{rate}%</div></div>', unsafe_allow_html=True)
    table_placeholder.dataframe(fetch_telemetry_records(), hide_index=True, use_container_width=True)

update_ui_metrics(16, 0, 0, 0)

with col_right:
    st.markdown("<h5 style='color: #334155;'>Live Surveillance View</h5>", unsafe_allow_html=True)
    
    source = st.radio(
        "Select Camera Feed:",
        ["Image Upload", "Video Upload", "Laptop Webcam", "IP Camera Stream"],
        horizontal=True
    )
    
    view = st.empty()
    
    if "last_log_time" not in st.session_state:
        st.session_state.last_log_time = 0
        
    if source == "Image Upload":
        uploaded = st.file_uploader("Parking lot snapshot upload karein", type=["jpg", "jpeg", "png"])
        if uploaded:
            buf = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            frame = cv2.imdecode(buf, 1)
            out_frame, tot, occ, vac = process_frame(frame, detector)
            rate = round((occ / tot) * 100, 1) if tot > 0 else 0.0
            
            insert_telemetry("Snapshot", tot, occ, vac, rate)
            update_ui_metrics(tot, vac, occ, rate)
            view.image(cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            
    elif source == "Video Upload":
        vid_file = st.file_uploader("Video footage upload karein", type=["mp4", "avi", "mov"])
        play = st.toggle("Stream Video")
        if vid_file and play:
            t = tempfile.NamedTemporaryFile(delete=False)
            t.write(vid_file.read())
            cap = cv2.VideoCapture(t.name)
            
            f_count = 0
            while cap.isOpened() and play:
                ret, frame = cap.read()
                if not ret:
                    break
                f_count += 1
                if f_count % 3 != 0:
                    continue
                out_frame, tot, occ, vac = process_frame(frame, detector)
                rate = round((occ / tot) * 100, 1) if tot > 0 else 0.0
                
                curr_t = time.time()
                if curr_t - st.session_state.last_log_time >= 4.0:
                    insert_telemetry("Video", tot, occ, vac, rate)
                    st.session_state.last_log_time = curr_t
                    
                update_ui_metrics(tot, vac, occ, rate)
                view.image(cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            cap.release()
            
    elif source == "Laptop Webcam":
        cam_active = st.toggle("Turn On Webcam")
        if cam_active:
            cap = cv2.VideoCapture(0)
            while cam_active:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Webcam accessible nahi hai.")
                    break
                out_frame, tot, occ, vac = process_frame(frame, detector)
                rate = round((occ / tot) * 100, 1) if tot > 0 else 0.0
                
                curr_t = time.time()
                if curr_t - st.session_state.last_log_time >= 4.0:
                    insert_telemetry("Webcam", tot, occ, vac, rate)
                    st.session_state.last_log_time = curr_t
                    
                update_ui_metrics(tot, vac, occ, rate)
                view.image(cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            cap.release()
            
    elif source == "IP Camera Stream":
        ip_url = st.text_input("RTSP ya HTTP Stream URL:", placeholder="http://192.168.1.15:8080/video")
        stream_active = st.toggle("Connect to Stream")
        if stream_active and ip_url:
            cap = cv2.VideoCapture(ip_url)
            while stream_active:
                ret, frame = cap.read()
                if not ret:
                    st.warning("IP stream disconnect ho gaya.")
                    break
                out_frame, tot, occ, vac = process_frame(frame, detector)
                rate = round((occ / tot) * 100, 1) if tot > 0 else 0.0
                
                curr_t = time.time()
                if curr_t - st.session_state.last_log_time >= 4.0:
                    insert_telemetry("IP Stream", tot, occ, vac, rate)
                    st.session_state.last_log_time = curr_t
                    
                update_ui_metrics(tot, vac, occ, rate)
                view.image(cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            cap.release()

