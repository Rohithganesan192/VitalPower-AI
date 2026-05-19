import os
import json
import tempfile
import torch
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, File, UploadFile, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import io
import time
import asyncio
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv

import google.generativeai as genai
from ml_engine import HybridTransformerGNN

load_dotenv()
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "")
SENDER_APP_PASSWORD = os.getenv("SENDER_APP_PASSWORD", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# --- MULTI-USER STATE ISOLATION ---
user_last_alert_times: Dict[str, float] = {}
user_system_states: Dict[str, Dict[str, Any]] = {}
user_datasets: Dict[str, pd.DataFrame] = {}
user_dataset_indices: Dict[str, int] = {}

def get_user_state(user_id: str) -> Dict[str, Any]:
    if user_id not in user_system_states:
        user_system_states[user_id] = {
            "latest_risk": 0.0,
            "active_appliances": [],
            "status": "Safe",
            "hazard_type": None,
            "hazard_message": "",
            "anomaly_detected": False
        }
    return user_system_states[user_id]

app = FastAPI(title="VitalPower AI Backend")
app.add_middleware(
    CORSMiddleware, 
    allow_origins=[
        "https://vitalpower-ai.web.app",
        "http://localhost:5500",
        "http://127.0.0.1:5500"
    ], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
)

# --- INITIALIZE REAL ML MODEL ---
NUM_FEATURES = 9
SEQ_LEN = 24
try:
    print("Initializing PyTorch HybridTransformerGNN...")
    model = HybridTransformerGNN(num_features=NUM_FEATURES, seq_len=SEQ_LEN, nhead=2, num_layers=2, hidden_dim=64)
    model.eval()
except Exception as e:
    print(f"Warning: Failed to initialize ML engine: {e}")
    model = None

# --- EMAIL BROADCASTER (FIXED) ---
async def send_emergency_alerts(user_id: str, hazard_type: str, details: str):
    if not SENDER_EMAIL or not SENDER_APP_PASSWORD:
        print("⚠️ Missing Email Credentials in .env")
        return

    contacts_path = os.path.join(os.path.dirname(__file__), f"contacts_{user_id}.json")
    try:
        with open(contacts_path, "r") as f:
            contacts = json.load(f)
    except Exception as e:
        print(f"⚠️ Could not read contacts file for {user_id}: {e}")
        return

    email_contacts = [c for c in contacts if c.get("alert_preference") in ("email", "both") and c.get("email")]
    if not email_contacts:
        return

    def _send_to(contact: dict):
        try:
            msg = EmailMessage()
            msg.set_content(f"Dear {contact.get('name', 'Caregiver')},\n\nURGENT: VitalPower AI has detected a {hazard_type}.\n\nDetails:\n{details}\n\nPlease check on the resident immediately.")
            msg['Subject'] = f"🚨 VITALPOWER AI HEALTH ALERT: {hazard_type}"
            msg['From'] = SENDER_EMAIL
            msg['To'] = contact["email"]

            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(SENDER_EMAIL, SENDER_APP_PASSWORD)
                server.send_message(msg)
            print(f"📧 Alert successfully sent to {contact['email']}")
        except Exception as e:
            print(f"⚠️ Failed to send alert: {e}")

    loop = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(None, _send_to, c) for c in email_contacts]
    await asyncio.gather(*tasks)

# --- DATASET BOOTLOADER ---
boot_hourly_df = None
try:
    default_path = os.path.join(os.path.dirname(__file__), "demo_datasets", "0_healthy_baseline.csv")
    if os.path.exists(default_path):
        boot_df = pd.read_csv(default_path)
        if 'datetime' not in boot_df.columns and 'Date' in boot_df.columns and 'Time' in boot_df.columns:
            boot_df['datetime'] = pd.to_datetime(boot_df['Date'] + ' ' + boot_df['Time'], dayfirst=True, errors='coerce')
        boot_df['datetime'] = pd.to_datetime(boot_df['datetime'])
        boot_df.set_index('datetime', inplace=True)
        boot_hourly_df = boot_df.resample('ME' if pd.__version__ >= '2.2.0' else '1h').mean(numeric_only=True).reset_index() if 'ME' in str(pd.__version__) else boot_df.resample('1h').mean(numeric_only=True).reset_index()
        boot_hourly_df = boot_df.resample('1h').mean(numeric_only=True).reset_index()
        boot_hourly_df.ffill(inplace=True)
except Exception as e:
    print(f"Failed to load baseline on boot: {e}")

# --- DATASET UPLOAD AND LIVE STREAMING ROUTES ---
@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...), x_user_id: str = Header("default_user")):
    try:
        contents = await file.read()
        try:
            df = pd.read_csv(io.BytesIO(contents), sep=';')
            if len(df.columns) < 5:
                df = pd.read_csv(io.BytesIO(contents), sep=',')
        except:
            df = pd.read_csv(io.BytesIO(contents), sep=',')
            
        df.ffill(inplace=True)
        user_datasets[x_user_id] = df
        user_dataset_indices[x_user_id] = 0
        return {"status": "success", "message": f"Dataset uploaded with {len(df)} rows."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process dataset: {str(e)}")

@app.post("/load-demo/{scenario_id}")
async def load_demo(scenario_id: int, x_user_id: str = Header("default_user")):
    demo_files = [
        "0_healthy_baseline.csv",
        "1_crisis_nutrition.csv",
        "2_crisis_hygiene.csv",
        "3_crisis_engagement.csv",
        "4_crisis_circadian.csv",
        "5_crisis_cognitive_hazard.csv",
        "6_crisis_fall_mobility.csv"
    ]
    if scenario_id < 0 or scenario_id >= len(demo_files):
        raise HTTPException(status_code=400, detail="Invalid scenario")
        
    file_path = os.path.join(os.path.dirname(__file__), "demo_datasets", demo_files[scenario_id])
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Demo file not found")
        
    df = pd.read_csv(file_path)
    
    if 'datetime' not in df.columns and 'Date' in df.columns and 'Time' in df.columns:
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True, errors='coerce')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    hourly_df = df.resample('1h').mean(numeric_only=True).reset_index()
    hourly_df.ffill(inplace=True)
    
    user_datasets[x_user_id] = hourly_df
    user_dataset_indices[x_user_id] = 0
    return {"status": "success", "message": f"Loaded {demo_files[scenario_id]} with {len(hourly_df)} rows"}

@app.get("/next-reading")
async def get_next_reading(x_user_id: str = Header("default_user")):
    df = user_datasets.get(x_user_id, boot_hourly_df)
    idx = user_dataset_indices.get(x_user_id, 0)
    
    if df is None or len(df) == 0:
        raise HTTPException(status_code=404, detail="No dataset uploaded yet.")
    
    if idx >= len(df):
        idx = 0 
        
    row = df.iloc[idx]
    user_dataset_indices[x_user_id] = idx + 1
    
    return {
        "datetime": str(row.get('datetime', row.get('timestamp', pd.Timestamp.now()))),
        "Global_active_power": float(row.get('Global_active_power', row.get('power_watts', 0.0))),
        "Global_reactive_power": float(row.get('Global_reactive_power', 0.1)),
        "Voltage": float(row.get('Voltage', 240.0)),
        "Global_intensity": float(row.get('Global_intensity', 0.0)),
        "Sub_metering_1": float(row.get('Sub_metering_1', 0.0)),
        "Sub_metering_2": float(row.get('Sub_metering_2', 0.0)),
        "Sub_metering_3": float(row.get('Sub_metering_3', 0.0)),
    }

# --- AI DATA INGESTION & HEALTH MAPPING ---
class Reading(BaseModel):
    datetime: str
    Global_active_power: float
    Global_reactive_power: float
    Voltage: float
    Global_intensity: float
    Sub_metering_1: float
    Sub_metering_2: float
    Sub_metering_3: float

class AnalyzeRequest(BaseModel):
    readings: List[Reading]
    vacation_mode: bool = False

def get_diagnostic_phase(anomaly_score: float, hour_of_day: int, avg_power: float, mse_per_feature: List[float], is_anomaly: bool) -> Dict[str, str]:
    if not is_anomaly and anomaly_score < 0.60:
        return {"phase_name": "Routine Normal", "severity": "Normal", "diagnostic_msg": "Routine patterns appear normal. No clinical risks detected.", "icon_key": "safe", "type": "None", "msg": "Normal routine detected."}
    
    is_low_power = avg_power < 0.25
    if is_low_power:
        return {"phase_name": "🚨 Mobility Phase Alert", "severity": "Critical", "diagnostic_msg": "Prolonged Sedentary State: Total house movement is critically below baseline. High risk of fall or acute illness.", "icon_key": "mobility", "type": "Missed Routine / Inactivity", "msg": "🚨 CRITICAL: Resident has missed expected daily routines. Potential fall."}
    
    if hour_of_day < 6 and avg_power > 1.0:
        return {"phase_name": "Circadian Phase", "severity": "Warning", "diagnostic_msg": "High energy usage detected during typical sleep hours. Risk: Sleep Disruption.", "icon_key": "sleep", "type": "Sleep Disruption", "msg": "⚠️ WARNING: Sleep disruption likely."}
        
    if avg_power > 2.5:
        return {"phase_name": "Cognitive Safety", "severity": "Critical", "diagnostic_msg": "Sustained high power draw detected. Risk: Potential hazard/left appliances on.", "icon_key": "hazard", "type": "Potential Hazard", "msg": "🚨 CRITICAL: Hazard risk. High power draw."}

    if mse_per_feature:
        power_errors = mse_per_feature[:7]
        culprit_idx = power_errors.index(max(power_errors))
        
        if culprit_idx == 4:
            return {"phase_name": "Nutrition Phase", "severity": "Warning", "diagnostic_msg": "Anomalous kitchen appliance usage. Risk: Nutritional Neglect.", "icon_key": "nutrition", "type": "Nutritional Anomaly", "msg": "⚠️ WARNING: Kitchen anomaly."}
        elif culprit_idx == 5:
            return {"phase_name": "Engagement Phase", "severity": "Warning", "diagnostic_msg": "Anomalous living room/laundry usage. Risk: Domestic Withdrawal.", "icon_key": "engagement", "type": "Engagement Anomaly", "msg": "⚠️ WARNING: Engagement anomaly."}
        elif culprit_idx == 6:
            return {"phase_name": "Hygiene Phase", "severity": "Warning", "diagnostic_msg": "Anomalous bathroom/water heater usage. Risk: Self-Care Deficit.", "icon_key": "hygiene", "type": "Hygiene Anomaly", "msg": "⚠️ WARNING: Hygiene anomaly."}

    if is_anomaly:
        return {"phase_name": "🚨 General Usage Anomaly", "severity": "Warning", "diagnostic_msg": "Unexpected energy draw detected across multiple appliances. Resident routine is likely disrupted.", "icon_key": "warning", "type": "Abnormal Usage", "msg": "⚠️ WARNING: Unusual patterns."}
    
    return {"phase_name": "Routine Normal", "severity": "Normal", "diagnostic_msg": "Routine patterns appear normal. No clinical risks detected.", "icon_key": "safe", "type": "None", "msg": "Normal routine detected."}

@app.post("/analyze-usage")
async def analyze_usage(request: AnalyzeRequest, x_user_id: str = Header("default_user")):
    if not model:
        raise HTTPException(status_code=500, detail="PyTorch Model not loaded.")
    
    df = pd.DataFrame([r.model_dump() for r in request.readings])
    if len(df) < SEQ_LEN:
        pad_df = pd.DataFrame([df.iloc[-1].to_dict()] * (SEQ_LEN - len(df)))
        df = pd.concat([pad_df, df], ignore_index=True)

    try:
        dt = pd.to_datetime(df['datetime'])
        hour_of_day = dt.iloc[-1].hour
        avg_power = df['Global_active_power'].mean()
        
        features = df[['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']].astype(float)
        features['hour'] = dt.dt.hour
        features['day'] = dt.dt.dayofweek
        tensor_seq = torch.tensor(features.values[-SEQ_LEN:], dtype=torch.float32)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data prep failed: {str(e)}")

    threshold = 2.0 if request.vacation_mode else 0.8
    anomaly_score, is_anomaly, mse_per_feature = model.evaluate_anomaly(tensor_seq, threshold=threshold)
    
    health_insight = get_diagnostic_phase(anomaly_score, hour_of_day, avg_power, mse_per_feature, is_anomaly)
    
    # 4. Trigger Alerts & Update User State
    current_time = time.time()
    state = get_user_state(x_user_id)
    
    state["latest_risk"] = float(anomaly_score)
    state["status"] = "Alert" if is_anomaly else "Safe"
    state["hazard_type"] = health_insight["type"]
    state["hazard_message"] = health_insight["diagnostic_msg"]
    state["anomaly_detected"] = bool(is_anomaly)
    
    last_time = user_last_alert_times.get(x_user_id, 0.0)
    
    if health_insight["severity"] == "Critical" and (current_time - last_time > 300):
        await send_emergency_alerts(x_user_id, health_insight["phase_name"], health_insight["diagnostic_msg"])
        user_last_alert_times[x_user_id] = current_time

    return {
        "status": "Alert" if is_anomaly else "Safe",
        "risk_score": float(anomaly_score),
        "health_behavior": health_insight["type"],
        "message": health_insight["msg"],
        "severity": health_insight["severity"].lower(),
        "health_insight": {
            "phase_name": health_insight["phase_name"],
            "severity": health_insight["severity"],
            "diagnostic_msg": health_insight["diagnostic_msg"],
            "icon_key": health_insight["icon_key"]
        }
    }

# --- CONTACT REGISTRY ROUTES ---
@app.get("/get-contacts")
async def get_contacts(x_user_id: str = Header("default_user")):
    try:
        contacts_path = os.path.join(os.path.dirname(__file__), f"contacts_{x_user_id}.json")
        with open(contacts_path, "r") as f:
            return {"contacts": json.load(f)}
    except:
        return {"contacts": []}

@app.post("/add-contact")
async def add_contact(contact: dict, x_user_id: str = Header("default_user")):
    contacts_path = os.path.join(os.path.dirname(__file__), f"contacts_{x_user_id}.json")
    try:
        with open(contacts_path, "r") as f:
            contacts = json.load(f)
    except:
        contacts = []
    contact["id"] = int(time.time() * 1000)
    contacts.append(contact)
    with open(contacts_path, "w") as f:
        json.dump(contacts, f, indent=2)
    return {"message": "Contact added successfully"}

@app.delete("/remove-contact/{contact_id}")
async def remove_contact(contact_id: int, x_user_id: str = Header("default_user")):
    contacts_path = os.path.join(os.path.dirname(__file__), f"contacts_{x_user_id}.json")
    try:
        with open(contacts_path, "r") as f:
            contacts = json.load(f)
    except:
        return {"message": "No contacts found"}
        
    initial_count = len(contacts)
    contacts = [c for c in contacts if c.get("id") != contact_id]
    
    if len(contacts) < initial_count:
        with open(contacts_path, "w") as f:
            json.dump(contacts, f, indent=2)
        return {"message": "Contact removed successfully"}
    else:
        raise HTTPException(status_code=404, detail="Contact not found")

# --- GEMINI CHATBOT ROUTE ---
@app.post("/chat-query")
async def chat_query(request: dict, x_user_id: str = Header("default_user")):
    if not GEMINI_API_KEY:
        return {"reply": "API Key missing.", "type": "warning"}
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gen_model = genai.GenerativeModel('gemini-2.5-flash')
        
        state = get_user_state(x_user_id)
        
        state_context = (
            f"Current Risk Score: {state['latest_risk']:.2f}. "
            f"Status: {state['status']}. "
            f"Recent Notifications: {state['hazard_message']}. "
            f"Anomaly Detected: {state['anomaly_detected']}."
        )
        
        fc = request.get('context', {})
        frontend_str = ""
        if fc:
            frontend_str = (
                f"Dashboard Context:\n"
                f"- CSV Uploaded: {fc.get('csv_uploaded')} (Filename: {fc.get('csv_filename')})\n"
                f"- Vacation Mode: {fc.get('vacation_mode')}\n"
                f"- Active Registered Contacts: {fc.get('contacts_count')}\n"
                f"- Health Routine (Morning: {fc.get('health_routine', {}).get('morning')}, "
                f"Midday: {fc.get('health_routine', {}).get('midday')}, "
                f"Night: {fc.get('health_routine', {}).get('night')})\n"
                f"- AI Assessment: {fc.get('health_routine', {}).get('assessment')}\n"
                f"- Analytics (Total Reads: {fc.get('analytics', {}).get('total_reads')}, "
                f"Anomalies: {fc.get('analytics', {}).get('anomalies')}, "
                f"Avg Risk: {fc.get('analytics', {}).get('avg_risk')})\n"
                f"- Peak Energy Value (kW): {fc.get('energy_data', {}).get('peak_value')}\n"
                f"- Recent Event Logs (Last 10): {', '.join(fc.get('event_log', []))}\n"
                f"- Last 24h Energy Usage (kW): {fc.get('energy_data', {}).get('actual_kw_24h')}\n"
            )
        
        prompt = (
            f"You are VitalPower AI, a highly intelligent senior health and safety monitor. "
            f"You HAVE full access to the user's dashboard data, charts, logs, and application state.\n"
            f"Here is the real-time system state from the user's home suite:\n{state_context}\n\n"
            f"Here is the detailed application data from the frontend dashboard:\n{frontend_str}\n\n"
            f"Answer the user's query thoughtfully, accurately, and conversationally. You can and should reference the graphs, logs, routines, and risk scores provided above when relevant to their question.\n"
            f"User Query: {request.get('message')}"
        )
        res = gen_model.generate_content(prompt)
        return {"reply": res.text, "type": "info"}
    except Exception as e:
        return {"reply": f"LLM Offline: {str(e)}", "type": "warning"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)