import os

WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
LOCAL_CACHE_DIR = os.path.join(WORKSPACE_ROOT, ".runtime_cache")
REMOTE_DEVICES_FILE = os.path.join(WORKSPACE_ROOT, "artifacts", "remote_devices.json")
os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(REMOTE_DEVICES_FILE), exist_ok=True)
os.environ.setdefault("TORCH_HOME", os.path.join(LOCAL_CACHE_DIR, "torch"))
os.environ.setdefault("ULTRALYTICS_CONFIG_DIR", os.path.join(LOCAL_CACHE_DIR, "ultralytics"))

import uvicorn
from fastapi import FastAPI, Request, File, UploadFile, HTTPException, Query
from yolo_inference import YOLOInference, PIXELS_PER_SQM
import cv2
import threading
import uvicorn
import numpy as np
import time
import json
import re
import uuid
from pathlib import Path
from typing import List
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import torch

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for camera
camera_active = False
camera_thread = None
current_camera_frame = None
frame_lock = threading.Lock()
camera_area_sqm = 50.0  # Default area in square meters
camera_source = 0
camera_source_label = "Webcam 0"
camera_stats = {
    "people_count": 0,
    "density": 0.0,
    "density_level": "Low",
    "alert_status": "Safe",
    "fps": 0,
    "source": camera_source_label,
    "heatmap_enabled": False,
    "last_snapshot": None,
    "last_error": None,
    "model": "fast",
    "model_label": "YOLOv8 Nano",
}

current_model_key = "fast"

MAX_VIDEO_SIZE_BYTES = 500 * 1024 * 1024
MAX_IMAGE_SIZE_BYTES = 50 * 1024 * 1024
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

processing_status = {
    "active": False,
    "filename": None,
    "progress_percent": 0.0,
    "processed_frames": 0,
    "total_frames": 0,
    "people_count": 0,
    "density": 0.0,
    "completed": False,
    "output_path": None,
    "error": None,
}

remote_devices_lock = threading.Lock()
mobile_frames_lock = threading.Lock()
mobile_frames = {}

# Setup directories
UPLOAD_FOLDER = os.path.join('static', 'uploads')
PROCESSED_FOLDER = os.path.join('static', 'processed')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs("snapshots", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory=os.path.join(os.getcwd(), "templates"))

# Initialize YOLO with optimized parameters for better accuracy
model_path = 'yolov8n.pt'
yolo_infer = YOLOInference(model_path=model_path)  # Use YOLOv8n with improved settings
camera_stats["model"] = yolo_infer.get_model_status()["key"]
camera_stats["model_label"] = yolo_infer.get_model_status()["label"]


def sanitize_filename(filename: str) -> str:
    base = Path(filename or "").name
    sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    return sanitized or f"upload_{int(time.time())}"


def validate_extension(filename: str, allowed_extensions: set[str], label: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Invalid {label} file type")
    return ext


def parse_camera_source(source_value):
    if source_value is None or source_value == "":
        return 0, "Webcam 0"

    text = str(source_value).strip()
    if text.isdigit():
        index = int(text)
        return index, f"Webcam {index}"
    return text, text


def load_remote_devices():
    if not os.path.exists(REMOTE_DEVICES_FILE):
        return []
    try:
        with open(REMOTE_DEVICES_FILE, "r", encoding="utf-8") as handle:
            data = json.load(handle)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def save_remote_devices(devices):
    with open(REMOTE_DEVICES_FILE, "w", encoding="utf-8") as handle:
        json.dump(devices, handle, indent=2)


def get_remote_devices():
    with remote_devices_lock:
        return load_remote_devices()


def upsert_remote_device(device):
    with remote_devices_lock:
        devices = load_remote_devices()
        for index, existing in enumerate(devices):
            if existing["id"] == device["id"]:
                devices[index] = device
                save_remote_devices(devices)
                return device
        devices.append(device)
        save_remote_devices(devices)
        return device


def remove_remote_device(device_id):
    with remote_devices_lock:
        devices = [device for device in load_remote_devices() if device["id"] != device_id]
        save_remote_devices(devices)


def build_source_from_device(device):
    if device["type"] == "mobile":
        return f"mobile:{device['id']}", device["name"]
    return device["source"], device["name"]


def resolve_source_value(source_value):
    parsed_source, parsed_label = parse_camera_source(source_value)
    if isinstance(parsed_source, str):
        for device in get_remote_devices():
            if parsed_source == device["id"]:
                return build_source_from_device(device)
    return parsed_source, parsed_label


def get_mobile_frame(device_id):
    with mobile_frames_lock:
        frame_info = mobile_frames.get(device_id)
        if not frame_info:
            return None, None
        return frame_info.get("frame"), frame_info.get("timestamp")


def reset_processing_status():
    processing_status.update({
        "active": False,
        "filename": None,
        "progress_percent": 0.0,
        "processed_frames": 0,
        "total_frames": 0,
        "people_count": 0,
        "density": 0.0,
        "completed": False,
        "output_path": None,
        "error": None,
    })


def update_processing_status(processed_frames, total_frames, people_count, density, completed=False):
    processing_status["active"] = not completed
    processing_status["processed_frames"] = int(processed_frames or 0)
    processing_status["total_frames"] = int(total_frames or 0)
    processing_status["people_count"] = int(people_count or 0)
    processing_status["density"] = float(density or 0.0)
    processing_status["completed"] = bool(completed)
    processing_status["progress_percent"] = round(
        (processed_frames / total_frames) * 100, 1
    ) if total_frames else (100.0 if completed else 0.0)


def process_video_job(video_path: str, processed_path: str, original_filename: str):
    reset_processing_status()
    processing_status["active"] = True
    processing_status["filename"] = original_filename
    processing_status["output_path"] = f"/static/processed/{os.path.basename(processed_path)}"
    try:
        yolo_infer.process_video(video_path, processed_path, progress_callback=update_processing_status)
    except Exception as exc:
        processing_status["active"] = False
        processing_status["completed"] = False
        processing_status["error"] = str(exc)
    else:
        processing_status["active"] = False
        processing_status["completed"] = True


def open_video_source(source):
    if isinstance(source, int):
        return cv2.VideoCapture(source, cv2.CAP_DSHOW)
    return cv2.VideoCapture(source)


def get_system_state():
    model_status = yolo_infer.get_model_status()
    return {
        "camera_active": camera_active,
        "camera_stats": camera_stats,
        "processing_status": processing_status,
        "models": yolo_infer.get_available_models(),
        "current_model": model_status,
        "remote_devices": get_remote_devices(),
    }

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/favicon.ico")
async def favicon():
    return {"message": "No favicon"}

# --- Feature Control Endpoints ---
@app.get("/toggle_heatmap")
async def toggle_heatmap():
    yolo_infer.set_heatmap_enabled(not yolo_infer.enable_heat_map)
    return RedirectResponse(url="/live_camera", status_code=302)

@app.get("/live_camera", response_class=HTMLResponse)
async def live_camera(request: Request):
    return templates.TemplateResponse("live_camera_v2.html", {"request": request})


@app.get("/remote_devices", response_class=HTMLResponse)
async def remote_devices_page(request: Request):
    return templates.TemplateResponse("remote_devices.html", {"request": request})


@app.get("/mobile_camera/{device_id}", response_class=HTMLResponse)
async def mobile_camera_page(request: Request, device_id: str):
    devices = get_remote_devices()
    device = next((item for item in devices if item["id"] == device_id and item["type"] == "mobile"), None)
    if not device:
        raise HTTPException(status_code=404, detail="Mobile device not found")
    return templates.TemplateResponse("mobile_camera.html", {"request": request, "device": device})


@app.get("/system_state")
async def system_state():
    return get_system_state()


@app.get("/api/devices")
async def list_devices():
    return {"devices": get_remote_devices()}


@app.post("/api/devices")
async def create_device(request: Request):
    payload = await request.json()
    device_type = payload.get("type", "").strip().lower()
    name = payload.get("name", "").strip() or "Unnamed Device"
    source = payload.get("source", "").strip()
    allowed_types = {"cctv", "ip", "mobile"}
    if device_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Device type must be cctv, ip, or mobile")
    if device_type != "mobile" and not source:
        raise HTTPException(status_code=400, detail="Source URL is required for CCTV/IP devices")

    device = {
        "id": payload.get("id") or uuid.uuid4().hex[:10],
        "name": name,
        "type": device_type,
        "source": source,
        "notes": payload.get("notes", "").strip(),
        "created_at": int(time.time()),
    }
    upsert_remote_device(device)
    return {
        "message": "Device saved",
        "device": device,
        "mobile_url": f"/mobile_camera/{device['id']}" if device_type == "mobile" else None,
        "live_source": f"mobile:{device['id']}" if device_type == "mobile" else device["source"],
    }


@app.delete("/api/devices/{device_id}")
async def delete_device(device_id: str):
    remove_remote_device(device_id)
    return {"message": "Device removed"}


@app.post("/api/mobile_frame/{device_id}")
async def ingest_mobile_frame(device_id: str, frame: UploadFile = File(...)):
    content = await frame.read()
    np_buffer = np.frombuffer(content, dtype=np.uint8)
    decoded = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(status_code=400, detail="Invalid image frame")

    with mobile_frames_lock:
        mobile_frames[device_id] = {"frame": decoded, "timestamp": time.time()}
    return {"message": "Frame received"}


@app.get("/api/mobile_frame/{device_id}/status")
async def mobile_frame_status(device_id: str):
    _, timestamp = get_mobile_frame(device_id)
    return {"connected": timestamp is not None and (time.time() - timestamp) < 5, "last_frame_ts": timestamp}


@app.post("/set_model")
async def set_model(request: Request):
    global current_model_key

    payload = await request.json()
    model_key = payload.get("model_key", "fast")
    if model_key not in yolo_infer.get_available_models():
        raise HTTPException(status_code=400, detail="Unknown model")
    if camera_active:
        raise HTTPException(status_code=409, detail="Stop the live camera before switching models")

    try:
        yolo_infer.load_model(model_key=model_key)
        current_model_key = model_key
        model_status = yolo_infer.get_model_status()
        camera_stats["model"] = model_status["key"]
        camera_stats["model_label"] = model_status["label"]
        return {"message": "Model updated", "model": model_status}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/start_camera")
async def start_camera(request: Request):
    global camera_active, camera_thread, camera_area_sqm, camera_source, camera_source_label

    if camera_active:
        return {"status": "Camera already active"}

    try:
        data = await request.json()
        area_sqm = data.get("area_sqm", 50.0)
        source_value = data.get("source", 0)
        if area_sqm <= 0:
            return {"status": "Invalid area value"}
        camera_area_sqm = float(area_sqm)
        camera_source, camera_source_label = resolve_source_value(source_value)
    except Exception:
        camera_area_sqm = 50.0
        camera_source, camera_source_label = 0, "Webcam 0"

    camera_active = True
    camera_thread = threading.Thread(target=run_camera_processing, daemon=True)
    camera_thread.start()

    camera_stats["source"] = camera_source_label
    camera_stats["last_error"] = None
    return {"status": "Camera started successfully", "source": camera_source_label}

@app.post("/stop_camera")
async def stop_camera():
    global camera_active
    camera_active = False
    camera_stats["fps"] = 0
    return {"status": "Camera stopped"}

@app.get("/camera_feed")
def camera_feed():
    """Fixed camera feed with proper streaming format"""
    def generate():
        global current_camera_frame, frame_lock
        while camera_active:
            with frame_lock:
                if current_camera_frame is not None:
                    try:
                        ret, buffer = cv2.imencode('.jpg', current_camera_frame, 
                                                 [cv2.IMWRITE_JPEG_QUALITY, 85])
                        if ret:
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + 
                                   buffer.tobytes() + b'\r\n')
                    except Exception as e:
                        print(f"Error encoding frame: {e}")
                else:
                    placeholder = np.zeros((240, 320, 3), dtype=np.uint8)
                    cv2.putText(placeholder, "Initializing Camera...", (180, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    ret, buffer = cv2.imencode('.jpg', placeholder)
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + 
                               buffer.tobytes() + b'\r\n')
            
            time.sleep(0.04)

    return StreamingResponse(generate(),
                           media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/camera_stats")
async def get_camera_stats():
    return camera_stats

@app.post("/toggle_camera_heatmap")
async def toggle_camera_heatmap():
    yolo_infer.set_heatmap_enabled(not yolo_infer.enable_heat_map)
    camera_stats["heatmap_enabled"] = yolo_infer.enable_heat_map
    return {"heatmap_enabled": yolo_infer.enable_heat_map}


@app.get("/processing_status")
async def get_processing_status():
    return processing_status

def run_camera_processing():
    """FIXED: Run camera processing with real YOLO model"""
    global current_camera_frame, camera_stats, camera_active, frame_lock

    try:
        yolo_infer.load_model(model_key=current_model_key)
        model = yolo_infer.model
        model_status = yolo_infer.get_model_status()
        camera_stats["model"] = model_status["key"]
        camera_stats["model_label"] = model_status["label"]
    except Exception as e:
        print(f"Error loading active model: {e}")
        camera_stats["last_error"] = str(e)
        camera_active = False
        return
    
    cap = None
    mobile_device_id = None
    if isinstance(camera_source, str) and camera_source.startswith("mobile:"):
        mobile_device_id = camera_source.split(":", 1)[1]
    else:
        cap = open_video_source(camera_source)
        if not cap.isOpened():
            print(f"Error: Could not open camera source {camera_source_label}")
            camera_stats["last_error"] = f"Could not open source: {camera_source_label}"
            camera_active = False
            return

        cap.set(cv2.CAP_PROP_FPS, 25)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("Camera processing started with YOLO model")

    fps_counter = 0
    fps_start_time = time.time()
    pulse_timer = 0
    crowd_counts = []
    frame_skip_counter = 0
    FRAME_SKIP_RATE = 1  # Process every frame for maximum FPS
    last_people_count = 0
    estimated_camera_area = None  # For dynamic estimation if not user-provided
    user_provided_area = camera_area_sqm != 50.0  # Assume 50.0 is default
    camera_stats["source"] = camera_source_label
    camera_stats["heatmap_enabled"] = yolo_infer.enable_heat_map

    while camera_active:
        if mobile_device_id:
            mobile_frame, frame_timestamp = get_mobile_frame(mobile_device_id)
            if mobile_frame is None or frame_timestamp is None or (time.time() - frame_timestamp) > 5:
                placeholder = np.zeros((480, 800, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Waiting for mobile camera connection...", (80, 210),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(placeholder, "Open the mobile broadcaster page and allow camera access.", (55, 255),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 220, 255), 2)
                with frame_lock:
                    current_camera_frame = placeholder
                camera_stats["last_error"] = "Waiting for mobile device frames"
                time.sleep(0.08)
                continue
            frame = mobile_frame.copy()
        else:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read from camera")
                camera_stats["last_error"] = f"Could not read from source: {camera_source_label}"
                break

        try:
            height, width = frame.shape[:2]

            # Initialize density map if not exists or wrong size
            if yolo_infer.density_map is None or yolo_infer.density_map.shape[:2] != (height, width):
                yolo_infer.density_map = np.zeros((height, width), dtype=np.float32)

            # Frame skipping for higher FPS - process every FRAME_SKIP_RATE frames
            frame_skip_counter += 1
            process_frame = (frame_skip_counter % FRAME_SKIP_RATE == 0)

            people_count = 0  # Initialize people_count
            detections = []

            if process_frame and model is not None:
                # Run YOLO inference with optimized parameters for better accuracy
                results = model(frame, conf=0.35, device=yolo_infer.device, verbose=False, imgsz=640, max_det=100, iou=0.45)

                # Process YOLO results
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes.data.cpu().numpy():
                            x1, y1, x2, y2, conf, class_id = box

                            # Check if detected object is a person (class_id = 0 in COCO dataset)
                            if int(class_id) == 0:  # Person class
                                people_count += 1
                                w, h = x2 - x1, y2 - y1
                                detections.append(([x1, y1, w, h], conf, "person"))

                                # Update density map for this person
                                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                                if 0 <= cx < width and 0 <= cy < height:
                                    cv2.circle(yolo_infer.density_map, (cx, cy), 25, (1.0,), thickness=-1)

                                # Draw bounding box around detected person
                                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                                # Add person label with confidence
                                label = f'Person {conf:.2f}'
                                cv2.putText(frame, label, (int(x1), int(y1) - 10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Estimate area if not user-provided and detections available
                if not user_provided_area and estimated_camera_area is None and detections:
                    estimated_camera_area = yolo_infer.estimate_area_from_detections(width, height, detections)
                    print(f"[INFO] Estimated camera area from detections: {estimated_camera_area} sqm")

                # Update last detection results
                last_people_count = people_count
            else:
                # Use last detection results for skipped frames
                people_count = last_people_count

            # Implement smoothing
            crowd_counts.append(people_count)
            if len(crowd_counts) > 30:
                crowd_counts.pop(0)
            smoothed_count = int(sum(crowd_counts) / len(crowd_counts)) if crowd_counts else 0

            # Calculate density based on area
            area_for_density = camera_area_sqm if user_provided_area else (estimated_camera_area if estimated_camera_area else camera_area_sqm)
            density = smoothed_count / area_for_density if area_for_density > 0 else 0

            # Enhanced pulsing red live indicator
            pulse_timer += 1
            pulse_intensity = int(128 + 127 * np.sin(pulse_timer * 0.2))

            # Draw pulsing red dot
            cv2.circle(frame, (20, height - 20), 10, (0, 0, pulse_intensity), -1)
            cv2.circle(frame, (20, height - 20), 10, (255, 255, 255), 2)
            cv2.putText(frame, "LIVE", (38, height - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Draw overlays
            frame = yolo_infer._draw_overlays(frame, smoothed_count, density)

            # Update stats with real detection data
            camera_stats["people_count"] = smoothed_count
            camera_stats["density"] = round(density, 3)
            camera_stats["density_level"] = yolo_infer.get_density_level(density)
            camera_stats["alert_status"] = yolo_infer.get_alert_status(density, smoothed_count)
            camera_stats["heatmap_enabled"] = yolo_infer.enable_heat_map

            alert_event = yolo_infer.handle_alert(frame, density, smoothed_count, source_label=camera_source_label)
            if alert_event:
                camera_stats["last_snapshot"] = alert_event["snapshot_path"].replace("\\", "/")
                camera_stats["alert_status"] = alert_event["status"]
            
            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_start_time >= 1.0:
                camera_stats["fps"] = fps_counter
                fps_counter = 0
                fps_start_time = time.time()
            
            with frame_lock:
                current_camera_frame = frame.copy()
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            camera_stats["last_error"] = str(e)
            with frame_lock:
                current_camera_frame = frame
        
        time.sleep(0.04)

    if cap is not None:
        cap.release()
    camera_stats["fps"] = 0
    print("Camera processing stopped")

@app.post("/upload")
async def upload(video: UploadFile = File(...)):
    if not video.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    validate_extension(video.filename, ALLOWED_VIDEO_EXTENSIONS, "video")
    safe_name = sanitize_filename(video.filename)
    video_path = os.path.join(UPLOAD_FOLDER, safe_name)
    content = await video.read()
    if len(content) > MAX_VIDEO_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="Video exceeds 500MB limit")

    with open(video_path, "wb") as buffer:
        buffer.write(content)

    processed_filename = f"processed_{safe_name}"
    processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)

    threading.Thread(
        target=process_video_job,
        args=(video_path, processed_path, safe_name),
        daemon=True
    ).start()

    return {
        "message": "File uploaded successfully",
        "filename": safe_name,
        "processed_file": f"/static/processed/{processed_filename}",
    }

@app.post("/upload_image")
async def upload_image(
    image: UploadFile = File(...),
    selected_area_sqm: float = 0.0,
    scale_factor: float = 1.0,
    personal_space: float = 1.8,
    region_type: str = "rectangle",
    region_data: str = "{}"
):
    if not image.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    validate_extension(image.filename, ALLOWED_IMAGE_EXTENSIONS, "image")
    safe_name = sanitize_filename(image.filename)
    content = await image.read()
    if len(content) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="Image exceeds 50MB limit")

    image_path = os.path.join(UPLOAD_FOLDER, safe_name)

    with open(image_path, "wb") as buffer:
        buffer.write(content)

    processed_filename = f"processed_{safe_name}"
    processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)

    # Parse region data
    try:
        region_info = {
            "type": region_type,
            "data": json.loads(region_data) if region_data != "{}" else None,
            "selected_area_sqm": selected_area_sqm,
            "scale_factor": scale_factor,
            "personal_space": personal_space
        }
    except json.JSONDecodeError:
        region_info = None

    # Run enhanced analysis
    result = process_image_with_area_analysis(image_path, processed_path, region_info)

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return result

def process_image_with_area_analysis(image_path, processed_path, region_info=None):
    """Enhanced image processing with area-based crowd analysis"""
    import time
    start_time = time.time()

    print(f"[INFO] Processing image with area analysis: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Could not load image: {image_path}")
        return {"error": "Could not load image"}

    height, width = frame.shape[:2]

    # Initialize density map
    yolo_infer.density_map = np.zeros((height, width), dtype=np.float32)

    # Run YOLO detection
    results = yolo_infer.model(frame, conf=0.4, device=yolo_infer.device, imgsz=416, max_det=50, iou=0.5)
    detections = []
    people_count = 0
    people_in_area = 0

    for result in results:
        for box in result.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, class_id = box
            class_id = int(class_id)

            # Draw bounding box for all detected objects
            w, h = x2 - x1, y2 - y1
            detections.append(([x1, y1, w, h], conf, class_id))

            # Special handling for people (class 0)
            if class_id == 0:  # Person class
                people_count += 1
                person_center = ((x1 + x2) / 2, (y1 + y2) / 2)

                # Check if person is within selected area
                if region_info and region_info["data"]:
                    if is_point_in_region(person_center, region_info):
                        people_in_area += 1
                        # Highlight person in area
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                        cv2.putText(frame, f'Person {conf:.2f}', (int(x1), int(y1) - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        # Person outside area - different color
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (128, 128, 128), 2)
                        cv2.putText(frame, f'Outside {conf:.2f}', (int(x1), int(y1) - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)
                else:
                    # No area selected - highlight all
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f'Person {conf:.2f}', (int(x1), int(y1) - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Update density map
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                cv2.circle(yolo_infer.density_map, (cx, cy), 25, (1.0,), thickness=-1)
            else:
                # Draw other detected objects
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, f'Class {class_id} {conf:.2f}', (int(x1), int(y1) - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Draw selected region on image
    if region_info and region_info["data"]:
        draw_region_on_image(frame, region_info)

    # Calculate metrics
    if region_info and region_info["selected_area_sqm"] > 0:
        selected_area_sqm = region_info["selected_area_sqm"]
    else:
        selected_area_sqm = yolo_infer.estimate_area_from_detections(width, height, detections, image_path)
    detected_people_count = people_in_area if region_info and region_info["data"] else people_count
    personal_space = region_info["personal_space"] if region_info else 1.8

    # Capacity estimation
    estimated_max_capacity = int(selected_area_sqm / personal_space)

    # Current density calculation
    current_density = detected_people_count / selected_area_sqm if selected_area_sqm > 0 else 0
    current_density_percentage = (detected_people_count / estimated_max_capacity * 100) if estimated_max_capacity > 0 else 0

    # Safety status determination
    if current_density_percentage < 50:
        safety_status = "safe"
    elif current_density_percentage < 80:
        safety_status = "caution"
    elif current_density_percentage < 100:
        safety_status = "warning"
    else:
        safety_status = "critical"

    # Draw stats on image
    stats_y = 30
    cv2.putText(frame, f"Selected Area: {selected_area_sqm:.2f} sqm", (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    stats_y += 30
    cv2.putText(frame, f"People in Area: {detected_people_count}", (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    stats_y += 30
    cv2.putText(frame, f"Max Capacity: {estimated_max_capacity}", (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    stats_y += 30
    cv2.putText(frame, f"Current Density: {current_density:.3f} p/sqm", (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    stats_y += 30
    cv2.putText(frame, f"Safety Status: {safety_status.upper()}", (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Save processed image
    cv2.imwrite(processed_path, frame)
    print(f"[INFO] Processed image saved to: {processed_path}")

    processing_time_ms = int((time.time() - start_time) * 1000)

    yolo_infer.latest_frame = frame.copy()

    return {
        "message": "Image uploaded and analyzed successfully",
        "selected_area_sqm": round(selected_area_sqm, 2),
        "detected_people_count": detected_people_count,
        "estimated_max_capacity": estimated_max_capacity,
        "current_density": round(current_density, 3),
        "current_density_percentage": round(current_density_percentage, 1),
        "safety_status": safety_status,
        "processing_time_ms": processing_time_ms,
        "total_people_detected": people_count,
        "processed_image": f"/static/processed/{os.path.basename(processed_path)}"
    }

def is_point_in_region(point, region_info):
    """Check if a point is within the selected region"""
    if not region_info or not region_info["data"]:
        return True

    x, y = point

    if region_info["type"] == "rectangle":
        data = region_info["data"]
        return (data["startX"] <= x <= data["endX"] and
                data["startY"] <= y <= data["endY"])

    elif region_info["type"] in ["polygon", "freehand"]:
        # Use ray casting algorithm for polygon point-inclusion
        points = region_info["data"]
        if not points or len(points) < 3:
            return False

        inside = False
        j = len(points) - 1
        for i in range(len(points)):
            if ((points[i]["y"] > y) != (points[j]["y"] > y) and
                (x < points[i]["x"] + (points[j]["x"] - points[i]["x"]) * (y - points[i]["y"]) / (points[j]["y"] - points[i]["y"] + 1e-10))):
                inside = not inside
            j = i
        return inside

    return False

def draw_region_on_image(frame, region_info):
    """Draw the selected region boundary on the image"""
    if not region_info or not region_info["data"]:
        return

    if region_info["type"] == "rectangle":
        data = region_info["data"]
        cv2.rectangle(frame,
                     (int(data["startX"]), int(data["startY"])),
                     (int(data["endX"]), int(data["endY"])),
                     (255, 0, 0), 3)
    elif region_info["type"] in ["polygon", "freehand"]:
        points = region_info["data"]
        if points and len(points) > 1:
            pts = np.array([[int(p["x"]), int(p["y"])] for p in points], np.int32)
            cv2.polylines(frame, [pts], region_info["type"] == "polygon", (255, 0, 0), 3)
            if region_info["type"] == "polygon":
                cv2.fillPoly(frame, [pts], (255, 0, 0, 50))

@app.get("/live_preview", response_class=HTMLResponse)
async def live_preview(request: Request):
    return templates.TemplateResponse("live_preview_v2.html", {"request": request})

@app.get("/video_feed")
async def video_feed():
    def generate():
        while True:
            if yolo_infer.latest_frame is None:
                time.sleep(0.1)
                continue
            ret, buffer = cv2.imencode('.jpg', yolo_infer.latest_frame)
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   buffer.tobytes() + b'\r\n')
            time.sleep(0.04)

    return StreamingResponse(generate(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/set_zoom")
async def set_zoom(row: int = Query(default=-1), col: int = Query(default=-1)):
    yolo_infer.set_zoom_cell(row, col)
    return {"status": "OK"}

@app.get("/zoom_feed")
async def zoom_feed():
    def gen():
        while True:
            subimg = yolo_infer.get_zoomed_subimage()
            if subimg is None:
                time.sleep(0.1)
                continue
            ret, buffer = cv2.imencode('.jpg', subimg)
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   buffer.tobytes() + b'\r\n')
            time.sleep(0.04)

    return StreamingResponse(gen(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/process_video")
async def process_video_route():
    input_video_path = os.path.join(UPLOAD_FOLDER, "input.mp4")
    output_video_path = os.path.join(PROCESSED_FOLDER, "output.mp4")
    
    threading.Thread(
        target=process_video_job,
        args=(input_video_path, output_video_path, "input.mp4"),
        daemon=True
    ).start()
    
    return RedirectResponse(url="/live_preview", status_code=302)

@app.post("/batch_process")
async def batch_process(
    files: List[UploadFile] = File(...),
    personal_space: float = 1.8,
    scale_factor: float = 1.0
):
    """Process multiple files in batch"""
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided")

    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed for batch processing")

    results = []

    for file in files:
        if not file.filename:
            continue

        safe_name = sanitize_filename(file.filename)
        ext = Path(safe_name).suffix.lower()
        if ext not in ALLOWED_IMAGE_EXTENSIONS.union(ALLOWED_VIDEO_EXTENSIONS):
            continue

        file_path = os.path.join(UPLOAD_FOLDER, f"batch_{safe_name}")
        processed_filename = f"batch_processed_{safe_name}"
        processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)

        content = await file.read()
        size_limit = MAX_IMAGE_SIZE_BYTES if ext in ALLOWED_IMAGE_EXTENSIONS else MAX_VIDEO_SIZE_BYTES
        if len(content) > size_limit:
            results.append({
                "filename": safe_name,
                "error": "File exceeds size limit",
                "detected_people_count": 0,
                "safety_status": "error"
            })
            continue

        with open(file_path, "wb") as buffer:
            buffer.write(content)

        try:
            content_type = file.content_type or ""
            if content_type.startswith('image/'):
                # Process image
                result = process_image_batch(file_path, processed_path, personal_space, scale_factor)
            elif content_type.startswith('video/'):
                # Process video (extract first frame for demo)
                result = process_video_batch(file_path, processed_path, personal_space, scale_factor)
            else:
                if ext in ALLOWED_IMAGE_EXTENSIONS:
                    result = process_image_batch(file_path, processed_path, personal_space, scale_factor)
                elif ext in ALLOWED_VIDEO_EXTENSIONS:
                    result = process_video_batch(file_path, processed_path, personal_space, scale_factor)
                else:
                    continue

            result["filename"] = file.filename
            result["processed_file"] = f"/static/processed/{processed_filename}"
            results.append(result)

        except Exception as e:
            print(f"Error processing {file.filename}: {e}")
            results.append({
                "filename": file.filename,
                "error": str(e),
                "detected_people_count": 0,
                "safety_status": "error"
            })
        finally:
            # Clean up uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)

    return results

def process_image_batch(image_path, processed_path, personal_space, scale_factor):
    """Process single image for batch processing"""
    import time
    start_time = time.time()

    frame = cv2.imread(image_path)
    if frame is None:
        return {"error": "Could not load image"}

    height, width = frame.shape[:2]
    yolo_infer.density_map = np.zeros((height, width), dtype=np.float32)

    # Run YOLO detection
    results = yolo_infer.model(frame, conf=0.4, device=yolo_infer.device, imgsz=416, max_det=50, iou=0.5)
    detections = []
    people_count = 0

    for result in results:
        for box in result.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, class_id = box
            class_id = int(class_id)
            w, h = x2 - x1, y2 - y1
            detections.append(([x1, y1, w, h], conf, class_id))

            if class_id == 0:  # Person class
                people_count += 1
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'Person {conf:.2f}', (int(x1), int(y1) - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                # Draw other detected objects
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, f'Class {class_id} {conf:.2f}', (int(x1), int(y1) - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Calculate metrics using improved area estimation
    selected_area_sqm = yolo_infer.estimate_area_from_detections(width, height, detections, image_path)
    # Apply scale factor if provided
    if scale_factor != 1.0:
        selected_area_sqm *= scale_factor
    estimated_max_capacity = int(selected_area_sqm / personal_space)
    current_density = people_count / selected_area_sqm if selected_area_sqm > 0 else 0
    current_density_percentage = (people_count / estimated_max_capacity * 100) if estimated_max_capacity > 0 else 0

    # Safety status
    if current_density_percentage < 50:
        safety_status = "safe"
    elif current_density_percentage < 80:
        safety_status = "caution"
    elif current_density_percentage < 100:
        safety_status = "warning"
    else:
        safety_status = "critical"

    # Draw stats
    cv2.putText(frame, f"People: {people_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Area: {selected_area_sqm:.1f} sqm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Capacity: {estimated_max_capacity}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Status: {safety_status.upper()}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imwrite(processed_path, frame)

    return {
        "detected_people_count": people_count,
        "selected_area_sqm": round(selected_area_sqm, 2),
        "estimated_max_capacity": estimated_max_capacity,
        "current_density": round(current_density, 3),
        "current_density_percentage": round(current_density_percentage, 1),
        "safety_status": safety_status,
        "processing_time_ms": int((time.time() - start_time) * 1000)
    }

def process_video_batch(video_path, processed_path, personal_space, scale_factor):
    """Process video for batch processing (extract first frame)"""
    import time
    start_time = time.time()

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return {"error": "Could not read video"}

    height, width = frame.shape[:2]
    yolo_infer.density_map = np.zeros((height, width), dtype=np.float32)

    # Run YOLO detection on first frame
    results = yolo_infer.model(frame, conf=0.4, device=yolo_infer.device, imgsz=416, max_det=50, iou=0.5)
    detections = []
    people_count = 0

    for result in results:
        for box in result.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, class_id = box
            class_id = int(class_id)
            w, h = x2 - x1, y2 - y1
            detections.append(([x1, y1, w, h], conf, class_id))

            if class_id == 0:  # Person class
                people_count += 1
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'Person {conf:.2f}', (int(x1), int(y1) - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                # Draw other detected objects
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, f'Class {class_id} {conf:.2f}', (int(x1), int(y1) - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Calculate metrics using improved area estimation
    selected_area_sqm = yolo_infer.estimate_area_from_detections(width, height, detections)
    # Apply scale factor if provided
    if scale_factor != 1.0:
        selected_area_sqm *= scale_factor
    estimated_max_capacity = int(selected_area_sqm / personal_space)
    current_density = people_count / selected_area_sqm if selected_area_sqm > 0 else 0
    current_density_percentage = (people_count / estimated_max_capacity * 100) if estimated_max_capacity > 0 else 0

    # Safety status
    if current_density_percentage < 50:
        safety_status = "safe"
    elif current_density_percentage < 80:
        safety_status = "caution"
    elif current_density_percentage < 100:
        safety_status = "warning"
    else:
        safety_status = "critical"

    # Draw stats
    cv2.putText(frame, f"People: {people_count} (first frame)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Area: {selected_area_sqm:.1f} sqm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Capacity: {estimated_max_capacity}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Status: {safety_status.upper()}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imwrite(processed_path, frame)

    return {
        "detected_people_count": people_count,
        "selected_area_sqm": round(selected_area_sqm, 2),
        "estimated_max_capacity": estimated_max_capacity,
        "current_density": round(current_density, 3),
        "current_density_percentage": round(current_density_percentage, 1),
        "safety_status": safety_status,
        "processing_time_ms": int((time.time() - start_time) * 1000),
        "note": "Video analysis based on first frame only"
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
