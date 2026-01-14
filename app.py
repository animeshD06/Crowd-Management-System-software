import uvicorn
from fastapi import FastAPI, Request, File, UploadFile, HTTPException, Query
from yolo_inference import YOLOInference, PIXELS_PER_SQM
import os
import cv2
import threading
import uvicorn
import numpy as np
import time
import json
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
camera_stats = {
    "people_count": 0,
    "density": 0.0,
    "density_level": "Low",
    "alert_status": "Safe",
    "fps": 0
}

# Setup directories
UPLOAD_FOLDER = os.path.join('static', 'uploads')
PROCESSED_FOLDER = os.path.join('static', 'processed')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory=os.path.join(os.getcwd(), "templates"))

# Initialize YOLO with optimized parameters for better accuracy
model_path = 'yolov8n'
yolo_infer = YOLOInference(model_path=model_path)  # Use YOLOv8n with improved settings

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
    return templates.TemplateResponse("live_camera.html", {"request": request})

@app.post("/start_camera")
async def start_camera(request: Request):
    global camera_active, camera_thread, camera_area_sqm

    if camera_active:
        return {"status": "Camera already active"}

    # Get area from request body
    try:
        data = await request.json()
        area_sqm = data.get("area_sqm", 50.0)
        if area_sqm <= 0:
            return {"status": "Invalid area value"}
        camera_area_sqm = float(area_sqm)
    except:
        camera_area_sqm = 50.0  # Default if no area provided

    camera_active = True
    camera_thread = threading.Thread(target=run_camera_processing, daemon=True)
    camera_thread.start()

    return {"status": "Camera started successfully"}

@app.post("/stop_camera")
async def stop_camera():
    global camera_active
    camera_active = False
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
    return {"heatmap_enabled": True}

def run_camera_processing():
    """FIXED: Run camera processing with real YOLO model"""
    global current_camera_frame, camera_stats, camera_active, frame_lock, model
    
    # Initialize YOLO model inside thread (thread-safe approach)
    try:
        print("Loading YOLO model...")
        # Update this path to your actual model path
        model_path = 'yolov8n'  # Faster model for better FPS
        # model_path = 'yolo11x'  # Or use pre-trained model
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = YOLO(model_path)
        model.to(device)
        print(f"YOLO model loaded successfully on {device}")
        
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        print("Falling back to pre-trained YOLO11n model...")
        try:
            model = YOLO("yolo11n")  # Download and use YOLO11n
            print("YOLO11n model loaded successfully")
        except Exception as e2:
            print(f"Error loading YOLO11n model: {e2}")
            print("Falling back to YOLOv8n...")
            try:
                model = YOLO("yolov8n.pt")  # Final fallback
                print("YOLOv8n fallback model loaded successfully")
            except Exception as e3:
                print(f"Error loading final fallback model: {e3}")
                camera_active = False
                return
    
    # Initialize camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open camera")
        camera_active = False
        return

    # Let camera use native resolution for correct aspect ratio
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
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
    last_density = 0.0
    estimated_camera_area = None  # For dynamic estimation if not user-provided
    user_provided_area = camera_area_sqm != 50.0  # Assume 50.0 is default

    while camera_active:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read from camera")
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
                results = model(frame, conf=0.4, device=model.device, verbose=False, imgsz=416, max_det=50, iou=0.5)

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
            camera_stats["density"] = density
            camera_stats["density_level"] = yolo_infer.get_density_level(density)
            camera_stats["alert_status"] = yolo_infer.get_alert_status(density, smoothed_count)
            
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
            with frame_lock:
                current_camera_frame = frame
        
        time.sleep(0.04)

    cap.release()
    print("Camera processing stopped")

@app.post("/upload")
async def upload(video: UploadFile = File(...)):
    if not video.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    video_path = os.path.join(UPLOAD_FOLDER, video.filename)
    
    # Save uploaded file
    with open(video_path, "wb") as buffer:
        content = await video.read()
        buffer.write(content)
    
    processed_filename = f"processed_{video.filename}"
    processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)
    
    # Start processing in background thread
    threading.Thread(
        target=yolo_infer.process_video,
        args=(video_path, processed_path),
        daemon=True
    ).start()
    
    return {"message": "File uploaded successfully"}

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

    # Check if it's an image
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    ext = os.path.splitext(image.filename)[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Invalid file type. Only image files are allowed.")

    image_path = os.path.join(UPLOAD_FOLDER, image.filename)

    # Save uploaded file
    with open(image_path, "wb") as buffer:
        content = await image.read()
        buffer.write(content)

    # Process the image with area-based analysis
    processed_filename = f"processed_{image.filename}"
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
    return templates.TemplateResponse("live_preview.html", {"request": request})

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
        target=yolo_infer.process_video,
        args=(input_video_path, output_video_path),
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

        # Check file type and size
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.mp4', '.mov', '.avi'}
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in allowed_extensions:
            continue

        file_path = os.path.join(UPLOAD_FOLDER, f"batch_{file.filename}")
        processed_filename = f"batch_processed_{file.filename}"
        processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)

        # Save uploaded file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        try:
            if file.content_type.startswith('image/'):
                # Process image
                result = process_image_batch(file_path, processed_path, personal_space, scale_factor)
            elif file.content_type.startswith('video/'):
                # Process video (extract first frame for demo)
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