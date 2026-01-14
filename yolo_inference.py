import cv2
import torch
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import threading
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from room_capacity import estimate_room_capacity

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
CROWD_THRESHOLD = 30  # For Discord alerts (overall count threshold)
ALERT_DELAY = 15  # Minimum seconds between alerts
MOTION_THRESHOLD = 5000  # For background subtraction
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1334259674580516979/pH92tTp_wnYG2a5j6KNgPHHHmbQRC7Hs8L01KANDKiQrw7iE4jPa6iuWqauLY1G6DqoD"
SNAPSHOT_FOLDER = "snapshots"
os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)

ROLLING_WINDOW = 30  # For smoothing crowd count
ENABLE_CROWD_PREDICTION = True
PREDICT_FRAMES_AHEAD = 30  # Not used in grid analysis

# --- Density calculation settings ---
PIXELS_PER_SQM = 25000  # Pixels per square meter - calibrate based on camera setup

# --- Real-world object sizes for scale estimation (in meters) ---
OBJECT_SIZES = {
    0: 1.7,   # person height
    2: 1.8,   # car width
    3: 1.5,   # motorcycle width
    5: 3.0,   # bus width
    7: 2.0,   # truck width
    39: 0.3,  # bottle height
    56: 0.5,  # chair width
    60: 1.2,  # dining table width
    62: 0.8,  # tv width
    63: 0.4,  # laptop width
    67: 0.15, # cell phone height
    # Add more as needed for better scale estimation
}

# --- Grid-based crowd analysis settings ---
ENABLE_GRID_ANALYSIS = True
NUM_GRID_ROWS = 6  # Grid rows
NUM_GRID_COLS = 6  # Grid cols
GRID_CELL_THRESHOLD = 3  # If cell count >= this, mark it

# DeepSort tracker
tracker = DeepSort(max_age=30, embedder="mobilenet")


class YOLOInference:
    def __init__(self, model_path="runs/detect/yolo11x_head12/weights/best.pt"):
        """Initialize YOLOv11 model and other settings."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_path = os.path.basename(model_path) if os.path.isabs(model_path) else model_path
        try:
            self.model = YOLO(model_path).to(self.device)
            print(f"[INFO] YOLOv11 model loaded on device: {self.device}")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            print("Falling back to pre-trained YOLOv8n model...")
            try:
                self.model = YOLO("yolov8n.pt").to(self.device)
                print("Fallback model loaded successfully")
            except Exception as e2:
                print(f"Error loading fallback model: {e2}")
                self.model = None

        self.last_alert_time = 0
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
        self.density_map = None
        self.latest_frame = None

        self.frame_indices = []
        self.crowd_counts = []

        # Heatmap toggle
        self.enable_heat_map = False
        self.density_map = None
        self.latest_frame = None
        self.frame_lock = threading.Lock()

        # Grid-based zooming
        self.zoom_row = None
        self.zoom_col = None

        # Store last processed overlay frame
        self.last_processed_overlay = None

        # Heatmap
        self.enable_heat_map = False
        self.density_map = None

        # Threading lock
        self.frame_lock = threading.Lock()

    def set_heatmap_enabled(self, state: bool):
        self.enable_heat_map = state
        if not state: # Reset density map when turned off
            if self.density_map is not None:
                self.density_map.fill(0)
        print(f"[INFO] Heatmap enabled: {self.enable_heat_map}")

    def set_zoom_cell(self, row: int, col: int):
        if row < 0 or col < 0:
            self.zoom_row, self.zoom_col = None, None
            print("[INFO] Zoom reset.")
        else:
            self.zoom_row, self.zoom_col = row, col
            print(f"[INFO] Zoom cell set to row={row}, col={col}")

    def get_zoomed_subimage(self):
        """Returns a magnified subimage of the latest frame."""
        with self.frame_lock:
            if self.latest_frame is None or self.zoom_row is None or self.zoom_col is None:
                return np.zeros((240, 320, 3), dtype=np.uint8) # Return a blank image
        
        height, width = self.latest_frame.shape[:2]
        cell_height = height // self.NUM_GRID_ROWS
        cell_width = width // self.NUM_GRID_COLS
        
        start_y, end_y = self.zoom_row * cell_height, (self.zoom_row + 1) * cell_height
        start_x, end_x = self.zoom_col * cell_width, (self.zoom_col + 1) * cell_width
        
        subimg = self.latest_frame[start_y:end_y, start_x:end_x]
        return cv2.resize(subimg, (width // 2, height // 2), interpolation=cv2.INTER_LINEAR)


    def _draw_overlays(self, frame, people_count, density):
        """Helper function to draw stats and optional heatmap/grid."""
        # Draw heatmap if enabled
        if self.enable_heat_map and self.density_map is not None:
            heatmap_viz = cv2.applyColorMap(
                cv2.normalize(self.density_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            frame = cv2.addWeighted(frame, 0.6, heatmap_viz, 0.4, 0)
            # Decay the heatmap over time
            self.density_map *= 0.97

        # Calculate density and alert
        density_level = self.get_density_level(density)
        alert_status = self.get_alert_status(density, people_count)

        # Draw stats
        cv2.putText(frame, f"People Count: {int(people_count)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Density: {density:.3f} p/sqm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Density Level: {density_level}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Alert: {alert_status}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame


    def set_zoom_cell(self, row: int, col: int):
        """Set the grid cell (row, col) to magnify; if negative, reset."""
        if row < 0 or col < 0:
            self.zoom_row = None
            self.zoom_col = None
            print("[INFO] Zoom reset (no cell selected).")
        else:
            self.zoom_row = row
            self.zoom_col = col
            print(f"[INFO] Zoom cell set to row={row}, col={col}")
    
    def get_zoomed_subimage(self):
        """Returns a zoomed subimage of the latest frame based on zoom_row and zoom_col."""
        if self.latest_frame is None or self.zoom_row is None or self.zoom_col is None:
            return None
            
        height, width = self.latest_frame.shape[:2]
        cell_height = height // NUM_GRID_ROWS
        cell_width = width // NUM_GRID_COLS
        
        # Calculate cell boundaries
        start_y = self.zoom_row * cell_height
        end_y = start_y + cell_height
        start_x = self.zoom_col * cell_width
        end_x = start_x + cell_width
        
        # Extract the subimage
        subimg = self.latest_frame[start_y:end_y, start_x:end_x].copy()
        
        # Resize to make it larger (optional)
        subimg = cv2.resize(subimg, (width // 2, height // 2))
        
        return subimg

    def process_video(self, input_path, output_path):
        """Processes the input video using YOLOv11 and DeepSort."""
        print(f"[INFO] Opening video: {input_path}")
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25
            print(f"[WARN] Invalid FPS detected. Defaulting to: {fps} fps")

        self.density_map = np.zeros((height, width), dtype=np.float32)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        estimated_area_sqm = None
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] No more frames. Processing complete.")
                break

            frame_count += 1

            # Run YOLO detection with optimized parameters for better accuracy
            results = self.model(frame, conf=0.4, device=self.device, imgsz=416, max_det=50, iou=0.5)
            detections = []

            for result in results:
                for box in result.boxes.data.cpu().numpy():
                    x1, y1, x2, y2, conf, class_id = box
                    if int(class_id) == 0:  # Person class
                        w, h = x2 - x1, y2 - y1
                        detections.append(([x1, y1, w, h], conf, "person"))

            # Estimate area from first frame detections
            if estimated_area_sqm is None and detections:
                estimated_area_sqm = self.estimate_area_from_detections(width, height, detections)
                print(f"[INFO] Estimated area from detections: {estimated_area_sqm} sqm")

            # Update DeepSort Tracker
            tracks = tracker.update_tracks(detections, frame=frame)

            # Count confirmed tracks for accurate people count
            people_count = sum(1 for track in tracks if track.is_confirmed())

            # Implement smoothing
            self.crowd_counts.append(people_count)
            if len(self.crowd_counts) > ROLLING_WINDOW:
                self.crowd_counts.pop(0)
            smoothed_count = int(sum(self.crowd_counts) / len(self.crowd_counts)) if self.crowd_counts else 0

            # Calculate density
            if estimated_area_sqm is None:
                estimated_area_sqm = (width * height) / PIXELS_PER_SQM
            density = smoothed_count / estimated_area_sqm if estimated_area_sqm > 0 else 0

            # Draw bounding boxes & track IDs
            for track in tracks:
                if not track.is_confirmed():
                    continue
                x1, y1, x2, y2 = map(int, track.to_tlbr())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"ID {track.track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Update density map
            for track in tracks:
                if not track.is_confirmed():
                    continue
                x1, y1, x2, y2 = map(int, track.to_tlbr())
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(self.density_map, (cx, cy), 25, (1.0,), thickness=-1)

            # Draw overlays (stats, heatmap)
            frame = self._draw_overlays(frame, smoothed_count, density)

            # Write frame
            out.write(frame)
            self.latest_frame = frame.copy()

        cap.release()
        out.release()
        print(f"[INFO] Finished processing. Output saved to: {output_path}")

    def process_image(self, image_path, output_path=None):
        """Processes a single image using YOLOv11 and returns stats and optionally saves processed image."""
        print(f"[INFO] Processing image: {image_path}")
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"[ERROR] Could not load image: {image_path}")
            return {"error": "Could not load image"}

        height, width = frame.shape[:2]
        self.density_map = np.zeros((height, width), dtype=np.float32)

        # Run YOLO detection with optimized parameters for better accuracy
        results = self.model(frame, conf=0.4, device=self.device, imgsz=416, max_det=50, iou=0.5)
        detections = []
        people_count = 0

        for result in results:
            for box in result.boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, class_id = box
                if int(class_id) == 0:  # Person class
                    people_count += 1
                    w, h = x2 - x1, y2 - y1
                    detections.append(([x1, y1, w, h], conf, "person"))
                    # Draw bounding box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f'Person {conf:.2f}', (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Update density map
        for det in detections:
            x1, y1, w, h = det[0]
            cx, cy = int(x1 + w/2), int(y1 + h/2)
            cv2.circle(self.density_map, (cx, cy), 25, (1.0,), thickness=-1)

        # Overlay density heatmap if enabled
        if self.enable_heat_map:
            heatmap = cv2.applyColorMap(cv2.normalize(self.density_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET)
            frame = cv2.addWeighted(frame, 0.5, heatmap, 0.5, 0)

        # Draw stats
        cv2.putText(frame, f"People Count: {people_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Calculate approximate area in square meters using detections
        estimated_area_sqm = self.estimate_area_from_detections(width, height, detections, image_path)

        # Calculate density
        density = people_count / estimated_area_sqm if estimated_area_sqm > 0 else 0

        # Determine density level and alert status
        density_level = self.get_density_level(density)
        alert_status = self.get_alert_status(density, people_count)

        # Draw density on image
        cv2.putText(frame, f"Density: {density:.3f} p/sqm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Density Level: {density_level}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Alert: {alert_status}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Save processed image if output_path provided
        if output_path:
            cv2.imwrite(output_path, frame)
            print(f"[INFO] Processed image saved to: {output_path}")

        self.latest_frame = frame.copy()

        return {
            "people_count": people_count,
            "density": density,
            "density_level": density_level,
            "alert_status": alert_status,
            "estimated_area_sqm": estimated_area_sqm,
            "processed_image_path": output_path if output_path else None
        }

    def get_density_level(self, density):
        """Determine crowd density level based on people per square meter"""
        if density < 0.05:
            return "None"
        elif density < 0.2:
            return "Low"
        elif density < 0.5:
            return "Medium"
        elif density < 1.0:
            return "High"
        else:
            return "Critical"

    def get_alert_status(self, density, people_count=0):
        """Determine alert status based on density and people count for enhanced risk analysis"""
        # Enhanced risk analysis considering both density and absolute count
        risk_score = density * 10 + (people_count / 10)  # Weighted combination

        if risk_score < 5:
            return "Safe"
        elif risk_score < 15:
            return "Caution"
        elif risk_score < 25:
            return "Warning"
        else:
            return "Critical Alert"

    def calculate_approximate_area(self, width, height):
        """Calculate approximate area in square meters based on image dimensions.
        Uses configurable PIXELS_PER_SQM for camera calibration."""
        pixel_area = width * height
        estimated_area = pixel_area / PIXELS_PER_SQM
        return round(estimated_area, 2)

    def estimate_area_from_detections(self, width, height, detections, image_path=None):
        """Estimate area using MiDaS depth estimation if image_path provided, else fallback to detected objects."""
        if image_path:
            try:
                result = estimate_room_capacity(image_path)
                area = result["estimated_area"]
                print(f"[DEBUG] MiDaS estimated area: {area:.2f} sqm")
                return area
            except Exception as e:
                print(f"[WARN] MiDaS estimation failed: {e}, falling back to detection-based method")

        # Fallback to original method
        if not detections:
            fallback_area = self.calculate_approximate_area(width, height)
            print(f"[DEBUG] No detections available, using fallback area: {fallback_area} sqm")
            return fallback_area

        # Collect dimensions for objects with known sizes
        scale_measurements = []
        person_heights = []
        for det in detections:
            if len(det[0]) == 4:  # [x1, y1, w, h]
                x1, y1, w, h = det[0]
                raw_class_id = det[2] if len(det) > 2 else 0  # Default to person if not specified
                
                # Handle both string "person" and integer class IDs
                if isinstance(raw_class_id, str):
                    class_id = 0 if raw_class_id.lower() == "person" else -1
                else:
                    class_id = int(raw_class_id)

                if class_id in OBJECT_SIZES:
                    real_size_m = OBJECT_SIZES[class_id]
                    if class_id == 0:  # Person - use height for better accuracy
                        pixel_size = h  # Use height in pixels for persons
                        pixels_per_meter = pixel_size / real_size_m
                        person_heights.append(pixels_per_meter)
                        print(f"[DEBUG] Person detection: w={w:.1f}, h={h:.1f}, pixels_per_meter={pixels_per_meter:.1f}")
                    else:
                        # For other objects, use larger dimension
                        pixel_size = max(w, h)
                        pixels_per_meter = pixel_size / real_size_m
                        scale_measurements.append(pixels_per_meter)
                        print(f"[DEBUG] Object class {class_id}: w={w:.1f}, h={h:.1f}, pixel_size={pixel_size:.1f}, pixels_per_meter={pixels_per_meter:.1f}")

        # Prioritize person heights for more accurate estimation
        if person_heights:
            # Use median to reduce outliers
            avg_pixels_per_meter = sorted(person_heights)[len(person_heights)//2]
            print(f"[DEBUG] Using median person height: {avg_pixels_per_meter:.1f} pixels/m from {len(person_heights)} detections")
        elif scale_measurements:
            avg_pixels_per_meter = sum(scale_measurements) / len(scale_measurements)
            print(f"[DEBUG] Using average from other objects: {avg_pixels_per_meter:.1f} pixels/m from {len(scale_measurements)} detections")
        else:
            fallback_area = self.calculate_approximate_area(width, height)
            print(f"[DEBUG] No usable scale measurements, using fallback area: {fallback_area} sqm")
            return fallback_area

        pixels_per_sqm = avg_pixels_per_meter ** 2
        pixel_area = width * height
        estimated_area = pixel_area / pixels_per_sqm

        print(f"[DEBUG] Final estimation: {width}x{height}={pixel_area} pixels, {pixels_per_sqm:.0f} pixels/sqm, area={estimated_area:.2f} sqm")
        return round(estimated_area, 2)


if __name__ == "__main__":
    yolo_infer = YOLOInference("yolo11x.pt")
    yolo_infer.process_video("input_video.mp4", "output_video.mp4")