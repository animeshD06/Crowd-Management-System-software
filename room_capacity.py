import torch
import cv2
import numpy as np
from skimage import measure
import os

# Load MiDaS model globally
print("Loading MiDaS model...")
try:
    model_type = "MiDaS_small"
    midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    transform = midas_transforms.small_transform
    print("MiDaS model loaded successfully")
except Exception as e:
    print(f"Failed to load MiDaS model: {e}")
    midas = None
    transform = None

def estimate_room_capacity(image_path: str) -> dict:
    """
    Estimate room capacity using MiDaS depth estimation.

    Args:
        image_path (str): Path to the input image.

    Returns:
        dict: Contains 'estimated_area' (float), 'maximum_people' (int),
              'depth_map_path' (str), 'floor_overlay_path' (str).
    """
    if midas is None or transform is None:
        raise RuntimeError("MiDaS model not loaded")

    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_batch = transform(img_rgb).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    # Normalize depth map to 0-1
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

    # Detect floor: assume floor is areas with depth > 0.6 (farthest parts)
    floor_mask = depth_map > 0.6

    # Find largest connected component as floor
    labels = measure.label(floor_mask)
    if labels.max() == 0:
        raise ValueError("No floor detected in the image")
    props = measure.regionprops(labels)
    largest = max(props, key=lambda x: x.area)
    floor_mask = labels == largest.label

    # Scale depth to meters, assuming max depth corresponds to camera height 1.6m
    scale = 1.6 / depth_map.max()
    depth_map_m = depth_map * scale

    # Camera parameters (assumed)
    focal_length = 0.005  # meters
    pixel_size = 1e-6  # meters
    angular_size = pixel_size / focal_length

    # Calculate area
    floor_depths = depth_map_m[floor_mask]
    area_per_pixel = (floor_depths * angular_size) ** 2
    total_area = np.sum(area_per_pixel)

    # Calculate maximum people (2 per m²)
    max_people = int(total_area * 2)

    # Create visualizations
    # Depth map visualization
    depth_vis = (depth_map * 255).astype(np.uint8)
    base_name = os.path.splitext(image_path)[0]
    depth_path = f"{base_name}_depth.png"
    cv2.imwrite(depth_path, depth_vis)

    # Floor segmentation overlay
    overlay = img.copy()
    overlay[floor_mask] = [0, 255, 0]  # Green overlay for floor
    overlay_path = f"{base_name}_floor_overlay.png"
    cv2.imwrite(overlay_path, overlay)

    return {
        "estimated_area": total_area,
        "maximum_people": max_people,
        "depth_map_path": depth_path,
        "floor_overlay_path": overlay_path
    }