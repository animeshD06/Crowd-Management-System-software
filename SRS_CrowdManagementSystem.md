# Software Requirements Specification

## Crowd Management System (CMS)

**Version 1.0**

**Prepared by:** [Project Team Name]

**Organization:** [Organization Name]

**Date:** January 8, 2026

**Document Status:** Preliminary

---

## Revision History

| Version | Date | Description | Author |
|---------|------|-------------|--------|
| 1.0 | January 8, 2026 | Initial Draft - Preliminary Stage | [Team Name] |

---

## Table of Contents

1. [Introduction](#1-introduction)
   - 1.1 Purpose
   - 1.2 Scope
   - 1.3 Definitions, Acronyms, and Abbreviations
   - 1.4 References
   - 1.5 Overview
2. [Overall Description](#2-overall-description)
   - 2.1 Product Perspective
   - 2.2 Product Features
   - 2.3 User Classes and Characteristics
   - 2.4 Operating Environment
   - 2.5 Design and Implementation Constraints
   - 2.6 Assumptions and Dependencies
3. [Specific Requirements](#3-specific-requirements)
   - 3.1 External Interface Requirements
   - 3.2 Functional Requirements
   - 3.3 Non-Functional Requirements
4. [System Features](#4-system-features)
   - 4.1 Real-time AI Surveillance
   - 4.2 Video Upload and Analysis
   - 4.3 Image Analysis
   - 4.4 Heatmap Visualization
   - 4.5 Safety Alert System
   - 4.6 Grid-Based Crowd Analysis
5. [Other Requirements](#5-other-requirements)
6. [Appendices](#6-appendices)

---

# 1. Introduction

## 1.1 Purpose

This Software Requirements Specification (SRS) document provides a comprehensive description of the requirements for the Crowd Management System (CMS). This document is intended for internal development teams, project stakeholders, quality assurance teams, and potential investors or evaluators.

The purpose of this document is to:
- Define the functional and non-functional requirements of the CMS
- Serve as a basis for project planning and cost estimation
- Provide a reference for validation and verification activities
- Establish agreements between stakeholders and development team

**Document Status:** This document reflects the preliminary stage of development and will be updated as the project evolves.

## 1.2 Scope

The Crowd Management System (CMS) is an advanced AI-powered web application designed to enhance safety, efficiency, and decision-making in events and public spaces. The system leverages state-of-the-art YOLO11x object detection for real-time crowd monitoring and analysis.

### 1.2.1 In-Scope Features

The CMS will provide the following capabilities:

1. **Real-time Crowd Surveillance** - Live video feed analysis with person detection and tracking
2. **Video Upload & Analysis** - Batch processing of uploaded video files for crowd analysis
3. **Image Analysis** - Single image processing with crowd density assessment
4. **Heatmap Visualization** - Dynamic visual representation of crowd density patterns
5. **Safety Alerts** - Automated notifications for overcrowding conditions
6. **Crowd Analytics** - Statistical analysis of crowd behavior and movement trends
7. **Grid-Based Analysis** - Spatial segmentation for detailed crowd distribution
8. **Interactive Dashboard** - Web-based interface for monitoring and control

### 1.2.2 Out-of-Scope Features (Future Phases)

The following features are planned for future development phases:
- Multi-camera network coordination
- Predictive crowd flow modeling
- Mobile application interface
- Integration with emergency response systems
- Historical data analytics and reporting
- Edge computing deployment

### 1.2.3 Benefits

- **Enhanced Public Safety** - Early detection of dangerous crowd densities
- **Improved Event Management** - Data-driven crowd control decisions
- **Reduced Response Time** - Automated alerts for rapid intervention
- **Cost Efficiency** - Reduced need for manual surveillance personnel

## 1.3 Definitions, Acronyms, and Abbreviations

| Term | Definition |
|------|------------|
| CMS | Crowd Management System |
| YOLO | You Only Look Once - Real-time object detection algorithm |
| DeepSort | Deep Simple Online and Realtime Tracking algorithm |
| API | Application Programming Interface |
| MJPEG | Motion JPEG - Video compression format |
| FPS | Frames Per Second |
| GPU | Graphics Processing Unit |
| CPU | Central Processing Unit |
| CUDA | Compute Unified Device Architecture (NVIDIA) |
| CORS | Cross-Origin Resource Sharing |
| mAP | Mean Average Precision |
| SRS | Software Requirements Specification |
| UI | User Interface |
| REST | Representational State Transfer |
| ASGI | Asynchronous Server Gateway Interface |
| Density | Number of persons per unit area (persons/m²) |
| Heatmap | Visual representation of data density using color gradients |
| Threshold | Predefined limit triggering specific actions |
| Bounding Box | Rectangle drawn around detected objects |

## 1.4 References

| Reference | Description | Version |
|-----------|-------------|---------|
| IEEE 830-1998 | IEEE Recommended Practice for Software Requirements Specifications | 1998 |
| Ultralytics YOLO11 Documentation | Object detection model documentation | Latest |
| FastAPI Documentation | Web framework documentation | 0.115+ |
| OpenCV Documentation | Computer vision library documentation | 4.7.0 |
| DeepSort Paper | "Simple Online and Realtime Tracking with a Deep Association Metric" | 2017 |

## 1.5 Overview

This document is organized according to IEEE 830-1998 standard:

- **Section 2** provides an overall description of the product including context, main features, user characteristics, constraints, and dependencies.
- **Section 3** details specific functional and non-functional requirements.
- **Section 4** describes detailed system features with use cases.
- **Section 5** covers additional requirements.
- **Section 6** contains appendices with supplementary information.

---

# 2. Overall Description

## 2.1 Product Perspective

### 2.1.1 System Context

The Crowd Management System is a standalone web-based application that interfaces with:

1. **Camera Input Sources** - Webcams, IP cameras, or RTSP streams
2. **File System** - For video/image uploads and processed outputs
3. **Web Browsers** - For user interaction via the dashboard
4. **External Notification Services** - Discord webhooks for alerts (optional)

```
┌─────────────────────────────────────────────────────────┐
│                    EXTERNAL SYSTEMS                      │
├──────────────┬───────────────┬───────────────┬──────────┤
│   Camera     │   File        │   Web         │ Discord  │
│   Devices    │   Storage     │   Browser     │ Webhook  │
└──────┬───────┴───────┬───────┴───────┬───────┴────┬─────┘
       │               │               │            │
       ▼               ▼               ▼            ▼
┌─────────────────────────────────────────────────────────┐
│              CROWD MANAGEMENT SYSTEM                     │
│  ┌─────────────────────────────────────────────────┐    │
│  │     FastAPI Backend + YOLO AI Engine            │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

### 2.1.2 System Interfaces

| Interface | Type | Description |
|-----------|------|-------------|
| Video Input | Hardware | USB webcam, IP camera, video files |
| Web Interface | Software | HTTP/HTTPS REST API endpoints |
| File System | Software | Local or cloud storage for media files |
| Notification | Software | Discord webhook API for alerts |

## 2.2 Product Features

The CMS provides the following major features:

### Feature Summary

| Feature ID | Feature Name | Priority | Status |
|------------|--------------|----------|--------|
| F-001 | Real-time AI Surveillance | High | In Development |
| F-002 | Video Upload & Analysis | High | In Development |
| F-003 | Image Analysis | Medium | In Development |
| F-004 | Heatmap Visualization | High | In Development |
| F-005 | Safety Alert System | High | Planned |
| F-006 | Grid-Based Analysis | Medium | In Development |
| F-007 | Zoom/Magnification | Low | In Development |
| F-008 | Batch Processing | Low | Planned |

## 2.3 User Classes and Characteristics

### 2.3.1 Primary Users

| User Class | Description | Technical Expertise | Frequency of Use |
|------------|-------------|---------------------|------------------|
| Security Personnel | Monitors live feeds for crowd safety | Low to Medium | Continuous |
| Event Managers | Plans and manages crowd flow | Medium | Daily/Event-based |
| System Administrators | Configures and maintains the system | High | Weekly |
| Emergency Responders | Receives alerts and coordinates response | Low | As needed |

### 2.3.2 User Characteristics

1. **Security Personnel**
   - Require simple, intuitive interface
   - Need real-time visual feedback
   - Must receive clear, actionable alerts
   - May work in high-stress environments

2. **Event Managers**
   - Need access to analytics and reports
   - Require historical data analysis
   - Use system for planning and optimization
   - Prefer dashboard-style interfaces

3. **System Administrators**
   - Require access to configuration options
   - Need ability to adjust detection parameters
   - Must manage system performance
   - Responsible for maintenance and updates

## 2.4 Operating Environment

### 2.4.1 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 |
| RAM | 8 GB | 16 GB or higher |
| GPU | Not required (CPU mode) | NVIDIA GTX 1060+ with CUDA |
| Storage | 10 GB free | 50 GB SSD |
| Network | 10 Mbps | 100 Mbps |
| Webcam | 720p USB webcam | 1080p IP camera |

### 2.4.2 Software Requirements

| Component | Requirement |
|-----------|-------------|
| Operating System | Windows 10/11, Linux (Ubuntu 20.04+), macOS 12+ |
| Python | Version 3.10 or higher |
| Web Browser | Chrome 90+, Firefox 90+, Edge 90+, Safari 15+ |
| CUDA (Optional) | Version 11.8 or 12.1 for GPU acceleration |

### 2.4.3 Network Requirements

- HTTP/HTTPS access for web interface
- Port 8000 (default) for API server
- Outbound internet access for Discord notifications (optional)

## 2.5 Design and Implementation Constraints

### 2.5.1 Technology Constraints

1. **AI Model Dependency** - System relies on YOLO11 model which requires PyTorch framework
2. **Real-time Processing** - Video streaming requires efficient frame processing
3. **Browser Compatibility** - MJPEG streaming must be supported by client browser
4. **GPU Availability** - Optimal performance requires NVIDIA GPU with CUDA support

### 2.5.2 Regulatory Constraints

1. **Privacy Regulations** - Must comply with local privacy laws regarding video surveillance
2. **Data Retention** - Stored video/images must follow data retention policies
3. **Consent Requirements** - Public notices may be required for surveillance areas

### 2.5.3 Development Constraints

1. **Open Source Dependencies** - System uses open-source libraries with various licenses
2. **Model Training Data** - Custom model trained on 2000+ annotated crowd images
3. **Real-time Performance** - Must maintain minimum 15 FPS for effective monitoring

## 2.6 Assumptions and Dependencies

### 2.6.1 Assumptions

| ID | Assumption |
|----|------------|
| A-001 | Users have access to a modern web browser |
| A-002 | Camera devices are properly configured and accessible |
| A-003 | Network connectivity is stable and reliable |
| A-004 | Users have basic computer literacy |
| A-005 | Server hardware meets minimum specifications |

### 2.6.2 Dependencies

| ID | Dependency | Impact |
|----|------------|--------|
| D-001 | PyTorch framework | Critical - Required for AI inference |
| D-002 | OpenCV library | Critical - Required for video processing |
| D-003 | Ultralytics YOLO | Critical - Required for object detection |
| D-004 | FastAPI framework | Critical - Required for web server |
| D-005 | DeepSort library | Medium - Required for object tracking |
| D-006 | CUDA toolkit | Optional - Required for GPU acceleration |

---

# 3. Specific Requirements

## 3.1 External Interface Requirements

### 3.1.1 User Interfaces

**UI-001: Main Dashboard**
- **Description:** Central hub displaying system status, feature access, and video upload capabilities
- **Requirements:**
  - Display system status and health indicators
  - Provide navigation to all major features
  - Support drag-and-drop file upload
  - Responsive design for various screen sizes
  - Dark theme with modern aesthetics

**UI-002: Live Camera Interface**
- **Description:** Real-time video feed display with overlays and controls
- **Requirements:**
  - Display live video stream with detection overlays
  - Show real-time statistics (person count, density)
  - Provide toggle controls for heatmap and analysis features
  - Support zoom/magnification functionality
  - Display alert status and notifications

**UI-003: Video Processing Interface**
- **Description:** Interface for viewing uploaded video analysis results
- **Requirements:**
  - Display processed video with annotations
  - Show frame-by-frame statistics
  - Provide playback controls
  - Display density heatmaps

### 3.1.2 Hardware Interfaces

**HW-001: Camera Interface**
- **Description:** Interface for video input devices
- **Requirements:**
  - Support USB webcam capture via OpenCV
  - Handle various resolutions (480p to 1080p)
  - Process at minimum 15 FPS input rate

**HW-002: GPU Interface (Optional)**
- **Description:** Interface for GPU acceleration
- **Requirements:**
  - Detect and utilize NVIDIA CUDA-enabled GPUs
  - Fallback to CPU processing when GPU unavailable
  - Support PyTorch CUDA operations

### 3.1.3 Software Interfaces

**SW-001: REST API Interface**

| Endpoint | Method | Description |
|----------|--------|-------------|
| / | GET | Main dashboard page |
| /live_camera | GET | Live camera interface |
| /start_camera | POST | Initialize camera processing |
| /stop_camera | POST | Terminate camera processing |
| /camera_feed | GET | MJPEG video stream |
| /camera_stats | GET | Real-time statistics JSON |
| /upload | POST | Video file upload |
| /upload_image | POST | Image file upload |
| /live_preview | GET | Video processing preview |
| /video_feed | GET | Processed video stream |
| /set_zoom | GET | Set grid cell for zoom |
| /toggle_heatmap | POST | Toggle heatmap display |

**SW-002: File System Interface**
- **Input Formats:** MP4, AVI, MOV, MKV (video); JPG, PNG, BMP (image)
- **Output Formats:** MP4 (processed video); JPG (processed images)
- **Storage Locations:** /static/uploads/, /static/processed/, /snapshots/

### 3.1.4 Communication Interfaces

**CI-001: HTTP Protocol**
- Protocol: HTTP/1.1 and HTTP/2
- Port: 8000 (default, configurable)
- Encoding: UTF-8

**CI-002: Streaming Protocol**
- Format: MJPEG (Motion JPEG)
- Content-Type: multipart/x-mixed-replace

**CI-003: Discord Webhook (Optional)**
- Protocol: HTTPS POST
- Format: JSON payload with alert details

## 3.2 Functional Requirements

### 3.2.1 Video Processing Requirements

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| FR-VP-001 | System shall capture video from webcam at minimum 15 FPS | High | Pending |
| FR-VP-002 | System shall process uploaded video files up to 500MB | Medium | Pending |
| FR-VP-003 | System shall generate annotated output video with bounding boxes | High | Pending |
| FR-VP-004 | System shall support common video formats (MP4, AVI, MOV) | Medium | Pending |
| FR-VP-005 | System shall display real-time video stream in web browser | High | Pending |

### 3.2.2 AI Detection Requirements

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| FR-AI-001 | System shall detect persons in video frames using YOLO11 model | High | Pending |
| FR-AI-002 | System shall achieve minimum 80% detection accuracy (mAP@0.5) | High | Pending |
| FR-AI-003 | System shall track individuals across video frames using DeepSort | Medium | Pending |
| FR-AI-004 | System shall display bounding boxes around detected persons | High | Pending |
| FR-AI-005 | System shall assign unique IDs to tracked individuals | Medium | Pending |

### 3.2.3 Crowd Analysis Requirements

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| FR-CA-001 | System shall calculate crowd density (persons per square meter) | High | Pending |
| FR-CA-002 | System shall classify density levels (Low, Moderate, High, Critical) | High | Pending |
| FR-CA-003 | System shall generate crowd density heatmaps | Medium | Pending |
| FR-CA-004 | System shall perform grid-based spatial analysis (6x6 grid) | Medium | Pending |
| FR-CA-005 | System shall identify high-density zones within the grid | Medium | Pending |

### 3.2.4 Alert Requirements

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| FR-AL-001 | System shall generate alerts when density exceeds threshold | High | Pending |
| FR-AL-002 | System shall display visual alert indicators on dashboard | High | Pending |
| FR-AL-003 | System shall support configurable density thresholds | Medium | Pending |
| FR-AL-004 | System shall send notifications via Discord webhook (optional) | Low | Pending |
| FR-AL-005 | System shall implement alert rate limiting to prevent flooding | Low | Pending |

### 3.2.5 User Interface Requirements

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| FR-UI-001 | System shall provide responsive web-based dashboard | High | Pending |
| FR-UI-002 | System shall support drag-and-drop file upload | Medium | Pending |
| FR-UI-003 | System shall display real-time statistics panel | High | Pending |
| FR-UI-004 | System shall provide heatmap toggle control | Medium | Pending |
| FR-UI-005 | System shall support grid cell zoom/magnification | Low | Pending |

## 3.3 Non-Functional Requirements

### 3.3.1 Performance Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-P-001 | Real-time video processing latency | < 200 ms |
| NFR-P-002 | Minimum frame rate for live stream | 15 FPS (CPU), 30 FPS (GPU) |
| NFR-P-003 | API response time | < 100 ms |
| NFR-P-004 | Maximum memory usage | < 4 GB (CPU mode) |
| NFR-P-005 | Video upload processing time | < 3x video duration |

### 3.3.2 Security Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| NFR-S-001 | System shall validate all file uploads for type and size | High |
| NFR-S-002 | System shall sanitize file paths to prevent traversal attacks | High |
| NFR-S-003 | System shall support HTTPS in production deployment | High |
| NFR-S-004 | System shall implement CORS for cross-origin protection | Medium |
| NFR-S-005 | System shall not store personally identifiable information | High |

### 3.3.3 Reliability Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-R-001 | System availability | 99% uptime |
| NFR-R-002 | Mean time between failures | > 72 hours |
| NFR-R-003 | Recovery from crash | < 30 seconds |
| NFR-R-004 | Graceful degradation on GPU failure | Automatic CPU fallback |

### 3.3.4 Usability Requirements

| ID | Requirement | Description |
|----|-------------|-------------|
| NFR-U-001 | Learnability | New users can start basic monitoring within 5 minutes |
| NFR-U-002 | Accessibility | UI complies with WCAG 2.1 Level A guidelines |
| NFR-U-003 | Feedback | System provides visual feedback for all user actions |
| NFR-U-004 | Error Messages | Clear, actionable error messages displayed to users |

### 3.3.5 Scalability Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-SC-001 | Concurrent users | Support 10-50 concurrent users |
| NFR-SC-002 | Video resolution | Support up to 1920x1080 input |
| NFR-SC-003 | Batch processing | Process up to 10 files concurrently |

### 3.3.6 Maintainability Requirements

| ID | Requirement | Description |
|----|-------------|-------------|
| NFR-M-001 | Modular architecture | Separation of frontend, backend, and AI components |
| NFR-M-002 | Documentation | Comprehensive code documentation and API docs |
| NFR-M-003 | Logging | Structured logging for debugging and monitoring |
| NFR-M-004 | Configuration | Externalized configuration for easy updates |

---

# 4. System Features

## 4.1 Real-time AI Surveillance

### 4.1.1 Description

Real-time AI surveillance enables live monitoring of crowd conditions through webcam or IP camera feeds. The system continuously analyzes video frames to detect individuals, track their movements, and assess crowd density.

### 4.1.2 Stimulus/Response Sequences

| Stimulus | Response |
|----------|----------|
| User navigates to Live Camera page | System displays camera interface with controls |
| User clicks "Start Camera" | System initializes camera capture and begins processing |
| Camera captures frame | System runs YOLO inference, updates detections |
| Detection results available | System draws bounding boxes and calculates statistics |
| Density exceeds threshold | System displays alert notification |
| User clicks "Stop Camera" | System terminates processing and releases camera |

### 4.1.3 Functional Requirements

- **F-001-01:** Initialize and capture video from local webcam
- **F-001-02:** Process frames at minimum 15 FPS
- **F-001-03:** Display annotated video stream in browser
- **F-001-04:** Show real-time person count and density
- **F-001-05:** Support start/stop controls for camera
- **F-001-06:** Implement smooth frame transitions

## 4.2 Video Upload and Analysis

### 4.2.1 Description

Video upload allows users to submit pre-recorded video files for offline crowd analysis. The system processes the entire video, applying detection and tracking algorithms to generate comprehensive analytics.

### 4.2.2 Stimulus/Response Sequences

| Stimulus | Response |
|----------|----------|
| User selects video file | System validates file type and size |
| User uploads video | System saves to uploads directory |
| Upload completes | System begins video processing |
| Processing progresses | System generates annotated output frames |
| Processing completes | System saves processed video and redirects to preview |
| User views preview | System streams processed video with statistics |

### 4.2.3 Functional Requirements

- **F-002-01:** Accept video uploads up to 500MB
- **F-002-02:** Support MP4, AVI, MOV, MKV formats
- **F-002-03:** Process video frame-by-frame with YOLO detection
- **F-002-04:** Generate annotated output video
- **F-002-05:** Provide progress indication during processing
- **F-002-06:** Store processed video for playback

## 4.3 Image Analysis

### 4.3.1 Description

Image analysis processes single images to provide crowd density assessment. Users can upload images for quick analysis without the overhead of video processing.

### 4.3.2 Stimulus/Response Sequences

| Stimulus | Response |
|----------|----------|
| User selects image file | System validates file format |
| User uploads image | System processes with YOLO detection |
| Processing completes | System returns annotated image with statistics |
| User views results | System displays bounding boxes and density info |

### 4.3.3 Functional Requirements

- **F-003-01:** Accept JPG, PNG, BMP image uploads
- **F-003-02:** Process image within 5 seconds
- **F-003-03:** Display detected person count
- **F-003-04:** Calculate and display density estimate
- **F-003-05:** Support region selection for targeted analysis
- **F-003-06:** Return processed image with annotations

## 4.4 Heatmap Visualization

### 4.4.1 Description

Heatmap visualization provides a color-coded overlay showing crowd density distribution across the monitored area. Hot zones indicate high-density regions requiring attention.

### 4.4.2 Stimulus/Response Sequences

| Stimulus | Response |
|----------|----------|
| User toggles heatmap on | System begins accumulating density data |
| Detection results processed | System updates heatmap based on person positions |
| Frame rendered | System overlays heatmap on video feed |
| User toggles heatmap off | System removes heatmap overlay |

### 4.4.3 Functional Requirements

- **F-004-01:** Generate real-time density heatmap overlay
- **F-004-02:** Use color gradient from green (low) to red (high)
- **F-004-03:** Toggle heatmap display on/off
- **F-004-04:** Update heatmap at frame rate
- **F-004-05:** Maintain heatmap persistence across frames

## 4.5 Safety Alert System

### 4.5.1 Description

The safety alert system monitors crowd density and triggers notifications when thresholds are exceeded. Alerts help operators take timely action to prevent dangerous situations.

### 4.5.2 Stimulus/Response Sequences

| Stimulus | Response |
|----------|----------|
| Density exceeds moderate threshold | System displays yellow warning indicator |
| Density exceeds high threshold | System displays orange alert indicator |
| Density exceeds critical threshold | System displays red critical alert, sends notification |
| Density returns to safe level | System clears alert indicators |

### 4.5.3 Functional Requirements

- **F-005-01:** Define configurable density thresholds
- **F-005-02:** Display color-coded alert indicators
- **F-005-03:** Send Discord webhook notifications (optional)
- **F-005-04:** Capture snapshot on critical alert
- **F-005-05:** Implement alert rate limiting

### 4.5.4 Alert Thresholds

| Level | Density (persons/m²) | Color | Action |
|-------|---------------------|-------|--------|
| Low | < 1.0 | Green | Normal operation |
| Moderate | 1.0 - 2.0 | Yellow | Monitor closely |
| High | 2.0 - 3.0 | Orange | Consider intervention |
| Critical | > 3.0 | Red | Immediate action required |

## 4.6 Grid-Based Crowd Analysis

### 4.6.1 Description

Grid-based analysis divides the video frame into a 6x6 grid for detailed spatial analysis. Each cell shows person count and density, enabling targeted monitoring of specific areas.

### 4.6.2 Stimulus/Response Sequences

| Stimulus | Response |
|----------|----------|
| Analysis enabled | System overlays 6x6 grid on video |
| Detection processed | System counts persons per cell |
| Cell exceeds threshold | System highlights cell with alert color |
| User clicks cell | System zooms to selected cell |

### 4.6.3 Functional Requirements

- **F-006-01:** Overlay 6x6 grid on video frame
- **F-006-02:** Count persons in each grid cell
- **F-006-03:** Highlight cells exceeding threshold
- **F-006-04:** Support cell-based zoom/magnification
- **F-006-05:** Display cell-level statistics

---

# 5. Other Requirements

## 5.1 Database Requirements

**Current Implementation:** File-based storage
- Uploaded files stored in `/static/uploads/`
- Processed outputs stored in `/static/processed/`
- Alert snapshots stored in `/snapshots/`

**Future Consideration:** PostgreSQL or MongoDB for:
- Session management
- Historical analytics storage
- User preferences and configuration

## 5.2 Internationalization Requirements

- Interface initially in English only
- Unicode support for text rendering
- Future support for multi-language UI

## 5.3 Legal Requirements

- Privacy policy for video surveillance
- User consent mechanisms
- Data retention compliance
- GDPR considerations for EU deployments

## 5.4 Installation and Deployment

### 5.4.1 Development Environment Setup

1. Clone repository
2. Create Python virtual environment
3. Install dependencies via pip or uv
4. Configure CUDA (optional, for GPU)
5. Run development server

### 5.4.2 Production Deployment

- Docker containerization recommended
- Nginx reverse proxy for HTTPS
- Process manager (systemd/supervisord)
- Log aggregation setup

---

# 6. Appendices

## Appendix A: Technology Stack

| Category | Technology | Version |
|----------|------------|---------|
| Backend Framework | FastAPI | 0.115+ |
| Web Server | Uvicorn | 0.34+ |
| AI Framework | PyTorch | 2.7+ |
| Object Detection | Ultralytics YOLO11 | 8.3+ |
| Object Tracking | DeepSort | 1.3+ |
| Computer Vision | OpenCV | 4.7.0 |
| Numerical Computing | NumPy | <2.0 |
| Visualization | Matplotlib | 3.10+ |
| Templating | Jinja2 | 3.1+ |
| Frontend | HTML5, CSS3, JavaScript | - |
| CSS Framework | Bootstrap 5 | 5.x |
| Animation | GSAP | Latest |

## Appendix B: AI Model Specifications

| Parameter | Value |
|-----------|-------|
| Base Model | YOLO11x (Extra Large) |
| Training Dataset | 2000+ annotated crowd images |
| Training Epochs | 100 |
| Batch Size | 4 |
| Training Image Size | 1024x1024 |
| Inference Image Size | 640x640 |
| Target Classes | Person (single class) |
| mAP@0.5 | 0.85 |
| Precision | 0.88 |
| Recall | 0.82 |

## Appendix C: API Response Formats

### Camera Statistics Response (GET /camera_stats)

```json
{
  "raw_count": 15,
  "smoothed_count": 14,
  "density": 1.25,
  "density_level": "Moderate",
  "fps": 25.5,
  "alert_status": {
    "level": "warning",
    "message": "Moderate crowd density detected"
  }
}
```

### Image Analysis Response (POST /upload_image)

```json
{
  "success": true,
  "processed_image": "/static/processed/processed_image.jpg",
  "people_count": 28,
  "density": 2.3,
  "density_level": "High",
  "area_sqm": 12.17,
  "alert_status": "high"
}
```

## Appendix D: Use Case Diagrams

### Primary Use Cases

```
                    ┌─────────────────────────────────────┐
                    │        Crowd Management System       │
                    └─────────────────────────────────────┘
                                      │
           ┌──────────────────────────┼──────────────────────────┐
           │                          │                          │
    ┌──────┴──────┐           ┌───────┴───────┐          ┌───────┴───────┐
    │ Monitor Live │           │ Analyze Video │          │ View Alerts   │
    │   Camera     │           │    Upload     │          │               │
    └──────────────┘           └───────────────┘          └───────────────┘
           │                          │                          │
    ┌──────┴──────┐           ┌───────┴───────┐          ┌───────┴───────┐
    │  View       │           │  Upload       │          │  Configure    │
    │  Heatmap    │           │   Image       │          │  Thresholds   │
    └──────────────┘           └───────────────┘          └───────────────┘
```

## Appendix E: Project Timeline (Preliminary)

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| Phase 1: Planning | 2 weeks | Requirements, Architecture |
| Phase 2: Core Development | 6 weeks | Backend, AI Integration |
| Phase 3: Frontend | 4 weeks | Dashboard, Live View |
| Phase 4: Testing | 2 weeks | Unit, Integration, UAT |
| Phase 5: Deployment | 1 week | Production Release |

**Total Estimated Duration:** 15 weeks

---

## Document Approval

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Project Manager | _____________ | _____________ | _____________ |
| Technical Lead | _____________ | _____________ | _____________ |
| QA Lead | _____________ | _____________ | _____________ |
| Stakeholder | _____________ | _____________ | _____________ |

---

**End of Software Requirements Specification**

*Document Version: 1.0 | Status: Preliminary | Classification: Internal*
