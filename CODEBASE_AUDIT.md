# Codebase Audit

Date: 2026-03-31

## Goal

Review the codebase for issues that affect startup reliability and basic app availability, fix the confirmed problems carefully, and record what was actually verified.

## What was verified

- The app imports successfully from the project virtual environment.
- Main route smoke tests passed with `fastapi.testclient`.
- The following endpoints returned `200` during verification:
- `/`
- `/remote_devices`
- `/live_camera`
- `/live_preview`
- `/system_state`
- `/api/devices`
- `/processing_status`
- HTTPS certificate files are present in `certs/`.

## Confirmed fixes

### 1. Fragile path handling

Problem:
- `app.py` depended on relative paths for `static`, `templates`, uploads, processed files, and snapshots.
- That could break rendering or file writes when the app was launched from a different working directory.

Fix:
- Switched those runtime paths to repository-root-based absolute paths.

Status:
- Fixed.

### 2. Reload behavior was too aggressive for this app

Problem:
- `CMS_RELOAD` defaulted to `true`.
- In this project, reload is expensive because startup loads heavy ML components, and it can lead to unnecessary duplicate startup work during normal runs.

Fix:
- Changed the default `CMS_RELOAD` behavior to `false`.
- Updated [start_https.ps1](c:/D/Crowd-Management-System-software-main/start_https.ps1) and [scripts/start_https.ps1](c:/D/Crowd-Management-System-software-main/scripts/start_https.ps1) to set `CMS_RELOAD=false`.
- Adjusted `uvicorn.run(...)` so it uses the app object directly when reload is disabled.

Status:
- Fixed.

### 3. MiDaS loaded during module import

Problem:
- [room_capacity.py](c:/D/Crowd-Management-System-software-main/room_capacity.py) loaded MiDaS at import time.
- That made normal startup slower and more failure-prone even when room-capacity estimation was not used.

Fix:
- Refactored MiDaS initialization to lazy-load only when `estimate_room_capacity(...)` is called.

Status:
- Fixed.

### 4. Snapshot path was relative

Problem:
- [yolo_inference.py](c:/D/Crowd-Management-System-software-main/yolo_inference.py) saved snapshots to a relative `snapshots` directory.

Fix:
- Anchored snapshot output to the repository root.

Status:
- Fixed.

## Notes on current behavior

- The app now imports without loading MiDaS immediately.
- YOLO still loads on app import. That is expected with the current architecture and was not changed in this pass.
- The main pages and state endpoints verified above are responding successfully.

## Remaining non-blocking issue

### `deep_sort_realtime` warning

Observed:
- Importing the app still emits a warning from `deep_sort_realtime` because that dependency uses `pkg_resources`, which is deprecated upstream.

Impact:
- This does not block startup or the smoke-tested routes.

Action:
- No code change was made here because the warning originates in a third-party package.
- If needed later, this can be handled by upgrading or replacing that dependency, or by pinning a compatible `setuptools` version.

## Limits of this audit

- This pass did not fully validate live webcam capture.
- This pass did not fully validate mobile camera streaming with a real phone.
- This pass did not run a full end-to-end video processing session with user-provided media.
- This pass focused on startup reliability and basic route availability.

## Result

- Startup behavior is more predictable.
- File and template resolution is more reliable.
- Unnecessary MiDaS loading during normal startup is removed.
- The app's main pages and core status endpoints are responding in smoke tests.
