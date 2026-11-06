# 🔒 Biometric Face Verification System
## Developer Handover & Deployment Guide — v2.0

This document is the complete handover reference for the incoming developer.
It covers architecture, AI mathematics, bug history, known limitations, and deployment steps.

---

## Part 1: What the System Does (For Managers)

An AI-powered **1-to-1 biometric face verification portal**. Employees select their Unique ID on a kiosk screen, look at a webcam, blink once to prove they are a live human, and the system verifies their face against their registered templates. Designed to prevent buddy-punching and session hijacking in Odoo ERP environments.

**Single-click setup**: Double-click `setup.bat` → open `http://localhost:5000`.

---

## Part 2: Architecture Diagram

```
Browser (index_user.html / index_admin.html)
    │
    │  Base64 JPEG frames via fetch() POST — 100ms intervals
    │  (canvas applies scaleX(-1) mirror before encoding)
    ▼
Flask Server (app.py) — Port 5000
    │
    ├─ /api/detect_blink  →  biometrics_engine.check_eyes_open()
    ├─ /api/enroll        →  biometrics_engine.RECOGNIZER.feature()
    ├─ /api/verify        →  np.dot(live_vector, registered_vector)
    ├─ /api/users         →  engine.GALLERY dict (in-memory cache)
    └─ /api/attendance/download  →  attendance.csv
    │
    ▼
biometrics_engine.py
    │
    ├─ DETECTOR  = YuNet  (yunet.onnx)   — face bounding box + 5 landmarks
    ├─ RECOGNIZER = SFace (sface.onnx)   — 128-D biometric feature vector
    ├─ EYE_CASCADE = Haar (haarcascade_eye.xml)  — open/closed eye detection
    └─ GALLERY  = { user_id: { "name": str, "vectors": [np.array] } }
    │
    ▼
embeddings_gallery.json  (file on disk, reloaded on mtime change)
static/avatars/{user_id}.jpg  (cropped face photo per enrolled user)
access_log.txt + attendance.csv  (audit trail, append-only)
```

---

## Part 3: Frontend Architecture

### Frame Capture & Transmission
```javascript
// captureFrame() in index_user.html and index_admin.html:
context.translate(canvas.width, 0);
context.scale(-1, 1);                            // mirrors horizontally
context.drawImage(video, 0, 0, canvas.width, canvas.height);
return canvas.toDataURL('image/jpeg', 0.65);     // 65% JPEG quality → ~50% smaller payload
```

> **Critical**: The horizontal mirror means landmark coordinates in the server's received image are **flipped relative to the display**. When the user turns to their **physical left**, their nose moves toward `right_eye_x` (YuNet landmark index 4) in the server-received image.

### Polling & Dropdown Stability
- The user portal polls `/api/users` every 3 seconds to pick up newly enrolled users.
- `selector.value` is read **before** clearing `innerHTML` and the matching option is re-selected afterward — preventing the dropdown from resetting while the user is making a selection.

### Liveness Loop State Machine
```
eyesWereOpen = false
eyesClosedSeen = false

Loop (100ms):
  if eyes_open == true:
      eyesWereOpen = true
      if eyesClosedSeen == true:
          → BLINK COMPLETE → trigger verify / enroll capture
      else:
          → show "Blink now"
  else (eyes closed):
      if eyesWereOpen == true:
          eyesClosedSeen = true   → "Blink detected, keep looking"
      else:
          → "Eyes not detected"
```

---

## Part 4: Backend Architecture

### In-Memory State

| Variable | Type | Purpose |
|----------|------|---------|
| `GALLERY` | `dict` | Cached biometric DB. Reloaded from disk when `mtime` changes. |
| `TEMP_ENROLLMENTS` | `dict` | Enrollment wizard buffer. Holds pose templates until step 4 commits atomically. |
| `COOLDOWN_DICT` | `dict` | Reserved — cooldown system removed but structure kept. |
| `ACTIVE_THRESHOLD` | `float` | Runtime-adjustable match threshold (default `0.365`). |
| `TEST_MODE` | `bool` | Set via `BIOMETRIC_TEST_MODE=true` env var. Bypasses blur + pose checks. |

### Database Schema (`embeddings_gallery.json`)
```json
{
  "emp001": {
    "name": "john_doe",
    "vectors": [
      [0.021, -0.134, ...],   // step 0 — face straight
      [0.018, -0.141, ...],   // step 1 — turned left
      [0.023, -0.129, ...],   // step 2 — turned right
      [0.019, -0.138, ...],   // step 3 — tilted up
      [0.022, -0.132, ...]    // step 4 — tilted down
    ]
  }
}
```

All IDs and names are sanitized via `re.sub(r'[^a-zA-Z0-9\s_-]', '', name)` on write and read.

### Enrollment Transaction Flow
1. Step 0 arrives → `TEMP_ENROLLMENTS[user_id]` created with `{"name", "vectors": [v0], "base_up_dist"}`
2. Steps 1–4 arrive → pose geometry validated → vector appended with double-post guard:
   ```python
   if len(TEMP_ENROLLMENTS[user_id]["vectors"]) < step + 1:
       TEMP_ENROLLMENTS[user_id]["vectors"].append(vector)
   ```
3. Step 4 → all 5 vectors committed to `embeddings_gallery.json` → gallery reloaded → temp session deleted

---

## Part 5: AI Pipeline — Mathematics

### 5.1 CLAHE Contrast Enhancement
```
BGR → LAB → CLAHE on L channel (clipLimit=2.0, tileGrid=8×8) → BGR
```
Normalizes brightness without distorting hue/saturation. Prevents dark-room enrollment failures.

### 5.2 YuNet Face Detection
- Input: BGR image at native camera resolution (auto-scaled via `setInputSize`)
- Confidence threshold: `0.8` | NMS threshold: `0.3` | Max faces: `5000`
- Output: `[x, y, w, h, rex, rey, lex, ley, nx, ny, rmx, rmy, lmx, lmy]`

### 5.3 Bilateral Moiré Anti-Spoof

$$\text{Texture} = |I_{\text{gray}} - \text{Bilateral}(I_{\text{gray}}, d=9, \sigma_c=75, \sigma_s=75)|$$

$$\text{Moiré Score} = \text{Var}(\text{Texture})$$

- Real skin: score ≈ **8–25**
- Phone screen: score ≈ **100–200**
- Rejection threshold: **95.0**

### 5.4 Laplacian Blur Detection

$$\text{Blur Score} = \text{Var}(\nabla^2 I_{\text{gray}})$$

- Sharp face: score ≈ **50–200**
- Blurry/motion: score ≈ **1–10**
- Rejection threshold: **12.0**

### 5.5 SFace Recognition
1. `alignCrop(image, face)` → 112×112px affine-warped canonical face
2. `feature(aligned)` → 128-dimensional float32 vector
3. `cv2.normalize(feature, norm_type=cv2.NORM_L2)` → unit vector

### 5.6 Matching

Since vectors are L2-normalized:
$$\text{Score} = \vec{A} \cdot \vec{B} = \cos\theta$$

Verification uses `max(scores)` across all 5 registered templates:
```python
scores = [float(np.dot(vec, live_vector)) for vec in user_vectors]
best_score = max(scores)
match = best_score > current_threshold
```

**Score distribution (SFace on LFW benchmark)**:
- Same person: typically **0.40–0.98**
- Different person: typically **0.05–0.30**
- Operational threshold: **0.365**

### 5.7 Pose Geometry (Enrollment Validation)

```python
right_eye_x = best_face[4]   # YuNet right eye
left_eye_x  = best_face[6]   # YuNet left eye
nose_x      = best_face[8]

# Mirrored image: user's LEFT → nose moves toward right_eye_x
nose_to_right_eye = abs(nose_x - right_eye_x)
nose_to_left_eye  = abs(nose_x - left_eye_x)

# Step 1 (user turns LEFT):  nose_to_right_eye < nose_to_left_eye
# Step 2 (user turns RIGHT): nose_to_left_eye  < nose_to_right_eye

up_dist = nose_y - (right_eye_y + left_eye_y) / 2.0
# Step 3 (tilt up):   up_dist < baseline − 0.5
# Step 4 (tilt down): up_dist > baseline + 0.5
```

---

## Part 6: Bug History & Fixes Applied

| # | Bug | Fix Applied |
|---|-----|-------------|
| 1 | `test_user_99` hardcoded backdoor | Replaced with `BIOMETRIC_TEST_MODE` env var |
| 2 | `left_dist`/`right_dist` variable names semantically swapped | Renamed to `nose_to_left_eye`/`nose_to_right_eye` with comments |
| 3 | Pose checks inverted (mirror not accounted for) | Flipped step 1/2 reject conditions to account for `scaleX(-1)` |
| 4 | Parallel enrollment race — same ID could start two sessions | Stale session cleared on step 0 restart |
| 5 | Admin step-0 enroll call missing `step: 0` in payload | Added explicitly |
| 6 | `temp_attendance.csv` left on disk after filtered export | File deleted immediately after `send_file()` |
| 7 | Dead `cooldown` status handler in user portal JS | Removed |
| 8 | Log column width `:<6` too narrow for `DELETE_PROFILE` | Widened to `:<16` |
| 9 | Enrollment dedup used strict match threshold (0.365) | Raised to `max(threshold+0.12, 0.50)` |
| 10 | Network retry could double-append same step vector | Added `current_count < step + 1` guard |
| 11 | Dropdown reset to first option every 3s poll | Preserve `selector.value` before clearing `innerHTML` |

---

## Part 7: Known Limitations & Recommended Fixes

| Limitation | Priority | Recommended Fix |
|------------|----------|----------------|
| Pitch threshold is fixed `±0.5px` | High | Use `±2% of face bounding box height` |
| No enrollment session timeout | High | Background thread to expire TEMP_ENROLLMENTS > 5 minutes |
| No DB file locking during concurrent writes | High | `threading.Lock()` around all JSON reads/writes |
| No rate limiting on `/api/verify` | Medium | Per-IP counter: max 10 req/30s |
| `max(scores)` matching — one lucky frame can pass | Medium | Use average of top-3 scores instead |
| No minimum face size check on enroll | Medium | Add `w_box < 100` gate same as verify |
| Flask binds to `0.0.0.0` — LAN-exposed | Medium | Bind to `127.0.0.1`, use nginx reverse proxy |
| Avatar fallback missing | Low | Serve placeholder SVG if avatar file absent |
| Failed verifications not logged | Low | Append failed attempts with score to `access_log.txt` |
| Vectors have no enrollment timestamp | Low | Add `enrolled_at` field to gallery schema |

---

## Part 8: Production Deployment

### Option A — Waitress (Windows, recommended)
```python
# serve.py
from waitress import serve
from app import app
serve(app, host="127.0.0.1", port=5000, threads=8)
```
```bash
pip install waitress
python serve.py
```

### Option B — Gunicorn (Linux)
```bash
pip install gunicorn
gunicorn -w 4 -b 127.0.0.1:5000 app:app
```

### IIS / Nginx Reverse Proxy (HTTPS)
Put a reverse proxy in front of the Flask app:
```nginx
server {
    listen 443 ssl;
    location / {
        proxy_pass http://127.0.0.1:5000;
    }
}
```

---

## Part 9: Odoo Integration

### Method 1 — URL Session Lock
```
http://localhost:5000/user?user_id=emp001
```
The portal hides the dropdown and shows a **"🔒 LOCKED FROM ODOO"** badge. Verification always runs against `emp001`.

### Method 2 — OWL Component (Odoo 16/17)
1. Add an OWL JS component to your custom addon
2. Use `navigator.mediaDevices.getUserMedia` to access the webcam
3. Capture and mirror frames, encode to Base64 JPEG at quality `0.65`
4. POST to `http://your-kiosk-server:5000/api/verify` with `user_id` from the Odoo session
5. On `{ match: true }`, call `this.env.services.orm.call("hr.attendance", "action_check_in", [[employee_id]])`

### Method 3 — Native Python in Odoo Addon
Copy into your addon folder:
- `biometrics_engine.py`
- `yunet.onnx`
- `sface.onnx`
- `haarcascade_eye.xml`

In your Odoo `__init__.py`:
```python
import biometrics_engine as engine
engine.load_models_and_db()
```

In your controller:
```python
@http.route('/biometric/verify', type='json', auth='user')
def verify(self, user_id, image_b64):
    img = engine.decode_base64_image(image_b64)
    # ... run detection, alignment, feature extraction, dot product
    return {"match": bool, "score": float}
```

---

## Part 10: Handover Checklist

- [ ] `embeddings_gallery.json` backed up and stored securely
- [ ] `static/avatars/` folder backed up
- [ ] `BIOMETRIC_TEST_MODE` env var is **not set** in production
- [ ] Flask server binding changed from `0.0.0.0` to `127.0.0.1` or behind a proxy
- [ ] Waitress or Gunicorn configured for multi-threaded production serving
- [ ] Date/time on host machine is correct (attendance timestamps depend on it)
- [ ] New developer has read the **Bug History** section (Part 6) to understand mirror geometry
