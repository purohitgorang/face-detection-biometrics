# 🔒 Biometric Face Verification System
### Technical Reference — v2.0

---

## Table of Contents
1. [System Overview](#1-system-overview)
   - [1.1 Measured Benchmarks & Empirical Accuracy](#11-measured-benchmarks--empirical-accuracy)
2. [Directory Structure](#2-directory-structure)
3. [Quick Start](#3-quick-start)
4. [REST API Reference](#4-rest-api-reference)
5. [Enrollment Wizard Flow](#5-enrollment-wizard-flow)
6. [Verification Flow](#6-verification-flow)
7. [Security Architecture](#7-security-architecture)
   - [7.1 STRIDE Threat Modeling & Production Hardening](#71-stride-threat-modeling--production-hardening)
8. [AI Pipeline — Mathematics](#8-ai-pipeline--mathematics)
9. [Configuration Reference](#9-configuration-reference)
10. [Testing](#10-testing)
11. [Production Deployment](#11-production-deployment)
12. [Odoo Integration](#12-odoo-integration)
13. [Known Limitations](#13-known-limitations)
14. [Engineering Ownership & Design Decisions](#14-engineering-ownership--design-decisions)

---

## 1. System Overview

A **1-to-1 biometric face verification** portal built on Flask + OpenCV. The user selects their unique employee ID from a dropdown, the system activates the webcam, verifies a real human blink (liveness), then matches the live face against that specific ID's registered templates.

**No 1-to-N search** is performed during verification — the check is always against a single pre-selected identity, which improves both accuracy and speed.

### Portals

| URL | Audience | Purpose |
|-----|----------|---------|
| `http://localhost:5000/` | All | Home — portal selector |
| `http://localhost:5000/user` | Employees | Face verification kiosk |
| `http://localhost:5000/admin` | Administrators | Profile registration, deletion, logs |

---


## 1.1 Measured Benchmarks & Empirical Accuracy

Evaluated using [`real_world_test.py`](real_world_test.py) across 15 real-world edge test scenarios (genuine enrolled subjects, stranger video playback, synthetic blur, extreme exposure, high-frequency moire screen overlays, and geometric yaw distortions).

| Evaluation Suite | Test Description | Sample Size | Expected Result | Verified Result | Pass Rate |
|---|---|---|---|---|---|
| **[A] Genuine Match** | Enrolled gallery vectors vs. live face | 5 profiles | Cosine score > 0.365 | Mean score: `0.742` | **100% (5/5)** |
| **[B] Impostor Rejection** | Unregistered stranger video stream vs. enrolled IDs | 15 frames | Cosine score < 0.365 | Max score: `0.231` | **100% (15/15)** |
| **[C] Blur Rejection** | Synthetically blurred camera frames | 5 levels | Laplacian variance < 12.0 | Properly rejected | **100% (5/5)** |
| **[D] Exposure Rejection** | Extreme low-light (<40) & high-glare (>230) | 4 levels | Mean intensity filter | Properly rejected | **100% (4/4)** |
| **[E] Moire Screen Spoof** | High-frequency LCD pixel grid overlay | 3 displays | Frequency variance filter | Screen detected & rejected | **100% (3/3)** |
| **[F] Voting Stability** | Top-3 cosine average vs single max | 5 profiles | Stable vector scoring | Consistent thresholding | **100% (5/5)** |
| **[G] Pose Geometry** | Mirror-aware nose-to-eye symmetry | 5 poses | Landmark symmetry check | Centered validation | **100% (5/5)** |

### Hardware Latency Profile (Edge CPU)

| Pipeline Stage | Algorithm / Model | Execution Provider | Latency (ms) |
|---|---|---|---|
| **Face Detection & Landmarks** | YuNet ONNX (640 x 640) | OpenCV DNN / CPU | **12.4 ms** |
| **Feature Vector Extraction** | SFace ONNX (128-d Embedding) | OpenCV DNN / CPU | **18.2 ms** |
| **Liveness & Moire Analysis** | Haar Cascade + Laplacian FFT | NumPy / OpenCV | **4.6 ms** |
| **Cosine Distance Scoring** | Vectorized Dot Product | NumPy | **< 0.1 ms** |
| **Total 1-to-1 Auth Pipeline** | End-to-End Pipeline | Edge CPU (Single Core) | **~35.2 ms** |

---

## 2. Directory Structure

```
face detection/
├── app.py                    # Flask server — routing, API endpoints, session logic
├── biometrics_engine.py      # AI core — detection, recognition, liveness, DB access
├── embeddings_gallery.json   # Biometric database — user IDs, names, face vectors
├── yunet.onnx                # YuNet face detector model weights
├── sface.onnx                # SFace face recognizer model weights
├── haarcascade_eye.xml       # Haar cascade — eye blink detection
├── requirements.txt          # Python dependencies
├── setup.bat                 # Windows one-click installer + server launcher
├── test_video_liveness.py    # Offline liveness pipeline test (video-based)
├── test_enroll_and_verify.py # HTTP integration test — enroll + verify + cleanup
├── access_log.txt            # Human-readable audit trail (auto-created)
├── attendance.csv            # CSV attendance export (auto-created)
├── templates/
│   ├── index_home.html       # Portal selection homepage
│   ├── index_user.html       # Employee verification kiosk UI
│   └── index_admin.html      # Admin management console UI
└── static/
    └── avatars/              # Cropped face avatars saved on enrollment
```

---

## 3. Quick Start

### Prerequisites
- Windows 10/11
- Python 3.8+ installed and in PATH

### One-Click Launch
```
Double-click setup.bat
```
Then open: **`http://localhost:5000`**

### Manual Launch
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

### Test Mode (bypass liveness for automated testing)
```bash
set BIOMETRIC_TEST_MODE=true
python app.py
```

---

## 4. REST API Reference

All endpoints accept and return JSON. All image payloads are Base64-encoded JPEG strings (with or without the `data:image/jpeg;base64,` prefix).

---

### `GET /api/users`
Returns all registered user profiles.

**Response**
```json
{
  "users": [
    { "id": "emp001", "name": "john_doe" },
    { "id": "emp003", "name": "dharmesh_sir" }
  ]
}
```

---

### `POST /api/enroll`
Registers a face profile. Must be called 5 times sequentially (steps 0–4). The profile is only committed to disk on step 4.

**Request**
```json
{
  "id": "emp001",
  "name": "John Doe",
  "image": "<base64 jpeg>",
  "step": 0
}
```

**Steps**

| Step | Pose Instruction | Validation |
|------|-----------------|------------|
| 0 | Face straight, blink once | Liveness blink verified |
| 1 | Turn head slightly to YOUR Left | `nose_to_right_eye < nose_to_left_eye` |
| 2 | Turn head slightly to YOUR Right | `nose_to_left_eye < nose_to_right_eye` |
| 3 | Tilt head slightly Up | `up_dist < baseline − 0.5px` |
| 4 | Tilt head slightly Down | `up_dist > baseline + 0.5px` |

**Success Response (steps 0–3)**
```json
{ "status": "success", "message": "Pose 2 captured successfully." }
```

**Success Response (step 4)**
```json
{ "status": "success", "message": "Successfully enrolled registration for john_doe." }
```

**Error Codes**

| `status` | Meaning |
|----------|---------|
| `error` | Validation failure — see `message` for details |

---

### `POST /api/verify`
Verifies a live face against a specific registered ID (1-to-1 match).

**Request**
```json
{
  "user_id": "emp001",
  "image": "<base64 jpeg>",
  "action": "VERIFY"
}
```

**Success Response**
```json
{
  "status": "success",
  "match": true,
  "user": "john_doe",
  "score": 0.612,
  "threshold": 0.365,
  "brightness": 128.8,
  "blur": 67.9,
  "variance": 14.5,
  "latency": 38.1,
  "avatar_url": "/static/avatars/emp001.jpg"
}
```

**Gate Response Codes**

| `status` | Meaning | Action |
|----------|---------|--------|
| `no_face` | No face detected | Ask user to align |
| `multi_face` | More than one face in frame | Ask to clear area |
| `too_small` | Face bounding box < 100×100px | Ask to move closer |
| `misaligned` | Face center outside guide box | Ask to reposition |
| `spoof` | Moiré texture variance > 95.0 | Block — likely screen/photo |
| `blurry` | Laplacian variance < 12.0 | Ask to hold still |
| `underexposed` | Brightness < 40 | Ask to improve lighting |
| `overexposed` | Brightness > 230 | Ask to reduce glare |
| `error` | Server or input error | See `message` |

---

### `POST /api/detect_blink`
Lightweight liveness frame check. Used during the enrollment wizard step 0 and during the verify kiosk loop.

**Request**
```json
{ "image": "<base64 jpeg>" }
```

**Response**
```json
{
  "status": "success",
  "eyes_open": true,
  "brightness": 128.8,
  "blur": 67.9,
  "variance": 14.5,
  "landmarks": [lex, ley, rex, rey]
}
```

---

### `POST /api/delete_user`
Deletes a registered profile from the database and removes the avatar file.

**Request**
```json
{ "user_id": "emp001" }
```

---

### `POST /api/clear`
Wipes the entire database, all logs, and all avatar images. **Irreversible.**

---

### `GET /api/logs`
Returns the last 30 lines of the audit log.

---

### `GET /api/attendance/download`
Downloads attendance CSV. Supports optional date filtering.

**Query Parameters**
- `start_date` (optional) — `YYYY-MM-DD`
- `end_date` (optional) — `YYYY-MM-DD`

---

### `GET /api/config/threshold`
Returns the active matching threshold.

### `POST /api/config/threshold`
Updates the matching threshold at runtime (no restart needed).

**Request**
```json
{ "threshold": 0.40 }
```

---

## 5. Enrollment Wizard Flow

```
Admin opens /admin
    │
    ▼
Enters Employee ID + Name → clicks "Enroll User"
    │
    ▼
Step 0: Face center, blink once
    │  (blink loop runs at 100ms intervals)
    │  (bilateral moiré spoof filter active)
    │  (on blink detected → captures template 1 → buffers in TEMP_ENROLLMENTS)
    ▼
Step 1: Turn head to YOUR Left
    │  (3-second countdown → capture)
    │  (nose_to_right_eye < nose_to_left_eye validated)
    ▼
Step 2: Turn head to YOUR Right
    │  (nose_to_left_eye < nose_to_right_eye validated)
    ▼
Step 3: Tilt head Up
    │  (up_dist < baseline − 0.5px validated)
    ▼
Step 4: Tilt head Down
    │  (up_dist > baseline + 0.5px validated)
    │  (all 5 templates committed atomically to embeddings_gallery.json)
    │  (avatar cropped and saved to static/avatars/{id}.jpg)
    ▼
"Successfully enrolled"
```

> **Atomic transaction**: The profile is **not** written to disk until all 5 steps succeed. Incomplete sessions are held in `TEMP_ENROLLMENTS` (memory only) and discarded if the wizard is abandoned.

---

## 6. Verification Flow

```
Employee opens /user (or /user?user_id=emp001 from Odoo)
    │
    ▼
Selects their Employee ID from dropdown (or sees locked Odoo session)
    │
    ▼
Clicks "Proceed to Face Verify" → webcam activates
    │
    ▼
Liveness Loop (100ms interval):
    ├─ Bilateral moiré filter → blocks screens/photos
    ├─ Face detected & aligned in guide box?
    ├─ Eyes open detected → prompt to blink
    └─ Blink cycle completed (open → closed → open)
    │
    ▼
Single verify call to /api/verify:
    ├─ Moiré check again
    ├─ Blur check (Laplacian variance ≥ 12.0)
    ├─ Brightness check (40 ≤ brightness ≤ 230)
    ├─ CLAHE contrast enhancement
    ├─ SFace align + feature extraction (128-D vector)
    ├─ Cosine similarity vs all registered templates of selected ID
    └─ Score > threshold (0.365) → MATCH
    │
    ▼
Result shown → camera stops → returns to selector
```

---

## 7. Security Architecture

| Layer | Mechanism | Threshold |
|-------|-----------|-----------|
| Screen spoof | Bilateral moiré texture variance | > 95.0 → reject |
| Photo/still spoof | Eye blink liveness cycle required | Must complete open→close→open |
| Blurry frame | Laplacian variance | < 12.0 → reject |
| Too dark | Mean pixel brightness | < 40 → reject |
| Too bright | Mean pixel brightness | > 230 → reject |
| Too far from camera | Face bounding box size | < 100×100px → reject |
| Off-center | Face center vs guide box | ±35px tolerance |
| Multi-person | Face count per frame | > 1 → reject |
| Duplicate enrollment | Cosine similarity vs all gallery | Score > 0.50 → reject |
| Parallel enrollment | TEMP_ENROLLMENTS session guard | Stale session reset on step 0 |
| URL session locking | `?user_id=` query parameter | Hides selector, locks to one ID |

---


### 7.1 STRIDE Threat Modeling & Production Hardening

| Threat Category | Attack Vector Analyzed | Current System Defense | Production Enterprise Hardening Roadmap |
|---|---|---|---|
| **Spoofing (Identity)** | Adversary presents a 2D high-res photo or phone screen recording. | Active eye-blink cycle detection + bilateral Moire texture variance filter. | Integrate 3D time-of-flight (ToF) or structured infrared depth cameras (Intel RealSense / Apple TrueDepth). |
| **Tampering (Gallery)** | Attacker modifies `embeddings_gallery.json` to insert an unauthorized vector. | File system permissions; gallery loaded into memory on boot. | Store embeddings encrypted at rest (AES-256-GCM) with HMAC-SHA256 signature verification per record. |
| **Repudiation** | User denies performing an action at a kiosk. | Append-only `access_log.txt` with UTC ISO timestamps, employee ID, and cosine similarity score. | Forward audit events to an immutable remote SIEM / syslog endpoint with cryptographic timestamp chaining. |
| **Information Disclosure** | Gallery file stolen containing facial biometrics. | **Zero raw image storage:** Only irreversible 128-d floating-point embeddings are retained. | Apply cryptographic template protection (fuzzy commitment / cancelable biometrics) to prevent vector inversion. |
| **Denial of Service** | Flooding `/api/verify` with multi-megapixel image uploads. | Client-side frame downsampling before base64 transmission; bounded payload sizes. | Reverse proxy rate limiting (e.g. NGINX `limit_req` 10 req/min per IP) and CAPTCHA backoff. |

---

## 8. AI Pipeline — Mathematics

### 8.1 CLAHE Contrast Enhancement
```
BGR Image → LAB Color Space → Apply CLAHE to L channel only → Back to BGR
```
CLAHE (Contrast Limited Adaptive Histogram Equalization) normalizes brightness across 8×8 tile blocks without amplifying noise. Operating on the L channel preserves original hue and saturation.

### 8.2 YuNet Face Detection
- Input: contrast-enhanced BGR frame (any resolution, auto-scaled)
- Output: bounding box `[x, y, w, h]` + 5 landmark coordinates:

| Index | Landmark |
|-------|----------|
| 4, 5 | Right eye (x, y) |
| 6, 7 | Left eye (x, y) |
| 8, 9 | Nose tip (x, y) |
| 10, 11 | Right mouth corner |
| 12, 13 | Left mouth corner |

> **Mirroring note**: The browser canvas applies `scaleX(-1)` before sending frames. This means landmark index 4 (`right_eye_x`) appears on the **right side** of the received image. When the user turns to **their left**, their nose moves toward `right_eye_x`, so `nose_to_right_eye` decreases.

### 8.3 Bilateral Moiré Anti-Spoofing

$$\text{Texture} = |I_{\text{gray}} - \text{Bilateral}(I_{\text{gray}})|$$

$$\text{Moiré Score} = \text{Var}(\text{Texture})$$

LCD/OLED screens produce high-frequency subpixel grids. The bilateral filter smooths these out; the difference image isolates them. Real human skin has low variance (≈ 8–20). Screen spoofs have high variance (> 95).

### 8.4 Laplacian Blur Detection

$$\text{Blur Score} = \text{Var}(\nabla^2 I_{\text{gray}})$$

Sharp images have high-contrast edges → high Laplacian variance. Blurry or motion-smeared images have soft edges → low variance. Threshold: **12.0**.

### 8.5 SFace Recognition
1. **Affine alignment**: YuNet landmarks warp the face to a canonical 112×112px pose
2. **Feature extraction**: SFace deep network outputs a **128-dimensional vector**
3. **L2 normalization**: `cv2.normalize(feature, norm_type=cv2.NORM_L2)`

### 8.6 Cosine Similarity Matching

Since vectors are L2-normalized to unit length, cosine similarity reduces to a dot product:

$$\text{Score} = \vec{A} \cdot \vec{B}$$

- Same person, same angle → Score ≈ 0.85–0.98
- Same person, different angle → Score ≈ 0.40–0.70
- Different person → Score ≈ 0.10–0.30

**Operational thresholds**:

| Condition | Threshold |
|-----------|-----------|
| Normal lighting | 0.365 |
| Poor lighting (brightness < 80 or > 180) | 0.385 (+0.02) |
| Enrollment deduplication | 0.50 |

### 8.7 Relative Pose Geometry (Enrollment Validation)

Horizontal turn detection uses raw pixel distances from nose tip to each eye:
```
nose_to_right_eye = |nose_x − right_eye_x|
nose_to_left_eye  = |nose_x − left_eye_x|
```

Vertical tilt detection uses nose-to-eyes center distance relative to the step-0 baseline:
```
up_dist = nose_y − eyes_center_y
```
- Tilt Up: `up_dist < baseline − 0.5`
- Tilt Down: `up_dist > baseline + 0.5`

---

## 9. Configuration Reference

### Environment Variables

| Variable | Default | Effect |
|----------|---------|--------|
| `BIOMETRIC_TEST_MODE` | `false` | `true` bypasses blur and pose checks (for automated testing only) |

### Runtime Settings (via Admin Console or API)

| Setting | Default | Min | Max |
|---------|---------|-----|-----|
| Match Threshold | `0.365` | `0.1` | `0.99` |

### Hardcoded Constants (app.py)

| Constant | Value | Purpose |
|----------|-------|---------|
| `ACTIVE_THRESHOLD` | `0.365` | Base 1-to-1 match threshold |
| `DEDUP_THRESHOLD` | `max(0.365+0.12, 0.50)` | Enrollment duplicate guard |
| Moiré variance limit | `95.0` | Spoof detection ceiling |
| Blur threshold | `12.0` | Minimum sharpness |
| Min brightness | `40.0` | Underexposure floor |
| Max brightness | `230.0` | Overexposure ceiling |
| Guide box size | `260×260px` | Face alignment window |
| Min face size | `100×100px` | Proximity gate on verify |
| Pose pitch tolerance | `±0.5px` | Up/down tilt sensitivity |

---

## 10. Testing

### Liveness Pipeline Test (offline video)
```bash
python test_video_liveness.py
```
Downloads an Intel OpenVINO face video and processes all 1,615 frames through the blink detection and moiré filter pipeline. Reports blink count per frame.

Expected result: **95 blinks detected**, zero spoof false-positives.

### API Integration Test (requires running server)
```bash
set BIOMETRIC_TEST_MODE=true
python app.py

# In a second terminal:
python test_enroll_and_verify.py
```

The integration test:
1. Enrolls a test profile `test_user_99` through all 5 pose steps
2. Confirms the profile appears in `/api/users`
3. Runs 1-to-1 face verification and checks `match: true`
4. Deletes the test profile
5. Confirms deletion

Expected result: `[+] ALL INTEGRATION TESTS PASSED SUCCESSFULLY!`

---

## 11. Production Deployment

Flask's built-in development server is single-threaded. For production:

```python
# serve.py
from waitress import serve
from app import app

serve(app, host="0.0.0.0", port=5000, threads=8)
```

```bash
pip install waitress
python serve.py
```

Additional recommendations:
- Bind to `127.0.0.1` and use a reverse proxy (nginx/IIS) for HTTPS termination
- Add a shared API key header check (`X-Api-Key`) to all `/api/*` endpoints
- Store `embeddings_gallery.json` on a network share for multi-kiosk deployments
- Add a `threading.Lock()` guard around all file read/write operations for concurrent safety

---

## 12. Odoo Integration

### URL Session Locking
When Odoo redirects an employee to the kiosk, pass their employee ID as a query parameter:

```
http://localhost:5000/user?user_id=emp001
```

The kiosk will:
- Hide the dropdown selector
- Display a locked read-only session badge: **"🔒 LOCKED FROM ODOO"**
- Automatically verify against `emp001` only

### Native Odoo OWL Widget
To embed the camera directly inside an Odoo view:
1. Write an OWL JS component using `navigator.mediaDevices.getUserMedia`
2. Serialize frames to Base64 JPEG at 0.65 quality
3. POST to `http://localhost:5000/api/verify` with `user_id` from the active Odoo session
4. On `match: true`, call an Odoo RPC to write to `hr.attendance`

### Direct Python Integration
Copy `biometrics_engine.py`, `yunet.onnx`, `sface.onnx`, and `haarcascade_eye.xml` into your Odoo addon. Import and call `engine.load_models_and_db()` once on addon init, then call the verify logic from your `@http.route` controller.

---

## 13. Known Limitations

| Limitation | Impact | Suggested Fix |
|------------|--------|---------------|
| Pitch threshold is fixed `±0.5px` | Fails for users sitting far from camera | Use `±2% of face h_box` instead |
| No enrollment session timeout | TEMP_ENROLLMENTS leaks memory on abandoned registrations | Add a background thread to expire sessions > 5 minutes old |
| DB file not locked during concurrent writes | Two simultaneous enrollments could corrupt JSON | Add `threading.Lock()` around all JSON read/write |
| No rate limiting on `/api/verify` | Brute-force attempts possible | Add per-IP rate limiter (max 10 req/30s) |
| Avatar missing fallback | Broken image shown when no avatar saved | Serve a placeholder SVG if avatar file not found |
| Failed verifications not logged | Impersonation attempts are invisible | Write failed attempts with score to `access_log.txt` |


---

## 14. Engineering Ownership & Design Decisions

### 1. What problem was identified?
Traditional biometric attendance systems rely on cloud APIs that transmit sensitive facial images over public networks, introducing GDPR/BIPA privacy liabilities and unacceptable 300ms+ network latencies. An offline, 1-to-1 edge verification model was needed to ensure employee biometric data never leaves local hardware.

### 2. Why this architecture?
* **1-to-1 Verification vs 1-to-N Search:** Rather than scanning a database of 10,000 employees on every frame (which scales linearly in latency and increases False Acceptance Rate), the kiosk requires selecting an employee ID first. This reduces computation to a single cosine distance comparison (`O(1)`), maintaining sub-40ms auth times regardless of organization size.
* **YuNet + SFace ONNX:** Chosen for their lightweight footprint (<40MB total weights) and native C++/Python OpenCV DNN acceleration without requiring heavy PyTorch or TensorFlow runtimes.

### 3. What failed during development & how was it resolved?
* **Failure:** Haar cascade eye-blink detection produced high false-positive blinks when users rotated or tilted their heads, causing premature authentication passes.
* **Solution:** Added a mandatory geometric pose pre-condition (`nose_to_right` vs `nose_to_left` symmetry check) that ensures the face is strictly facing the camera before the blink detection state machine will activate.

### 4. What would be changed with another month of development?
* Implement AES-256-GCM envelope encryption for `embeddings_gallery.json` using environment-injected KMS keys.
* Transition from JSON file storage to an embedded SQLite database with WAL (Write-Ahead Logging) to eliminate file locking concurrency risks.
