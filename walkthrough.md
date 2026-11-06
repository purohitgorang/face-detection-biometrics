# Walkthrough: Odoo-Style 1-to-1 Unique ID Verification with Moire Spoof Gating

We have successfully refactored and optimized the biometric attendance system:

## Step 1: Session Selector (Camera Closed)
* The employee enters the portal and is presented with a dropdown list of Unique IDs (representing their Odoo session log credentials).
* **The webcam remains completely inactive and turned off.**

## Step 2: Verification Gating (Camera Active + Eye Blink Liveness + Moire Spoof Gating)
* Clicking **"Proceed to Face Verify"** activates the webcam stream.
* **Bilateral Texture Moire Filter**:
  * We compute the difference between the original gray image and a bilaterally filtered version. Bilateral filtering smooths out micro-grids (like OLED/LCD pixels) while preserving face edges.
  * Subtracting the two yields the high-frequency grid signature.
  * **Digital screens and printed photos** generate high-frequency moiré patterns (`variance > 95.0`).
  * If a screen spoof is detected, the check is immediately aborted: **"REJECTED - SPOOF DETECTED"**.
* **Eye-Blink Liveness**: Once the liveness filter confirms it is a real human face, it monitors the eyes for a **real human blink** (`Open eyes ➔ Closed eyes ➔ Open eyes`) before proceeding.
* **Match Verification**: Once a blink is confirmed, it verifies the face 1-to-1 against the selected ID's registered templates.
* **Success**: Grants access (*"✓ SUCCESS: VERIFIED AS John Doe"*), stops the camera, and redirects back to the selector.
* **Failure**: Denies access (*"✗ ACCESS DENIED: Face does not match John Doe"*), stops the camera, and redirects back to the selector.

---

## Technical Robustness Enhancements
1. **Lowercased Casing Standard (Deletion Fix)**: Standardized `sanitize_name` in `biometrics_engine.py` to match `app.py`'s lowercase rule. This ensures that when checking for duplicate IDs or deleting a user, the lookup keys strictly match without any casing mismatches (e.g. `emp001` vs `EMP001`).
2. **Missing Database Cache Recovery**: Updated `reload_gallery` and `check_and_reload_db` to properly clear the in-memory variables to `{}` if the database file is completely missing or deleted.
3. **2x Larger Webcam Viewport**: Rescaled the camera viewport panel layout and aspect-ratio CSS bindings. The webcam display panel area is now **2.25x larger** on both the User Kiosk and Admin Console.
4. **Auto-Polling & Cache-Busters**: 
   * Added `?t=timestamp` cache-busters to AJAX GET requests to prevent browser cache retention.
   * Enabled a background **3-second polling interval** in `/user` to dynamically update selection options if an administrator adds or deletes profiles.
5. **Interactive Eye HUD Feedback Overlay**: The backend now returns computed coordinates for the right and left eye centers, drawing glowing biometric rings directly over the eyes.
6. **Video-Based Liveness Test Suite**: Created `test_video_liveness.py` which downloads a sample face video clip from the Intel OpenVINO dataset and runs it frame-by-frame through the liveness pipeline.
7. **Balanced Moiré Thresholds**: Raised the upper moiré texture variance threshold to `95.0` to prevent false-positives under noisy camera grains.
8. **Wide Context Eye Crops**: Increased proportional eye crop radius to `14%` of face bounding box width, providing the Haar Cascade with essential eyebrow and eye socket structural context.
9. **Sensitive Detection Tuning**: Relaxed scale factor to `1.05` to increase search iterations.
10. **Duplicate Person Face Gating**: During enrollment, the system extracts the live facial feature vector and calculates the similarity against **all registered vectors of all users** to prevent double-enrollments.
11. **Bilateral Moire Screen Spoof Filter**: Blocks phone screens displaying a blinking video.
12. **Correct Image Targeting**: Texture analysis runs on the original camera image.
13. **Either-Eye Relaxed Check**: Trigger if **at least one eye** is detected open.
14. **Telemetry Diagnostics**: Dynamically displays face brightness, Laplacian blur, and Moire texture variance.
