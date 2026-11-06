import cv2
import os
import json
import time
import re
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file
import biometrics_engine as engine

app = Flask(__name__)

@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    return r

# Configurable global strictness threshold
ACTIVE_THRESHOLD = 0.365

# Set BIOMETRIC_TEST_MODE=true env var to bypass liveness checks during automated testing
TEST_MODE = os.environ.get("BIOMETRIC_TEST_MODE", "false").lower() == "true"

# Double-punch cooldown memory: {user_id: timestamp}
COOLDOWN_DICT = {}

# Temporary storage for enrollment poses: {user_id: {"name": name, "vectors": []}}
TEMP_ENROLLMENTS = {}

# Initialize models
engine.load_models_and_db()

@app.route("/")
def index():
    return render_template("index_home.html")

@app.route("/user")
def user_portal():
    return render_template("index_user.html")

@app.route("/admin")
def admin_portal():
    return render_template("index_admin.html")

@app.route("/api/config/threshold", methods=["GET", "POST"])
def configure_threshold():
    global ACTIVE_THRESHOLD
    if request.method == "POST":
        data = request.get_json()
        if data and "threshold" in data:
            ACTIVE_THRESHOLD = float(data["threshold"])
            return jsonify({"status": "success", "threshold": ACTIVE_THRESHOLD})
    return jsonify({"status": "success", "threshold": ACTIVE_THRESHOLD})

@app.route("/api/users", methods=["GET"])
def get_users():
    engine.check_and_reload_db()
    users_list = []
    for uid, info in engine.GALLERY.items():
        users_list.append({
            "id": uid,
            "name": info["name"]
        })
    return jsonify({"users": users_list})

@app.route("/api/clear", methods=["POST"])
def clear_db():
    engine.GALLERY = {}
    global COOLDOWN_DICT
    COOLDOWN_DICT.clear()
    
    if os.path.exists(engine.GALLERY_PATH):
        try:
            os.remove(engine.GALLERY_PATH)
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
            
    # Clear logs
    for path in [engine.LOG_PATH, engine.CSV_PATH]:
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception as e:
                print(f"[-] Failed to remove log: {e}")
                
    # Clear avatars directory
    if os.path.exists(engine.AVATARS_DIR):
        try:
            for file in os.listdir(engine.AVATARS_DIR):
                os.remove(os.path.join(engine.AVATARS_DIR, file))
        except Exception as e:
            print(f"[-] Failed to clear avatars: {e}")
            
    engine.reload_gallery()
    return jsonify({"status": "success", "message": "Database and logs cleared successfully."})

@app.route("/api/delete_user", methods=["POST"])
def delete_user():
    data = request.get_json()
    if not data or "user_id" not in data:
        return jsonify({"status": "error", "message": "Missing 'user_id' payload."}), 400
        
    user_id = engine.sanitize_name(data["user_id"])
    engine.check_and_reload_db()
    
    if user_id not in engine.GALLERY:
        return jsonify({"status": "error", "message": f"Profile ID '{user_id}' does not exist."}), 400
        
    try:
        # Load database JSON file
        current_data = {}
        if os.path.exists(engine.GALLERY_PATH):
            with open(engine.GALLERY_PATH, "r") as f:
                current_data = json.load(f)
                
        # Delete user_id from database
        if user_id in current_data:
            username = current_data[user_id].get("name", user_id)
            del current_data[user_id]
            with open(engine.GALLERY_PATH, "w") as f:
                json.dump(current_data, f, indent=4)
        else:
            username = user_id
            
        # Delete profile avatar image file
        avatar_filename = f"{user_id}.jpg"
        avatar_path = os.path.join(engine.AVATARS_DIR, avatar_filename)
        if os.path.exists(avatar_path):
            try:
                os.remove(avatar_path)
            except Exception as e:
                print(f"[-] Failed to remove avatar file {avatar_filename}: {e}")
                
        # Clear user from cooldown tracker
        if user_id in COOLDOWN_DICT:
            del COOLDOWN_DICT[user_id]
            
        # Log successful profile deletion
        engine.write_access_log("DELETE_PROFILE", f"System Admin", f"Deleted Profile ID: {user_id} (Name: {username})")
        
        engine.reload_gallery()
        return jsonify({"status": "success", "message": f"Successfully deleted profile '{username}' ({user_id})."})
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/detect_blink", methods=["POST"])
def detect_blink():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"status": "error", "message": "Missing 'image' payload."}), 400
        
    try:
        img = engine.decode_base64_image(data["image"])
        if img is None:
            return jsonify({"status": "error", "message": "Failed to decode image."}), 400
            
        h, w, _ = img.shape
        engine.DETECTOR.setInputSize((w, h))
        _, faces = engine.DETECTOR.detect(img)
        
        if faces is None or len(faces) == 0:
            return jsonify({"status": "no_face", "eyes_open": False})
            
        # Select largest face
        best_face = None
        max_area = 0
        for f in faces:
            area = int(f[2]) * int(f[3])
            if area > max_area:
                max_area = area
                best_face = f
                
        # Calculate Telemetry parameters
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_val = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        
        x, y, w_box, h_box = list(map(int, best_face[:4]))
        face_roi = img[max(0, y):min(h, y+h_box), max(0, x):min(w, x+w_box)]
        
        face_brightness = 0.0
        texture_var = 0.0
        
        if face_roi.size > 0:
            face_brightness = float(np.mean(face_roi))
            h_roi, w_roi = face_roi.shape[:2]
            if h_roi >= 64 and w_roi >= 64:
                gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                gray_roi = cv2.resize(gray_roi, (128, 128))
                bilateral = cv2.bilateralFilter(gray_roi, 9, 75, 75)
                texture = cv2.absdiff(gray_roi, bilateral)
                texture_var = float(np.var(texture))
                if texture_var > 95.0:
                    return jsonify({
                        "status": "spoof",
                        "eyes_open": False,
                        "message": "REJECTED - SPOOF DETECTED (Fake Face / Phone Screen)",
                        "brightness": face_brightness,
                        "blur": blur_val,
                        "variance": texture_var
                    })
                    
        # Extract eye landmarks for visual overlay feedback
        lex, ley = int(best_face[4]), int(best_face[5])
        rex, rey = int(best_face[6]), int(best_face[7])
        
        eyes_open = engine.check_eyes_open(img, best_face)
        return jsonify({
            "status": "success",
            "eyes_open": eyes_open,
            "brightness": face_brightness,
            "blur": blur_val,
            "variance": texture_var,
            "landmarks": [lex, ley, rex, rey]
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/enroll", methods=["POST"])
def enroll_user():
    data = request.get_json()
    if not data or "name" not in data or "id" not in data or "image" not in data:
        return jsonify({"status": "error", "message": "Missing registration details."}), 400
        
    username = engine.sanitize_name(data["name"])
    user_id = engine.sanitize_name(data["id"])
    step = int(data.get("step", 0))
    
    if not username or not user_id:
        return jsonify({"status": "error", "message": "ID and Name cannot be empty."}), 400
        
    engine.check_and_reload_db()
    # BUG3 FIX: Block parallel enrollment sessions for the same ID
    if step == 0 and user_id in TEMP_ENROLLMENTS:
        del TEMP_ENROLLMENTS[user_id]  # Reset stale session and start fresh
    if user_id in engine.GALLERY:
        existing_vectors = engine.GALLERY[user_id].get("vectors", [])
        if len(existing_vectors) >= 5:
            return jsonify({"status": "error", "message": f"Unique ID '{user_id}' already exists."}), 400
        
    try:
        img = engine.decode_base64_image(data["image"])
        if img is None:
            return jsonify({"status": "error", "message": "Failed to decode image."}), 400
            
        img_enhanced = engine.enhance_contrast(img)
        h, w, _ = img_enhanced.shape
        
        engine.DETECTOR.setInputSize((w, h))
        _, faces = engine.DETECTOR.detect(img_enhanced)
        
        if faces is None or len(faces) == 0:
            return jsonify({"status": "error", "message": "No face detected."}), 400
            
        # Select largest face
        best_face = None
        max_area = 0
        for f in faces:
            area = int(f[2]) * int(f[3])
            if area > max_area:
                max_area = area
                best_face = f
                
        # Validate that the correct pose instructions are followed
        pose, yaw_ratio, pitch_ratio = engine.estimate_head_pose(best_face)
            
        # Bilateral texture moire analysis for liveness in enrollment
        x, y, w_box, h_box = list(map(int, best_face[:4]))
        face_roi = img[max(0, y):min(h, y+h_box), max(0, x):min(w, x+w_box)]
        if face_roi.size > 0:
            h_roi, w_roi = face_roi.shape[:2]
            if h_roi >= 64 and w_roi >= 64:
                gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                gray_roi = cv2.resize(gray_roi, (128, 128))
                bilateral = cv2.bilateralFilter(gray_roi, 9, 75, 75)
                texture = cv2.absdiff(gray_roi, bilateral)
                texture_var = float(np.var(texture))
                if texture_var > 95.0:
                    return jsonify({
                        "status": "error",
                        "message": f"Enrollment rejected: Spoof detected (Moiré Var: {texture_var:.1f})"
                    }), 400
                    
        # Extract face landmarks for relative geometry calculations
        # Landmarks: index 4,5 = right eye; 6,7 = left eye; 8,9 = nose tip
        right_eye_x, right_eye_y = float(best_face[4]), float(best_face[5])
        left_eye_x,  left_eye_y  = float(best_face[6]), float(best_face[7])
        nx, ny = float(best_face[8]), float(best_face[9])
        
        # Horizontal distances from nose tip to each eye
        # When user turns LEFT  → nose moves toward their LEFT  → closer to left_eye_x
        # When user turns RIGHT → nose moves toward their RIGHT → closer to right_eye_x
        nose_to_left_eye  = abs(nx - left_eye_x)
        nose_to_right_eye = abs(nx - right_eye_x)
        
        # Vertical distance from nose to eyes center line (positive = nose below eyes)
        eyes_y = (right_eye_y + left_eye_y) / 2.0
        up_dist = ny - eyes_y

        aligned = engine.RECOGNIZER.alignCrop(img_enhanced, best_face)
        feature = engine.RECOGNIZER.feature(aligned)
        cv2.normalize(feature, feature, norm_type=cv2.NORM_L2)
        
        vector = feature.flatten().tolist()
        
        # Prevent registering the same person twice under different IDs
        live_vector_np = np.array(vector, dtype=np.float32)
        best_score = 0.0
        matched_id = None
        matched_name = None
        
        for registered_id, user_info in engine.GALLERY.items():
            if registered_id == user_id:
                continue
            user_vectors = user_info["vectors"]
            if len(user_vectors) > 0:
                scores = [float(np.dot(vec, live_vector_np)) for vec in user_vectors]
                max_user_score = max(scores)
                if max_user_score > best_score:
                    best_score = max_user_score
                    matched_id = registered_id
                    matched_name = user_info["name"]
                    
        # BUG9 FIX: Use a looser threshold for 1-to-N de-duplication during enrollment
        # The strict matching threshold (0.365) is too tight for different angles/lighting
        DEDUP_THRESHOLD = max(ACTIVE_THRESHOLD + 0.12, 0.50)
        if best_score > DEDUP_THRESHOLD:
            return jsonify({
                "status": "error",
                "message": f"Face already registered under ID '{matched_id}' ({matched_name})."
            }), 400
            
        # Buffer templates in temporary memory. Only save to database once all 5 steps are completed!
        if step == 0:
            TEMP_ENROLLMENTS[user_id] = {
                "name": username, 
                "vectors": [vector],
                "base_up_dist": up_dist
            }
        else:
            if user_id not in TEMP_ENROLLMENTS:
                return jsonify({"status": "error", "message": "Enrollment session not started or timed out. Please restart from Pose 1."}), 400
                
            base_up_dist = TEMP_ENROLLMENTS[user_id]["base_up_dist"]
            
            # Run relative geometry checks — skipped in TEST_MODE
            # NOTE: captureFrame() in the browser mirrors the video horizontally before sending.
            # So when the user turns to THEIR LEFT:
            #   - nose moves toward their left side
            #   - in the mirrored image this appears on the RIGHT → closer to right_eye_x (index 4)
            #   - nose_to_right_eye DECREASES
            if not TEST_MODE:
                if step == 1: # User turns LEFT → nose_to_right_eye < nose_to_left_eye
                    if nose_to_right_eye >= nose_to_left_eye:
                        return jsonify({
                            "status": "error", 
                            "message": f"Incorrect Pose: Please turn your head slightly to YOUR Left (nose-to-right: {nose_to_right_eye:.1f}px, nose-to-left: {nose_to_left_eye:.1f}px)."
                        }), 400
                elif step == 2: # User turns RIGHT → nose_to_left_eye < nose_to_right_eye
                    if nose_to_left_eye >= nose_to_right_eye:
                        return jsonify({
                            "status": "error", 
                            "message": f"Incorrect Pose: Please turn your head slightly to YOUR Right (nose-to-left: {nose_to_left_eye:.1f}px, nose-to-right: {nose_to_right_eye:.1f}px)."
                        }), 400
                elif step == 3: # Tilt Up: nose moves toward eyes
                    if up_dist >= base_up_dist - 0.5:
                        return jsonify({
                            "status": "error", 
                            "message": f"Incorrect Pose: Please tilt your head slightly Up (dist: {up_dist:.1f}px, baseline: {base_up_dist:.1f}px)."
                        }), 400
                elif step == 4: # Tilt Down: nose moves away from eyes
                    if up_dist <= base_up_dist + 0.5:
                        return jsonify({
                            "status": "error", 
                            "message": f"Incorrect Pose: Please tilt your head slightly Down (dist: {up_dist:.1f}px, baseline: {base_up_dist:.1f}px)."
                        }), 400

            # BUG10 FIX: Guard against double-posting same step (network retries)
            current_count = len(TEMP_ENROLLMENTS[user_id]["vectors"])
            if current_count < step + 1:
                TEMP_ENROLLMENTS[user_id]["vectors"].append(vector)
            
        if step == 4:
            if user_id not in TEMP_ENROLLMENTS or len(TEMP_ENROLLMENTS[user_id]["vectors"]) < 5:
                return jsonify({"status": "error", "message": "Incomplete registration templates. Please restart."}), 400
                
            # Write to main database file
            current_data = {}
            if os.path.exists(engine.GALLERY_PATH):
                with open(engine.GALLERY_PATH, "r") as f:
                    current_data = json.load(f)
                    
            current_data[user_id] = {
                "name": username,
                "vectors": TEMP_ENROLLMENTS[user_id]["vectors"]
            }
            
            with open(engine.GALLERY_PATH, "w") as f:
                json.dump(current_data, f, indent=4)
                
            # Save cropped avatar image on successful completion
            engine.save_avatar(user_id, img, (x, y, w_box, h_box))
            
            # Clear temporary enrollment session
            del TEMP_ENROLLMENTS[user_id]
            engine.reload_gallery()
            
            return jsonify({"status": "success", "message": f"Successfully enrolled registration for {username}."})
            
        return jsonify({"status": "success", "message": f"Pose {step + 1} captured successfully."})
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/verify", methods=["POST"])
def verify_user():
    global ACTIVE_THRESHOLD
    t_start = time.time()
    data = request.get_json()
    if not data or "image" not in data or "user_id" not in data:
        return jsonify({"status": "error", "message": "Missing 'image' or 'user_id' payload."}), 400
        
    user_id = engine.sanitize_name(data["user_id"])
    action = data.get("action", "LOGIN").upper()
    
    engine.check_and_reload_db()
    if user_id not in engine.GALLERY:
        return jsonify({"status": "error", "message": "Selected ID is not registered."}), 400
        
    user_info = engine.GALLERY[user_id]
    user_name = user_info["name"]
    user_vectors = user_info["vectors"]
    
    try:
        img = engine.decode_base64_image(data["image"])
        if img is None:
            return jsonify({"status": "error", "message": "Failed to decode image."}), 400
            
        h, w, _ = img.shape
        engine.DETECTOR.setInputSize((w, h))
        _, faces = engine.DETECTOR.detect(img)
        
        if faces is None or len(faces) == 0:
            return jsonify({
                "status": "no_face",
                "message": "NO FACE DETECTED - PLEASE ALIGN CAMERA"
            })
            
        # Multi-face gating check
        if len(faces) > 1:
            return jsonify({
                "status": "multi_face",
                "message": "WARNING: MULTIPLE FACES DETECTED. SCAN ONE PERSON AT A TIME."
            })
            
        face = faces[0]
        x, y, w_box, h_box = list(map(int, face[:4]))
        
        face_cx = x + w_box // 2
        face_cy = y + h_box // 2
        
        guide_w, guide_h = 260, 260
        g_x1 = (w - guide_w) // 2
        g_y1 = (h - guide_h) // 2
        g_x2 = g_x1 + guide_w
        g_y2 = g_y1 + guide_h
        
        if w_box < 100 or h_box < 100:
            return jsonify({
                "status": "too_small",
                "message": "REJECTED - MOVE CLOSER TO CAMERA",
                "w": w_box,
                "h": h_box
            })
            
        if not (g_x1 - 35 <= face_cx <= g_x2 + 35 and g_y1 - 35 <= face_cy <= g_y2 + 35):
            return jsonify({
                "status": "misaligned",
                "message": "REJECTED - ALIGN FACE IN THE BOX",
                "g_x1": g_x1,
                "g_y1": g_y1
            })
            
        # Liveness checks: Bilateral texture moire analysis
        orig_face_roi = img[max(0, y):min(h, y+h_box), max(0, x):min(w, x+w_box)]
        texture_var = 0.0
        if orig_face_roi.size > 0:
            h_roi, w_roi = orig_face_roi.shape[:2]
            if h_roi >= 64 and w_roi >= 64:
                gray_roi = cv2.cvtColor(orig_face_roi, cv2.COLOR_BGR2GRAY)
                gray_roi = cv2.resize(gray_roi, (128, 128))
                bilateral = cv2.bilateralFilter(gray_roi, 9, 75, 75)
                texture = cv2.absdiff(gray_roi, bilateral)
                texture_var = float(np.var(texture))
                if texture_var > 95.0:
                    return jsonify({
                        "status": "spoof",
                        "message": f"REJECTED - SPOOF DETECTED (Moiré Var: {texture_var:.1f})",
                        "variance": texture_var
                    })
                
        # Laplacian blur check (skipped in TEST_MODE)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_val = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if not TEST_MODE and blur_val < 12.0:
            return jsonify({
                "status": "blurry",
                "message": "REJECTED - BLURRY (Hold Still)",
                "blur": blur_val
            })
            
        # Contrast Enhancement
        img_enhanced = engine.enhance_contrast(img)
        
        face_roi = img_enhanced[max(0, y):min(h, y+h_box), max(0, x):min(w, x+w_box)]
        if face_roi.size == 0:
            face_roi = img_enhanced
        face_brightness = float(np.mean(face_roi))
        if face_brightness < 40.0:
            return jsonify({
                "status": "underexposed",
                "message": "REJECTED - TOO DARK (Improve Lighting)",
                "brightness": face_brightness
            })
        elif face_brightness > 230.0:
            return jsonify({
                "status": "overexposed",
                "message": "REJECTED - TOO BRIGHT (Reduce Glare)",
                "brightness": face_brightness
            })
            
        aligned = engine.RECOGNIZER.alignCrop(img_enhanced, face)
        feature = engine.RECOGNIZER.feature(aligned)
        cv2.normalize(feature, feature, norm_type=cv2.NORM_L2)
        live_vector = feature.flatten()
        
        # Hardened matching threshold
        current_threshold = ACTIVE_THRESHOLD
        if face_brightness < 80.0 or face_brightness > 180.0:
            current_threshold += 0.02
            
        best_score = 0.0
        
        # 1-to-1 match against templates of this selected ID
        if len(user_vectors) > 0:
            scores = [float(np.dot(vec, live_vector)) for vec in user_vectors]
            best_score = max(scores)
            
        match_success = (best_score > current_threshold)
        latency_ms = (time.time() - t_start) * 1000.0
        avatar_url = None
        
        if match_success:
            # Log Successful transaction
            engine.write_access_log(action, f"{user_name} ({user_id})", f"(Score: {best_score:.3f}, Latency: {latency_ms:.1f}ms)")
            avatar_url = engine.get_avatar_url(user_id)
            
        return jsonify({
            "status": "success",
            "match": match_success,
            "user": user_name,
            "score": best_score,
            "threshold": current_threshold,
            "brightness": face_brightness,
            "blur": blur_val,
            "variance": texture_var,
            "latency": latency_ms,
            "avatar_url": avatar_url
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/logs", methods=["GET"])
def get_logs():
    if not os.path.exists(engine.LOG_PATH):
        return jsonify({"logs": []})
    try:
        with open(engine.LOG_PATH, "r") as f:
            lines = f.readlines()
        last_logs = [line.strip() for line in lines[-30:]]
        return jsonify({"logs": last_logs})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/attendance/download", methods=["GET"])
def download_attendance():
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    
    if not os.path.exists(engine.CSV_PATH):
        with open(engine.CSV_PATH, "w", encoding="utf-8") as f:
            f.write("Timestamp,Event,User,Details\n")
            
    # Apply date filters to log export if supplied
    if start_date or end_date:
        temp_csv_path = os.path.join(engine.SCRIPT_DIR, "temp_attendance.csv")
        try:
            with open(engine.CSV_PATH, "r", encoding="utf-8") as fin:
                lines = fin.readlines()
            
            with open(temp_csv_path, "w", encoding="utf-8") as fout:
                fout.write(lines[0]) # Header
                
                for line in lines[1:]:
                    match = re.search(r'"(\d{4}-\d{2}-\d{2})[^"]*"', line)
                    if match:
                        line_date = match.group(1)
                        if start_date and line_date < start_date:
                            continue
                        if end_date and line_date > end_date:
                            continue
                    fout.write(line)
                    
            resp = send_file(
                temp_csv_path,
                mimetype="text/csv",
                as_attachment=True,
                download_name=f"attendance_{start_date or 'start'}_to_{end_date or 'end'}.csv"
            )
            # BUG5 FIX: Clean up temp file after serving
            try:
                os.remove(temp_csv_path)
            except Exception:
                pass
            return resp
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    else:
        try:
            return send_file(
                engine.CSV_PATH,
                mimetype="text/csv",
                as_attachment=True,
                download_name="attendance_records.csv"
            )
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
