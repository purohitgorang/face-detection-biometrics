import cv2
import os
import json
import numpy as np
import base64
import time
import re

# Resolve absolute paths dynamically relative to script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GALLERY_PATH = os.path.join(SCRIPT_DIR, "embeddings_gallery.json")
YUNET_PATH = os.path.join(SCRIPT_DIR, "yunet.onnx")
SFACE_PATH = os.path.join(SCRIPT_DIR, "sface.onnx")
LOG_PATH = os.path.join(SCRIPT_DIR, "access_log.txt")
CSV_PATH = os.path.join(SCRIPT_DIR, "attendance.csv")
AVATARS_DIR = os.path.join(SCRIPT_DIR, "static", "avatars")

os.makedirs(AVATARS_DIR, exist_ok=True)

# Global caches
DETECTOR = None
RECOGNIZER = None
EYE_CASCADE = None
GALLERY = {}
LAST_LOAD_TIME = 0

def sanitize_name(name):
    if not name:
        return ""
    # Remove special characters, keep alphanumeric, replace spaces/dashes with underscores and lowercase strictly
    s = re.sub(r'[^a-zA-Z0-9\s_-]', '', name).strip()
    return re.sub(r'[\s-]+', '_', s).lower()

def write_access_log(event, username, details=""):
    sanitized_user = sanitize_name(username)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {event:<16} | User: {sanitized_user} {details}\n"
    
    try:
        with open(LOG_PATH, "a") as f:
            f.write(log_line)
    except Exception as e:
        print(f"[-] Failed to write log: {e}")
        
    try:
        write_header = not os.path.exists(CSV_PATH)
        with open(CSV_PATH, "a", encoding="utf-8") as f:
            if write_header:
                f.write("Timestamp,Event,User,Details\n")
            f.write(f'"{timestamp}","{event}","{sanitized_user}","{details}"\n')
    except Exception as e:
        print(f"[-] Failed to write CSV: {e}")

def load_models_and_db():
    global DETECTOR, RECOGNIZER, EYE_CASCADE
    print("[*] Loading YuNet, SFace, and Eye Cascade models...")
    DETECTOR = cv2.FaceDetectorYN.create(YUNET_PATH, "", (320, 320), 0.8, 0.3, 5000)
    RECOGNIZER = cv2.FaceRecognizerSF.create(SFACE_PATH, "")
    eye_xml_path = os.path.join(SCRIPT_DIR, "haarcascade_eye.xml")
    EYE_CASCADE = cv2.CascadeClassifier(eye_xml_path)
    reload_gallery()

def check_and_reload_db():
    global LAST_LOAD_TIME
    if not os.path.exists(GALLERY_PATH):
        if len(GALLERY) > 0:
            reload_gallery()
        return
        
    mtime = os.path.getmtime(GALLERY_PATH)
    if mtime > LAST_LOAD_TIME:
        reload_gallery()

def reload_gallery():
    global GALLERY, LAST_LOAD_TIME
    if not os.path.exists(GALLERY_PATH):
        GALLERY = {}
        LAST_LOAD_TIME = 0
        return
        
    try:
        with open(GALLERY_PATH, "r") as f:
            gallery_data = json.load(f)
        
        GALLERY = {}
        needs_migration = False
        migrated_data = {}
        
        for uid, user_info in gallery_data.items():
            sanitized_id = sanitize_name(uid)
            if isinstance(user_info, dict) and "name" in user_info and "vectors" in user_info:
                name = sanitize_name(user_info["name"])
                vecs = [np.array(v, dtype=np.float32) for v in user_info["vectors"]]
                GALLERY[sanitized_id] = {"name": name, "vectors": vecs}
                migrated_data[sanitized_id] = {"name": name, "vectors": [v.tolist() for v in vecs]}
            elif isinstance(user_info, list):
                # Old format conversion: set ID and Name equal to the folder key
                name = sanitized_id
                vecs = [np.array(v, dtype=np.float32) for v in user_info]
                GALLERY[sanitized_id] = {"name": name, "vectors": vecs}
                migrated_data[sanitized_id] = {"name": name, "vectors": [v.tolist() for v in vecs]}
                needs_migration = True
        
        if needs_migration:
            print("[*] Migrating database to new Unique ID format...")
            with open(GALLERY_PATH, "w") as f:
                json.dump(migrated_data, f, indent=4)
            
        LAST_LOAD_TIME = os.path.getmtime(GALLERY_PATH)
    except Exception as e:
        print(f"[-] Failed to load database: {e}")

def check_eyes_open(img, face):
    global EYE_CASCADE
    if EYE_CASCADE is None or EYE_CASCADE.empty():
        return True
        
    try:
        h, w, _ = img.shape
        lx, ly = int(face[4]), int(face[5])
        rx, ry = int(face[6]), int(face[7])
        
        w_box = face[2]
        # Proportional radius: 14% of face bounding box width to include eyebrow/socket context
        r = max(24, int(w_box * 0.14))
        
        ly1, ly2 = max(0, ly - r), min(h, ly + r)
        lx1, lx2 = max(0, lx - r), min(w, lx + r)
        ry1, ry2 = max(0, ry - r), min(h, ry + r)
        rx1, rx2 = max(0, rx - r), min(w, rx + r)
        
        left_crop = img[ly1:ly2, lx1:lx2]
        right_crop = img[ry1:ry2, rx1:rx2]
        
        if left_crop.size > 0 and right_crop.size > 0:
            left_gray = cv2.cvtColor(left_crop, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_crop, cv2.COLOR_BGR2GRAY)
            
            # Equalize histogram to handle shadows and poor lighting
            left_gray = cv2.equalizeHist(left_gray)
            right_gray = cv2.equalizeHist(right_gray)
            
            # Detect eyes with sensitive scaleFactor and minNeighbors=1 for high reliability
            left_eyes = EYE_CASCADE.detectMultiScale(left_gray, 1.05, 1, minSize=(10, 10))
            right_eyes = EYE_CASCADE.detectMultiScale(right_gray, 1.05, 1, minSize=(10, 10))
            
            # Requiring either eye to be open makes it robust to head turns or hair blocking
            return len(left_eyes) > 0 or len(right_eyes) > 0
    except Exception as e:
        print(f"[-] Eye check failed: {e}")
    return False

def save_avatar(user_id, frame, face_box):
    sanitized_id = sanitize_name(user_id)
    try:
        h, w, _ = frame.shape
        x, y, w_box, h_box = face_box
        
        pad_w = int(w_box * 0.15)
        pad_h = int(h_box * 0.15)
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(w, x + w_box + pad_w)
        y2 = min(h, y + h_box + pad_h)
        
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size > 0:
            avatar_path = os.path.join(AVATARS_DIR, f"{sanitized_id}.jpg")
            cv2.imwrite(avatar_path, face_crop)
            return f"/static/avatars/{sanitized_id}.jpg"
    except Exception as e:
        print(f"[-] Failed to save avatar: {e}")
    return None

def get_avatar_url(user_id):
    sanitized_id = sanitize_name(user_id)
    avatar_path = os.path.join(AVATARS_DIR, f"{sanitized_id}.jpg")
    if os.path.exists(avatar_path):
        return f"/static/avatars/{sanitized_id}.jpg"
    return None

def enhance_contrast(bgr_img):
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def decode_base64_image(base64_str):
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    img_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def estimate_head_pose(face):
    try:
        # Landmarks: Right Eye (4,5), Left Eye (6,7), Nose (8,9), Right Mouth (10,11), Left Mouth (12,13)
        rex, rey = float(face[4]), float(face[5])
        lex, ley = float(face[6]), float(face[7])
        nx, ny = float(face[8]), float(face[9])
        rmx, rmy = float(face[10]), float(face[11])
        lmx, lmy = float(face[12]), float(face[13])
        
        # Horizontal Eye span
        eye_span_x = abs(lex - rex)
        if eye_span_x == 0:
            return "straight", 0.5, 0.5
            
        # Yaw estimation: relative horizontal position of nose between eyes
        yaw_ratio = (nx - rex) / eye_span_x
        
        # Vertical Span: eyes center to mouth center
        eyes_y = (rey + ley) / 2.0
        mouth_y = (rmy + lmy) / 2.0
        vert_span = abs(mouth_y - eyes_y)
        if vert_span == 0:
            return "straight", yaw_ratio, 0.5
            
        # Pitch estimation: relative vertical position of nose between eyes and mouth
        pitch_ratio = (ny - eyes_y) / vert_span
        
        # Classify pose based on swapped yaw (for mirroring) and relaxed thresholds
        pose = "straight"
        if yaw_ratio < 0.43:
            pose = "left"
        elif yaw_ratio > 0.57:
            pose = "right"
        elif pitch_ratio < 0.46:
            pose = "up"
        elif pitch_ratio > 0.58:
            pose = "down"
            
        return pose, yaw_ratio, pitch_ratio
    except Exception:
        return "straight", 0.5, 0.5
