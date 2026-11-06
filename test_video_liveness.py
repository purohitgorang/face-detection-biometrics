import os
import sys
import time
import urllib.request
import cv2
import numpy as np

# Add parent dir to path so we can import biometrics_engine
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import biometrics_engine as engine

VIDEO_URL = "https://github.com/intel-iot-devkit/sample-videos/raw/master/head-pose-face-detection-female.mp4"
VIDEO_PATH = "test_stranger_face.mp4"

def download_sample_video():
    if not os.path.exists(VIDEO_PATH):
        print(f"[*] Downloading test video clip from Intel OpenVINO dataset...")
        try:
            urllib.request.urlretrieve(VIDEO_URL, VIDEO_PATH)
            print("[+] Download complete!")
        except Exception as e:
            print(f"[-] Download failed: {e}")
            sys.exit(1)
    else:
        print("[*] Test video file already exists.")

def run_video_test():
    print("[*] Initializing YuNet Face Detector and Eye Cascades...")
    engine.load_models_and_db()
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("[-] Failed to open test video file.")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[+] Loaded video: {VIDEO_PATH} ({total_frames} frames, {fps:.1f} FPS)")
    
    frame_idx = 0
    blink_cycles = 0
    eyes_were_open = False
    eyes_closed_seen = False
    
    print("\n" + "="*80)
    print(f"{'Frame':<8} | {'Face Detect':<12} | {'Eye State':<12} | {'Moire Var':<10} | {'Status/Liveness':<25}")
    print("="*80)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        h, w, _ = frame.shape
        engine.DETECTOR.setInputSize((w, h))
        _, faces = engine.DETECTOR.detect(frame)
        
        face_status = "No Face"
        eye_status = "--"
        moire_status = "--"
        liveness = "Initializing"
        
        if faces is not None and len(faces) > 0:
            face_status = "Detected"
            # Get largest face
            best_face = faces[0]
            max_area = 0
            for f in faces:
                area = int(f[2]) * int(f[3])
                if area > max_area:
                    max_area = area
                    best_face = f
            
            # 1. Moire Spoof Check
            x, y, w_box, h_box = list(map(int, best_face[:4]))
            face_roi = frame[max(0, y):min(h, y+h_box), max(0, x):min(w, x+w_box)]
            texture_var = 0.0
            is_spoof = False
            
            if face_roi.size > 0:
                h_roi, w_roi = face_roi.shape[:2]
                if h_roi >= 64 and w_roi >= 64:
                    gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                    gray_roi = cv2.resize(gray_roi, (128, 128))
                    bilateral = cv2.bilateralFilter(gray_roi, 9, 75, 75)
                    texture = cv2.absdiff(gray_roi, bilateral)
                    texture_var = float(np.var(texture))
                    moire_status = f"{texture_var:.2f}"
                    if texture_var > 95.0:
                        is_spoof = True
            
            if is_spoof:
                liveness = "[SPOOF REJECTED]"
            else:
                # 2. Eye Blink state
                eyes_open = engine.check_eyes_open(frame, best_face)
                if eyes_open:
                    eye_status = "Open"
                    eyes_were_open = True
                    if eyes_closed_seen:
                        blink_cycles += 1
                        eyes_closed_seen = False
                        liveness = f"[BLINK VERIFIED] (#{blink_cycles})"
                    else:
                        liveness = "Waiting for Blink"
                else:
                    eye_status = "Closed"
                    if eyes_were_open:
                        eyes_closed_seen = True
                        liveness = "Blink Detected..."
                    else:
                        liveness = "Aligning Eyes"
                        
        print(f"{frame_idx:<8} | {face_status:<12} | {eye_status:<12} | {moire_status:<10} | {liveness:<25}")
        
        # Limit speed to match real-time playback
        time.sleep(1.0 / fps)
        
    cap.release()
    print("="*80)
    print(f"\n[+] Test complete! Total Blinks Verified: {blink_cycles}")

if __name__ == "__main__":
    download_sample_video()
    run_video_test()
