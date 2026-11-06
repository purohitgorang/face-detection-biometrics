import os
import sys
import time
import cv2
import base64
import urllib.request
import json

SERVER_URL = "http://localhost:5000"
TEST_ID = "test_user_99"
TEST_NAME = "Test User Integration"

def frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
    return base64.b64encode(buffer).decode('utf-8')

def post_json(url, data_dict):
    json_data = json.dumps(data_dict).encode('utf-8')
    req = urllib.request.Request(
        url,
        data=json_data,
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    try:
        with urllib.request.urlopen(req) as response:
            status_code = response.status
            body = response.read().decode('utf-8')
            return status_code, body
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode('utf-8')
    except Exception as e:
        return 500, str(e)

def get_json(url):
    try:
        with urllib.request.urlopen(url) as response:
            status_code = response.status
            body = response.read().decode('utf-8')
            return status_code, body
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode('utf-8')
    except Exception as e:
        return 500, str(e)

def run_integration_test():
    print("[*] Starting Registration & Verification API Integration Test...")
    
    # 1. Load the test video to extract frames
    video_path = "test_stranger_face.mp4"
    if not os.path.exists(video_path):
        print(f"[-] Missing {video_path}. Please run test_video_liveness.py first to download it.")
        sys.exit(1)
        
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < 100:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"[+] Loaded {len(frames)} test video frames.")
    
    # We will pick 5 frames to represent:
    # Pose 1 (Center): Frame 0
    # Pose 2 (Left): Frame 20
    # Pose 3 (Right): Frame 40
    # Pose 4 (Up): Frame 60
    # Pose 5 (Down): Frame 80
    selected_indices = [0, 20, 40, 60, 80]
    
    # Let's clean up any previous failed runs first
    print("[*] Performing preliminary cleanup for test ID...")
    post_json(f"{SERVER_URL}/api/delete_user", {"user_id": TEST_ID})
    
    # 2. Run through the enrollment steps
    print("\n=== Phase 1: Step-by-Step Enrollment Wizard ===")
    
    for step_idx in range(5):
        frame_to_use = frames[selected_indices[step_idx]]
        b64_img = frame_to_base64(frame_to_use)
        
        payload = {
            "id": TEST_ID,
            "name": TEST_NAME,
            "image": b64_img,
            "step": step_idx
        }
        
        print(f"[*] Submitting Pose {step_idx + 1}/5 (Step index: {step_idx})...")
        status, body = post_json(f"{SERVER_URL}/api/enroll", payload)
        print(f"    Response Status Code: {status}")
        print(f"    Response Body: {body}")
        
        if status != 200:
            print("[-] Test failed: Enrollment rejected.")
            sys.exit(1)
            
    # Verify user was created in gallery database
    print("\n[*] Verifying database registry...")
    status, body = get_json(f"{SERVER_URL}/api/users")
    users_data = json.loads(body)
    registered_ids = [u["id"] for u in users_data.get("users", [])]
    print(f"    Registered IDs: {registered_ids}")
    if TEST_ID not in registered_ids:
        print("[-] Test failed: User not committed to database after step 5.")
        sys.exit(1)
    print("[+] Database registration verified successfully!")
    
    # 3. Verify user matches straight face (1-to-1 match check)
    print("\n=== Phase 2: 1-to-1 Verification check ===")
    verify_frame = frames[20] # Sharp, in-focus face
    b64_verify = frame_to_base64(verify_frame)
    
    verify_payload = {
        "user_id": TEST_ID,
        "image": b64_verify,
        "action": "VERIFY"
    }
    
    print("[*] Submitting verification check...")
    status, body = post_json(f"{SERVER_URL}/api/verify", verify_payload)
    print(f"    Response Status Code: {status}")
    v_data = json.loads(body)
    print(f"    Response Body: {json.dumps(v_data, indent=2)}")
    
    if status == 200 and v_data.get("match") is True:
        print("[+] Face Verification SUCCESSFUL!")
    else:
        print("[-] Face Verification FAILED!")
        sys.exit(1)
        
    # 4. Clean up test user
    print("\n=== Phase 3: Cleanup ===")
    print("[*] Deleting test profile...")
    status, body = post_json(f"{SERVER_URL}/api/delete_user", {"user_id": TEST_ID})
    print(f"    Delete status: {status}")
    print("[+] Test user deleted successfully!")
    print("\n" + "="*50 + "\n[+] ALL INTEGRATION TESTS PASSED SUCCESSFULLY!\n" + "="*50)

if __name__ == "__main__":
    run_integration_test()
