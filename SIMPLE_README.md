# 🔒 Biometric Face Verification System
### Quick Start Guide — For Non-Technical Users

---

## How to Start the System

You do not need to write any code. Just follow these 3 steps:

**Step 1**: Open the project folder and **double-click `setup.bat`**

**Step 2**: A black terminal window will open and set up the software automatically.
Keep it open in the background — do not close it.

**Step 3**: Open Google Chrome or Microsoft Edge and go to:

> 🌐 **`http://localhost:5000`**

---

## What You Will See

The home page has two buttons:

### 👤 Employee Portal
Used by **employees** to clock in / clock out.

1. Select your name from the dropdown list
2. Click **"Proceed to Face Verify"**
3. Look at the camera — the system will ask you to **blink once**
4. The system verifies your face and shows ✅ Success or ❌ Rejected

### 🛠 Admin Console
Used by **managers** to register new employees, view logs, and export attendance.

- **Register New Employee**: Enter an ID and name, then follow the 5-pose camera wizard (look straight, left, right, up, down)
- **Delete Employee**: Select a profile from the dropdown and click Delete
- **Export Attendance**: Choose a date range and click Download CSV
- **Adjust Strictness**: Slide the threshold bar to make matching more or less strict

---

## What Each File Does (In Plain English)

| File / Folder | What It Is |
|---------------|-----------|
| `setup.bat` | The on/off switch. Double-click to start everything. |
| `app.py` | The brain of the website. Handles all user requests. |
| `biometrics_engine.py` | The AI scanner. Detects faces, checks for blinks, and matches identities. |
| `yunet.onnx` | AI model that finds faces in the camera image. |
| `sface.onnx` | AI model that recognizes whose face it is. |
| `haarcascade_eye.xml` | Model that detects open/closed eyes for blink verification. |
| `embeddings_gallery.json` | The database. Stores all registered employee face data. |
| `access_log.txt` | A record of every time someone verified their face. |
| `attendance.csv` | Same records, but in a spreadsheet format you can open in Excel. |
| `templates/` | The visual designs for the web pages you see in the browser. |
| `static/avatars/` | Folder where employee photos are automatically saved during registration. |
| `requirements.txt` | A list of software components the system needs to run (installed automatically). |

---

## Frequently Asked Questions

**Q: The page says "No Registered Profiles" — what do I do?**
Go to the Admin Console (`http://localhost:5000/admin`) and register at least one employee first.

**Q: My face keeps being rejected — what should I do?**
- Make sure the room is well-lit (not too dark, not a bright light behind you)
- Hold your face steady inside the guide box on screen
- Make sure no one else is visible in the camera frame
- Remove sunglasses or heavily tinted glasses

**Q: The system says "SPOOF DETECTED" but I am a real person**
This happens when the camera picks up high-frequency patterns — usually from wearing a patterned shirt near your face, or from reflections. Try moving to a plain background.

**Q: How do I stop the server?**
Click on the black terminal window and press `Ctrl + C`.

**Q: Can I use this on multiple computers?**
Yes. Install the system on one computer and access it from others on the same network using that computer's IP address instead of `localhost` (e.g. `http://192.168.1.10:5000`).

---

## Security Notes

- Employee face data is stored **locally** on this computer only — nothing is sent to the internet
- Face data is stored as mathematical vectors (128 numbers), not as photographs
- The system requires a real human to blink in front of the camera — photos and phone screens are automatically rejected
- The Admin Console has no password by default — it is designed for internal network use only
