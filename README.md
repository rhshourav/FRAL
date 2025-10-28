# 🧠 FRAL – Face Recognition Auto Lock (with Pose Awareness)

**FRAL** (Face Recognition Auto Lock) is a smart workstation security tool that automatically **locks your computer** when the **authorized person** (you) leaves or stands up — even if there are **other people** in front of the camera.

It combines:
- 🧍 **YOLOv8 Pose Detection** (to track posture: sitting/standing)
- 🧑‍💻 **Face Recognition** (to identify the authorized user)
- 🔒 **Auto Lock Mechanism** (for Windows systems)

---

## 🚀 Features

✅ Recognizes only the **authorized user** (ignores others)  
✅ Detects posture (**Sitting / Standing**) using YOLOv8-pose  
✅ Locks workstation if the authorized user:
- Stands up and leaves the frame  
- Is not visible for a certain time (`LOCK_DELAY`)  
✅ Runs on Windows with standard webcam  
✅ Lightweight and real-time (~20–25 FPS depending on model)

---

## 🛠️ Requirements

- Python 3.9+
- A working webcam
- Windows OS (for auto-lock via `ctypes.windll.user32.LockWorkStation`)

### Python Dependencies

Install everything in one go:

```bash
pip install ultralytics face_recognition opencv-python numpy dlib
```

---

## 📂 Folder Structure

```
FRAL/
│
├── TranningData/           # Folder containing known face(s)
│   └── Shourav.jpg         # Example: your face image
│
├── merged_face_pose_lock.py
└── README.md
```

> **Note:** Each file in `TranningData/` should contain a single clear image of one known face.  
> The filename (without extension) is used as the name label.

---

## ⚙️ Configuration

You can adjust settings inside `merged_face_pose_lock.py`:

| Variable | Description | Default |
|-----------|--------------|----------|
| `FACES_DIR` | Folder containing known faces | `"TranningData"` |
| `POSE_MODEL` | YOLO pose model file | `"yolov8n-pose.pt"` |
| `LOCK_DELAY` | Seconds before workstation locks | `5` |
| `FACE_SKIP_FRAMES` | Run face recognition every N frames | `2` |

Use `yolov8s-pose.pt` for higher accuracy (slower) or `yolov8n-pose.pt` for faster detection.

---

## 🧩 How It Works

1. Loads known faces from the `TranningData` folder.  
2. Continuously captures frames from the webcam.  
3. Checks if the **known person** is in view.  
4. Uses YOLOv8 Pose Estimation to determine posture (sitting / standing).  
5. If the person:
   - Stands up and leaves, or  
   - Is not seen for more than `LOCK_DELAY` seconds  
   → Locks the Windows workstation automatically.  

🧠 **Other people do not prevent locking** — only the known face matters.

---

## ▶️ Usage

1. Place your photo(s) inside the `TranningData/` folder.  
2. Run the script:

   ```bash
   python merged_face_pose_lock.py
   ```

3. The app will show:
   - ✅ Green text when your face is detected  
   - ❌ Red text when you’re gone  
   - 💺 “Standing / Sitting” status on screen  
4. When you leave → 💻 the system automatically locks.

---

## 🧱 Example Output

```
[*] Pose model loaded.
[+] Loaded Shourav.jpg
[*] Camera started. Press 'q' to quit.
[🔒] Locking workstation - known person left or stood up.
```

Display window example:

```
User: Shourav
Posture: Sitting
```

---

## 🧠 Future Ideas

- 🔓 Auto-unlock when authorized face reappears  
- 🪟 Cross-platform support (Linux/Mac)  
- 🧩 System tray background mode  
- 🧍 Multi-person monitoring with access levels  

---

## 📜 License

This project is open-source under the **MIT License**.  
Feel free to modify and use it for personal or research purposes.

---

**Author:** [RH Shourav](https://github.com/rhshoura)  
**Repository:** [github.com/rhshoura/FRAL](https://github.com/rhshoura/FRAL)

> “Secure your workstation — intelligently.”
