# Face Recognition System - Quick Start Guide

## 🚀 How to Run

### 1. Activate Virtual Environment (if not already activated)
```bash
# Windows
.\env\Scripts\Activate.ps1

# Or if using Command Prompt
.\env\Scripts\activate.bat
```

### 2. Start the Server
```bash
uvicorn app:app --reload
```

You should see:
```
============================================================
Starting Face Recognition System...
============================================================

Loading college faces from: D:\FaceDetection\college_faces
Loaded: STUDENT_001 - Manith Mettu
Loaded: STUDENT_002 - Shiva Krishna Baigalla
Loaded: STUDENT_003 - Sai Kiran Reddy
Loaded: STUDENT_004 - Mahendar Rao Jella
Loaded: STUDENT_005 - [Name]

Total persons loaded: 5

============================================================
[OK] Server ready!
  API Docs: http://localhost:8000/docs
============================================================
```

### 3. Access the API

**Option A: Use Swagger UI (Recommended for Testing)**
1. Open browser: http://localhost:8000/docs
2. Click on `POST /incident/upload-video`
3. Click "Try it out"
4. Upload your incident video
5. Copy the `video_id` from response
6. Use `GET /incident/status/{video_id}` to check progress
7. Use `GET /incident/results/{video_id}` to get matched persons

**Option B: Use cURL**
```bash
# Upload video
curl -X POST -F "video=@data/college_2.mp4" http://localhost:8000/incident/upload-video

# Check status (replace VIDEO_ID with actual ID from upload response)
curl http://localhost:8000/incident/status/VIDEO_ID

# Get results
curl http://localhost:8000/incident/results/VIDEO_ID
```

**Option C: Use Python**
```python
import requests

# Upload video
with open('data/college_2.mp4', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/incident/upload-video',
        files={'video': f}
    )
    video_id = response.json()['video_id']
    print(f"Video ID: {video_id}")

# Check status
status = requests.get(f'http://localhost:8000/incident/status/{video_id}')
print(status.json())

# Get results (when completed)
results = requests.get(f'http://localhost:8000/incident/results/{video_id}')
print(results.json())
```

---

## 📁 Project Structure

```
FaceDetection/
├── app.py                    # FastAPI server - START HERE
├── config.py                 # Configuration settings
├── requirements.txt          # Dependencies
├── README.md                 # Full documentation
├── QUICKSTART.md            # This file
│
├── core/                     # Core components
│   ├── face_detector.py      # Face detection + recognition
│   ├── face_recognizer.py    # Similarity matching
│   ├── face_database.py      # In-memory database
│   ├── frame_extractor.py    # Video frame extraction
│   └── video_processor.py    # Async processing pipeline
│
├── college_faces/            # Face database
│   ├── STUDENT_001/
│   │   ├── name.txt          # "Manith Mettu"
│   │   └── face.jpg
│   ├── STUDENT_002/
│   │   ├── name.txt          # "Shiva Krishna Baigalla"
│   │   └── face.jpg
│   └── ...
│
├── uploads/                  # Uploaded videos (auto-created)
└── data/                     # Sample test videos
```

---

## 🎯 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info & database size |
| `/incident/upload-video` | POST | Upload video, get `video_id` |
| `/incident/status/{video_id}` | GET | Check processing status |
| `/incident/results/{video_id}` | GET | Get matched persons |
| `/incident/{video_id}` | DELETE | Delete video & results |

---

## 📊 Expected Response Format

### Upload Response
```json
{
  "status": "success",
  "video_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Video uploaded successfully. Processing started.",
  "filename": "incident.mp4",
  "size_mb": 5.08
}
```

### Status Response (Processing)
```json
{
  "video_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "filename": "incident.mp4",
  "progress": {
    "frames_processed": 12,
    "total_frames": 22,
    "percentage": 54.55
  }
}
```

### Results Response (Completed)
```json
{
  "status": "completed",
  "video_id": "550e8400-e29b-41d4-a716-446655440000",
  "total_frames_processed": 22,
  "faces_detected": 8,
  "matched_persons": [
    {
      "person_id": "STUDENT_001",
      "name": "Manith Mettu",
      "confidence": 0.87,
      "first_seen_frame": 3,
      "last_seen_frame": 18,
      "total_appearances": 6
    }
  ],
  "processing_time_seconds": 8.5
}
```

---

## ⚙️ Configuration

Edit `config.py` to customize:

```python
FACE_DETECTION_CONFIDENCE = 0.8  # Min confidence for detection (0-1)
FACE_MATCH_THRESHOLD = 0.6       # Min similarity for matching (0-1)
FRAME_EXTRACTION_RATE = 1        # Frames per second
MAX_VIDEO_SIZE_MB = 500          # Max upload size
USE_GPU = False                   # Set True if GPU available
```

---

## 🔧 Troubleshooting

### Server won't start
```bash
# Make sure virtual environment is activated
.\env\Scripts\Activate.ps1

# Check if dependencies are installed
pip list | Select-String -Pattern "fastapi|uvicorn|insightface"

# Reinstall if needed
pip install -r requirements.txt
```

### No faces detected in database
- Check that each `STUDENT_XXX` folder has:
  - `name.txt` file with person's name
  - `face.jpg` (or .png, .bmp) with clear face photo
- Restart server to reload database

### Low matching confidence
- Use clearer face images in database
- Ensure good lighting in both database photos and incident videos
- Adjust `FACE_MATCH_THRESHOLD` in `config.py` (lower = more lenient)

### Port already in use
```bash
# Use different port
uvicorn app:app --reload --port 8001
```

---

## 🎓 Adding More Students

1. Create new folder: `college_faces/STUDENT_006/`
2. Add `name.txt` with student name
3. Add `face.jpg` with clear frontal photo
4. Restart server (it will auto-load new faces)

---

## 📝 Notes

- **Processing is async**: Upload returns immediately, processing happens in background
- **Frame rate**: Extracts 1 frame/second (configurable)
- **Storage**: In-memory for POC (can upgrade to PostgreSQL later)
- **GPU**: Set `USE_GPU = True` in config.py for faster processing

---

## 📚 Full Documentation

See [README.md](file:///d:/FaceDetection/README.md) for complete documentation.
