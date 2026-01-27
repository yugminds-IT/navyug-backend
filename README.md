# Incident Video Face Recognition System

## 📖 Overview
A production-ready, async video processing system that identifies individuals in incident footage by matching faces against a college database. Built with FastAPI and state-of-the-art Deep Learning models.

## 🏗️ System Architecture

```mermaid
graph TD
    Client[Client / User]
    
    subgraph "API Layer (FastAPI)"
        UploadEP[POST /upload-video]
        StatusEP[GET /status]
        ResultsEP[GET /results]
    end
    
    subgraph "Storage"
        UploadsDir[uploads/]
        FaceDBDir[college_faces/]
        JobStore[(In-Memory Job Store)]
    end
    
    subgraph "Core Processing (Async)"
        VideoProc[VideoProcessor]
        FrameExt[FrameExtractor]
        
        subgraph "Face Analysis Pipeline"
            PreProc[Pre-Processing<br/>(Gamma, Sharpen)]
            Detector[FaceDetector<br/>(RetinaFace)]
            
            subgraph "Multi-Model Embedding"
                IF[InsightFace]
                AF[ArcFace]
                FN[FaceNet]
                Fusion[Fusion Logic<br/>(Average/Concat)]
            end
            
            Dedup[Deduplication &<br/>Quality Scoring]
        end
        
        Matcher[FaceRecognizer<br/>(Cosine Similarity)]
    end
    
    Client -->|Upload Video| UploadEP
    UploadEP -->|Save File| UploadsDir
    UploadEP -->|Init Job| JobStore
    
    Client -->|Poll Status| StatusEP
    StatusEP -->|Read| JobStore
    
    Client -->|Get Results| ResultsEP
    ResultsEP -->|Read| JobStore
    
    UploadEP -.->|Trigger Background Task| VideoProc
    
    VideoProc -->|Read Video| UploadsDir
    VideoProc -->|Extract Frames| FrameExt
    FrameExt -->|Raw Frames| PreProc
    PreProc -->|Enhanced Frames| Detector
    Detector -->|Detected Faces| IF & AF & FN
    IF & AF & FN -->|Vectors| Fusion
    Fusion -->|Fused Embeddings| Dedup
    Dedup -->|Unique Best Faces| Matcher
    
    FaceDBDir -->|Load at Startup| Matcher
    Matcher -->|Match Results| JobStore
```

### Workflow
1.  **Ingestion**: User uploads video; server saves it securely.
2.  **Extraction**: Frames are extracted at 1 FPS.
3.  **Detection**: Faces are detected using **RetinaFace** with Gamma correction.
4.  **Embedding**: **Multi-Model Fusion** combines vectors from InsightFace, ArcFace, and FaceNet.
5.  **Deduplication**: Intelligent quality scoring keeps only the best face per person.
6.  **Recognition**: Matches against database using Cosine Similarity.

## 🛠️ Technology Stack

### Core Frameworks
- **Language**: Python 3.10+
- **API**: FastAPI, Uvicorn (ASGI)
- **Async**: Python `asyncio`

### Computer Vision & AI
- **Face Detection**: InsightFace (RetinaFace/SCRFD)
- **Face Recognition Models**:
    - **InsightFace**: ResNet100 (ArcFace based)
    - **FaceNet**: InceptionResnetV1 (vggface2 pretrained)
    - **ArcFace**: Helper model via InsightFace
- **Utils**: OpenCV (cv2), NumPy, PyTorch

---

## 🧠 Algorithm Details

### 1. Robust Preprocessing
Before detection, every video frame undergoes:
- **Gamma Correction**: Adjusts brightness for dark/overexposed scenes.
- **Sharpening Kernel**: Enhances facial features.
- **CLAHE**: Improves local contrast.

### 2. Multi-Model Embedding Fusion
To achieve high accuracy, we generate 3 embeddings for every face and fuse them:
- **Method**: Weighted Average or Concatenation (Configurable).
- **Benefit**: Reduced false positives; if one model is unsure, others compensate.

### 3. Intelligent Deduplication
The system encounters the same face multiple times in a video. It uses a **Quality Score** to pick the best one:
- **Metric**: `Score = 0.25*Resolution + 0.30*Sharpness + 0.20*Brightness + 0.15*Contrast + 0.10*Size`
- **Logic**: If `Similarity(Face A, Face B) > 0.8`, keep the one with higher Score.

---

## 🚀 Build & Run Guide

### 1. Prerequisites
- Python 3.10+
- [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (Windows) / Build Essentials (Linux)

### 2. Installation
```bash
# Clone
git clone <repo_url>
cd FaceDetection

# Environment
python -m venv env
.\env\Scripts\activate  # Windows
# source env/bin/activate # Mac/Linux

# Dependencies
pip install -r requirements.txt
```

### 3. Setup Database
Create `college_faces/` structure:
```text
college_faces/
├── STUDENT_001/
│   ├── name.txt       # "John Doe"
│   └── face.jpg       # Clear photo
└── STUDENT_002/
    ├── name.txt
    └── face.jpg
```

### 4. Configuration (Optional)
Edit `config.py` to tune:
- `USE_GPU = True` (Recommended for speed)
- `ENABLE_GAMMA_CORRECTION = True`
- `FACE_MATCH_THRESHOLD = 0.5`

### 5. Start Server
```bash
uvicorn app:app --reload
```

---

## 📡 API Usage

### Upload Video
```http
POST /incident/upload-video
Content-Type: multipart/form-data; boundary=boundary

--boundary
Content-Disposition: form-data; name="video"; filename="incident.mp4"
Content-Type: video/mp4

<video_bytes>
```

### Check Status
```http
GET /incident/status/{video_id}
```
**Response**:
```json
{
  "status": "processing",
  "progress": {
    "percentage": 45.5,
    "stage": "processing_stream"
  }
}
```

### Get Results
```http
GET /incident/results/{video_id}
```

---

## 📁 Project Structure
```
FaceDetection/
├── app.py                  # API Server
├── config.py               # Settings
├── requirements.txt        # Deps
├── core/
│   ├── face_detector.py    # Detection + Preprocessing
│   ├── multi_model_embedder.py # Fusion Logic
│   └── video_processor.py  # Pipeline
└── college_faces/          # Face DB
```
