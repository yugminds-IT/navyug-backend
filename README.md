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

## 🐳 Docker: What it is and how the process works

If you're new to Docker, this section explains the basics and the exact steps for this project.

### What is Docker?

- **Image** = A snapshot of your app and everything it needs (Python, dependencies, your code). Think of it like a recipe + packaged ingredients.
- **Container** = A running instance of that image. Like turning the recipe into an actual running app.
- **Dockerfile** = The instructions to build the image (e.g. “use Python 3.11”, “install these packages”, “copy my code”, “run this command”).

So: you write a **Dockerfile** → Docker **builds** an **image** → you **run** that image as a **container**. The same image runs the same way on your laptop, Coolify, or any server.

### The process in 4 steps

| Step | What you do | What happens |
|------|-------------|--------------|
| **1. Install Docker** | Install Docker Desktop (Mac/Windows) or Docker Engine (Linux) | You get the `docker` command to build and run containers |
| **2. Build** | Run `docker build -t face-recognition-api:latest .` in the project folder | Docker reads the Dockerfile, installs Python/deps, copies your code, and creates an image named `face-recognition-api:latest` |
| **3. Run (test)** | Run `docker run -p 8000:8000 --env-file .env face-recognition-api:latest` | A container starts; your app listens on port 8000. You can open http://localhost:8000 |
| **4. Deploy** | Push the image to a registry (e.g. Docker Hub), then in Coolify choose “Deploy from image” and set env vars | Coolify (or any host) pulls the image and runs it; your app is live |

### Step 1: Install Docker

- **Mac / Windows:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) — install and start it. You should see the Docker icon in the menu bar.
- **Linux:** Install Docker Engine (e.g. `apt install docker.io` or follow [docs.docker.com](https://docs.docker.com/engine/install/)).

Check it works:

```bash
docker --version
# Example: Docker version 24.0.x
```

### Step 2: Build the image

Open a terminal in the **project root** (the folder that contains `Dockerfile` and `app.py`).

```bash
cd /path/to/Face-detect-college-POC
docker build -t face-recognition-api:latest .
```

- `docker build` = build an image.
- `-t face-recognition-api:latest` = tag (name) the image so you can refer to it later.
- `.` = use the current folder as “build context” (Docker will read the Dockerfile and copy files from here).

The first time can take a few minutes (downloading Python base image, installing packages). When it finishes, you’ll have an image named `face-recognition-api:latest`. List it:

```bash
docker images
# You should see face-recognition-api   latest   ...
```

### Step 3: Run the container (test on your machine)

Your app needs a database and (optionally) S3. The easiest way is to pass your existing `.env` file so the container gets `DATABASE_URL`, `SECRET_KEY`, etc.

```bash
docker run --rm -p 8000:8000 --env-file .env face-recognition-api:latest
```

- `docker run` = start a container from an image.
- `--rm` = remove the container when you stop it (keeps things tidy).
- `-p 8000:8000` = map port 8000 on your machine to port 8000 in the container (so you can open http://localhost:8000).
- `--env-file .env` = load environment variables from `.env` (database, S3, secrets).
- `face-recognition-api:latest` = which image to run.

You should see logs (migrations, “Uvicorn running on http://0.0.0.0:8000”). Then:

- Open **http://localhost:8000/docs** for the API.
- Open **http://localhost:8000/health** to see health status.

To stop: press `Ctrl+C` in the terminal.

### Step 4: Push the image and deploy (e.g. Coolify)

To run the same image on a server (Coolify on Hostinger), you put the image in a **registry** so the server can download it.

1. **Create an account** on [Docker Hub](https://hub.docker.com) (or use [GitHub Container Registry](https://ghcr.io)).
2. **Log in** on your machine:
   ```bash
   docker login
   # Enter your Docker Hub username and password
   ```
3. **Tag the image** with your username and (optionally) a version:
   ```bash
   docker tag face-recognition-api:latest YOUR_DOCKERHUB_USERNAME/face-recognition-api:latest
   ```
   Replace `YOUR_DOCKERHUB_USERNAME` with your actual Docker Hub username.
4. **Push** the image:
   ```bash
   docker push YOUR_DOCKERHUB_USERNAME/face-recognition-api:latest
   ```
5. **In Coolify:** Create a new application → choose **“Docker Image”** (or “Pre-built image”) → set:
   - **Image:** `YOUR_DOCKERHUB_USERNAME/face-recognition-api:latest`
   - **Port:** `8000`
   - **Environment variables:** Add the same ones as in your `.env` (e.g. `DATABASE_URL`, `SECRET_KEY`, S3 vars). Do not upload your `.env` file; type or paste each variable in Coolify.

Coolify will pull the image and run it. Point your domain to that Coolify app and you’re deployed.

### Quick reference

| Goal | Command |
|------|--------|
| Build image | `docker build -t face-recognition-api:latest .` |
| Run locally | `docker run --rm -p 8000:8000 --env-file .env face-recognition-api:latest` |
| List images | `docker images` |
| Tag for Docker Hub | `docker tag face-recognition-api:latest USERNAME/face-recognition-api:latest` |
| Push to Docker Hub | `docker push USERNAME/face-recognition-api:latest` |
| Stop a running container | `Ctrl+C` in the terminal where it’s running |

### Optional: use the build script

We provide a script that builds (and optionally tags and pushes) in one go:

```bash
./scripts/build-docker.sh                                    # build only
./scripts/build-docker.sh -t USERNAME/face-recognition-api:latest --push   # build, tag, push
```

---

## 🐳 Build Docker image & deploy

### Build the image

From the project root:

```bash
# Build only (tag: face-recognition-api:latest)
docker build -t face-recognition-api:latest .

# Or use the script (same result)
chmod +x scripts/build-docker.sh
./scripts/build-docker.sh
```

**Tag and push to a registry** (for Coolify “pull image” or any Docker host):

```bash
# Docker Hub
./scripts/build-docker.sh -t YOUR_DOCKERHUB_USER/face-recognition-api:latest --push

# GitHub Container Registry
./scripts/build-docker.sh -t ghcr.io/YOUR_GITHUB_USER/face-recognition-api:latest --push

# Any private registry
./scripts/build-docker.sh -t registry.yourdomain.com/face-recognition-api:v1.0 --push
```

Log in first if needed: `docker login` (Docker Hub) or `docker login ghcr.io` (GHCR).

### Run locally (test)

**Option A: Use your existing database** (e.g. remote Hostinger DB)

```bash
docker run --rm -p 8000:8000 --env-file .env face-recognition-api:latest
```

If you see **"Connection refused"** to the DB host (e.g. `72.61.241.118:5432`), the container cannot reach PostgreSQL. Common causes:
- The database server is down or not listening on that IP/port.
- A firewall or cloud security group is blocking port 5432 from your IP (or from the machine running the container).
- PostgreSQL is bound only to `127.0.0.1` and not to the public IP.

Fix by ensuring the DB is running, listening on the right interface, and that your IP (or the server’s IP when you deploy) is allowed. For **local testing**, use Option B below.

**Option B: Run app + PostgreSQL with Docker Compose** (no remote DB needed)

This starts a local Postgres and the app; the app uses it via `DATABASE_URL=postgresql://postgres:postgres@db:5432/face_recognition`:

```bash
# Start Postgres + app (app builds from Dockerfile)
docker compose up --build

# Or use the pre-built image: in docker-compose.yml set "image: face-ai" and "build: ." → comment build, use image
```

Then open `http://localhost:8000/docs` and `http://localhost:8000/health`. To use a **pre-built** image (e.g. `face-ai`) with compose, add under `app:`: `image: face-ai` and comment out or remove the `build: .` line so compose doesn’t rebuild.

### Deploy

- **Coolify**: In Coolify create an application and either **build from Git** (Dockerfile) or **use a pre-built image** (set image to `YOUR_REGISTRY/face-recognition-api:latest`). Set port **8000** and all required env vars.
- **Any Docker host**: `docker pull YOUR_REGISTRY/face-recognition-api:latest` then run the container with `DATABASE_URL` and other env vars, exposing port 8000.

---

## 🐳 Deploy with Coolify (Hostinger KVM 1)

Deploy this app as a Docker image on [Coolify](https://coolify.io) running on your Hostinger KVM 1 server.

### 1. Coolify on the KVM

- Install Coolify on the KVM (one-click or [self-hosting guide](https://coolify.io/docs/self-hosting)).
- Ensure Docker is available on the server (Coolify uses it to build and run containers).

### 2. Database (PostgreSQL)

The app needs PostgreSQL. In Coolify:

- **Option A**: Add a **PostgreSQL** service (Coolify → Add Resource → Database → PostgreSQL). Note the internal URL (e.g. `postgresql://user:pass@postgres:5432/dbname`).
- **Option B**: Use an external DB (e.g. Hostinger DB or your existing `72.61.241.118`). Use that connection string as `DATABASE_URL`.

### 3. Create the Application in Coolify

- **New Resource** → **Application** → **Docker Compose** or **Dockerfile** (recommended: **Public Repository** or **Dockerfile**).
- Connect your Git repo (this project) and set:
  - **Build Pack**: Dockerfile  
  - **Dockerfile path**: `Dockerfile` (root)
  - **Port**: `8000`

If you build from **Dockerfile** (no compose), Coolify will build the image and run the container; expose port **8000**.

### 4. Environment Variables

In Coolify → your application → **Environment Variables**, set (at least):

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | Full PostgreSQL URL (from step 2) |
| `SECRET_KEY` | Strong random secret (e.g. `openssl rand -hex 32`) |
| `ENCRYPTION_KEY` | Fernet key for sensitive data (generate once, keep safe) |
| `USE_S3` | `true` if using S3/MinIO |
| `USE_S3_ONLY` | `true` for S3-only storage (recommended in production) |
| `CAMPUS_SECURITY_BUCKET_NAME` | Your bucket name |
| `S3_REGION`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY` | S3 credentials (if `USE_S3=true`) |
| `S3_ENDPOINT_URL` | Optional; for MinIO/custom S3 (e.g. `https://files.navedhana.com`) |
| `CORS_ORIGINS` | Comma-separated frontend origins (e.g. `https://yourdomain.com`) |
| `TRUSTED_HOSTS` | Comma-separated allowed hosts (e.g. `yourdomain.com,api.yourdomain.com`) |
| `ENVIRONMENT` | `production` |

Do **not** commit `.env`; configure these only in Coolify (or your CI/CD secrets).

### 5. Domain & HTTPS

- In Coolify, add a **Domain** for the application and enable **HTTPS** (Coolify can issue certificates).
- Set `CORS_ORIGINS` and `TRUSTED_HOSTS` to your real domain(s).

### 6. Volumes (optional)

If you need persistent local uploads (when not using S3-only), add a volume in Coolify:

- Container path: `/app/uploads`  
- Mount a persistent volume or host path.

With `USE_S3_ONLY=true`, uploads and face storage use S3; no local volumes are required for that.

### 7. Build & Deploy

- Trigger **Build** then **Deploy**. The Dockerfile runs `alembic upgrade head` on startup and serves the app on port 8000.
- The `/health` endpoint is used by the image **HEALTHCHECK**; Coolify can use it for readiness.

### Quick checklist

- [ ] Coolify installed on Hostinger KVM 1  
- [ ] PostgreSQL created (Coolify or external) and `DATABASE_URL` set  
- [ ] App added from Git with Dockerfile, port 8000  
- [ ] All env vars set (no `.env` in image; `.dockerignore` excludes it)  
- [ ] Domain + HTTPS configured; `CORS_ORIGINS` and `TRUSTED_HOSTS` updated  

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
