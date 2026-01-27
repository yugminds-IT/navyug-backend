# College Face Database

This directory contains the college face database for the incident video analysis system.

## Directory Structure

Each person should have their own folder with the following structure:

```
college_faces/
├── STUDENT_001/
│   ├── name.txt          # Person's name (one line)
│   └── face.jpg          # Clear face photo
├── STUDENT_002/
│   ├── name.txt
│   └── face.jpg
└── STUDENT_003/
    ├── name.txt
    └── face.jpg
```

## Adding New Persons

1. Create a new folder with a unique ID (e.g., `STUDENT_003`, `FACULTY_001`, etc.)
2. Add a `name.txt` file containing the person's name (one line)
3. Add a face image file (supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`)
   - Name the file `face.jpg` (or `face.png`, etc.)
   - Or use any image filename - the system will auto-detect

## Face Image Requirements

For best results, face images should:
- ✅ Show a clear, frontal view of the face
- ✅ Have good lighting (avoid shadows)
- ✅ Contain only ONE face
- ✅ Be at least 200x200 pixels
- ✅ Show the person looking at the camera
- ❌ Avoid sunglasses or face coverings
- ❌ Avoid extreme angles or side profiles

## Example

**STUDENT_001/name.txt:**
```
John Doe
```

**STUDENT_001/face.jpg:**
- A clear frontal photo of John Doe

## Loading the Database

The database is automatically loaded when the FastAPI server starts:

```bash
uvicorn app:app --reload
```

You'll see output like:
```
Loading college faces from: d:\FaceDetection\college_faces
Loaded: STUDENT_001 - John Doe
Loaded: STUDENT_002 - Jane Smith

Total persons loaded: 2
```

## Troubleshooting

### "No face detected in image for STUDENT_XXX"
- The image doesn't contain a detectable face
- Try a different photo with better lighting and frontal view

### "No face image found for STUDENT_XXX"
- The folder doesn't contain a supported image file
- Add a `.jpg`, `.jpeg`, `.png`, or `.bmp` file

### Person not being matched in videos
- Check if the face image quality is good
- Ensure the person's appearance in the video is similar to the database photo
- Adjust `FACE_MATCH_THRESHOLD` in `config.py` if needed
