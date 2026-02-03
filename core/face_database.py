import os
import re
import cv2
import numpy as np
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from collections import defaultdict
from core.face_detector import FaceDetector
from core.image_quality import remove_duplicates_keep_best
from core.logger import logger
import config

try:
    from storage.s3_paths import campus_student_embeddings_path
except ImportError:
    campus_student_embeddings_path = None

class FaceDatabase:
    """
    In-memory face database for POC.
    Stores person information and face embeddings.
    """
    
    def __init__(self, similarity_threshold: float = 0.6):
        """
        Initialize face database.
        
        Args:
            similarity_threshold: Minimum similarity for a match (0-1)
        """
        self.persons: Dict[str, Dict] = {}  # {person_id: {name, embedding}}
        self.similarity_threshold = similarity_threshold
        self.face_detector = None
    
    def load_from_directory(self, directory_path: str, ctx_id: int = -1):
        """
        Load face database from directory structure.
        NOW SUPPORTS MULTIPLE IMAGES PER PERSON for better robustness!
        
        Expected structure:
        directory_path/
            PERSON_001/
                name.txt (optional, contains person name)
                face.jpg (or face1.jpg, face2.jpg, photo1.png, etc.)
                face2.jpg (optional - additional photos)
                face3.jpg (optional - more variations)
            PERSON_002/
                name.txt
                image1.jpg
                image2.jpg
        
        Args:
            directory_path: Path to directory containing person folders
            ctx_id: GPU context ID (-1 for CPU)
        """
        # Initialize face detector with recognition
        if self.face_detector is None:
            self.face_detector = FaceDetector(ctx_id=ctx_id, recognition=True)
        
        directory = Path(directory_path)
        if not directory.exists():
            raise ValueError(f"Directory not found: {directory_path}")
        
        loaded_count = 0
        
        # Iterate through person directories
        for person_dir in directory.iterdir():
            if not person_dir.is_dir():
                continue
            
            # Skip README and other non-person directories
            if person_dir.name.upper() in ['README', 'EXTRACTED_1', 'EXTRACTED_2']:
                continue
            
            person_id = person_dir.name
            
            # Look for name file
            name_file = person_dir / "name.txt"
            if name_file.exists():
                with open(name_file, 'r', encoding='utf-8') as f:
                    person_name = f.read().strip()
            else:
                person_name = person_id  # Use ID as name if no name file
            
            # Find ALL image files in the directory
            image_files = []
            for file in person_dir.iterdir():
                if file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    image_files.append(file)
            
            if len(image_files) == 0:
                logger.warning(f"No face images found for {person_id}")
                continue
            
            # Step 1: Load all images and extract embeddings with metadata
            image_data_list = []
            
            for image_path in image_files:
                # Load image
                image = cv2.imread(str(image_path))
                if image is None:
                    logger.warning(f"Could not load {image_path.name} for {person_id}")
                    continue
                
                # Detect face and get embedding with enhanced detection
                faces = self.face_detector.detect_and_embed(
                    image,
                    preprocess=config.ENABLE_IMAGE_PREPROCESSING,
                    align_faces=config.ENABLE_FACE_ALIGNMENT
                )
                
                if len(faces) == 0:
                    logger.warning(f"No face detected in {image_path.name} for {person_id}")
                    continue
                
                # Filter by minimum face size if configured
                if config.MIN_FACE_SIZE > 0:
                    faces = [
                        f for f in faces
                        if (f["bbox"][2] - f["bbox"][0]) >= config.MIN_FACE_SIZE and
                           (f["bbox"][3] - f["bbox"][1]) >= config.MIN_FACE_SIZE
                    ]
                    if len(faces) == 0:
                        logger.warning(f"Face too small in {image_path.name} for {person_id} (min size: {config.MIN_FACE_SIZE}px)")
                        continue
                
                # Use the first (largest) detected face
                face_data = faces[0]
                
                # Store image data with metadata for duplicate detection
                image_data_list.append({
                    'embedding': face_data["embedding"],
                    'image_path': image_path,
                    'image': image,  # Keep image for quality scoring
                    'face_bbox': face_data["bbox"]  # Keep bbox for quality scoring
                })
            
            if len(image_data_list) == 0:
                logger.warning(f"No valid face embeddings for {person_id}")
                continue
            
            # Step 2: Remove duplicates, keeping best quality images
            if len(image_data_list) > 1 and config.REMOVE_DUPLICATE_IMAGES:
                logger.debug(f"Checking for duplicates in {person_id}")
                filtered_image_data, deleted_files, stats = remove_duplicates_keep_best(
                    image_data_list,
                    similarity_threshold=config.DUPLICATE_SIMILARITY_THRESHOLD,
                    dry_run=config.DUPLICATE_DETECTION_DRY_RUN
                )
                
                if stats["images_removed"] > 0:
                    logger.info(f"Removed {stats['images_removed']} duplicate image(s) for {person_id}, "
                               f"kept {stats['images_kept']} unique image(s)")
                else:
                    logger.debug(f"No duplicates found for {person_id} ({len(image_data_list)} unique images)")
                
                image_data_list = filtered_image_data
            
            # Step 3: Extract embeddings from filtered list
            embeddings = [img_data['embedding'] for img_data in image_data_list]
            successful_images = len(embeddings)
            
            # Store ALL embeddings for this person (for robust matching)
            self.persons[person_id] = {
                "name": person_name,
                "embeddings": embeddings,  # List of embeddings (one per image)
                "num_images": successful_images
            }
            
            loaded_count += 1
            logger.info(f"Loaded: {person_id} - {person_name} ({successful_images} image{'s' if successful_images > 1 else ''})")
        
        logger.info(f"Total persons loaded: {loaded_count}")
        return loaded_count

    def load_from_s3(
        self,
        college_code: str,
        s3_client: Optional[Any] = None,
        ctx_id: int = -1,
    ) -> int:
        """
        Load face database from S3. Uses this structure per student:
          students/student_id=STU001/profile/passport.jpg
          students/student_id=STU001/face-gallery/img_01.jpg, ...
          students/student_id=STU001/embeddings/face_embedding.npy
        Prefers precomputed embeddings/face_embedding.npy when present; otherwise
        downloads profile + face-gallery images and extracts embeddings.

        Args:
            college_code: College code (e.g. COL001)
            s3_client: S3 client instance (uses config.s3_client if None)
            ctx_id: GPU context ID (-1 for CPU)

        Returns:
            Number of persons loaded
        """
        client = s3_client or getattr(config, "s3_client", None)
        if not client:
            logger.warning("load_from_s3: no S3 client available")
            return 0
        if self.face_detector is None:
            self.face_detector = FaceDetector(ctx_id=ctx_id, recognition=True)
        prefix = f"colleges/college_id={college_code}/students/"
        try:
            keys = client.list_objects(prefix=prefix, college_code=college_code)
        except Exception as e:
            logger.warning(f"load_from_s3: list_objects failed for prefix {prefix!r}: {e}")
            return 0
        # Fallback: if no keys under college prefix, try root-level students/ (e.g. students/student_id=STU001/...)
        if not keys:
            fallback_prefix = "students/"
            try:
                keys = client.list_objects(prefix=fallback_prefix, college_code=college_code)
                if keys:
                    logger.info(f"load_from_s3: using root-level prefix {fallback_prefix!r}, got {len(keys)} keys")
                    prefix = fallback_prefix
            except Exception as e:
                logger.debug(f"load_from_s3: fallback list_objects({fallback_prefix!r}) failed: {e}")
        logger.info(f"load_from_s3: listed prefix {prefix!r}, got {len(keys)} keys")
        if not keys:
            logger.warning(f"load_from_s3: no keys under {prefix!r}; ensure S3 has students (e.g. .../students/student_id=STU001/profile/passport.jpg or .../face-gallery/img_01.jpg)")
        # When using root-level students/, embedding key has no college prefix
        use_root_students = prefix == "students/"
        # Group image keys by student_id: .../students/student_id=STU001/profile/... or .../face-gallery/...
        student_keys: Dict[str, List[str]] = defaultdict(list)
        pattern_img = re.compile(r"students/student_id=([^/]+)/(profile|face-gallery)/[^\s]+\.(jpg|jpeg|png|bmp)", re.I)
        pattern_emb = re.compile(r"students/student_id=([^/]+)/embeddings/face_embedding\.npy$", re.I)
        students_with_embeddings: set = set()
        for key in keys:
            m_emb = pattern_emb.search(key)
            if m_emb:
                students_with_embeddings.add(m_emb.group(1))
                continue
            m = pattern_img.search(key)
            if m:
                student_id = m.group(1)
                student_keys[student_id].append(key)
        # Also consider students that only have embeddings (no images in list)
        all_student_ids = set(student_keys.keys()) | students_with_embeddings
        loaded_count = 0
        for person_id in all_student_ids:
            image_keys = student_keys.get(person_id, [])
            # Prefer precomputed embeddings/face_embedding.npy when available
            if use_root_students:
                emb_key = f"students/student_id={person_id}/embeddings/face_embedding.npy"
            else:
                emb_key = (
                    campus_student_embeddings_path(college_code, person_id, "face_embedding.npy")
                    if campus_student_embeddings_path
                    else f"colleges/college_id={college_code}/students/student_id={person_id}/embeddings/face_embedding.npy"
                )
            try:
                buf = client.download_fileobj(emb_key, college_code=college_code)
                data = buf.read()
                if data:
                    arr = np.load(BytesIO(data), allow_pickle=False)
                    embeddings = [arr] if arr.ndim == 1 else [arr[i] for i in range(len(arr))]
                    if embeddings:
                        self.persons[person_id] = {
                            "name": person_id,
                            "embeddings": embeddings,
                            "num_images": len(embeddings),
                        }
                        loaded_count += 1
                        logger.info(f"Loaded from S3 (embeddings): {person_id} ({len(embeddings)} embedding(s))")
                        continue
            except Exception as e:
                # 404 / NotFound is normal when embeddings/face_embedding.npy does not exist yet
                err_msg = str(e).lower()
                if "404" not in err_msg and "not found" not in err_msg:
                    logger.debug(f"load_from_s3: no precomputed embedding for {person_id}: {e}")
            # Fall back to profile + face-gallery images
            if not image_keys:
                continue
            person_name = person_id
            image_data_list: List[Dict[str, Any]] = []
            for s3_key in image_keys:
                try:
                    buf = client.download_fileobj(s3_key, college_code=college_code)
                    data = buf.read()
                    if not data:
                        continue
                    nparr = np.frombuffer(data, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if image is None:
                        continue
                    faces = self.face_detector.detect_and_embed(
                        image,
                        preprocess=config.ENABLE_IMAGE_PREPROCESSING,
                        align_faces=config.ENABLE_FACE_ALIGNMENT,
                    )
                    if not faces:
                        continue
                    if config.MIN_FACE_SIZE > 0:
                        faces = [
                            f for f in faces
                            if (f["bbox"][2] - f["bbox"][0]) >= config.MIN_FACE_SIZE
                            and (f["bbox"][3] - f["bbox"][1]) >= config.MIN_FACE_SIZE
                        ]
                        if not faces:
                            continue
                    face_data = faces[0]
                    image_data_list.append({
                        "embedding": face_data["embedding"],
                        "image_path": s3_key,
                        "image": image,
                        "face_bbox": face_data["bbox"],
                    })
                except Exception as e:
                    logger.debug(f"load_from_s3: skip {s3_key}: {e}")
                    continue
            if not image_data_list:
                logger.warning(f"No valid face embeddings from S3 for {person_id}")
                continue
            if len(image_data_list) > 1 and config.REMOVE_DUPLICATE_IMAGES:
                filtered_image_data, _, _ = remove_duplicates_keep_best(
                    image_data_list,
                    similarity_threshold=config.DUPLICATE_SIMILARITY_THRESHOLD,
                    dry_run=config.DUPLICATE_DETECTION_DRY_RUN,
                )
                image_data_list = filtered_image_data
            embeddings = [d["embedding"] for d in image_data_list]
            self.persons[person_id] = {
                "name": person_name,
                "embeddings": embeddings,
                "num_images": len(embeddings),
            }
            loaded_count += 1
            logger.info(f"Loaded from S3: {person_id} - {person_name} ({len(embeddings)} image(s))")
        logger.info(f"load_from_s3: total persons loaded: {loaded_count}")
        return loaded_count

    def add_person(self, person_id: str, name: str, embedding: np.ndarray):
        """
        Add a person to the database.
        
        Args:
            person_id: Unique identifier for person
            name: Person's name
            embedding: Face embedding vector
        """
        self.persons[person_id] = {
            "name": name,
            "embedding": embedding
        }

    def add_embeddings_to_person(
        self, person_id: str, name: str, new_embeddings: List[np.ndarray]
    ) -> int:
        """
        Add or merge embeddings for a person. Used when student uploads passport/360°
        so face_embedding.npy can be updated without app restart.
        
        Args:
            person_id: Unique identifier (e.g. roll_number, college_student_id)
            name: Display name
            new_embeddings: List of embedding vectors from new photo(s)
            
        Returns:
            Total number of embeddings now stored for this person
        """
        if not new_embeddings:
            return len(self._get_embeddings_list(self.persons.get(person_id, {})))
        existing = self._get_embeddings_list(self.persons.get(person_id, {}))
        merged = list(existing) + list(new_embeddings)
        # Cap at 20 embeddings per person to avoid unbounded growth
        max_embeddings = getattr(config, "MAX_EMBEDDINGS_PER_PERSON", 20)
        if len(merged) > max_embeddings:
            merged = merged[-max_embeddings:]
        self.persons[person_id] = {"name": name, "embeddings": merged}
        return len(merged)
    
    def get_all_embeddings(self) -> Dict[str, List[np.ndarray]]:
        """
        Get all embeddings from database.
        Now returns multiple embeddings per person for robust matching!
        
        Returns:
            Dict of {person_id: [embedding1, embedding2, ...]}
        """
        return {
            person_id: self._get_embeddings_list(data)
            for person_id, data in self.persons.items()
        }
    
    def _get_embeddings_list(self, data: Dict[str, Any]) -> List[np.ndarray]:
        """Return list of embeddings from person data (supports 'embeddings' or single 'embedding')."""
        if "embeddings" in data:
            return data["embeddings"]
        if "embedding" in data:
            return [data["embedding"]]
        return []
    
    def get_embeddings_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """
        Get all embeddings as a single matrix for vectorized matching.
        
        Returns:
            Tuple of:
            - embeddings_matrix: (N, D) array where N is total faces, D is embedding dim
            - person_ids: List of N person_ids corresponding to each row
        """
        all_embeddings = []
        person_ids = []
        
        for person_id, data in self.persons.items():
            for emb in self._get_embeddings_list(data):
                all_embeddings.append(emb)
                person_ids.append(person_id)
                
        if not all_embeddings:
            return np.array([]), []
            
        return np.vstack(all_embeddings), person_ids
    
    def get_person_info(self, person_id: str) -> Optional[Dict]:
        """
        Get person information.
        
        Args:
            person_id: Person identifier
            
        Returns:
            Dict with person info or None if not found
        """
        if person_id in self.persons:
            return {
                "person_id": person_id,
                "name": self.persons[person_id]["name"]
            }
        return None
    
    def get_database_size(self) -> int:
        """Get number of persons in database."""
        return len(self.persons)
    
    def clear(self):
        """Clear all data from database."""
        self.persons.clear()
