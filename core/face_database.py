import os
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from core.face_detector import FaceDetector
from core.image_quality import remove_duplicates_keep_best
from core.logger import logger
import config

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
    
    def get_all_embeddings(self) -> Dict[str, List[np.ndarray]]:
        """
        Get all embeddings from database.
        Now returns multiple embeddings per person for robust matching!
        
        Returns:
            Dict of {person_id: [embedding1, embedding2, ...]}
        """
        return {
            person_id: data["embeddings"] 
            for person_id, data in self.persons.items()
        }
    
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
            for emb in data["embeddings"]:
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
