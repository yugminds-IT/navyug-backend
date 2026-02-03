"""
Person service for face database operations.
"""
from io import BytesIO
from typing import Optional, List
from sqlalchemy.orm import Session
import json
import numpy as np

from database.models import Person
from core.logger import logger

try:
    import config as _config
except ImportError:
    _config = None


class PersonService:
    """Service for person/face database operations."""
    
    @staticmethod
    def create_person(
        db: Session,
        person_id: str,
        name: str,
        embeddings: List[np.ndarray],
        s3_image_path: Optional[str] = None,
        local_image_path: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> Person:
        """
        Create or update a person in the database.
        
        Args:
            db: Database session
            person_id: Unique person ID
            name: Person's name
            embeddings: List of face embeddings
            s3_image_path: S3 path to image
            local_image_path: Local path to image
            metadata: Additional metadata
            
        Returns:
            Created/updated Person
        """
        # Convert embeddings to JSON-serializable format
        embeddings_json = [emb.tolist() for emb in embeddings]
        
        # Check if person exists
        person = db.query(Person).filter(Person.person_id == person_id).first()
        
        if person:
            # Update existing person
            person.name = name
            person.embedding = json.dumps(embeddings_json)
            person.num_images = len(embeddings)
            person.s3_image_path = s3_image_path or person.s3_image_path
            person.local_image_path = local_image_path or person.local_image_path
            if metadata:
                person.extra_metadata = metadata
        else:
            # Create new person
            person = Person(
                person_id=person_id,
                name=name,
                embedding=json.dumps(embeddings_json),
                num_images=len(embeddings),
                s3_image_path=s3_image_path,
                local_image_path=local_image_path,
                extra_metadata=metadata
            )
            db.add(person)
        
        db.commit()
        db.refresh(person)
        logger.info(f"Saved person to database: {person_id} - {name}")
        return person

    @staticmethod
    def upload_embeddings_to_s3(person_id: str, embeddings: List[np.ndarray], college_code: str) -> bool:
        """
        Upload face embeddings as face_embedding.npy to S3 at
        colleges/college_id=X/students/student_id=Y/embeddings/face_embedding.npy
        """
        if not embeddings or not college_code:
            return False
        try:
            from storage.s3_paths import campus_student_embeddings_path
            if not _config or not getattr(_config, "s3_client", None):
                return False
            s3_key = campus_student_embeddings_path(college_code, person_id, "face_embedding.npy")
            arr = np.vstack(embeddings) if len(embeddings) > 1 else embeddings[0]
            buf = BytesIO()
            np.save(buf, arr, allow_pickle=False)
            buf.seek(0)
            _config.s3_client.upload_fileobj(
                buf,
                s3_key,
                college_code=college_code,
                content_type="application/octet-stream",
            )
            logger.debug(f"Uploaded face_embedding.npy to S3 for {person_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to upload face_embedding.npy to S3 for {person_id}: {e}")
            return False
    
    @staticmethod
    def get_person(db: Session, person_id: str) -> Optional[Person]:
        """Get person by ID."""
        return db.query(Person).filter(Person.person_id == person_id).first()
    
    @staticmethod
    def get_all_persons(db: Session, active_only: bool = True) -> List[Person]:
        """Get all persons."""
        query = db.query(Person)
        if active_only:
            query = query.filter(Person.is_active == True)
        return query.all()
    
    @staticmethod
    def get_embeddings(db: Session, person_id: str) -> Optional[List[np.ndarray]]:
        """Get embeddings for a person."""
        person = PersonService.get_person(db, person_id)
        if not person or not person.embedding:
            return None
        
        embeddings_json = json.loads(person.embedding)
        return [np.array(emb) for emb in embeddings_json]
    
    @staticmethod
    def get_all_embeddings_matrix(db: Session) -> tuple[np.ndarray, List[str]]:
        """
        Get all embeddings as a matrix for vectorized matching.
        
        Returns:
            Tuple of (embeddings_matrix, person_ids)
        """
        persons = PersonService.get_all_persons(db, active_only=True)
        
        all_embeddings = []
        person_ids = []
        
        for person in persons:
            if person.embedding:
                embeddings_json = json.loads(person.embedding)
                for emb in embeddings_json:
                    all_embeddings.append(np.array(emb))
                    person_ids.append(person.person_id)
        
        if not all_embeddings:
            return np.array([]), []
        
        return np.vstack(all_embeddings), person_ids
    
    @staticmethod
    def delete_person(db: Session, person_id: str) -> bool:
        """Delete person (soft delete by setting is_active=False)."""
        person = PersonService.get_person(db, person_id)
        if not person:
            return False
        
        person.is_active = False
        db.commit()
        logger.info(f"Deactivated person: {person_id}")
        return True
