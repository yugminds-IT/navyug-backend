import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict

class FaceRecognizer:
    """
    Enhanced face recognition using cosine similarity with quality weighting,
    confidence calibration, and improved matching strategies.
    """
    
    def __init__(self, similarity_threshold: float = 0.6, use_quality_weighting: bool = True, use_statistical: bool = True):
        """
        Initialize face recognizer.
        
        Args:
            similarity_threshold: Minimum cosine similarity for a match (0-1)
            use_quality_weighting: If True, weight matches by face quality
            use_statistical: If True, use statistical matching (top-3 average)
        """
        self.similarity_threshold = similarity_threshold
        self.use_quality_weighting = use_quality_weighting
        self.use_statistical = use_statistical
    
    @staticmethod
    def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        """
        # Normalize embeddings
        embedding1_norm = embedding1 / (np.linalg.norm(embedding1) + 1e-10)
        embedding2_norm = embedding2 / (np.linalg.norm(embedding2) + 1e-10)
        return float(np.dot(embedding1_norm, embedding2_norm))

    @staticmethod
    def compute_similarity_matrix(query_embeddings: np.ndarray, db_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute similarity matrix between queries and database.
        Shape: (Q, D) x (N, D).T -> (Q, N)
        """
        # Normalize
        q_norm = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        db_norm = np.linalg.norm(db_embeddings, axis=1, keepdims=True)
        
        # Avoid division by zero
        q_normalized = query_embeddings / (q_norm + 1e-10)
        db_normalized = db_embeddings / (db_norm + 1e-10)
        
        return np.dot(q_normalized, db_normalized.T)

    def match_face(
        self, 
        query_embedding: np.ndarray, 
        database_data: Union[Dict[str, List[np.ndarray]], Tuple[np.ndarray, List[str]]],
        face_quality: float = 1.0,
        use_adaptive_threshold: bool = True
    ) -> Tuple[Optional[str], float]:
        """
        Enhanced face matching with quality weighting and adaptive thresholds.
        Supports both legacy dictionary database and optimized matrix/id tuple.
        """
        if isinstance(database_data, tuple):
            # Vectorized path
            db_matrix, person_ids = database_data
            if len(person_ids) == 0:
                return None, 0.0
                
            # Ensure query is (1, D)
            if query_embedding.ndim == 1:
                query_matrix = query_embedding.reshape(1, -1)
            else:
                query_matrix = query_embedding
                
            sim_matrix = self.compute_similarity_matrix(query_matrix, db_matrix)
            # sim_matrix is (1, N)
            similarities = sim_matrix[0]
            
            # Find best match per person
            person_max_sims = defaultdict(float)
            person_all_sims = defaultdict(list)
            
            for idx, sim in enumerate(similarities):
                pid = person_ids[idx]
                person_all_sims[pid].append(sim)
                person_max_sims[pid] = max(person_max_sims[pid], sim)
                
            # Statistical aggregation (same logic as before)
            best_match_id = None
            best_similarity = 0.0
            
            for pid, max_sim in person_max_sims.items():
                if self.use_statistical and len(person_all_sims[pid]) >= 3:
                     top_3 = sorted(person_all_sims[pid], reverse=True)[:3]
                     avg_sim = np.mean(top_3)
                     final_sim = 0.7 * max_sim + 0.3 * avg_sim
                else:
                    final_sim = max_sim
                    
                if final_sim > best_similarity:
                    best_similarity = final_sim
                    best_match_id = pid
            
            all_sims_flat = similarities.tolist()
            
        else:
            # Legacy loop path
            database_embeddings = database_data
            best_match_id = None
            best_similarity = 0.0
            all_sims_flat = [] 
            
            for person_id, person_embeddings in database_embeddings.items():
                person_similarities = []
                for db_embedding in person_embeddings:
                    similarity = self.cosine_similarity(query_embedding, db_embedding)
                    person_similarities.append(similarity)
                    all_sims_flat.append(similarity)
                
                if self.use_statistical and len(person_similarities) >= 3:
                    top_3 = sorted(person_similarities, reverse=True)[:3]
                    avg_similarity = np.mean(top_3)
                    max_similarity = max(person_similarities)
                    max_similarity_for_person = 0.7 * max_similarity + 0.3 * avg_similarity
                else:
                    max_similarity_for_person = max(person_similarities) if person_similarities else 0.0
                
                if max_similarity_for_person > best_similarity:
                    best_similarity = max_similarity_for_person
                    best_match_id = person_id

        # Common threshold and calibration logic
        effective_threshold = self.similarity_threshold
        if use_adaptive_threshold and face_quality > 0.7:
            effective_threshold = max(0.3, self.similarity_threshold - 0.05)
        elif use_adaptive_threshold and face_quality < 0.5:
            effective_threshold = min(0.8, self.similarity_threshold + 0.05)
            
        calibrated_confidence = self._calibrate_confidence(
            best_similarity, 
            face_quality, 
            all_sims_flat
        )
        
        if best_similarity >= effective_threshold:
            return best_match_id, calibrated_confidence
        else:
            return None, 0.0
    
    def _calibrate_confidence(
        self, 
        similarity: float, 
        face_quality: float, 
        all_similarities: List[float]
    ) -> float:
        """
        Calibrate confidence score based on face quality and similarity distribution.
        """
        if not all_similarities:
            return similarity
        
        calibrated = similarity
        
        if face_quality > 0.7:
            quality_boost = (face_quality - 0.7) * 0.1
            calibrated = min(1.0, calibrated + quality_boost)
        
        if face_quality < 0.5:
            quality_penalty = (0.5 - face_quality) * 0.15
            calibrated = max(0.0, calibrated - quality_penalty)
        
        if len(all_similarities) > 1:
            # Quick sort for top 2
            # Using partition is faster than full sort for large lists
            arr = np.array(all_similarities)
            if len(arr) >= 2:
                # We need top 2 values. partition puts Nth element in sorted position
                # and everything smaller before, larger after.
                # We want largest.
                idx = np.argpartition(arr, -2)[-2:]
                top_2 = arr[idx]
                sorted_top_2 = np.sort(top_2)[::-1]
                
                gap = sorted_top_2[0] - sorted_top_2[1]
                if gap > 0.15:
                    calibrated = min(1.0, calibrated + 0.05)
                elif gap < 0.05:
                    calibrated = max(0.0, calibrated - 0.05)
        
        return round(float(calibrated), 3)
    
    def match_with_quality_weighting(
        self,
        query_embedding: np.ndarray,
        database_data: Union[Dict, Tuple],
        face_quality: float = 1.0
    ) -> Tuple[Optional[str], float]:
        return self.match_face(query_embedding, database_data, face_quality, use_adaptive_threshold=True)
    
    def match_faces_batch(
        self,
        query_embeddings: np.ndarray,
        database_data: Tuple[np.ndarray, List[str]],
        face_qualities: Optional[np.ndarray] = None,
        use_adaptive_threshold: bool = True
    ) -> List[Tuple[Optional[str], float]]:
        """
        Match multiple face embeddings against database in a single vectorized operation.
        Much faster than matching one-by-one.
        
        Args:
            query_embeddings: (N, D) array of query embeddings
            database_data: Tuple of (db_matrix, person_ids)
            face_qualities: Optional (N,) array of quality scores
            use_adaptive_threshold: Whether to use adaptive thresholding
            
        Returns:
            List of (person_id, confidence) tuples
        """
        db_matrix, person_ids = database_data
        
        if len(query_embeddings) == 0:
            return []
        
        if len(person_ids) == 0:
            return [(None, 0.0)] * len(query_embeddings)
        
        # Ensure query_embeddings is 2D
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
        
        # Compute similarity matrix: (N_queries, N_db)
        sim_matrix = self.compute_similarity_matrix(query_embeddings, db_matrix)
        
        # Default quality scores if not provided
        if face_qualities is None:
            face_qualities = np.ones(len(query_embeddings))
        elif face_qualities.ndim == 0:
            face_qualities = np.array([face_qualities])
        
        results = []
        
        # Process each query embedding
        for i, (sim_row, quality) in enumerate(zip(sim_matrix, face_qualities)):
            # Find best match per person
            person_max_sims = defaultdict(float)
            person_all_sims = defaultdict(list)
            
            for idx, sim in enumerate(sim_row):
                pid = person_ids[idx]
                person_all_sims[pid].append(sim)
                person_max_sims[pid] = max(person_max_sims[pid], sim)
            
            # Statistical aggregation
            best_match_id = None
            best_similarity = 0.0
            
            for pid, max_sim in person_max_sims.items():
                if self.use_statistical and len(person_all_sims[pid]) >= 3:
                    top_3 = sorted(person_all_sims[pid], reverse=True)[:3]
                    avg_sim = np.mean(top_3)
                    final_sim = 0.7 * max_sim + 0.3 * avg_sim
                else:
                    final_sim = max_sim
                    
                if final_sim > best_similarity:
                    best_similarity = final_sim
                    best_match_id = pid
            
            # Apply threshold
            effective_threshold = self.similarity_threshold
            if use_adaptive_threshold and quality > 0.7:
                effective_threshold = max(0.3, self.similarity_threshold - 0.05)
            elif use_adaptive_threshold and quality < 0.5:
                effective_threshold = min(0.8, self.similarity_threshold + 0.05)
            
            # Calibrate confidence
            all_sims_flat = sim_row.tolist()
            calibrated_confidence = self._calibrate_confidence(
                best_similarity,
                float(quality),
                all_sims_flat
            )
            
            if best_similarity >= effective_threshold:
                results.append((best_match_id, calibrated_confidence))
            else:
                results.append((None, 0.0))
        
        return results

