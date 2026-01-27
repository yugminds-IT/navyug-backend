"""
Temporal face tracking and smoothing for improved matching across frames.
"""
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np


class TemporalFaceTracker:
    """
    Tracks faces across frames and applies temporal smoothing for better matching.
    """
    
    def __init__(self, smoothing_window: int = 3, min_consistent_matches: int = 2):
        """
        Initialize temporal tracker.
        
        Args:
            smoothing_window: Number of frames to consider for smoothing
            min_consistent_matches: Minimum consistent matches to confirm identity
        """
        self.smoothing_window = smoothing_window
        self.min_consistent_matches = min_consistent_matches
        self.track_history = defaultdict(list)  # {track_id: [(frame_id, person_id, confidence), ...]}
        self.next_track_id = 0
    
    def track_face(
        self,
        frame_id: int,
        embedding: np.ndarray,
        person_id: Optional[str],
        confidence: float,
        similarity_threshold: float = 0.85
    ) -> Tuple[Optional[str], float, int]:
        """
        Track a face across frames and apply temporal smoothing.
        
        Args:
            frame_id: Current frame ID
            embedding: Face embedding
            person_id: Matched person ID (or None)
            confidence: Match confidence
            similarity_threshold: Similarity threshold for track matching
            
        Returns:
            Tuple of (smoothed_person_id, smoothed_confidence, track_id)
        """
        # Try to match to existing track
        track_id = self._find_matching_track(embedding, similarity_threshold)
        
        if track_id is None:
            # Create new track
            track_id = self.next_track_id
            self.next_track_id += 1
        
        # Add to track history
        self.track_history[track_id].append((frame_id, person_id, confidence))
        
        # Keep only recent history
        if len(self.track_history[track_id]) > self.smoothing_window:
            self.track_history[track_id] = self.track_history[track_id][-self.smoothing_window:]
        
        # Apply temporal smoothing
        smoothed_id, smoothed_conf = self._apply_smoothing(track_id)
        
        return smoothed_id, smoothed_conf, track_id
    
    def _find_matching_track(
        self,
        embedding: np.ndarray,
        similarity_threshold: float
    ) -> Optional[int]:
        """
        Find existing track that matches this embedding.
        
        Args:
            embedding: Face embedding to match
            similarity_threshold: Similarity threshold
            
        Returns:
            Track ID or None if no match
        """
        # For simplicity, we'll use a basic approach
        # In a full implementation, we'd store track embeddings and compare
        # For now, we'll use frame proximity and person ID consistency
        
        # This is a simplified version - full implementation would compare embeddings
        return None
    
    def _apply_smoothing(self, track_id: int) -> Tuple[Optional[str], float]:
        """
        Apply temporal smoothing to a track.
        
        Args:
            track_id: Track ID to smooth
            
        Returns:
            Tuple of (smoothed_person_id, smoothed_confidence)
        """
        history = self.track_history[track_id]
        
        if len(history) == 0:
            return None, 0.0
        
        # Get most recent match
        _, recent_id, recent_conf = history[-1]
        
        # Count consistent matches in window
        person_counts = defaultdict(int)
        person_confidences = defaultdict(list)
        
        for _, person_id, conf in history:
            if person_id is not None:
                person_counts[person_id] += 1
                person_confidences[person_id].append(conf)
        
        if len(person_counts) == 0:
            return None, 0.0
        
        # Find most consistent person
        most_common_person = max(person_counts.items(), key=lambda x: x[1])
        most_common_id, count = most_common_person
        
        # Only return match if consistent enough
        if count >= self.min_consistent_matches:
            # Average confidence for this person
            avg_conf = np.mean(person_confidences[most_common_id])
            return most_common_id, float(avg_conf)
        else:
            # Not consistent enough, return most recent
            return recent_id, recent_conf
    
    def get_track_statistics(self) -> Dict:
        """
        Get statistics about all tracks.
        
        Returns:
            Dictionary with track statistics
        """
        stats = {
            "total_tracks": len(self.track_history),
            "active_tracks": sum(1 for h in self.track_history.values() if len(h) > 0)
        }
        return stats
