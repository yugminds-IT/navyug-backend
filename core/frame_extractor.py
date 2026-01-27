import cv2
from typing import Generator, Dict, Optional

class FrameExtractor:
    """
    Extract frames from video at specified intervals.
    """
    
    def __init__(self, video_path: str, frames_per_second: int = 1):
        """
        Initialize frame extractor.
        
        Args:
            video_path: Path to video file
            frames_per_second: Number of frames to extract per second
        """
        self.video_path = video_path
        self.frames_per_second = frames_per_second
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_interval = int(self.fps / frames_per_second)
        
        # Calculate total frames to extract
        self.total_extract_frames = self.total_frames // self.frame_interval
    
    def extract_frames(self) -> Generator[Dict, None, None]:
        """
        Generator that yields frames at specified intervals.
        
        Yields:
            Dict with frame, frame_id, and timestamp
        """
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Extract frame at intervals
            if frame_count % self.frame_interval == 0:
                timestamp = frame_count / self.fps
                
                yield {
                    "frame": frame,
                    "frame_id": extracted_count,
                    "original_frame_id": frame_count,
                    "timestamp": round(timestamp, 3),
                    "total_frames": self.total_extract_frames
                }
                
                extracted_count += 1
            
            frame_count += 1
    
    def release(self):
        """Release video capture resources."""
        if self.cap:
            self.cap.release()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
    
    def get_video_info(self) -> Dict:
        """
        Get video metadata.
        
        Returns:
            Dict with video information
        """
        return {
            "fps": self.fps,
            "total_frames": self.total_frames,
            "duration_seconds": self.total_frames / self.fps,
            "frames_to_extract": self.total_extract_frames,
            "extraction_rate": self.frames_per_second
        }
