import cv2

class VideoReader:
    def __init__(self,video_path):
        self.video_path=video_path
        self.cap=cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video:{video_path}")
        
        self.fps=self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame=0


    def read(self):
        ret,frame=self.cap.read()
        if not ret:
            return None
        
        self.current_frame +=1
        timestamp=self.current_frame/self.fps

        return {
            "frame": frame,
            "frame_id": self.current_frame,
            "timestamp": round(timestamp, 3)
        }
    

    def release(self):
        self.cap.release()