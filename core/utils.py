import numpy as np
from typing import List, Dict

def compute_iou(box1: List[int], box2: List[int]) -> float:
    """
    Compute Intersection over Union (IoU) between two boxes.
    Box format: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Area of each box
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Union area
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
        
    return intersection / union

def apply_nms(faces: List[Dict], iou_threshold: float = 0.6) -> List[Dict]:
    """
    Apply Non-Maximum Suppression to filter overlapping faces.
    Keeps the face with higher confidence.
    """
    if not faces:
        return []
        
    # Sort faces by confidence (descending)
    sorted_faces = sorted(faces, key=lambda x: x["confidence"], reverse=True)
    
    keep = []
    while sorted_faces:
        current = sorted_faces.pop(0)
        keep.append(current)
        
        # Remove overlapping faces
        sorted_faces = [
            face for face in sorted_faces 
            if compute_iou(current["bbox"], face["bbox"]) < iou_threshold
        ]
        
    return keep
