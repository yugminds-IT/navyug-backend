"""
Image quality assessment and duplicate detection utilities.
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
from core.face_recognizer import FaceRecognizer


def calculate_image_quality_score(image: np.ndarray, face_bbox: List[int] = None) -> float:
    """
    Calculate a quality score for an image based on multiple factors.
    
    Quality factors:
    1. Resolution (higher = better)
    2. Sharpness (Laplacian variance)
    3. Brightness (optimal range)
    4. Contrast
    5. Face size (if bbox provided)
    
    Args:
        image: Image array (BGR format from OpenCV)
        face_bbox: Optional bounding box [x1, y1, x2, y2] of detected face
        
    Returns:
        Quality score (0-1, higher is better)
    """
    if image is None or image.size == 0:
        return 0.0
    
    scores = []
    
    # 1. Resolution score (normalized by typical good resolution)
    height, width = image.shape[:2]
    pixel_count = width * height
    # Normalize: 200x200 = 0.5, 400x400 = 1.0, 800x800 = 1.0 (capped)
    resolution_score = min(1.0, pixel_count / (400 * 400))
    scores.append(("resolution", resolution_score * 0.25))  # 25% weight
    
    # 2. Sharpness score (Laplacian variance)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Normalize: typical good images have 100-500, excellent > 500
    sharpness_score = min(1.0, laplacian_var / 300.0)
    scores.append(("sharpness", sharpness_score * 0.30))  # 30% weight
    
    # 3. Brightness score (optimal range: 100-200 in 0-255 scale)
    mean_brightness = np.mean(gray)
    # Optimal brightness is around 127 (middle of 0-255)
    # Score peaks at 127, decreases as we move away
    brightness_diff = abs(mean_brightness - 127)
    brightness_score = max(0.0, 1.0 - (brightness_diff / 127.0))
    scores.append(("brightness", brightness_score * 0.20))  # 20% weight
    
    # 4. Contrast score (standard deviation of pixel values)
    contrast = np.std(gray)
    # Good contrast is typically 30-60
    contrast_score = min(1.0, contrast / 50.0)
    scores.append(("contrast", contrast_score * 0.15))  # 15% weight
    
    # 5. Face size score (if bbox provided)
    if face_bbox is not None:
        x1, y1, x2, y2 = face_bbox
        face_width = x2 - x1
        face_height = y2 - y1
        face_area = face_width * face_height
        image_area = width * height
        
        # Face should be reasonably large (at least 10% of image, optimal 20-40%)
        face_ratio = face_area / image_area if image_area > 0 else 0
        if face_ratio < 0.1:
            face_size_score = face_ratio / 0.1  # Penalize small faces
        elif face_ratio > 0.4:
            face_size_score = 1.0 - ((face_ratio - 0.4) / 0.4)  # Slight penalty for too large
        else:
            face_size_score = 1.0  # Optimal range
        
        scores.append(("face_size", face_size_score * 0.10))  # 10% weight
    
    # Calculate weighted sum
    total_score = sum(score for _, score in scores)
    
    return min(1.0, max(0.0, total_score))


def detect_duplicate_images(
    image_data_list: List[Dict],
    similarity_threshold: float = 0.85
) -> List[List[int]]:
    """
    Detect duplicate images based on face embedding similarity.
    
    Args:
        image_data_list: List of dicts with keys:
            - 'embedding': face embedding (np.ndarray)
            - 'image_path': path to image file
            - 'image': image array (optional, for quality scoring)
            - 'face_bbox': bounding box (optional, for quality scoring)
        similarity_threshold: Minimum cosine similarity to consider images duplicates (0-1)
        
    Returns:
        List of duplicate groups, where each group is a list of indices into image_data_list
        Example: [[0, 2, 5], [1, 3]] means images 0,2,5 are duplicates, and 1,3 are duplicates
    """
    if len(image_data_list) < 2:
        return []
    
    duplicate_groups = []
    processed = set()
    
    for i in range(len(image_data_list)):
        if i in processed:
            continue
        
        # Start a new duplicate group
        group = [i]
        embedding_i = image_data_list[i]['embedding']
        
        # Check against all other images
        for j in range(i + 1, len(image_data_list)):
            if j in processed:
                continue
            
            embedding_j = image_data_list[j]['embedding']
            
            # Calculate similarity using static method
            similarity = FaceRecognizer.cosine_similarity(embedding_i, embedding_j)
            
            # If similarity is above threshold, they're duplicates
            if similarity >= similarity_threshold:
                group.append(j)
                processed.add(j)
        
        # Only add groups with duplicates (size > 1)
        if len(group) > 1:
            duplicate_groups.append(group)
            processed.add(i)
    
    return duplicate_groups


def _path_display_name(path) -> str:
    """Display name for path: Path.name, or last segment of S3 key, or str(path)."""
    if hasattr(path, "name"):
        return getattr(path, "name", str(path))
    if isinstance(path, str):
        return path.split("/")[-1] if "/" in path else path
    return str(path)


def remove_duplicates_keep_best(
    image_data_list: List[Dict],
    similarity_threshold: float = 0.85,
    dry_run: bool = False
) -> Tuple[List[Dict], List, Dict]:
    """
    Remove duplicate images, keeping only the best quality image from each duplicate group.
    
    Args:
        image_data_list: List of dicts with keys:
            - 'embedding': face embedding (np.ndarray)
            - 'image_path': path to image (Path object or str S3 key)
            - 'image': image array (for quality scoring)
            - 'face_bbox': bounding box (for quality scoring)
        similarity_threshold: Minimum similarity to consider duplicates
        dry_run: If True, don't delete files, just return what would be deleted
        
    Returns:
        Tuple of:
        - filtered_image_data_list: List with duplicates removed (only best quality kept)
        - deleted_files: List of paths that were/would be deleted (only local Paths are unlinked)
        - stats: Dict with statistics about the deduplication
    """
    if len(image_data_list) < 2:
        return image_data_list, [], {"duplicate_groups": 0, "images_removed": 0}
    
    # Detect duplicate groups
    duplicate_groups = detect_duplicate_images(image_data_list, similarity_threshold)
    
    if not duplicate_groups:
        return image_data_list, [], {"duplicate_groups": 0, "images_removed": 0}
    
    # Track which indices to keep and which to remove
    indices_to_remove = set()
    kept_indices = set(range(len(image_data_list)))
    deleted_files = []
    
    # Process each duplicate group
    for group in duplicate_groups:
        # Calculate quality scores for all images in this group
        quality_scores = []
        for idx in group:
            img_data = image_data_list[idx]
            quality_score = calculate_image_quality_score(
                img_data.get('image'),
                img_data.get('face_bbox')
            )
            quality_scores.append((idx, quality_score, img_data['image_path']))
        
        # Sort by quality score (descending), then by resolution as tiebreaker
        def get_resolution(idx):
            img = image_data_list[idx].get('image')
            if img is not None and img.size > 0:
                return img.shape[0] * img.shape[1]
            return 0
        
        quality_scores.sort(key=lambda x: (
            -x[1],  # Quality score (negative for descending)
            -get_resolution(x[0])  # Resolution as tiebreaker
        ))
        
        # Keep the best one, mark others for removal
        best_idx = quality_scores[0][0]
        for idx, score, path in quality_scores[1:]:
            indices_to_remove.add(idx)
            deleted_files.append(path)
            kept_indices.discard(idx)
        
        print(f"  Duplicate group: {len(group)} images")
        print(f"    Keeping: {_path_display_name(image_data_list[best_idx]['image_path'])} (quality: {quality_scores[0][1]:.3f})")
        for idx, score, path in quality_scores[1:]:
            print(f"    Removing: {_path_display_name(path)} (quality: {score:.3f})")
    
    # Filter the list to keep only non-duplicate images
    filtered_list = [image_data_list[i] for i in range(len(image_data_list)) if i not in indices_to_remove]
    
    stats = {
        "duplicate_groups": len(duplicate_groups),
        "images_removed": len(deleted_files),
        "images_kept": len(filtered_list)
    }
    
    # Delete only local files if not dry run (never unlink S3 keys)
    if not dry_run:
        for file_path in deleted_files:
            if isinstance(file_path, Path) and file_path.exists():
                try:
                    file_path.unlink()
                    print(f"    Deleted: {file_path.name}")
                except Exception as e:
                    print(f"    Warning: Could not delete {file_path.name}: {e}")
            # S3 keys (str) are not deleted from disk; they are just dropped from the in-memory list
    
    return filtered_list, deleted_files, stats
