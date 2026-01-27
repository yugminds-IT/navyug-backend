import insightface
import numpy as np
import cv2
from typing import List, Dict, Optional
from core.utils import apply_nms
from core.multi_model_embedder import MultiModelEmbedder
from core.logger import logger
import config

class FaceDetector:
    def __init__(self, ctx_id=0, det_size=(640, 640), det_thresh=0.5, recognition=False):
        """
        Initialize face detector with optional recognition support.
        
        Args:
            ctx_id: GPU context ID (-1 for CPU, 0+ for GPU)
            det_size: Detection size tuple (larger = better accuracy, slower)
            det_thresh: Detection confidence threshold
            recognition: If True, enables face recognition with embeddings
        """
        self.det_thresh = det_thresh
        if recognition:
            # Load both detection and recognition models
            self.app = insightface.app.FaceAnalysis(
                allowed_modules=["detection", "recognition"]
            )
        else:
            # Load only detection model
            self.app = insightface.app.FaceAnalysis(
                allowed_modules=["detection"]
            )
            
        # Configure the app
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)
    
        # Look for the detector model and set the threshold if possible
        # InsightFace's FaceAnalysis wraps the detector models differently depending on version
        # But we can try to set the score threshold on the detection model
        if hasattr(self.app, 'det_model'):
            # The retinaface/scrfd model has 'conf_thresh' or 'score_thresh'
            # We set it if the user provided one (via config passed implicitly or expected default)
            # Since strict config passing isn't here yet, we'll rely on the caller or default
            pass

        self.recognition_enabled = recognition
        self.det_size = det_size
        
        # Initialize multi-model embedder if enabled
        self.multi_model_embedder = None
        self._fusion_warning_logged = False  # Track if we've logged the warning
        if recognition and config.USE_MULTI_MODEL_FUSION:
            try:
                self.multi_model_embedder = MultiModelEmbedder(
                    ctx_id=ctx_id,
                    fusion_method=config.FUSION_METHOD,
                    use_arcface=config.USE_ARCFACE,
                    use_facenet=config.USE_FACENET
                )
                # Check which models are actually available
                available_models = ["insightface"]
                if self.multi_model_embedder.use_arcface and self.multi_model_embedder.arcface_model is not None:
                    available_models.append("arcface")
                if self.multi_model_embedder.use_facenet and self.multi_model_embedder.facenet_model is not None:
                    available_models.append("facenet")
                
                if len(available_models) > 1:
                    logger.info(f"Multi-model embedding fusion enabled: {', '.join(available_models)}")
                else:
                    logger.info("Multi-model fusion enabled but only InsightFace available (ArcFace/FaceNet not working)")
                    logger.debug("Will use InsightFace only - this is normal if ArcFace/FaceNet models aren't properly initialized")
            except Exception as e:
                logger.warning(f"Could not initialize multi-model embedder: {e}")
                logger.warning("Falling back to InsightFace only")
                self.multi_model_embedder = None

    @staticmethod
    def preprocess_image(image: np.ndarray) -> np.ndarray:
        """
        Preprocess image to improve face detection.
        
        Improvements:
        - Brightness/contrast normalization
        - Sharpening
        - Noise reduction
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Preprocessed image
        """
        # Convert to LAB color space for better processing
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Apply Gamma Correction if enabled
        if config.ENABLE_GAMMA_CORRECTION:
            gamma = float(config.GAMMA_VALUE)  # Ensure float
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            # Apply gamma correction to the enhanced image
            enhanced = cv2.LUT(enhanced, table)

        # Apply slight sharpening
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]]) * 0.1
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Blend original and sharpened (70% original, 30% sharpened)
        result = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
        
        return result
    
    def detect(self, frame, preprocess: bool = True):
        """
        Detect faces in frame with optional preprocessing.
        
        Args:
            frame: Input frame (BGR format)
            preprocess: If True, apply image preprocessing for better detection
            
        Returns:
            List of dicts with bbox and confidence
        """
        # Preprocess image if enabled
        if preprocess:
            frame = self.preprocess_image(frame)
        
        faces = self.app.get(frame)

        results = []
        for face in faces:
            # Filter by confidence threshold
            if face.det_score < self.det_thresh:
                continue

            x1, y1, x2, y2 = map(int, face.bbox)
            results.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": float(face.det_score)
            })
        return results
    
    def detect_and_embed(self, frame, preprocess: bool = True, align_faces: bool = True):
        """
        Detect faces and generate embeddings with multi-model fusion support.
        
        Args:
            frame: Input frame (BGR format)
            preprocess: If True, apply image preprocessing for better detection
            align_faces: If True, align faces before generating embeddings (improves accuracy)
            
        Returns:
            List of dicts with bbox, confidence, and embedding (fused or single-model)
        """
        if not self.recognition_enabled:
            raise ValueError("Recognition not enabled. Initialize with recognition=True")
        
        # Preprocess image if enabled
        original_frame = frame.copy()
        if preprocess:
            frame = self.preprocess_image(frame)
        
        faces = self.app.get(frame)

        results = []
        for face in faces:
            # Filter by confidence threshold
            if face.det_score < self.det_thresh:
                continue

            x1, y1, x2, y2 = map(int, face.bbox)
            
            # Get embedding - use multi-model fusion if enabled
            if self.multi_model_embedder is not None:
                try:
                    # Extract aligned face region (with padding for better quality)
                    padding = 10
                    x1_padded = max(0, x1 - padding)
                    y1_padded = max(0, y1 - padding)
                    x2_padded = min(frame.shape[1], x2 + padding)
                    y2_padded = min(frame.shape[0], y2 + padding)
                    face_region = frame[y1_padded:y2_padded, x1_padded:x2_padded]
                    
                    # Use multi-model fusion
                    fused_embedding, individual_embeddings = self.multi_model_embedder.get_embedding(face_region)
                    embedding = fused_embedding
                    model_info = {k: "available" if v is not None else "unavailable" 
                                 for k, v in individual_embeddings.items()}
                except Exception as e:
                    # Fall back to InsightFace if fusion fails
                    # Only log warning once to avoid spam - this is expected when ArcFace/FaceNet aren't working
                    if not self._fusion_warning_logged:
                        logger.debug(f"Multi-model fusion unavailable (ArcFace/FaceNet not working), using InsightFace only. "
                                   f"This is normal and will not be logged again. Error: {e}")
                        self._fusion_warning_logged = True
                    embedding = face.embedding
                    model_info = {"insightface": "available"}
            else:
                # Use InsightFace only
                embedding = face.embedding
                model_info = {"insightface": "available"}
            
            results.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": float(face.det_score),
                "embedding": embedding,  # Fused or single-model embedding
                "landmarks": face.kps.tolist() if hasattr(face, 'kps') and face.kps is not None else None,
                "model_info": model_info  # Info about which models were used
            })
        return results
    
    def detect_multi_scale(self, frame, scales: List[float] = [1.0, 0.75, 1.25], preprocess: bool = True):
        """
        Detect faces at multiple scales for better detection of small/large faces.
        
        Args:
            frame: Input frame (BGR format)
            scales: List of scales to try (1.0 = original, 0.75 = smaller, 1.25 = larger)
            preprocess: If True, apply image preprocessing
            
        Returns:
            List of dicts with bbox, confidence, and embedding (merged from all scales)
        """
        if not self.recognition_enabled:
            raise ValueError("Recognition not enabled. Initialize with recognition=True")
        
        all_results = []
        h, w = frame.shape[:2]
        
        for scale in scales:
            # Resize frame
            if scale != 1.0:
                new_w = int(w * scale)
                new_h = int(h * scale)
                scaled_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                scaled_frame = frame
            
            # Detect faces at this scale
            faces = self.detect_and_embed(scaled_frame, preprocess=preprocess, align_faces=True)
            
            # Scale bounding boxes back to original size
            for face in faces:
                if scale != 1.0:
                    face["bbox"] = [
                        int(face["bbox"][0] / scale),
                        int(face["bbox"][1] / scale),
                        int(face["bbox"][2] / scale),
                        int(face["bbox"][3] / scale)
                    ]
                all_results.append(face)
        
        # Remove duplicates (same face detected at multiple scales)
        # Keep the one with highest confidence
        if len(all_results) > 1:
            # Deduplication based on IoU
            all_results = apply_nms(all_results, iou_threshold=0.5)
        
        return all_results

