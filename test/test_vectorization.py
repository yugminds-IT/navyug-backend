import sys
import os
import numpy as np
import time
from typing import Dict, List

# Add project root to path
sys.path.append(os.getcwd())

from core.face_recognizer import FaceRecognizer

def test_vectorized_matching():
    print("Initializing test data...")
    recognizer = FaceRecognizer()
    
    # Create fake database: 100 people, 5 embeddings each
    NUM_PEOPLE = 100
    EMBEDDINGS_PER_PERSON = 5
    EMBEDDING_DIM = 512
    
    db_dict = {}
    all_embeddings = []
    all_ids = []
    
    for i in range(NUM_PEOPLE):
        pid = f"person_{i}"
        embeddings = []
        for _ in range(EMBEDDINGS_PER_PERSON):
            # Random normalized vector
            emb = np.random.randn(EMBEDDING_DIM)
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)
            all_embeddings.append(emb)
            all_ids.append(pid)
        db_dict[pid] = embeddings
        
    db_matrix = np.vstack(all_embeddings)
    db_tuple = (db_matrix, all_ids)
    
    # Create query vector (matches person_0)
    query = db_dict["person_0"][0] + np.random.randn(EMBEDDING_DIM) * 0.01
    query = query / np.linalg.norm(query)
    
    print(f"Database size: {len(all_embeddings)} embeddings")
    
    # Test 1: Compare Result Correctness
    print("\nTest 1: Correctness (Dict vs Tuple)")
    
    start = time.time()
    match_dict, conf_dict = recognizer.match_face(query, db_dict)
    time_dict = time.time() - start
    
    start = time.time()
    match_tuple, conf_tuple = recognizer.match_face(query, db_tuple)
    time_tuple = time.time() - start
    
    print(f"Dict result: {match_dict} ({conf_dict:.3f}) - Time: {time_dict*1000:.3f}ms")
    print(f"Tuple result: {match_tuple} ({conf_tuple:.3f}) - Time: {time_tuple*1000:.3f}ms")
    
    assert match_dict == match_tuple, f"Mismatch! {match_dict} vs {match_tuple}"
    assert abs(conf_dict - conf_tuple) < 0.01, f"Confidence mismatch! {conf_dict} vs {conf_tuple}"
    print("✅ Logic Correctness Verified")
    
    # Test 2: Benchmark
    print("\nTest 2: Benchmark (1000 queries)")
    NUM_QUERIES = 100
    queries = [np.random.randn(EMBEDDING_DIM) for _ in range(NUM_QUERIES)]
    queries = [q / np.linalg.norm(q) for q in queries]
    
    # Dict Loop
    start = time.time()
    for q in queries:
        recognizer.match_face(q, db_dict)
    end = time.time()
    fps_dict = NUM_QUERIES / (end - start)
    
    # Tuple Loop
    start = time.time()
    for q in queries:
        recognizer.match_face(q, db_tuple)
    end = time.time()
    fps_tuple = NUM_QUERIES / (end - start)
    
    print(f"Classic Dict Approach: {fps_dict:.2f} queries/sec")
    print(f"Vectorized Tuple Approach: {fps_tuple:.2f} queries/sec")
    print(f"Speedup: {fps_tuple/fps_dict:.2f}x")

if __name__ == "__main__":
    test_vectorized_matching()
