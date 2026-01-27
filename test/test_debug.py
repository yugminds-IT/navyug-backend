import requests
import time
import sys

def test_system():
    url = "http://localhost:8000"
    video_path = "d:/FaceDetection/data/college_5.mp4"
    
    print(f"Uploading {video_path}...")
    try:
        with open(video_path, 'rb') as f:
            files = {'video': f}
            response = requests.post(f"{url}/incident/upload-video", files=files)
            
        if response.status_code != 200:
            print(f"Upload failed: {response.text}")
            return
            
        data = response.json()
        video_id = data['video_id']
        print(f"Video ID: {video_id}")
        
        # Poll status
        while True:
            response = requests.get(f"{url}/incident/status/{video_id}")
            status_data = response.json()
            status = status_data['status']
            progress = status_data.get('progress', {})
            percentage = progress.get('percentage', 0)
            
            print(f"Status: {status} ({percentage}%)")
            
            if status == 'completed':
                break
            elif status == 'failed':
                print(f"Processing failed: {status_data.get('error')}")
                return
                
            time.sleep(1)
            
        # Get results
        response = requests.get(f"{url}/incident/results/{video_id}")
        results = response.json()
        print("\nResults:")
        print(f"Faces detected: {results['faces_detected']}")
        print(f"Debug faces saved: {results.get('debug_faces_saved', 0)}")
        print(f"Matched persons: {len(results['matched_persons'])}")
        
        for person in results['matched_persons']:
            print(f"- {person['name']} ({person['person_id']}): {person['confidence']}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_system()
