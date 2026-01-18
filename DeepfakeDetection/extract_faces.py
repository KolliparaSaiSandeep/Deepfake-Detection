import cv2
import os
from facenet_pytorch import MTCNN

# Setup detector
detector = MTCNN(margin=20, keep_all=False, post_process=False, device='cpu')

def process_videos(video_folder, save_folder, max_videos=500):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)
    
    videos = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
    for v_file in videos[:max_videos]:
        cap = cv2.VideoCapture(os.path.join(video_folder, v_file))
        success, frame = cap.read()
        if success:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            save_path = os.path.join(save_folder, f"{v_file}.jpg")
            detector(frame_rgb, save_path=save_path)
        cap.release()

# Change these to your actual Download paths from your image
real_path = "C:/Users/u1196158/Downloads/archive/Celeb-real"
fake_path = "C:/Users/u1196158/Downloads/archive/Celeb-synthesis"

process_videos(real_path, "data/train/real")
process_videos(fake_path, "data/train/fake")
print("✅ Faces extracted to data/ folder.")

# Change paths to your test folders
real_test_path = "C:/Users/u1196158/Downloads/archive/Celeb-real" 
fake_test_path = "C:/Users/u1196158/Downloads/archive/Celeb-synthesis"

# Extract to the 'test' folder
process_videos(real_test_path, "data/test/real", max_videos=100)
process_videos(fake_test_path, "data/test/fake", max_videos=100)
print("🧪 Test faces extracted.")