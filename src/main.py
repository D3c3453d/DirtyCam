import cv2
import os

def extract_frames(video_path: str, output_folder: str, frame_interval: int = 1):
    cap = cv2.VideoCapture(f"../{video_path}")
    os.makedirs(f"../{output_folder}", exist_ok=True)
    frame_count = 0
    saved_count = 0
    if cap.isOpened():
        print("Video file successfully retrieved")
    else:
        print("Video file wasn't retrieved properly")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            cv2.imwrite(f"../{output_folder}/{saved_count:04d}.jpg", frame)
            saved_count += 1
        frame_count += 1
    cap.release()
    print(f"Извлечено {saved_count} кадров")

if __name__ == "__main__":
    extract_frames("autofocus1.mp4", "frames", 7)
