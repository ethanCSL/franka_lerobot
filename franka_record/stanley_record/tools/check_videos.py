import cv2
import os
from pathlib import Path

def check_video_metadata(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Cannot open: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_reported = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps_reported if fps_reported > 0 else 0
    fps_computed = total_frames / duration if duration > 0 else 0

    print(f"\n📹 {video_path.name}")
    print(f"  ➤ Total Frames     : {total_frames}")
    print(f"  ➤ Reported FPS     : {fps_reported:.2f}")
    print(f"  ➤ Duration (sec)   : {duration:.2f}")
    print(f"  ➤ Computed FPS     : {fps_computed:.2f}")

    if abs(fps_computed - fps_reported) > 1:
        print("  ⚠️  WARNING: FPS mismatch – may cause indexing errors!")

    cap.release()


if __name__ == "__main__":
    # Change this path to where your video files are stored
    video_dir = Path("~/.cache/huggingface/lerobot/StanleyChueh/franka_lerobot/videos/chunk-000").expanduser()

    # Recursively find all .mp4 files
    for video_file in video_dir.rglob("*.mp4"):
        check_video_metadata(video_file)
