### video_writer.py
import cv2
from pathlib import Path
import numpy as np

class VideoWriterModule:
    """
    Handles saving a sequence of frames (np.ndarray) to an MP4 file.
    """
    def __init__(self, fps: int, codec: str = 'mp4v'):
        self.fps = fps
        self.codec = codec

    def write_episode(
        self,
        frames: np.ndarray,
        output_path: Path,
    ) -> None:
        """
        frames: np.ndarray of shape (N, H, W, 3), dtype=uint8
        output_path: Path to .mp4 file
        """
        if frames.size == 0:
            return
        height, width = frames.shape[1], frames.shape[2]
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, self.fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open video writer for {output_path}")
        for frame in frames:
            # frame expected BGR or RGB; assume BGR
            writer.write(frame)
        writer.release()
